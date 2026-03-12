"""File system tools: read, write, edit, list."""
# 文件系统工具模块 - 提供读取、写入、编辑、列出文件/目录的功能

import difflib  # 用于生成文件差异对比
from pathlib import Path  # 文件路径处理
from typing import Any  # 类型提示：任意类型

from nanobot.agent.tools.base import Tool  # 导入工具基类


def _resolve_path(
    path: str, workspace: Path | None = None, allowed_dir: Path | None = None
) -> Path:
    """Resolve path against workspace (if relative) and enforce directory restriction."""
    # 解析路径：将相对路径解析为绝对路径，并检查目录限制
    # path: 原始路径字符串
    # workspace: 工作空间路径（用于解析相对路径）
    # allowed_dir: 允许访问的目录（用于安全限制）
    
    p = Path(path).expanduser()  # 展开用户目录（如 ~）
    if not p.is_absolute() and workspace:
        # 如果是相对路径且有工作空间，则相对于工作空间解析
        p = workspace / p
    resolved = p.resolve()  # 解析为绝对路径（处理 .. 等）
    
    if allowed_dir:
        # 如果设置了允许目录，检查路径是否在允许范围内
        try:
            resolved.relative_to(allowed_dir.resolve())
        except ValueError:
            # 路径不在允许目录内，抛出权限错误
            raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


class _FsTool(Tool):
    """Shared base for filesystem tools — common init and path resolution."""
    # 文件系统工具共享基类 - 提供通用的初始化和路径解析

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        # 构造函数
        # workspace: 工作空间路径
        # allowed_dir: 允许访问的目录
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    def _resolve(self, path: str) -> Path:
        # 解析路径的便捷方法
        return _resolve_path(path, self._workspace, self._allowed_dir)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class ReadFileTool(_FsTool):
    """Read file contents with optional line-based pagination."""
    # 读取文件工具 - 支持基于行的分页读取

    _MAX_CHARS = 128_000  # 最大返回字符数
    _DEFAULT_LIMIT = 2000  # 默认读取行数

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. Returns numbered lines. "
            "Use offset and limit to paginate through large files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to read"},
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed, default 1)",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default 2000)",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        }

    async def execute(self, path: str, offset: int = 1, limit: int | None = None, **kwargs: Any) -> str:
        try:
            fp = self._resolve(path)  # 解析路径
            if not fp.exists():
                return f"Error: File not found: {path}"  # 文件不存在
            if not fp.is_file():
                return f"Error: Not a file: {path}"  # 不是文件

            all_lines = fp.read_text(encoding="utf-8").splitlines()  # 读取所有行
            total = len(all_lines)  # 总行数

            if offset < 1:
                offset = 1
            if total == 0:
                return f"(Empty file: {path})"  # 空文件
            if offset > total:
                return f"Error: offset {offset} is beyond end of file ({total} lines)"  # 偏移超出范围

            start = offset - 1  # 转换为 0-based 索引
            end = min(start + (limit or self._DEFAULT_LIMIT), total)  # 计算结束行
            # 添加行号
            numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(all_lines[start:end])]
            result = "\n".join(numbered)

            # 如果结果超过最大字符数，截断
            if len(result) > self._MAX_CHARS:
                trimmed, chars = [], 0
                for line in numbered:
                    chars += len(line) + 1
                    if chars > self._MAX_CHARS:
                        break
                    trimmed.append(line)
                end = start + len(trimmed)
                result = "\n".join(trimmed)

            # 添加分页提示
            if end < total:
                result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
            else:
                result += f"\n\n(End of file — {total} lines total)"
            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {e}"


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

class WriteFileTool(_FsTool):
    """Write content to a file."""
    # 写入文件工具

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file at the given path. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write to"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            fp = self._resolve(path)  # 解析路径
            fp.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录（如果不存在）
            fp.write_text(content, encoding="utf-8")  # 写入内容
            return f"Successfully wrote {len(content)} bytes to {fp}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {e}"


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

def _find_match(content: str, old_text: str) -> tuple[str | None, int]:
    """Locate old_text in content: exact first, then line-trimmed sliding window.

    Both inputs should use LF line endings (caller normalises CRLF).
    Returns (matched_fragment, count) or (None, 0).
    """
    # 在内容中定位 old_text：先精确匹配，再尝试行修剪滑动窗口匹配
    # content: 文件内容
    # old_text: 要查找的旧文本
    # 返回 (匹配的片段, 匹配次数) 或 (None, 0)
    
    if old_text in content:
        # 精确匹配
        return old_text, content.count(old_text)

    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0
    stripped_old = [l.strip() for l in old_lines]  # 去除每行空白
    content_lines = content.splitlines()

    candidates = []
    # 滑动窗口匹配
    for i in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[i : i + len(stripped_old)]
        if [l.strip() for l in window] == stripped_old:
            candidates.append("\n".join(window))

    if candidates:
        return candidates[0], len(candidates)
    return None, 0


class EditFileTool(_FsTool):
    """Edit a file by replacing text with fallback matching."""
    # 编辑文件工具 - 支持回退匹配的文本替换

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing old_text with new_text. "
            "Supports minor whitespace/line-ending differences. "
            "Set replace_all=true to replace every occurrence."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_text": {"type": "string", "description": "The text to find and replace"},
                "new_text": {"type": "string", "description": "The text to replace with"},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default false)",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self, path: str, old_text: str, new_text: str,
        replace_all: bool = False, **kwargs: Any,
    ) -> str:
        try:
            fp = self._resolve(path)  # 解析路径
            if not fp.exists():
                return f"Error: File not found: {path}"

            raw = fp.read_bytes()
            uses_crlf = b"\r\n" in raw  # 检测是否使用 CRLF 换行
            content = raw.decode("utf-8").replace("\r\n", "\n")  # 统一转换为 LF
            match, count = _find_match(content, old_text.replace("\r\n", "\n"))

            if match is None:
                return self._not_found_msg(old_text, content, path)  # 未找到匹配
            if count > 1 and not replace_all:
                return (
                    f"Warning: old_text appears {count} times. "
                    "Provide more context to make it unique, or set replace_all=true."
                )

            norm_new = new_text.replace("\r\n", "\n")  # 新文本也转换为 LF
            new_content = content.replace(match, norm_new) if replace_all else content.replace(match, norm_new, 1)
            if uses_crlf:
                new_content = new_content.replace("\n", "\r\n")  # 恢复 CRLF

            fp.write_bytes(new_content.encode("utf-8"))
            return f"Successfully edited {fp}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {e}"

    @staticmethod
    def _not_found_msg(old_text: str, content: str, path: str) -> str:
        # 生成未找到匹配时的提示信息，包含最佳匹配建议
        lines = content.splitlines(keepends=True)
        old_lines = old_text.splitlines(keepends=True)
        window = len(old_lines)

        best_ratio, best_start = 0.0, 0
        # 寻找最佳匹配
        for i in range(max(1, len(lines) - window + 1)):
            ratio = difflib.SequenceMatcher(None, old_lines, lines[i : i + window]).ratio()
            if ratio > best_ratio:
                best_ratio, best_start = ratio, i

        if best_ratio > 0.5:
            # 生成差异对比
            diff = "\n".join(difflib.unified_diff(
                old_lines, lines[best_start : best_start + window],
                fromfile="old_text (provided)",
                tofile=f"{path} (actual, line {best_start + 1})",
                lineterm="",
            ))
            return f"Error: old_text not found in {path}.\nBest match ({best_ratio:.0%} similar) at line {best_start + 1}:\n{diff}"
        return f"Error: old_text not found in {path}. No similar text found. Verify the file content."


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------

class ListDirTool(_FsTool):
    """List directory contents with optional recursion."""
    # 列出目录工具 - 支持递归列出

    _DEFAULT_MAX = 200  # 默认最大条目数
    _IGNORE_DIRS = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".coverage", "htmlcov",
    }  # 默认忽略的目录

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return (
            "List the contents of a directory. "
            "Set recursive=true to explore nested structure. "
            "Common noise directories (.git, node_modules, __pycache__, etc.) are auto-ignored."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The directory path to list"},
                "recursive": {
                    "type": "boolean",
                    "description": "Recursively list all files (default false)",
                },
                "max_entries": {
                    "type": "integer",
                    "description": "Maximum entries to return (default 200)",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        }

    async def execute(
        self, path: str, recursive: bool = False,
        max_entries: int | None = None, **kwargs: Any,
    ) -> str:
        try:
            dp = self._resolve(path)  # 解析路径
            if not dp.exists():
                return f"Error: Directory not found: {path}"
            if not dp.is_dir():
                return f"Error: Not a directory: {path}"

            cap = max_entries or self._DEFAULT_MAX
            items: list[str] = []
            total = 0

            if recursive:
                # 递归列出
                for item in sorted(dp.rglob("*")):
                    if any(p in self._IGNORE_DIRS for p in item.parts):
                        continue  # 跳过忽略的目录
                    total += 1
                    if len(items) < cap:
                        rel = item.relative_to(dp)
                        items.append(f"{rel}/" if item.is_dir() else str(rel))
            else:
                # 非递归列出
                for item in sorted(dp.iterdir()):
                    if item.name in self._IGNORE_DIRS:
                        continue  # 跳过忽略的目录
                    total += 1
                    if len(items) < cap:
                        pfx = "📁 " if item.is_dir() else "📄 "
                        items.append(f"{pfx}{item.name}")

            if not items and total == 0:
                return f"Directory {path} is empty"  # 空目录

            result = "\n".join(items)
            if total > cap:
                result += f"\n\n(truncated, showing first {cap} of {total} entries)"
            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {e}"
