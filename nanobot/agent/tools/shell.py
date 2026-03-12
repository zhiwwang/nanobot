"""Shell execution tool."""
# 命令执行工具模块 - 用于执行 shell 命令

import asyncio  # 异步 I/O 支持
import os  # 操作系统接口
import re  # 正则表达式
from pathlib import Path  # 文件路径处理
from typing import Any  # 类型提示：任意类型

from nanobot.agent.tools.base import Tool  # 导入工具基类


class ExecTool(Tool):
    """Tool to execute shell commands."""
    # 命令执行工具类

    def __init__(
        self,
        timeout: int = 60,  # 默认超时时间（秒）
        working_dir: str | None = None,  # 默认工作目录
        deny_patterns: list[str] | None = None,  # 禁止的命令模式列表
        allow_patterns: list[str] | None = None,  # 允许的命令模式列表（白名单）
        restrict_to_workspace: bool = False,  # 是否限制在工作空间内
        path_append: str = "",  # 附加到 PATH 的路径
    ):
        self.timeout = timeout  # 保存超时时间
        self.working_dir = working_dir  # 保存工作目录
        # 默认禁止的危险命令模式
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr - 删除文件/目录
            r"\bdel\s+/[fq]\b",              # del /f, del /q - Windows 删除
            r"\brmdir\s+/s\b",               # rmdir /s - Windows 删除目录
            r"(?:^|[;&|]\s*)format\b",       # format - 格式化磁盘
            r"\b(mkfs|diskpart)\b",          # mkfs, diskpart - 磁盘操作
            r"\bdd\s+if=",                   # dd if=... - 磁盘写入
            r">\s*/dev/sd",                  # 写入到磁盘设备
            r"\b(shutdown|reboot|poweroff)\b",  # shutdown, reboot, poweroff - 系统电源
            r":\(\)\s*\{.*\};\s*:",          # fork bomb - 叉子炸弹
        ]
        self.allow_patterns = allow_patterns or []  # 允许的命令模式
        self.restrict_to_workspace = restrict_to_workspace  # 是否限制工作空间
        self.path_append = path_append  # 附加 PATH

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = 600  # 最大超时时间（秒）
    _MAX_OUTPUT = 10_000  # 最大输出字符数

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds. Increase for long-running commands "
                        "like compilation or installation (default 60, max 600)."
                    ),
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self, command: str, working_dir: str | None = None,
        timeout: int | None = None, **kwargs: Any,
    ) -> str:
        # 执行命令
        cwd = working_dir or self.working_dir or os.getcwd()  # 确定工作目录
        guard_error = self._guard_command(command, cwd)  # 安全检查
        if guard_error:
            return guard_error  # 安全检查失败

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)  # 确定有效超时

        # 设置环境变量
        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            # 创建子进程
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,  # 捕获标准输出
                stderr=asyncio.subprocess.PIPE,  # 捕获标准错误
                cwd=cwd,  # 工作目录
                env=env,  # 环境变量
            )

            try:
                # 等待命令完成，带超时
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                # 超时处理
                process.kill()  # 终止进程
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)  # 等待进程结束
                except asyncio.TimeoutError:
                    pass
                return f"Error: Command timed out after {effective_timeout} seconds"

            output_parts = []

            # 处理标准输出
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            # 处理标准错误
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            # 添加退出码
            output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # 头部+尾部截断，保留输出的开头和结尾
            max_len = self._MAX_OUTPUT
            if len(result) > max_len:
                half = max_len // 2
                result = (
                    result[:half]
                    + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                    + result[-half:]
                )

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        # 尽最大努力的安全检查，防止危险命令
        # 返回错误信息，如果没有问题返回 None
        
        cmd = command.strip()
        lower = cmd.lower()

        # 检查禁止模式
        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        # 如果设置了白名单，检查是否在白名单中
        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        # 如果限制在工作空间内，检查路径遍历
        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()

            # 检查所有绝对路径是否都在工作空间内
            for raw in self._extract_absolute_paths(cmd):
                try:
                    expanded = os.path.expandvars(raw.strip())  # 展开环境变量
                    p = Path(expanded).expanduser().resolve()  # 展开用户目录并解析
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        # 从命令中提取绝对路径
        # Windows 路径: C:\...
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)
        # POSIX 绝对路径: /...
        posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command)
        # 用户目录快捷方式: ~
        home_paths = re.findall(r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command)
        return win_paths + posix_paths + home_paths
