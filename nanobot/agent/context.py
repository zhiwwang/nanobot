"""Context builder for assembling agent prompts."""
# 上下文构建器模块 - 负责组装发送给 LLM 的完整提示词和消息列表

import base64  # 用于图片的 base64 编码
import mimetypes  # 用于检测文件 MIME 类型
import platform  # 用于获取操作系统信息
import time  # 用于获取时区信息
from datetime import datetime  # 用于获取当前时间
from pathlib import Path  # 用于处理文件路径
from typing import Any  # 类型提示：任意类型

# 导入记忆存储，用于读取长期记忆
from nanobot.agent.memory import MemoryStore
# 导入技能加载器，用于加载技能信息
from nanobot.agent.skills import SkillsLoader
# 导入工具函数：构建助手消息、检测图片 MIME 类型
from nanobot.utils.helpers import build_assistant_message, detect_image_mime


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""
    # 上下文构建器类 - 构建系统提示词和消息列表

    # 启动时需要加载的引导文件列表
    # 这些文件位于工作空间根目录，用于定义 Agent 的行为和身份
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    
    # 运行时上下文标签，用于标记非指令性的元数据
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

    def __init__(self, workspace: Path):
        # 构造函数，初始化上下文构建器
        # workspace: 工作空间路径，所有文件操作都在此目录下进行
        self.workspace = workspace  # 保存工作空间路径
        self.memory = MemoryStore(workspace)  # 创建记忆存储实例
        self.skills = SkillsLoader(workspace)  # 创建技能加载器实例

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        # 构建系统提示词（System Prompt）
        # skill_names: 需要激活的特定技能列表
        
        parts = [self._get_identity()]  # 第一部分：Agent 身份信息

        # 加载引导文件内容
        bootstrap = self._load_bootstrap_files()
        if bootstrap:  # 如果引导文件有内容
            parts.append(bootstrap)  # 添加到提示词各部分中

        # 加载长期记忆内容
        memory = self.memory.get_memory_context()
        if memory:  # 如果有长期记忆
            parts.append(f"# Memory\n\n{memory}")  # 添加记忆部分

        # 加载始终激活的技能（always=true 的技能）
        always_skills = self.skills.get_always_skills()
        if always_skills:  # 如果有始终激活的技能
            # 加载这些技能的完整内容
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # 构建所有可用技能的摘要列表
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:  # 如果有技能
            # 添加技能说明，告诉 Agent 如何使用技能
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")

        # 用分隔线连接所有部分，形成完整的系统提示词
        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """Get the core identity section."""
        # 获取 Agent 核心身份信息
        
        # 获取工作空间的绝对路径（展开用户目录并解析为绝对路径）
        workspace_path = str(self.workspace.expanduser().resolve())
        
        # 获取操作系统信息
        system = platform.system()
        # 构建运行时信息字符串：操作系统 + 架构 + Python 版本
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        # 根据操作系统生成平台策略说明
        platform_policy = ""
        if system == "Windows":
            # Windows 平台的特殊说明
            platform_policy = """## Platform Policy (Windows)
- You are running on Windows. Do not assume GNU tools like `grep`, `sed`, or `awk` exist.
- Prefer Windows-native commands or file tools when they are more reliable.
- If terminal output is garbled, retry with UTF-8 output enabled.
"""
        else:
            # POSIX 平台（Linux/macOS）的说明
            platform_policy = """## Platform Policy (POSIX)
- You are running on a POSIX system. Prefer UTF-8 and standard shell tools.
- Use file tools when they are simpler or more reliable than shell commands.
"""

        # 返回完整的身份信息模板，包含所有变量
        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md (write important facts here)
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM].
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

{platform_policy}

## nanobot Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Before modifying a file, read it first. Do not assume files or directories exist.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        # 构建运行时上下文元数据块
        # 这些是非指令性的元数据，注入到用户消息之前
        # channel: 消息来源频道（如 telegram、discord）
        # chat_id: 聊天会话 ID
        
        # 获取当前时间，格式：YYYY-MM-DD HH:MM (星期) (时区)
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"  # 获取时区，默认为 UTC
        lines = [f"Current Time: {now} ({tz})"]  # 第一行：当前时间
        if channel and chat_id:  # 如果有频道和聊天 ID
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]  # 添加频道信息
        # 返回带标签的运行时上下文字符串
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        # 从工作空间加载所有引导文件
        parts = []  # 存储各文件内容的列表

        # 遍历预定义的引导文件列表
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename  # 构建文件完整路径
            if file_path.exists():  # 如果文件存在
                content = file_path.read_text(encoding="utf-8")  # 读取文件内容
                parts.append(f"## {filename}\n\n{content}")  # 添加带标题的内容

        # 用空行连接所有部分，如果没有内容则返回空字符串
        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],  # 历史消息列表
        current_message: str,  # 当前用户消息
        skill_names: list[str] | None = None,  # 要激活的技能
        media: list[str] | None = None,  # 媒体文件路径列表（图片等）
        channel: str | None = None,  # 消息频道
        chat_id: str | None = None,  # 聊天 ID
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        # 构建完整的 LLM 消息列表
        
        # 构建运行时上下文（时间、频道等信息）
        runtime_ctx = self._build_runtime_context(channel, chat_id)
        # 构建用户内容（文本 + 媒体）
        user_content = self._build_user_content(current_message, media)

        # 合并运行时上下文和用户内容到单个用户消息中
        # 避免连续相同角色的消息（某些 LLM 提供商拒绝这种情况）
        if isinstance(user_content, str):  # 如果用户内容是纯文本
            merged = f"{runtime_ctx}\n\n{user_content}"  # 直接拼接字符串
        else:  # 如果用户内容是多模态（包含图片）
            # 将运行时上下文作为文本块，加上图片内容
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        # 返回完整的消息列表：系统提示词 + 历史消息 + 当前用户消息
        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *history,  # 展开历史消息
            {"role": "user", "content": merged},  # 当前用户消息
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        # 构建用户消息内容，支持可选的 base64 编码图片
        # text: 用户文本消息
        # media: 媒体文件路径列表
        
        if not media:  # 如果没有媒体文件
            return text  # 直接返回文本

        images = []  # 存储图片内容的列表
        for path in media:  # 遍历每个媒体文件路径
            p = Path(path)  # 转为 Path 对象
            if not p.is_file():  # 如果不是文件则跳过
                continue
            raw = p.read_bytes()  # 读取文件原始字节
            # 从文件魔数检测真实 MIME 类型；如果失败则根据文件名猜测
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):  # 如果不是图片则跳过
                continue
            b64 = base64.b64encode(raw).decode()  # 将图片转为 base64 编码
            # 构建 OpenAI 格式的图片 URL 对象
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:  # 如果没有有效图片
            return text  # 返回纯文本
        return images + [{"type": "text", "text": text}]  # 图片在前，文本在后

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        # 将工具执行结果添加到消息列表
        # messages: 当前消息列表
        # tool_call_id: 工具调用的唯一 ID（用于关联请求和响应）
        # tool_name: 工具名称
        # result: 工具执行结果字符串
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,  # 助手回复内容
        tool_calls: list[dict[str, Any]] | None = None,  # 工具调用列表
        reasoning_content: str | None = None,  # 推理过程内容（如 DeepSeek R1）
        thinking_blocks: list[dict] | None = None,  # 思考块（如 Claude）
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        # 将助手消息添加到消息列表
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages
