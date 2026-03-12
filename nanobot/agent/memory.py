"""Memory system for persistent agent memory."""
# 记忆系统模块 - 用于持久化 Agent 的长期记忆
# 包含两层记忆：长期记忆（MEMORY.md）和历史日志（HISTORY.md）

from __future__ import annotations  # 启用 postponed evaluation of annotations

import asyncio  # 异步 I/O 支持
import json  # JSON 序列化
import weakref  # 弱引用，用于缓存管理
from pathlib import Path  # 文件路径处理
from typing import TYPE_CHECKING, Any, Callable  # 类型提示

from loguru import logger  # 结构化日志记录

# 导入工具函数
from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

# TYPE_CHECKING 为 True 时导入的类型仅用于类型检查
if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider  # LLM 提供商基类
    from nanobot.session.manager import Session, SessionManager  # 会话管理相关


# 定义保存记忆的工具调用格式（用于让 LLM 决定保存什么到长期记忆）
_SAVE_MEMORY_TOOL = [
    {
        "type": "function",  # 工具类型：函数
        "function": {
            "name": "save_memory",  # 工具名称
            "description": "Save important facts to persistent memory. Call this when you learn something that should persist across sessions (user preferences, project context, relationships).",  # 工具描述
            "parameters": {  # 参数定义
                "type": "object",
                "properties": {
                    "facts": {  # facts 参数：要保存的事实列表
                        "type": "array",
                        "items": {"type": "string"},  # 每个元素是字符串
                        "description": "List of facts to save to MEMORY.md. Each fact should be a complete, standalone sentence.",
                    }
                },
                "required": ["facts"],  # 必需参数
            },
        },
    }
]


class MemoryStore:
    """Manages persistent memory files (MEMORY.md and HISTORY.md)."""
    # 记忆存储类 - 管理持久化记忆文件

    def __init__(self, workspace: Path):
        # 构造函数，初始化记忆存储
        # workspace: 工作空间路径，记忆文件存储在 workspace/memory/ 目录下
        self.workspace = workspace  # 保存工作空间路径
        # 记忆文件路径
        self.memory_file = workspace / "memory" / "MEMORY.md"  # 长期记忆文件
        self.history_file = workspace / "memory" / "HISTORY.md"  # 历史日志文件

    def get_memory_context(self) -> str:
        """Get the current memory content for injection into prompts."""
        # 获取当前记忆内容，用于注入到提示词中
        if not self.memory_file.exists():  # 如果记忆文件不存在
            return ""  # 返回空字符串
        # 读取并返回文件内容
        return self.memory_file.read_text(encoding="utf-8")

    def append_to_history(self, entry: str) -> None:
        """Append an entry to HISTORY.md."""
        # 向历史日志追加条目
        ensure_dir(self.history_file.parent)  # 确保父目录存在
        # 以追加模式写入，添加换行符
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")


class MemoryConsolidator:
    """
    Consolidates old session messages into MEMORY.md.
    
    Uses a two-step process:
    1. Extract facts from old messages  # 第一步：从旧消息中提取事实
    2. Save facts to MEMORY.md  # 第二步：将事实保存到 MEMORY.md
    """
    # 记忆整合器 - 将旧会话消息整合到长期记忆中

    def __init__(
        self,
        workspace: Path,  # 工作空间路径
        provider: LLMProvider,  # LLM 提供商
        model: str,  # 使用的模型
        sessions: SessionManager,  # 会话管理器
        context_window_tokens: int,  # 上下文窗口 token 限制
        build_messages: Callable,  # 构建消息的函数
        get_tool_definitions: Callable,  # 获取工具定义的函数
    ):
        self.workspace = workspace  # 保存工作空间路径
        self.provider = provider  # LLM 提供商
        self.model = model  # 模型名称
        self.sessions = sessions  # 会话管理器
        self.context_window_tokens = context_window_tokens  # 上下文 token 限制
        self.build_messages = build_messages  # 构建消息函数
        self.get_tool_definitions = get_tool_definitions  # 获取工具定义函数
        
        # 记忆文件路径
        self.memory_file = workspace / "memory" / "MEMORY.md"
        self.history_file = workspace / "memory" / "HISTORY.md"
        
        # 整合阈值：当消息 token 数超过此值时触发整合
        # 使用上下文窗口的 75% 作为阈值
        self.consolidate_threshold = int(context_window_tokens * 0.75)
        
        # 异步锁，防止并发整合
        self._consolidation_lock = asyncio.Lock()
        
        # 缓存：会话密钥 -> (token 数, 弱引用)
        # 用于避免重复计算 token 数
        self._token_cache: dict[str, tuple[int, weakref.ref]] = {}

    def _get_cached_token_count(self, session: Session) -> int:
        """Get cached token count or compute and cache it."""
        # 获取缓存的 token 数，如果没有则计算并缓存
        key = session.key
        # 检查缓存是否存在且会话对象相同
        if key in self._token_cache:
            count, ref = self._token_cache[key]
            if ref() is session:  # 确认是同一个会话对象
                return count  # 返回缓存值
        # 计算 token 数
        count = estimate_prompt_tokens_chain(session.messages)
        # 存入缓存（使用弱引用避免内存泄漏）
        self._token_cache[key] = (count, weakref.ref(session))
        return count

    async def maybe_consolidate_by_tokens(self, session: Session) -> bool:
        """Consolidate if token count exceeds threshold."""
        # 如果 token 数超过阈值，则执行整合
        # 返回是否执行了整合
        
        # 快速检查：如果消息太少，直接返回
        if len(session.messages) < 4:
            return False
        
        # 获取当前 token 数
        token_count = self._get_cached_token_count(session)
        
        # 如果未超过阈值，不整合
        if token_count < self.consolidate_threshold:
            return False
        
        # 获取锁，防止并发整合
        async with self._consolidation_lock:
            # 重新检查（可能在等待锁期间已整合）
            token_count = self._get_cached_token_count(session)
            if token_count < self.consolidate_threshold:
                return False
            
            # 执行整合
            return await self._consolidate(session)

    async def _consolidate(self, session: Session) -> bool:
        """Consolidate old messages into MEMORY.md."""
        # 将旧消息整合到 MEMORY.md
        # 返回是否成功
        
        logger.info("Consolidating memory for session {}", session.key)
        
        # 确定要归档的消息数量（保留最近 10 条）
        messages_to_archive = session.messages[:-10] if len(session.messages) > 10 else []
        if not messages_to_archive:  # 如果没有要归档的消息
            return False
        
        # 提取事实
        facts = await self._extract_facts(messages_to_archive)
        if not facts:  # 如果没有提取到事实
            logger.info("No facts extracted for consolidation")
            # 仍然移除已处理的消息
            session.messages = session.messages[-10:]
            return True
        
        # 保存事实到记忆文件
        await self._save_facts(facts)
        
        # 更新会话：只保留最近的消息
        session.messages = session.messages[-10:]
        
        # 使 token 缓存失效
        self._token_cache.pop(session.key, None)
        
        logger.info("Consolidated {} facts for session {}", len(facts), session.key)
        return True

    async def _extract_facts(self, messages: list[dict]) -> list[str]:
        """Extract important facts from messages using LLM."""
        # 使用 LLM 从消息中提取重要事实
        
        # 构建提取提示词
        system_prompt = """You are a memory extraction assistant. Your task is to read the conversation history and extract important facts that should be remembered for future sessions.

Extract facts that are:
- User preferences (communication style, technical level, etc.)
- Project context (what they're working on, tools they use)
- Relationships (who is involved, their roles)
- Important decisions or conclusions

Return a JSON array of strings, each being a complete, standalone fact."""
        
        # 构建消息列表
        extract_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract facts from this conversation:\n\n{json.dumps(messages, ensure_ascii=False, indent=2)}"},
        ]
        
        try:
            # 调用 LLM 提取事实
            response = await self.provider.chat_with_retry(
                messages=extract_messages,
                model=self.model,
            )
            
            # 解析 JSON 响应
            content = response.content or "[]"
            # 尝试从代码块中提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            facts = json.loads(content.strip())
            if isinstance(facts, list):  # 确保是列表
                return [f for f in facts if isinstance(f, str)]  # 过滤字符串
            return []
        except Exception as e:
            logger.error("Failed to extract facts: {}", e)
            return []

    async def _save_facts(self, facts: list[str]) -> None:
        """Save facts to MEMORY.md."""
        # 将事实保存到 MEMORY.md
        
        ensure_dir(self.memory_file.parent)  # 确保目录存在
        
        # 读取现有内容
        existing = ""
        if self.memory_file.exists():
            existing = self.memory_file.read_text(encoding="utf-8")
        
        # 构建新内容
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # 新事实条目
        new_entry = f"\n## {timestamp}\n\n" + "\n".join(f"- {fact}" for fact in facts)
        
        # 追加到现有内容
        if existing:
            new_content = existing + new_entry
        else:
            new_content = f"# Memory\n\n{new_entry}"
        
        # 写入文件
        self.memory_file.write_text(new_content, encoding="utf-8")

    async def archive_unconsolidated(self, session: Session) -> bool:
        """Archive any unconsolidated messages before clearing session."""
        # 在清空会话前归档任何未整合的消息
        # 用于 /new 命令
        
        if len(session.messages) < 2:  # 如果消息太少，无需归档
            return True
        
        async with self._consolidation_lock:
            # 提取所有消息的事实（不只是旧的）
            facts = await self._extract_facts(session.messages)
            if facts:  # 如果有事实
                await self._save_facts(facts)  # 保存
            return True
