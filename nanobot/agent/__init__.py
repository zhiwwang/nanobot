"""Agent core module."""
# 这是 Agent 核心模块的入口文件
# 负责导出主要的 Agent 组件类

# 从 context 模块导入上下文构建器
# ContextBuilder 用于构建发送给 LLM 的完整上下文（系统提示词 + 历史消息）
from nanobot.agent.context import ContextBuilder

# 从 loop 模块导入 Agent 主循环
# AgentLoop 是 Agent 的核心处理引擎，负责消息处理循环
from nanobot.agent.loop import AgentLoop

# 从 memory 模块导入记忆存储
# MemoryStore 用于管理长期记忆（MEMORY.md）和历史日志（HISTORY.md）
from nanobot.agent.memory import MemoryStore

# 从 skills 模块导入技能加载器
# SkillsLoader 用于加载和管理 Agent 的技能（SKILL.md 文件）
from nanobot.agent.skills import SkillsLoader

# __all__ 定义了当使用 from nanobot.agent import * 时导出的名称
__all__ = ["AgentLoop", "ContextBuilder", "MemoryStore", "SkillsLoader"]
