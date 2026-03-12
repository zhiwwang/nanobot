"""Agent tools module."""
# Agent 工具模块入口文件
# 负责导出工具基类和工具注册表

# 从 base 模块导入 Tool 抽象基类
# Tool 是所有工具的基类，定义了工具的接口
from nanobot.agent.tools.base import Tool

# 从 registry 模块导入 ToolRegistry 工具注册表
# ToolRegistry 用于动态注册和管理工具
from nanobot.agent.tools.registry import ToolRegistry

# __all__ 定义了当使用 from nanobot.agent.tools import * 时导出的名称
__all__ = ["Tool", "ToolRegistry"]
