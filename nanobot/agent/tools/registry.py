"""Tool registry for dynamic tool management."""
# 工具注册表模块 - 用于动态注册和管理工具

from typing import Any  # 类型提示：任意类型

from nanobot.agent.tools.base import Tool  # 导入工具基类


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """
    # 工具注册表类 - 允许动态注册和执行工具

    def __init__(self):
        # 构造函数，初始化空的工具字典
        # _tools: 工具名称 -> 工具实例 的映射
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        # 注册一个工具
        # tool: 要注册的工具实例
        # 工具名称作为字典键，工具实例作为值
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        # 按名称注销工具
        # name: 要注销的工具名称
        # 如果工具存在则删除，不存在则静默处理
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        # 按名称获取工具
        # name: 工具名称
        # 返回工具实例，如果不存在返回 None
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        # 检查工具是否已注册
        # name: 工具名称
        # 返回布尔值
        return name in self._tools

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        # 列出所有已注册的工具名称
        # 返回工具名称列表
        return list(self._tools.keys())

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get OpenAI-style tool definitions for all registered tools."""
        # 获取所有已注册工具的 OpenAI 格式定义
        # 用于发送给 LLM 进行 function calling
        # 返回工具定义列表
        return [tool.get_definition() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool by name with the given arguments.
        
        Returns the tool result as a string.
        """
        # 按名称执行工具
        # name: 工具名称
        # arguments: 参数字典
        # 返回工具执行结果字符串
        
        tool = self._tools.get(name)  # 获取工具实例
        if not tool:  # 如果工具不存在
            return f"Error: Tool '{name}' not found"
        
        # 验证参数
        is_valid, error = tool.validate_args(arguments)
        if not is_valid:  # 如果参数无效
            return f"Error: {error}"
        
        try:
            # 执行工具并返回结果
            return await tool.execute(**arguments)
        except Exception as e:
            # 捕获异常并返回错误信息
            return f"Error executing tool '{name}': {str(e)}"
