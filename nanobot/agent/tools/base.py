"""Base class for agent tools."""
# 工具基类模块 - 定义所有工具的抽象基类

from abc import ABC, abstractmethod  # 抽象基类和抽象方法装饰器
from typing import Any  # 类型提示：任意类型


class Tool(ABC):
    """
    Abstract base class for agent tools.

    Tools are capabilities that the agent can use to interact with
    the environment, such as reading files, executing commands, etc.
    """
    # 工具抽象基类 - 所有工具必须继承此类
    # 工具是 Agent 与环境交互的能力，如读取文件、执行命令等

    # 类型映射：JSON Schema 类型 -> Python 类型
    # 用于验证工具参数
    _TYPE_MAP = {
        "string": str,      # JSON string -> Python str
        "integer": int,     # JSON integer -> Python int
        "number": (int, float),  # JSON number -> Python int 或 float
        "boolean": bool,    # JSON boolean -> Python bool
        "array": list,      # JSON array -> Python list
        "object": dict,     # JSON object -> Python dict
    }

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name — must be unique in the registry."""
        # 工具名称 - 必须在注册表中唯一
        # 子类必须实现此属性
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description shown to the LLM."""
        # 工具描述 - 显示给 LLM，说明工具的用途
        # 子类必须实现此属性
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """
        JSON Schema for tool parameters.
        
        Example:
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        """
        # 工具参数的 JSON Schema 定义
        # 用于告诉 LLM 如何调用此工具
        # 子类必须实现此属性
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """
        Execute the tool with the given arguments.
        
        Returns a string result (success message or error).
        """
        # 执行工具
        # kwargs: 工具参数
        # 返回字符串结果（成功消息或错误信息）
        # 子类必须实现此方法
        ...

    def get_definition(self) -> dict[str, Any]:
        """Return the OpenAI-style tool definition."""
        # 返回 OpenAI 格式的工具定义
        # 用于 function calling API
        return {
            "type": "function",  # 工具类型：函数
            "function": {
                "name": self.name,           # 函数名
                "description": self.description,  # 函数描述
                "parameters": self.parameters,    # 参数定义
            },
        }

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate arguments against the parameter schema.
        
        Returns (is_valid, error_message).
        """
        # 根据参数模式验证参数
        # args: 参数字典
        # 返回 (是否有效, 错误信息)
        
        schema = self.parameters  # 获取参数模式
        
        # 检查必需参数
        required = schema.get("required", [])  # 获取必需参数列表
        for param in required:
            if param not in args:  # 如果缺少必需参数
                return False, f"Missing required parameter: {param}"
        
        # 检查参数类型
        properties = schema.get("properties", {})  # 获取参数属性定义
        for key, value in args.items():
            if key in properties:  # 如果参数在定义中
                expected_type = properties[key].get("type")  # 获取期望类型
                if expected_type and expected_type in self._TYPE_MAP:
                    python_type = self._TYPE_MAP[expected_type]  # 转换为 Python 类型
                    if not isinstance(value, python_type):  # 类型不匹配
                        return False, f"Parameter '{key}' should be {expected_type}"
        
        return True, ""  # 验证通过
