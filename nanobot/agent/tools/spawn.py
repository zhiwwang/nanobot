"""Spawn tool for creating background subagents."""
# 子代理启动工具模块 - 用于创建后台子代理

from typing import TYPE_CHECKING, Any  # 类型提示

from nanobot.agent.tools.base import Tool  # 导入工具基类

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager  # 仅在类型检查时导入


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""
    # 子代理启动工具类 - 启动后台任务执行的子代理

    def __init__(self, manager: "SubagentManager"):
        # 构造函数
        # manager: 子代理管理器实例
        self._manager = manager
        self._origin_channel = "cli"  # 默认源频道
        self._origin_chat_id = "direct"  # 默认源聊天 ID
        self._session_key = "cli:direct"  # 默认会话密钥

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        # 设置子代理通知的源上下文
        # 当子代理完成任务时，会向这个频道发送通知
        # channel: 频道名称
        # chat_id: 聊天 ID
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        # 启动子代理执行给定任务
        # task: 任务描述
        # label: 任务标签（用于显示）
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
        )
