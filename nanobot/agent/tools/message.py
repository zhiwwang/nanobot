"""Message tool for sending messages to users."""
# 消息发送工具模块 - 用于向用户发送消息

from typing import Any, Awaitable, Callable  # 类型提示

from nanobot.agent.tools.base import Tool  # 导入工具基类
from nanobot.bus.events import OutboundMessage  # 导入出站消息类


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""
    # 消息发送工具类 - 在聊天频道上向用户发送消息

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,  # 发送回调函数
        default_channel: str = "",  # 默认频道
        default_chat_id: str = "",  # 默认聊天 ID
        default_message_id: str | None = None,  # 默认消息 ID（用于回复）
    ):
        self._send_callback = send_callback  # 保存发送回调
        self._default_channel = default_channel  # 保存默认频道
        self._default_chat_id = default_chat_id  # 保存默认聊天 ID
        self._default_message_id = default_message_id  # 保存默认消息 ID
        self._sent_in_turn: bool = False  # 本轮是否已发送过消息

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        # 设置当前消息上下文
        # channel: 频道名称
        # chat_id: 聊天 ID
        # message_id: 消息 ID（可选，用于回复）
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        # 设置发送消息的回调函数
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        # 重置每轮发送跟踪
        # 在每一轮对话开始时调用
        self._sent_in_turn = False

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        # 执行消息发送
        # 使用传入的参数或默认值
        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id
        message_id = message_id or self._default_message_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"  # 没有指定目标

        if not self._send_callback:
            return "Error: Message sending not configured"  # 没有配置发送回调

        # 构建出站消息
        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata={
                "message_id": message_id,
            },
        )

        try:
            await self._send_callback(msg)  # 调用回调发送消息
            # 如果发送到当前上下文，标记本轮已发送
            if channel == self._default_channel and chat_id == self._default_chat_id:
                self._sent_in_turn = True
            media_info = f" with {len(media)} attachments" if media else ""
            return f"Message sent to {channel}:{chat_id}{media_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
