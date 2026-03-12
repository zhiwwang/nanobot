"""Cron tool for scheduling reminders and tasks."""
# 定时任务工具模块 - 用于调度提醒和周期性任务

from contextvars import ContextVar  # 上下文变量，用于在异步任务间传递上下文
from typing import Any  # 类型提示：任意类型

from nanobot.agent.tools.base import Tool  # 导入工具基类
from nanobot.cron.service import CronService  # 导入定时服务
from nanobot.cron.types import CronSchedule  # 导入定时计划类型


class CronTool(Tool):
    """Tool to schedule reminders and recurring tasks."""
    # 定时任务工具类 - 调度提醒和周期性任务

    def __init__(self, cron_service: CronService):
        # 构造函数，初始化定时任务工具
        # cron_service: 定时服务实例，用于实际管理定时任务
        self._cron = cron_service  # 保存定时服务引用
        self._channel = ""  # 当前频道（用于发送提醒消息）
        self._chat_id = ""  # 当前聊天 ID（用于发送提醒消息）
        # 上下文变量：标记是否在定时任务回调中执行
        # 用于防止在定时任务中再创建定时任务（避免无限循环）
        self._in_cron_context: ContextVar[bool] = ContextVar("cron_in_context", default=False)

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current session context for delivery."""
        # 设置当前会话上下文，用于发送提醒消息
        # channel: 频道名称（如 telegram、discord）
        # chat_id: 聊天 ID
        self._channel = channel
        self._chat_id = chat_id

    def set_cron_context(self, active: bool):
        """Mark whether the tool is executing inside a cron job callback."""
        # 标记是否在定时任务回调中执行
        # active: 是否处于定时任务上下文
        # 返回 token，用于后续重置
        return self._in_cron_context.set(active)

    def reset_cron_context(self, token) -> None:
        """Restore previous cron context."""
        # 恢复之前的定时任务上下文
        # token: set_cron_context 返回的 token
        self._in_cron_context.reset(token)

    @property
    def name(self) -> str:
        # 工具名称
        return "cron"

    @property
    def description(self) -> str:
        # 工具描述
        return "Schedule reminders and recurring tasks. Actions: add, list, remove."

    @property
    def parameters(self) -> dict[str, Any]:
        # 工具参数定义（JSON Schema 格式）
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove"],  # 允许的操作
                    "description": "Action to perform",
                },
                "message": {"type": "string", "description": "Reminder message (for add)"},  # 提醒消息
                "every_seconds": {
                    "type": "integer",
                    "description": "Interval in seconds (for recurring tasks)",  # 间隔秒数
                },
                "cron_expr": {
                    "type": "string",
                    "description": "Cron expression like '0 9 * * *' (for scheduled tasks)",  # Cron 表达式
                },
                "tz": {
                    "type": "string",
                    "description": "IANA timezone for cron expressions (e.g. 'America/Vancouver')",  # 时区
                },
                "at": {
                    "type": "string",
                    "description": "ISO datetime for one-time execution (e.g. '2026-02-12T10:30:00')",  # 一次性执行时间
                },
                "job_id": {"type": "string", "description": "Job ID (for remove)"},  # 任务 ID
            },
            "required": ["action"],  # 必需参数：action
        }

    async def execute(
        self,
        action: str,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        # 执行定时任务操作
        # 根据 action 参数分发到不同的处理方法
        
        if action == "add":
            # 添加任务
            if self._in_cron_context.get():
                # 如果在定时任务回调中，禁止创建新任务（防止无限循环）
                return "Error: cannot schedule new jobs from within a cron job execution"
            return self._add_job(message, every_seconds, cron_expr, tz, at)
        elif action == "list":
            # 列出任务
            return self._list_jobs()
        elif action == "remove":
            # 移除任务
            return self._remove_job(job_id)
        return f"Unknown action: {action}"  # 未知操作

    def _add_job(
        self,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
    ) -> str:
        # 添加定时任务
        # message: 提醒消息
        # every_seconds: 间隔秒数（周期性任务）
        # cron_expr: Cron 表达式（定时任务）
        # tz: 时区
        # at: 一次性执行时间
        
        if not message:
            return "Error: message is required for add"  # 消息是必需的
        if not self._channel or not self._chat_id:
            return "Error: no session context (channel/chat_id)"  # 需要会话上下文
        if tz and not cron_expr:
            return "Error: tz can only be used with cron_expr"  # 时区只能和 Cron 表达式一起使用
        if tz:
            from zoneinfo import ZoneInfo

            try:
                ZoneInfo(tz)  # 验证时区是否有效
            except (KeyError, Exception):
                return f"Error: unknown timezone '{tz}'"

        # 构建定时计划
        delete_after = False  # 是否执行后删除
        if every_seconds:
            # 周期性任务：每 N 秒执行一次
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            # Cron 定时任务：按 Cron 表达式执行
            schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
        elif at:
            # 一次性任务：在指定时间执行
            from datetime import datetime

            try:
                dt = datetime.fromisoformat(at)  # 解析 ISO 格式时间
            except ValueError:
                return f"Error: invalid ISO datetime format '{at}'. Expected format: YYYY-MM-DDTHH:MM:SS"
            at_ms = int(dt.timestamp() * 1000)  # 转换为毫秒时间戳
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            delete_after = True  # 一次性任务执行后删除
        else:
            return "Error: either every_seconds, cron_expr, or at is required"

        # 添加任务到定时服务
        job = self._cron.add_job(
            name=message[:30],  # 任务名称（取消息前 30 字符）
            schedule=schedule,
            message=message,
            deliver=True,  # 需要发送消息
            channel=self._channel,
            to=self._chat_id,
            delete_after_run=delete_after,
        )
        return f"Created job '{job.name}' (id: {job.id})"

    def _list_jobs(self) -> str:
        # 列出所有定时任务
        jobs = self._cron.list_jobs()  # 获取任务列表
        if not jobs:
            return "No scheduled jobs."  # 没有任务
        lines = [f"- {j.name} (id: {j.id}, {j.schedule.kind})" for j in jobs]  # 格式化每行
        return "Scheduled jobs:\n" + "\n".join(lines)

    def _remove_job(self, job_id: str | None) -> str:
        # 移除定时任务
        # job_id: 任务 ID
        if not job_id:
            return "Error: job_id is required for remove"  # 需要任务 ID
        if self._cron.remove_job(job_id):
            return f"Removed job {job_id}"
        return f"Job {job_id} not found"
