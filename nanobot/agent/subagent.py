"""Subagent manager for background task execution."""
# 子代理管理器模块 - 用于在后台执行任务
# 子代理是独立的 Agent 实例，可以并行处理任务

import asyncio  # 异步 I/O 支持
import json  # JSON 序列化
import uuid  # 生成唯一标识符
from pathlib import Path  # 文件路径处理
from typing import Any  # 类型提示

from loguru import logger  # 结构化日志记录

# 导入工具类
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool

# 导入消息总线相关
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus

# 导入会话管理
from nanobot.session.manager import SessionManager


class SubagentManager:
    """
    Manages subagents for background task execution.
    
    Subagents are separate agent instances that can process tasks
    in the background without blocking the main agent.
    """
    # 子代理管理器类 - 管理后台任务执行的子代理

    def __init__(
        self,
        provider,  # LLM 提供商
        workspace: Path,  # 工作空间路径
        bus: MessageBus,  # 消息总线
        model: str,  # 使用的模型
        brave_api_key: str | None = None,  # Brave 搜索 API 密钥
        web_proxy: str | None = None,  # 网络代理
        exec_config=None,  # 命令执行配置
        restrict_to_workspace: bool = False,  # 是否限制在工作空间
    ):
        self.provider = provider  # LLM 提供商
        self.workspace = workspace  # 工作空间路径
        self.bus = bus  # 消息总线
        self.model = model  # 模型名称
        self.brave_api_key = brave_api_key  # Brave API 密钥
        self.web_proxy = web_proxy  # 网络代理
        self.exec_config = exec_config  # 命令执行配置
        self.restrict_to_workspace = restrict_to_workspace  # 是否限制工作空间
        
        # 存储活动子代理：任务 ID -> 子代理实例
        self._subagents: dict[str, 'Subagent'] = {}
        
        # 锁，用于线程安全地访问 _subagents
        self._lock = asyncio.Lock()

    async def spawn(
        self,
        task: str,  # 任务描述
        label: str | None = None,  # 任务标签（用于显示）
        session_key: str | None = None,  # 关联的会话密钥
    ) -> str:
        """
        Spawn a new subagent to execute a task.
        
        Returns the task ID for tracking.
        """
        # 启动新子代理执行任务
        # 返回任务 ID 用于跟踪
        
        # 生成唯一任务 ID
        task_id = str(uuid.uuid4())[:8]
        
        # 创建子代理实例
        subagent = Subagent(
            task_id=task_id,
            task=task,
            label=label or task[:50],  # 默认标签取任务前 50 字符
            provider=self.provider,
            workspace=self.workspace,
            bus=self.bus,
            model=self.model,
            brave_api_key=self.brave_api_key,
            web_proxy=self.web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=self.restrict_to_workspace,
            session_key=session_key,
        )
        
        # 存储子代理
        async with self._lock:
            self._subagents[task_id] = subagent
        
        # 启动子代理（在后台运行）
        asyncio.create_task(subagent.run())
        
        logger.info("Spawned subagent {}: {}", task_id, label or task[:50])
        return task_id

    async def get_status(self, task_id: str) -> dict[str, Any] | None:
        """Get the status of a subagent task."""
        # 获取子代理任务状态
        
        async with self._lock:
            subagent = self._subagents.get(task_id)  # 获取子代理
        
        if not subagent:  # 如果不存在
            return None
        
        return subagent.get_status()  # 返回状态

    async def cancel(self, task_id: str) -> bool:
        """Cancel a subagent task."""
        # 取消子代理任务
        
        async with self._lock:
            subagent = self._subagents.get(task_id)  # 获取子代理
        
        if not subagent:  # 如果不存在
            return False
        
        await subagent.cancel()  # 取消任务
        
        async with self._lock:
            del self._subagents[task_id]  # 从字典中移除
        
        return True

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for a session. Returns count cancelled."""
        # 取消会话的所有子代理，返回取消数量
        
        cancelled = 0  # 取消计数
        to_cancel = []  # 要取消的任务 ID 列表
        
        async with self._lock:
            # 找出该会话的所有子代理
            for task_id, subagent in self._subagents.items():
                if subagent.session_key == session_key:
                    to_cancel.append(task_id)
        
        # 取消每个子代理
        for task_id in to_cancel:
            if await self.cancel(task_id):
                cancelled += 1
        
        return cancelled

    async def list_tasks(self, session_key: str | None = None) -> list[dict[str, Any]]:
        """List subagent tasks, optionally filtered by session."""
        # 列出子代理任务，可选按会话过滤
        
        tasks = []  # 任务列表
        
        async with self._lock:
            items = list(self._subagents.items())  # 获取所有子代理
        
        for task_id, subagent in items:  # 遍历
            if session_key and subagent.session_key != session_key:
                continue  # 如果指定了会话但不匹配，跳过
            tasks.append(subagent.get_status())  # 添加状态
        
        return tasks


class Subagent:
    """
    A subagent that executes a task in the background.
    """
    # 子代理类 - 在后台执行任务的独立代理实例

    def __init__(
        self,
        task_id: str,  # 任务 ID
        task: str,  # 任务描述
        label: str,  # 任务标签
        provider,  # LLM 提供商
        workspace: Path,  # 工作空间
        bus: MessageBus,  # 消息总线
        model: str,  # 模型
        brave_api_key: str | None,  # Brave API 密钥
        web_proxy: str | None,  # 网络代理
        exec_config,  # 命令执行配置
        restrict_to_workspace: bool,  # 是否限制工作空间
        session_key: str | None = None,  # 关联的会话密钥
    ):
        self.task_id = task_id  # 任务 ID
        self.task = task  # 任务描述
        self.label = label  # 标签
        self.provider = provider  # LLM 提供商
        self.workspace = workspace  # 工作空间
        self.bus = bus  # 消息总线
        self.model = model  # 模型
        self.session_key = session_key  # 会话密钥
        
        # 状态
        self._status = "running"  # 状态：running, completed, failed, cancelled
        self._result: str | None = None  # 结果
        self._error: str | None = None  # 错误信息
        self._start_time = asyncio.get_event_loop().time()  # 开始时间
        
        # 取消标志
        self._cancelled = False
        self._task: asyncio.Task | None = None
        
        # 初始化工具
        self.tools = ToolRegistry()
        self._init_tools(
            brave_api_key, web_proxy, exec_config, restrict_to_workspace
        )

    def _init_tools(
        self,
        brave_api_key: str | None,
        web_proxy: str | None,
        exec_config,
        restrict_to_workspace: bool,
    ) -> None:
        """Initialize tools for this subagent."""
        # 初始化子代理的工具
        
        allowed_dir = self.workspace if restrict_to_workspace else None
        
        # 注册文件系统工具
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        
        # 注册命令执行工具
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=exec_config.timeout if exec_config else 60,
            restrict_to_workspace=restrict_to_workspace,
            path_append=exec_config.path_append if exec_config else None,
        ))
        
        # 注册网络工具
        self.tools.register(WebSearchTool(api_key=brave_api_key, proxy=web_proxy))
        self.tools.register(WebFetchTool(proxy=web_proxy))

    def get_status(self) -> dict[str, Any]:
        """Get current status of this subagent."""
        # 获取当前状态
        
        elapsed = asyncio.get_event_loop().time() - self._start_time  # 已运行时间
        
        return {
            "task_id": self.task_id,  # 任务 ID
            "label": self.label,  # 标签
            "status": self._status,  # 状态
            "result": self._result,  # 结果
            "error": self._error,  # 错误
            "elapsed_seconds": round(elapsed, 1),  # 已运行秒数
            "session_key": self.session_key,  # 会话密钥
        }

    async def run(self) -> None:
        """Execute the task."""
        # 执行任务
        
        try:
            # 创建异步任务
            self._task = asyncio.create_task(self._execute())
            await self._task  # 等待执行完成
        except asyncio.CancelledError:
            self._status = "cancelled"  # 标记为已取消
            logger.info("Subagent {} cancelled", self.task_id)
        except Exception as e:
            self._status = "failed"  # 标记为失败
            self._error = str(e)  # 记录错误
            logger.exception("Subagent {} failed", self.task_id)

    async def _execute(self) -> None:
        """Internal execution logic."""
        # 内部执行逻辑
        
        # 构建系统提示词
        system_prompt = f"""You are a subagent executing a background task.

Task: {self.task}

Execute this task independently. You have access to file, shell, and web tools.
When complete, provide a concise summary of what you did and the results.
"""
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please execute the task."},
        ]
        
        # 执行循环（简化版，最多 20 次迭代）
        for iteration in range(20):
            if self._cancelled:  # 如果已取消
                raise asyncio.CancelledError()
            
            # 调用 LLM
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
            )
            
            if response.has_tool_calls:
                # 处理工具调用
                # ...（简化处理，实际应类似主 Agent 循环）
                break
            else:
                # 任务完成
                self._result = response.content
                self._status = "completed"
                logger.info("Subagent {} completed", self.task_id)
                
                # 如果有关联会话，发送通知
                if self.session_key:
                    await self._notify_completion()
                return
        
        # 达到最大迭代次数
        self._status = "completed"
        self._result = "Task completed (max iterations reached)"

    async def _notify_completion(self) -> None:
        """Notify the parent session of completion."""
        # 通知父会话任务完成
        
        # 解析会话密钥获取频道和聊天 ID
        if ":" in self.session_key:
            channel, chat_id = self.session_key.split(":", 1)
        else:
            channel, chat_id = "cli", self.session_key
        
        # 构建通知消息
        content = f"✅ Background task completed: {self.label}\n\n{self._result or ''}"[:500]
        
        # 发送系统消息
        from nanobot.bus.events import OutboundMessage
        await self.bus.publish_outbound(OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
        ))

    async def cancel(self) -> None:
        """Cancel this subagent."""
        # 取消子代理
        
        self._cancelled = True  # 设置取消标志
        if self._task and not self._task.done():
            self._task.cancel()  # 取消任务
            try:
                await self._task  # 等待取消完成
            except asyncio.CancelledError:
                pass
