"""Agent loop: the core processing engine."""
# Agent 主循环模块 - 核心处理引擎
# 负责接收消息、调用 LLM、执行工具、返回响应的完整循环

from __future__ import annotations  # 启用 postponed evaluation of annotations（PEP 563）

import asyncio  # 异步 I/O 支持
import json  # JSON 序列化/反序列化
import os  # 操作系统接口（用于重启）
import re  # 正则表达式
import sys  # 系统相关参数和函数（用于重启）
from contextlib import AsyncExitStack  # 异步上下文管理器栈，用于 MCP 连接
from pathlib import Path  # 文件路径处理
from typing import TYPE_CHECKING, Any, Awaitable, Callable  # 类型提示

from loguru import logger  # 结构化日志记录

# 导入 Agent 核心组件
from nanobot.agent.context import ContextBuilder  # 上下文构建器
from nanobot.agent.memory import MemoryConsolidator  # 记忆整合器
from nanobot.agent.subagent import SubagentManager  # 子代理管理器

# 导入各种工具实现
from nanobot.agent.tools.cron import CronTool  # 定时任务工具
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool  # 文件系统工具
from nanobot.agent.tools.message import MessageTool  # 消息发送工具
from nanobot.agent.tools.registry import ToolRegistry  # 工具注册表
from nanobot.agent.tools.shell import ExecTool  # 命令执行工具
from nanobot.agent.tools.spawn import SpawnTool  # 子代理启动工具
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool  # 网络工具

# 导入消息总线相关
from nanobot.bus.events import InboundMessage, OutboundMessage  # 入站/出站消息
from nanobot.bus.queue import MessageBus  # 消息总线

# 导入 LLM 提供商基类
from nanobot.providers.base import LLMProvider

# 导入会话管理
from nanobot.session.manager import Session, SessionManager

# TYPE_CHECKING 为 True 时导入的类型仅用于类型检查，运行时不会导入
if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig  # 配置类型
    from nanobot.cron.service import CronService  # 定时服务


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus  # 从消息总线接收消息
    2. Builds context with history, memory, skills  # 构建包含历史、记忆、技能的上下文
    3. Calls the LLM  # 调用大语言模型
    4. Executes tool calls  # 执行工具调用
    5. Sends responses back  # 发送响应回去
    """

    # 工具结果的最大字符数，超过会被截断
    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,  # 消息总线实例
        provider: LLMProvider,  # LLM 提供商实例
        workspace: Path,  # 工作空间路径
        model: str | None = None,  # 使用的模型名称，None 则使用提供商默认模型
        max_iterations: int = 40,  # 最大工具调用迭代次数，防止无限循环
        context_window_tokens: int = 65_536,  # 上下文窗口的 token 限制
        brave_api_key: str | None = None,  # Brave 搜索 API 密钥
        web_proxy: str | None = None,  # 网络代理设置
        exec_config: ExecToolConfig | None = None,  # 命令执行工具配置
        cron_service: CronService | None = None,  # 定时任务服务
        restrict_to_workspace: bool = False,  # 是否限制文件操作在工作空间内
        session_manager: SessionManager | None = None,  # 会话管理器
        mcp_servers: dict | None = None,  # MCP 服务器配置
        channels_config: ChannelsConfig | None = None,  # 频道配置
    ):
        from nanobot.config.schema import ExecToolConfig  # 延迟导入避免循环依赖
        self.bus = bus  # 保存消息总线引用
        self.channels_config = channels_config  # 频道配置
        self.provider = provider  # LLM 提供商
        self.workspace = workspace  # 工作空间路径
        self.model = model or provider.get_default_model()  # 使用指定模型或默认模型
        self.max_iterations = max_iterations  # 最大迭代次数
        self.context_window_tokens = context_window_tokens  # 上下文 token 限制
        self.brave_api_key = brave_api_key  # Brave API 密钥
        self.web_proxy = web_proxy  # 网络代理
        self.exec_config = exec_config or ExecToolConfig()  # 命令执行配置，默认空配置
        self.cron_service = cron_service  # 定时服务
        self.restrict_to_workspace = restrict_to_workspace  # 是否限制在工作空间

        # 初始化核心组件
        self.context = ContextBuilder(workspace)  # 上下文构建器
        self.sessions = session_manager or SessionManager(workspace)  # 会话管理器
        self.tools = ToolRegistry()  # 工具注册表
        
        # 初始化子代理管理器，传入必要的依赖
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        # 运行状态标志
        self._running = False  # 是否正在运行
        
        # MCP 相关状态
        self._mcp_servers = mcp_servers or {}  # MCP 服务器配置
        self._mcp_stack: AsyncExitStack | None = None  # MCP 连接的上下文栈
        self._mcp_connected = False  # 是否已连接 MCP
        self._mcp_connecting = False  # 是否正在连接 MCP
        
        # 任务管理
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> 任务列表
        self._processing_lock = asyncio.Lock()  # 处理锁，确保单条消息顺序处理
        
        # 记忆整合器，用于自动归档旧消息
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,  # 传入构建消息的方法
            get_tool_definitions=self.tools.get_definitions,  # 传入获取工具定义的方法
        )
        
        # 注册默认工具
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # 注册默认工具集到工具注册表
        
        # 如果限制在工作空间内，则设置允许目录为工作空间，否则为 None（无限制）
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        
        # 注册文件系统工具：读、写、编辑、列目录
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        
        # 注册命令执行工具
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),  # 工作目录
            timeout=self.exec_config.timeout,  # 超时时间
            restrict_to_workspace=self.restrict_to_workspace,  # 是否限制工作空间
            path_append=self.exec_config.path_append,  # 附加 PATH
        ))
        
        # 注册网络搜索工具
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        # 注册网页获取工具
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        # 注册消息发送工具，传入发布出站消息的回调
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        # 注册子代理启动工具
        self.tools.register(SpawnTool(manager=self.subagents))
        
        # 如果有定时服务，注册定时任务工具
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        # 连接到配置的 MCP 服务器（一次性、延迟连接）
        # MCP = Model Context Protocol，用于连接外部工具服务器
        
        # 如果已连接、正在连接、或没有配置服务器，则直接返回
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        
        self._mcp_connecting = True  # 标记正在连接
        from nanobot.agent.tools.mcp import connect_mcp_servers  # 延迟导入
        try:
            self._mcp_stack = AsyncExitStack()  # 创建异步上下文栈
            await self._mcp_stack.__aenter__()  # 进入上下文
            # 连接所有 MCP 服务器，将工具注册到注册表
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True  # 标记已连接
        except Exception as e:
            # 连接失败，记录错误，下次消息时会重试
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()  # 关闭上下文栈
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False  # 重置连接中标志

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        # 为需要路由信息的工具设置上下文
        # channel: 频道名称
        # chat_id: 聊天 ID
        # message_id: 消息 ID（可选）
        
        # 遍历需要设置上下文的工具
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):  # 获取工具实例
                if hasattr(tool, "set_context"):  # 如果工具有 set_context 方法
                    # 调用 set_context，message 工具额外传入 message_id
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        # 移除某些模型（如 DeepSeek）在内容中嵌入的 <think> 思考块
        if not text:
            return None
        # 使用正则表达式移除 <think>...</think> 及其内容
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        # 将工具调用格式化为简洁的提示，例如 'web_search("query")'
        def _fmt(tc):
            # 获取参数，处理列表或字典格式
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            # 获取第一个参数值
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):  # 如果不是字符串，只返回工具名
                return tc.name
            # 截断过长的参数值
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)  # 连接所有工具调用提示

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],  # 初始消息列表
        on_progress: Callable[..., Awaitable[None]] | None = None,  # 进度回调函数
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop."""
        # 运行 Agent 迭代循环
        # 返回: (最终回复内容, 使用的工具列表, 所有消息列表)
        
        messages = initial_messages  # 当前消息列表
        iteration = 0  # 迭代计数器
        final_content = None  # 最终回复内容
        tools_used: list[str] = []  # 记录使用的工具

        # 循环直到达到最大迭代次数
        while iteration < self.max_iterations:
            iteration += 1

            # 获取当前所有工具的定义（用于 LLM function calling）
            tool_defs = self.tools.get_definitions()

            # 调用 LLM，传入消息和工具定义
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs,
                model=self.model,
            )

            # 如果 LLM 调用了工具
            if response.has_tool_calls:
                # 如果有进度回调，发送思考内容和工具提示
                if on_progress:
                    thought = self._strip_think(response.content)  # 提取思考内容
                    if thought:
                        await on_progress(thought)  # 发送思考内容
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)  # 发送工具提示

                # 将工具调用转换为 OpenAI 格式
                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                # 添加助手消息（包含工具调用）到消息列表
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                # 执行每个工具调用
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)  # 记录工具名
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)  # 序列化参数
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])  # 记录日志
                    # 执行工具并获取结果
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    # 将工具结果添加到消息列表
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # LLM 没有调用工具，返回最终回复
                clean = self._strip_think(response.content)  # 清理思考块
                # 如果完成原因是错误，不保存到会话历史（避免污染上下文）
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                # 添加助手消息到列表
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean  # 保存最终内容
                break  # 退出循环

        # 如果达到最大迭代次数仍未完成
        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        # 运行 Agent 循环，将消息作为任务分发以响应 /stop 命令
        self._running = True  # 设置运行标志
        await self._connect_mcp()  # 连接 MCP 服务器
        logger.info("Agent loop started")  # 记录启动日志

        # 主循环
        while self._running:
            try:
                # 等待入站消息，超时 1 秒以便检查运行状态
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # 超时，继续循环

            # 处理命令
            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)  # 处理停止命令
            elif cmd == "/restart":
                await self._handle_restart(msg)  # 处理重启命令
            else:
                # 创建任务处理普通消息
                task = asyncio.create_task(self._dispatch(msg))
                # 记录到活动任务列表
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                # 任务完成时从列表移除
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        # 处理 /stop 命令：取消会话的所有活动任务和子代理
        
        # 获取并移除该会话的所有活动任务
        tasks = self._active_tasks.pop(msg.session_key, [])
        # 取消未完成的任务
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        # 等待所有任务完成（或取消）
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        # 取消该会话的子代理
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled  # 总取消数
        content = f"Stopped {total} task(s)." if total else "No active task to stop."
        # 发送响应
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_restart(self, msg: InboundMessage) -> None:
        """Restart the process in-place via os.execv."""
        # 处理 /restart 命令：通过 os.execv 原地重启进程
        
        # 发送重启中消息
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        # 定义异步重启函数
        async def _do_restart():
            await asyncio.sleep(1)  # 等待 1 秒确保消息发送
            os.execv(sys.executable, [sys.executable] + sys.argv)  # 执行新进程替换当前进程

        asyncio.create_task(_do_restart())  # 创建重启任务

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        # 在全局锁下处理消息
        async with self._processing_lock:  # 获取处理锁
            try:
                response = await self._process_message(msg)  # 处理消息
                if response is not None:  # 如果有响应
                    await self.bus.publish_outbound(response)  # 发送响应
                elif msg.channel == "cli":  # CLI 模式下即使没有内容也发送空响应
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:  # 任务被取消
                logger.info("Task cancelled for session {}", msg.session_key)
                raise  # 重新抛出以便上层处理
            except Exception:  # 其他异常
                logger.exception("Error processing message for session {}", msg.session_key)
                # 发送错误响应
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        # 关闭 MCP 连接
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()  # 关闭上下文栈
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK 的取消范围清理可能会产生噪音但无害
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        # 停止 Agent 循环
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,  # 入站消息
        session_key: str | None = None,  # 会话密钥（可选，用于覆盖）
        on_progress: Callable[[str], Awaitable[None]] | None = None,  # 进度回调
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # 处理单条入站消息并返回响应
        
        # 系统消息处理：从 chat_id 解析 origin（格式: "channel:chat_id"）
        if msg.channel == "system":
            # 解析频道和聊天 ID
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"  # 构建会话密钥
            session = self.sessions.get_or_create(key)  # 获取或创建会话
            # 检查是否需要整合记忆
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            # 设置工具上下文
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            # 获取历史消息（max_messages=0 表示获取全部）
            history = session.get_history(max_messages=0)
            # 构建消息列表
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            # 运行 Agent 循环
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            # 保存本轮对话
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)  # 保存会话
            # 再次检查是否需要整合记忆
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            # 返回出站消息
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        # 记录消息预览（前 80 字符）
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # 获取会话密钥和会话对象
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # 处理斜杠命令
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # /new 命令：开始新会话，归档未整合的记忆
            try:
                if not await self.memory_consolidator.archive_unconsolidated(session):
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Memory archival failed, session not cleared. Please try again.",
                    )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )

            session.clear()  # 清空会话
            self.sessions.save(session)  # 保存
            self.sessions.invalidate(session.key)  # 使缓存失效
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        
        if cmd == "/help":
            # /help 命令：显示帮助信息
            lines = [
                "🐈 nanobot commands:",
                "/new — Start a new conversation",
                "/stop — Stop the current task",
                "/restart — Restart the bot",
                "/help — Show available commands",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )
        
        # 检查是否需要整合记忆（token 超限）
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        # 设置工具上下文
        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        # 如果存在消息工具，开始新一轮
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        # 获取历史消息
        history = session.get_history(max_messages=0)
        # 构建初始消息列表
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,  # 媒体文件
            channel=msg.channel, chat_id=msg.chat_id,
        )

        # 定义进度回调函数（用于发送进度更新）
        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})  # 复制元数据
            meta["_progress"] = True  # 标记为进度消息
            meta["_tool_hint"] = tool_hint  # 标记是否为工具提示
            # 发送进度消息
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        # 运行 Agent 循环，获取最终回复
        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        # 如果最终内容为空，设置默认消息
        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # 保存本轮对话
        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)  # 保存会话
        # 再次检查是否需要整合记忆
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        # 检查本轮是否发送过消息（通过 message 工具）
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None  # 如果已发送过消息，不再发送响应

        # 记录响应预览
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        # 返回出站消息
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        # 保存新一轮的消息到会话，截断过大的工具结果
        # session: 会话对象
        # messages: 所有消息列表
        # skip: 跳过的消息数（历史消息）
        
        from datetime import datetime  # 导入 datetime 用于时间戳
        
        # 遍历新消息（跳过历史消息）
        for m in messages[skip:]:
            entry = dict(m)  # 复制消息字典
            role, content = entry.get("role"), entry.get("content")
            
            # 跳过空的助手消息（它们会污染会话上下文）
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue
            
            # 截断过长的工具结果
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            
            # 处理用户消息：移除运行时上下文前缀，处理多模态内容
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # 从字符串内容中剥离运行时上下文前缀
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]  # 保留用户文本
                    else:
                        continue  # 没有实际内容，跳过
                
                # 处理多模态内容（列表格式）
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        # 从多模态消息中剥离运行时上下文
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # 跳过运行时上下文块
                        # 将 base64 图片替换为 [image] 占位符
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)  # 保留其他内容
                    if not filtered:  # 如果没有有效内容
                        continue
                    entry["content"] = filtered  # 更新为过滤后的内容
            
            # 添加时间戳（如果不存在）
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)  # 添加到会话消息列表
        
        session.updated_at = datetime.now()  # 更新会话更新时间

    async def process_direct(
        self,
        content: str,  # 消息内容
        session_key: str = "cli:direct",  # 会话密钥，默认为 CLI 直接模式
        channel: str = "cli",  # 频道，默认为 CLI
        chat_id: str = "direct",  # 聊天 ID
        on_progress: Callable[[str], Awaitable[None]] | None = None,  # 进度回调
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        # 直接处理消息（用于 CLI 或定时任务）
        # 不通过消息总线，直接返回字符串结果
        
        await self._connect_mcp()  # 确保 MCP 已连接
        # 创建入站消息对象
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        # 处理消息
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""  # 返回内容或空字符串
