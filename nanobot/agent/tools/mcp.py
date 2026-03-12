"""MCP client: connects to MCP servers and wraps their tools as native nanobot tools."""
# MCP 客户端模块 - 连接到 MCP 服务器并将其工具包装为 nanobot 原生工具
# MCP = Model Context Protocol，用于连接外部工具服务器

import asyncio  # 异步 I/O 支持
from contextlib import AsyncExitStack  # 异步上下文管理器栈
from typing import Any  # 类型提示：任意类型

import httpx  # 异步 HTTP 客户端
from loguru import logger  # 结构化日志记录

from nanobot.agent.tools.base import Tool  # 导入工具基类
from nanobot.agent.tools.registry import ToolRegistry  # 导入工具注册表


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as a nanobot Tool."""
    # MCP 工具包装器 - 将单个 MCP 服务器工具包装为 nanobot 工具

    def __init__(self, session, server_name: str, tool_def, tool_timeout: int = 30):
        # 构造函数
        # session: MCP 客户端会话
        # server_name: MCP 服务器名称
        # tool_def: MCP 工具定义
        # tool_timeout: 工具调用超时时间（秒）
        self._session = session  # 保存 MCP 会话
        self._original_name = tool_def.name  # 原始工具名称
        self._name = f"mcp_{server_name}_{tool_def.name}"  # 包装后的工具名称（添加前缀）
        self._description = tool_def.description or tool_def.name  # 工具描述
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}  # 参数模式
        self._tool_timeout = tool_timeout  # 超时时间

    @property
    def name(self) -> str:
        # 工具名称（带 mcp_ 前缀）
        return self._name

    @property
    def description(self) -> str:
        # 工具描述
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        # 工具参数定义
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        # 执行 MCP 工具
        # kwargs: 工具参数
        from mcp import types  # 延迟导入

        try:
            # 调用 MCP 工具，带超时
            result = await asyncio.wait_for(
                self._session.call_tool(self._original_name, arguments=kwargs),
                timeout=self._tool_timeout,
            )
        except asyncio.TimeoutError:
            # 超时处理
            logger.warning("MCP tool '{}' timed out after {}s", self._name, self._tool_timeout)
            return f"(MCP tool call timed out after {self._tool_timeout}s)"
        except asyncio.CancelledError:
            # MCP SDK 的 anyio 取消范围可能在超时/失败时泄漏 CancelledError
            # 仅当任务被外部取消（如 /stop）时才重新抛出
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            logger.warning("MCP tool '{}' was cancelled by server/SDK", self._name)
            return "(MCP tool call was cancelled)"
        except Exception as exc:
            # 其他异常
            logger.exception(
                "MCP tool '{}' failed: {}: {}",
                self._name,
                type(exc).__name__,
                exc,
            )
            return f"(MCP tool call failed: {type(exc).__name__})"

        # 处理结果
        parts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                # 文本内容
                parts.append(block.text)
            else:
                # 其他类型转为字符串
                parts.append(str(block))
        return "\n".join(parts) or "(no output)"


async def connect_mcp_servers(
    mcp_servers: dict, registry: ToolRegistry, stack: AsyncExitStack
) -> None:
    """Connect to configured MCP servers and register their tools."""
    # 连接到配置的 MCP 服务器并注册其工具
    # mcp_servers: MCP 服务器配置字典
    # registry: 工具注册表
    # stack: 异步上下文栈，用于管理连接生命周期
    
    from mcp import ClientSession, StdioServerParameters  # MCP 客户端会话和 stdio 参数
    from mcp.client.sse import sse_client  # SSE 传输客户端
    from mcp.client.stdio import stdio_client  # stdio 传输客户端
    from mcp.client.streamable_http import streamable_http_client  # HTTP 流式传输客户端

    for name, cfg in mcp_servers.items():
        # 遍历每个 MCP 服务器配置
        try:
            # 确定传输类型
            transport_type = cfg.type
            if not transport_type:
                # 如果没有指定类型，根据配置推断
                if cfg.command:
                    transport_type = "stdio"  # 有命令则为 stdio
                elif cfg.url:
                    # 约定：URL 以 /sse 结尾使用 SSE 传输；其他使用 streamableHttp
                    transport_type = (
                        "sse" if cfg.url.rstrip("/").endswith("/sse") else "streamableHttp"
                    )
                else:
                    logger.warning("MCP server '{}': no command or url configured, skipping", name)
                    continue

            if transport_type == "stdio":
                # stdio 传输：通过子进程通信
                params = StdioServerParameters(
                    command=cfg.command, args=cfg.args, env=cfg.env or None
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif transport_type == "sse":
                # SSE 传输：服务器发送事件
                def httpx_client_factory(
                    headers: dict[str, str] | None = None,
                    timeout: httpx.Timeout | None = None,
                    auth: httpx.Auth | None = None,
                ) -> httpx.AsyncClient:
                    # 合并配置的 headers 和传入的 headers
                    merged_headers = {**(cfg.headers or {}), **(headers or {})}
                    return httpx.AsyncClient(
                        headers=merged_headers or None,
                        follow_redirects=True,
                        timeout=timeout,
                        auth=auth,
                    )

                read, write = await stack.enter_async_context(
                    sse_client(cfg.url, httpx_client_factory=httpx_client_factory)
                )
            elif transport_type == "streamableHttp":
                # HTTP 流式传输
                # 始终提供显式的 httpx 客户端，使 MCP HTTP 传输不会继承 httpx 的默认 5s 超时
                http_client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=cfg.headers or None,
                        follow_redirects=True,
                        timeout=None,  # 无超时，由上层控制
                    )
                )
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(cfg.url, http_client=http_client)
                )
            else:
                logger.warning("MCP server '{}': unknown transport type '{}'", name, transport_type)
                continue

            # 创建 MCP 会话
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()  # 初始化会话

            # 列出并注册工具
            tools = await session.list_tools()
            for tool_def in tools.tools:
                wrapper = MCPToolWrapper(session, name, tool_def, tool_timeout=cfg.tool_timeout)
                registry.register(wrapper)  # 注册到工具注册表
                logger.debug("MCP: registered tool '{}' from server '{}'", wrapper.name, name)

            logger.info("MCP server '{}': connected, {} tools registered", name, len(tools.tools))
        except Exception as e:
            logger.error("MCP server '{}': failed to connect: {}", name, e)
