"""Web tools: web_search and web_fetch."""
# 网络工具模块 - 提供网络搜索和网页获取功能

import html  # HTML 实体解码
import json  # JSON 序列化
import os  # 操作系统接口（获取环境变量）
import re  # 正则表达式
from typing import Any  # 类型提示
from urllib.parse import urlparse  # URL 解析

import httpx  # 异步 HTTP 客户端
from loguru import logger  # 结构化日志记录

from nanobot.agent.tools.base import Tool  # 导入工具基类

# 共享常量
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"  # 用户代理字符串
MAX_REDIRECTS = 5  # 最大重定向次数，防止 DoS 攻击


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    # 移除 HTML 标签并解码实体
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)  # 移除 script 标签
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)  # 移除 style 标签
    text = re.sub(r'<[^>]+>', '', text)  # 移除所有 HTML 标签
    return html.unescape(text).strip()  # 解码 HTML 实体并去除空白


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    # 规范化空白字符
    text = re.sub(r'[ \t]+', ' ', text)  # 多个空格/制表符替换为单个空格
    return re.sub(r'\n{3,}', '\n\n', text).strip()  # 多个换行替换为两个


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    # 验证 URL：必须是 http(s) 且有有效域名
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


class WebSearchTool(Tool):
    """Search the web using Brave Search API."""
    # 网络搜索工具 - 使用 Brave Search API

    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10}
        },
        "required": ["query"]
    }

    def __init__(self, api_key: str | None = None, max_results: int = 5, proxy: str | None = None):
        self._init_api_key = api_key  # 初始 API 密钥
        self.max_results = max_results  # 最大结果数
        self.proxy = proxy  # 代理设置

    @property
    def api_key(self) -> str:
        """Resolve API key at call time so env/config changes are picked up."""
        # 在调用时解析 API 密钥，以便环境/配置更改被捕获
        return self._init_api_key or os.environ.get("BRAVE_API_KEY", "")

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        if not self.api_key:
            return (
                "Error: Brave Search API key not configured. Set it in "
                "~/.nanobot/config.json under tools.web.search.apiKey "
                "(or export BRAVE_API_KEY), then restart the gateway."
            )

        try:
            n = min(max(count or self.max_results, 1), 10)  # 确定结果数量（1-10）
            logger.debug("WebSearch: {}", "proxy enabled" if self.proxy else "direct connection")
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                    timeout=10.0
                )
                r.raise_for_status()

            results = r.json().get("web", {}).get("results", [])[:n]  # 获取结果
            if not results:
                return f"No results for: {query}"

            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results, 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if desc := item.get("description"):
                    lines.append(f"   {desc}")
            return "\n".join(lines)
        except httpx.ProxyError as e:
            logger.error("WebSearch proxy error: {}", e)
            return f"Proxy error: {e}"
        except Exception as e:
            logger.error("WebSearch error: {}", e)
            return f"Error: {e}"


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Readability."""
    # 网页获取工具 - 使用 Readability 提取内容

    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100}
        },
        "required": ["url"]
    }

    def __init__(self, max_chars: int = 50000, proxy: str | None = None):
        self.max_chars = max_chars  # 最大字符数
        self.proxy = proxy  # 代理设置

    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        from readability import Document  # 延迟导入

        max_chars = maxChars or self.max_chars
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url}, ensure_ascii=False)

        try:
            logger.debug("WebFetch: {}", "proxy enabled" if self.proxy else "direct connection")
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0,
                proxy=self.proxy,
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()

            ctype = r.headers.get("content-type", "")

            if "application/json" in ctype:
                # JSON 内容：格式化输出
                text, extractor = json.dumps(r.json(), indent=2, ensure_ascii=False), "json"
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                # HTML 内容：使用 Readability 提取
                doc = Document(r.text)
                content = self._to_markdown(doc.summary()) if extractMode == "markdown" else _strip_tags(doc.summary())
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                # 其他内容：原样返回
                text, extractor = r.text, "raw"

            truncated = len(text) > max_chars
            if truncated: text = text[:max_chars]

            return json.dumps({"url": url, "finalUrl": str(r.url), "status": r.status_code,
                              "extractor": extractor, "truncated": truncated, "length": len(text), "text": text}, ensure_ascii=False)
        except httpx.ProxyError as e:
            logger.error("WebFetch proxy error for {}: {}", url, e)
            return json.dumps({"error": f"Proxy error: {e}", "url": url}, ensure_ascii=False)
        except Exception as e:
            logger.error("WebFetch error for {}: {}", url, e)
            return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)

    def _to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        # 将 HTML 转换为 Markdown
        # 在移除标签前转换链接、标题、列表
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))
