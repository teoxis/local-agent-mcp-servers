#!/usr/bin/env python3
"""Web search MCP server for a local coding agent.

Exposes three tools on top of a self-hosted SearXNG instance:
  - web_search: metasearch (no API key, no quotas, no telemetry)
  - web_fetch:  downloads a URL and extracts its main content as text
                (PDF and Office documents included, via MarkItDown)
  - doc_read:   converts a local document (PDF, Word, Excel, PowerPoint,
                HTML, CSV, EPUB...) to Markdown, page by page

Tool descriptions are deliberately short: they travel with every request and
a local model's context budget is scarce.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import httpx
import trafilatura
from markitdown import MarkItDown, StreamInfo
from mcp.server.mcpserver import MCPServer

# httpx logs every request at INFO level; in an MCP server that is pure noise.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("trafilatura").setLevel(logging.ERROR)

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888")
HTTP_TIMEOUT = float(os.environ.get("WEBSEARCH_TIMEOUT", "25"))

# Present as a browser: quite a few sites return 403 to generic clients.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

# Content types trafilatura does not understand but MarkItDown does.
DOC_TYPES = ("application/pdf", "officedocument", "msword", "ms-excel", "ms-powerpoint",
             "application/epub", "text/csv")
_md = MarkItDown(enable_plugins=False)


def _page(text: str, start: int, max_chars: int, source: str) -> str:
    """Split long text: the agent asks for the next chunk instead of blowing up its context."""
    total = len(text)
    chunk = text[start:start + max_chars]
    footer = (f"\n\n[characters {start}-{start + len(chunk)} of {total}. "
              f"Next chunk: start={start + len(chunk)}]" if start + len(chunk) < total else "")
    return f"# {source}\n\n{chunk}{footer}"


server = MCPServer(
    name="websearch",
    instructions=(
        "Web search via a local SearXNG. Use web_search to find sources "
        "and web_fetch to read a specific page."
    ),
)


@server.tool()
async def web_search(
    query: str,
    max_results: int = 8,
    language: str = "auto",
    time_range: str = "",
) -> str:
    """Searches the web. Returns title, URL and snippet for each result.

    Args:
        query: Search terms.
        max_results: Maximum number of results (1-20).
        language: Language code, e.g. 'es', 'en', or 'auto'.
        time_range: Empty, or one of: day, week, month, year.
    """
    max_results = max(1, min(int(max_results), 20))
    params: dict[str, str] = {"q": query, "format": "json", "safesearch": "0"}
    if language and language != "auto":
        params["language"] = language
    if time_range:
        params["time_range"] = time_range

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(f"{SEARXNG_URL}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        return (
            f"ERROR: no response from SearXNG at {SEARXNG_URL}. "
            "Start it, or point SEARXNG_URL at a running instance."
        )
    except Exception as exc:  # noqa: BLE001 - the agent needs the reason in plain words
        return f"ERROR querying SearXNG: {type(exc).__name__}: {exc}"

    results = data.get("results", [])[:max_results]
    if not results:
        return f"No results for: {query}"

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "(untitled)").strip()
        url = (r.get("url") or "").strip()
        snippet = " ".join((r.get("content") or "").split())
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        lines.append(f"{i}. {title}\n   {url}" + (f"\n   {snippet}" if snippet else ""))

    answers = data.get("answers") or []
    header = f"Direct answer: {answers[0]}\n\n" if answers else ""
    return f"{header}{len(results)} results for '{query}':\n\n" + "\n\n".join(lines)


@server.tool()
async def web_fetch(url: str, max_chars: int = 20000) -> str:
    """Downloads a URL and returns its main content as readable text.

    Args:
        url: Full URL (http:// or https://).
        max_chars: Truncates the result to this length to protect the context.
    """
    if not url.startswith(("http://", "https://")):
        return "ERROR: the URL must start with http:// or https://"

    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "").lower()
            if any(t in ctype for t in DOC_TYPES) or url.lower().split("?")[0].endswith(
                    (".pdf", ".docx", ".xlsx", ".pptx", ".epub")):
                res = _md.convert_stream(io.BytesIO(resp.content),
                                         stream_info=StreamInfo(mimetype=ctype.split(";")[0] or None,
                                                                extension=Path(url.split("?")[0]).suffix or None))
                text = (res.text_content or "").strip()
                if not text:
                    return f"The document at {url} has no extractable text (scanned PDF? needs a vision model)"
                return _page(text, 0, max(1000, int(max_chars)), f"Content of {url}")
            html = resp.text
    except httpx.HTTPStatusError as exc:
        return f"ERROR HTTP {exc.response.status_code} downloading {url}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR downloading {url}: {type(exc).__name__}: {exc}"

    text = trafilatura.extract(
        html,
        include_links=False,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if not text:
        # trafilatura fails on very dynamic pages; fall back to plain text.
        text = trafilatura.html2txt(html) or ""
    text = text.strip()
    if not text:
        return f"Could not extract readable text from {url} (JavaScript-only page?)"

    max_chars = max(1000, int(max_chars))
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    head = f"# Content of {url}\n\n"
    tail = f"\n\n[...truncated at {max_chars} characters...]" if truncated else ""
    return head + text + tail


@server.tool()
def doc_read(path: str, start: int = 0, max_chars: int = 15000) -> str:
    """Converts a local document to Markdown: PDF, Word, Excel, PowerPoint, HTML, CSV, EPUB, ZIP.

    Args:
        path: Absolute path, or starting with ~.
        start: First character; use it to read the next chunk of a long document.
        max_chars: Maximum length of the returned chunk.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return f"ERROR: file not found: {p}"
    try:
        text = (_md.convert(str(p)).text_content or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR converting {p.name}: {type(exc).__name__}: {exc}"
    if not text:
        return f"{p.name} has no extractable text (scanned PDF or image? needs a vision model)"
    return _page(text, max(0, int(start)), max(1000, int(max_chars)), p.name)


if __name__ == "__main__":
    try:
        server.run()
    except KeyboardInterrupt:
        sys.exit(0)
