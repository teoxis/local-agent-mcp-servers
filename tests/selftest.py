#!/usr/bin/env python3
"""End-to-end checks for both MCP servers, driven through a real MCP client over stdio.

- memory: full write -> index -> read -> search -> delete cycle in a throwaway workspace.
  The embeddings/reranker URLs point to a closed port on purpose, so the run is hermetic
  and also proves the keyword-only fallback (and that it warns about it).
- websearch: tool contract always; live searches only if SearXNG answers at SEARXNG_URL.

Exit code 0 if everything passes, 1 otherwise.
Usage: python tests/selftest.py
"""

import asyncio
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT / "websearch" / "server.py"
MEM = ROOT / "memory" / "server.py"
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888")

# Reserved name: created and deleted on every run.
CANARY = "zz-test-canary"

failures: list[str] = []
steps = 0


def check(cond: bool, msg: str) -> bool:
    global steps
    steps += 1
    print(f"    {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        failures.append(msg)
    return cond


def session(script: Path, env: dict[str, str]):
    params = StdioServerParameters(command=sys.executable, args=[str(script)], env=env)
    return stdio_client(params)


async def call(s: ClientSession, tool: str, args: dict) -> str:
    res = await s.call_tool(tool, args)
    return "".join(getattr(c, "text", "") for c in res.content)


def searxng_up() -> bool:
    try:
        with urllib.request.urlopen(SEARXNG_URL, timeout=3):
            return True
    except Exception:  # noqa: BLE001
        return False


async def test_websearch() -> None:
    print("[websearch]")
    async with session(WS, dict(os.environ)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            check(names == {"web_search", "web_fetch", "doc_read"},
                  f"exposes web_search, web_fetch and doc_read (saw: {sorted(names)})")

            out = await call(s, "web_fetch", {"url": "not-a-url"})
            check("ERROR" in out, "web_fetch rejects an invalid URL instead of crashing")

            out = await call(s, "doc_read", {"path": "/definitely/missing/file.pdf"})
            check("ERROR" in out, "doc_read reports a missing file cleanly")

            if not searxng_up():
                print(f"    skip live searches: no SearXNG at {SEARXNG_URL}")
                return
            out = await call(s, "web_search", {"query": "Model Context Protocol", "max_results": 3})
            check("ERROR" not in out, "web_search returns no error")
            check("http" in out, "web_search returns at least one URL")

            # A real, stable page: example.com does not always resolve and used to give a false
            # negative that blamed web_fetch for a network failure.
            out = await call(s, "web_fetch", {
                "url": "https://en.wikipedia.org/wiki/Model_Context_Protocol", "max_chars": 2000})
            check("Content of" in out, "web_fetch extracts text from a real page")


async def test_memory(workspace: Path) -> None:
    print("[memory]")
    env = dict(os.environ, AGENT_WORKSPACE=str(workspace),
               EMBED_URL="http://127.0.0.1:9/v1/embeddings", RERANK_URL="http://127.0.0.1:9/v1/rerank")
    async with session(MEM, env) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            expected = {
                "memory_index", "memory_load_core", "memory_read", "memory_write", "memory_search",
                "memory_delete", "knowledge_index", "knowledge_search", "dossier",
                "scratch_append", "scratch_read", "scratch_clear",
            }
            check(names == expected, f"exposes the {len(expected)} tools "
                                     f"(missing: {sorted(expected - names)}, extra: {sorted(names - expected)})")

            # Full cycle: write -> index -> read -> search -> delete.
            out = await call(s, "memory_write", {
                "name": CANARY,
                "content": "Unique verification phrase: xyzzy-plugh-42.",
                "description": "Temporary memory created by the test suite.",
                "kind": "reference",
            })
            written = check("created" in out or "updated" in out, "memory_write creates the memory")

            await call(s, "memory_write", {
                "name": "zz-test-user", "content": "Prefers short answers.",
                "description": "User preference used by the test suite.", "kind": "user"})

            out = await call(s, "memory_index", {})
            check(CANARY in out, "memory_index lists the newly created memory")

            out = await call(s, "memory_read", {"name": CANARY})
            check("xyzzy-plugh-42" in out, "memory_read returns the exact content")

            out = await call(s, "memory_search", {"query": "xyzzy-plugh-42"})
            check(CANARY in out, "memory_search finds it by content")
            check("KEYWORD-ONLY" in out, "memory_search warns when it falls back to keywords")

            # This tool exists so the agent never has to pick or declare anything:
            # it returns the content, not a list of names.
            out = await call(s, "memory_load_core", {})
            check("Prefers short answers." in out, "memory_load_core returns real content, not just names")
            check(CANARY not in out, "memory_load_core does not drag in memories of kind reference")

            out = await call(s, "memory_write", {"name": CANARY, "content": "x", "description": "y", "kind": "made-up"})
            check("ERROR" in out, "memory_write rejects an invalid kind")

            out = await call(s, "memory_write", {"name": "../../escape", "content": "x", "description": "y",
                                                 "kind": "project"})
            check(not (workspace.parent / "escape.md").exists() and (workspace / "memory" / "escape.md").exists(),
                  "memory names cannot escape the workspace (path traversal)")

            out = await call(s, "memory_read", {"name": "never-exists-xyz"})
            check("does not exist" in out, "memory_read reports cleanly when missing")

            # Scratch buffer.
            out = await call(s, "scratch_append", {"note": "test canary"})
            check("saved" in out.lower(), "scratch_append writes to the buffer")
            out = await call(s, "scratch_read", {})
            check("test canary" in out, "scratch_read gets back what was written")

            # Dossiers.
            out = await call(s, "dossier", {"action": "open", "name": "Test dossier", "text": "verify dossiers"})
            check("opened" in out, "dossier open")
            out = await call(s, "dossier", {"action": "note", "name": "Test dossier", "text": "first note"})
            check("updated" in out, "dossier note")
            out = await call(s, "dossier", {"action": "list"})
            check("test-dossier (open" in out, "dossier list shows it as open")

            if written:
                out = await call(s, "memory_delete", {"name": CANARY})
                check("deleted" in out, "memory_delete removes the canary")
                out = await call(s, "memory_index", {})
                check(CANARY not in out, "the canary no longer appears in the index")


def test_chunking() -> None:
    print("[semantic]")
    sys.path.insert(0, str(ROOT / "memory"))
    from semantic import Index
    text = "Contents . . . . . . . 3\n\n12\n\nFirst  paragraph   of\nthe document.\n\nSecond paragraph."
    chunks = Index._chunk_prose(text)
    joined = " ".join(t for _, t in chunks)
    check("First paragraph of the document." in joined, "prose chunker joins layout line breaks and normalizes spaces")
    check(". . . . ." not in joined and " 12 " not in f" {joined} ", "prose chunker drops TOC dot leaders and page numbers")
    long = "\n".join(f"line {i}" for i in range(100))
    starts = [line for line, _ in Index._chunk_lines(long)]
    check(starts[:3] == [1, 26, 51], f"line chunker uses 30-line windows with 5 lines of overlap (starts: {starts[:3]})")


async def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="mcp-selftest-")) / "workspace"
    try:
        test_chunking()
        await test_websearch()
        await test_memory(workspace)
    except Exception as exc:  # noqa: BLE001
        print(f"    FAIL unhandled exception: {type(exc).__name__}: {exc}")
        failures.append(str(exc))
    finally:
        shutil.rmtree(workspace.parent, ignore_errors=True)

    print(f"\n  {steps - len(failures)}/{steps} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
