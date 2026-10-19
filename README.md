# local-agent-mcp-servers

Two [Model Context Protocol](https://modelcontextprotocol.io) servers, written in Python, that give a **local** coding agent the two things it lacks most: access to the web without third-party APIs, and memory that outlives its small context window.

They run against self-hosted services only (SearXNG, llama.cpp). I built them for my own fully local agent (opencode + llama.cpp on an Apple Silicon MacBook), but they speak plain MCP over stdio. Any MCP client can use them, Claude Code included.

## Why

A model running on a laptop has a 32–128K token context window, and its tool definitions and system prompt eat a large share of it. In my setup, the system prompt plus tool definitions cost 13,805 tokens on every turn (measured October 2026): 10.5% of a 128K window, or 21% of a 64K one. That shaped two design rules:

- **Tool descriptions are deliberately short.** They are sent with every request.
- **Long content is paged, never dumped.** Large pages and documents come back in chunks with a `start=` cursor, so the agent decides how much it reads.

## The servers

### `websearch/` — web access without API keys

| Tool | What it does |
|---|---|
| `web_search` | Metasearch through a local [SearXNG](https://github.com/searxng/searxng) instance: no API key, no quotas, no telemetry |
| `web_fetch` | Downloads a URL and extracts the main content with [trafilatura](https://github.com/adbar/trafilatura). PDFs and Office files go through [MarkItDown](https://github.com/microsoft/markitdown) |
| `doc_read` | Converts a local document (PDF, Word, Excel, PowerPoint, HTML, CSV, EPUB…) to Markdown, page by page |

Errors are returned as plain sentences (`ERROR: no response from SearXNG at …`) instead of exceptions. The model needs the reason, not a stack trace.

### `memory/` — persistent memory on disk

Everything lives under `AGENT_WORKSPACE` (default `~/.agent-memory`), and memory names are slugified so nothing can escape that directory.

| Level | Tools | Purpose |
|---|---|---|
| **Long-term memory** (`memory/`) | `memory_index`, `memory_load_core`, `memory_read`, `memory_write`, `memory_delete`, `memory_search` | Durable facts, one Markdown file per fact, typed as `user`, `feedback`, `project` or `reference`. An `INDEX.md` is regenerated on every write |
| **Scratch buffer** (`.scratch/`) | `scratch_append`, `scratch_read`, `scratch_clear` | A per-day journal for the task in progress. When the context resets halfway through a long task, the agent re-reads it and carries on |
| **Dossiers** (`dossiers/`) | `dossier` (`list`/`open`/`note`/`read`/`close`) | Working documents per project that last weeks. They are flagged as stale after 30 days without activity |
| **Knowledge base** | `knowledge_index`, `knowledge_search` | Index any folder or document (code, Markdown, PDF, Office) incrementally and search it |

`memory_load_core` returns the *content* of every `user` and `feedback` memory in a single call. Small models are unreliable at "first list, then decide what to read": this removes the decision.

If the workspace is a git repository, every memory change is committed automatically. This happens inside the server, the one place every write goes through, rather than in a harness hook that might not fire.

#### Hybrid search (`memory/semantic.py`)

One SQLite file holds two indexes:

- **FTS5** for literal matches: names, paths, error codes.
- **Embedding vectors** ([Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)) for paraphrases: "the laptop" finds "the MacBook".

Both rankings are merged with **Reciprocal Rank Fusion**, so no score calibration is needed. An optional cross-encoder reranker ([Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)) reorders the top 20.

When the embeddings server is down, search **degrades to keywords and says so** in its output. A partial search must never look like a complete one. Files indexed without embeddings are retried automatically once the server comes back.

## Measured, not assumed

- **Chunking converted documents by prose rather than by lines.** On a 60-page legal PDF with 14 questions whose answer appears verbatim, correct-answer-in-top-5 went from **9/14 to 13/14**. That chunker normalizes whitespace, drops tables of contents and page numbers, and groups paragraphs into ~500-character chunks. Sizes tried: 300 → 10/14, 400 → 12/14, **500 → 13/14**, 600/800 → 11/14, 1200 → 7/14.
- **Normalizing whitespace before reranking.** Justified PDFs carry runs of spaces. The same relevant chunk scored **0.30 as-is and 0.85 normalized**.

## Running it

Requirements: Python 3.11+ and macOS or Linux (only tested on macOS).

```bash
git clone https://github.com/teoxis/local-agent-mcp-servers.git && cd local-agent-mcp-servers
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Optional local services:

```bash
# Web search: any SearXNG with the JSON output format enabled
export SEARXNG_URL=http://127.0.0.1:8888

# Semantic memory: embeddings + reranker served by llama.cpp
llama-server -m Qwen3-Embedding-0.6B-Q8_0.gguf --embeddings --pooling last --port 8081 -c 4096 -ub 512 -b 512
llama-server -m qwen3-reranker-0.6b-q8_0.gguf --reranking --port 8082 -c 8192 -ub 1024 -b 1024
```

Example client configuration (Claude Code `.mcp.json`; opencode and other clients use the same command):

```json
{
  "mcpServers": {
    "websearch": { "command": "/path/to/.venv/bin/python", "args": ["/path/to/websearch/server.py"] },
    "memory": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/memory/server.py"],
      "env": { "AGENT_WORKSPACE": "~/.agent-memory" }
    }
  }
}
```

### Configuration

| Variable | Default | |
|---|---|---|
| `SEARXNG_URL` | `http://127.0.0.1:8888` | SearXNG instance |
| `WEBSEARCH_TIMEOUT` | `25` | HTTP timeout in seconds |
| `AGENT_WORKSPACE` | `~/.agent-memory` | Root of all memory files |
| `EMBED_URL` | `http://127.0.0.1:8081/v1/embeddings` | OpenAI-compatible embeddings endpoint |
| `RERANK_URL` | `http://127.0.0.1:8082/v1/rerank` | Reranker endpoint (set it empty to disable reranking) |
| `EMBED_KEY` | *(empty)* | Bearer token for the embeddings and reranker servers |
| `MEMORY_CORE_EXCLUDE` | *(empty)* | Comma-separated memory names that `memory_load_core` should skip (e.g. ones already in the system prompt) |

## Tests

```bash
.venv/bin/python tests/selftest.py
```

The suite drives both servers through a real MCP client over stdio:
- the tool contracts;
- the full memory write → index → read → search → delete cycle in a throwaway workspace;
- path traversal, scratch buffer and dossiers;
- the chunkers;
- the keyword-only fallback (embeddings are pointed at a closed port on purpose, so the memory tests are hermetic).

Live web searches run only if SearXNG answers.

## Limitations

- Tested on macOS only. Nothing is platform-specific, but Linux is untested.
- Built against the MCP Python SDK 2.x (`mcp.server.mcpserver.MCPServer`).
- The vector search is brute-force over NumPy arrays, which is fine for a personal knowledge base. It is not designed for millions of chunks.
- `web_fetch` cannot read JavaScript-only pages, and scanned PDFs need a vision model.

## License

MIT
