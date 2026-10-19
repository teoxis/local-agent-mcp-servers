#!/usr/bin/env python3
"""Persistent memory MCP server for a local coding agent.

A local model has a small context window. This server turns the disk into
external memory, with two clearly separated levels:

  LONG-TERM MEMORY (memory/)   -> durable facts, one file per fact.
                                  Survives everything. Indexed in INDEX.md.
  SCRATCH BUFFER   (.scratch/) -> notes for the task in progress, per day.
                                  This is what the agent re-reads when its
                                  context was reset halfway through a long task.

Plus "dossiers" (dossiers/): semi-permanent working documents, one per project
or topic, that expire after 30 days without activity.

Every path resolves inside AGENT_WORKSPACE: no memory name can escape the
directory.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mcp.server.mcpserver import MCPServer

WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", "~/.agent-memory")).expanduser()
MEM_DIR = WORKSPACE / "memory"
BUF_DIR = WORKSPACE / ".scratch"
DOSSIER_DIR = WORKSPACE / "dossiers"
INDEX = MEM_DIR / "INDEX.md"

VALID_KINDS = ("user", "feedback", "project", "reference")
# Memories that memory_load_core should skip, e.g. because they are already in the
# agent's system prompt (loading them again only duplicates context).
CORE_EXCLUDE = {s.strip() for s in os.environ.get("MEMORY_CORE_EXCLUDE", "").split(",") if s.strip()}
MAX_BUFFER_CHARS = 60_000  # the daily journal rotates beyond this size

server = MCPServer(
    name="memory",
    instructions=(
        "Disk memory. memory/ stores durable facts; .scratch/ stores notes for "
        "the task in progress that survive a context reset."
    ),
)


def _slug(name: str) -> str:
    """Normalizes a memory name to a safe slug (no path traversal)."""
    s = re.sub(r"[^\w\s-]", "", name.strip().lower(), flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s[:80] or "unnamed"


def _ensure_dirs() -> None:
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    BUF_DIR.mkdir(parents=True, exist_ok=True)


def _today_journal() -> Path:
    day = dt.date.today().isoformat()
    d = BUF_DIR / day
    d.mkdir(parents=True, exist_ok=True)
    return d / "journal.md"


def _parse_front(text: str) -> dict[str, str]:
    """Extracts the simple YAML front matter (name/description/kind) of a file."""
    meta: dict[str, str] = {}
    if not text.startswith("---"):
        return meta
    end = text.find("\n---", 3)
    if end == -1:
        return meta
    for line in text[3:end].splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"')
    return meta


# --------------------------------------------------------------------------
# Long-term memory
# --------------------------------------------------------------------------


@server.tool()
def memory_index() -> str:
    """Lists all long-term memory, one line per entry. Cheap on context.

    Call this at the start of a session to know what you remember.
    """
    _ensure_dirs()
    files = sorted(p for p in MEM_DIR.glob("*.md") if p.name != "INDEX.md")
    if not files:
        return "Memory is empty. Use memory_write to store the first fact."

    lines: list[str] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            lines.append(f"- {p.stem} — [unreadable: {exc}]")
            continue
        meta = _parse_front(text)
        desc = meta.get("description", "")
        if not desc:
            body = text.split("---", 2)[-1].strip()
            desc = " ".join(body.split())[:100]
        kind = meta.get("kind", meta.get("type", "?"))
        lines.append(f"- [{kind}] {p.stem} — {desc}")

    return f"{len(files)} memories:\n" + "\n".join(lines)


@server.tool()
def memory_load_core() -> str:
    """Loads the FULL content of the memories that always apply.

    Call this at the start of a session. Returns, in one go, every memory of kind
    `user` and `feedback` (who the user is, how to work with them, their
    preferences). Nothing to choose or declare: the content comes in the response.
    """
    _ensure_dirs()
    parts: list[str] = []
    for f in sorted(MEM_DIR.glob("*.md")):
        if f.name == "INDEX.md" or f.stem in CORE_EXCLUDE:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        if _parse_front(text).get("kind") in ("user", "feedback"):
            parts.append(f"===== {f.stem} =====\n{text.strip()}")

    if not parts:
        return "There are no memories of kind user or feedback yet."
    return f"{len(parts)} memories that always apply:\n\n" + "\n\n".join(parts)


@server.tool()
def memory_read(name: str) -> str:
    """Reads a full memory by name (the slug shown by memory_index)."""
    _ensure_dirs()
    p = MEM_DIR / f"{_slug(name)}.md"
    if not p.exists():
        # Tolerate the model using the original name with capitals/accents.
        cands = [q for q in MEM_DIR.glob("*.md") if _slug(q.stem) == _slug(name)]
        if not cands:
            return f"Memory '{name}' does not exist. Use memory_index to see the available ones."
        p = cands[0]
    return p.read_text(encoding="utf-8")


@server.tool()
def memory_write(name: str, content: str, description: str, kind: str = "project") -> str:
    """Creates or updates a durable memory. One fact per memory.

    Args:
        name: Short name, e.g. 'terminal-preference'.
        content: The fact itself, in markdown. Include the why if it is a lesson.
        description: One-line summary (used in the index to decide whether to read it).
        kind: user | feedback | project | reference
    """
    _ensure_dirs()
    kind = kind.strip().lower()
    if kind not in VALID_KINDS:
        return f"ERROR: kind must be one of {', '.join(VALID_KINDS)} (got: '{kind}')"

    slug = _slug(name)
    p = MEM_DIR / f"{slug}.md"
    existed = p.exists()
    doc = (
        "---\n"
        f"name: {slug}\n"
        f"description: {' '.join(description.split())}\n"
        f"kind: {kind}\n"
        f"updated: {dt.date.today().isoformat()}\n"
        "---\n\n"
        f"{content.strip()}\n"
    )
    p.write_text(doc, encoding="utf-8")
    _rebuild_index()
    verb = "updated" if existed else "created"
    _commit(f"memory: {verb} '{slug}' ({kind})")
    return f"Memory '{slug}' {verb} ({len(doc)} bytes)."


_index = None


def _idx():
    """Lazy hybrid index: memory/ and dossiers/ are re-indexed (incrementally) on every search."""
    global _index
    from semantic import Index
    if _index is None:
        _index = Index(WORKSPACE / ".index" / "semantic.db")
    return _index


def _format(hits: list[dict], warning: str) -> str:
    if not hits:
        return (warning + "\n" if warning else "") + "No results. Do not make the fact up: say so."
    out = [warning] if warning else []
    for h in hits:
        rel = h["path"].replace(str(WORKSPACE) + "/", "")
        out.append(f"--- {rel}:{h['line']}\n{h['text'][:700]}")
    return "\n".join(out)


@server.tool()
def memory_search(query: str, k: int = 5) -> str:
    """Searches long-term memory and dossiers by MEANING and by keywords.

    Finds paraphrases ("the laptop" -> "the Mac"). Returns snippets with file:line.
    """
    _ensure_dirs()
    if not query.strip():
        return "ERROR: empty query."
    ix = _idx()
    ix.index(MEM_DIR, "memory")
    DOSSIER_DIR.mkdir(exist_ok=True)
    ix.index(DOSSIER_DIR, "dossiers")
    hits = ix.search(query, k, ["memory", "dossiers"])
    return _format(hits, ix.warning)


@server.tool()
def knowledge_index(path: str, scope: str = "docs") -> str:
    """Adds a folder or file (md, code, PDF, Word, Excel...) to the searchable knowledge base.

    Incremental: repeating it only processes what changed. `scope` groups documents (e.g. 'project-x').
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"ERROR: {path} does not exist"
    r = _idx().index(p, scope)
    warning = _idx().warning
    return (warning + "\n" if warning else "") + (f"Indexed '{scope}': {r['files']} files seen, "
                                                  f"{r['reindexed']} processed, {r['chunks']} new chunks.")


@server.tool()
def knowledge_search(query: str, scope: str = "", k: int = 5) -> str:
    """Searches the indexed knowledge base (by meaning and keywords). Empty `scope` = everything."""
    ix = _idx()
    hits = ix.search(query, k, [scope] if scope else None)
    return _format(hits, ix.warning)


@server.tool()
def dossier(action: str, name: str = "", text: str = "") -> str:
    """Semi-permanent working documents, one per project or topic (memory for weeks).

    action: 'list' | 'open' (text=goal) | 'note' (text=note) | 'read' | 'close' (text=conclusion).
    They expire after 30 days without activity: listing flags them for review.
    """
    DOSSIER_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    if action == "list":
        rows = []
        for p in sorted(DOSSIER_DIR.glob("*.md")):
            m = _parse_front(p.read_text(encoding="utf-8"))
            upd = dt.date.fromisoformat(m.get("updated", today.isoformat()))
            flag = " [STALE: review or close]" if m.get("status") == "open" and (today - upd).days > 30 else ""
            rows.append(f"- {p.stem} ({m.get('status', '?')}, {m.get('updated', '?')}){flag}: {m.get('goal', '')}")
        return "\n".join(rows) or "There are no dossiers."
    slug = _slug(name)
    if not name.strip():
        return "ERROR: missing name."
    f = DOSSIER_DIR / f"{slug}.md"
    if action == "open":
        if f.exists():
            return f"'{slug}' already exists. Use read or note."
        f.write_text(f"---\nstatus: open\ngoal: {text.strip()}\ncreated: {today}\nupdated: {today}\n---\n\n"
                     f"# {name}\n\n**Goal:** {text.strip()}\n\n## Notes\n", encoding="utf-8")
        return f"Dossier '{slug}' opened."
    if not f.exists():
        return f"Dossier '{slug}' does not exist. Use dossier('list')."
    doc = f.read_text(encoding="utf-8")
    if action == "read":
        return doc[-12000:]
    if action in ("note", "close"):
        if not text.strip():
            return "ERROR: empty text."
        doc = re.sub(r"^updated: .*$", f"updated: {today}", doc, count=1, flags=re.M)
        if action == "close":
            doc = re.sub(r"^status: .*$", "status: closed", doc, count=1, flags=re.M)
            doc += f"\n## Conclusion ({today})\n{text.strip()}\n"
        else:
            doc += f"- [{dt.datetime.now():%Y-%m-%d %H:%M}] {text.strip()}\n"
        f.write_text(doc, encoding="utf-8")
        return f"Dossier '{slug}' {'closed' if action == 'close' else 'updated'}."
    return "ERROR: unknown action (list|open|note|read|close)."


@server.tool()
def memory_delete(name: str) -> str:
    """Deletes a memory that turned out to be wrong or has become obsolete."""
    _ensure_dirs()
    p = MEM_DIR / f"{_slug(name)}.md"
    if not p.exists():
        return f"Memory '{name}' does not exist."
    p.unlink()
    _rebuild_index()
    _commit(f"memory: deleted '{_slug(name)}'")
    return f"Memory '{_slug(name)}' deleted."


def _commit(message: str) -> None:
    """Versions the change if the workspace is a git repository.

    It is done here and not in a harness hook because this is the only place
    every memory change goes through: deterministic, without depending on any
    event being emitted.
    """
    if not (WORKSPACE / ".git").is_dir():
        return
    try:
        # The agent's own instructions (AGENTS.md) are as irreplaceable as the memories: versioned too.
        subprocess.run(["git", "add", "-A", "memory", "dossiers", "AGENTS.md"], cwd=WORKSPACE,
                       check=False, capture_output=True, timeout=10)
        # If there are no changes git exits with 1 and nothing happens:
        # no noise and no exceptions because of that.
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=WORKSPACE,
                       check=False, capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001 - versioning must never break a write
        pass


def _rebuild_index() -> None:
    """Regenerates INDEX.md. It is a derived artifact: never edited by hand."""
    files = sorted(p for p in MEM_DIR.glob("*.md") if p.name != "INDEX.md")
    lines = [
        "# Memory index",
        "",
        "<!-- Generated by the memory MCP server. Do not edit by hand. -->",
        "",
    ]
    for p in files:
        try:
            meta = _parse_front(p.read_text(encoding="utf-8"))
        except OSError:
            continue
        lines.append(f"- [{meta.get('kind', '?')}] **{p.stem}** — {meta.get('description', '')}")
    INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Scratch buffer
# --------------------------------------------------------------------------


@server.tool()
def scratch_append(note: str) -> str:
    """Appends a note to today's scratch buffer.

    Use it during long tasks so you don't lose findings, decisions, or the exact
    point where you left off if the context gets reset. It is cheap: do it often.
    """
    _ensure_dirs()
    j = _today_journal()
    stamp = dt.datetime.now().strftime("%H:%M")
    with j.open("a", encoding="utf-8") as fh:
        fh.write(f"- [{stamp}] {note.strip()}\n")

    size = j.stat().st_size
    if size > MAX_BUFFER_CHARS:
        rotated = j.with_name(f"journal-{dt.datetime.now().strftime('%H%M%S')}.md")
        j.rename(rotated)
        return f"Note saved. Buffer rotated to {rotated.name} (exceeded {MAX_BUFFER_CHARS} bytes)."
    return f"Note saved to today's buffer ({size} bytes)."


@server.tool()
def scratch_read(day: str = "") -> str:
    """Re-reads the scratch buffer. Without an argument it reads today's.

    Args:
        day: ISO date (YYYY-MM-DD). Empty = today.
    """
    _ensure_dirs()
    target = day.strip() or dt.date.today().isoformat()
    d = BUF_DIR / target
    if not d.exists():
        return f"No buffer for {target}."
    parts: list[str] = []
    for p in sorted(d.glob("*.md")):
        parts.append(f"## {p.name}\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) if parts else f"Buffer for {target} is empty."


@server.tool()
def scratch_clear() -> str:
    """Empties today's scratch buffer. Use it when closing a task, not before."""
    _ensure_dirs()
    d = BUF_DIR / dt.date.today().isoformat()
    if not d.exists():
        return "Today's buffer is already empty."
    n = 0
    for p in d.glob("*.md"):
        p.unlink()
        n += 1
    return f"Today's buffer emptied ({n} files)."


if __name__ == "__main__":
    _ensure_dirs()
    try:
        server.run()
    except KeyboardInterrupt:
        sys.exit(0)
