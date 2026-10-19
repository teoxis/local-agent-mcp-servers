"""Hybrid index (keywords + meaning) over memory, dossiers and documents.

A single SQLite file: FTS5 for literal matches (names, paths, IPs) and
Qwen3-Embedding vectors for paraphrases ("the laptop" ~ "the Mac"). Both
rankings are merged with Reciprocal Rank Fusion, which needs no score
calibration. An optional cross-encoder reranker reorders the top candidates.

If the embeddings server does not respond, search degrades to keywords only
and SAYS SO in the result: a half search must not look like a complete one.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.request
from pathlib import Path

import numpy as np

EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")
EMBED_KEY = os.environ.get("EMBED_KEY", "")
RERANK_URL = os.environ.get("RERANK_URL", "http://127.0.0.1:8082/v1/rerank")
TEXT = {".md", ".txt", ".py", ".sh", ".js", ".ts", ".json", ".yml", ".yaml", ".toml", ".ini", ".conf",
        ".csv", ".html", ".sql", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".ps1", ".xml", ".log"}
CONVERTIBLE = {".pdf", ".docx", ".xlsx", ".pptx", ".epub", ".doc", ".xls"}
IGNORE = re.compile(r"/INDEX\.md$|/(\.git|node_modules|\.venv|venv|__pycache__|site-packages|\.cache|\.index)(/|$)")
MAX_FILE = 5_000_000
# Qwen3-Embedding: queries carry an instruction, documents do not.
INSTRUCTION = "Instruct: Given a query, retrieve the passages that answer it\nQuery: "


class Index:
    def __init__(self, db: Path):
        db.parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(db)
        self.c.executescript("""
            CREATE TABLE IF NOT EXISTS docs(path TEXT PRIMARY KEY, mtime REAL, scope TEXT);
            CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, path TEXT, line INT, text TEXT, emb BLOB);
            CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, content='chunks', content_rowid='id',
                                                              tokenize='unicode61 remove_diacritics 2');
        """)
        self.warning = ""

    # --- embeddings --------------------------------------------------------
    def _embed(self, texts: list[str], query: bool = False) -> np.ndarray | None:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        inputs = [(INSTRUCTION + t if query else t)[:6000] + "<|endoftext|>" for t in texts]
        vecs = []
        try:
            for i in range(0, len(inputs), 16):
                req = urllib.request.Request(EMBED_URL, data=json.dumps({"input": inputs[i:i + 16]}).encode(),
                                             headers={"Content-Type": "application/json",
                                                      **({"Authorization": f"Bearer {EMBED_KEY}"} if EMBED_KEY else {})})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = json.load(r)["data"]
                vecs += [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
        except Exception as e:  # noqa: BLE001
            self.warning = f"WARNING: embeddings server unavailable ({type(e).__name__}); KEYWORD-ONLY search."
            return None
        m = np.asarray(vecs, dtype=np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    # --- indexing ----------------------------------------------------------
    @staticmethod
    def _read(p: Path) -> str | None:
        if p.stat().st_size > MAX_FILE:
            return None
        if p.suffix.lower() in CONVERTIBLE:
            try:
                from markitdown import MarkItDown
                return MarkItDown().convert(str(p)).text_content
            except Exception:  # noqa: BLE001
                return None
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    @staticmethod
    def _chunk_lines(text: str, lines: int = 30, overlap: int = 5) -> list[tuple[int, str]]:
        ls = text.splitlines()
        out, i = [], 0
        while i < len(ls):
            block = "\n".join(ls[i:i + lines]).strip()
            if block:
                # Huge chunks (very long lines, minified files) are cut by characters.
                for k in range(0, len(block), 2000):
                    out.append((i + 1, block[k:k + 2000]))
            i += lines - overlap
        return out

    @staticmethod
    def _chunk_prose(text: str, size: int = 500, overlap: int = 100) -> list[tuple[int, str]]:
        """Converted documents (PDF, Word…): their line breaks are layout, not meaning.

        Normalizes whitespace (justified PDFs carry double spaces that confuse embeddings and the
        reranker), drops table-of-contents lines with dot leaders and lone page numbers, and groups
        paragraphs into ~size-character chunks with overlap. Measured on a 60-page Spanish law PDF with
        14 questions whose answer appears verbatim: 30-line chunks 9/14 · 1200 chars 7/14 · 800 11/14 ·
        600 11/14 · 500 13/14 · 400 12/14 · 300 10/14.
        """
        paragraphs, current, first_line = [], [], 1
        for n, l in enumerate(text.splitlines(), 1):
            l = re.sub(r"[ \t ]+", " ", l).strip()
            if re.search(r"(\. ?){6,}", l) or re.fullmatch(r"\d{1,4}", l):
                continue   # table of contents with dot leaders, or a page number
            if not l:
                if current:
                    paragraphs.append((first_line, " ".join(current))); current = []
                continue
            if not current:
                first_line = n
            current.append(l)
        if current:
            paragraphs.append((first_line, " ".join(current)))
        out, buf, start = [], "", 1
        for n, p in paragraphs:
            if buf and len(buf) + len(p) > size:
                out.append((start, buf))
                tail = buf[-overlap:]
                buf, start = tail[tail.find(" ") + 1:] + " " + p, n
            else:
                buf, start = (buf + " " + p if buf else p), (start if buf else n)
            while len(buf) > 2 * size:   # giant paragraph: cut by characters
                out.append((start, buf[:size])); buf = buf[size - overlap:]
        if buf.strip():
            out.append((start, buf))
        return out

    def index(self, root: Path, scope: str) -> dict[str, int]:
        root = root.expanduser().resolve()
        files = [root] if root.is_file() else [
            p for p in root.rglob("*") if p.is_file() and not IGNORE.search(str(p))
            and p.suffix.lower() in TEXT | CONVERTIBLE]
        seen, reindexed, chunks = set(), 0, 0
        for p in files:
            path = str(p)
            seen.add(path)
            mt = p.stat().st_mtime
            row = self.c.execute("SELECT mtime FROM docs WHERE path=?", (path,)).fetchone()
            if row and row[0] == mt:
                continue
            text = self._read(p)
            if text is None:
                continue
            cs = self._chunk_prose(text) if p.suffix.lower() in CONVERTIBLE else self._chunk_lines(text)
            embs = self._embed([f"{p.name}\n{t}" for _, t in cs])
            self._delete(path)
            for j, (line, t) in enumerate(cs):
                cur = self.c.execute("INSERT INTO chunks(path, line, text, emb) VALUES(?,?,?,?)",
                                     (path, line, t, embs[j].tobytes() if embs is not None else None))
                self.c.execute("INSERT INTO fts(rowid, text) VALUES(?,?)", (cur.lastrowid, f"{p.name} {t}"))
            # Without embeddings the file is not marked as indexed: it is retried when the server is back.
            self.c.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?)", (path, mt if embs is not None else 0, scope))
            reindexed += 1
            chunks += len(cs)
        # Files deleted from disk under this root
        prefix = str(root) if root.is_dir() else None
        if prefix:
            for (path,) in self.c.execute("SELECT path FROM docs WHERE path LIKE ?", (prefix + "%",)).fetchall():
                if path not in seen:
                    self._delete(path)
                    self.c.execute("DELETE FROM docs WHERE path=?", (path,))
        self.c.commit()
        return {"files": len(files), "reindexed": reindexed, "chunks": chunks}

    def _delete(self, path: str) -> None:
        for (i, t) in self.c.execute("SELECT id, text FROM chunks WHERE path=?", (path,)).fetchall():
            self.c.execute("INSERT INTO fts(fts, rowid, text) VALUES('delete', ?, ?)", (i, f"{Path(path).name} {t}"))
        self.c.execute("DELETE FROM chunks WHERE path=?", (path,))

    # --- search ------------------------------------------------------------
    def _rerank(self, query: str, ids: list[int]) -> list[int]:
        """Cross-encoder reranking of the candidates; if it does not respond, the RRF order stays."""
        if len(ids) < 2 or not RERANK_URL:
            return ids
        # Normalized whitespace: justified PDFs carry double spaces that sink the reranker score
        # (measured: the same chunk scored 0.30 as-is and 0.85 normalized).
        texts = [re.sub(r"\s+", " ", self.c.execute("SELECT text FROM chunks WHERE id=?", (i,)).fetchone()[0])[:1500]
                 for i in ids]
        try:
            req = urllib.request.Request(RERANK_URL, data=json.dumps({"query": query, "documents": texts}).encode(),
                                         headers={"Content-Type": "application/json",
                                                  **({"Authorization": f"Bearer {EMBED_KEY}"} if EMBED_KEY else {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                res = json.load(r)["results"]
        except Exception as e:  # noqa: BLE001
            self.warning += f" WARNING: reranker unavailable ({type(e).__name__}); less precise order."
            return ids
        return [ids[x["index"]] for x in sorted(res, key=lambda x: -x["relevance_score"])]

    def search(self, query: str, k: int = 5, scopes: list[str] | None = None) -> list[dict]:
        self.warning = ""
        where, params = "", []
        if scopes:
            where = f" AND t.path IN (SELECT path FROM docs WHERE scope IN ({','.join('?' * len(scopes))}))"
            params = list(scopes)
        ranks: dict[int, float] = {}
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        if words:
            q = " OR ".join(f'"{w}"' for w in words)
            rows = self.c.execute(f"SELECT t.id FROM fts JOIN chunks t ON t.id=fts.rowid WHERE fts MATCH ?{where} "
                                  f"ORDER BY bm25(fts) LIMIT 40", [q] + params).fetchall()
            for r, (i,) in enumerate(rows):
                ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
        qv = self._embed([query], query=True)
        if qv is not None:
            rows = self.c.execute(f"SELECT t.id, t.emb FROM chunks t WHERE t.emb IS NOT NULL{where}", params).fetchall()
            if rows:
                ids = np.array([f[0] for f in rows])
                m = np.frombuffer(b"".join(f[1] for f in rows), dtype=np.float32).reshape(len(rows), -1)
                sims = m @ qv[0]
                for r, j in enumerate(np.argsort(-sims)[:40]):
                    ranks[int(ids[j])] = ranks.get(int(ids[j]), 0) + 1 / (60 + r)
        best = self._rerank(query, sorted(ranks, key=lambda i: -ranks[i])[:20])[:k]
        out = []
        for i in best:
            path, line, text = self.c.execute("SELECT path, line, text FROM chunks WHERE id=?", (i,)).fetchone()
            out.append({"path": path, "line": line, "text": text})
        return out
