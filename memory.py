"""Long-term memory: SQLite (source of truth) + ChromaDB (semantic search)."""

import sqlite3
from datetime import datetime

import chromadb

from rag_config import config, get_embedding_function


class JarvisMemory:
    def __init__(self, rag_config=None):
        self.cfg = rag_config or config
        self._init_sqlite()
        self.chroma = chromadb.PersistentClient(path=self.cfg.chroma_path)
        self.collection = self.chroma.get_or_create_collection(
            name=self.cfg.collection_name,
            embedding_function=get_embedding_function(self.cfg.embedding_model),
        )
        self._sync_sqlite_to_chroma()

    def _connect(self):
        return sqlite3.connect(self.cfg.sqlite_path)

    def _init_sqlite(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_text TEXT NOT NULL,
                jarvis_response TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _doc_text(self, user_text, jarvis_response):
        return self.cfg.chunk_template.format(user=user_text, jarvis=jarvis_response)

    def _sync_sqlite_to_chroma(self):
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, timestamp, user_text, jarvis_response FROM conversations ORDER BY id"
        ).fetchall()
        conn.close()

        if not rows:
            return

        existing = set()
        if self.collection.count() > 0:
            existing = set(self.collection.get(include=[])["ids"])

        ids, docs, metas = [], [], []
        for row_id, ts, user_text, jarvis_response in rows:
            doc_id = str(row_id)
            if doc_id in existing:
                continue
            ids.append(doc_id)
            docs.append(self._doc_text(user_text, jarvis_response))
            metas.append({
                "timestamp": ts,
                "user_text": user_text,
                "jarvis_response": jarvis_response,
            })

        if ids:
            self.collection.add(ids=ids, documents=docs, metadatas=metas)
            print(f"[Memory] Synced {len(ids)} past conversation(s) into ChromaDB")

    def save(self, user_text, jarvis_response):
        ts = datetime.now().isoformat()
        conn = self._connect()
        cur = conn.execute(
            "INSERT INTO conversations (timestamp, user_text, jarvis_response) VALUES (?, ?, ?)",
            (ts, user_text, jarvis_response),
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()

        self.collection.add(
            ids=[str(row_id)],
            documents=[self._doc_text(user_text, jarvis_response)],
            metadatas=[{
                "timestamp": ts,
                "user_text": user_text,
                "jarvis_response": jarvis_response,
            }],
        )
        return row_id

    def get_recent(self, limit=None):
        limit = limit if limit is not None else self.cfg.recent_turns
        conn = self._connect()
        rows = conn.execute(
            "SELECT user_text, jarvis_response FROM conversations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [{"user": row[0], "jarvis": row[1]} for row in reversed(rows)]

    def search_relevant(self, query, n_results=None):
        n_results = n_results if n_results is not None else self.cfg.semantic_top_k
        if self.collection.count() == 0:
            return []

        count = min(n_results, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=count)

        memories = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = None
            if results.get("distances") and results["distances"][0]:
                distance = results["distances"][0][i]
            memories.append({
                "id": doc_id,
                "user": meta.get("user_text", ""),
                "jarvis": meta.get("jarvis_response", ""),
                "timestamp": meta.get("timestamp", ""),
                "distance": distance,
            })
        return memories

    def build_system_instruction(self, query):
        base = self.cfg.jarvis_persona
        memories = self.search_relevant(query)
        if not memories:
            return base

        lines = []
        for m in memories:
            lines.append(self.cfg.memory_line_template.format(
                date=m["timestamp"][:10],
                user=m["user"],
                jarvis=m["jarvis"],
            ))
        memory_block = "\n".join(lines)
        print(f"[Memory] Retrieved {len(memories)} relevant past conversation(s)")
        return f"{base}\n\n{self.cfg.memory_instruction}\n{memory_block}"
