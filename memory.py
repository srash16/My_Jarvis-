"""Long-term memory: SQLite (source of truth) + ChromaDB (semantic search)."""

import re
import sqlite3
from datetime import datetime

import chromadb

from rag_config import config, get_embedding_function


def _part_text(content) -> str:
    """Extract plain text from a Gemini-style content dict or string."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts") or []
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("text"):
            texts.append(part["text"])
        elif isinstance(part, str):
            texts.append(part)
    return " ".join(texts).strip()


def _looks_like_clarifying_question(text: str) -> bool:
    t = text.lower()
    cues = (
        "which", "what kind", "what type", "could you tell", "let me know",
        "are you looking", "do you mean", "be more specific", "more specific",
        "interested in", "prefer", "choose", "option", "catch up on",
    )
    return any(c in t for c in cues) or ("?" in text and len(text) > 40)


def _looks_like_short_followup(text: str) -> bool:
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) <= 8:
        return True
    starters = (
        "that one", "the first", "the second", "the third", "yes", "yeah",
        "no", "about ", "tell me about", "regarding", "related to",
    )
    t = text.lower().strip()
    return any(t.startswith(s) for s in starters)


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

    def build_retrieval_query(self, current_query: str, conversation_history=None) -> str:
        """Rewrite short follow-ups using recent dialogue so RAG matches the open thread."""
        current = (current_query or "").strip()
        history = conversation_history or []

        # Parse recent user/model texts from live history (excluding the just-appended current user turn)
        turns = []
        for item in history:
            role = item.get("role") if isinstance(item, dict) else None
            text = _part_text(item)
            if role and text:
                turns.append((role, text))

        # Drop the trailing user turn if it matches current_query (already appended in process_command)
        if turns and turns[-1][0] == "user" and turns[-1][1].strip().lower() == current.lower():
            turns = turns[:-1]

        recent = turns[-(self.cfg.context_turns_for_query * 2):]
        last_model = ""
        last_user = ""
        for role, text in reversed(recent):
            if role == "model" and not last_model:
                last_model = text
            elif role == "user" and not last_user:
                last_user = text
            if last_model and last_user:
                break

        clarifying = last_model and _looks_like_clarifying_question(last_model)
        followup = _looks_like_short_followup(current)

        if clarifying and followup and last_user:
            # e.g. user asked news → JARVIS asked topic → user said Hollywood
            # retrieval should target "news update hollywood", not "hollywoodlywood definition"
            rewritten = (
                f"Continuing prior request. Earlier user ask: {last_user}. "
                f"User follow-up answer: {current}. "
                f"Topic focus: {current} related to: {last_user}"
            )
            print(f"[Memory] Contextual retrieval query (follow-up): {rewritten[:160]}...")
            return rewritten

        if recent and followup:
            ctx_bits = [t for _, t in recent[-4:]]
            rewritten = " | ".join(ctx_bits + [current])
            print(f"[Memory] Contextual retrieval query: {rewritten[:160]}...")
            return rewritten

        return current

    def search_relevant(self, query, n_results=None, exclude_texts=None):
        n_results = n_results if n_results is not None else self.cfg.semantic_top_k
        if self.collection.count() == 0:
            return []

        # Fetch extra then filter duplicates against live conversation
        fetch = min(n_results + 4, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=fetch)

        exclude = {t.strip().lower() for t in (exclude_texts or []) if t and t.strip()}
        memories = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            user = meta.get("user_text", "")
            jarvis = meta.get("jarvis_response", "")
            if user.strip().lower() in exclude or jarvis.strip().lower() in exclude:
                continue
            distance = None
            if results.get("distances") and results["distances"][0]:
                distance = results["distances"][0][i]
            memories.append({
                "id": doc_id,
                "user": user,
                "jarvis": jarvis,
                "timestamp": meta.get("timestamp", ""),
                "distance": distance,
            })
            if len(memories) >= n_results:
                break
        return memories

    def build_system_instruction(self, query, conversation_history=None):
        base = self.cfg.jarvis_persona
        history = conversation_history or []

        # Live-history texts to avoid re-injecting via RAG
        exclude = []
        for item in history[-12:]:
            text = _part_text(item)
            if text:
                exclude.append(text)

        retrieval_query = self.build_retrieval_query(query, history)
        memories = self.search_relevant(retrieval_query, exclude_texts=exclude)

        # Extra follow-up nudge when last JARVIS turn was a clarifying question
        followup_note = ""
        turns = [(item.get("role"), _part_text(item)) for item in history if isinstance(item, dict)]
        if turns and turns[-1][0] == "user":
            # previous model message is second-to-last content before current user
            for role, text in reversed(turns[:-1]):
                if role == "model" and text:
                    if _looks_like_clarifying_question(text):
                        followup_note = (
                            "\n\nActive follow-up: Your last message asked the user to clarify. "
                            f'Their latest reply was: "{query}". Interpret it as answering that '
                            "clarification (e.g. if you asked which news topic, give that topic's "
                            "news — do not redefine the topic word itself)."
                        )
                    break

        if not memories:
            return base + followup_note

        lines = []
        for m in memories:
            lines.append(self.cfg.memory_line_template.format(
                date=m["timestamp"][:10] if m["timestamp"] else "????",
                user=m["user"],
                jarvis=m["jarvis"],
            ))
        memory_block = "\n".join(lines)
        print(f"[Memory] Retrieved {len(memories)} relevant past conversation(s)")
        return (
            f"{base}{followup_note}\n\n"
            f"{self.cfg.memory_instruction}\n{memory_block}"
        )
