"""
RAG configuration for JARVIS memory.

Architecture: Hybrid conversational RAG
  - Recent turns (sliding window) → Gemini `contents`
  - Semantic retrieval (ChromaDB)   → Gemini `system_instruction`

Tune via environment variables in `.env` or edit defaults below.
"""

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass
class RAGConfig:
    # --- Vector database (ChromaDB) ---
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "chroma_data"))
    collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION", "jarvis_conversations")
    )
    sqlite_path: str = field(default_factory=lambda: os.getenv("SQLITE_PATH", "jarvis_memory.db"))

    # --- Embedding model ---
    # Options: "default" | "minilm" | "minilm_l12"
    # "default" / "minilm" → all-MiniLM-L6-v2 (384 dims, local ONNX, Chroma built-in)
    # "minilm_l12"         → all-MiniLM-L12-v2 (384 dims, slightly heavier)
    # Changing this after data exists requires a new collection (see memory.reset_collection).
    embedding_model: str = field(
        default_factory=lambda: os.getenv("RAG_EMBEDDING_MODEL", "default").lower()
    )

    # --- Chunking (turn-level, not text splitting) ---
    # Each stored "chunk" = one full user + JARVIS exchange.
    # Template placeholders: {user}, {jarvis}
    chunk_template: str = field(
        default_factory=lambda: os.getenv(
            "RAG_CHUNK_TEMPLATE",
            "User: {user}\nJARVIS: {jarvis}",
        )
    )

    # --- Retrieval ---
    # Lean defaults (3/3) cut tokens per Gemini call vs older 5/5 — helps avoid
    # TPM stacking with RPD. Does NOT reduce request count. Override via .env if
    # context recall feels thinner: RAG_SEMANTIC_TOP_K=5 and RAG_RECENT_TURNS=5.
    semantic_top_k: int = field(default_factory=lambda: _int("RAG_SEMANTIC_TOP_K", 3))
    recent_turns: int = field(default_factory=lambda: _int("RAG_RECENT_TURNS", 3))

    # --- Augmentation (what Gemini sees) ---
    jarvis_persona: str = field(
        default_factory=lambda: os.getenv(
            "JARVIS_PERSONA",
            "You are JARVIS, a helpful and intelligent AI assistant. "
            "Be concise, smart, and slightly witty — like Tony Stark's AI.\n"
            "Conversation continuity is critical: if your previous message asked a "
            "clarifying question (topic, category, preference), treat the user's next "
            "message as the answer to that question and continue that thread. "
            "Do not restart with a generic definition of their words unless they "
            "clearly changed the subject.",
        )
    )
    memory_instruction: str = field(
        default_factory=lambda: os.getenv(
            "RAG_MEMORY_INSTRUCTION",
            "Relevant memories from past sessions (background only — never override "
            "the current conversation thread or a clarifying follow-up):\n",
        )
    )
    memory_line_template: str = field(
        default_factory=lambda: os.getenv(
            "RAG_MEMORY_LINE",
            '- [{date}] User: "{user}" → JARVIS: "{jarvis}"',
        )
    )
    # When rewriting short follow-ups for retrieval, use this many recent exchanges
    context_turns_for_query: int = field(
        default_factory=lambda: _int("RAG_CONTEXT_TURNS", 2)
    )


def get_embedding_function(model_key: str):
    """Return a Chroma-compatible embedding function for the chosen preset."""
    from chromadb.utils import embedding_functions

    key = (model_key or "default").lower()
    if key in ("default", "minilm"):
        return embedding_functions.DefaultEmbeddingFunction()
    if key == "minilm_l12":
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L12-v2"
        )
    raise ValueError(
        f"Unknown RAG_EMBEDDING_MODEL={model_key!r}. "
        "Use: default, minilm, or minilm_l12"
    )


# Singleton used by memory.py — import `config` or `RAGConfig()` after env is loaded.
config = RAGConfig()
