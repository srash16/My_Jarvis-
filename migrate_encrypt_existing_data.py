#!/usr/bin/env python3
"""
One-time migration: encrypt existing plaintext jarvis_memory.db + rebuild chroma_data/.

Prerequisites:
  1. Set JARVIS_DB_KEY in .env (64-char hex). Never commit this file.
  2. Stop JARVIS before running this script.

Usage:
  python migrate_encrypt_existing_data.py

After success:
  - Delete or keep the *.pre_encrypt.bak backups once you verify JARVIS starts.
  - Do not re-run unless you restored plaintext backups.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from memory_crypto import (  # noqa: E402
    encrypt_text,
    get_fernet,
    is_encrypted_value,
    require_db_key,
)
from rag_config import config, get_embedding_function  # noqa: E402


def main() -> int:
    require_db_key()
    fernet = get_fernet()

    sqlite_path = Path(config.sqlite_path)
    chroma_path = Path(config.chroma_path)

    if not sqlite_path.exists():
        print(f"No {sqlite_path} found - nothing to migrate.")
        return 0

    bak_db = Path(str(sqlite_path) + ".pre_encrypt.bak")
    if not bak_db.exists():
        shutil.copy2(sqlite_path, bak_db)
        print(f"Backed up SQLite -> {bak_db}")
    else:
        print(f"Using existing backup {bak_db}")

    conn = sqlite3.connect(str(sqlite_path))
    rows = conn.execute(
        "SELECT id, timestamp, user_text, jarvis_response FROM conversations ORDER BY id"
    ).fetchall()

    updated = 0
    for row_id, ts, user_text, jarvis_response in rows:
        if is_encrypted_value(user_text) and is_encrypted_value(jarvis_response):
            continue
        conn.execute(
            "UPDATE conversations SET user_text = ?, jarvis_response = ? WHERE id = ?",
            (
                encrypt_text(user_text, fernet),
                encrypt_text(jarvis_response, fernet),
                row_id,
            ),
        )
        updated += 1
    conn.commit()
    conn.close()
    print(f"Encrypted {updated} SQLite row(s) (already-encrypted rows skipped).")

    # Rebuild Chroma from encrypted SQLite (plaintext embeddings, ciphertext docs)
    if chroma_path.exists():
        bak_chroma = Path(str(chroma_path) + ".pre_encrypt.bak")
        if bak_chroma.exists():
            shutil.rmtree(bak_chroma, ignore_errors=True)
        shutil.move(str(chroma_path), str(bak_chroma))
        print(f"Moved old Chroma -> {bak_chroma}")

    import chromadb

    embed_fn = get_embedding_function(config.embedding_model)
    client = chromadb.PersistentClient(path=str(chroma_path))
    # Drop collection if somehow present
    try:
        client.delete_collection(config.collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=embed_fn,
    )

    from memory_crypto import decrypt_text  # noqa: E402

    conn = sqlite3.connect(str(sqlite_path))
    rows = conn.execute(
        "SELECT id, timestamp, user_text, jarvis_response FROM conversations ORDER BY id"
    ).fetchall()
    conn.close()

    if rows:
        ids, docs, metas, embeddings = [], [], [], []
        for row_id, ts, enc_user, enc_jarvis in rows:
            user = decrypt_text(enc_user, fernet)
            jarvis = decrypt_text(enc_jarvis, fernet)
            plain_doc = config.chunk_template.format(user=user, jarvis=jarvis)
            embeddings.append(embed_fn([plain_doc])[0])
            docs.append(encrypt_text(plain_doc, fernet))
            metas.append({
                "timestamp": ts,
                "user_text": encrypt_text(user, fernet),
                "jarvis_response": encrypt_text(jarvis, fernet),
            })
            ids.append(str(row_id))
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        print(f"Rebuilt Chroma with {len(ids)} encrypted document(s).")
    else:
        print("No conversations to put into Chroma.")

    print(
        "\nMigration complete.\n"
        "1. Start JARVIS and confirm memory still works.\n"
        "2. When satisfied, delete the *.pre_encrypt.bak backups "
        "(they still contain plaintext).\n"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"Migration failed: {e}")
        raise SystemExit(1)
