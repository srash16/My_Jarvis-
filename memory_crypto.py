"""App-level encryption helpers keyed from JARVIS_DB_KEY (Fernet / local only).

SQLCipher wheels do not install cleanly on Windows + Python 3.13, so conversation
TEXT fields and Chroma documents/metadata are Fernet-encrypted at rest instead.
Same key requirement: never run with plaintext when JARVIS_DB_KEY is unset.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import sys

from cryptography.fernet import Fernet, InvalidToken

ENC_PREFIX = "jenc:"


def generate_db_key() -> str:
    return secrets.token_hex(32)


def require_db_key() -> str:
    """
    Return JARVIS_DB_KEY from the environment.
    If missing, print a one-time setup message (with a generated key) and exit.
    """
    key = (os.getenv("JARVIS_DB_KEY") or "").strip()
    if key:
        return key

    generated = generate_db_key()
    print(
        "\n"
        "============================================================\n"
        " JARVIS_DB_KEY is missing — memory encryption is required.\n"
        "============================================================\n"
        " Add this line to your .env file (do not commit .env):\n"
        f"\n  JARVIS_DB_KEY={generated}\n"
        "\n Then restart JARVIS.\n"
        " If you already have jarvis_memory.db / chroma_data from before\n"
        " encryption, run after saving the key:\n"
        "   python migrate_encrypt_existing_data.py\n"
        "============================================================\n"
    )
    sys.exit(1)


def _fernet_from_key(key_hex: str) -> Fernet:
    """Derive a Fernet key from the hex JARVIS_DB_KEY."""
    try:
        raw = bytes.fromhex(key_hex)
    except ValueError as e:
        raise SystemExit(
            "JARVIS_DB_KEY must be a hex string (64 chars from secrets.token_hex(32))."
        ) from e
    if len(raw) < 16:
        raise SystemExit("JARVIS_DB_KEY is too short.")
    # Normalize to 32 bytes for Fernet
    material = hashlib.sha256(raw).digest()
    return Fernet(base64.urlsafe_b64encode(material))


def get_fernet(key_hex: str | None = None) -> Fernet:
    return _fernet_from_key(key_hex or require_db_key())


def is_encrypted_value(value: str | None) -> bool:
    return isinstance(value, str) and value.startswith(ENC_PREFIX)


def encrypt_text(plain: str, fernet: Fernet | None = None) -> str:
    if plain is None:
        plain = ""
    if is_encrypted_value(plain):
        return plain
    f = fernet or get_fernet()
    token = f.encrypt(plain.encode("utf-8")).decode("ascii")
    return ENC_PREFIX + token


def decrypt_text(value: str, fernet: Fernet | None = None) -> str:
    if value is None:
        return ""
    if not is_encrypted_value(value):
        return value
    f = fernet or get_fernet()
    try:
        return f.decrypt(value[len(ENC_PREFIX) :].encode("ascii")).decode("utf-8")
    except InvalidToken as e:
        raise SystemExit(
            "Could not decrypt memory data with JARVIS_DB_KEY. "
            "Check that .env has the same key used when the data was encrypted."
        ) from e
