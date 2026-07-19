#!/usr/bin/env python
"""
JARVIS Memory Dashboard - Read-only dashboard to view conversation history and audit log.

To run (only accessible from localhost):
    streamlit run dashboard.py --server.address=127.0.0.1

This dashboard reuses the decryption logic from memory_crypto to decrypt
conversation history stored in SQLite. It reads the audit log (JSON lines) directly.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from audit_log import AUDIT_LOG_PATH  # noqa: E402
from memory_crypto import decrypt_text, get_fernet, is_encrypted_value  # noqa: E402
from rag_config import config  # noqa: E402

SQLITE_PATH = config.sqlite_path


def decrypt_field(value: str, fernet) -> str:
    """Decrypt an encrypted field; return plaintext/empty safely without SystemExit."""
    if not value:
        return ""
    if not is_encrypted_value(value):
        return value
    try:
        return decrypt_text(value, fernet)
    except SystemExit:
        return "[decrypt failed — check JARVIS_DB_KEY]"
    except Exception:
        return "[decrypt failed]"


@st.cache_data
def get_unique_tools() -> list[str]:
    path = str(AUDIT_LOG_PATH)
    if not os.path.exists(path):
        return []
    tools = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                name = entry.get("tool_name", "")
                if name:
                    tools.add(name)
            except json.JSONDecodeError:
                pass
    return sorted(tools)


def load_conversations(
    search_text: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
    fernet=None,
):
    """Load conversations from SQLite, decrypt, and apply filters."""
    if not os.path.exists(SQLITE_PATH):
        return []

    fernet = fernet or get_fernet()
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row

    query = "SELECT timestamp, user_text, jarvis_response FROM conversations"
    conditions = []
    params = []

    if start_date:
        conditions.append("date(timestamp) >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("date(timestamp) <= ?")
        params.append(end_date)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    results = []
    for row in rows:
        user_text = decrypt_field(row["user_text"], fernet)
        jarvis_text = decrypt_field(row["jarvis_response"], fernet)
        timestamp = row["timestamp"]

        if search_text:
            needle = search_text.lower()
            if needle not in user_text.lower() and needle not in jarvis_text.lower():
                continue

        results.append({
            "timestamp": timestamp,
            "user_text": user_text,
            "jarvis_response": jarvis_text,
        })

    return results


def load_audit_log(tool_filter: str | None = None, confirmed_only: bool = False):
    """Load audit log from JSON lines file and apply filters."""
    path = str(AUDIT_LOG_PATH)
    if not os.path.exists(path):
        return []

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            tool_name = entry.get("tool_name", "")
            confirmed = entry.get("confirmed", None)

            if tool_filter and tool_name != tool_filter:
                continue
            if confirmed_only and confirmed is not True:
                continue

            entries.append({
                "timestamp": entry.get("timestamp", ""),
                "tool_name": tool_name,
                "args": entry.get("args", {}),
                "result": entry.get("result", ""),
                "confirmed": confirmed,
            })

    entries.reverse()
    return entries


def main():
    st.set_page_config(page_title="JARVIS Memory Dashboard", layout="wide")
    st.title("JARVIS Memory Dashboard")
    st.caption(
        "Read-only view of conversation history and audit log. "
        "Data is decrypted on the fly using JARVIS_DB_KEY from .env."
    )

    key = (os.getenv("JARVIS_DB_KEY") or "").strip()
    if not key:
        st.error(
            "JARVIS_DB_KEY is missing from .env. "
            "Add it and restart the dashboard (same key used by JARVIS)."
        )
        st.stop()

    try:
        fernet = get_fernet(key)
    except SystemExit as e:
        st.error(str(e) if str(e) else "Invalid JARVIS_DB_KEY.")
        st.stop()

    if not os.path.exists(SQLITE_PATH):
        st.warning(f"Conversation database not found: {SQLITE_PATH}")

    if not os.path.exists(str(AUDIT_LOG_PATH)):
        st.info(f"No audit log yet at {AUDIT_LOG_PATH} (appears after tool calls).")

    with st.sidebar:
        st.header("Filters")
        st.subheader("Conversation History")
        search_text = st.text_input("Search in conversations", placeholder="user or JARVIS text")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From date", value=None)
        with col2:
            end_date = st.date_input("To date", value=None)
        start_date_str = start_date.isoformat() if start_date else None
        end_date_str = end_date.isoformat() if end_date else None

        st.divider()
        st.subheader("Audit Log")
        tool_options = ["All"] + get_unique_tools()
        selected_tool = st.selectbox("Tool name", options=tool_options, index=0)
        tool_filter = None if selected_tool == "All" else selected_tool
        confirmed_only = st.checkbox("Show only confirmed actions", value=False)

    conversations = load_conversations(
        search_text=search_text,
        start_date=start_date_str,
        end_date=end_date_str,
        fernet=fernet,
    )
    audit_entries = load_audit_log(
        tool_filter=tool_filter,
        confirmed_only=confirmed_only,
    )

    st.subheader("Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matching Conversations", len(conversations))
    with col2:
        tool_counts: dict[str, int] = {}
        for entry in audit_entries:
            tool = entry["tool_name"]
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        if tool_counts:
            st.write("Matching tool calls:")
            st.table([{"Tool": k, "Count": v} for k, v in sorted(tool_counts.items())])
        else:
            st.write("No matching tool calls.")
    with col3:
        timestamps = []
        if conversations:
            timestamps.append(conversations[0]["timestamp"])
        if audit_entries:
            timestamps.append(audit_entries[0]["timestamp"])
        if timestamps:
            latest = max(timestamps)
            try:
                dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                formatted = latest
            st.metric("Most Recent Activity", formatted)
        else:
            st.metric("Most Recent Activity", "N/A")

    st.subheader("Conversation History")
    if conversations:
        display_data = []
        for conv in conversations:
            display_data.append({
                "Timestamp": conv["timestamp"][:19].replace("T", " "),
                "User": (
                    conv["user_text"][:100] + "..."
                    if len(conv["user_text"]) > 100
                    else conv["user_text"]
                ),
                "JARVIS": (
                    conv["jarvis_response"][:100] + "..."
                    if len(conv["jarvis_response"]) > 100
                    else conv["jarvis_response"]
                ),
            })
        st.dataframe(display_data, use_container_width=True, hide_index=True)
    else:
        st.info("No conversation records match the current filters.")

    st.subheader("Audit Log")
    if audit_entries:
        display_audit = []
        for entry in audit_entries:
            args_str = json.dumps(entry["args"], ensure_ascii=False)
            result_str = (
                entry["result"]
                if isinstance(entry["result"], str)
                else json.dumps(entry["result"], ensure_ascii=False)
            )
            if len(args_str) > 100:
                args_str = args_str[:97] + "..."
            if len(result_str) > 100:
                result_str = result_str[:97] + "..."
            confirmed = entry["confirmed"]
            if confirmed is True:
                confirmed_str = "Yes"
            elif confirmed is False:
                confirmed_str = "No"
            else:
                confirmed_str = "N/A"
            display_audit.append({
                "Timestamp": (entry["timestamp"] or "")[:19].replace("T", " "),
                "Tool": entry["tool_name"],
                "Args": args_str,
                "Result": result_str,
                "Confirmed": confirmed_str,
            })
        st.dataframe(display_audit, use_container_width=True, hide_index=True)
    else:
        st.info("No audit log entries match the current filters.")

    st.divider()
    st.caption(
        "This dashboard runs locally (bind to 127.0.0.1). "
        "Keep .env private; never commit it."
    )


if __name__ == "__main__":
    main()
