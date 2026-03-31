"""Conversation memory with async SQLite backend."""

from __future__ import annotations

import uuid
import aiosqlite
from datetime import datetime, timezone
from pathlib import Path


class ConversationMemory:
    """Multi-turn conversation memory with token-budget-aware retrieval."""

    def __init__(self, db_path: str | Path = "conversations.db", tokenizer=None):
        self.db_path = str(db_path)
        self.tokenizer = tokenizer
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv_id
                ON messages(conversation_id, id);
        """)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens using actual tokenization, not heuristics."""
        if self.tokenizer is None:
            # Fallback: rough estimate (only used if tokenizer unavailable)
            return len(text) // 3
        return len(self.tokenizer.encode(text).ids)

    async def create_conversation(self) -> str:
        """Create a new conversation and return its ID."""
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
            (conv_id, now, now),
        )
        await self._db.commit()
        return conv_id

    async def add_message(
        self, conversation_id: str, role: str, content: str
    ) -> int:
        """Add a message to a conversation. Returns the message ID."""
        token_count = self._count_tokens(content)
        now = datetime.now(timezone.utc).isoformat()

        cursor = await self._db.execute(
            "INSERT INTO messages (conversation_id, role, content, token_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, token_count, now),
        )
        await self._db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_context(
        self, conversation_id: str, token_budget: int = 3000
    ) -> list[dict]:
        """Get messages that fit within token budget.

        Strategy: always include system message, then fill from most recent
        backward. No summarization — just truncate oldest turns.
        """
        cursor = await self._db.execute(
            "SELECT role, content, token_count FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return []

        messages = [{"role": r, "content": c, "token_count": t} for r, c, t in rows]

        # Separate system messages and conversation messages
        system_msgs = [m for m in messages if m["role"] == "system"]
        conv_msgs = [m for m in messages if m["role"] != "system"]

        result = []
        remaining_budget = token_budget

        # Always include system messages first
        for msg in system_msgs:
            if remaining_budget >= msg["token_count"]:
                result.append({"role": msg["role"], "content": msg["content"]})
                remaining_budget -= msg["token_count"]

        # Fill from most recent conversation messages backward
        recent = []
        for msg in reversed(conv_msgs):
            if remaining_budget < msg["token_count"]:
                break
            recent.insert(0, {"role": msg["role"], "content": msg["content"]})
            remaining_budget -= msg["token_count"]

        return result + recent

    async def list_conversations(self, limit: int = 20) -> list[dict]:
        """List recent conversations."""
        cursor = await self._db.execute(
            "SELECT id, created_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [{"id": r[0], "created_at": r[1], "updated_at": r[2]} for r in rows]

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages."""
        await self._db.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
        )
        await self._db.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        await self._db.commit()
