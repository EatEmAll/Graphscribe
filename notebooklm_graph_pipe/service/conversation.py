from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    text: str
    evidence: tuple[dict[str, Any], ...] = ()


class ConversationStore:
    def __init__(self, path: str | Path, *, max_turns: int = 6):
        if max_turns <= 0:
            raise ValueError("max_turns must be positive.")
        self.path = Path(path).resolve()
        self.max_messages = max_turns * 2
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS conversation_message ("
                "conversation_id TEXT NOT NULL, corpus_id TEXT NOT NULL, sequence INTEGER NOT NULL, "
                "role TEXT NOT NULL, text TEXT NOT NULL, evidence_json TEXT NOT NULL, created_at REAL NOT NULL, "
                "PRIMARY KEY(conversation_id, corpus_id, sequence))"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def history(self, conversation_id: str, corpus_id: str) -> list[ConversationTurn]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT role, text, evidence_json FROM conversation_message "
                "WHERE conversation_id = ? AND corpus_id = ? ORDER BY sequence",
                (conversation_id, corpus_id),
            ).fetchall()
        return [ConversationTurn(role, text, tuple(json.loads(evidence))) for role, text, evidence in rows]

    def append_exchange(
        self,
        conversation_id: str,
        corpus_id: str,
        question: str,
        answer: str,
        evidence: list[dict[str, Any]],
    ) -> None:
        if not conversation_id.strip() or len(conversation_id) > 128:
            raise ValueError("conversation_id must contain between 1 and 128 characters.")
        with self._lock, self._connect() as connection:
            next_sequence = int(
                connection.execute(
                    "SELECT coalesce(max(sequence), -1) + 1 FROM conversation_message "
                    "WHERE conversation_id = ? AND corpus_id = ?",
                    (conversation_id, corpus_id),
                ).fetchone()[0]
            )
            connection.executemany(
                "INSERT INTO conversation_message VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (conversation_id, corpus_id, next_sequence, "user", question, "[]", time.time()),
                    (
                        conversation_id,
                        corpus_id,
                        next_sequence + 1,
                        "assistant",
                        answer,
                        json.dumps(evidence, ensure_ascii=True, sort_keys=True),
                        time.time(),
                    ),
                ],
            )
            connection.execute(
                "DELETE FROM conversation_message WHERE conversation_id = ? AND corpus_id = ? "
                "AND sequence NOT IN (SELECT sequence FROM conversation_message "
                "WHERE conversation_id = ? AND corpus_id = ? ORDER BY sequence DESC LIMIT ?)",
                (conversation_id, corpus_id, conversation_id, corpus_id, self.max_messages),
            )


def contextualize_question(question: str, history: list[ConversationTurn]) -> str:
    if not history:
        return question
    conversation = "\n".join(f"{turn.role}: {turn.text}" for turn in history)
    return (
        f"Current question: {question}\n\n"
        "Conversation context (use only to resolve references; this is not source evidence):\n"
        f"{conversation}"
    )
