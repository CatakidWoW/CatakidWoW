from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantum_earth.config import settings

ARTEFACT_DIRS = [
    "RAW",
    "PROCESSED",
    "ASSIMILATED",
    "TRAINING",
    "VALIDATION",
    "FORECAST",
    "GROUND_TRUTH",
    "MODELS",
    "EXPERIMENTS",
    "METRICS",
    "EVENTS",
]


class StorageLayout:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or settings.data_root)
        for name in ARTEFACT_DIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, category: str, *parts: str) -> Path:
        if category not in ARTEFACT_DIRS:
            raise ValueError(f"Unknown storage category: {category}")
        p = self.root / category
        for part in parts:
            p = p / part
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, category: str, relative: str, payload: Any) -> Path:
        path = self.path(category, relative)
        text = json.dumps(payload, indent=2, default=str)
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def checksum_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def checksum_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


class MetadataDB:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS data_sources (
                  source_id TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS models (
                  model_id TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                  experiment_id TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS verification (
                  id TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS health (
                  component TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                """
            )

    def upsert(self, table: str, key_col: str, key: str, payload: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {table} ({key_col}, payload, updated_at) VALUES (?, ?, ?) "
                f"ON CONFLICT({key_col}) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (key, json.dumps(payload, default=str), now),
            )

    def get(self, table: str, key_col: str, key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT payload FROM {table} WHERE {key_col}=?", (key,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_all(self, table: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(f"SELECT payload FROM {table}").fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def insert_verification(self, record_id: str, payload: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO verification (id, payload, created_at) VALUES (?, ?, ?)",
                (record_id, json.dumps(payload, default=str), now),
            )
