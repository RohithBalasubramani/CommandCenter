"""
Memory Stratification — Upgrade 7

Three-tier memory: Session (in-memory), Operator (SQLite), Site (SQLite).
Session clears on exit, Operator has 90-day TTL, Site is permanent.
"""

import os
import re
import time
import json
import uuid
import sqlite3
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    SESSION = "session"
    OPERATOR = "operator"
    SITE = "site"


@dataclass
class MemoryEntry:
    id: str = ""
    tier: MemoryTier = MemoryTier.SESSION
    key: str = ""
    value: dict = field(default_factory=dict)
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    created_by: str = ""
    source: str = "explicit"
    ttl: Optional[float] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def to_dict(self) -> dict:
        return {
            "id": self.id, "tier": self.tier.value,
            "key": self.key, "value": self.value,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "created_by": self.created_by,
            "source": self.source,
        }


class MemoryStore:
    """Three-tier memory store backed by SQLite (operator/site) and dict (session)."""

    TIER_TTL_DEFAULTS = {
        MemoryTier.SESSION: None,
        MemoryTier.OPERATOR: 90 * 86400,
        MemoryTier.SITE: None,
    }

    MAX_SESSION_ENTRIES = 500

    def __init__(self, db_path: str = None):
        self._session_store: dict[str, dict[str, MemoryEntry]] = {}
        self._db_path = db_path
        if db_path:
            self._init_db()

    def _init_db(self):
        """Initialize SQLite tables for operator and site memory."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS operator_memory (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at REAL,
                    accessed_at REAL,
                    access_count INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'explicit',
                    UNIQUE(user_id, key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS site_memory (
                    id TEXT PRIMARY KEY,
                    site_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at REAL,
                    accessed_at REAL,
                    access_count INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'explicit',
                    UNIQUE(site_id, key)
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Memory DB init failed: {e}")

    def remember(self, entry: MemoryEntry, session_id: str = None,
                  user_id: str = None, site_id: str = "default"):
        """Store a memory entry in the appropriate tier."""
        if entry.tier == MemoryTier.SESSION:
            if not session_id:
                return
            if session_id not in self._session_store:
                self._session_store[session_id] = {}
            store = self._session_store[session_id]
            # Prune if too large
            if len(store) >= self.MAX_SESSION_ENTRIES:
                oldest_key = min(store, key=lambda k: store[k].created_at)
                del store[oldest_key]
            store[entry.key] = entry

        elif entry.tier == MemoryTier.OPERATOR:
            if not user_id or not self._db_path:
                return
            self._db_upsert("operator_memory", "user_id", user_id, entry)

        elif entry.tier == MemoryTier.SITE:
            if not self._db_path:
                return
            self._db_upsert("site_memory", "site_id", site_id, entry)

    def recall(self, key: str, tier: MemoryTier = None,
                session_id: str = None, user_id: str = None,
                site_id: str = "default") -> Optional[MemoryEntry]:
        """Recall a memory entry. Search order: Session → Operator → Site."""
        # Session
        if (tier is None or tier == MemoryTier.SESSION) and session_id:
            store = self._session_store.get(session_id, {})
            entry = store.get(key)
            if entry and not entry.is_expired():
                entry.access_count += 1
                entry.accessed_at = time.time()
                return entry

        # Operator
        if (tier is None or tier == MemoryTier.OPERATOR) and user_id and self._db_path:
            entry = self._db_recall("operator_memory", "user_id", user_id, key)
            if entry:
                # Reduce confidence if stale (>30 days since last access)
                if time.time() - entry.accessed_at > 30 * 86400:
                    entry.confidence *= 0.5
                return entry

        # Site
        if (tier is None or tier == MemoryTier.SITE) and self._db_path:
            entry = self._db_recall("site_memory", "site_id", site_id, key)
            if entry:
                return entry

        return None

    def recall_relevant(self, query: str, tier: MemoryTier = None,
                         session_id: str = None, user_id: str = None,
                         site_id: str = "default",
                         limit: int = 5) -> list[MemoryEntry]:
        """BM25-like search across memory entries using keyword matching."""
        results = []
        query_terms = set(re.findall(r'[a-z0-9]+', query.lower()))

        # Search session
        if (tier is None or tier == MemoryTier.SESSION) and session_id:
            for entry in self._session_store.get(session_id, {}).values():
                if entry.is_expired():
                    continue
                score = self._bm25_score(query_terms, entry)
                if score > 0:
                    results.append((score, entry))

        # Search operator
        if (tier is None or tier == MemoryTier.OPERATOR) and user_id and self._db_path:
            entries = self._db_search("operator_memory", "user_id", user_id)
            for entry in entries:
                score = self._bm25_score(query_terms, entry)
                if score > 0:
                    results.append((score, entry))

        # Search site
        if (tier is None or tier == MemoryTier.SITE) and self._db_path:
            entries = self._db_search("site_memory", "site_id", site_id)
            for entry in entries:
                score = self._bm25_score(query_terms, entry)
                if score > 0:
                    results.append((score, entry))

        # Sort by score desc, return top-k
        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:limit]]

    def forget_session(self, session_id: str):
        """Clear all session-tier memory for a session."""
        self._session_store.pop(session_id, None)

    def promote(self, entry: MemoryEntry, to_tier: MemoryTier,
                 user_id: str = None, site_id: str = "default"):
        """Promote a memory entry to a higher tier."""
        new_entry = MemoryEntry(
            tier=to_tier,
            key=entry.key,
            value=entry.value,
            confidence=entry.confidence,
            created_by=entry.created_by,
            source=entry.source,
        )
        self.remember(new_entry, user_id=user_id, site_id=site_id)

    def _bm25_score(self, query_terms: set, entry: MemoryEntry) -> float:
        """Simple BM25-like scoring based on keyword overlap."""
        entry_text = f"{entry.key} {json.dumps(entry.value)}".lower()
        # Split on non-alphanumeric (so pump_004 → pump, 004)
        entry_terms = set(re.findall(r'[a-z0-9]+', entry_text))
        overlap = query_terms & entry_terms
        if not overlap:
            return 0.0
        return len(overlap) / max(len(query_terms), 1) * entry.confidence

    def _db_upsert(self, table: str, owner_col: str, owner_id: str, entry: MemoryEntry):
        """Insert or update in SQLite."""
        retries = 3
        for attempt in range(retries):
            try:
                conn = sqlite3.connect(self._db_path, timeout=5)
                conn.execute(f"""
                    INSERT OR REPLACE INTO {table}
                    (id, {owner_col}, key, value, confidence, created_at, accessed_at, access_count, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.id, owner_id, entry.key,
                    json.dumps(entry.value), entry.confidence,
                    entry.created_at, entry.accessed_at,
                    entry.access_count, entry.source,
                ))
                conn.commit()
                conn.close()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    logger.warning(f"Memory DB upsert failed: {e}")

    def _db_recall(self, table: str, owner_col: str, owner_id: str,
                    key: str) -> Optional[MemoryEntry]:
        """Recall from SQLite."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5)
            row = conn.execute(
                f"SELECT id, key, value, confidence, created_at, accessed_at, access_count, source "
                f"FROM {table} WHERE {owner_col} = ? AND key = ?",
                (owner_id, key)
            ).fetchone()
            if row:
                # Update access
                conn.execute(
                    f"UPDATE {table} SET accessed_at = ?, access_count = access_count + 1 "
                    f"WHERE id = ?", (time.time(), row[0])
                )
                conn.commit()
                conn.close()
                tier = MemoryTier.OPERATOR if table == "operator_memory" else MemoryTier.SITE
                return MemoryEntry(
                    id=row[0], tier=tier, key=row[1],
                    value=json.loads(row[2]), confidence=row[3],
                    created_at=row[4], accessed_at=row[5],
                    access_count=row[6] + 1, source=row[7],
                )
            conn.close()
        except Exception as e:
            logger.warning(f"Memory DB recall failed: {e}")
        return None

    def _db_search(self, table: str, owner_col: str, owner_id: str) -> list[MemoryEntry]:
        """Search all entries in a table for an owner."""
        entries = []
        try:
            conn = sqlite3.connect(self._db_path, timeout=5)
            rows = conn.execute(
                f"SELECT id, key, value, confidence, created_at, accessed_at, access_count, source "
                f"FROM {table} WHERE {owner_col} = ? LIMIT 100",
                (owner_id,)
            ).fetchall()
            tier = MemoryTier.OPERATOR if table == "operator_memory" else MemoryTier.SITE
            for row in rows:
                entries.append(MemoryEntry(
                    id=row[0], tier=tier, key=row[1],
                    value=json.loads(row[2]), confidence=row[3],
                    created_at=row[4], accessed_at=row[5],
                    access_count=row[6], source=row[7],
                ))
            conn.close()
        except Exception as e:
            logger.warning(f"Memory DB search failed: {e}")
        return entries
