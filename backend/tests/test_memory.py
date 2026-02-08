"""
Tests for Upgrade 7: Memory Stratification

Test IDs: MS-B01 through MS-B09
"""

import os
import pytest
import time
from layer2.memory import MemoryStore, MemoryEntry, MemoryTier


@pytest.fixture
def memory_store(tmp_path):
    """Create a memory store with temp SQLite DB."""
    db_path = str(tmp_path / "test_memory.sqlite3")
    return MemoryStore(db_path=db_path)


@pytest.fixture
def session_only_store():
    """Memory store without SQLite (session only)."""
    return MemoryStore(db_path=None)


class TestSessionMemory:
    """MS-B01/B02: Session-tier memory."""

    def test_remember_and_recall_session(self, session_only_store):
        """MS-B01: Store and recall session memory."""
        store = session_only_store
        entry = MemoryEntry(
            tier=MemoryTier.SESSION,
            key="pump_004:last_query",
            value={"query": "show pump 4 vibration", "widgets": ["trend", "kpi"]},
        )
        store.remember(entry, session_id="sess1")
        recalled = store.recall("pump_004:last_query", session_id="sess1")
        assert recalled is not None
        assert recalled.value["query"] == "show pump 4 vibration"

    def test_session_isolation(self, session_only_store):
        """MS-B02: Sessions are isolated."""
        store = session_only_store
        entry = MemoryEntry(
            tier=MemoryTier.SESSION,
            key="test:key",
            value={"data": "session1"},
        )
        store.remember(entry, session_id="sess1")
        assert store.recall("test:key", session_id="sess1") is not None
        assert store.recall("test:key", session_id="sess2") is None

    def test_forget_session(self, session_only_store):
        """MS-B03: Forget clears all session memory."""
        store = session_only_store
        entry = MemoryEntry(tier=MemoryTier.SESSION, key="test", value={})
        store.remember(entry, session_id="sess1")
        store.forget_session("sess1")
        assert store.recall("test", session_id="sess1") is None


class TestOperatorMemory:
    """MS-B04/B05: Operator-tier memory with SQLite."""

    def test_remember_and_recall_operator(self, memory_store):
        """MS-B04: Store and recall operator memory."""
        entry = MemoryEntry(
            tier=MemoryTier.OPERATOR,
            key="widget_preference",
            value={"preferred": ["trend", "kpi"], "suppressed": ["alerts"]},
            created_by="user1",
        )
        memory_store.remember(entry, user_id="user1")
        recalled = memory_store.recall("widget_preference", user_id="user1")
        assert recalled is not None
        assert recalled.value["preferred"] == ["trend", "kpi"]

    def test_operator_memory_persists(self, tmp_path):
        """MS-B05: Operator memory survives store recreation."""
        db_path = str(tmp_path / "persist_test.sqlite3")
        store1 = MemoryStore(db_path=db_path)
        entry = MemoryEntry(
            tier=MemoryTier.OPERATOR,
            key="persist_test",
            value={"data": "persistent"},
        )
        store1.remember(entry, user_id="user1")

        # Create new store with same DB
        store2 = MemoryStore(db_path=db_path)
        recalled = store2.recall("persist_test", user_id="user1")
        assert recalled is not None
        assert recalled.value["data"] == "persistent"


class TestSiteMemory:
    """MS-B06: Site-tier memory."""

    def test_remember_and_recall_site(self, memory_store):
        """MS-B06: Site memory shared across users."""
        entry = MemoryEntry(
            tier=MemoryTier.SITE,
            key="pump_004:known_issues",
            value={"issues": [{"id": "i1", "description": "Bearing wear pattern"}]},
        )
        memory_store.remember(entry, site_id="plant_a")
        recalled = memory_store.recall("pump_004:known_issues", site_id="plant_a")
        assert recalled is not None
        assert len(recalled.value["issues"]) == 1


class TestMemorySearch:
    """MS-B07: BM25-like search."""

    def test_recall_relevant(self, memory_store):
        """MS-B07: Search finds relevant entries."""
        entries = [
            MemoryEntry(tier=MemoryTier.SITE, key="pump_004:vibration_history",
                         value={"avg": 2.1, "max": 3.5}),
            MemoryEntry(tier=MemoryTier.SITE, key="pump_004:temperature_history",
                         value={"avg": 42, "max": 55}),
            MemoryEntry(tier=MemoryTier.SITE, key="motor_001:power_history",
                         value={"avg": 15, "max": 22}),
        ]
        for e in entries:
            memory_store.remember(e, site_id="plant_a")

        results = memory_store.recall_relevant(
            "pump 004 vibration", site_id="plant_a"
        )
        assert len(results) >= 1
        assert "pump_004" in results[0].key


class TestMemoryTierSearch:
    """MS-B08: Search order Session → Operator → Site."""

    def test_session_takes_priority(self, memory_store):
        """MS-B08: Session memory found before operator/site."""
        # Store in site
        memory_store.remember(
            MemoryEntry(tier=MemoryTier.SITE, key="test_key", value={"tier": "site"}),
            site_id="default",
        )
        # Store in session
        memory_store.remember(
            MemoryEntry(tier=MemoryTier.SESSION, key="test_key", value={"tier": "session"}),
            session_id="sess1",
        )
        # Recall should find session first
        recalled = memory_store.recall("test_key", session_id="sess1", site_id="default")
        assert recalled.value["tier"] == "session"


class TestMemoryPromotion:
    """MS-B09: Promote entries between tiers."""

    def test_promote_session_to_operator(self, memory_store):
        """MS-B09: Promote from session to operator."""
        entry = MemoryEntry(
            tier=MemoryTier.SESSION,
            key="preferred_layout",
            value={"hero": "trend", "secondary": ["kpi", "alerts"]},
        )
        memory_store.remember(entry, session_id="sess1")
        memory_store.promote(entry, to_tier=MemoryTier.OPERATOR, user_id="user1")

        # Should be in operator tier now
        recalled = memory_store.recall("preferred_layout", user_id="user1")
        assert recalled is not None
        assert recalled.value["hero"] == "trend"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
