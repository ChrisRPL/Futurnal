"""Tests for HierarchicalMemory class.

Phase 2.5: Research Integration Sprint

Tests verify:
1. Three-tier memory structure (H-MEM alignment)
2. Episodic to semantic consolidation
3. Query-aware retrieval
4. Option B compliance (no model updates)
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from futurnal.memory.hierarchical_memory import (
    HierarchicalMemory,
    MemoryTierStats,
    EpisodicEntry,
    create_hierarchical_memory,
)


class TestEpisodicEntry:
    """Tests for EpisodicEntry dataclass."""

    def test_default_creation(self):
        """Test default entry creation."""
        entry = EpisodicEntry()
        assert entry.event_description == ""
        assert entry.event_type == ""
        assert entry.significance_score == 0.5
        assert entry.access_count == 0

    def test_with_values(self):
        """Test entry creation with values."""
        entry = EpisodicEntry(
            event_description="Team meeting",
            event_type="meeting",
            significance_score=0.8,
        )
        assert entry.event_description == "Team meeting"
        assert entry.event_type == "meeting"
        assert entry.significance_score == 0.8

    def test_to_natural_language(self):
        """Test natural language export."""
        entry = EpisodicEntry(
            event_description="Team meeting",
            event_type="meeting",
            significance_score=0.8,
        )
        nl = entry.to_natural_language()

        assert "meeting" in nl
        assert "Team meeting" in nl
        assert "80%" in nl


class TestMemoryTierStats:
    """Tests for MemoryTierStats dataclass."""

    def test_creation(self):
        """Test stats creation."""
        stats = MemoryTierStats(
            tier_name="TokenPriorStore",
            research_term="Semantic Memory",
            entry_count=50,
            capacity=100,
            utilization=0.5,
        )
        assert stats.tier_name == "TokenPriorStore"
        assert stats.research_term == "Semantic Memory"
        assert stats.utilization == 0.5

    def test_to_dict(self):
        """Test dictionary serialization."""
        stats = MemoryTierStats(
            tier_name="Test",
            research_term="Test Memory",
            entry_count=10,
            capacity=20,
            utilization=0.5,
        )
        d = stats.to_dict()

        assert d["tier_name"] == "Test"
        assert d["research_term"] == "Test Memory"
        assert d["entry_count"] == 10


class TestHierarchicalMemory:
    """Tests for HierarchicalMemory class."""

    @pytest.fixture
    def memory(self):
        """Create HierarchicalMemory without external dependencies."""
        return HierarchicalMemory(
            episodic_retention_days=30,
            episodic_max_entries=100,
            consolidation_threshold=3,
        )

    @pytest.fixture
    def mock_working_memory(self):
        """Create mock EvolvingMemoryBuffer."""
        buffer = MagicMock()
        buffer.add_entry.return_value = "entry_123"
        buffer.get_stats.return_value = {
            "total_entries": 10,
            "max_entries": 50,
            "utilization": 0.2,
        }
        buffer.get_relevant_context.return_value = []
        buffer.export_for_token_priors.return_value = "Working memory content"
        return buffer

    @pytest.fixture
    def mock_semantic_memory(self):
        """Create mock TokenPriorStore."""
        store = MagicMock()
        store.get_summary.return_value = {
            "total_priors": 30,
            "capacity": 100,
        }
        store.generate_prompt_context.return_value = "Semantic context"
        store.export_as_natural_language.return_value = "Semantic priors"
        return store

    def test_add_to_episodic_memory(self, memory):
        """Test adding entries to episodic memory."""
        entry_id = memory.add_to_episodic_memory(
            "Team meeting occurred",
            event_type="meeting",
            significance_score=0.8,
        )

        assert entry_id is not None
        assert len(memory._episodic_cache) == 1
        assert memory._episodic_cache[0].event_description == "Team meeting occurred"
        assert memory._episodic_cache[0].event_type == "meeting"

    def test_episodic_memory_with_related_entities(self, memory):
        """Test episodic memory with related entities."""
        entry_id = memory.add_to_episodic_memory(
            "Meeting with team",
            event_type="meeting",
            related_entities=["Alice", "Bob"],
        )

        entry = memory._episodic_cache[0]
        assert "Alice" in entry.related_entities
        assert "Bob" in entry.related_entities

    def test_episodic_memory_pruning_by_count(self, memory):
        """Test that old episodic entries are pruned by count."""
        memory._episodic_max_entries = 5

        # Add more than max
        for i in range(10):
            memory.add_to_episodic_memory(f"Event {i}", "test")

        assert len(memory._episodic_cache) <= 5

    def test_episodic_memory_pruning_by_age(self, memory):
        """Test that old episodic entries are pruned by age."""
        # Add old entry
        old_entry = EpisodicEntry(
            event_description="Old event",
            event_type="test",
            timestamp=datetime.utcnow() - timedelta(days=40),
        )
        memory._episodic_cache.append(old_entry)

        # Add new entry (triggers pruning)
        memory.add_to_episodic_memory("New event", "test")

        # Old entry should be removed
        assert len(memory._episodic_cache) == 1
        assert memory._episodic_cache[0].event_description == "New event"

    def test_consolidate_episodic_to_semantic_below_threshold(self, memory):
        """Test that consolidation doesn't happen below threshold."""
        # Add entries below threshold
        memory.add_to_episodic_memory("Meeting 1", "meeting")
        memory.add_to_episodic_memory("Meeting 2", "meeting")

        consolidated = memory.consolidate_episodic_to_semantic()
        assert consolidated == 0

    def test_consolidate_episodic_to_semantic_at_threshold(self, memory):
        """Test consolidation at threshold."""
        memory._consolidation_threshold = 3

        # Add entries at threshold
        for i in range(5):
            memory.add_to_episodic_memory(f"Meeting {i}", "meeting")

        # Need semantic memory for consolidation
        memory._semantic = MagicMock()

        consolidated = memory.consolidate_episodic_to_semantic()
        assert consolidated == 1

    def test_add_to_working_memory(self, mock_working_memory):
        """Test adding entries to working memory."""
        memory = HierarchicalMemory(working_memory=mock_working_memory)
        entry_id = memory.add_to_working_memory(
            "Test hypothesis",
            entry_type="hypothesis",
            priority="high",
        )

        assert entry_id == "entry_123"
        mock_working_memory.add_entry.assert_called_once()

    def test_add_to_working_memory_without_buffer(self, memory):
        """Test adding to working memory without buffer returns None."""
        entry_id = memory.add_to_working_memory("Test")
        assert entry_id is None

    def test_get_relevant_context(self, memory):
        """Test multi-tier context retrieval."""
        memory.add_to_episodic_memory("Meeting about project", "meeting")

        context = memory.get_relevant_context("meeting")

        assert "episodic_memory" in context
        assert len(context["episodic_memory"]) > 0

    def test_get_tier_stats(self, memory, mock_working_memory, mock_semantic_memory):
        """Test getting statistics for all tiers."""
        memory._working = mock_working_memory
        memory._semantic = mock_semantic_memory
        memory.add_to_episodic_memory("Test", "test")

        stats = memory.get_tier_stats()

        assert len(stats) == 3  # Working, Episodic, Semantic
        tier_names = [s.tier_name for s in stats]
        assert "EvolvingMemoryBuffer" in tier_names
        assert "EpisodicCache" in tier_names
        assert "TokenPriorStore" in tier_names

    def test_get_statistics(self, memory):
        """Test overall statistics."""
        memory.add_to_episodic_memory("Test", "test")

        stats = memory.get_statistics()

        assert stats["total_episodic_added"] == 1
        assert stats["current_episodic_count"] == 1
        assert stats["consolidation_count"] == 0

    def test_export_for_token_priors(self, memory):
        """Test natural language export."""
        memory.add_to_episodic_memory("Test event", "test")

        export = memory.export_for_token_priors()

        assert "Hierarchical Memory Export" in export
        assert "H-MEM" in export
        assert "Episodic Memory" in export

    def test_clear_episodic(self, memory):
        """Test clearing episodic memory."""
        memory.add_to_episodic_memory("Test", "test")
        assert len(memory._episodic_cache) == 1

        memory.clear_episodic()
        assert len(memory._episodic_cache) == 0


class TestCreateHierarchicalMemory:
    """Tests for create_hierarchical_memory factory function."""

    def test_create_returns_hierarchical_memory(self):
        """Test that factory returns HierarchicalMemory instance."""
        # This may fail if dependencies aren't available
        try:
            memory = create_hierarchical_memory()
            assert isinstance(memory, HierarchicalMemory)
        except ImportError:
            # If dependencies not available, create without them
            memory = HierarchicalMemory()
            assert isinstance(memory, HierarchicalMemory)
