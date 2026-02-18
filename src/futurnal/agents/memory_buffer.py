"""Evolving Memory Buffer for AgentFlow Analysis State.

Phase 2E: AgentFlow Architecture - Step 15

Maintains bounded memory of analysis state across sessions:
- Hypothesis history
- Verification results
- Investigation progress
- Learned patterns

Research Foundation:
- ProPerSim (2509.21730v1): Multi-turn context preservation
- Training-Free GRPO (2510.08191v1): Natural language learning
- H-MEM (arxiv:2507.22925): Hierarchical memory for long-term reasoning
  - EvolvingMemoryBuffer acts as "Working Memory" in H-MEM terminology
  - Active context window for current analysis session

Memory Architecture Alignment (H-MEM arxiv:2507.22925):
- EvolvingMemoryBuffer = Working Memory tier (active context)
- Max 50 entries with priority-based retention
- Compresses old entries to make room for new

Option B Compliance:
- No model parameter updates
- Memory stored as natural language entries
- Ghost model FROZEN
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class MemoryEntryType(str, Enum):
    """Types of memory entries."""

    HYPOTHESIS = "hypothesis"  # Generated correlation hypothesis
    EVIDENCE = "evidence"  # Evidence for/against hypothesis
    VERIFICATION = "verification"  # User verification result
    INVESTIGATION = "investigation"  # Investigation progress
    INSIGHT = "insight"  # Generated insight
    PATTERN = "pattern"  # Detected pattern
    FEEDBACK = "feedback"  # User feedback on analysis


class MemoryPriority(str, Enum):
    """Priority levels for memory retention."""

    CRITICAL = "critical"  # Always retain (verified causal)
    HIGH = "high"  # Retain unless space needed
    NORMAL = "normal"  # Standard retention
    LOW = "low"  # First to be compressed/removed


@dataclass
class MemoryEntry:
    """A single entry in the evolving memory buffer.

    Attributes:
        entry_id: Unique identifier
        entry_type: Type of memory entry
        content: Natural language content
        priority: Retention priority
        timestamp: When entry was created
        related_entries: IDs of related entries
        metadata: Additional structured data
        access_count: How many times entry was accessed
        last_accessed: When entry was last accessed
    """

    entry_id: str = field(default_factory=lambda: str(uuid4()))
    entry_type: MemoryEntryType = MemoryEntryType.INSIGHT
    content: str = ""
    priority: MemoryPriority = MemoryPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    related_entries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response.

        Returns camelCase keys for compatibility with frontend/Tauri.
        """
        return {
            "entryId": self.entry_id,
            "entryType": self.entry_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "relatedEntries": self.related_entries,
            "metadata": self.metadata,
            "accessCount": self.access_count,
            "lastAccessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary.

        Accepts both snake_case and camelCase keys for backwards compatibility.
        """
        return cls(
            entry_id=data.get("entryId") or data.get("entry_id", str(uuid4())),
            entry_type=MemoryEntryType(data.get("entryType") or data.get("entry_type", "insight")),
            content=data.get("content", ""),
            priority=MemoryPriority(data.get("priority", "normal")),
            timestamp=datetime.fromisoformat(
                data.get("timestamp", datetime.utcnow().isoformat())
            ),
            related_entries=data.get("relatedEntries") or data.get("related_entries", []),
            metadata=data.get("metadata", {}),
            access_count=data.get("accessCount") or data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data.get("lastAccessed") or data["last_accessed"])
            if data.get("lastAccessed") or data.get("last_accessed")
            else None,
        )

    def to_natural_language(self) -> str:
        """Convert entry to natural language for token priors."""
        lines = [
            f"[{self.entry_type.value.upper()}] {self.timestamp.strftime('%Y-%m-%d')}:",
            self.content,
        ]

        if self.metadata.get("confidence"):
            lines.append(f"Confidence: {self.metadata['confidence']:.0%}")

        if self.metadata.get("status"):
            lines.append(f"Status: {self.metadata['status']}")

        return "\n".join(lines)


class EvolvingMemoryBuffer:
    """Bounded memory buffer for AgentFlow analysis state.

    Maintains a fixed-size buffer of memory entries, automatically
    compressing or removing low-priority entries when space is needed.

    Key Features:
    - Bounded size (default 50 entries)
    - Priority-based retention
    - Automatic compression of old entries
    - Context retrieval for relevant queries
    - Persistence across sessions

    Option B Compliance:
    - No model updates
    - All entries stored as natural language
    - Ghost model FROZEN

    Usage:
        buffer = EvolvingMemoryBuffer()
        buffer.add_entry(MemoryEntry(
            entry_type=MemoryEntryType.HYPOTHESIS,
            content="Monday meetings correlate with higher productivity",
            priority=MemoryPriority.NORMAL,
        ))
        relevant = buffer.get_relevant_context("productivity patterns")
    """

    DEFAULT_STORAGE_PATH = "~/.futurnal/agents/memory_buffer.json"
    DEFAULT_MAX_ENTRIES = 50
    COMPRESSION_THRESHOLD = 0.8  # Compress when 80% full

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        storage_path: Optional[str] = None,
    ):
        """Initialize memory buffer.

        Args:
            max_entries: Maximum number of entries to retain
            storage_path: Path to persist memory
        """
        self.max_entries = max_entries
        self._storage_path = Path(
            os.path.expanduser(storage_path or self.DEFAULT_STORAGE_PATH)
        )
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.entries: List[MemoryEntry] = []
        self._load()

        logger.info(
            f"EvolvingMemoryBuffer initialized "
            f"({len(self.entries)}/{max_entries} entries)"
        )

    def _load(self) -> None:
        """Load entries from storage."""
        if not self._storage_path.exists():
            return

        try:
            data = json.loads(self._storage_path.read_text())
            self.entries = [MemoryEntry.from_dict(e) for e in data.get("entries", [])]
            logger.info(f"Loaded {len(self.entries)} memory entries")
        except Exception as e:
            logger.warning(f"Failed to load memory buffer: {e}")

    def _save(self) -> None:
        """Save entries to storage."""
        try:
            data = {
                "entries": [e.to_dict() for e in self.entries],
                "last_saved": datetime.utcnow().isoformat(),
            }
            self._storage_path.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved {len(self.entries)} memory entries")
        except Exception as e:
            logger.warning(f"Failed to save memory buffer: {e}")

    def add_entry(self, entry: MemoryEntry) -> str:
        """Add a new entry to the buffer.

        If buffer is full, low-priority entries may be compressed
        or removed to make space.

        Args:
            entry: The entry to add

        Returns:
            Entry ID
        """
        # Check if compression/removal needed
        if len(self.entries) >= self.max_entries * self.COMPRESSION_THRESHOLD:
            self._make_space()

        self.entries.append(entry)
        self._save()

        logger.debug(
            f"Added memory entry: {entry.entry_type.value} "
            f"(priority={entry.priority.value})"
        )

        return entry.entry_id

    def _make_space(self) -> None:
        """Make space by compressing or removing entries."""
        if len(self.entries) < self.max_entries:
            return

        # Sort by priority (critical first) and recency
        priority_order = {
            MemoryPriority.CRITICAL: 0,
            MemoryPriority.HIGH: 1,
            MemoryPriority.NORMAL: 2,
            MemoryPriority.LOW: 3,
        }

        # Remove lowest priority entries first
        self.entries.sort(
            key=lambda e: (
                priority_order.get(e.priority, 2),
                -e.access_count,
                e.timestamp,
            )
        )

        # Keep only max_entries - buffer (10% buffer for new entries)
        target_size = int(self.max_entries * 0.9)
        removed_count = len(self.entries) - target_size

        if removed_count > 0:
            # Try to compress LOW priority entries first
            compressed = self._compress_old_entries()

            # If still need space, remove
            while len(self.entries) > target_size:
                removed = self.entries.pop()
                logger.debug(f"Removed memory entry: {removed.entry_id}")

            logger.info(
                f"Made space in memory buffer: compressed {compressed}, "
                f"removed {removed_count - compressed}"
            )

    def _compress_old_entries(self) -> int:
        """Compress old LOW priority entries into summaries.

        Returns:
            Number of entries compressed
        """
        # Find LOW priority entries older than 7 days
        cutoff = datetime.utcnow().replace(hour=0, minute=0, second=0)
        from datetime import timedelta

        cutoff = cutoff - timedelta(days=7)

        old_low = [
            e
            for e in self.entries
            if e.priority == MemoryPriority.LOW and e.timestamp < cutoff
        ]

        if len(old_low) < 3:
            return 0

        # Group by type and create summary
        by_type: Dict[MemoryEntryType, List[MemoryEntry]] = {}
        for entry in old_low:
            if entry.entry_type not in by_type:
                by_type[entry.entry_type] = []
            by_type[entry.entry_type].append(entry)

        compressed = 0
        for entry_type, entries in by_type.items():
            if len(entries) < 2:
                continue

            # Create summary entry
            summary_content = f"Summary of {len(entries)} {entry_type.value} entries:\n"
            for e in entries[:5]:  # Sample first 5
                summary_content += f"- {e.content[:100]}...\n"
            if len(entries) > 5:
                summary_content += f"... and {len(entries) - 5} more"

            summary = MemoryEntry(
                entry_type=entry_type,
                content=summary_content,
                priority=MemoryPriority.LOW,
                metadata={"compressed_from": [e.entry_id for e in entries]},
            )

            # Remove old entries and add summary
            for e in entries:
                self.entries.remove(e)
                compressed += 1

            self.entries.append(summary)

        return compressed

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get an entry by ID.

        Args:
            entry_id: The entry ID

        Returns:
            MemoryEntry if found, None otherwise
        """
        for entry in self.entries:
            if entry.entry_id == entry_id:
                entry.access_count += 1
                entry.last_accessed = datetime.utcnow()
                return entry
        return None

    def get_relevant_context(
        self,
        query: str,
        max_entries: int = 5,
        entry_types: Optional[List[MemoryEntryType]] = None,
    ) -> List[MemoryEntry]:
        """Get entries relevant to a query.

        Simple keyword-based relevance for Option B compliance.

        Args:
            query: The query to match against
            max_entries: Maximum entries to return
            entry_types: Optional filter by entry types

        Returns:
            List of relevant entries
        """
        query_words = set(query.lower().split())

        scored_entries: List[tuple[float, MemoryEntry]] = []

        for entry in self.entries:
            # Filter by type if specified
            if entry_types and entry.entry_type not in entry_types:
                continue

            # Simple keyword scoring
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)

            if overlap == 0:
                continue

            # Score based on overlap, priority, and recency
            priority_boost = {
                MemoryPriority.CRITICAL: 2.0,
                MemoryPriority.HIGH: 1.5,
                MemoryPriority.NORMAL: 1.0,
                MemoryPriority.LOW: 0.5,
            }

            days_old = (datetime.utcnow() - entry.timestamp).days
            recency_factor = max(0.5, 1.0 - (days_old / 30) * 0.5)

            score = overlap * priority_boost.get(entry.priority, 1.0) * recency_factor
            scored_entries.append((score, entry))

        # Sort by score and return top entries
        scored_entries.sort(key=lambda x: -x[0])

        results = []
        for score, entry in scored_entries[:max_entries]:
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            results.append(entry)

        return results

    def get_by_type(
        self,
        entry_type: MemoryEntryType,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Get entries by type.

        Args:
            entry_type: Type to filter by
            limit: Maximum entries to return

        Returns:
            List of entries of that type
        """
        matching = [e for e in self.entries if e.entry_type == entry_type]
        matching.sort(key=lambda e: e.timestamp, reverse=True)
        return matching[:limit]

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get most recent entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of recent entries
        """
        sorted_entries = sorted(self.entries, key=lambda e: e.timestamp, reverse=True)
        return sorted_entries[:limit]

    def update_entry(
        self,
        entry_id: str,
        content: Optional[str] = None,
        priority: Optional[MemoryPriority] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing entry.

        Args:
            entry_id: Entry to update
            content: New content (if provided)
            priority: New priority (if provided)
            metadata: Metadata to merge (if provided)

        Returns:
            True if entry was found and updated
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False

        if content is not None:
            entry.content = content
        if priority is not None:
            entry.priority = priority
        if metadata is not None:
            entry.metadata.update(metadata)

        self._save()
        return True

    def link_entries(self, entry_id_1: str, entry_id_2: str) -> bool:
        """Create a bidirectional link between entries.

        Args:
            entry_id_1: First entry ID
            entry_id_2: Second entry ID

        Returns:
            True if both entries were found and linked
        """
        entry_1 = self.get_entry(entry_id_1)
        entry_2 = self.get_entry(entry_id_2)

        if not entry_1 or not entry_2:
            return False

        if entry_id_2 not in entry_1.related_entries:
            entry_1.related_entries.append(entry_id_2)
        if entry_id_1 not in entry_2.related_entries:
            entry_2.related_entries.append(entry_id_1)

        self._save()
        return True

    def export_for_token_priors(self) -> str:
        """Export memory buffer as natural language for token priors.

        Option B Compliance: Learning through natural language context.

        Returns:
            Natural language summary of memory
        """
        lines = ["Analysis Memory Buffer:"]

        # Group by type
        by_type: Dict[MemoryEntryType, List[MemoryEntry]] = {}
        for entry in self.entries:
            if entry.entry_type not in by_type:
                by_type[entry.entry_type] = []
            by_type[entry.entry_type].append(entry)

        for entry_type, entries in by_type.items():
            lines.append(f"\n## {entry_type.value.title()} ({len(entries)} entries)")
            for entry in sorted(entries, key=lambda e: -e.access_count)[:3]:
                lines.append(entry.to_natural_language())

        return "\n".join(lines)

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        count = len(self.entries)
        self.entries = []
        self._save()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns camelCase keys for compatibility with frontend/Tauri.
        """
        by_type = {}
        by_priority = {}

        for entry in self.entries:
            by_type[entry.entry_type.value] = by_type.get(entry.entry_type.value, 0) + 1
            by_priority[entry.priority.value] = (
                by_priority.get(entry.priority.value, 0) + 1
            )

        return {
            "totalEntries": len(self.entries),
            "maxEntries": self.max_entries,
            "utilization": len(self.entries) / self.max_entries,
            "byType": by_type,
            "byPriority": by_priority,
        }


# Global instance
_memory_buffer: Optional[EvolvingMemoryBuffer] = None


def get_memory_buffer() -> EvolvingMemoryBuffer:
    """Get the default memory buffer singleton."""
    global _memory_buffer
    if _memory_buffer is None:
        _memory_buffer = EvolvingMemoryBuffer()
    return _memory_buffer
