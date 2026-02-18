"""Unified Temporal Source - Aggregates temporal data from all PKG sources.

Phase 2.5 P0 Fix: Extends temporal analysis beyond Event nodes to include
Documents, Entities, and other timestamped content in the PKG.

This enables insights generation from:
- Event nodes (traditional)
- Document nodes (modified_at, updated_at, created_at, ingested_at)
- Entity mentions (via document timestamps)

Research Foundation:
- Addresses the core product issue: user's PKG may have no Event nodes
- Documents with timestamps are the primary personal data source
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from futurnal.pkg.schema.models import DocumentNode, EventNode

if TYPE_CHECKING:
    from futurnal.pkg.queries.temporal import TemporalGraphQueries

logger = logging.getLogger(__name__)


@dataclass
class TemporalItem:
    """Unified representation of a timestamped item.

    Provides a common interface for correlation detection regardless
    of whether the source is an Event, Document, or Entity mention.
    """

    item_id: str
    item_type: str  # 'event', 'document', 'entity_mention'
    timestamp: datetime
    category: str  # For correlation grouping (event_type, doc_type, entity_type)
    source_node: Any = None  # Original EventNode or DocumentNode
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        """Alias for category to maintain compatibility with correlation detector."""
        return self.category


class UnifiedTemporalSource:
    """Aggregates temporal data from all PKG sources.

    Provides a unified interface for temporal correlation detection
    that works with Events, Documents, and Entity mentions.

    This addresses the core product issue where users may have
    documents with timestamps but no Event nodes.

    Example:
        >>> source = UnifiedTemporalSource(pkg_queries)
        >>> items = source.get_temporal_items(
        ...     start=datetime(2024, 1, 1),
        ...     end=datetime(2024, 12, 31),
        ...     include_documents=True,
        ... )
        >>> print(f"Found {len(items)} temporal items")
    """

    def __init__(self, pkg_queries: "TemporalGraphQueries"):
        """Initialize the unified temporal source.

        Args:
            pkg_queries: PKG temporal queries service
        """
        self._pkg = pkg_queries

    def get_temporal_items(
        self,
        start: datetime,
        end: datetime,
        include_events: bool = True,
        include_documents: bool = True,
        include_entity_mentions: bool = False,  # Not yet implemented
        document_category_field: str = "source_type",
    ) -> List[TemporalItem]:
        """Get all timestamped items from PKG.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (inclusive)
            include_events: Include Event nodes (default True)
            include_documents: Include Document nodes (default True)
            include_entity_mentions: Include entity mentions (not yet implemented)
            document_category_field: Field to use for document categorization

        Returns:
            List of TemporalItem instances ordered by timestamp
        """
        items: List[TemporalItem] = []

        # Get Event nodes
        if include_events:
            try:
                events = self._pkg.query_events_in_timerange(
                    start=start,
                    end=end,
                )
                for event in events:
                    items.append(self._event_to_temporal_item(event))
                logger.debug(f"Retrieved {len(events)} events")
            except Exception as e:
                logger.warning(f"Failed to query events: {e}")

        # Get Document nodes
        if include_documents:
            try:
                documents = self._pkg.query_documents_in_timerange(
                    start=start,
                    end=end,
                )
                for doc in documents:
                    item = self._document_to_temporal_item(doc, document_category_field)
                    if item:
                        items.append(item)
                logger.debug(f"Retrieved {len(documents)} documents")
            except Exception as e:
                logger.warning(f"Failed to query documents: {e}")

        # Sort by timestamp
        items.sort(key=lambda x: x.timestamp)

        logger.info(
            f"UnifiedTemporalSource: {len(items)} items in range "
            f"{start.isoformat()} to {end.isoformat()}"
        )

        return items

    def _event_to_temporal_item(self, event: EventNode) -> TemporalItem:
        """Convert an EventNode to a TemporalItem."""
        timestamp = event.timestamp
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)

        return TemporalItem(
            item_id=event.id,
            item_type="event",
            timestamp=timestamp,
            category=event.event_type or "unknown_event",
            source_node=event,
            metadata={
                "name": event.name,
                "description": getattr(event, "description", None),
            },
        )

    def _document_to_temporal_item(
        self,
        doc: DocumentNode,
        category_field: str = "doc_type",
    ) -> Optional[TemporalItem]:
        """Convert a DocumentNode to a TemporalItem.

        Uses best available timestamp: modified_at > updated_at > created_at > ingested_at
        """
        # Get timestamp with priority
        timestamp = self._get_document_timestamp(doc)
        if timestamp is None:
            return None

        # Strip timezone for consistency
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)

        # Get category from doc
        category = getattr(doc, category_field, None) or "document"

        # If no specific type, try to infer from file extension or path
        if category == "document":
            title = getattr(doc, "title", "") or getattr(doc, "doc_title", "") or ""
            if title.endswith(".md"):
                category = "markdown_note"
            elif title.endswith(".pdf"):
                category = "pdf_document"
            elif "meeting" in title.lower():
                category = "meeting_note"
            elif "decision" in title.lower():
                category = "decision_document"

        return TemporalItem(
            item_id=doc.id,
            item_type="document",
            timestamp=timestamp,
            category=category,
            source_node=doc,
            metadata={
                "title": getattr(doc, "title", None) or getattr(doc, "doc_title", None),
                "source_path": getattr(doc, "source_path", None),
            },
        )

    def _get_document_timestamp(self, doc: DocumentNode) -> Optional[datetime]:
        """Get the best available timestamp for a document.

        Priority: modified_at > updated_at > created_at > ingested_at
        """
        for field in ["modified_at", "updated_at", "created_at", "ingested_at"]:
            ts = getattr(doc, field, None)
            if ts is not None:
                return ts
        return None

    def get_categories(
        self,
        items: List[TemporalItem],
    ) -> Dict[str, List[TemporalItem]]:
        """Group temporal items by category.

        Args:
            items: List of temporal items

        Returns:
            Dictionary mapping category to list of items
        """
        by_category: Dict[str, List[TemporalItem]] = {}
        for item in items:
            if item.category not in by_category:
                by_category[item.category] = []
            by_category[item.category].append(item)
        return by_category

    def get_summary(self, items: List[TemporalItem]) -> Dict[str, Any]:
        """Get summary statistics for temporal items.

        Args:
            items: List of temporal items

        Returns:
            Dictionary with summary statistics
        """
        if not items:
            return {
                "total_items": 0,
                "by_type": {},
                "by_category": {},
                "time_range": None,
            }

        by_type: Dict[str, int] = {}
        by_category: Dict[str, int] = {}

        for item in items:
            by_type[item.item_type] = by_type.get(item.item_type, 0) + 1
            by_category[item.category] = by_category.get(item.category, 0) + 1

        return {
            "total_items": len(items),
            "by_type": by_type,
            "by_category": by_category,
            "time_range": {
                "start": items[0].timestamp.isoformat(),
                "end": items[-1].timestamp.isoformat(),
            },
        }
