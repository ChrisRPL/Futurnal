"""User Insight Storage for Chat-Generated Discoveries.

Phase C: Save Insight Command

This module handles saving user-generated insights from chat conversations
to the knowledge graph. Implements hybrid storage:
1. Immediate Insight node in Neo4j (high signal)
2. Markdown document for pipeline processing (learning)

Research Foundation:
- Training-Free GRPO (2510.08191v1): Natural language learning
- ProPerSim (2509.21730v1): Multi-turn context preservation

Option B Compliance:
- No model parameter updates
- Insights stored as natural language for token priors
- Ghost model FROZEN
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class UserInsight:
    """Represents a user-saved insight from chat conversation.

    Attributes:
        insight_id: Unique identifier
        content: The insight content (user description or AI summary)
        conversation_id: Source conversation ID
        related_entities: Entity IDs mentioned in conversation
        confidence: User confidence (1.0 for explicit saves)
        created_at: Timestamp
        source: How the insight was created ('user_explicit', 'ai_suggested')
        tags: Optional user-defined tags
    """

    insight_id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    conversation_id: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)
    confidence: float = 1.0  # User-saved insights have high confidence
    created_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "user_explicit"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "insight_id": self.insight_id,
            "content": self.content,
            "conversation_id": self.conversation_id,
            "related_entities": self.related_entities,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "tags": self.tags,
        }

    def to_markdown(self) -> str:
        """Convert to markdown format for document storage."""
        lines = [
            "---",
            f"insight_id: {self.insight_id}",
            f"created_at: {self.created_at.isoformat()}",
            f"source: {self.source}",
            f"confidence: {self.confidence}",
        ]

        if self.conversation_id:
            lines.append(f"conversation_id: {self.conversation_id}")

        if self.related_entities:
            lines.append(f"related_entities: {', '.join(self.related_entities)}")

        if self.tags:
            lines.append(f"tags: {', '.join(self.tags)}")

        lines.extend([
            "---",
            "",
            "# User Insight",
            "",
            self.content,
            "",
        ])

        return "\n".join(lines)


class InsightStorageService:
    """Service for storing user-generated insights.

    Implements hybrid storage:
    1. Neo4j Insight node for immediate graph access
    2. Markdown file for pipeline processing

    Usage:
        service = InsightStorageService()
        insight = await service.save_insight(
            content="My productivity drops with pre-10am meetings",
            conversation_id="conv_123",
            related_entities=["meeting", "productivity"]
        )
    """

    DEFAULT_INSIGHTS_DIR = "~/.futurnal/insights"

    def __init__(
        self,
        insights_dir: Optional[str] = None,
        neo4j_writer: Optional[Any] = None,
    ):
        """Initialize insight storage.

        Args:
            insights_dir: Directory for markdown insight files
            neo4j_writer: Neo4j writer for graph storage (optional)
        """
        self.insights_dir = Path(
            os.path.expanduser(insights_dir or self.DEFAULT_INSIGHTS_DIR)
        )
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.neo4j_writer = neo4j_writer

        logger.info(f"InsightStorageService initialized (dir={self.insights_dir})")

    async def save_insight(
        self,
        content: str,
        conversation_id: Optional[str] = None,
        related_entities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        source: str = "user_explicit",
    ) -> UserInsight:
        """Save a user-generated insight.

        Args:
            content: The insight content
            conversation_id: Source conversation ID
            related_entities: Entity IDs mentioned
            tags: Optional user-defined tags
            source: How the insight was created

        Returns:
            Created UserInsight
        """
        insight = UserInsight(
            content=content,
            conversation_id=conversation_id,
            related_entities=related_entities or [],
            tags=tags or [],
            source=source,
        )

        # 1. Save as markdown document
        await self._save_markdown(insight)

        # 2. Save to Neo4j if writer available
        if self.neo4j_writer:
            await self._save_to_neo4j(insight)

        logger.info(
            f"Saved insight {insight.insight_id}: "
            f"{content[:50]}... ({len(insight.related_entities)} entities)"
        )

        return insight

    async def _save_markdown(self, insight: UserInsight) -> Path:
        """Save insight as markdown file."""
        # Generate filename: YYYY-MM-DD-insight-{short_id}.md
        date_str = insight.created_at.strftime("%Y-%m-%d")
        short_id = insight.insight_id[:8]
        filename = f"{date_str}-insight-{short_id}.md"

        filepath = self.insights_dir / filename
        filepath.write_text(insight.to_markdown(), encoding="utf-8")

        logger.debug(f"Saved insight markdown: {filepath}")
        return filepath

    async def _save_to_neo4j(self, insight: UserInsight) -> None:
        """Save insight node to Neo4j."""
        if not self.neo4j_writer:
            return

        try:
            # Create Insight node
            cypher = """
            MERGE (i:Insight {id: $id})
            SET i.content = $content,
                i.conversation_id = $conversation_id,
                i.confidence = $confidence,
                i.created_at = datetime($created_at),
                i.source = $source,
                i.user_verified = true
            """

            params = {
                "id": insight.insight_id,
                "content": insight.content,
                "conversation_id": insight.conversation_id,
                "confidence": insight.confidence,
                "created_at": insight.created_at.isoformat(),
                "source": insight.source,
            }

            await self.neo4j_writer.execute(cypher, params)

            # Link to related entities
            if insight.related_entities:
                link_cypher = """
                MATCH (i:Insight {id: $insight_id})
                MATCH (e) WHERE e.id IN $entity_ids OR e.name IN $entity_ids
                MERGE (i)-[:RELATES_TO]->(e)
                """
                await self.neo4j_writer.execute(
                    link_cypher,
                    {
                        "insight_id": insight.insight_id,
                        "entity_ids": insight.related_entities,
                    },
                )

            logger.debug(f"Saved insight to Neo4j: {insight.insight_id}")

        except Exception as e:
            logger.warning(f"Failed to save insight to Neo4j: {e}")

    async def list_insights(
        self,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> List[UserInsight]:
        """List saved insights.

        Args:
            limit: Maximum number to return
            since: Only return insights after this date

        Returns:
            List of UserInsights
        """
        insights: List[UserInsight] = []

        for filepath in sorted(
            self.insights_dir.glob("*-insight-*.md"),
            reverse=True,
        )[:limit]:
            try:
                insight = await self._load_from_markdown(filepath)
                if insight:
                    if since and insight.created_at < since:
                        continue
                    insights.append(insight)
            except Exception as e:
                logger.warning(f"Failed to load insight from {filepath}: {e}")

        return insights

    async def _load_from_markdown(self, filepath: Path) -> Optional[UserInsight]:
        """Load insight from markdown file."""
        content = filepath.read_text(encoding="utf-8")

        # Parse frontmatter
        if not content.startswith("---"):
            return None

        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter = parts[1].strip()
        body = parts[2].strip()

        # Parse frontmatter fields
        fields = {}
        for line in frontmatter.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()

        # Extract content (after # User Insight header)
        if "# User Insight" in body:
            body = body.split("# User Insight", 1)[1].strip()

        return UserInsight(
            insight_id=fields.get("insight_id", str(uuid4())),
            content=body,
            conversation_id=fields.get("conversation_id"),
            related_entities=fields.get("related_entities", "").split(", ") if fields.get("related_entities") else [],
            confidence=float(fields.get("confidence", "1.0")),
            created_at=datetime.fromisoformat(fields.get("created_at", datetime.utcnow().isoformat())),
            source=fields.get("source", "user_explicit"),
            tags=fields.get("tags", "").split(", ") if fields.get("tags") else [],
        )

    async def delete_insight(self, insight_id: str) -> bool:
        """Delete an insight by ID.

        Args:
            insight_id: The insight ID to delete

        Returns:
            True if deleted, False if not found
        """
        short_id = insight_id[:8]

        # Find and delete markdown file
        for filepath in self.insights_dir.glob(f"*-insight-{short_id}.md"):
            filepath.unlink()
            logger.info(f"Deleted insight file: {filepath}")

            # Also delete from Neo4j if available
            if self.neo4j_writer:
                try:
                    await self.neo4j_writer.execute(
                        "MATCH (i:Insight {id: $id}) DETACH DELETE i",
                        {"id": insight_id},
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete insight from Neo4j: {e}")

            return True

        return False


# Convenience function for quick access
_default_service: Optional[InsightStorageService] = None


def get_insight_service() -> InsightStorageService:
    """Get the default insight storage service."""
    global _default_service
    if _default_service is None:
        _default_service = InsightStorageService()
    return _default_service
