"""Curiosity Engine for Phase 2 (Analyst) knowledge gap detection.

Phase 2 will use this module to identify significant gaps in the user's knowledge
network that warrant investigation. The Curiosity Engine complements Emergent
Insights by proactively highlighting what's MISSING rather than what's present.

Example Knowledge Gaps:
- "I notice you have 15 notes referencing 'Project Titan' but none are linked
  to your stated aspiration of 'Lead high-impact projects'—this gap might
  indicate a documentation blind spot or potential misalignment."
- "You've read 20 articles about causal inference in the past 3 months, but
  haven't created any notes synthesizing key concepts. Would you like to explore
  what you've learned?"
- "Your research notes cluster around three main topics, but there are no
  connections between them. This might represent an opportunity for novel
  insights by bridging these domains."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class GapType(str, Enum):
    """Categories of knowledge gaps."""

    MISSING_SYNTHESIS = "missing_synthesis"  # Consumed but not synthesized
    ASPIRATION_DISCONNECT = "aspiration_disconnect"  # Referenced but not linked to goals
    ISOLATED_CLUSTER = "isolated_cluster"  # Topics without connections
    INCOMPLETE_THOUGHT = "incomplete_thought"  # Started but not finished
    BROKEN_LINK = "broken_link"  # References to non-existent content


@dataclass
class KnowledgeGap:
    """Represents a detected gap in the user's knowledge network.

    Knowledge gaps are opportunities for exploration and synthesis identified
    by the Curiosity Engine through graph analysis.

    Attributes:
        gap_id: Unique identifier
        gap_type: Category of gap
        title: Short description of the gap
        description: Detailed explanation
        severity: How significant the gap is (0.0 to 1.0)
        related_notes: List of note URIs related to this gap
        related_aspirations: Links to aspirations if gap represents misalignment
        suggested_exploration: Prompts or questions to explore this gap
        created_at: When gap was detected
        addressed: Whether user has addressed this gap
    """

    gap_id: str = field(default_factory=lambda: str(uuid4()))
    gap_type: GapType = GapType.MISSING_SYNTHESIS
    title: str = ""
    description: str = ""
    severity: float = 0.0  # 0.0 to 1.0
    related_notes: List[str] = field(default_factory=list)
    related_aspirations: List[str] = field(default_factory=list)
    suggested_exploration: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    addressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type.value,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "related_notes": self.related_notes,
            "related_aspirations": self.related_aspirations,
            "suggested_exploration": self.suggested_exploration,
            "created_at": self.created_at.isoformat(),
            "addressed": self.addressed,
        }


class CuriosityEngine:
    """Identifies knowledge gaps and exploration opportunities in the PKG.

    Phase 2 (Analyst) implementation will analyze the user's knowledge graph
    to find significant gaps, disconnections, and opportunities for synthesis.
    This enables the Ghost to guide users toward deeper understanding.

    Planned Capabilities:
    - Detect references without corresponding content (broken links)
    - Identify consumed content without synthesis notes
    - Find disconnects between aspirations and related notes
    - Discover isolated concept clusters that could be bridged
    - Highlight incomplete thought traces (started but not finished)

    Example Usage (Phase 2):
        engine = CuriosityEngine()
        gaps = engine.detect_gaps(pkg_graph, aspirations)
        for gap in gaps:
            if gap.severity > 0.7:
                notify_user(gap)
    """

    def __init__(self, min_severity: float = 0.5):
        """Initialize curiosity engine.

        Args:
            min_severity: Minimum severity threshold for reporting gaps
        """
        self.min_severity = min_severity

    def detect_gaps(
        self,
        pkg_graph: Any,
        aspirations: Optional[List[Any]] = None,
        recent_events: Optional[List[Any]] = None
    ) -> List[KnowledgeGap]:
        """Detect knowledge gaps in the user's PKG.

        TODO: Phase 2 - Implement gap detection algorithms
        TODO: Phase 2 - Add graph analysis for isolated clusters
        TODO: Phase 2 - Detect aspiration misalignments
        TODO: Phase 2 - Find missing synthesis opportunities
        TODO: Phase 2 - Identify incomplete thought chains

        Args:
            pkg_graph: User's Personal Knowledge Graph
            aspirations: User's aspirations for alignment checking
            recent_events: Recent experiential events for context

        Returns:
            List of detected knowledge gaps ranked by severity
        """
        # Phase 2 implementation placeholder
        gaps: List[KnowledgeGap] = []

        # TODO: Implement gap detection
        # 1. Analyze graph structure for disconnected components
        # 2. Find references without corresponding nodes
        # 3. Identify content consumed without synthesis
        # 4. Detect aspiration-note disconnects
        # 5. Generate exploration suggestions

        return gaps

    def detect_missing_synthesis(
        self,
        pkg_graph: Any,
        recent_events: List[Any]
    ) -> List[KnowledgeGap]:
        """Detect content consumed but not synthesized into notes.

        Example: "You've read 20 articles about causal inference but haven't
        created synthesis notes"

        TODO: Phase 2 - Implement synthesis gap detection
        """
        # Phase 2 implementation placeholder
        return []

    def detect_aspiration_disconnects(
        self,
        pkg_graph: Any,
        aspirations: List[Any]
    ) -> List[KnowledgeGap]:
        """Detect notes not linked to relevant aspirations.

        Example: "15 'Project Titan' notes not linked to 'Lead projects' aspiration"

        TODO: Phase 2 - Implement aspiration alignment analysis
        """
        # Phase 2 implementation placeholder
        return []

    def detect_isolated_clusters(
        self,
        pkg_graph: Any,
        min_cluster_size: int = 3
    ) -> List[KnowledgeGap]:
        """Detect concept clusters without connections to other topics.

        Example: "Three main research areas with no bridges between them"

        TODO: Phase 2 - Implement graph community detection
        """
        # Phase 2 implementation placeholder
        return []

    def detect_incomplete_thoughts(
        self,
        pkg_graph: Any,
        recent_events: List[Any]
    ) -> List[KnowledgeGap]:
        """Detect thought traces that were started but not completed.

        Example: "Note 'Draft: Causal Reasoning Framework' last edited 3 months ago"

        TODO: Phase 2 - Implement incomplete thought detection
        """
        # Phase 2 implementation placeholder
        return []

    def suggest_exploration_prompts(
        self,
        gap: KnowledgeGap,
        user_context: Dict[str, Any]
    ) -> List[str]:
        """Generate exploration prompts for a detected gap.

        Example prompts:
        - "Would you like to synthesize what you've learned about causal inference?"
        - "How does Project Titan relate to your goal of leading high-impact projects?"
        - "What connections do you see between your research on X and Y?"

        TODO: Phase 2 - Implement prompt generation
        """
        # Phase 2 implementation placeholder
        return []
