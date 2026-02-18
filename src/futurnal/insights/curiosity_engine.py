"""Curiosity Engine for AGI-Level Knowledge Gap Detection.

AGI Phase 4: Implements curiosity-driven autotelic intelligence that
proactively identifies knowledge gaps and exploration opportunities.

Research Foundation:
- Curiosity-driven Autotelic AI (Oudeyer 2024): Information gain scoring
- DyMemR (2024): Memory forgetting curve analysis
- Intrinsic Motivation Theory: Learning progress as curiosity signal

Key Innovation:
Unlike reactive Q&A systems, the CuriosityEngine proactively:
1. Detects what's MISSING in user's knowledge (not just what's present)
2. Prioritizes gaps by information gain potential
3. Generates natural exploration prompts
4. Identifies forgotten but important memories

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

Option B Compliance:
- Graph analysis only, no model parameter updates
- All discoveries expressed as natural language for token priors
- Ghost model FROZEN throughout
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from futurnal.insights.community_detection import (
    BridgeOpportunity,
    Community,
    MemoryDecay,
    PKGCommunityDetector,
)

logger = logging.getLogger(__name__)


class GapType(str, Enum):
    """Categories of knowledge gaps."""

    MISSING_SYNTHESIS = "missing_synthesis"  # Consumed but not synthesized
    ASPIRATION_DISCONNECT = "aspiration_disconnect"  # Referenced but not linked to goals
    ISOLATED_CLUSTER = "isolated_cluster"  # Topics without connections
    INCOMPLETE_THOUGHT = "incomplete_thought"  # Started but not finished
    BROKEN_LINK = "broken_link"  # References to non-existent content
    FORGOTTEN_MEMORY = "forgotten_memory"  # DyMemR: Important but decaying
    BRIDGE_OPPORTUNITY = "bridge_opportunity"  # Potential cross-domain connection
    EXPLORATION_FRONTIER = "exploration_frontier"  # Edge of current knowledge


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
        information_gain: Expected info gain from addressing (0.0 to 1.0)
        related_notes: List of note URIs related to this gap
        related_aspirations: Links to aspirations if gap represents misalignment
        suggested_exploration: Prompts or questions to explore this gap
        created_at: When gap was detected
        addressed: Whether user has addressed this gap
        confidence: Confidence in gap detection
    """

    gap_id: str = field(default_factory=lambda: str(uuid4()))
    gap_type: GapType = GapType.MISSING_SYNTHESIS
    title: str = ""
    description: str = ""
    severity: float = 0.0  # 0.0 to 1.0
    information_gain: float = 0.0  # AGI Phase 4: Expected info gain
    related_notes: List[str] = field(default_factory=list)
    related_aspirations: List[str] = field(default_factory=list)
    suggested_exploration: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    addressed: bool = False
    confidence: float = 0.5  # Confidence in detection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type.value,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "information_gain": self.information_gain,
            "related_notes": self.related_notes,
            "related_aspirations": self.related_aspirations,
            "suggested_exploration": self.suggested_exploration,
            "created_at": self.created_at.isoformat(),
            "addressed": self.addressed,
            "confidence": self.confidence,
        }

    @property
    def priority_score(self) -> float:
        """Combined priority score for ranking gaps."""
        return (
            0.4 * self.severity +
            0.4 * self.information_gain +
            0.2 * self.confidence
        )


class CuriosityEngine:
    """Identifies knowledge gaps and exploration opportunities in the PKG.

    AGI Phase 4 implementation that analyzes the user's knowledge graph
    to find significant gaps, disconnections, and opportunities for synthesis.

    This implements curiosity-driven autotelic AI principles:
    1. Information Gain Maximization: Prioritize gaps with highest learning potential
    2. DyMemR Forgetting: Detect important memories that need reinforcement
    3. Aspiration Alignment: Find disconnects between knowledge and goals
    4. Bridge Detection: Identify cross-domain connection opportunities

    Example Usage:
        engine = CuriosityEngine()
        gaps = engine.detect_gaps(pkg_graph, aspirations)
        for gap in gaps:
            if gap.priority_score > 0.7:
                notify_user(gap)

    Option B Compliance:
    - No model updates, only graph analysis
    - Outputs natural language for token prior storage
    """

    # Configuration
    MIN_SEVERITY = 0.3  # Minimum severity to report
    MIN_INFO_GAIN = 0.2  # Minimum info gain to consider
    MAX_GAPS_PER_TYPE = 3  # Maximum gaps per category (reduced to avoid clutter)
    SYNTHESIS_THRESHOLD_DAYS = 30  # Days before expecting synthesis

    def __init__(
        self,
        min_severity: float = MIN_SEVERITY,
        min_info_gain: float = MIN_INFO_GAIN,
        community_detector: Optional[PKGCommunityDetector] = None,
        storage_path: Optional[str] = None,
    ):
        """Initialize curiosity engine.

        Args:
            min_severity: Minimum severity threshold for reporting gaps
            min_info_gain: Minimum information gain to consider gap valuable
            community_detector: Optional custom community detector
            storage_path: Path to persist gaps (default: ~/.futurnal/insights/gaps.json)
        """
        import os
        import json
        from pathlib import Path

        self.min_severity = min_severity
        self.min_info_gain = min_info_gain
        self._detector = community_detector or PKGCommunityDetector()

        # Persistent storage for knowledge gaps
        self._storage_path = Path(
            storage_path or os.path.expanduser("~/.futurnal/insights/gaps.json")
        )
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load cached gaps from storage
        self._cached_gaps: List[Dict[str, Any]] = []
        if self._storage_path.exists():
            try:
                self._cached_gaps = json.loads(self._storage_path.read_text())
                logger.info(f"Loaded {len(self._cached_gaps)} cached knowledge gaps")
            except Exception as e:
                logger.warning(f"Could not load cached gaps: {e}")

        logger.info(
            f"CuriosityEngine initialized "
            f"(min_severity={min_severity}, min_info_gain={min_info_gain})"
        )

    def detect_gaps(
        self,
        pkg_graph: Any,
        aspirations: Optional[List[Any]] = None,
        recent_events: Optional[List[Any]] = None,
        access_history: Optional[Dict[str, List[datetime]]] = None,
    ) -> List[KnowledgeGap]:
        """Detect knowledge gaps in the user's PKG.

        AGI Phase 4: Comprehensive gap detection using multiple strategies:
        1. Isolated cluster detection (bridge opportunities)
        2. DyMemR forgetting analysis (forgotten memories)
        3. Aspiration disconnect detection
        4. Missing synthesis detection
        5. Incomplete thought detection

        Args:
            pkg_graph: User's Personal Knowledge Graph
            aspirations: User's aspirations for alignment checking
            recent_events: Recent experiential events for context
            access_history: Node access history for DyMemR analysis

        Returns:
            List of detected knowledge gaps ranked by priority
        """
        all_gaps: List[KnowledgeGap] = []

        # 1. Detect isolated clusters
        isolated_gaps = self.detect_isolated_clusters(pkg_graph)
        all_gaps.extend(isolated_gaps[:self.MAX_GAPS_PER_TYPE])

        # 2. Detect forgotten memories (DyMemR)
        forgotten_gaps = self._detect_forgotten_memories(
            pkg_graph,
            access_history,
        )
        all_gaps.extend(forgotten_gaps[:self.MAX_GAPS_PER_TYPE])

        # 3. Detect aspiration disconnects
        if aspirations:
            aspiration_gaps = self.detect_aspiration_disconnects(
                pkg_graph,
                aspirations,
            )
            all_gaps.extend(aspiration_gaps[:self.MAX_GAPS_PER_TYPE])

        # 4. Detect missing synthesis
        if recent_events:
            synthesis_gaps = self.detect_missing_synthesis(
                pkg_graph,
                recent_events,
            )
            all_gaps.extend(synthesis_gaps[:self.MAX_GAPS_PER_TYPE])

        # 5. Detect incomplete thoughts
        if recent_events:
            incomplete_gaps = self.detect_incomplete_thoughts(
                pkg_graph,
                recent_events,
            )
            all_gaps.extend(incomplete_gaps[:self.MAX_GAPS_PER_TYPE])

        # 6. Detect bridge opportunities
        bridge_gaps = self._detect_bridge_opportunities(pkg_graph)
        all_gaps.extend(bridge_gaps[:self.MAX_GAPS_PER_TYPE])

        # Filter by thresholds
        filtered_gaps = [
            gap for gap in all_gaps
            if gap.severity >= self.min_severity
            and gap.information_gain >= self.min_info_gain
        ]

        # Sort by priority score
        filtered_gaps.sort(key=lambda g: -g.priority_score)

        # Cache and persist new gaps
        if filtered_gaps:
            self._cache_gaps(filtered_gaps)

        logger.info(
            f"Detected {len(filtered_gaps)} gaps "
            f"(from {len(all_gaps)} candidates)"
        )

        return filtered_gaps

    def _cache_gaps(self, gaps: List[KnowledgeGap]) -> None:
        """Cache and persist knowledge gaps to storage."""
        import json

        # Convert gaps to dictionaries for storage
        new_entries = []
        for gap in gaps:
            entry = {
                "gapId": gap.gap_id,
                "gapType": gap.gap_type.value if hasattr(gap.gap_type, 'value') else str(gap.gap_type),
                "title": gap.title,
                "description": gap.description,
                "informationGain": gap.information_gain,
                "relatedTopics": gap.related_notes or [],  # Fixed: was related_topics
                "explorationPrompts": gap.suggested_exploration or [],  # Fixed: was exploration_prompts
                "createdAt": gap.created_at.isoformat() if hasattr(gap.created_at, 'isoformat') else str(gap.created_at),
                "isAddressed": gap.addressed,  # Fixed: was is_addressed
            }
            new_entries.append(entry)

        # Add to cache (avoid duplicates by gap_id)
        existing_ids = {g.get("gapId") for g in self._cached_gaps}
        for entry in new_entries:
            if entry["gapId"] not in existing_ids:
                self._cached_gaps.append(entry)

        # Limit cache size (keep most recent 50)
        if len(self._cached_gaps) > 50:
            self._cached_gaps = self._cached_gaps[-50:]

        # Persist to storage
        try:
            self._storage_path.write_text(json.dumps(self._cached_gaps, indent=2))
            logger.debug(f"Persisted {len(self._cached_gaps)} gaps to {self._storage_path}")
        except Exception as e:
            logger.warning(f"Failed to persist gaps: {e}")

    def detect_isolated_clusters(
        self,
        pkg_graph: Any,
        min_cluster_size: int = 3,
    ) -> List[KnowledgeGap]:
        """Detect concept clusters without connections to other topics.

        Example: "Three main research areas with no bridges between them"

        Args:
            pkg_graph: Personal Knowledge Graph
            min_cluster_size: Minimum nodes for a valid cluster

        Returns:
            List of gaps for isolated clusters
        """
        gaps: List[KnowledgeGap] = []

        try:
            # Detect communities
            communities = self._detector.detect_communities(pkg_graph)

            # Find isolated ones
            isolated = self._detector.find_isolated_communities(
                communities,
                pkg_graph,
            )

            for community in isolated:
                # Compute information gain
                info_gain = self._detector.compute_information_gain(
                    community,
                    pkg_graph,
                )

                # Generate exploration prompts
                prompts = self._generate_cluster_prompts(community)

                gap = KnowledgeGap(
                    gap_type=GapType.ISOLATED_CLUSTER,
                    title=f"Isolated knowledge cluster: {', '.join(community.central_topics[:2])}",
                    description=(
                        f"Found an isolated cluster of {community.size} related concepts "
                        f"around {', '.join(community.central_topics)}. "
                        f"This cluster has few connections to your other knowledge areas, "
                        f"representing an opportunity for cross-domain insights."
                    ),
                    severity=0.6 if community.size > 5 else 0.4,
                    information_gain=info_gain,
                    related_notes=list(community.nodes)[:10],
                    suggested_exploration=prompts,
                    confidence=0.8,
                )

                gaps.append(gap)

        except Exception as e:
            logger.error(f"Error detecting isolated clusters: {e}")

        return gaps

    def _detect_forgotten_memories(
        self,
        pkg_graph: Any,
        access_history: Optional[Dict[str, List[datetime]]] = None,
    ) -> List[KnowledgeGap]:
        """Detect important but decaying memories using DyMemR principles.

        DyMemR (2024) models memory forgetting to identify memories
        that need reinforcement before being lost.

        Args:
            pkg_graph: Personal Knowledge Graph
            access_history: Node access history

        Returns:
            List of gaps for forgotten memories
        """
        gaps: List[KnowledgeGap] = []

        try:
            # Analyze memory decay
            decay_results = self._detector.analyze_memory_decay(
                pkg_graph,
                access_history,
            )

            # Find memories needing reinforcement
            for memory in decay_results:
                if not memory.needs_reinforcement:
                    continue

                # Compute info gain from reinforcement
                info_gain = self._detector.compute_information_gain(
                    memory,
                    pkg_graph,
                )

                # Generate prompts
                prompts = self._generate_memory_prompts(memory)

                gap = KnowledgeGap(
                    gap_type=GapType.FORGOTTEN_MEMORY,
                    title=f"Fading memory: {memory.title}",
                    description=(
                        f"'{memory.title}' is an important note "
                        f"(importance: {memory.importance_score:.0%}) "
                        f"that hasn't been accessed in a while "
                        f"(decay: {memory.decay_score:.0%}). "
                        f"Reinforcing this memory could help maintain "
                        f"your knowledge connections."
                    ),
                    severity=memory.importance_score * (1 - memory.decay_score),
                    information_gain=info_gain,
                    related_notes=[memory.node_id],
                    suggested_exploration=prompts,
                    confidence=0.7,
                )

                gaps.append(gap)

                if len(gaps) >= self.MAX_GAPS_PER_TYPE:
                    break

        except Exception as e:
            logger.error(f"Error detecting forgotten memories: {e}")

        return gaps

    def _detect_bridge_opportunities(
        self,
        pkg_graph: Any,
    ) -> List[KnowledgeGap]:
        """Detect opportunities to bridge isolated communities.

        Args:
            pkg_graph: Personal Knowledge Graph

        Returns:
            List of gaps for bridge opportunities
        """
        gaps: List[KnowledgeGap] = []

        try:
            # Detect communities
            communities = self._detector.detect_communities(pkg_graph)

            # Find bridge opportunities
            opportunities = self._detector.find_bridge_opportunities(
                communities,
                pkg_graph,
            )

            for opportunity in opportunities:
                # Generate prompts
                prompts = self._generate_bridge_prompts(opportunity, communities)

                gap = KnowledgeGap(
                    gap_type=GapType.BRIDGE_OPPORTUNITY,
                    title=f"Bridge opportunity: {opportunity.community_a} ↔ {opportunity.community_b}",
                    description=(
                        f"Two knowledge areas could potentially be connected. "
                        f"Bridging them could lead to novel insights "
                        f"(expected info gain: {opportunity.information_gain:.0%})."
                    ),
                    severity=0.5,
                    information_gain=opportunity.information_gain,
                    related_notes=[
                        node for pair in opportunity.suggested_connections
                        for node in pair
                    ],
                    suggested_exploration=prompts,
                    confidence=0.6,
                )

                gaps.append(gap)

        except Exception as e:
            logger.error(f"Error detecting bridge opportunities: {e}")

        return gaps

    def detect_missing_synthesis(
        self,
        pkg_graph: Any,
        recent_events: List[Any],
    ) -> List[KnowledgeGap]:
        """Detect content consumed but not synthesized into notes.

        Example: "You've read 20 articles about causal inference but haven't
        created synthesis notes"

        Args:
            pkg_graph: Personal Knowledge Graph
            recent_events: Recent consumption events

        Returns:
            List of gaps for missing synthesis
        """
        gaps: List[KnowledgeGap] = []

        try:
            # Group events by topic/theme
            topic_events: Dict[str, List[Any]] = {}

            for event in recent_events:
                # Extract topic from event
                topic = self._extract_topic(event)
                if topic:
                    if topic not in topic_events:
                        topic_events[topic] = []
                    topic_events[topic].append(event)

            # Check for topics with consumption but no synthesis
            for topic, events in topic_events.items():
                if len(events) < 3:  # Need multiple events to suggest synthesis
                    continue

                # Check if synthesis exists
                has_synthesis = self._check_synthesis_exists(pkg_graph, topic)

                if not has_synthesis:
                    # Calculate info gain
                    info_gain = min(1.0, len(events) / 10)

                    # Generate prompts
                    prompts = self._generate_synthesis_prompts(topic, events)

                    gap = KnowledgeGap(
                        gap_type=GapType.MISSING_SYNTHESIS,
                        title=f"Synthesis opportunity: {topic}",
                        description=(
                            f"You've consumed {len(events)} items about '{topic}' "
                            f"but haven't created synthesis notes. "
                            f"Synthesizing what you've learned could solidify understanding "
                            f"and create valuable reference material."
                        ),
                        severity=min(0.9, 0.4 + len(events) * 0.05),
                        information_gain=info_gain,
                        related_notes=[
                            e.id if hasattr(e, "id") else str(e)
                            for e in events[:10]
                        ],
                        suggested_exploration=prompts,
                        confidence=0.7,
                    )

                    gaps.append(gap)

        except Exception as e:
            logger.error(f"Error detecting missing synthesis: {e}")

        return gaps

    def detect_aspiration_disconnects(
        self,
        pkg_graph: Any,
        aspirations: List[Any],
    ) -> List[KnowledgeGap]:
        """Detect notes not linked to relevant aspirations.

        Example: "15 'Project Titan' notes not linked to 'Lead projects' aspiration"

        Args:
            pkg_graph: Personal Knowledge Graph
            aspirations: User's aspirations

        Returns:
            List of gaps for aspiration disconnects
        """
        gaps: List[KnowledgeGap] = []

        try:
            for aspiration in aspirations:
                asp_id = aspiration.id if hasattr(aspiration, "id") else str(aspiration)
                asp_name = aspiration.name if hasattr(aspiration, "name") else str(aspiration)

                # Find nodes semantically related but not linked
                related_but_unlinked = self._find_unlinked_related_nodes(
                    pkg_graph,
                    aspiration,
                )

                if len(related_but_unlinked) >= 3:
                    # Generate prompts
                    prompts = self._generate_aspiration_prompts(
                        asp_name,
                        related_but_unlinked,
                    )

                    gap = KnowledgeGap(
                        gap_type=GapType.ASPIRATION_DISCONNECT,
                        title=f"Aspiration gap: {asp_name}",
                        description=(
                            f"Found {len(related_but_unlinked)} notes that seem related to "
                            f"your aspiration '{asp_name}' but aren't explicitly linked. "
                            f"Connecting these could help track progress toward your goal."
                        ),
                        severity=min(0.8, 0.4 + len(related_but_unlinked) * 0.04),
                        information_gain=0.6,
                        related_notes=related_but_unlinked[:10],
                        related_aspirations=[asp_id],
                        suggested_exploration=prompts,
                        confidence=0.6,
                    )

                    gaps.append(gap)

        except Exception as e:
            logger.error(f"Error detecting aspiration disconnects: {e}")

        return gaps

    def detect_incomplete_thoughts(
        self,
        pkg_graph: Any,
        recent_events: List[Any],
    ) -> List[KnowledgeGap]:
        """Detect thought traces that were started but not completed.

        Example: "Note 'Draft: Causal Reasoning Framework' last edited 3 months ago"

        Args:
            pkg_graph: Personal Knowledge Graph
            recent_events: Recent events for context

        Returns:
            List of gaps for incomplete thoughts
        """
        gaps: List[KnowledgeGap] = []

        try:
            # Get all nodes with draft markers
            draft_nodes = self._find_draft_nodes(pkg_graph)

            now = datetime.utcnow()

            for node_id, node_data in draft_nodes:
                # Check if stale (not edited recently)
                last_modified = node_data.get("modified", now - timedelta(days=90))
                if isinstance(last_modified, str):
                    last_modified = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))

                days_since_edit = (now - last_modified).days

                if days_since_edit > 30:  # Stale draft
                    title = node_data.get("title", node_id)

                    # Generate prompts
                    prompts = self._generate_incomplete_prompts(title, days_since_edit)

                    gap = KnowledgeGap(
                        gap_type=GapType.INCOMPLETE_THOUGHT,
                        title=f"Incomplete thought: {title}",
                        description=(
                            f"'{title}' appears to be an incomplete draft "
                            f"last edited {days_since_edit} days ago. "
                            f"Would you like to complete this thought or archive it?"
                        ),
                        severity=min(0.7, 0.3 + days_since_edit * 0.003),
                        information_gain=0.4,
                        related_notes=[node_id],
                        suggested_exploration=prompts,
                        confidence=0.8,
                    )

                    gaps.append(gap)

        except Exception as e:
            logger.error(f"Error detecting incomplete thoughts: {e}")

        return gaps

    def suggest_exploration_prompts(
        self,
        gap: KnowledgeGap,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generate exploration prompts for a detected gap.

        Creates natural, curiosity-inspiring questions to guide exploration.

        Args:
            gap: The knowledge gap
            user_context: Optional user context for personalization

        Returns:
            List of exploration prompts
        """
        prompts = list(gap.suggested_exploration)

        # Add generic prompts based on gap type
        if gap.gap_type == GapType.ISOLATED_CLUSTER:
            prompts.append(
                "What connections might exist between this topic and your other interests?"
            )
        elif gap.gap_type == GapType.FORGOTTEN_MEMORY:
            prompts.append(
                "What new insights have you gained since you last explored this topic?"
            )
        elif gap.gap_type == GapType.MISSING_SYNTHESIS:
            prompts.append(
                "What key patterns or principles have you noticed across these materials?"
            )
        elif gap.gap_type == GapType.ASPIRATION_DISCONNECT:
            prompts.append(
                "How might this knowledge help you progress toward your goal?"
            )

        return prompts[:5]  # Limit to 5 prompts

    def compute_information_gain(
        self,
        gap: KnowledgeGap,
        pkg_graph: Any,
    ) -> float:
        """Compute expected information gain from addressing a gap.

        Based on curiosity-driven exploration principles:
        - Higher gain for gaps connecting diverse knowledge areas
        - Lower gain for highly familiar or highly alien content

        Args:
            gap: The knowledge gap
            pkg_graph: Current knowledge graph

        Returns:
            Information gain score (0-1)
        """
        return self._detector.compute_information_gain(gap, pkg_graph)

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _generate_cluster_prompts(
        self,
        community: Community,
    ) -> List[str]:
        """Generate prompts for isolated cluster gaps."""
        topics = ", ".join(community.central_topics[:3])
        return [
            f"What aspects of {topics} might connect to your other work?",
            f"Are there insights from {topics} that apply to other areas?",
            f"What questions do you have about {topics} that remain unanswered?",
        ]

    def _generate_memory_prompts(
        self,
        memory: MemoryDecay,
    ) -> List[str]:
        """Generate prompts for forgotten memory gaps."""
        return [
            f"What do you remember most about '{memory.title}'?",
            f"Has your understanding of this topic evolved since then?",
            f"How might revisiting this connect to your current work?",
        ]

    def _generate_bridge_prompts(
        self,
        opportunity: BridgeOpportunity,
        communities: List[Community],
    ) -> List[str]:
        """Generate prompts for bridge opportunity gaps."""
        return [
            "What unexpected connections might exist between these areas?",
            "How might combining these perspectives lead to new insights?",
            "What questions arise when you consider these topics together?",
        ]

    def _generate_synthesis_prompts(
        self,
        topic: str,
        events: List[Any],
    ) -> List[str]:
        """Generate prompts for missing synthesis gaps."""
        return [
            f"What are the key takeaways from your exploration of {topic}?",
            f"What patterns have you noticed across these {len(events)} items?",
            f"How would you explain the main ideas of {topic} to someone else?",
        ]

    def _generate_aspiration_prompts(
        self,
        aspiration_name: str,
        related_notes: List[str],
    ) -> List[str]:
        """Generate prompts for aspiration disconnect gaps."""
        return [
            f"How do these notes relate to '{aspiration_name}'?",
            f"What progress do these represent toward '{aspiration_name}'?",
            f"What's missing to fully achieve '{aspiration_name}'?",
        ]

    def _generate_incomplete_prompts(
        self,
        title: str,
        days_ago: int,
    ) -> List[str]:
        """Generate prompts for incomplete thought gaps."""
        return [
            f"What was the original intention behind '{title}'?",
            f"What would it take to complete this thought?",
            f"Is this still relevant, or should it be archived?",
        ]

    def _extract_topic(self, event: Any) -> Optional[str]:
        """Extract topic from an event."""
        if hasattr(event, "topic"):
            return event.topic
        if hasattr(event, "tags"):
            tags = event.tags
            if tags:
                return tags[0]
        if hasattr(event, "title"):
            # Simple extraction from title
            title = event.title
            if ":" in title:
                return title.split(":")[0].strip()
            return title[:30]
        return None

    def _check_synthesis_exists(
        self,
        pkg_graph: Any,
        topic: str,
    ) -> bool:
        """Check if synthesis notes exist for a topic."""
        # Look for nodes with synthesis markers
        if hasattr(pkg_graph, "search"):
            try:
                results = pkg_graph.search(f"synthesis {topic}")
                return len(results) > 0
            except Exception:
                pass
        return False

    def _find_unlinked_related_nodes(
        self,
        pkg_graph: Any,
        aspiration: Any,
    ) -> List[str]:
        """Find nodes related to aspiration but not linked."""
        related: List[str] = []

        # This would use semantic search in production
        # For now, return empty to avoid false positives
        return related

    def _find_draft_nodes(
        self,
        pkg_graph: Any,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Find nodes marked as drafts."""
        drafts: List[Tuple[str, Dict[str, Any]]] = []

        if hasattr(pkg_graph, "nodes"):
            try:
                for node_id in pkg_graph.nodes():
                    data = pkg_graph.nodes[node_id]
                    title = data.get("title", "").lower()
                    status = data.get("status", "").lower()

                    if "draft" in title or status in ["draft", "incomplete", "wip"]:
                        drafts.append((node_id, dict(data)))
            except Exception:
                pass

        return drafts
