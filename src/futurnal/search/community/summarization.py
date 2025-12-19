"""
Community Summarization.

Generates summaries for communities at different hierarchy levels:
- Entity-level summaries
- Relationship summaries
- Hierarchical aggregation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from collections import defaultdict
import hashlib
import json

from .detection import Community
from .hierarchy import CommunityHierarchy

logger = logging.getLogger(__name__)


@dataclass
class CommunitySummary:
    """Summary of a community."""
    community_id: str
    title: str
    summary: str
    keywords: List[str]
    key_entities: List[str]
    key_relationships: List[str]
    statistics: Dict[str, Any]
    level: int
    embedding: Optional[List[float]] = None


@dataclass
class SummaryCache:
    """Cache for community summaries."""
    summaries: Dict[str, CommunitySummary] = field(default_factory=dict)
    content_hashes: Dict[str, str] = field(default_factory=dict)
    max_size: int = 10000

    def get(self, community_id: str) -> Optional[CommunitySummary]:
        """Get cached summary."""
        return self.summaries.get(community_id)

    def put(
        self,
        community_id: str,
        summary: CommunitySummary,
        content_hash: Optional[str] = None
    ) -> None:
        """Cache a summary."""
        if len(self.summaries) >= self.max_size:
            # Simple eviction: remove oldest
            oldest_key = next(iter(self.summaries))
            del self.summaries[oldest_key]

        self.summaries[community_id] = summary
        if content_hash:
            self.content_hashes[community_id] = content_hash

    def is_valid(self, community_id: str, current_hash: str) -> bool:
        """Check if cached summary is still valid."""
        return (
            community_id in self.summaries and
            self.content_hashes.get(community_id) == current_hash
        )

    def invalidate(self, community_id: str) -> None:
        """Invalidate a cached summary."""
        self.summaries.pop(community_id, None)
        self.content_hashes.pop(community_id, None)


class CommunitySummarizer:
    """
    Generates summaries for individual communities.

    Uses LLM for natural language summarization with
    structured extraction of key information.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        use_local_llm: bool = True,
        max_entities_in_prompt: int = 50,
        max_relationships_in_prompt: int = 30
    ):
        self.llm_client = llm_client
        self.use_local_llm = use_local_llm
        self.max_entities = max_entities_in_prompt
        self.max_relationships = max_relationships_in_prompt

    async def summarize(
        self,
        community: Community,
        entity_details: Dict[str, Dict[str, Any]],
        relationship_details: List[Dict[str, Any]],
        child_summaries: Optional[List[CommunitySummary]] = None
    ) -> CommunitySummary:
        """
        Generate summary for a community.

        Args:
            community: The community to summarize
            entity_details: Details about entities in the community
            relationship_details: Details about relationships
            child_summaries: Summaries of child communities (for hierarchical)

        Returns:
            CommunitySummary
        """
        # Gather key entities
        key_entities = self._extract_key_entities(community.nodes, entity_details)

        # Gather key relationships
        key_relationships = self._extract_key_relationships(relationship_details)

        # Compute statistics
        statistics = {
            "num_entities": len(community.nodes),
            "num_relationships": len(relationship_details),
            "level": community.level,
            "has_children": len(community.child_ids) > 0
        }

        # Generate summary using LLM
        summary_text, title, keywords = await self._generate_summary(
            key_entities,
            key_relationships,
            child_summaries,
            community.level
        )

        return CommunitySummary(
            community_id=community.id,
            title=title,
            summary=summary_text,
            keywords=keywords,
            key_entities=key_entities[:10],
            key_relationships=key_relationships[:10],
            statistics=statistics,
            level=community.level
        )

    def summarize_sync(
        self,
        community: Community,
        entity_details: Dict[str, Dict[str, Any]],
        relationship_details: List[Dict[str, Any]],
        child_summaries: Optional[List[CommunitySummary]] = None
    ) -> CommunitySummary:
        """Synchronous version of summarize."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.summarize(community, entity_details, relationship_details, child_summaries)
        )

    def _extract_key_entities(
        self,
        node_ids: Set[str],
        entity_details: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Extract and rank key entities."""
        ranked_entities = []

        for node_id in node_ids:
            if node_id in entity_details:
                details = entity_details[node_id]
                # Score based on connectivity, type importance, etc.
                score = details.get("degree", 0) + details.get("importance", 0)
                name = details.get("name", node_id)
                ranked_entities.append((name, score))

        ranked_entities.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in ranked_entities[:self.max_entities]]

    def _extract_key_relationships(
        self,
        relationship_details: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract and format key relationships."""
        formatted = []

        for rel in relationship_details[:self.max_relationships]:
            head = rel.get("head_name", rel.get("head_id", "?"))
            tail = rel.get("tail_name", rel.get("tail_id", "?"))
            rel_type = rel.get("type", "related_to")
            formatted.append(f"{head} --[{rel_type}]--> {tail}")

        return formatted

    async def _generate_summary(
        self,
        key_entities: List[str],
        key_relationships: List[str],
        child_summaries: Optional[List[CommunitySummary]],
        level: int
    ) -> tuple[str, str, List[str]]:
        """Generate natural language summary using LLM."""
        prompt = self._build_summary_prompt(
            key_entities, key_relationships, child_summaries, level
        )

        if self.llm_client:
            response = await self._call_llm(prompt)
            return self._parse_summary_response(response)
        else:
            # Fallback: generate simple summary without LLM
            return self._generate_fallback_summary(
                key_entities, key_relationships, child_summaries
            )

    def _build_summary_prompt(
        self,
        key_entities: List[str],
        key_relationships: List[str],
        child_summaries: Optional[List[CommunitySummary]],
        level: int
    ) -> str:
        """Build prompt for LLM summarization."""
        prompt = """Generate a concise summary for a knowledge community.

## Key Entities:
"""
        for entity in key_entities[:20]:
            prompt += f"- {entity}\n"

        prompt += "\n## Key Relationships:\n"
        for rel in key_relationships[:15]:
            prompt += f"- {rel}\n"

        if child_summaries:
            prompt += "\n## Child Community Summaries:\n"
            for child in child_summaries[:5]:
                prompt += f"- {child.title}: {child.summary[:100]}...\n"

        prompt += """
## Output Format:
Please provide:
1. TITLE: A brief title for this community (max 10 words)
2. SUMMARY: A 2-3 sentence summary of what this community represents
3. KEYWORDS: 5-10 keywords that best describe this community

Format your response as:
TITLE: <title>
SUMMARY: <summary>
KEYWORDS: <keyword1>, <keyword2>, ...
"""
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for summary generation."""
        try:
            if hasattr(self.llm_client, "generate"):
                response = await self.llm_client.generate(prompt)
                return response
            elif hasattr(self.llm_client, "chat"):
                response = await self.llm_client.chat([{"role": "user", "content": prompt}])
                return response.get("content", "")
            else:
                return ""
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""

    def _parse_summary_response(self, response: str) -> tuple[str, str, List[str]]:
        """Parse LLM response into structured summary."""
        title = "Community"
        summary = ""
        keywords = []

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("TITLE:"):
                title = line[6:].strip()
            elif line.startswith("SUMMARY:"):
                summary = line[8:].strip()
            elif line.startswith("KEYWORDS:"):
                kw_str = line[9:].strip()
                keywords = [k.strip() for k in kw_str.split(",")]

        if not summary:
            summary = response[:200] if response else "No summary available."

        return summary, title, keywords

    def _generate_fallback_summary(
        self,
        key_entities: List[str],
        key_relationships: List[str],
        child_summaries: Optional[List[CommunitySummary]]
    ) -> tuple[str, str, List[str]]:
        """Generate simple summary without LLM."""
        if key_entities:
            title = f"Community: {', '.join(key_entities[:3])}"
            summary = f"A community containing {len(key_entities)} entities including {', '.join(key_entities[:5])}."
            keywords = key_entities[:10]
        else:
            title = "Unknown Community"
            summary = "A community with no identified entities."
            keywords = []

        if key_relationships:
            summary += f" Key relationships include: {', '.join(key_relationships[:3])}."

        return summary, title, keywords


class HierarchicalSummarizer:
    """
    Generates hierarchical summaries across community levels.

    Implements bottom-up summarization with aggregation.
    """

    def __init__(
        self,
        base_summarizer: Optional[CommunitySummarizer] = None,
        cache: Optional[SummaryCache] = None
    ):
        self.summarizer = base_summarizer or CommunitySummarizer()
        self.cache = cache or SummaryCache()

    async def summarize_hierarchy(
        self,
        hierarchy: CommunityHierarchy,
        entity_loader: Callable[[Set[str]], Dict[str, Dict[str, Any]]],
        relationship_loader: Callable[[Set[str]], List[Dict[str, Any]]]
    ) -> Dict[str, CommunitySummary]:
        """
        Generate summaries for all communities in hierarchy.

        Uses bottom-up approach: summarize leaves first, then aggregate.

        Args:
            hierarchy: Community hierarchy
            entity_loader: Function to load entity details
            relationship_loader: Function to load relationship details

        Returns:
            Dictionary of community_id -> CommunitySummary
        """
        all_summaries: Dict[str, CommunitySummary] = {}

        # Process levels bottom-up
        for level_idx in range(len(hierarchy.levels) - 1, -1, -1):
            level = hierarchy.levels[level_idx]

            for community in level.communities:
                # Check cache
                content_hash = self._compute_content_hash(community)
                if self.cache.is_valid(community.id, content_hash):
                    cached = self.cache.get(community.id)
                    if cached:
                        all_summaries[community.id] = cached
                        continue

                # Load entity and relationship details
                entity_details = entity_loader(community.nodes)
                relationship_details = relationship_loader(community.nodes)

                # Get child summaries
                child_summaries = [
                    all_summaries[child_id]
                    for child_id in community.child_ids
                    if child_id in all_summaries
                ]

                # Generate summary
                summary = await self.summarizer.summarize(
                    community,
                    entity_details,
                    relationship_details,
                    child_summaries if child_summaries else None
                )

                # Cache and store
                self.cache.put(community.id, summary, content_hash)
                all_summaries[community.id] = summary

        logger.info(f"Generated {len(all_summaries)} community summaries")
        return all_summaries

    def summarize_hierarchy_sync(
        self,
        hierarchy: CommunityHierarchy,
        entity_loader: Callable[[Set[str]], Dict[str, Dict[str, Any]]],
        relationship_loader: Callable[[Set[str]], List[Dict[str, Any]]]
    ) -> Dict[str, CommunitySummary]:
        """Synchronous version of summarize_hierarchy."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.summarize_hierarchy(hierarchy, entity_loader, relationship_loader)
        )

    def get_summary_for_query(
        self,
        query: str,
        hierarchy: CommunityHierarchy,
        summaries: Dict[str, CommunitySummary],
        top_k: int = 5
    ) -> List[CommunitySummary]:
        """
        Find most relevant community summaries for a query.

        Args:
            query: User query
            hierarchy: Community hierarchy
            summaries: Pre-computed summaries
            top_k: Number of summaries to return

        Returns:
            Most relevant community summaries
        """
        # Simple keyword matching for now
        # Could be enhanced with embedding similarity
        query_tokens = set(query.lower().split())
        scored_summaries = []

        for summary in summaries.values():
            # Score based on keyword overlap
            keywords_lower = [k.lower() for k in summary.keywords]
            title_tokens = set(summary.title.lower().split())
            summary_tokens = set(summary.summary.lower().split())

            keyword_overlap = len(query_tokens & set(keywords_lower))
            title_overlap = len(query_tokens & title_tokens)
            summary_overlap = len(query_tokens & summary_tokens)

            score = keyword_overlap * 3 + title_overlap * 2 + summary_overlap * 0.5

            # Prefer higher-level (more general) communities for overview queries
            # Prefer lower-level (more specific) for detailed queries
            if any(w in query.lower() for w in ["overview", "summary", "general"]):
                score += (hierarchy.max_depth - summary.level) * 0.5
            else:
                score += summary.level * 0.3

            scored_summaries.append((summary, score))

        scored_summaries.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored_summaries[:top_k]]

    def _compute_content_hash(self, community: Community) -> str:
        """Compute hash of community content for cache validation."""
        content = json.dumps({
            "nodes": sorted(list(community.nodes)),
            "level": community.level,
            "children": sorted(community.child_ids)
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def invalidate_subtree(self, community_id: str, hierarchy: CommunityHierarchy) -> None:
        """Invalidate cache for a community and all its ancestors."""
        self.cache.invalidate(community_id)

        comm = hierarchy.get_community(community_id)
        if comm and comm.parent_id:
            self.invalidate_subtree(comm.parent_id, hierarchy)
