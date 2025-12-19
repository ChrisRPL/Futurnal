"""
Personalized Deep Research with User Profiles.

Implements research agents that adapt to user preferences,
knowledge level, and interests for personalized research.

Research Foundation:
- Personalized Deep Research (2509.25106v1)
- User-centric knowledge graphs (PGraphRAG)
- Adaptive information retrieval

Key Features:
- User profile learning
- Interest modeling
- Adaptive depth/breadth
- Source preference learning

Option B Compliance:
- User profiles stored as natural language priors
- No model fine-tuning
- All personalization via retrieval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExpertiseLevel(str, Enum):
    """User expertise levels."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ResearchDepth(str, Enum):
    """Depth of research."""
    OVERVIEW = "overview"  # Quick summary
    STANDARD = "standard"  # Normal depth
    DETAILED = "detailed"  # In-depth
    EXHAUSTIVE = "exhaustive"  # Leave no stone unturned


@dataclass
class UserProfile:
    """Profile for personalized research."""
    user_id: str
    name: str = ""

    # Expertise areas
    expertise_areas: Dict[str, ExpertiseLevel] = field(default_factory=dict)
    interests: List[str] = field(default_factory=list)
    anti_interests: List[str] = field(default_factory=list)  # Topics to avoid

    # Preferences
    preferred_depth: ResearchDepth = ResearchDepth.STANDARD
    preferred_source_types: List[str] = field(default_factory=list)
    language_preference: str = "en"

    # Learning history
    queries_history: List[str] = field(default_factory=list)
    topics_explored: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Quality preferences
    prefer_academic: bool = False
    prefer_recent: bool = True
    max_source_age_days: Optional[int] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def get_expertise(self, topic: str) -> ExpertiseLevel:
        """Get expertise level for a topic."""
        # Direct match
        if topic.lower() in {k.lower(): v for k, v in self.expertise_areas.items()}:
            return self.expertise_areas.get(topic, ExpertiseLevel.NOVICE)

        # Partial match
        for area, level in self.expertise_areas.items():
            if area.lower() in topic.lower() or topic.lower() in area.lower():
                return level

        return ExpertiseLevel.NOVICE

    def is_interested(self, topic: str) -> bool:
        """Check if user is interested in a topic."""
        topic_lower = topic.lower()

        # Check anti-interests first
        for anti in self.anti_interests:
            if anti.lower() in topic_lower:
                return False

        # Check interests
        for interest in self.interests:
            if interest.lower() in topic_lower:
                return True

        return True  # Default to interested


@dataclass
class ResearchResult:
    """Result from personalized research."""
    query: str
    user_id: str

    # Findings
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    detailed_findings: List[Dict[str, Any]] = field(default_factory=list)

    # Sources
    sources: List[Dict[str, Any]] = field(default_factory=list)
    num_sources_consulted: int = 0

    # Personalization info
    expertise_level_used: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    depth_used: ResearchDepth = ResearchDepth.STANDARD

    # Quality
    confidence: float = 0.0
    relevance_score: float = 0.0

    # Metadata
    research_time_seconds: float = 0.0


class PersonalizedResearchAgent:
    """
    Agent that conducts personalized deep research.

    Adapts research strategy based on:
    - User expertise level
    - Topic interests
    - Source preferences
    - Historical queries
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        web_browser: Optional[Any] = None,
        knowledge_graph: Optional[Any] = None,
        vector_store: Optional[Any] = None
    ):
        """Initialize research agent.

        Args:
            llm_client: LLM for synthesis
            web_browser: Web browsing agent
            knowledge_graph: Neo4j driver
            vector_store: Vector database
        """
        self.llm_client = llm_client
        self.web_browser = web_browser
        self.kg = knowledge_graph
        self.vector_store = vector_store

        # User profiles
        self.profiles: Dict[str, UserProfile] = {}

    async def research(
        self,
        query: str,
        user_id: str,
        depth: Optional[ResearchDepth] = None
    ) -> ResearchResult:
        """Conduct personalized research.

        Args:
            query: Research query
            user_id: User identifier
            depth: Optional depth override

        Returns:
            ResearchResult with findings
        """
        import time
        start_time = time.time()

        # Get or create profile
        profile = self.get_or_create_profile(user_id)

        # Determine research parameters
        expertise = profile.get_expertise(query)
        research_depth = depth or profile.preferred_depth

        logger.info(
            f"Research for {user_id}: '{query[:50]}...' "
            f"(expertise: {expertise.value}, depth: {research_depth.value})"
        )

        # Gather information from multiple sources
        findings = []

        # 1. Search knowledge graph
        kg_results = await self._search_knowledge_graph(query, profile)
        findings.extend(kg_results)

        # 2. Search vector store
        vector_results = await self._search_vector_store(query, profile)
        findings.extend(vector_results)

        # 3. Web search if needed
        if research_depth in [ResearchDepth.DETAILED, ResearchDepth.EXHAUSTIVE]:
            web_results = await self._search_web(query, profile)
            findings.extend(web_results)

        # 4. Synthesize findings
        summary, key_points = await self._synthesize(
            query, findings, profile, expertise
        )

        # 5. Calculate quality metrics
        confidence = self._calculate_confidence(findings)
        relevance = self._calculate_relevance(findings, query)

        # Update profile
        self._update_profile(profile, query)

        return ResearchResult(
            query=query,
            user_id=user_id,
            summary=summary,
            key_points=key_points,
            detailed_findings=findings,
            sources=[f.get("source", {}) for f in findings if f.get("source")],
            num_sources_consulted=len(findings),
            expertise_level_used=expertise,
            depth_used=research_depth,
            confidence=confidence,
            relevance_score=relevance,
            research_time_seconds=time.time() - start_time,
        )

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create a user profile."""
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id)
        return self.profiles[user_id]

    def update_profile(
        self,
        user_id: str,
        expertise_areas: Optional[Dict[str, ExpertiseLevel]] = None,
        interests: Optional[List[str]] = None,
        **kwargs
    ):
        """Update user profile."""
        profile = self.get_or_create_profile(user_id)

        if expertise_areas:
            profile.expertise_areas.update(expertise_areas)
        if interests:
            profile.interests.extend(interests)

        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.updated_at = datetime.utcnow()

    async def _search_knowledge_graph(
        self,
        query: str,
        profile: UserProfile
    ) -> List[Dict[str, Any]]:
        """Search the knowledge graph."""
        findings = []

        if not self.kg:
            return findings

        # Build query based on user interests
        interest_filter = ""
        if profile.interests:
            topics = ", ".join([f"'{i}'" for i in profile.interests[:5]])
            interest_filter = f"AND (n.topic IN [{topics}] OR n.category IN [{topics}])"

        cypher = f"""
        MATCH (n)
        WHERE n.name CONTAINS $query OR n.description CONTAINS $query
        {interest_filter}
        RETURN n.name as name, n.description as desc, labels(n) as labels
        LIMIT 20
        """

        try:
            with self.kg.session() as session:
                result = session.run(cypher, query=query)
                for record in result:
                    findings.append({
                        "content": record["desc"] or record["name"],
                        "type": "knowledge_graph",
                        "source": {
                            "type": "kg",
                            "labels": record["labels"],
                        },
                        "relevance": 0.7,
                    })
        except Exception as e:
            logger.warning(f"KG search failed: {e}")

        return findings

    async def _search_vector_store(
        self,
        query: str,
        profile: UserProfile
    ) -> List[Dict[str, Any]]:
        """Search the vector store."""
        findings = []

        if not self.vector_store:
            return findings

        try:
            if hasattr(self.vector_store, "query"):
                results = self.vector_store.query(
                    query_texts=[query],
                    n_results=10
                )

                docs = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                for doc, meta, dist in zip(docs, metadatas, distances):
                    # Filter by user preferences
                    if not profile.is_interested(doc):
                        continue

                    findings.append({
                        "content": doc,
                        "type": "vector_search",
                        "source": meta,
                        "relevance": 1.0 / (1.0 + dist),
                    })
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

        return findings

    async def _search_web(
        self,
        query: str,
        profile: UserProfile
    ) -> List[Dict[str, Any]]:
        """Search the web."""
        findings = []

        if not self.web_browser:
            return findings

        try:
            # Adapt query for expertise level
            expertise = profile.get_expertise(query)
            adapted_query = self._adapt_query_for_expertise(query, expertise)

            if hasattr(self.web_browser, "browse"):
                result = await self.web_browser.browse(adapted_query)

                for finding in result.findings:
                    findings.append({
                        "content": finding.get("fact", ""),
                        "type": "web_search",
                        "source": {
                            "url": finding.get("source_url"),
                            "title": finding.get("source_title"),
                        },
                        "relevance": 0.6,
                    })
        except Exception as e:
            logger.warning(f"Web search failed: {e}")

        return findings

    def _adapt_query_for_expertise(
        self,
        query: str,
        expertise: ExpertiseLevel
    ) -> str:
        """Adapt query based on expertise level."""
        if expertise == ExpertiseLevel.NOVICE:
            return f"{query} beginner introduction basics"
        elif expertise == ExpertiseLevel.INTERMEDIATE:
            return f"{query} guide tutorial"
        elif expertise == ExpertiseLevel.ADVANCED:
            return f"{query} advanced techniques best practices"
        else:  # EXPERT
            return f"{query} research papers state of the art"

    async def _synthesize(
        self,
        query: str,
        findings: List[Dict[str, Any]],
        profile: UserProfile,
        expertise: ExpertiseLevel
    ) -> tuple[str, List[str]]:
        """Synthesize findings into summary and key points."""
        if not findings:
            return "No relevant information found.", []

        if not self.llm_client:
            # Simple aggregation without LLM
            contents = [f["content"] for f in findings[:5]]
            return "\n".join(contents), contents[:3]

        # Build prompt based on expertise
        complexity = {
            ExpertiseLevel.NOVICE: "simple, beginner-friendly",
            ExpertiseLevel.INTERMEDIATE: "clear and informative",
            ExpertiseLevel.ADVANCED: "detailed and technical",
            ExpertiseLevel.EXPERT: "highly technical with nuances",
        }

        findings_text = "\n".join([
            f"- {f['content'][:200]}..."
            for f in findings[:10]
        ])

        prompt = f"""Synthesize these findings into a {complexity[expertise]} summary:

Query: {query}

Findings:
{findings_text}

Provide:
1. A concise summary (2-3 paragraphs)
2. 5 key points

Format:
SUMMARY:
<summary>

KEY POINTS:
- point 1
- point 2
..."""

        try:
            if hasattr(self.llm_client, "generate"):
                response = await self.llm_client.generate(prompt)

                # Parse response
                summary = ""
                key_points = []

                if "SUMMARY:" in response:
                    parts = response.split("KEY POINTS:")
                    summary = parts[0].replace("SUMMARY:", "").strip()
                    if len(parts) > 1:
                        key_points = [
                            p.strip().lstrip("- ")
                            for p in parts[1].strip().split("\n")
                            if p.strip()
                        ]

                return summary or response, key_points
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")

        return findings[0]["content"], []

    def _calculate_confidence(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on findings."""
        if not findings:
            return 0.0

        # Base on number and diversity of sources
        num_findings = min(len(findings), 10)
        source_types = len(set(f["type"] for f in findings))

        confidence = 0.3 + (num_findings / 10) * 0.4 + (source_types / 3) * 0.3
        return min(1.0, confidence)

    def _calculate_relevance(
        self,
        findings: List[Dict[str, Any]],
        query: str
    ) -> float:
        """Calculate average relevance of findings."""
        if not findings:
            return 0.0

        return sum(f.get("relevance", 0.5) for f in findings) / len(findings)

    def _update_profile(self, profile: UserProfile, query: str):
        """Update profile with new query."""
        profile.queries_history.append(query)
        if len(profile.queries_history) > 100:
            profile.queries_history = profile.queries_history[-100:]

        # Extract topics from query
        words = query.lower().split()
        for word in words:
            if len(word) > 3:
                profile.topics_explored[word] += 1

        profile.updated_at = datetime.utcnow()
