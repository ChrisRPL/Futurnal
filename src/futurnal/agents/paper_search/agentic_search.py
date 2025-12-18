"""Agentic Paper Search - LLM-powered intelligent paper discovery.

Research Foundation:
- PaSa (2501.10120): Crawler + Selector agent architecture
- Consensus: Planning Agent + Search Agent coordination
- Elicit: Semantic search without perfect keyword match

Architecture:
1. QueryAnalyzer: Parse user intent, extract concepts, suggest expansions
2. SearchPlanner: Generate multiple search strategies
3. RelevanceScorer: LLM evaluates title/abstract relevance to user need
4. ResultSynthesizer: Rank, filter, summarize, suggest refinements

This transforms simple API calls into an intelligent research assistant that:
- Tries multiple query formulations
- Evaluates if results actually match user's specific need
- Iterates until finding relevant papers or exhausting strategies
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import PaperMetadata, SearchResult
from .semantic_scholar import SemanticScholarClient
from .multi_provider import MultiProviderSearch, get_default_multi_provider_search

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class QueryAnalysis:
    """Result of analyzing user's query intent."""

    original_query: str
    core_concepts: List[str]  # Main topics to search for
    qualifiers: List[str]  # e.g., "SOTA", "recent", "survey", "benchmark"
    domain: Optional[str] = None  # e.g., "computer vision", "NLP"
    intent: str = "find_papers"  # find_papers, find_survey, find_sota, etc.
    suggested_expansions: List[str] = field(default_factory=list)


@dataclass
class SearchStrategy:
    """A single search strategy to try."""

    query: str
    strategy_type: str  # "exact", "expanded", "synonym", "broader", "narrower"
    priority: int = 1  # Lower = higher priority
    rationale: str = ""


@dataclass
class ScoredPaper:
    """A paper with relevance scoring."""

    paper: PaperMetadata
    relevance_score: float  # 0-1, how relevant to user's need
    title_match: float  # 0-1, how well title matches
    abstract_match: float  # 0-1, how well abstract matches
    rationale: str = ""  # Why this score
    from_strategy: str = ""  # Which strategy found this


@dataclass
class AgenticSearchResult:
    """Result of agentic paper search."""

    query: str
    papers: List[ScoredPaper]
    strategies_tried: List[SearchStrategy]
    total_papers_evaluated: int
    synthesis: str  # LLM-generated summary of findings
    suggestions: List[str]  # Follow-up search suggestions
    search_time_ms: int = 0


# =============================================================================
# Query Analyzer
# =============================================================================


class QueryAnalyzer:
    """Analyzes user query to understand intent and extract searchable concepts.

    Uses LLM to:
    1. Extract core concepts from natural language query
    2. Identify qualifiers (SOTA, recent, survey, etc.)
    3. Suggest query expansions (synonyms, related terms)
    """

    # Known qualifiers that affect search strategy
    QUALIFIERS = {
        "sota": ["state-of-the-art", "best", "top", "leading"],
        "recent": ["2024", "2023", "latest", "new"],
        "survey": ["review", "overview", "comprehensive"],
        "benchmark": ["evaluation", "comparison", "dataset"],
        "tutorial": ["introduction", "guide", "primer"],
    }

    # Domain-specific term expansions
    DOMAIN_EXPANSIONS = {
        "object detection": ["detection", "localization", "bounding box"],
        "zero-shot": ["zero shot", "0-shot", "without training"],
        "few-shot": ["few shot", "low-shot", "n-shot"],
        "transformer": ["attention", "self-attention", "BERT", "GPT"],
        "llm": ["large language model", "language model", "GPT", "LLaMA"],
        "rag": ["retrieval augmented generation", "retrieval-augmented"],
        "knowledge graph": ["KG", "knowledge base", "ontology"],
    }

    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize with optional LLM client for advanced analysis."""
        self.llm_client = llm_client

    async def analyze(self, query: str) -> QueryAnalysis:
        """Analyze user query and extract search components.

        Args:
            query: Natural language query from user

        Returns:
            QueryAnalysis with extracted concepts and suggestions
        """
        query_lower = query.lower()

        # Extract qualifiers
        found_qualifiers = []
        for qualifier, variants in self.QUALIFIERS.items():
            if qualifier in query_lower or any(v in query_lower for v in variants):
                found_qualifiers.append(qualifier)

        # Extract core concepts (remove qualifiers, clean up)
        core_query = query_lower
        for qualifier in found_qualifiers:
            core_query = core_query.replace(qualifier, "")
            for variant in self.QUALIFIERS.get(qualifier, []):
                core_query = core_query.replace(variant, "")

        # Clean up and extract concepts
        core_concepts = [
            c.strip()
            for c in core_query.split()
            if len(c.strip()) > 2 and c.strip() not in ["the", "and", "for", "about", "on", "in", "a", "an"]
        ]

        # Detect domain
        domain = None
        domain_keywords = {
            "computer vision": ["image", "vision", "visual", "detection", "segmentation", "object"],
            "nlp": ["language", "text", "nlp", "translation", "sentiment", "parsing"],
            "reinforcement learning": ["rl", "reinforcement", "agent", "reward", "policy"],
            "knowledge graph": ["knowledge graph", "kg", "ontology", "triple", "entity"],
        }
        for dom, keywords in domain_keywords.items():
            if any(k in query_lower for k in keywords):
                domain = dom
                break

        # Generate expansions based on detected concepts
        expansions = []
        for concept in core_concepts:
            for term, related in self.DOMAIN_EXPANSIONS.items():
                if concept in term or term in concept:
                    expansions.extend(related)

        # Determine intent
        intent = "find_papers"
        if "survey" in found_qualifiers:
            intent = "find_survey"
        elif "sota" in found_qualifiers or "best" in query_lower:
            intent = "find_sota"
        elif "benchmark" in found_qualifiers:
            intent = "find_benchmark"

        return QueryAnalysis(
            original_query=query,
            core_concepts=core_concepts,
            qualifiers=found_qualifiers,
            domain=domain,
            intent=intent,
            suggested_expansions=list(set(expansions))[:5],
        )


# =============================================================================
# Search Planner
# =============================================================================


class SearchPlanner:
    """Generates multiple search strategies based on query analysis.

    Creates a ranked list of search queries to try, including:
    - Exact query
    - Expanded with synonyms
    - Broader terms
    - Narrower/specific terms
    - Domain-specific variations
    """

    def generate_strategies(self, analysis: QueryAnalysis) -> List[SearchStrategy]:
        """Generate search strategies from query analysis.

        Args:
            analysis: QueryAnalysis from QueryAnalyzer

        Returns:
            List of SearchStrategy ordered by priority
        """
        strategies = []

        # Strategy 1: Clean exact query
        exact_query = " ".join(analysis.core_concepts)
        if exact_query:
            strategies.append(
                SearchStrategy(
                    query=exact_query,
                    strategy_type="exact",
                    priority=1,
                    rationale="Direct search for core concepts",
                )
            )

        # Strategy 2: Original query (may have context)
        if analysis.original_query != exact_query:
            strategies.append(
                SearchStrategy(
                    query=analysis.original_query,
                    strategy_type="original",
                    priority=2,
                    rationale="Original user phrasing",
                )
            )

        # Strategy 3: Add domain context if detected
        if analysis.domain and exact_query:
            strategies.append(
                SearchStrategy(
                    query=f"{exact_query} {analysis.domain}",
                    strategy_type="domain_contextualized",
                    priority=3,
                    rationale=f"Core concepts with domain context: {analysis.domain}",
                )
            )

        # Strategy 4: Qualifier-enhanced searches
        if "survey" in analysis.qualifiers and exact_query:
            strategies.append(
                SearchStrategy(
                    query=f"{exact_query} survey review",
                    strategy_type="survey_focused",
                    priority=2,
                    rationale="Searching for survey/review papers",
                )
            )

        if "sota" in analysis.qualifiers and exact_query:
            strategies.append(
                SearchStrategy(
                    query=f"{exact_query} state-of-the-art benchmark",
                    strategy_type="sota_focused",
                    priority=2,
                    rationale="Searching for SOTA/benchmark papers",
                )
            )

        # Strategy 5: Expansion-based searches
        for i, expansion in enumerate(analysis.suggested_expansions[:3]):
            if expansion not in exact_query:
                strategies.append(
                    SearchStrategy(
                        query=f"{exact_query} {expansion}",
                        strategy_type="expanded",
                        priority=4 + i,
                        rationale=f"Expanded with related term: {expansion}",
                    )
                )

        # Strategy 6: Broader search (fewer terms)
        if len(analysis.core_concepts) > 2:
            broader = " ".join(analysis.core_concepts[:2])
            strategies.append(
                SearchStrategy(
                    query=broader,
                    strategy_type="broader",
                    priority=5,
                    rationale="Broader search with fewer constraints",
                )
            )

        # Sort by priority
        strategies.sort(key=lambda s: s.priority)

        return strategies


# =============================================================================
# Relevance Scorer
# =============================================================================


class RelevanceScorer:
    """Scores paper relevance using local LLM analysis via Ollama.

    Evaluates how well a paper matches the user's specific need by:
    1. Having an LLM read the title and abstract
    2. Scoring semantic relevance to the query intent
    3. Providing a rationale for the score
    """

    # Prompt template for LLM scoring
    SCORING_PROMPT = """Score this paper's relevance to the research query.

QUERY: {query}
USER INTENT: {intent}

PAPER TITLE: {title}
ABSTRACT: {abstract}

Rate relevance from 0-10 where:
- 0-2: Not relevant
- 3-4: Tangentially related
- 5-6: Moderately relevant
- 7-8: Highly relevant
- 9-10: Perfect match

Respond in this exact format:
SCORE: <number>
REASON: <one sentence explanation>"""

    def __init__(self, use_llm: bool = True, model: str = "mistral:7b-instruct"):
        """Initialize relevance scorer.

        Args:
            use_llm: Whether to use LLM for scoring (falls back to keywords if False)
            model: Ollama model name for scoring
        """
        self.use_llm = use_llm
        self.model = model
        self._ollama_client = None

    def _get_ollama(self):
        """Lazy-load Ollama client."""
        if self._ollama_client is None:
            try:
                from futurnal.extraction.ollama_client import OllamaLLMClient, ollama_available
                if ollama_available():
                    self._ollama_client = OllamaLLMClient(model_name=self.model)
                    logger.info(f"LLM scoring enabled with {self.model}")
                else:
                    logger.warning("Ollama not available, falling back to keyword matching")
                    self.use_llm = False
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama: {e}")
                self.use_llm = False
        return self._ollama_client

    def _parse_llm_score(self, response: str) -> Tuple[float, str]:
        """Parse LLM response into score and rationale."""
        import re

        # Extract score
        score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', response)
        score = float(score_match.group(1)) / 10.0 if score_match else 0.5
        score = min(1.0, max(0.0, score))

        # Extract reason
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response)
        reason = reason_match.group(1).strip() if reason_match else "LLM evaluation"

        return score, reason

    async def score_paper(
        self,
        paper: PaperMetadata,
        query_analysis: QueryAnalysis,
    ) -> ScoredPaper:
        """Score a single paper's relevance to user query.

        Args:
            paper: Paper metadata to evaluate
            query_analysis: Analysis of user's query

        Returns:
            ScoredPaper with relevance scores
        """
        title_lower = paper.title.lower()
        abstract_lower = (paper.abstract or "").lower()

        # Try LLM scoring first
        if self.use_llm:
            ollama = self._get_ollama()
            if ollama:
                try:
                    prompt = self.SCORING_PROMPT.format(
                        query=query_analysis.original_query,
                        intent=query_analysis.intent,
                        title=paper.title,
                        abstract=(paper.abstract or "No abstract available")[:1000],
                    )

                    # Run sync Ollama in executor to not block
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: ollama.generate(prompt, max_tokens=100, temperature=0.1)
                    )

                    relevance_score, rationale = self._parse_llm_score(response)

                    # Calculate title/abstract match for metadata
                    concepts_in_title = sum(
                        1 for c in query_analysis.core_concepts if c.lower() in title_lower
                    )
                    title_match = min(1.0, concepts_in_title / max(1, len(query_analysis.core_concepts)))

                    concepts_in_abstract = sum(
                        1 for c in query_analysis.core_concepts if c.lower() in abstract_lower
                    )
                    abstract_match = min(1.0, concepts_in_abstract / max(1, len(query_analysis.core_concepts)))

                    return ScoredPaper(
                        paper=paper,
                        relevance_score=relevance_score,
                        title_match=title_match,
                        abstract_match=abstract_match,
                        rationale=rationale,
                    )
                except Exception as e:
                    logger.warning(f"LLM scoring failed for {paper.title[:30]}: {e}")
                    # Fall through to keyword matching

        # Fallback: keyword matching
        concepts_in_title = sum(
            1 for c in query_analysis.core_concepts if c.lower() in title_lower
        )
        title_match = min(1.0, concepts_in_title / max(1, len(query_analysis.core_concepts)))

        concepts_in_abstract = sum(
            1 for c in query_analysis.core_concepts if c.lower() in abstract_lower
        )
        abstract_match = min(1.0, concepts_in_abstract / max(1, len(query_analysis.core_concepts)))

        # Check qualifiers
        qualifier_bonus = 0.0
        if "survey" in query_analysis.qualifiers:
            if any(term in title_lower for term in ["survey", "review", "overview"]):
                qualifier_bonus += 0.2

        if "sota" in query_analysis.qualifiers:
            if any(term in title_lower or term in abstract_lower
                   for term in ["state-of-the-art", "sota", "benchmark", "outperform"]):
                qualifier_bonus += 0.2

        relevance_score = min(1.0, (
            title_match * 0.4 +
            abstract_match * 0.4 +
            qualifier_bonus +
            (0.1 if paper.citation_count and paper.citation_count > 50 else 0)
        ))

        rationale_parts = []
        if title_match > 0.5:
            rationale_parts.append(f"Title matches {int(title_match * 100)}%")
        if abstract_match > 0.5:
            rationale_parts.append(f"Abstract matches {int(abstract_match * 100)}%")
        if qualifier_bonus > 0:
            rationale_parts.append("Matches qualifiers")

        return ScoredPaper(
            paper=paper,
            relevance_score=relevance_score,
            title_match=title_match,
            abstract_match=abstract_match,
            rationale="; ".join(rationale_parts) if rationale_parts else "Low keyword match",
        )

    async def score_papers(
        self,
        papers: List[PaperMetadata],
        query_analysis: QueryAnalysis,
        max_concurrent: int = 5,
    ) -> List[ScoredPaper]:
        """Score multiple papers with controlled concurrency.

        Args:
            papers: List of papers to score
            query_analysis: Analysis of user's query
            max_concurrent: Max concurrent LLM calls

        Returns:
            List of ScoredPaper sorted by relevance
        """
        # Use semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_with_limit(paper):
            async with semaphore:
                return await self.score_paper(paper, query_analysis)

        tasks = [score_with_limit(paper) for paper in papers]
        scored = await asyncio.gather(*tasks)

        # Sort by relevance score descending
        scored.sort(key=lambda s: s.relevance_score, reverse=True)

        return scored


# =============================================================================
# Result Synthesizer
# =============================================================================


class ResultSynthesizer:
    """Synthesizes search results into actionable output.

    Generates:
    - Summary of findings
    - Filtered list of relevant papers
    - Suggestions for follow-up searches
    """

    RELEVANCE_THRESHOLD = 0.3  # Minimum score to include

    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize with optional LLM client."""
        self.llm_client = llm_client

    def synthesize(
        self,
        scored_papers: List[ScoredPaper],
        query_analysis: QueryAnalysis,
        strategies_tried: List[SearchStrategy],
    ) -> Tuple[str, List[str]]:
        """Synthesize results into summary and suggestions.

        Args:
            scored_papers: Papers with relevance scores
            query_analysis: Original query analysis
            strategies_tried: Search strategies that were executed

        Returns:
            Tuple of (synthesis text, list of suggestions)
        """
        # Filter to relevant papers
        relevant = [p for p in scored_papers if p.relevance_score >= self.RELEVANCE_THRESHOLD]

        # Generate synthesis
        if not relevant:
            synthesis = f"No highly relevant papers found for '{query_analysis.original_query}'. "
            synthesis += "The search tried multiple query formulations but results had low relevance scores."
        elif len(relevant) <= 3:
            synthesis = f"Found {len(relevant)} relevant paper(s) for '{query_analysis.original_query}'. "
            if relevant[0].relevance_score > 0.7:
                synthesis += f"Top match: '{relevant[0].paper.title}' appears highly relevant."
        else:
            synthesis = f"Found {len(relevant)} relevant papers for '{query_analysis.original_query}'. "
            top_authors = set()
            for p in relevant[:5]:
                if p.paper.authors:
                    top_authors.add(p.paper.author_names[0] if p.paper.author_names else "Unknown")
            if top_authors:
                synthesis += f"Notable authors: {', '.join(list(top_authors)[:3])}."

        # Generate suggestions based on results
        suggestions = []

        if not relevant:
            # No good results - suggest alternatives
            if query_analysis.suggested_expansions:
                suggestions.append(
                    f"Try searching with related terms: {', '.join(query_analysis.suggested_expansions[:3])}"
                )
            suggestions.append("Try a broader search with fewer specific terms")
            suggestions.append("Check if there are survey papers that cover this topic")

        elif len(relevant) < 5:
            # Few results - suggest expansions
            if query_analysis.domain:
                suggestions.append(f"Explore more papers in {query_analysis.domain}")
            suggestions.append("Check citations of the found papers for related work")

        else:
            # Good results - suggest refinements
            if "survey" not in query_analysis.qualifiers:
                suggestions.append("Search for survey papers to get a comprehensive overview")
            suggestions.append("Filter by year to find most recent work")
            suggestions.append("Look at highly-cited papers for foundational work")

        return synthesis, suggestions


# =============================================================================
# Agentic Paper Search Agent
# =============================================================================


class AgenticPaperSearchAgent:
    """Intelligent paper search agent that iteratively finds relevant papers.

    Orchestrates:
    1. Query analysis and understanding
    2. Multi-strategy search planning
    3. Parallel search execution
    4. LLM-based relevance scoring
    5. Result synthesis and suggestions

    Example:
        agent = AgenticPaperSearchAgent()
        result = await agent.search("zero shot object detection SOTA")
        print(f"Found {len(result.papers)} relevant papers")
        print(f"Synthesis: {result.synthesis}")
    """

    def __init__(
        self,
        search_client: Optional[Any] = None,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        llm_client: Optional[Any] = None,
        max_strategies: int = 5,
        max_papers_per_strategy: int = 10,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ):
        """Initialize agentic search agent.

        Args:
            search_client: Paper search client (MultiProviderSearch, SemanticScholarClient, etc.)
                          If not provided, uses MultiProviderSearch (OpenAlex + arXiv)
            semantic_scholar_client: Deprecated, use search_client instead
            llm_client: Optional LLM client for advanced analysis
            max_strategies: Maximum search strategies to try
            max_papers_per_strategy: Papers to fetch per strategy
            on_progress: Progress callback (message, progress 0-1)
        """
        # Support both new search_client and legacy semantic_scholar_client
        if search_client is not None:
            self.search_client = search_client
        elif semantic_scholar_client is not None:
            self.search_client = semantic_scholar_client
        else:
            # Default: Use multi-provider search (OpenAlex + arXiv - no rate limit issues)
            self.search_client = get_default_multi_provider_search()
        self.query_analyzer = QueryAnalyzer(llm_client)
        self.search_planner = SearchPlanner()
        self.relevance_scorer = RelevanceScorer(use_llm=True)
        self.result_synthesizer = ResultSynthesizer(llm_client)

        self.max_strategies = max_strategies
        self.max_papers_per_strategy = max_papers_per_strategy
        self.on_progress = on_progress

    def _report_progress(self, message: str, progress: float):
        """Report progress to callback."""
        if self.on_progress:
            self.on_progress(message, progress)
        logger.info(f"[Agentic Search {progress:.0%}] {message}")

    async def search(self, query: str) -> AgenticSearchResult:
        """Execute agentic paper search.

        Args:
            query: Natural language query from user

        Returns:
            AgenticSearchResult with scored papers and synthesis
        """
        import time
        start_time = time.time()

        # Step 1: Analyze query
        self._report_progress(f"Analyzing query: {query}", 0.1)
        analysis = await self.query_analyzer.analyze(query)
        logger.info(f"Query analysis: concepts={analysis.core_concepts}, qualifiers={analysis.qualifiers}")

        # Step 2: Generate search strategies
        self._report_progress("Planning search strategies", 0.2)
        strategies = self.search_planner.generate_strategies(analysis)
        strategies = strategies[: self.max_strategies]
        logger.info(f"Generated {len(strategies)} search strategies")

        # Step 3: Execute searches
        all_papers: Dict[str, PaperMetadata] = {}  # Dedupe by paper_id
        strategies_tried: List[SearchStrategy] = []

        for i, strategy in enumerate(strategies):
            progress = 0.2 + (0.4 * (i + 1) / len(strategies))
            self._report_progress(f"Searching: {strategy.query} ({strategy.strategy_type})", progress)

            try:
                result = await self.search_client.search(
                    query=strategy.query,
                    limit=self.max_papers_per_strategy,
                )

                for paper in result.papers:
                    if paper.paper_id not in all_papers:
                        all_papers[paper.paper_id] = paper

                strategies_tried.append(strategy)
                logger.info(f"Strategy '{strategy.strategy_type}' found {len(result.papers)} papers")

            except Exception as e:
                logger.warning(f"Search strategy failed: {e}")
                continue

        # Step 4: Score relevance
        self._report_progress(f"Scoring {len(all_papers)} papers for relevance", 0.7)
        scored_papers = await self.relevance_scorer.score_papers(
            list(all_papers.values()),
            analysis,
        )

        # Add strategy info to scored papers
        for sp in scored_papers:
            for strategy in strategies_tried:
                if strategy.query.lower() in sp.paper.title.lower():
                    sp.from_strategy = strategy.strategy_type
                    break

        # Step 5: Synthesize results
        self._report_progress("Synthesizing results", 0.9)
        synthesis, suggestions = self.result_synthesizer.synthesize(
            scored_papers,
            analysis,
            strategies_tried,
        )

        # Filter to relevant papers only
        relevant_papers = [
            p for p in scored_papers
            if p.relevance_score >= self.result_synthesizer.RELEVANCE_THRESHOLD
        ]

        search_time_ms = int((time.time() - start_time) * 1000)
        self._report_progress(f"Found {len(relevant_papers)} relevant papers", 1.0)

        return AgenticSearchResult(
            query=query,
            papers=relevant_papers,
            strategies_tried=strategies_tried,
            total_papers_evaluated=len(scored_papers),
            synthesis=synthesis,
            suggestions=suggestions,
            search_time_ms=search_time_ms,
        )
