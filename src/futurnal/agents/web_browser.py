"""
WebDancer-Style Web Browsing Agent.

Implements an end-to-end information-seeking agent that can:
- Navigate the web autonomously
- Extract relevant information
- Synthesize findings into coherent answers

Research Foundation:
- WebDancer (2505.22648v3): End-to-end web agents
- WebVoyager: Vision-language web navigation
- SeeAct: Visual understanding for web interactions

Key Features:
- URL navigation and content extraction
- Query-driven information seeking
- Multi-page synthesis
- Source attribution

Option B Compliance:
- No fine-tuning of browsing model
- LLM used for planning and synthesis only
- All extracted content is local
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from urllib.parse import urlparse, urljoin
import asyncio

logger = logging.getLogger(__name__)


class BrowsingAction(str, Enum):
    """Actions the web agent can take."""
    NAVIGATE = "navigate"  # Go to URL
    SEARCH = "search"  # Search query
    EXTRACT = "extract"  # Extract information
    CLICK = "click"  # Click element (simulated)
    SCROLL = "scroll"  # Scroll page (simulated)
    BACK = "back"  # Go back
    SUMMARIZE = "summarize"  # Summarize current page
    COMPLETE = "complete"  # Task complete


class SourceReliability(str, Enum):
    """Reliability levels for sources."""
    HIGH = "high"  # Academic, official sources
    MEDIUM = "medium"  # News, reputable blogs
    LOW = "low"  # Forums, unknown sites
    UNKNOWN = "unknown"


@dataclass
class WebPage:
    """Represents a web page."""
    url: str
    title: str = ""
    content: str = ""
    links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Extraction results
    extracted_facts: List[str] = field(default_factory=list)
    relevance_score: float = 0.0

    # Source info
    reliability: SourceReliability = SourceReliability.UNKNOWN
    fetch_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BrowsingState:
    """Current state of the browsing agent."""
    query: str
    current_url: Optional[str] = None
    current_page: Optional[WebPage] = None
    history: List[WebPage] = field(default_factory=list)

    # Extracted information
    findings: List[Dict[str, Any]] = field(default_factory=list)
    sources_visited: List[str] = field(default_factory=list)

    # Progress
    steps_taken: int = 0
    max_steps: int = 20
    is_complete: bool = False

    # Error tracking
    errors: List[str] = field(default_factory=list)


@dataclass
class BrowsingResult:
    """Final result from web browsing agent."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]

    # Quality metrics
    confidence: float = 0.0
    coverage: float = 0.0  # How well query was covered
    num_sources: int = 0

    # Metadata
    total_steps: int = 0
    total_pages: int = 0
    duration_seconds: float = 0.0

    # Raw findings
    findings: List[Dict[str, Any]] = field(default_factory=list)


class WebBrowsingAgent:
    """
    WebDancer-style agent for autonomous web information seeking.

    Implements a planning-execution loop:
    1. Plan: Decide next action based on query and state
    2. Execute: Take the action (navigate, extract, etc.)
    3. Update: Update state with results
    4. Check: Determine if goal is met
    """

    # Domain reliability mappings
    RELIABLE_DOMAINS = {
        "wikipedia.org": SourceReliability.HIGH,
        "arxiv.org": SourceReliability.HIGH,
        "github.com": SourceReliability.MEDIUM,
        "medium.com": SourceReliability.MEDIUM,
        "stackoverflow.com": SourceReliability.MEDIUM,
        ".gov": SourceReliability.HIGH,
        ".edu": SourceReliability.HIGH,
    }

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        web_fetcher: Optional[Callable] = None,
        search_engine: Optional[Callable] = None,
        max_steps: int = 20,
        max_pages: int = 10
    ):
        """Initialize web browsing agent.

        Args:
            llm_client: LLM for planning and synthesis
            web_fetcher: Function to fetch web pages
            search_engine: Function to search the web
            max_steps: Maximum browsing steps
            max_pages: Maximum pages to visit
        """
        self.llm_client = llm_client
        self.web_fetcher = web_fetcher
        self.search_engine = search_engine
        self.max_steps = max_steps
        self.max_pages = max_pages

        # Statistics
        self.stats = {
            "total_queries": 0,
            "total_pages_fetched": 0,
            "avg_steps_per_query": 0,
        }

    async def browse(
        self,
        query: str,
        context: Optional[str] = None
    ) -> BrowsingResult:
        """Execute web browsing to answer a query.

        Args:
            query: The query to answer
            context: Optional context about what to look for

        Returns:
            BrowsingResult with synthesized answer
        """
        import time
        start_time = time.time()

        self.stats["total_queries"] += 1

        # Initialize state
        state = BrowsingState(
            query=query,
            max_steps=self.max_steps,
        )

        logger.info(f"Starting web browse for: {query}")

        # Main browsing loop
        while not state.is_complete and state.steps_taken < state.max_steps:
            # Plan next action
            action, params = await self._plan_action(state, context)

            logger.debug(f"Step {state.steps_taken + 1}: {action.value} - {params}")

            # Execute action
            await self._execute_action(state, action, params)

            state.steps_taken += 1

            # Check if we have enough information
            if await self._should_complete(state):
                state.is_complete = True

        # Synthesize final answer
        answer = await self._synthesize_answer(state, context)

        duration = time.time() - start_time

        # Build sources list
        sources = [
            {
                "url": page.url,
                "title": page.title,
                "reliability": page.reliability.value,
                "relevance": page.relevance_score,
            }
            for page in state.history
            if page.relevance_score > 0.3
        ]

        # Calculate metrics
        confidence = self._calculate_confidence(state)
        coverage = self._calculate_coverage(state, query)

        return BrowsingResult(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
            coverage=coverage,
            num_sources=len(sources),
            total_steps=state.steps_taken,
            total_pages=len(state.history),
            duration_seconds=duration,
            findings=state.findings,
        )

    def browse_sync(
        self,
        query: str,
        context: Optional[str] = None
    ) -> BrowsingResult:
        """Synchronous version of browse."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.browse(query, context))

    async def _plan_action(
        self,
        state: BrowsingState,
        context: Optional[str]
    ) -> tuple[BrowsingAction, Dict[str, Any]]:
        """Plan the next browsing action."""

        # If no current page, start with search
        if state.current_url is None and len(state.history) == 0:
            return BrowsingAction.SEARCH, {"query": state.query}

        # If we have enough findings, complete
        if len(state.findings) >= 5:
            return BrowsingAction.COMPLETE, {}

        # Use LLM for planning if available
        if self.llm_client:
            action, params = await self._llm_plan(state, context)
            return action, params

        # Rule-based planning
        return self._rule_based_plan(state)

    async def _llm_plan(
        self,
        state: BrowsingState,
        context: Optional[str]
    ) -> tuple[BrowsingAction, Dict[str, Any]]:
        """Use LLM for action planning."""
        prompt = f"""You are a web browsing agent. Plan the next action.

Query: {state.query}
{f'Context: {context}' if context else ''}

Current state:
- Current URL: {state.current_url or 'None'}
- Pages visited: {len(state.history)}
- Findings so far: {len(state.findings)}
- Steps taken: {state.steps_taken}/{state.max_steps}

Recent findings:
{self._format_findings(state.findings[-3:]) if state.findings else 'None yet'}

Available actions:
- SEARCH: Search for more information (query: str)
- NAVIGATE: Go to a specific URL (url: str)
- EXTRACT: Extract information from current page
- SUMMARIZE: Summarize current page
- COMPLETE: Finish browsing (if enough info gathered)

Choose the best action and parameters. Format:
ACTION: <action_name>
PARAMS: <json_params>
"""

        try:
            if hasattr(self.llm_client, "generate"):
                response = await self.llm_client.generate(prompt)
            else:
                return self._rule_based_plan(state)

            # Parse response
            action_match = re.search(r"ACTION:\s*(\w+)", response, re.IGNORECASE)
            params_match = re.search(r"PARAMS:\s*({.*?}|\w+)", response, re.IGNORECASE)

            if action_match:
                action_str = action_match.group(1).upper()
                try:
                    action = BrowsingAction(action_str.lower())
                except ValueError:
                    action = BrowsingAction.SEARCH

                params = {}
                if params_match:
                    try:
                        import json
                        params = json.loads(params_match.group(1))
                    except json.JSONDecodeError:
                        params = {"query": state.query}

                return action, params

        except Exception as e:
            logger.warning(f"LLM planning failed: {e}")

        return self._rule_based_plan(state)

    def _rule_based_plan(
        self,
        state: BrowsingState
    ) -> tuple[BrowsingAction, Dict[str, Any]]:
        """Rule-based action planning."""

        # No pages visited - search
        if not state.history:
            return BrowsingAction.SEARCH, {"query": state.query}

        # Current page not extracted - extract
        if state.current_page and not state.current_page.extracted_facts:
            return BrowsingAction.EXTRACT, {}

        # Not enough findings and have links - navigate
        if len(state.findings) < 5 and state.current_page and state.current_page.links:
            relevant_links = [
                link for link in state.current_page.links
                if self._is_relevant_link(link, state.query) and link not in state.sources_visited
            ]
            if relevant_links:
                return BrowsingAction.NAVIGATE, {"url": relevant_links[0]}

        # Enough findings - complete
        if len(state.findings) >= 3:
            return BrowsingAction.COMPLETE, {}

        # Default - search with refined query
        return BrowsingAction.SEARCH, {"query": f"{state.query} details"}

    async def _execute_action(
        self,
        state: BrowsingState,
        action: BrowsingAction,
        params: Dict[str, Any]
    ):
        """Execute a browsing action."""

        if action == BrowsingAction.SEARCH:
            await self._action_search(state, params.get("query", state.query))

        elif action == BrowsingAction.NAVIGATE:
            await self._action_navigate(state, params.get("url", ""))

        elif action == BrowsingAction.EXTRACT:
            await self._action_extract(state)

        elif action == BrowsingAction.SUMMARIZE:
            await self._action_summarize(state)

        elif action == BrowsingAction.COMPLETE:
            state.is_complete = True

        elif action == BrowsingAction.BACK:
            if len(state.history) > 1:
                state.current_page = state.history[-2]
                state.current_url = state.current_page.url

    async def _action_search(self, state: BrowsingState, query: str):
        """Execute a search."""
        if self.search_engine:
            try:
                results = await self.search_engine(query)
                if results:
                    # Navigate to first result
                    first_url = results[0].get("url", results[0].get("link", ""))
                    if first_url:
                        await self._action_navigate(state, first_url)
            except Exception as e:
                state.errors.append(f"Search failed: {e}")
        else:
            # Mock search results for testing
            logger.warning("No search engine configured - using mock results")

    async def _action_navigate(self, state: BrowsingState, url: str):
        """Navigate to a URL."""
        if not url:
            return

        if len(state.sources_visited) >= self.max_pages:
            return

        if url in state.sources_visited:
            return

        state.sources_visited.append(url)

        # Fetch page
        page = await self._fetch_page(url)
        if page:
            state.current_page = page
            state.current_url = url
            state.history.append(page)
            self.stats["total_pages_fetched"] += 1

    async def _action_extract(self, state: BrowsingState):
        """Extract information from current page."""
        if not state.current_page:
            return

        page = state.current_page

        # Use LLM for extraction if available
        if self.llm_client:
            facts = await self._llm_extract(page.content, state.query)
            page.extracted_facts = facts

            for fact in facts:
                state.findings.append({
                    "fact": fact,
                    "source_url": page.url,
                    "source_title": page.title,
                    "reliability": page.reliability.value,
                })
        else:
            # Simple keyword extraction
            keywords = state.query.lower().split()
            sentences = page.content.split(".")
            relevant = [
                s.strip() for s in sentences
                if any(kw in s.lower() for kw in keywords)
            ][:5]

            page.extracted_facts = relevant
            for fact in relevant:
                state.findings.append({
                    "fact": fact,
                    "source_url": page.url,
                    "source_title": page.title,
                    "reliability": page.reliability.value,
                })

        # Calculate relevance
        page.relevance_score = self._calculate_page_relevance(page, state.query)

    async def _action_summarize(self, state: BrowsingState):
        """Summarize current page."""
        if not state.current_page or not self.llm_client:
            return

        page = state.current_page

        prompt = f"""Summarize the following web page content in relation to the query: "{state.query}"

Title: {page.title}
Content: {page.content[:3000]}

Provide a 2-3 sentence summary of relevant information."""

        try:
            if hasattr(self.llm_client, "generate"):
                summary = await self.llm_client.generate(prompt)
                state.findings.append({
                    "fact": f"Summary: {summary}",
                    "source_url": page.url,
                    "source_title": page.title,
                    "reliability": page.reliability.value,
                })
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")

    async def _fetch_page(self, url: str) -> Optional[WebPage]:
        """Fetch a web page."""
        if self.web_fetcher:
            try:
                content = await self.web_fetcher(url)
                return self._parse_page(url, content)
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                return None
        else:
            # Mock page for testing
            return WebPage(
                url=url,
                title=f"Page at {urlparse(url).netloc}",
                content="Mock content for testing.",
                reliability=self._assess_reliability(url),
            )

    def _parse_page(self, url: str, content: str) -> WebPage:
        """Parse raw content into WebPage."""
        # Simple HTML parsing - in production use BeautifulSoup
        title_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE)
        title = title_match.group(1) if title_match else urlparse(url).netloc

        # Extract text content (simplified)
        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()

        # Extract links
        links = re.findall(r'href=["\']([^"\']+)["\']', content)
        links = [urljoin(url, link) for link in links if link.startswith(("http", "/"))]

        return WebPage(
            url=url,
            title=title,
            content=text[:10000],  # Limit content length
            links=links[:20],
            reliability=self._assess_reliability(url),
        )

    def _assess_reliability(self, url: str) -> SourceReliability:
        """Assess source reliability based on domain."""
        domain = urlparse(url).netloc.lower()

        for pattern, reliability in self.RELIABLE_DOMAINS.items():
            if pattern in domain or domain.endswith(pattern):
                return reliability

        return SourceReliability.UNKNOWN

    async def _llm_extract(self, content: str, query: str) -> List[str]:
        """Extract relevant facts using LLM."""
        prompt = f"""Extract key facts from the following content that answer or relate to: "{query}"

Content:
{content[:4000]}

List up to 5 relevant facts, one per line. Just facts, no explanations."""

        try:
            if hasattr(self.llm_client, "generate"):
                response = await self.llm_client.generate(prompt)
                lines = response.strip().split("\n")
                return [line.strip().lstrip("- •*") for line in lines if line.strip()]
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")

        return []

    async def _should_complete(self, state: BrowsingState) -> bool:
        """Determine if browsing should complete."""
        # Hard limits
        if state.steps_taken >= state.max_steps:
            return True

        if len(state.sources_visited) >= self.max_pages:
            return True

        # Soft limits - enough information
        if len(state.findings) >= 5:
            return True

        # High quality findings
        high_quality = [
            f for f in state.findings
            if f.get("reliability") in ["high", "medium"]
        ]
        if len(high_quality) >= 3:
            return True

        return False

    async def _synthesize_answer(
        self,
        state: BrowsingState,
        context: Optional[str]
    ) -> str:
        """Synthesize final answer from findings."""
        if not state.findings:
            return "I couldn't find relevant information to answer this query."

        if self.llm_client:
            findings_text = self._format_findings(state.findings)

            prompt = f"""Synthesize an answer to the following query based on the gathered information.

Query: {state.query}
{f'Context: {context}' if context else ''}

Information gathered:
{findings_text}

Provide a comprehensive answer that:
1. Directly addresses the query
2. Cites sources where appropriate
3. Notes any uncertainty or conflicting information
"""

            try:
                if hasattr(self.llm_client, "generate"):
                    answer = await self.llm_client.generate(prompt)
                    return answer
            except Exception as e:
                logger.warning(f"Synthesis failed: {e}")

        # Fallback: concatenate findings
        return "Based on web research:\n" + "\n".join(
            f"- {f['fact']} (Source: {f['source_title']})"
            for f in state.findings[:5]
        )

    def _format_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Format findings for display."""
        lines = []
        for f in findings:
            lines.append(f"- {f['fact']}")
            lines.append(f"  Source: {f.get('source_title', f.get('source_url', 'Unknown'))}")
        return "\n".join(lines)

    def _calculate_page_relevance(self, page: WebPage, query: str) -> float:
        """Calculate page relevance to query."""
        query_terms = set(query.lower().split())
        content_terms = set(page.content.lower().split())

        # Term overlap
        overlap = len(query_terms & content_terms)
        if not query_terms:
            return 0.0

        return min(1.0, overlap / len(query_terms))

    def _calculate_confidence(self, state: BrowsingState) -> float:
        """Calculate confidence in the answer."""
        if not state.findings:
            return 0.0

        # Base on source reliability and finding count
        reliability_scores = {
            "high": 1.0,
            "medium": 0.7,
            "low": 0.4,
            "unknown": 0.3,
        }

        total_score = sum(
            reliability_scores.get(f.get("reliability", "unknown"), 0.3)
            for f in state.findings
        )

        return min(1.0, total_score / 5.0)

    def _calculate_coverage(self, state: BrowsingState, query: str) -> float:
        """Calculate how well the query was covered."""
        if not state.findings:
            return 0.0

        query_terms = set(query.lower().split())
        covered_terms = set()

        for finding in state.findings:
            fact = finding.get("fact", "").lower()
            for term in query_terms:
                if term in fact:
                    covered_terms.add(term)

        return len(covered_terms) / len(query_terms) if query_terms else 0.0

    def _is_relevant_link(self, link: str, query: str) -> bool:
        """Check if a link is likely relevant to the query."""
        query_terms = set(query.lower().split())
        link_lower = link.lower()

        # Check if any query term appears in URL
        for term in query_terms:
            if term in link_lower:
                return True

        return False
