"""Semantic Context Gate for Query-Aware Prior Filtering.

Filters token priors by relevance to the current query before injection,
preventing irrelevant priors from polluting the context.

Research Foundation:
- Training-Free GRPO (2510.08191v1): Selective prior injection based on
  semantic relevance rather than just confidence thresholds
- RLHI: Context-relevant knowledge retrieval

Option B Compliance:
- Ghost model frozen (no fine-tuning)
- Uses lightweight embedding similarity for filtering
- All priors remain as natural language text
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from futurnal.learning.token_priors import (
        EntityTypePrior,
        RelationTypePrior,
        TemporalPatternPrior,
    )

logger = logging.getLogger(__name__)


# Default thresholds
DEFAULT_MIN_RELEVANCE = 0.3
DEFAULT_TOP_K_PRIORS = 5
DEFAULT_COHERENCE_THRESHOLD = 0.4


@dataclass
class CoherenceReport:
    """Report on coherence between query, priors, and results.

    Captures quality metrics about how well the retrieved results
    align with the applied priors and original query.
    """

    is_coherent: bool = False
    coherence_score: float = 0.0
    query_result_alignment: float = 0.0
    prior_result_alignment: float = 0.0
    conflicting_results: List[str] = field(default_factory=list)
    missing_expected_topics: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    computed_at: datetime = field(default_factory=datetime.utcnow)

    def to_natural_language(self) -> str:
        """Convert to natural language summary."""
        if self.is_coherent:
            return (
                f"Results are coherent (score: {self.coherence_score:.2f}). "
                f"Query alignment: {self.query_result_alignment:.2f}, "
                f"Prior alignment: {self.prior_result_alignment:.2f}."
            )
        else:
            issues = []
            if self.conflicting_results:
                issues.append(f"{len(self.conflicting_results)} conflicting results")
            if self.missing_expected_topics:
                issues.append(f"missing topics: {', '.join(self.missing_expected_topics[:3])}")
            return (
                f"Results have coherence issues (score: {self.coherence_score:.2f}). "
                f"Issues: {'; '.join(issues) if issues else 'low alignment'}."
            )


@dataclass
class PriorRelevanceScore:
    """Relevance score for a prior with explanation."""

    prior_name: str
    prior_type: str  # "entity", "relation", "temporal"
    relevance_score: float
    matching_terms: List[str] = field(default_factory=list)
    explanation: str = ""


class SemanticContextGate:
    """Query-aware prior filtering with coherence checking.

    Ensures token priors are relevant to the current query before
    injection into prompts, preventing context pollution.

    Research Foundation:
    - Training-Free GRPO: Selective prior injection
    - RLHI: Context-relevant knowledge retrieval

    Example:
        >>> gate = SemanticContextGate()
        >>> relevant = gate.filter_relevant_priors(
        ...     query="What meetings did I have last week?",
        ...     priors=token_store.entity_priors,
        ...     top_k=5,
        ... )
        >>> # Only priors relevant to "meetings" and "temporal" are returned
    """

    def __init__(
        self,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
        top_k: int = DEFAULT_TOP_K_PRIORS,
        coherence_threshold: float = DEFAULT_COHERENCE_THRESHOLD,
        use_embedding_similarity: bool = False,
        embedding_model: Optional[Any] = None,
    ):
        """Initialize the semantic context gate.

        Args:
            min_relevance: Minimum relevance score for prior inclusion
            top_k: Maximum priors per category to include
            coherence_threshold: Threshold for coherence checking
            use_embedding_similarity: Use embedding-based similarity (requires model)
            embedding_model: Optional embedding model for similarity computation
        """
        self.min_relevance = min_relevance
        self.top_k = top_k
        self.coherence_threshold = coherence_threshold
        self.use_embedding_similarity = use_embedding_similarity
        self._embedding_model = embedding_model

        # Build keyword sets for different query intents
        self._temporal_keywords = {
            "when", "yesterday", "today", "tomorrow", "last", "next",
            "week", "month", "year", "ago", "before", "after", "during",
            "time", "date", "schedule", "calendar", "meeting", "event",
        }
        self._causal_keywords = {
            "why", "because", "cause", "effect", "result", "led", "reason",
            "consequence", "impact", "influence", "trigger", "due",
        }
        self._entity_keywords = {
            "who", "what", "which", "person", "project", "company",
            "document", "file", "note", "topic", "concept",
        }

    def compute_query_prior_relevance(
        self,
        query: str,
        prior: Union["EntityTypePrior", "RelationTypePrior", "TemporalPatternPrior"],
    ) -> PriorRelevanceScore:
        """Compute semantic relevance between query and a prior.

        Uses term overlap, intent matching, and optional embedding similarity
        to determine how relevant a prior is to the current query.

        Args:
            query: User's natural language query
            prior: Token prior to evaluate

        Returns:
            PriorRelevanceScore with relevance score and explanation
        """
        query_lower = query.lower()
        query_terms = set(self._tokenize(query_lower))

        # Get prior information
        prior_type = self._get_prior_type(prior)
        prior_name = self._get_prior_name(prior)
        prior_text = self._get_prior_searchable_text(prior)
        prior_terms = set(self._tokenize(prior_text.lower()))

        # Calculate term overlap
        common_terms = query_terms & prior_terms
        term_overlap = len(common_terms) / max(len(query_terms), 1)

        # Calculate intent alignment
        intent_score = self._calculate_intent_alignment(query_lower, prior_type, prior)

        # Calculate name relevance (is prior name mentioned or related?)
        name_relevance = self._calculate_name_relevance(query_lower, prior_name)

        # Combined score (weighted average)
        relevance_score = (
            0.3 * term_overlap +
            0.4 * intent_score +
            0.3 * name_relevance
        )

        # Use embedding similarity if available
        if self.use_embedding_similarity and self._embedding_model:
            embedding_sim = self._compute_embedding_similarity(query, prior_text)
            relevance_score = 0.5 * relevance_score + 0.5 * embedding_sim

        # Build explanation
        explanation_parts = []
        if term_overlap > 0.1:
            explanation_parts.append(f"term overlap ({term_overlap:.2f})")
        if intent_score > 0.3:
            explanation_parts.append(f"intent match ({intent_score:.2f})")
        if name_relevance > 0.3:
            explanation_parts.append(f"name relevance ({name_relevance:.2f})")

        explanation = ", ".join(explanation_parts) if explanation_parts else "low relevance"

        return PriorRelevanceScore(
            prior_name=prior_name,
            prior_type=prior_type,
            relevance_score=relevance_score,
            matching_terms=list(common_terms)[:5],
            explanation=explanation,
        )

    def filter_relevant_priors(
        self,
        query: str,
        priors: Dict[str, Any],
        top_k: Optional[int] = None,
        min_relevance: Optional[float] = None,
    ) -> List[Tuple[Any, float]]:
        """Filter priors to only those relevant to the query.

        Returns list of (prior, relevance_score) tuples sorted by relevance.

        Args:
            query: User's natural language query
            priors: Dictionary of priors (keyed by name)
            top_k: Maximum priors to return (default: self.top_k)
            min_relevance: Minimum relevance threshold (default: self.min_relevance)

        Returns:
            List of (prior, relevance_score) tuples, most relevant first
        """
        if not query or not priors:
            return []

        top_k = top_k or self.top_k
        min_relevance = min_relevance or self.min_relevance

        # Score all priors
        scored_priors: List[Tuple[Any, float]] = []
        for name, prior in priors.items():
            score = self.compute_query_prior_relevance(query, prior)
            if score.relevance_score >= min_relevance:
                scored_priors.append((prior, score.relevance_score))

        # Sort by relevance descending
        scored_priors.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        result = scored_priors[:top_k]

        logger.debug(
            f"Filtered priors: {len(result)}/{len(priors)} passed relevance gate "
            f"(threshold={min_relevance})"
        )

        return result

    def filter_all_prior_categories(
        self,
        query: str,
        entity_priors: Dict[str, Any],
        relation_priors: Dict[str, Any],
        temporal_priors: Dict[str, Any],
        top_k_per_category: Optional[int] = None,
        min_relevance: Optional[float] = None,
    ) -> Dict[str, List[Tuple[Any, float]]]:
        """Filter all prior categories by query relevance.

        Args:
            query: User's natural language query
            entity_priors: Entity type priors
            relation_priors: Relation type priors
            temporal_priors: Temporal pattern priors
            top_k_per_category: Max priors per category
            min_relevance: Minimum relevance threshold

        Returns:
            Dictionary with filtered priors per category
        """
        return {
            "entity": self.filter_relevant_priors(
                query, entity_priors, top_k_per_category, min_relevance
            ),
            "relation": self.filter_relevant_priors(
                query, relation_priors, top_k_per_category, min_relevance
            ),
            "temporal": self.filter_relevant_priors(
                query, temporal_priors, top_k_per_category, min_relevance
            ),
        }

    def check_result_coherence(
        self,
        query: str,
        results: List[Any],
        applied_priors: List[Any],
    ) -> CoherenceReport:
        """Check if results are coherent with query and applied priors.

        Identifies conflicts and gaps in retrieved results relative
        to what the priors suggest should be found.

        Args:
            query: Original user query
            results: Retrieved search results
            applied_priors: Priors that were applied during retrieval

        Returns:
            CoherenceReport with alignment metrics and recommendations
        """
        if not results:
            return CoherenceReport(
                is_coherent=False,
                coherence_score=0.0,
                recommendations=["No results returned - try broader query"],
            )

        query_lower = query.lower()
        query_terms = set(self._tokenize(query_lower))

        # Analyze query-result alignment
        result_terms: Set[str] = set()
        for r in results:
            content = self._get_result_content(r)
            result_terms.update(self._tokenize(content.lower()))

        query_result_overlap = len(query_terms & result_terms) / max(len(query_terms), 1)

        # Analyze prior-result alignment
        prior_expected_terms: Set[str] = set()
        for prior in applied_priors:
            prior_text = self._get_prior_searchable_text(prior)
            prior_expected_terms.update(self._tokenize(prior_text.lower()))

        prior_result_overlap = (
            len(prior_expected_terms & result_terms) / max(len(prior_expected_terms), 1)
            if prior_expected_terms else 1.0
        )

        # Calculate coherence score
        coherence_score = (query_result_overlap + prior_result_overlap) / 2

        # Identify missing topics
        missing_topics = []
        for term in query_terms:
            if len(term) > 3 and term not in result_terms:
                # Check if any result contains a related term
                found_related = any(
                    term in content.lower()
                    for r in results
                    for content in [self._get_result_content(r)]
                )
                if not found_related:
                    missing_topics.append(term)

        # Build recommendations
        recommendations = []
        if query_result_overlap < 0.3:
            recommendations.append("Results may not match query intent - consider rephrasing")
        if prior_result_overlap < 0.3 and applied_priors:
            recommendations.append("Results don't align with learned patterns - may need more data")
        if missing_topics:
            recommendations.append(f"Missing topics: {', '.join(missing_topics[:3])}")

        is_coherent = coherence_score >= self.coherence_threshold

        return CoherenceReport(
            is_coherent=is_coherent,
            coherence_score=coherence_score,
            query_result_alignment=query_result_overlap,
            prior_result_alignment=prior_result_overlap,
            conflicting_results=[],  # TODO: Implement conflict detection
            missing_expected_topics=missing_topics[:5],
            recommendations=recommendations,
        )

    def _get_prior_type(self, prior: Any) -> str:
        """Determine the type of a prior."""
        type_name = type(prior).__name__
        if "Entity" in type_name:
            return "entity"
        elif "Relation" in type_name:
            return "relation"
        elif "Temporal" in type_name:
            return "temporal"
        return "unknown"

    def _get_prior_name(self, prior: Any) -> str:
        """Get the name/identifier of a prior."""
        for attr in ["entity_type", "relation_type", "pattern_type"]:
            if hasattr(prior, attr):
                return getattr(prior, attr)
        return str(prior)

    def _get_prior_searchable_text(self, prior: Any) -> str:
        """Get searchable text from a prior for similarity computation."""
        parts = []

        # Get name
        parts.append(self._get_prior_name(prior))

        # Get context pattern or extraction guidance
        for attr in ["context_pattern", "extraction_guidance"]:
            if hasattr(prior, attr) and getattr(prior, attr):
                parts.append(getattr(prior, attr))

        # Get examples
        if hasattr(prior, "examples") and prior.examples:
            parts.extend(prior.examples[:3])

        return " ".join(parts)

    def _get_result_content(self, result: Any) -> str:
        """Extract content text from a search result."""
        if isinstance(result, dict):
            return result.get("content", "") or result.get("text", "")
        elif hasattr(result, "content"):
            return result.content
        return str(result)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for term matching."""
        # Remove punctuation and split
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        # Filter stopwords
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "i", "you", "he", "she", "it", "we", "they", "this", "that",
            "and", "or", "but", "if", "then", "else", "for", "of", "to",
            "in", "on", "at", "by", "from", "with", "about", "as", "into",
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _calculate_intent_alignment(
        self,
        query_lower: str,
        prior_type: str,
        prior: Any,
    ) -> float:
        """Calculate alignment between query intent and prior type."""
        score = 0.0

        # Check temporal intent
        temporal_matches = sum(1 for k in self._temporal_keywords if k in query_lower)
        if temporal_matches > 0:
            if prior_type == "temporal":
                score += 0.5 + min(0.3, temporal_matches * 0.1)
            elif "event" in self._get_prior_name(prior).lower():
                score += 0.2

        # Check causal intent
        causal_matches = sum(1 for k in self._causal_keywords if k in query_lower)
        if causal_matches > 0:
            if prior_type == "relation":
                score += 0.5 + min(0.3, causal_matches * 0.1)

        # Check entity intent
        entity_matches = sum(1 for k in self._entity_keywords if k in query_lower)
        if entity_matches > 0:
            if prior_type == "entity":
                score += 0.4 + min(0.3, entity_matches * 0.1)

        return min(1.0, score)

    def _calculate_name_relevance(
        self,
        query_lower: str,
        prior_name: str,
    ) -> float:
        """Calculate relevance of prior name to query."""
        prior_name_lower = prior_name.lower()

        # Direct mention
        if prior_name_lower in query_lower:
            return 1.0

        # Partial match
        prior_terms = set(self._tokenize(prior_name_lower))
        query_terms = set(self._tokenize(query_lower))

        if prior_terms & query_terms:
            return 0.7

        # Check for semantic similarity (simple heuristics)
        # Map common variations
        semantic_groups = [
            {"meeting", "meetings", "meet", "met", "conference", "call"},
            {"document", "documents", "doc", "docs", "file", "files", "note", "notes"},
            {"person", "people", "user", "users", "contact", "contacts"},
            {"project", "projects", "task", "tasks", "work"},
            {"email", "emails", "mail", "message", "messages"},
            {"code", "coding", "programming", "development", "dev"},
        ]

        for group in semantic_groups:
            if (prior_terms & group) and (query_terms & group):
                return 0.5

        return 0.0

    def _compute_embedding_similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:
        """Compute embedding-based similarity (if model available)."""
        if not self._embedding_model:
            return 0.0

        try:
            # Assuming embedding model has an encode method
            emb_a = self._embedding_model.encode(text_a)
            emb_b = self._embedding_model.encode(text_b)

            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(emb_a, emb_b))
            norm_a = sum(a * a for a in emb_a) ** 0.5
            norm_b = sum(b * b for b in emb_b) ** 0.5

            if norm_a > 0 and norm_b > 0:
                return dot_product / (norm_a * norm_b)
        except Exception as e:
            logger.warning(f"Embedding similarity computation failed: {e}")

        return 0.0
