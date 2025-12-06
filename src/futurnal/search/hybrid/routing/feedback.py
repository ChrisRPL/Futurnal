"""Search Quality Feedback for GRPO Integration.

Collects search quality signals and connects to the Ghost->Animal
evolution system via TrainingFreeGRPO.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Signal Types:
- click: User clicked on result (positive)
- refinement: User refined query (negative - first query wasn't good enough)
- no_results: No results found (negative)
- satisfaction: Explicit satisfaction rating

Integration:
- Converts signals to SemanticAdvantage format for GRPO
- Triggers experiential knowledge updates after threshold
- Maintains query history for pattern analysis

Option B Compliance:
- No model parameter updates
- Learning via token priors (experiential knowledge)
- Thought templates evolve, model stays frozen
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from futurnal.search.hybrid.types import QueryPlan

if TYPE_CHECKING:
    from futurnal.extraction.schema.experiential import TrainingFreeGRPO
    from futurnal.extraction.schema.models import SemanticAdvantage

logger = logging.getLogger(__name__)


class SearchQualitySignal(BaseModel):
    """Quality signal from search interaction.

    Records user interactions with search results for
    experiential learning.

    Attributes:
        query_id: Reference to QueryPlan.query_id
        signal_type: Type of signal (click, refinement, no_results, satisfaction)
        signal_value: Value from -1.0 (negative) to 1.0 (positive)
        timestamp: When signal was recorded
        context: Additional context (e.g., clicked entity, refinement text)
    """

    query_id: str = Field(description="Reference to QueryPlan.query_id")
    signal_type: str = Field(
        description="Signal type: click, refinement, no_results, satisfaction"
    )
    signal_value: float = Field(
        ge=-1.0,
        le=1.0,
        description="Signal value (-1 to 1)",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Signal timestamp",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context",
    )


class SearchQualityFeedback:
    """Collect search quality signals for GRPO experiential learning.

    Connects hybrid search to Ghost->Animal evolution system.

    This class:
    1. Records query plans for tracking
    2. Collects quality signals from user interactions
    3. Aggregates signals into patterns
    4. Triggers GRPO updates when threshold reached
    5. Converts signals to SemanticAdvantage format

    Example:
        grpo = TrainingFreeGRPO(model=ghost_model)
        feedback = SearchQualityFeedback(grpo_engine=grpo)

        # Record query
        feedback.record_query(query_plan)

        # Record user interactions
        feedback.record_signal(
            query_id=plan.query_id,
            signal_type="click",
            signal_value=1.0,
            context={"entity_id": "clicked_entity"}
        )

        # After threshold signals, GRPO is automatically updated
    """

    # Configuration
    GRPO_TRIGGER_THRESHOLD = 10  # Trigger update after N signals
    MAX_QUERY_HISTORY = 1000  # Keep last N queries
    MAX_SIGNALS = 5000  # Keep last N signals

    def __init__(
        self,
        grpo_engine: Optional["TrainingFreeGRPO"] = None,
    ):
        """Initialize search quality feedback collector.

        Args:
            grpo_engine: TrainingFreeGRPO instance for experiential learning
        """
        self.grpo = grpo_engine
        self.query_history: List[QueryPlan] = []
        self.signals: List[SearchQualitySignal] = []

        logger.info(
            f"SearchQualityFeedback initialized "
            f"(GRPO: {'connected' if grpo_engine else 'not connected'})"
        )

    def record_query(self, plan: QueryPlan):
        """Record query for tracking.

        Called by QueryRouter after routing.

        Args:
            plan: Query execution plan
        """
        self.query_history.append(plan)

        # Trim history if too large
        if len(self.query_history) > self.MAX_QUERY_HISTORY:
            self.query_history = self.query_history[-self.MAX_QUERY_HISTORY:]

        logger.debug(f"Recorded query: {plan.query_id} ({plan.intent.value})")

    def record_signal(
        self,
        query_id: str,
        signal_type: str,
        signal_value: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Record quality signal from user interaction.

        Signal types:
        - "click": User clicked on result (positive, typically 1.0)
        - "refinement": User refined query (negative, typically -0.5)
        - "no_results": No results found (negative, typically -1.0)
        - "satisfaction": Explicit rating (value from -1 to 1)

        Args:
            query_id: Reference to QueryPlan.query_id
            signal_type: Type of signal
            signal_value: Value from -1.0 to 1.0
            context: Additional context
        """
        # Validate signal type
        valid_types = {"click", "refinement", "no_results", "satisfaction"}
        if signal_type not in valid_types:
            logger.warning(f"Unknown signal type: {signal_type}")

        signal = SearchQualitySignal(
            query_id=query_id,
            signal_type=signal_type,
            signal_value=max(-1.0, min(1.0, signal_value)),
            timestamp=datetime.utcnow(),
            context=context or {},
        )

        self.signals.append(signal)

        # Trim signals if too large
        if len(self.signals) > self.MAX_SIGNALS:
            self.signals = self.signals[-self.MAX_SIGNALS:]

        logger.debug(
            f"Recorded signal: {signal_type}={signal_value:.2f} "
            f"for query {query_id}"
        )

        # Check if we should trigger GRPO update
        if len(self.signals) >= self.GRPO_TRIGGER_THRESHOLD and self.grpo:
            self._trigger_grpo_update()

    def _trigger_grpo_update(self):
        """Update experiential knowledge based on search signals.

        This is how search quality contributes to Ghost->Animal evolution.
        Converts search signals to SemanticAdvantage format for GRPO.
        """
        logger.info("Triggering GRPO update from search signals")

        # Group signals by query
        query_signals = self._aggregate_signals()

        # Convert to semantic advantages
        advantages = self._extract_advantages(query_signals)

        # Update GRPO experiential knowledge
        if advantages and self.grpo:
            try:
                self.grpo.update_experiential_knowledge(advantages)
                logger.info(f"GRPO updated with {len(advantages)} advantages")
            except Exception as e:
                logger.error(f"GRPO update failed: {e}")

        # Clear processed signals
        self.signals = []

    def _aggregate_signals(self) -> Dict[str, List[SearchQualitySignal]]:
        """Group signals by query ID.

        Returns:
            Dictionary mapping query_id to list of signals
        """
        grouped: Dict[str, List[SearchQualitySignal]] = {}
        for signal in self.signals:
            if signal.query_id not in grouped:
                grouped[signal.query_id] = []
            grouped[signal.query_id].append(signal)
        return grouped

    def _extract_advantages(
        self,
        query_signals: Dict[str, List[SearchQualitySignal]],
    ) -> List[Dict[str, Any]]:
        """Extract semantic advantages from signal patterns.

        Patterns detected:
        - High click rate -> good intent classification
        - Query refinement -> poor initial understanding
        - No results -> strategy selection issue

        Args:
            query_signals: Signals grouped by query

        Returns:
            List of SemanticAdvantage-compatible dictionaries
        """
        advantages = []

        for query_id, signals in query_signals.items():
            # Find corresponding query plan
            plan = next(
                (p for p in self.query_history if p.query_id == query_id),
                None,
            )

            if not plan:
                continue

            # Analyze signal patterns
            click_signals = [s for s in signals if s.signal_type == "click"]
            refine_signals = [s for s in signals if s.signal_type == "refinement"]
            no_result_signals = [s for s in signals if s.signal_type == "no_results"]

            # Good results - record successful pattern
            if len(click_signals) >= 2:
                advantages.append({
                    "better_approach": (
                        f"Intent '{plan.intent.value}' with strategy "
                        f"'{plan.primary_strategy}' works well for queries like: "
                        f"{plan.original_query[:50]}"
                    ),
                    "worse_approach": "Alternative intent classifications",
                    "reasoning": (
                        f"High engagement: {len(click_signals)} clicks, "
                        f"confidence: {plan.intent_confidence:.2f}"
                    ),
                    "confidence": min(0.9, 0.6 + len(click_signals) * 0.1),
                })

            # Query refinement needed - room for improvement
            if len(refine_signals) > 0:
                advantages.append({
                    "better_approach": (
                        f"Consider alternative routing for queries like: "
                        f"{plan.original_query[:50]}"
                    ),
                    "worse_approach": (
                        f"Current approach: intent '{plan.intent.value}' "
                        f"with strategy '{plan.primary_strategy}'"
                    ),
                    "reasoning": (
                        f"Query refinement needed: {len(refine_signals)} refinements"
                    ),
                    "confidence": 0.6,
                })

            # No results - strategy may need adjustment
            if len(no_result_signals) > 0:
                advantages.append({
                    "better_approach": (
                        f"Expand search scope for intent '{plan.intent.value}'"
                    ),
                    "worse_approach": (
                        f"Current strategy '{plan.primary_strategy}' "
                        f"returned no results"
                    ),
                    "reasoning": "No results found for user query",
                    "confidence": 0.7,
                })

        return advantages

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get current search quality metrics for World State Model.

        Returns metrics that can be used to track search quality over time.

        Returns:
            Dictionary with quality metrics
        """
        if not self.signals:
            return {
                "insufficient_data": True,
                "signal_count": 0,
            }

        total_signals = len(self.signals)
        click_count = len([s for s in self.signals if s.signal_type == "click"])
        refine_count = len([s for s in self.signals if s.signal_type == "refinement"])
        no_result_count = len([s for s in self.signals if s.signal_type == "no_results"])

        click_rate = click_count / total_signals if total_signals > 0 else 0
        refine_rate = refine_count / total_signals if total_signals > 0 else 0
        no_result_rate = no_result_count / total_signals if total_signals > 0 else 0

        return {
            "click_rate": click_rate,
            "refinement_rate": refine_rate,
            "no_result_rate": no_result_rate,
            "satisfaction_trend": click_rate - refine_rate,
            "signal_count": total_signals,
            "query_count": len(self.query_history),
        }

    def get_intent_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by intent type.

        Returns:
            Dictionary mapping intent to performance metrics
        """
        intent_signals: Dict[str, List[SearchQualitySignal]] = {}

        # Group signals by intent
        for signal in self.signals:
            plan = next(
                (p for p in self.query_history if p.query_id == signal.query_id),
                None,
            )
            if plan:
                intent = plan.intent.value
                if intent not in intent_signals:
                    intent_signals[intent] = []
                intent_signals[intent].append(signal)

        # Calculate metrics per intent
        result: Dict[str, Dict[str, float]] = {}
        for intent, signals in intent_signals.items():
            total = len(signals)
            clicks = len([s for s in signals if s.signal_type == "click"])
            refines = len([s for s in signals if s.signal_type == "refinement"])

            result[intent] = {
                "click_rate": clicks / total if total > 0 else 0,
                "refinement_rate": refines / total if total > 0 else 0,
                "signal_count": total,
            }

        return result

    def clear_history(self):
        """Clear all recorded queries and signals.

        Useful for testing or resetting state.
        """
        self.query_history.clear()
        self.signals.clear()
        logger.info("Cleared search quality history")

    def export_data(self) -> Dict[str, Any]:
        """Export all data for analysis or persistence.

        Returns:
            Dictionary with all recorded data
        """
        return {
            "queries": [p.model_dump() for p in self.query_history],
            "signals": [s.model_dump() for s in self.signals],
            "metrics": self.get_quality_metrics(),
            "intent_performance": self.get_intent_performance(),
        }
