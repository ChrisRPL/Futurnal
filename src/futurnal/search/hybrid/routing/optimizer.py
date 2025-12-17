"""Search Ranking Optimizer for Bidirectional Learning.

AGI Phase 3: Closes the feedback loop by updating routing strategy weights
based on search quality signals.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Research Foundation:
- SEAgent (2508.04700v2): Complete experiential learning loop
- Training-Free GRPO (2510.08191v1): Learning without model updates

Key Innovation:
Standard RAG systems treat routing as static configuration. This optimizer
implements bidirectional learning where search quality signals flow back
to dynamically adjust strategy weights, creating a self-improving system.

Option B Compliance:
- Ghost model FROZEN: No parameter updates
- Learning via weight adjustments and token priors
- All learning is interpretable natural language
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from futurnal.search.hybrid.types import QueryIntent

if TYPE_CHECKING:
    from futurnal.search.hybrid.routing.feedback import (
        SearchQualityFeedback,
        SearchQualitySignal,
    )
    from futurnal.learning.token_priors import TokenPriorStore

logger = logging.getLogger(__name__)


@dataclass
class StrategyEffectiveness:
    """Effectiveness metrics for a routing strategy.

    Tracks performance of a strategy-intent combination over time
    to enable learning-based weight adjustments.
    """

    strategy: str
    intent: QueryIntent

    # Core metrics
    total_queries: int = 0
    successful_queries: int = 0  # Had clicks
    failed_queries: int = 0  # Had refinements or no results

    # Signal aggregates
    total_clicks: int = 0
    total_refinements: int = 0
    total_no_results: int = 0
    total_satisfaction_score: float = 0.0
    satisfaction_count: int = 0

    # Time-weighted metrics (recent signals weighted more)
    weighted_success_rate: float = 0.5  # Initial neutral

    # Confidence in our estimate
    confidence: float = 0.0  # Increases with more data

    # Last update timestamp
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def raw_success_rate(self) -> float:
        """Calculate raw success rate from signals."""
        total = self.successful_queries + self.failed_queries
        if total == 0:
            return 0.5  # Neutral prior
        return self.successful_queries / total

    @property
    def click_rate(self) -> float:
        """Click rate across all queries."""
        if self.total_queries == 0:
            return 0.0
        return self.total_clicks / self.total_queries

    @property
    def refinement_rate(self) -> float:
        """Refinement rate across all queries."""
        if self.total_queries == 0:
            return 0.0
        return self.total_refinements / self.total_queries

    @property
    def avg_satisfaction(self) -> float:
        """Average satisfaction score."""
        if self.satisfaction_count == 0:
            return 0.0
        return self.total_satisfaction_score / self.satisfaction_count

    @property
    def composite_score(self) -> float:
        """Composite effectiveness score (0-1).

        Combines multiple signals into single effectiveness measure:
        - Click rate (positive)
        - Refinement rate (negative)
        - No-result rate (negative)
        - Satisfaction (positive)
        """
        if self.total_queries == 0:
            return 0.5  # Neutral prior

        # Base on click rate
        score = self.click_rate

        # Penalty for refinements (query wasn't good enough)
        score -= 0.3 * self.refinement_rate

        # Penalty for no results
        no_result_rate = self.total_no_results / self.total_queries if self.total_queries > 0 else 0
        score -= 0.5 * no_result_rate

        # Boost from satisfaction
        if self.satisfaction_count > 0:
            score += 0.2 * max(0, self.avg_satisfaction)

        return max(0.0, min(1.0, score))


@dataclass
class WeightUpdate:
    """A proposed weight update for a strategy-intent pair."""

    intent: QueryIntent
    strategy: str
    old_weight: float
    new_weight: float
    reason: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def delta(self) -> float:
        """Weight change magnitude."""
        return self.new_weight - self.old_weight

    @property
    def is_significant(self) -> bool:
        """Is this update significant enough to apply?"""
        return abs(self.delta) > 0.02  # >2% change


class SearchRankingOptimizer:
    """Optimizes search routing based on quality feedback.

    AGI Phase 3 core component that closes the bidirectional learning loop.

    Unlike standard RAG systems with static routing, this optimizer:
    1. Tracks strategy effectiveness per intent
    2. Computes optimal weight adjustments
    3. Updates STRATEGY_CONFIGS dynamically
    4. Exports learned configs as token priors

    Learning Algorithm:
    - Exponential moving average for time-weighted success rates
    - Thompson Sampling inspired exploration vs exploitation
    - Bonferroni-like confidence thresholds before updates

    Example:
        optimizer = SearchRankingOptimizer()
        optimizer.process_signal(signal, plan)

        # When threshold reached
        if optimizer.should_update():
            updates = optimizer.compute_weight_updates(current_configs)
            for update in updates:
                router.update_strategy_config(update.intent, {"weights": ...})
    """

    # Configuration
    MIN_SIGNALS_FOR_UPDATE = 20  # Minimum signals before computing updates
    UPDATE_INTERVAL_HOURS = 1  # Minimum hours between updates
    MAX_WEIGHT_CHANGE = 0.15  # Maximum weight change per update (15%)
    EMA_ALPHA = 0.1  # Exponential moving average decay
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to apply update
    EXPLORATION_RATE = 0.1  # 10% exploration for under-tested strategies

    def __init__(
        self,
        token_prior_store: Optional["TokenPriorStore"] = None,
        min_signals: int = MIN_SIGNALS_FOR_UPDATE,
    ):
        """Initialize optimizer.

        Args:
            token_prior_store: For exporting learned configs as priors
            min_signals: Minimum signals before computing updates
        """
        self._token_prior_store = token_prior_store
        self._min_signals = min_signals

        # Track effectiveness per strategy-intent pair
        self._effectiveness: Dict[Tuple[str, QueryIntent], StrategyEffectiveness] = {}

        # Track all processed signals for analysis
        self._signal_count = 0
        self._last_update: Optional[datetime] = None

        # History of weight updates for analysis
        self._update_history: List[WeightUpdate] = []

        # Default strategy weights (baseline)
        self._default_weights: Dict[QueryIntent, Dict[str, float]] = {
            QueryIntent.TEMPORAL: {"temporal": 0.7, "hybrid": 0.3},
            QueryIntent.CAUSAL: {"causal": 0.6, "temporal": 0.4},
            QueryIntent.LOOKUP: {"hybrid": 1.0},
            QueryIntent.EXPLORATORY: {"hybrid": 0.6, "temporal": 0.4},
        }

        # Current learned weights (start with defaults)
        self._learned_weights: Dict[QueryIntent, Dict[str, float]] = {
            intent: weights.copy()
            for intent, weights in self._default_weights.items()
        }

        logger.info(
            f"SearchRankingOptimizer initialized "
            f"(min_signals={min_signals}, token_priors={'connected' if token_prior_store else 'not connected'})"
        )

    def process_signal(
        self,
        signal: "SearchQualitySignal",
        intent: QueryIntent,
        primary_strategy: str,
        secondary_strategy: Optional[str] = None,
    ):
        """Process a quality signal and update effectiveness tracking.

        This is called by SearchQualityFeedback after recording a signal.

        Args:
            signal: The quality signal
            intent: Query intent that was classified
            primary_strategy: Primary strategy that was used
            secondary_strategy: Secondary strategy (if any)
        """
        self._signal_count += 1

        # Get or create effectiveness tracker for primary strategy
        key = (primary_strategy, intent)
        if key not in self._effectiveness:
            self._effectiveness[key] = StrategyEffectiveness(
                strategy=primary_strategy,
                intent=intent,
            )

        eff = self._effectiveness[key]
        eff.total_queries += 1
        eff.last_updated = datetime.utcnow()

        # Update based on signal type
        if signal.signal_type == "click":
            eff.total_clicks += 1
            eff.successful_queries += 1
            self._update_weighted_success(eff, success=True)

        elif signal.signal_type == "refinement":
            eff.total_refinements += 1
            eff.failed_queries += 1
            self._update_weighted_success(eff, success=False)

        elif signal.signal_type == "no_results":
            eff.total_no_results += 1
            eff.failed_queries += 1
            self._update_weighted_success(eff, success=False)

        elif signal.signal_type == "satisfaction":
            eff.total_satisfaction_score += signal.signal_value
            eff.satisfaction_count += 1
            # Satisfaction > 0 is success, < 0 is failure
            self._update_weighted_success(eff, success=signal.signal_value > 0)

        # Update confidence based on sample size
        self._update_confidence(eff)

        logger.debug(
            f"Processed signal for {primary_strategy}/{intent.value}: "
            f"composite_score={eff.composite_score:.3f}, confidence={eff.confidence:.3f}"
        )

    def _update_weighted_success(
        self,
        eff: StrategyEffectiveness,
        success: bool,
    ):
        """Update time-weighted success rate using EMA.

        Recent signals have more influence than older ones.
        """
        signal_value = 1.0 if success else 0.0
        eff.weighted_success_rate = (
            self.EMA_ALPHA * signal_value +
            (1 - self.EMA_ALPHA) * eff.weighted_success_rate
        )

    def _update_confidence(self, eff: StrategyEffectiveness):
        """Update confidence based on sample size.

        Uses logistic function to model confidence growth.
        """
        # Confidence grows with more samples, saturating around 50 samples
        n = eff.total_queries
        eff.confidence = 1.0 / (1.0 + math.exp(-0.1 * (n - 20)))

    def should_update(self) -> bool:
        """Check if we should compute weight updates.

        Returns True if:
        - Have enough signals
        - Enough time has passed since last update
        """
        if self._signal_count < self._min_signals:
            return False

        if self._last_update is not None:
            hours_since_update = (
                datetime.utcnow() - self._last_update
            ).total_seconds() / 3600
            if hours_since_update < self.UPDATE_INTERVAL_HOURS:
                return False

        return True

    def compute_weight_updates(
        self,
        current_configs: Dict[QueryIntent, Dict[str, Any]],
    ) -> List[WeightUpdate]:
        """Compute weight updates based on accumulated effectiveness data.

        AGI Phase 3 core algorithm: Computes optimal weight adjustments
        based on observed strategy effectiveness.

        Algorithm:
        1. For each intent, compute relative effectiveness of strategies
        2. Adjust weights proportional to effectiveness
        3. Apply damping to prevent oscillation
        4. Cap maximum change to ensure stability

        Args:
            current_configs: Current STRATEGY_CONFIGS from QueryRouter

        Returns:
            List of proposed weight updates
        """
        updates: List[WeightUpdate] = []

        for intent in QueryIntent:
            if intent not in current_configs:
                continue

            current_weights = current_configs[intent].get("weights", {})
            if not current_weights:
                continue

            # Get effectiveness for all strategies used by this intent
            strategy_scores: Dict[str, Tuple[float, float]] = {}  # strategy -> (score, confidence)

            for strategy_key in current_weights:
                # Map weight key to strategy name
                strategy_map = {
                    "temporal": "temporal_query",
                    "causal": "causal_chain",
                    "hybrid": "hybrid_retrieval",
                }
                strategy = strategy_map.get(strategy_key, strategy_key)

                key = (strategy, intent)
                if key in self._effectiveness:
                    eff = self._effectiveness[key]
                    strategy_scores[strategy_key] = (eff.composite_score, eff.confidence)
                else:
                    # No data - use neutral score with low confidence
                    strategy_scores[strategy_key] = (0.5, 0.0)

            # Compute new weights based on effectiveness
            new_weights = self._compute_new_weights(
                current_weights,
                strategy_scores,
                intent,
            )

            # Generate updates for changed weights
            for strategy_key, new_weight in new_weights.items():
                old_weight = current_weights.get(strategy_key, 0.5)

                if abs(new_weight - old_weight) > 0.02:  # >2% change
                    # Get confidence for this update
                    _, confidence = strategy_scores.get(strategy_key, (0.5, 0.0))

                    reason = self._generate_update_reason(
                        strategy_key,
                        intent,
                        old_weight,
                        new_weight,
                        strategy_scores,
                    )

                    updates.append(WeightUpdate(
                        intent=intent,
                        strategy=strategy_key,
                        old_weight=old_weight,
                        new_weight=new_weight,
                        reason=reason,
                        confidence=confidence,
                    ))

        # Filter by confidence threshold
        significant_updates = [
            u for u in updates
            if u.confidence >= self.CONFIDENCE_THRESHOLD and u.is_significant
        ]

        logger.info(
            f"Computed {len(significant_updates)} significant weight updates "
            f"(from {len(updates)} total)"
        )

        return significant_updates

    def _compute_new_weights(
        self,
        current_weights: Dict[str, float],
        strategy_scores: Dict[str, Tuple[float, float]],
        intent: QueryIntent,
    ) -> Dict[str, float]:
        """Compute new weights based on strategy effectiveness.

        Algorithm:
        1. Normalize effectiveness scores
        2. Blend with current weights (damping)
        3. Apply maximum change constraint
        4. Normalize to sum to 1.0
        """
        new_weights: Dict[str, float] = {}

        # Calculate total effectiveness for normalization
        total_score = sum(
            score * confidence  # Weight by confidence
            for score, confidence in strategy_scores.values()
        )

        if total_score == 0:
            # No data - return current weights
            return current_weights.copy()

        for strategy_key, current_weight in current_weights.items():
            score, confidence = strategy_scores.get(strategy_key, (0.5, 0.0))

            # Target weight proportional to effectiveness
            if total_score > 0:
                target_weight = (score * confidence) / total_score
            else:
                target_weight = current_weight

            # Blend current and target (damping)
            # Higher confidence -> more influence from target
            blend_factor = min(0.5, confidence * 0.5)
            blended_weight = (
                blend_factor * target_weight +
                (1 - blend_factor) * current_weight
            )

            # Apply maximum change constraint
            delta = blended_weight - current_weight
            if abs(delta) > self.MAX_WEIGHT_CHANGE:
                delta = self.MAX_WEIGHT_CHANGE if delta > 0 else -self.MAX_WEIGHT_CHANGE

            new_weights[strategy_key] = current_weight + delta

        # Normalize to sum to 1.0
        weight_sum = sum(new_weights.values())
        if weight_sum > 0:
            new_weights = {k: v / weight_sum for k, v in new_weights.items()}

        return new_weights

    def _generate_update_reason(
        self,
        strategy_key: str,
        intent: QueryIntent,
        old_weight: float,
        new_weight: float,
        scores: Dict[str, Tuple[float, float]],
    ) -> str:
        """Generate human-readable reason for weight update."""
        score, confidence = scores.get(strategy_key, (0.5, 0.0))

        direction = "increased" if new_weight > old_weight else "decreased"
        delta_pct = abs(new_weight - old_weight) * 100

        return (
            f"{strategy_key.capitalize()} weight {direction} by {delta_pct:.1f}% "
            f"for {intent.value} queries. "
            f"Effectiveness score: {score:.2f}, confidence: {confidence:.2f}"
        )

    def apply_updates(
        self,
        updates: List[WeightUpdate],
        router: Any,  # QueryRouter
    ):
        """Apply weight updates to the QueryRouter.

        Args:
            updates: List of weight updates to apply
            router: QueryRouter instance
        """
        # Group updates by intent
        intent_updates: Dict[QueryIntent, Dict[str, float]] = defaultdict(dict)

        for update in updates:
            intent_updates[update.intent][update.strategy] = update.new_weight
            self._update_history.append(update)

        # Apply to router
        for intent, weights in intent_updates.items():
            # Get current config and update weights
            current_config = router.get_strategy_config(intent)
            current_weights = current_config.get("weights", {})
            current_weights.update(weights)

            # Update router
            router.update_strategy_config(intent, {"weights": current_weights})

            # Update our learned weights
            if intent not in self._learned_weights:
                self._learned_weights[intent] = {}
            self._learned_weights[intent].update(weights)

            logger.info(
                f"Applied weight update for {intent.value}: {weights}"
            )

        # Record update time
        self._last_update = datetime.utcnow()

        # Export to token priors if connected
        if self._token_prior_store:
            self._export_to_token_priors()

    def _export_to_token_priors(self):
        """Export learned weights as token priors for persistence.

        This is how learned routing knowledge persists across sessions
        in Option B compliance (learning via natural language).
        """
        if not self._token_prior_store:
            return

        try:
            from futurnal.learning.token_priors import TemporalPatternPrior

            # Create temporal pattern prior for routing knowledge
            learned_config_text = self._format_learned_configs()

            prior = TemporalPatternPrior(
                pattern_type="routing_weights",
                description="Learned search routing weights from user feedback",
                learned_weights=learned_config_text,
                confidence=self._compute_overall_confidence(),
                observation_count=self._signal_count,
            )

            self._token_prior_store.add_temporal_pattern(prior)

            logger.info("Exported learned routing weights to token priors")

        except Exception as e:
            logger.error(f"Failed to export to token priors: {e}")

    def _format_learned_configs(self) -> str:
        """Format learned configs as natural language for token priors."""
        lines = ["Learned search routing preferences:"]

        for intent, weights in self._learned_weights.items():
            weight_desc = ", ".join(
                f"{k}={v:.2f}" for k, v in weights.items()
            )
            lines.append(f"- {intent.value}: {weight_desc}")

        # Add top insights
        for key, eff in sorted(
            self._effectiveness.items(),
            key=lambda x: x[1].composite_score,
            reverse=True,
        )[:3]:
            strategy, intent = key
            lines.append(
                f"- {strategy} performs well for {intent.value} "
                f"(score: {eff.composite_score:.2f})"
            )

        return "\n".join(lines)

    def _compute_overall_confidence(self) -> float:
        """Compute overall confidence in learned weights."""
        if not self._effectiveness:
            return 0.0

        confidences = [eff.confidence for eff in self._effectiveness.values()]
        return sum(confidences) / len(confidences)

    def export_learned_configs(self) -> str:
        """Export learned configs as JSON for external use.

        Returns:
            JSON string of learned configurations
        """
        data = {
            "learned_weights": {
                intent.value: weights
                for intent, weights in self._learned_weights.items()
            },
            "effectiveness": {
                f"{strategy}_{intent.value}": {
                    "composite_score": eff.composite_score,
                    "click_rate": eff.click_rate,
                    "refinement_rate": eff.refinement_rate,
                    "confidence": eff.confidence,
                    "total_queries": eff.total_queries,
                }
                for (strategy, intent), eff in self._effectiveness.items()
            },
            "signal_count": self._signal_count,
            "last_update": self._last_update.isoformat() if self._last_update else None,
        }

        return json.dumps(data, indent=2)

    def get_strategy_effectiveness(
        self,
        intent: QueryIntent,
    ) -> Dict[str, float]:
        """Get effectiveness scores for all strategies of an intent.

        Args:
            intent: Query intent

        Returns:
            Dictionary mapping strategy to composite score
        """
        result: Dict[str, float] = {}

        for (strategy, eff_intent), eff in self._effectiveness.items():
            if eff_intent == intent:
                # Map strategy back to weight key
                strategy_map = {
                    "temporal_query": "temporal",
                    "causal_chain": "causal",
                    "hybrid_retrieval": "hybrid",
                }
                key = strategy_map.get(strategy, strategy)
                result[key] = eff.composite_score

        return result

    def reset(self):
        """Reset all learned state."""
        self._effectiveness.clear()
        self._signal_count = 0
        self._last_update = None
        self._update_history.clear()
        self._learned_weights = {
            intent: weights.copy()
            for intent, weights in self._default_weights.items()
        }
        logger.info("SearchRankingOptimizer reset")
