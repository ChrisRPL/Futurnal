"""
FutureX-Style Future Prediction System.

Implements dynamic future event prediction based on:
- Temporal patterns in knowledge graph
- Causal relationships
- Historical trends
- LLM reasoning

Research Foundation:
- FutureX (2508.11987v3): Dynamic benchmark for prediction
- Temporal reasoning for future events
- Causal inference for prediction

Key Features:
- Trend extrapolation from temporal patterns
- Causal chain projection
- Uncertainty quantification
- What-if scenario analysis

Option B Compliance:
- Rule-based prediction algorithms
- LLM for reasoning synthesis only
- No model fine-tuning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import math

logger = logging.getLogger(__name__)


class PredictionType(str, Enum):
    """Types of future predictions."""
    EVENT = "event"  # Predict if/when event will occur
    TREND = "trend"  # Predict trend direction
    OUTCOME = "outcome"  # Predict outcome of situation
    TEMPORAL = "temporal"  # Predict timing
    CAUSAL = "causal"  # Predict causal consequences


class ConfidenceLevel(str, Enum):
    """Confidence levels for predictions."""
    HIGH = "high"  # >80% confidence
    MEDIUM = "medium"  # 50-80% confidence
    LOW = "low"  # 20-50% confidence
    SPECULATIVE = "speculative"  # <20% confidence


@dataclass
class TemporalPattern:
    """A temporal pattern used for prediction."""
    pattern_type: str  # "cyclic", "trend", "seasonal", "random"
    period_days: Optional[float] = None
    trend_direction: Optional[str] = None  # "increasing", "decreasing", "stable"
    strength: float = 0.0
    samples: int = 0


@dataclass
class FuturePrediction:
    """A prediction about future events."""
    prediction_id: str
    prediction_type: PredictionType

    # What is predicted
    subject: str  # What the prediction is about
    prediction: str  # The actual prediction statement
    predicted_outcome: Optional[str] = None

    # Timing
    predicted_date: Optional[datetime] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    horizon_days: int = 0

    # Confidence
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.SPECULATIVE
    uncertainty_range: Optional[Tuple[float, float]] = None

    # Evidence
    supporting_patterns: List[TemporalPattern] = field(default_factory=list)
    causal_factors: List[str] = field(default_factory=list)
    historical_analogies: List[str] = field(default_factory=list)

    # Reasoning
    reasoning: str = ""
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "prediction_type": self.prediction_type.value,
            "subject": self.subject,
            "prediction": self.prediction,
            "predicted_date": self.predicted_date.isoformat() if self.predicted_date else None,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "reasoning": self.reasoning,
        }


@dataclass
class ScenarioAnalysis:
    """What-if scenario analysis result."""
    scenario_id: str
    scenario_description: str

    # Conditions
    conditions: List[str]
    changed_factors: Dict[str, Any]

    # Outcomes
    predicted_outcomes: List[FuturePrediction]
    probability: float = 0.0

    # Comparison to baseline
    baseline_outcome: Optional[str] = None
    deviation_from_baseline: float = 0.0


class FuturePredictionEngine:
    """
    Engine for predicting future events and outcomes.

    Combines:
    1. Temporal pattern analysis
    2. Causal inference
    3. Trend extrapolation
    4. LLM reasoning
    """

    def __init__(
        self,
        neo4j_driver: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        temporal_engine: Optional[Any] = None,
        default_horizon_days: int = 30
    ):
        """Initialize prediction engine.

        Args:
            neo4j_driver: Neo4j driver for graph queries
            llm_client: LLM for reasoning
            temporal_engine: Temporal query engine
            default_horizon_days: Default prediction horizon
        """
        self.driver = neo4j_driver
        self.llm_client = llm_client
        self.temporal_engine = temporal_engine
        self.default_horizon = default_horizon_days

        # Cache
        self._pattern_cache: Dict[str, List[TemporalPattern]] = {}

    async def predict(
        self,
        subject: str,
        prediction_type: PredictionType = PredictionType.EVENT,
        horizon_days: Optional[int] = None,
        context: Optional[str] = None
    ) -> FuturePrediction:
        """Make a prediction about the future.

        Args:
            subject: What to predict about
            prediction_type: Type of prediction
            horizon_days: How far ahead to predict
            context: Additional context

        Returns:
            FuturePrediction with results
        """
        from uuid import uuid4

        horizon = horizon_days or self.default_horizon

        # Gather evidence
        patterns = await self._analyze_patterns(subject)
        causal_factors = await self._get_causal_factors(subject)
        analogies = await self._find_historical_analogies(subject)

        # Generate prediction based on type
        if prediction_type == PredictionType.EVENT:
            prediction = await self._predict_event(subject, patterns, horizon)
        elif prediction_type == PredictionType.TREND:
            prediction = await self._predict_trend(subject, patterns)
        elif prediction_type == PredictionType.TEMPORAL:
            prediction = await self._predict_timing(subject, patterns, horizon)
        elif prediction_type == PredictionType.CAUSAL:
            prediction = await self._predict_causal(subject, causal_factors)
        else:  # OUTCOME
            prediction = await self._predict_outcome(subject, patterns, causal_factors, context)

        # Calculate confidence
        prediction.confidence = self._calculate_confidence(
            patterns, causal_factors, analogies
        )
        prediction.confidence_level = self._confidence_to_level(prediction.confidence)

        # Generate reasoning
        prediction.reasoning = await self._generate_reasoning(
            subject, prediction, patterns, causal_factors, context
        )

        # Add evidence
        prediction.supporting_patterns = patterns
        prediction.causal_factors = causal_factors
        prediction.historical_analogies = analogies
        prediction.horizon_days = horizon

        return prediction

    def predict_sync(
        self,
        subject: str,
        prediction_type: PredictionType = PredictionType.EVENT,
        horizon_days: Optional[int] = None,
        context: Optional[str] = None
    ) -> FuturePrediction:
        """Synchronous version of predict."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.predict(subject, prediction_type, horizon_days, context)
        )

    async def analyze_scenario(
        self,
        scenario_description: str,
        changed_factors: Dict[str, Any],
        baseline_subject: Optional[str] = None
    ) -> ScenarioAnalysis:
        """Perform what-if scenario analysis.

        Args:
            scenario_description: Description of the scenario
            changed_factors: Factors that are different in this scenario
            baseline_subject: Subject for baseline comparison

        Returns:
            ScenarioAnalysis with outcomes
        """
        from uuid import uuid4

        # Get baseline prediction if subject provided
        baseline_outcome = None
        if baseline_subject:
            baseline = await self.predict(baseline_subject, PredictionType.OUTCOME)
            baseline_outcome = baseline.prediction

        # Generate scenario-specific predictions
        predictions = []
        for factor, value in changed_factors.items():
            pred = await self.predict(
                f"{factor} changed to {value}",
                PredictionType.OUTCOME,
                context=scenario_description
            )
            predictions.append(pred)

        # Calculate overall scenario probability
        probability = 1.0
        for pred in predictions:
            probability *= pred.confidence

        # Calculate deviation from baseline
        deviation = 0.0
        if baseline_outcome and predictions:
            # Simple text-based deviation
            deviation = 0.5  # Placeholder

        return ScenarioAnalysis(
            scenario_id=str(uuid4()),
            scenario_description=scenario_description,
            conditions=[f"{k}: {v}" for k, v in changed_factors.items()],
            changed_factors=changed_factors,
            predicted_outcomes=predictions,
            probability=probability,
            baseline_outcome=baseline_outcome,
            deviation_from_baseline=deviation,
        )

    async def _analyze_patterns(self, subject: str) -> List[TemporalPattern]:
        """Analyze temporal patterns for a subject."""
        patterns = []

        if subject in self._pattern_cache:
            return self._pattern_cache[subject]

        if not self.driver:
            return patterns

        # Query for event occurrences
        query = """
        MATCH (e:Event)
        WHERE e.name CONTAINS $subject OR e.event_type CONTAINS $subject
        RETURN e.timestamp as ts, e.name as name
        ORDER BY e.timestamp
        """

        timestamps = []
        try:
            with self.driver.session() as session:
                result = session.run(query, subject=subject)
                for record in result:
                    if record["ts"]:
                        timestamps.append(record["ts"])
        except Exception as e:
            logger.warning(f"Pattern query failed: {e}")

        if len(timestamps) < 3:
            return patterns

        # Analyze patterns
        patterns.extend(self._detect_cyclic_pattern(timestamps))
        patterns.extend(self._detect_trend_pattern(timestamps))

        self._pattern_cache[subject] = patterns
        return patterns

    def _detect_cyclic_pattern(self, timestamps: List[datetime]) -> List[TemporalPattern]:
        """Detect cyclic/periodic patterns."""
        patterns = []

        if len(timestamps) < 4:
            return patterns

        # Calculate inter-event intervals
        intervals = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i - 1]).total_seconds() / 86400
            intervals.append(delta)

        if not intervals:
            return patterns

        # Check for regularity
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        # Low variance indicates cyclic pattern
        if avg_interval > 0:
            cv = std_dev / avg_interval  # Coefficient of variation
            if cv < 0.3:  # Low variability
                patterns.append(TemporalPattern(
                    pattern_type="cyclic",
                    period_days=avg_interval,
                    strength=1.0 - cv,
                    samples=len(timestamps),
                ))

        return patterns

    def _detect_trend_pattern(self, timestamps: List[datetime]) -> List[TemporalPattern]:
        """Detect trend patterns (increasing/decreasing frequency)."""
        patterns = []

        if len(timestamps) < 5:
            return patterns

        # Calculate intervals in first half vs second half
        mid = len(timestamps) // 2

        first_half_intervals = []
        second_half_intervals = []

        for i in range(1, mid):
            delta = (timestamps[i] - timestamps[i - 1]).total_seconds() / 86400
            first_half_intervals.append(delta)

        for i in range(mid + 1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i - 1]).total_seconds() / 86400
            second_half_intervals.append(delta)

        if not first_half_intervals or not second_half_intervals:
            return patterns

        avg_first = sum(first_half_intervals) / len(first_half_intervals)
        avg_second = sum(second_half_intervals) / len(second_half_intervals)

        # Determine trend
        if avg_first > 0 and avg_second > 0:
            ratio = avg_first / avg_second

            if ratio > 1.3:  # Intervals decreasing = frequency increasing
                direction = "increasing"
                strength = min(1.0, (ratio - 1) / 0.5)
            elif ratio < 0.7:  # Intervals increasing = frequency decreasing
                direction = "decreasing"
                strength = min(1.0, (1 - ratio) / 0.5)
            else:
                direction = "stable"
                strength = 1.0 - abs(ratio - 1)

            patterns.append(TemporalPattern(
                pattern_type="trend",
                trend_direction=direction,
                strength=strength,
                samples=len(timestamps),
            ))

        return patterns

    async def _get_causal_factors(self, subject: str) -> List[str]:
        """Get causal factors for a subject."""
        factors = []

        if not self.driver:
            return factors

        query = """
        MATCH (cause)-[r:CAUSES|LEADS_TO|TRIGGERS]->(e)
        WHERE e.name CONTAINS $subject OR e.event_type CONTAINS $subject
        RETURN DISTINCT cause.name as factor, type(r) as rel_type
        LIMIT 10
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, subject=subject)
                for record in result:
                    if record["factor"]:
                        factors.append(f"{record['factor']} ({record['rel_type']})")
        except Exception as e:
            logger.warning(f"Causal factors query failed: {e}")

        return factors

    async def _find_historical_analogies(self, subject: str) -> List[str]:
        """Find historical analogies for a subject."""
        analogies = []

        if not self.driver:
            return analogies

        # Find similar past events
        query = """
        MATCH (e:Event)
        WHERE e.name CONTAINS $subject
        AND e.timestamp < datetime()
        RETURN e.name as name, e.timestamp as ts, e.description as desc
        ORDER BY e.timestamp DESC
        LIMIT 5
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, subject=subject)
                for record in result:
                    if record["name"]:
                        ts = record["ts"]
                        date_str = ts.strftime("%Y-%m-%d") if ts else "unknown date"
                        analogies.append(f"{record['name']} ({date_str})")
        except Exception as e:
            logger.warning(f"Analogies query failed: {e}")

        return analogies

    async def _predict_event(
        self,
        subject: str,
        patterns: List[TemporalPattern],
        horizon_days: int
    ) -> FuturePrediction:
        """Predict if/when an event will occur."""
        from uuid import uuid4

        predicted_date = None
        prediction_text = f"Event related to '{subject}'"

        # Use cyclic pattern to predict next occurrence
        cyclic = next((p for p in patterns if p.pattern_type == "cyclic"), None)
        if cyclic and cyclic.period_days:
            # Predict next occurrence based on period
            predicted_date = datetime.utcnow() + timedelta(days=cyclic.period_days)
            prediction_text = (
                f"'{subject}' is expected to occur again in approximately "
                f"{cyclic.period_days:.0f} days based on observed cyclic pattern."
            )

        # Use trend pattern
        trend = next((p for p in patterns if p.pattern_type == "trend"), None)
        if trend:
            if trend.trend_direction == "increasing":
                prediction_text += f" Frequency is increasing."
            elif trend.trend_direction == "decreasing":
                prediction_text += f" Frequency is decreasing."

        return FuturePrediction(
            prediction_id=str(uuid4()),
            prediction_type=PredictionType.EVENT,
            subject=subject,
            prediction=prediction_text,
            predicted_date=predicted_date,
        )

    async def _predict_trend(
        self,
        subject: str,
        patterns: List[TemporalPattern]
    ) -> FuturePrediction:
        """Predict trend direction."""
        from uuid import uuid4

        trend = next((p for p in patterns if p.pattern_type == "trend"), None)

        if trend:
            direction = trend.trend_direction or "unknown"
            prediction_text = (
                f"The trend for '{subject}' appears to be {direction} "
                f"with strength {trend.strength:.0%}."
            )
        else:
            prediction_text = f"No clear trend detected for '{subject}'."

        return FuturePrediction(
            prediction_id=str(uuid4()),
            prediction_type=PredictionType.TREND,
            subject=subject,
            prediction=prediction_text,
            predicted_outcome=trend.trend_direction if trend else None,
        )

    async def _predict_timing(
        self,
        subject: str,
        patterns: List[TemporalPattern],
        horizon_days: int
    ) -> FuturePrediction:
        """Predict timing of an event."""
        from uuid import uuid4

        cyclic = next((p for p in patterns if p.pattern_type == "cyclic"), None)

        if cyclic and cyclic.period_days:
            # Calculate probability of occurrence within horizon
            periods_in_horizon = horizon_days / cyclic.period_days
            probability = min(1.0, periods_in_horizon)

            predicted_date = datetime.utcnow() + timedelta(days=cyclic.period_days)
            date_range = (
                datetime.utcnow() + timedelta(days=cyclic.period_days * 0.8),
                datetime.utcnow() + timedelta(days=cyclic.period_days * 1.2),
            )

            prediction_text = (
                f"'{subject}' is predicted to occur around {predicted_date.strftime('%Y-%m-%d')} "
                f"(±{cyclic.period_days * 0.2:.0f} days) with {probability:.0%} probability "
                f"within the {horizon_days}-day horizon."
            )
        else:
            predicted_date = None
            date_range = None
            prediction_text = f"Insufficient data to predict timing for '{subject}'."

        return FuturePrediction(
            prediction_id=str(uuid4()),
            prediction_type=PredictionType.TEMPORAL,
            subject=subject,
            prediction=prediction_text,
            predicted_date=predicted_date,
            date_range=date_range,
        )

    async def _predict_causal(
        self,
        subject: str,
        causal_factors: List[str]
    ) -> FuturePrediction:
        """Predict causal consequences."""
        from uuid import uuid4

        if causal_factors:
            factors_str = ", ".join(causal_factors[:5])
            prediction_text = (
                f"'{subject}' is influenced by: {factors_str}. "
                f"Changes in these factors may affect future occurrences."
            )
        else:
            prediction_text = f"No clear causal factors identified for '{subject}'."

        return FuturePrediction(
            prediction_id=str(uuid4()),
            prediction_type=PredictionType.CAUSAL,
            subject=subject,
            prediction=prediction_text,
            causal_factors=causal_factors,
        )

    async def _predict_outcome(
        self,
        subject: str,
        patterns: List[TemporalPattern],
        causal_factors: List[str],
        context: Optional[str]
    ) -> FuturePrediction:
        """Predict outcome using LLM reasoning."""
        from uuid import uuid4

        if self.llm_client:
            pattern_str = "\n".join(
                f"- {p.pattern_type}: {p.trend_direction or p.period_days or 'N/A'} (strength: {p.strength:.0%})"
                for p in patterns
            )
            factors_str = "\n".join(f"- {f}" for f in causal_factors)

            prompt = f"""Based on the following patterns and causal factors, predict the likely outcome for "{subject}":

Temporal Patterns:
{pattern_str or 'None detected'}

Causal Factors:
{factors_str or 'None identified'}

{f'Additional Context: {context}' if context else ''}

Provide:
1. Most likely outcome (1-2 sentences)
2. Key assumptions
3. Potential risks/uncertainties"""

            try:
                if hasattr(self.llm_client, "generate"):
                    response = await self.llm_client.generate(prompt)

                    # Parse response
                    prediction_text = response.split("\n")[0] if response else f"Unable to predict outcome for '{subject}'"

                    return FuturePrediction(
                        prediction_id=str(uuid4()),
                        prediction_type=PredictionType.OUTCOME,
                        subject=subject,
                        prediction=prediction_text,
                        reasoning=response,
                    )
            except Exception as e:
                logger.warning(f"LLM prediction failed: {e}")

        # Fallback
        return FuturePrediction(
            prediction_id=str(uuid4()),
            prediction_type=PredictionType.OUTCOME,
            subject=subject,
            prediction=f"Outcome prediction for '{subject}' requires more data.",
        )

    def _calculate_confidence(
        self,
        patterns: List[TemporalPattern],
        causal_factors: List[str],
        analogies: List[str]
    ) -> float:
        """Calculate prediction confidence."""
        confidence = 0.1  # Base confidence

        # Pattern strength contributes
        if patterns:
            avg_pattern_strength = sum(p.strength for p in patterns) / len(patterns)
            confidence += avg_pattern_strength * 0.4

        # Causal factors contribute
        if causal_factors:
            confidence += min(0.2, len(causal_factors) * 0.04)

        # Historical analogies contribute
        if analogies:
            confidence += min(0.2, len(analogies) * 0.04)

        # Sample size bonus
        if patterns:
            max_samples = max(p.samples for p in patterns)
            if max_samples >= 10:
                confidence += 0.1
            elif max_samples >= 5:
                confidence += 0.05

        return min(1.0, confidence)

    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to level."""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.SPECULATIVE

    async def _generate_reasoning(
        self,
        subject: str,
        prediction: FuturePrediction,
        patterns: List[TemporalPattern],
        causal_factors: List[str],
        context: Optional[str]
    ) -> str:
        """Generate reasoning explanation."""
        reasoning_parts = [f"Prediction for '{subject}':"]

        # Pattern-based reasoning
        if patterns:
            reasoning_parts.append("\nTemporal patterns detected:")
            for p in patterns:
                if p.pattern_type == "cyclic":
                    reasoning_parts.append(f"- Cyclic pattern with ~{p.period_days:.0f} day period")
                elif p.pattern_type == "trend":
                    reasoning_parts.append(f"- {p.trend_direction.capitalize()} trend")

        # Causal reasoning
        if causal_factors:
            reasoning_parts.append("\nCausal factors considered:")
            for f in causal_factors[:3]:
                reasoning_parts.append(f"- {f}")

        # Confidence explanation
        reasoning_parts.append(
            f"\nConfidence: {prediction.confidence:.0%} ({prediction.confidence_level.value})"
        )

        return "\n".join(reasoning_parts)
