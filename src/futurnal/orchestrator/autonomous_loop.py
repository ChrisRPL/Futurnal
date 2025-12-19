"""Autonomous Loop - The Brain's Heartbeat.

This module closes the autonomous learning loop by:
1. Listening for ingestion completion events
2. Triggering CuriosityEngine.detect_gaps() in the background
3. Running InsightGenerator on new data patterns
4. Updating token priors with discovered knowledge

Research Foundation:
- SEAgent (2508.04700v2): Complete experiential learning loop
- ICDA (2024): Interactive Causal Discovery Agent
- Curiosity-driven Autotelic AI (Oudeyer 2024): Information gain scoring
- DyMemR (2024): Memory decay and forgetting curves

This is the component that transforms Futurnal from a reactive system
into a proactive intelligence engine. Without this, the Ghost just sleeps.

Option B Compliance:
- Ghost model remains FROZEN
- All learning expressed as natural language token priors
- No gradient updates or model fine-tuning
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from weakref import WeakSet

if TYPE_CHECKING:
    from futurnal.insights.curiosity_engine import CuriosityEngine
    from futurnal.insights.emergent_insights import InsightGenerator
    from futurnal.insights.hypothesis_generation import HypothesisPipeline
    from futurnal.search.temporal.correlation import TemporalCorrelationDetector
    from futurnal.search.causal.bradford_hill import BradfordHillValidator
    from futurnal.learning.token_priors import TokenPriorStore
    from futurnal.orchestrator.insight_jobs import InsightJobExecutor

logger = logging.getLogger(__name__)


class AutonomousEventType(str, Enum):
    """Types of events in the autonomous loop."""

    # Ingestion events
    INGESTION_COMPLETED = "ingestion_completed"
    INGESTION_BATCH_COMPLETED = "ingestion_batch_completed"

    # Knowledge events
    NEW_ENTITIES_DISCOVERED = "new_entities_discovered"
    NEW_RELATIONSHIPS_FOUND = "new_relationships_found"

    # Insight events
    CORRELATION_DETECTED = "correlation_detected"
    GAP_IDENTIFIED = "gap_identified"
    INSIGHT_GENERATED = "insight_generated"

    # Causal discovery events
    HYPOTHESIS_GENERATED = "hypothesis_generated"
    HYPOTHESIS_VALIDATED = "hypothesis_validated"
    ICDA_VERIFICATION_PENDING = "icda_verification_pending"
    BRADFORD_HILL_COMPLETED = "bradford_hill_completed"

    # Learning events
    PRIOR_UPDATED = "prior_updated"
    FEEDBACK_RECEIVED = "feedback_received"


@dataclass
class AutonomousEvent:
    """Event in the autonomous loop."""

    event_type: AutonomousEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


EventHandler = Callable[[AutonomousEvent], None]
AsyncEventHandler = Callable[[AutonomousEvent], Any]


class AutonomousEventBus:
    """Event bus for autonomous loop communication.

    Enables loose coupling between ingestion, analysis, and learning
    components. When ingestion completes, interested listeners can
    respond without tight coupling.

    Example:
        bus = AutonomousEventBus()

        # Register handler
        bus.subscribe(
            AutonomousEventType.INGESTION_COMPLETED,
            on_ingestion_complete,
        )

        # Emit event
        bus.emit(AutonomousEvent(
            event_type=AutonomousEventType.INGESTION_COMPLETED,
            payload={"files_processed": 50},
        ))
    """

    def __init__(self):
        """Initialize event bus."""
        self._handlers: Dict[AutonomousEventType, List[EventHandler]] = {}
        self._async_handlers: Dict[AutonomousEventType, List[AsyncEventHandler]] = {}
        self._event_history: List[AutonomousEvent] = []
        self._max_history = 1000

    def subscribe(
        self,
        event_type: AutonomousEventType,
        handler: EventHandler,
    ) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Type of event to listen for
            handler: Callback function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def subscribe_async(
        self,
        event_type: AutonomousEventType,
        handler: AsyncEventHandler,
    ) -> None:
        """Subscribe async handler to an event type.

        Args:
            event_type: Type of event to listen for
            handler: Async callback function
        """
        if event_type not in self._async_handlers:
            self._async_handlers[event_type] = []
        self._async_handlers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: AutonomousEventType,
        handler: EventHandler,
    ) -> None:
        """Unsubscribe from an event type.

        Args:
            event_type: Type of event
            handler: Handler to remove
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    def emit(self, event: AutonomousEvent) -> None:
        """Emit an event to all subscribers.

        Args:
            event: Event to emit
        """
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Call sync handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

        # Schedule async handlers
        async_handlers = self._async_handlers.get(event.event_type, [])
        for handler in async_handlers:
            try:
                asyncio.create_task(handler(event))
            except Exception as e:
                logger.error(f"Async event handler error: {e}")

    async def emit_async(self, event: AutonomousEvent) -> None:
        """Emit an event and await all async handlers.

        Args:
            event: Event to emit
        """
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Call sync handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

        # Await async handlers
        async_handlers = self._async_handlers.get(event.event_type, [])
        tasks = []
        for handler in async_handlers:
            try:
                tasks.append(handler(event))
            except Exception as e:
                logger.error(f"Async event handler error: {e}")

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_recent_events(
        self,
        event_type: Optional[AutonomousEventType] = None,
        limit: int = 100,
    ) -> List[AutonomousEvent]:
        """Get recent events from history.

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum events to return

        Returns:
            List of recent events
        """
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]


@dataclass
class AutonomousLoopConfig:
    """Configuration for the autonomous loop."""

    # Enable/disable the loop
    enabled: bool = True

    # Batch size for triggering analysis
    min_ingestion_batch_size: int = 5

    # Cooldown between insight generations (seconds)
    insight_cooldown_seconds: int = 300  # 5 minutes

    # Debounce ingestion events (seconds)
    ingestion_debounce_seconds: int = 30

    # Maximum concurrent insight jobs
    max_concurrent_insight_jobs: int = 2

    # Scheduled job timing
    daily_correlation_hour: int = 3  # 3am
    weekly_curiosity_day: int = 6  # Sunday
    weekly_curiosity_hour: int = 9  # 9am


class AutonomousLoop:
    """The autonomous learning loop.

    This is the "brain's heartbeat" - the component that makes Futurnal
    proactive rather than purely reactive.

    The loop:
    1. Listens for ingestion completion events
    2. Accumulates changes until a threshold is met
    3. Triggers CuriosityEngine to find knowledge gaps
    4. Triggers InsightGenerator to discover patterns
    5. Updates TokenPriors with new knowledge
    6. Notifies the user of important discoveries

    All learning happens through natural language (Option B compliance).

    Example:
        loop = AutonomousLoop(
            event_bus=event_bus,
            insight_executor=executor,
            config=AutonomousLoopConfig(),
        )

        await loop.start()

        # When ingestion completes, the loop automatically triggers
        # analysis and learning without user intervention.
    """

    def __init__(
        self,
        event_bus: AutonomousEventBus,
        insight_executor: Optional["InsightJobExecutor"] = None,
        curiosity_engine: Optional["CuriosityEngine"] = None,
        insight_generator: Optional["InsightGenerator"] = None,
        hypothesis_pipeline: Optional["HypothesisPipeline"] = None,
        correlation_detector: Optional["TemporalCorrelationDetector"] = None,
        bradford_hill_validator: Optional["BradfordHillValidator"] = None,
        token_prior_store: Optional["TokenPriorStore"] = None,
        pkg_graph: Optional[Any] = None,
        config: Optional[AutonomousLoopConfig] = None,
        notification_callback: Optional[Callable] = None,
    ):
        """Initialize autonomous loop.

        Args:
            event_bus: Event bus for communication
            insight_executor: Executor for insight jobs
            curiosity_engine: CuriosityEngine for gap detection
            insight_generator: InsightGenerator for pattern discovery
            hypothesis_pipeline: Pipeline for causal hypothesis generation
            correlation_detector: Detector for temporal correlations
            bradford_hill_validator: Validator for causal claims
            token_prior_store: Store for learned priors
            pkg_graph: Personal Knowledge Graph
            config: Loop configuration
            notification_callback: Optional callback for user notifications
        """
        self._event_bus = event_bus
        self._insight_executor = insight_executor
        self._curiosity_engine = curiosity_engine
        self._insight_generator = insight_generator
        self._hypothesis_pipeline = hypothesis_pipeline
        self._correlation_detector = correlation_detector
        self._bradford_hill_validator = bradford_hill_validator
        self._token_prior_store = token_prior_store
        self._pkg_graph = pkg_graph
        self._config = config or AutonomousLoopConfig()
        self._notification_callback = notification_callback

        # State tracking
        self._running = False
        self._pending_ingestion_count = 0
        self._last_analysis_time: Optional[datetime] = None
        self._active_insight_jobs = 0

        # Background tasks
        self._scheduled_task: Optional[asyncio.Task] = None
        self._debounce_task: Optional[asyncio.Task] = None

        logger.info("AutonomousLoop initialized")

    async def start(self) -> None:
        """Start the autonomous loop.

        Registers event handlers and begins listening for ingestion events.
        """
        if self._running:
            return

        self._running = True

        # Subscribe to ingestion events
        self._event_bus.subscribe_async(
            AutonomousEventType.INGESTION_COMPLETED,
            self._on_ingestion_completed,
        )
        self._event_bus.subscribe_async(
            AutonomousEventType.INGESTION_BATCH_COMPLETED,
            self._on_batch_completed,
        )
        self._event_bus.subscribe_async(
            AutonomousEventType.FEEDBACK_RECEIVED,
            self._on_feedback_received,
        )

        # Start scheduled analysis task
        self._scheduled_task = asyncio.create_task(self._scheduled_analysis_loop())

        logger.info("AutonomousLoop started - The Ghost awakens")

    async def stop(self) -> None:
        """Stop the autonomous loop."""
        if not self._running:
            return

        self._running = False

        # Cancel scheduled task
        if self._scheduled_task:
            self._scheduled_task.cancel()
            try:
                await self._scheduled_task
            except asyncio.CancelledError:
                pass

        # Cancel debounce task
        if self._debounce_task:
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass

        logger.info("AutonomousLoop stopped")

    async def _on_ingestion_completed(self, event: AutonomousEvent) -> None:
        """Handle ingestion completion event.

        Args:
            event: The ingestion completed event
        """
        if not self._config.enabled:
            return

        files_processed = event.payload.get("files_processed", 0)
        self._pending_ingestion_count += files_processed

        logger.debug(
            f"Ingestion completed: {files_processed} files, "
            f"pending: {self._pending_ingestion_count}"
        )

        # Debounce analysis trigger
        await self._maybe_trigger_analysis()

    async def _on_batch_completed(self, event: AutonomousEvent) -> None:
        """Handle batch ingestion completion.

        Args:
            event: The batch completed event
        """
        if not self._config.enabled:
            return

        batch_size = event.payload.get("batch_size", 0)
        self._pending_ingestion_count += batch_size

        # Larger batch, more likely to trigger analysis
        if batch_size >= self._config.min_ingestion_batch_size:
            await self._trigger_analysis()

    async def _on_feedback_received(self, event: AutonomousEvent) -> None:
        """Handle user feedback event.

        Feedback drives immediate learning update.

        Args:
            event: The feedback event
        """
        if not self._config.enabled:
            return

        feedback_type = event.payload.get("feedback_type")
        target_id = event.payload.get("target_id")

        logger.info(f"Feedback received: {feedback_type} for {target_id}")

        # Immediate learning update on feedback
        if self._insight_executor:
            await self._insight_executor.execute_learning_update()

    async def _maybe_trigger_analysis(self) -> None:
        """Check if analysis should be triggered.

        Uses debouncing to avoid excessive analysis.
        """
        # Check cooldown
        if self._last_analysis_time:
            elapsed = (datetime.utcnow() - self._last_analysis_time).total_seconds()
            if elapsed < self._config.insight_cooldown_seconds:
                # Schedule delayed analysis if not already scheduled
                if not self._debounce_task or self._debounce_task.done():
                    delay = self._config.insight_cooldown_seconds - elapsed
                    self._debounce_task = asyncio.create_task(
                        self._delayed_analysis(delay)
                    )
                return

        # Check threshold
        if self._pending_ingestion_count >= self._config.min_ingestion_batch_size:
            await self._trigger_analysis()

    async def _delayed_analysis(self, delay: float) -> None:
        """Trigger analysis after a delay.

        Args:
            delay: Seconds to wait
        """
        try:
            await asyncio.sleep(delay)
            if self._pending_ingestion_count > 0:
                await self._trigger_analysis()
        except asyncio.CancelledError:
            pass

    async def _trigger_analysis(self) -> None:
        """Trigger the full analysis pipeline.

        This is the core of the autonomous loop:
        1. Run CuriosityEngine to detect gaps
        2. Run correlation detection for temporal patterns
        3. Generate hypotheses from correlations
        4. Validate with Bradford-Hill criteria
        5. Update TokenPriors with discoveries
        """
        # Check concurrent job limit
        if self._active_insight_jobs >= self._config.max_concurrent_insight_jobs:
            logger.debug("Max concurrent insight jobs reached, deferring")
            return

        self._active_insight_jobs += 1
        self._last_analysis_time = datetime.utcnow()
        processed_count = self._pending_ingestion_count
        self._pending_ingestion_count = 0

        logger.info(
            f"Autonomous analysis triggered: processing {processed_count} accumulated changes"
        )

        try:
            insights_generated = 0
            gaps_found = 0
            correlations_found = 0
            hypotheses_generated = 0
            hypotheses_validated = 0

            # Step 1: Run CuriosityEngine to find knowledge gaps
            gaps = []
            if self._curiosity_engine and self._pkg_graph:
                try:
                    gaps = self._curiosity_engine.detect_gaps(self._pkg_graph)
                    gaps_found = len(gaps)
                    logger.info(f"CuriosityEngine detected {gaps_found} knowledge gaps")

                    # Emit event for gap discovery
                    for gap in gaps[:5]:  # Limit notifications
                        self._event_bus.emit(AutonomousEvent(
                            event_type=AutonomousEventType.GAP_IDENTIFIED,
                            payload={"gap": gap.__dict__ if hasattr(gap, '__dict__') else str(gap)},
                        ))
                except Exception as e:
                    logger.error(f"CuriosityEngine error: {e}")

            # Step 2: Run correlation detection
            correlations = []
            if self._correlation_detector:
                try:
                    if hasattr(self._correlation_detector, 'scan_all_correlations'):
                        correlations = self._correlation_detector.scan_all_correlations()
                    correlations_found = len(correlations)
                    logger.info(f"Detected {correlations_found} temporal correlations")

                    # Emit correlation events
                    for corr in correlations[:5]:
                        self._event_bus.emit(AutonomousEvent(
                            event_type=AutonomousEventType.CORRELATION_DETECTED,
                            payload={
                                "event_a": corr.event_a_type if hasattr(corr, 'event_a_type') else str(corr),
                                "event_b": corr.event_b_type if hasattr(corr, 'event_b_type') else "",
                                "confidence": corr.confidence if hasattr(corr, 'confidence') else 0.0,
                            },
                        ))
                except Exception as e:
                    logger.error(f"Correlation detection error: {e}")

            # Step 3: Generate hypotheses from correlations
            hypotheses = []
            if correlations and self._hypothesis_pipeline:
                try:
                    hypotheses = await self._hypothesis_pipeline.process_correlations(
                        correlations=correlations,
                        max_hypotheses=10,
                    )
                    hypotheses_generated = len(hypotheses)
                    logger.info(f"Generated {hypotheses_generated} causal hypotheses")

                    # Emit hypothesis generation events
                    for hyp in hypotheses:
                        self._event_bus.emit(AutonomousEvent(
                            event_type=AutonomousEventType.HYPOTHESIS_GENERATED,
                            payload={
                                "hypothesis_id": hyp.hypothesis_id,
                                "cause": hyp.cause_type,
                                "effect": hyp.effect_type,
                                "confidence": hyp.confidence,
                            },
                        ))
                except Exception as e:
                    logger.error(f"Hypothesis generation error: {e}")

            # Step 4: Validate hypotheses with Bradford-Hill
            validated_hypotheses = []
            if hypotheses and self._bradford_hill_validator:
                for hyp in hypotheses:
                    try:
                        corr = next(
                            (c for c in correlations
                             if hasattr(c, 'event_a_type') and c.event_a_type == hyp.cause_type
                             and hasattr(c, 'event_b_type') and c.event_b_type == hyp.effect_type),
                            None
                        )
                        if corr:
                            report = await self._bradford_hill_validator.validate(
                                correlation=corr,
                                hypothesis=hyp,
                            )
                            self._event_bus.emit(AutonomousEvent(
                                event_type=AutonomousEventType.BRADFORD_HILL_COMPLETED,
                                payload={
                                    "hypothesis_id": hyp.hypothesis_id,
                                    "overall_score": report.overall_score,
                                    "verdict": report.verdict.value if hasattr(report.verdict, 'value') else str(report.verdict),
                                },
                            ))
                            if report.overall_score >= 0.6:
                                validated_hypotheses.append(hyp)
                                hyp.bradford_hill_score = report.overall_score
                    except Exception as e:
                        logger.warning(f"Bradford-Hill validation error: {e}")

                hypotheses_validated = len(validated_hypotheses)
                logger.info(f"Validated {hypotheses_validated} hypotheses with Bradford-Hill")

                # Emit ICDA pending event for validated hypotheses
                if validated_hypotheses:
                    self._event_bus.emit(AutonomousEvent(
                        event_type=AutonomousEventType.ICDA_VERIFICATION_PENDING,
                        payload={
                            "hypothesis_count": len(validated_hypotheses),
                            "hypotheses": [
                                {"id": h.hypothesis_id, "text": h.hypothesis_text[:100]}
                                for h in validated_hypotheses[:5]
                            ],
                        },
                    ))

            # Step 5: Run insight generation (via executor or directly)
            if self._insight_executor:
                try:
                    result = await self._insight_executor.execute_insight_generation()
                    insights_generated = result.insights_generated
                    # Use executor's correlation count if we didn't detect separately
                    if correlations_found == 0:
                        correlations_found = result.correlations_found

                    logger.info(
                        f"InsightGenerator produced {insights_generated} insights, "
                        f"{correlations_found} correlations"
                    )
                except Exception as e:
                    logger.error(f"InsightGenerator error: {e}")
            elif self._insight_generator:
                # Direct generation without executor
                try:
                    insights = self._insight_generator.generate_insights(
                        correlations=correlations,
                        knowledge_gaps=gaps,
                    )
                    insights_generated = len(insights)

                    # Store as token priors
                    if insights and self._token_prior_store:
                        priors_text = self._insight_generator.export_for_token_priors(insights)
                        self._store_priors(priors_text)

                except Exception as e:
                    logger.error(f"Direct insight generation error: {e}")

            # Store validated hypotheses as priors
            if validated_hypotheses and self._token_prior_store:
                for hyp in validated_hypotheses:
                    prior_text = f"Validated causal hypothesis: {hyp.hypothesis_text}"
                    self._store_priors(prior_text)

            # Emit insight generated events
            if insights_generated > 0 or hypotheses_generated > 0:
                self._event_bus.emit(AutonomousEvent(
                    event_type=AutonomousEventType.INSIGHT_GENERATED,
                    payload={
                        "insights_count": insights_generated,
                        "gaps_found": gaps_found,
                        "correlations_found": correlations_found,
                        "hypotheses_generated": hypotheses_generated,
                        "hypotheses_validated": hypotheses_validated,
                    },
                ))

            # Notify user of significant discoveries
            if self._notification_callback and (
                gaps_found > 3 or insights_generated > 2 or hypotheses_validated > 0
            ):
                try:
                    await self._notification_callback({
                        "type": "autonomous_discovery",
                        "insights": insights_generated,
                        "gaps": gaps_found,
                        "hypotheses": hypotheses_validated,
                        "message": (
                            f"Discovered {insights_generated} insights, "
                            f"{gaps_found} knowledge gaps, and "
                            f"{hypotheses_validated} validated causal hypotheses"
                        ),
                    })
                except Exception as e:
                    logger.warning(f"Notification callback error: {e}")

            logger.info(
                f"Autonomous analysis complete: "
                f"{insights_generated} insights, {gaps_found} gaps, "
                f"{correlations_found} correlations, {hypotheses_validated} validated hypotheses"
            )

        finally:
            self._active_insight_jobs -= 1

    def _store_priors(self, priors_text: str) -> None:
        """Store discovered knowledge as token priors.

        Args:
            priors_text: Natural language priors to store
        """
        if not self._token_prior_store:
            return

        try:
            from futurnal.learning.token_priors import TemporalPatternPrior

            prior = TemporalPatternPrior(
                pattern_type="autonomous_discovery",
                description="Autonomously discovered insight from background analysis",
                learned_weights=priors_text,
                confidence=0.7,
                observation_count=1,
            )

            self._token_prior_store.add_temporal_pattern(prior)

            # Emit prior updated event
            self._event_bus.emit(AutonomousEvent(
                event_type=AutonomousEventType.PRIOR_UPDATED,
                payload={"prior_type": "autonomous_discovery"},
            ))

        except Exception as e:
            logger.warning(f"Failed to store priors: {e}")

    async def _scheduled_analysis_loop(self) -> None:
        """Background loop for scheduled analysis.

        Runs daily correlation scans and weekly curiosity scans
        independent of ingestion events.
        """
        try:
            while self._running:
                now = datetime.utcnow()

                # Check for daily correlation scan (3am)
                if (
                    now.hour == self._config.daily_correlation_hour
                    and now.minute < 5
                ):
                    logger.info("Running scheduled daily correlation scan")
                    if self._insight_executor:
                        await self._insight_executor.execute_correlation_scan()

                # Check for weekly curiosity scan (Sunday 9am)
                if (
                    now.weekday() == self._config.weekly_curiosity_day
                    and now.hour == self._config.weekly_curiosity_hour
                    and now.minute < 5
                ):
                    logger.info("Running scheduled weekly curiosity scan")
                    if self._insight_executor:
                        await self._insight_executor.execute_curiosity_scan()

                # Sleep for 5 minutes between checks
                await asyncio.sleep(300)

        except asyncio.CancelledError:
            logger.debug("Scheduled analysis loop cancelled")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get current autonomous loop status.

        Returns:
            Dictionary with status information
        """
        return {
            "running": self._running,
            "enabled": self._config.enabled,
            "pending_ingestion_count": self._pending_ingestion_count,
            "last_analysis_time": (
                self._last_analysis_time.isoformat()
                if self._last_analysis_time
                else None
            ),
            "active_insight_jobs": self._active_insight_jobs,
            "components": {
                "curiosity_engine": self._curiosity_engine is not None,
                "insight_generator": self._insight_generator is not None,
                "hypothesis_pipeline": self._hypothesis_pipeline is not None,
                "correlation_detector": self._correlation_detector is not None,
                "bradford_hill_validator": self._bradford_hill_validator is not None,
                "token_prior_store": self._token_prior_store is not None,
                "pkg_graph": self._pkg_graph is not None,
            },
            "config": {
                "min_batch_size": self._config.min_ingestion_batch_size,
                "cooldown_seconds": self._config.insight_cooldown_seconds,
                "daily_correlation_hour": self._config.daily_correlation_hour,
                "weekly_curiosity_day": self._config.weekly_curiosity_day,
            },
        }

    async def trigger_manual_scan(self) -> Dict[str, Any]:
        """Trigger a manual analysis scan.

        Useful for testing or after significant data changes.

        Returns:
            Dictionary with scan results summary
        """
        logger.info("Manual scan triggered")

        # Force trigger analysis regardless of pending count
        original_count = self._pending_ingestion_count
        self._pending_ingestion_count = self._config.min_ingestion_batch_size

        await self._trigger_analysis()

        # Restore count if nothing was actually processed
        if original_count > 0:
            self._pending_ingestion_count = original_count

        return {
            "triggered": True,
            "last_analysis_time": (
                self._last_analysis_time.isoformat()
                if self._last_analysis_time
                else None
            ),
        }


def create_autonomous_loop(
    orchestrator: Any,
    curiosity_engine: Optional["CuriosityEngine"] = None,
    insight_generator: Optional["InsightGenerator"] = None,
    insight_executor: Optional["InsightJobExecutor"] = None,
    hypothesis_pipeline: Optional["HypothesisPipeline"] = None,
    correlation_detector: Optional["TemporalCorrelationDetector"] = None,
    bradford_hill_validator: Optional["BradfordHillValidator"] = None,
    token_prior_store: Optional["TokenPriorStore"] = None,
    pkg_graph: Optional[Any] = None,
    config: Optional[AutonomousLoopConfig] = None,
    notification_callback: Optional[Callable] = None,
) -> tuple[AutonomousEventBus, AutonomousLoop]:
    """Factory function to create and wire up the autonomous loop.

    This is the recommended way to initialize the autonomous loop,
    as it handles all the wiring between components.

    Args:
        orchestrator: IngestionOrchestrator to integrate with
        curiosity_engine: CuriosityEngine instance
        insight_generator: InsightGenerator instance
        insight_executor: InsightJobExecutor instance
        hypothesis_pipeline: HypothesisPipeline for causal hypothesis generation
        correlation_detector: TemporalCorrelationDetector for pattern detection
        bradford_hill_validator: BradfordHillValidator for causal validation
        token_prior_store: TokenPriorStore instance
        pkg_graph: Personal Knowledge Graph
        config: Loop configuration
        notification_callback: Optional callback for user notifications

    Returns:
        Tuple of (AutonomousEventBus, AutonomousLoop)

    Example:
        event_bus, loop = create_autonomous_loop(
            orchestrator=orchestrator,
            curiosity_engine=curiosity_engine,
            insight_generator=insight_generator,
            hypothesis_pipeline=hypothesis_pipeline,
            correlation_detector=correlation_detector,
            bradford_hill_validator=bradford_hill_validator,
        )

        await loop.start()
    """
    event_bus = AutonomousEventBus()

    loop = AutonomousLoop(
        event_bus=event_bus,
        insight_executor=insight_executor,
        curiosity_engine=curiosity_engine,
        insight_generator=insight_generator,
        hypothesis_pipeline=hypothesis_pipeline,
        correlation_detector=correlation_detector,
        bradford_hill_validator=bradford_hill_validator,
        token_prior_store=token_prior_store,
        pkg_graph=pkg_graph,
        config=config,
        notification_callback=notification_callback,
    )

    logger.info(
        "Autonomous loop created and wired to orchestrator - "
        "The Ghost now has a heartbeat (with causal discovery enabled)"
    )

    return event_bus, loop
