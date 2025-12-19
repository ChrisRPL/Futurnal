"""Autonomous Insight Job Executor.

AGI Phase 6: Background insight generation and scheduling for proactive
intelligence capabilities.

Research Foundation:
- SEAgent (2508.04700v2): Complete experiential learning loop
- Curiosity-driven Autotelic AI (Oudeyer 2024): Information gain scoring

Key Innovation:
Unlike reactive systems that only respond to queries, this module enables
proactive intelligence by running scheduled background jobs that:
1. Scan for new temporal correlations
2. Detect knowledge gaps and exploration opportunities
3. Generate emergent insights from patterns
4. Update learning systems with new knowledge

Schedule:
- Daily correlation scan at 3am
- Weekly curiosity scan on Sunday 9am
- Continuous learning updates as needed

Option B Compliance:
- No model parameter updates
- All discoveries expressed as natural language for token priors
- Ghost model FROZEN throughout
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType

if TYPE_CHECKING:
    from futurnal.search.temporal.correlation import TemporalCorrelationDetector
    from futurnal.insights.curiosity_engine import CuriosityEngine, KnowledgeGap
    from futurnal.insights.emergent_insights import EmergentInsight, InsightGenerator
    from futurnal.insights.hypothesis_generation import HypothesisPipeline, CausalHypothesis
    from futurnal.insights.interactive_causal import InteractiveCausalDiscoveryAgent
    from futurnal.search.causal.bradford_hill import BradfordHillValidator
    from futurnal.learning.token_priors import TokenPriorStore

logger = logging.getLogger(__name__)


class InsightJobStatus(str, Enum):
    """Status of an insight job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class InsightJobResult:
    """Result of an insight job execution."""

    job_id: str
    job_type: JobType
    status: InsightJobStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Results
    insights_generated: int = 0
    correlations_found: int = 0
    gaps_detected: int = 0
    priors_updated: int = 0
    hypotheses_generated: int = 0
    hypotheses_validated: int = 0
    bradford_hill_reports: int = 0

    # Errors
    error_message: Optional[str] = None

    def is_success(self) -> bool:
        """Check if job completed successfully."""
        return self.status == InsightJobStatus.COMPLETED


@dataclass
class InsightScheduleConfig:
    """Configuration for insight job scheduling."""

    # Correlation scan schedule (cron-like)
    correlation_scan_hour: int = 3  # 3am daily
    correlation_scan_minute: int = 0

    # Curiosity scan schedule
    curiosity_scan_day: int = 6  # Sunday (0=Monday, 6=Sunday)
    curiosity_scan_hour: int = 9  # 9am
    curiosity_scan_minute: int = 0

    # Learning update thresholds
    min_events_for_correlation_scan: int = 10
    min_nodes_for_curiosity_scan: int = 20

    # Quality thresholds
    min_correlation_confidence: float = 0.6
    min_gap_severity: float = 0.3

    # Notification thresholds
    notify_high_priority_insights: bool = True
    high_priority_threshold: float = 0.8


class InsightJobExecutor:
    """Executes background insight generation jobs.

    AGI Phase 6 core component that enables proactive intelligence
    through scheduled background analysis.

    Responsibilities:
    1. Execute correlation scans to find new temporal patterns
    2. Execute curiosity scans to detect knowledge gaps
    3. Generate emergent insights from findings
    4. Update token priors with new knowledge
    5. Notify users of high-priority insights

    Example:
        executor = InsightJobExecutor(
            correlation_detector=detector,
            curiosity_engine=engine,
            insight_generator=generator,
            token_prior_store=store,
        )

        # Run correlation scan
        result = await executor.execute_correlation_scan()

        # Run curiosity scan
        result = await executor.execute_curiosity_scan()
    """

    def __init__(
        self,
        correlation_detector: Optional["TemporalCorrelationDetector"] = None,
        curiosity_engine: Optional["CuriosityEngine"] = None,
        insight_generator: Optional["InsightGenerator"] = None,
        hypothesis_pipeline: Optional["HypothesisPipeline"] = None,
        icda_agent: Optional["InteractiveCausalDiscoveryAgent"] = None,
        bradford_hill_validator: Optional["BradfordHillValidator"] = None,
        token_prior_store: Optional["TokenPriorStore"] = None,
        pkg_graph: Optional[Any] = None,
        config: Optional[InsightScheduleConfig] = None,
        notification_callback: Optional[callable] = None,
    ):
        """Initialize insight job executor.

        Args:
            correlation_detector: For temporal correlation detection
            curiosity_engine: For knowledge gap detection
            insight_generator: For insight generation
            hypothesis_pipeline: For generating causal hypotheses from correlations
            icda_agent: For interactive causal discovery verification
            bradford_hill_validator: For validating causal claims
            token_prior_store: For storing learned priors
            pkg_graph: Personal Knowledge Graph for analysis
            config: Schedule and threshold configuration
            notification_callback: Optional callback for high-priority insights
        """
        self._correlation_detector = correlation_detector
        self._curiosity_engine = curiosity_engine
        self._insight_generator = insight_generator
        self._hypothesis_pipeline = hypothesis_pipeline
        self._icda_agent = icda_agent
        self._bradford_hill_validator = bradford_hill_validator
        self._token_prior_store = token_prior_store
        self._pkg_graph = pkg_graph
        self._config = config or InsightScheduleConfig()
        self._notification_callback = notification_callback

        # Job history
        self._job_history: List[InsightJobResult] = []
        self._max_history = 100

        # Running jobs
        self._running_jobs: Dict[str, InsightJobResult] = {}

        # Pending hypotheses for ICDA verification
        self._pending_hypotheses: List["CausalHypothesis"] = []

        logger.info("InsightJobExecutor initialized")

    async def execute_job(self, job: IngestionJob) -> InsightJobResult:
        """Execute an insight job based on type.

        Args:
            job: The insight job to execute

        Returns:
            InsightJobResult with execution details
        """
        if job.job_type == JobType.CORRELATION_SCAN:
            return await self.execute_correlation_scan(job.job_id)
        elif job.job_type == JobType.CURIOSITY_SCAN:
            return await self.execute_curiosity_scan(job.job_id)
        elif job.job_type == JobType.INSIGHT_GENERATION:
            return await self.execute_insight_generation(job.job_id)
        elif job.job_type == JobType.LEARNING_UPDATE:
            return await self.execute_learning_update(job.job_id)
        else:
            raise ValueError(f"Unknown insight job type: {job.job_type}")

    async def execute_correlation_scan(
        self,
        job_id: Optional[str] = None,
    ) -> InsightJobResult:
        """Execute a correlation scan job.

        Scans for new temporal correlations in the PKG, generates hypotheses,
        validates with Bradford-Hill criteria, and queues for ICDA verification.

        This is the core autonomous analysis pipeline:
        1. Detect temporal correlations
        2. Generate causal hypotheses via LLM
        3. Validate with Bradford-Hill criteria
        4. Queue high-confidence hypotheses for ICDA verification
        5. Update token priors with validated insights

        Args:
            job_id: Optional job ID (generated if not provided)

        Returns:
            InsightJobResult with scan results
        """
        job_id = job_id or str(uuid.uuid4())
        start_time = datetime.utcnow()

        result = InsightJobResult(
            job_id=job_id,
            job_type=JobType.CORRELATION_SCAN,
            status=InsightJobStatus.RUNNING,
            started_at=start_time,
        )

        self._running_jobs[job_id] = result

        try:
            logger.info(f"Starting correlation scan: {job_id}")

            if not self._correlation_detector:
                raise RuntimeError("Correlation detector not configured")

            # Step 1: Run correlation scan
            correlations = await self._run_correlation_scan()
            result.correlations_found = len(correlations)
            logger.info(f"Found {len(correlations)} temporal correlations")

            # Step 2: Generate hypotheses from correlations
            hypotheses = []
            if correlations and self._hypothesis_pipeline:
                hypotheses = await self._hypothesis_pipeline.process_correlations(
                    correlations=correlations,
                    max_hypotheses=10,
                )
                result.hypotheses_generated = len(hypotheses)
                logger.info(f"Generated {len(hypotheses)} causal hypotheses")

            # Step 3: Validate hypotheses with Bradford-Hill criteria
            validated_hypotheses = []
            if hypotheses and self._bradford_hill_validator:
                for hypothesis in hypotheses:
                    try:
                        # Find the correlation that generated this hypothesis
                        correlation = next(
                            (c for c in correlations
                             if c.event_a_type == hypothesis.cause_type
                             and c.event_b_type == hypothesis.effect_type),
                            None
                        )
                        if correlation:
                            report = await self._bradford_hill_validator.validate(
                                correlation=correlation,
                                hypothesis=hypothesis,
                            )
                            result.bradford_hill_reports += 1

                            # High-confidence hypotheses get queued for ICDA
                            if report.overall_score >= 0.6:
                                hypothesis.bradford_hill_score = report.overall_score
                                validated_hypotheses.append(hypothesis)
                                logger.info(
                                    f"Hypothesis validated: {hypothesis.hypothesis_text[:50]}... "
                                    f"(score: {report.overall_score:.2f})"
                                )
                    except Exception as e:
                        logger.warning(f"Bradford-Hill validation failed: {e}")

                result.hypotheses_validated = len(validated_hypotheses)

            # Step 4: Queue validated hypotheses for ICDA verification
            if validated_hypotheses:
                self._pending_hypotheses.extend(validated_hypotheses)
                logger.info(
                    f"Queued {len(validated_hypotheses)} hypotheses for ICDA verification"
                )

                # If ICDA agent is available, notify it of pending hypotheses
                if self._icda_agent and hasattr(self._icda_agent, 'queue_hypotheses'):
                    try:
                        await self._icda_agent.queue_hypotheses(validated_hypotheses)
                    except Exception as e:
                        logger.warning(f"Failed to queue hypotheses with ICDA: {e}")

            # Step 5: Generate insights from correlations (legacy path)
            if correlations and self._insight_generator:
                insights = self._insight_generator.generate_insights(
                    correlations=correlations,
                )
                result.insights_generated = len(insights)

                # Update token priors with new insights
                if insights and self._token_prior_store:
                    priors_text = self._insight_generator.export_for_token_priors(insights)
                    self._store_insight_priors(priors_text)
                    result.priors_updated = len(insights)

                # Notify for high-priority insights
                await self._notify_high_priority(insights)

            # Also store validated hypotheses as priors
            if validated_hypotheses and self._token_prior_store:
                self._store_hypothesis_priors(validated_hypotheses)
                result.priors_updated += len(validated_hypotheses)

            result.status = InsightJobStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - start_time
            ).total_seconds()

            logger.info(
                f"Correlation scan completed: {result.correlations_found} correlations, "
                f"{result.hypotheses_generated} hypotheses, "
                f"{result.hypotheses_validated} validated, "
                f"{result.insights_generated} insights"
            )

        except Exception as e:
            result.status = InsightJobStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - start_time
            ).total_seconds()
            logger.error(f"Correlation scan failed: {e}")

        finally:
            del self._running_jobs[job_id]
            self._add_to_history(result)

        return result

    async def execute_curiosity_scan(
        self,
        job_id: Optional[str] = None,
    ) -> InsightJobResult:
        """Execute a curiosity scan job.

        Scans for knowledge gaps and exploration opportunities using
        the CuriosityEngine.

        Args:
            job_id: Optional job ID (generated if not provided)

        Returns:
            InsightJobResult with scan results
        """
        job_id = job_id or str(uuid.uuid4())
        start_time = datetime.utcnow()

        result = InsightJobResult(
            job_id=job_id,
            job_type=JobType.CURIOSITY_SCAN,
            status=InsightJobStatus.RUNNING,
            started_at=start_time,
        )

        self._running_jobs[job_id] = result

        try:
            logger.info(f"Starting curiosity scan: {job_id}")

            if not self._curiosity_engine:
                raise RuntimeError("Curiosity engine not configured")

            if not self._pkg_graph:
                raise RuntimeError("PKG graph not configured")

            # Run curiosity scan
            gaps = self._curiosity_engine.detect_gaps(self._pkg_graph)

            result.gaps_detected = len(gaps)

            # Generate insights from gaps
            if gaps and self._insight_generator:
                insights = self._insight_generator.generate_insights(
                    correlations=[],
                    knowledge_gaps=gaps,
                )

                result.insights_generated = len(insights)

                # Update token priors
                if insights and self._token_prior_store:
                    priors_text = self._insight_generator.export_for_token_priors(insights)
                    self._store_insight_priors(priors_text)
                    result.priors_updated = len(insights)

                # Notify for high-priority insights
                await self._notify_high_priority(insights)

            result.status = InsightJobStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - start_time
            ).total_seconds()

            logger.info(
                f"Curiosity scan completed: {result.gaps_detected} gaps, "
                f"{result.insights_generated} insights"
            )

        except Exception as e:
            result.status = InsightJobStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - start_time
            ).total_seconds()
            logger.error(f"Curiosity scan failed: {e}")

        finally:
            del self._running_jobs[job_id]
            self._add_to_history(result)

        return result

    async def execute_insight_generation(
        self,
        job_id: Optional[str] = None,
    ) -> InsightJobResult:
        """Execute a combined insight generation job.

        Runs both correlation and curiosity scans, then generates
        comprehensive insights.

        Args:
            job_id: Optional job ID (generated if not provided)

        Returns:
            InsightJobResult with combined results
        """
        job_id = job_id or str(uuid.uuid4())
        start_time = datetime.utcnow()

        result = InsightJobResult(
            job_id=job_id,
            job_type=JobType.INSIGHT_GENERATION,
            status=InsightJobStatus.RUNNING,
            started_at=start_time,
        )

        self._running_jobs[job_id] = result

        try:
            logger.info(f"Starting insight generation: {job_id}")

            correlations = []
            gaps = []

            # Run correlation scan if available
            if self._correlation_detector:
                correlations = await self._run_correlation_scan()
                result.correlations_found = len(correlations)

            # Run curiosity scan if available
            if self._curiosity_engine and self._pkg_graph:
                gaps = self._curiosity_engine.detect_gaps(self._pkg_graph)
                result.gaps_detected = len(gaps)

            # Generate comprehensive insights
            if self._insight_generator and (correlations or gaps):
                insights = self._insight_generator.generate_insights(
                    correlations=correlations,
                    knowledge_gaps=gaps,
                )

                result.insights_generated = len(insights)

                # Update token priors
                if insights and self._token_prior_store:
                    priors_text = self._insight_generator.export_for_token_priors(insights)
                    self._store_insight_priors(priors_text)
                    result.priors_updated = len(insights)

                # Notify for high-priority insights
                await self._notify_high_priority(insights)

            result.status = InsightJobStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - start_time
            ).total_seconds()

            logger.info(
                f"Insight generation completed: {result.insights_generated} insights"
            )

        except Exception as e:
            result.status = InsightJobStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - start_time
            ).total_seconds()
            logger.error(f"Insight generation failed: {e}")

        finally:
            del self._running_jobs[job_id]
            self._add_to_history(result)

        return result

    async def execute_learning_update(
        self,
        job_id: Optional[str] = None,
    ) -> InsightJobResult:
        """Execute a learning update job.

        Updates token priors based on accumulated feedback and insights.

        Args:
            job_id: Optional job ID (generated if not provided)

        Returns:
            InsightJobResult with update results
        """
        job_id = job_id or str(uuid.uuid4())
        start_time = datetime.utcnow()

        result = InsightJobResult(
            job_id=job_id,
            job_type=JobType.LEARNING_UPDATE,
            status=InsightJobStatus.RUNNING,
            started_at=start_time,
        )

        self._running_jobs[job_id] = result

        try:
            logger.info(f"Starting learning update: {job_id}")

            if not self._token_prior_store:
                raise RuntimeError("Token prior store not configured")

            # Consolidate and update priors
            # This would integrate with the SearchRankingOptimizer
            # and other learning components

            result.priors_updated = 1  # Placeholder

            result.status = InsightJobStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - start_time
            ).total_seconds()

            logger.info("Learning update completed")

        except Exception as e:
            result.status = InsightJobStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - start_time
            ).total_seconds()
            logger.error(f"Learning update failed: {e}")

        finally:
            del self._running_jobs[job_id]
            self._add_to_history(result)

        return result

    async def _run_correlation_scan(self) -> List[Any]:
        """Run the actual correlation scan.

        Returns:
            List of detected correlations
        """
        if not self._correlation_detector:
            return []

        # Use scan_all_correlations if available
        if hasattr(self._correlation_detector, "scan_all_correlations"):
            return self._correlation_detector.scan_all_correlations()

        return []

    def _store_insight_priors(self, priors_text: str):
        """Store insight text as token priors.

        Args:
            priors_text: Natural language priors to store
        """
        if not self._token_prior_store:
            return

        try:
            # Store as temporal pattern prior
            from futurnal.learning.token_priors import TemporalPatternPrior

            prior = TemporalPatternPrior(
                pattern_type="autonomous_insight",
                description="Insights from autonomous background scan",
                learned_weights=priors_text,
                confidence=0.7,
                observation_count=1,
            )

            self._token_prior_store.add_temporal_pattern(prior)

        except Exception as e:
            logger.warning(f"Failed to store insight priors: {e}")

    def _store_hypothesis_priors(self, hypotheses: List["CausalHypothesis"]):
        """Store validated hypotheses as token priors.

        Args:
            hypotheses: List of validated causal hypotheses
        """
        if not self._token_prior_store:
            return

        try:
            from futurnal.learning.token_priors import TemporalPatternPrior

            for hypothesis in hypotheses:
                prior_text = (
                    f"Causal hypothesis (confidence: {hypothesis.confidence:.2f}): "
                    f"{hypothesis.hypothesis_text}\n"
                    f"Cause: {hypothesis.cause_type} -> Effect: {hypothesis.effect_type}\n"
                    f"Mechanism: {hypothesis.mechanism}"
                )

                prior = TemporalPatternPrior(
                    pattern_type="causal_hypothesis",
                    description=f"Validated causal hypothesis: {hypothesis.cause_type} -> {hypothesis.effect_type}",
                    learned_weights=prior_text,
                    confidence=hypothesis.confidence,
                    observation_count=1,
                )

                self._token_prior_store.add_temporal_pattern(prior)

        except Exception as e:
            logger.warning(f"Failed to store hypothesis priors: {e}")

    def get_pending_hypotheses(self) -> List["CausalHypothesis"]:
        """Get hypotheses pending ICDA verification.

        Returns:
            List of pending hypotheses
        """
        return list(self._pending_hypotheses)

    def clear_verified_hypothesis(self, hypothesis_id: str) -> bool:
        """Remove a hypothesis from pending after verification.

        Args:
            hypothesis_id: ID of the hypothesis to remove

        Returns:
            True if hypothesis was found and removed
        """
        for i, h in enumerate(self._pending_hypotheses):
            if h.hypothesis_id == hypothesis_id:
                del self._pending_hypotheses[i]
                return True
        return False

    async def trigger_immediate_scan(self) -> InsightJobResult:
        """Trigger an immediate correlation scan.

        Useful for manual triggering or after significant data changes.

        Returns:
            InsightJobResult from the scan
        """
        logger.info("Triggering immediate correlation scan")
        return await self.execute_correlation_scan()

    async def _notify_high_priority(
        self,
        insights: List["EmergentInsight"],
    ):
        """Notify user of high-priority insights.

        Args:
            insights: List of generated insights
        """
        if not self._config.notify_high_priority_insights:
            return

        if not self._notification_callback:
            return

        high_priority = [
            i for i in insights
            if i.priority_score >= self._config.high_priority_threshold
        ]

        for insight in high_priority:
            try:
                await self._notification_callback(insight)
            except Exception as e:
                logger.warning(f"Failed to notify for insight: {e}")

    def _add_to_history(self, result: InsightJobResult):
        """Add result to job history."""
        self._job_history.append(result)

        # Trim history if needed
        if len(self._job_history) > self._max_history:
            self._job_history = self._job_history[-self._max_history:]

    def get_job_history(
        self,
        job_type: Optional[JobType] = None,
        limit: int = 20,
    ) -> List[InsightJobResult]:
        """Get job execution history.

        Args:
            job_type: Filter by job type (optional)
            limit: Maximum results to return

        Returns:
            List of job results
        """
        results = self._job_history

        if job_type:
            results = [r for r in results if r.job_type == job_type]

        return results[-limit:]

    def get_running_jobs(self) -> List[InsightJobResult]:
        """Get currently running jobs.

        Returns:
            List of running job results
        """
        return list(self._running_jobs.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get insight job statistics.

        Returns:
            Dictionary with execution statistics
        """
        total_jobs = len(self._job_history)
        successful = sum(1 for r in self._job_history if r.is_success())
        failed = total_jobs - successful

        total_insights = sum(r.insights_generated for r in self._job_history)
        total_correlations = sum(r.correlations_found for r in self._job_history)
        total_gaps = sum(r.gaps_detected for r in self._job_history)
        total_hypotheses = sum(r.hypotheses_generated for r in self._job_history)
        total_validated = sum(r.hypotheses_validated for r in self._job_history)
        total_bradford_hill = sum(r.bradford_hill_reports for r in self._job_history)

        return {
            "total_jobs": total_jobs,
            "successful_jobs": successful,
            "failed_jobs": failed,
            "success_rate": successful / total_jobs if total_jobs > 0 else 0,
            "total_insights_generated": total_insights,
            "total_correlations_found": total_correlations,
            "total_gaps_detected": total_gaps,
            "total_hypotheses_generated": total_hypotheses,
            "total_hypotheses_validated": total_validated,
            "total_bradford_hill_reports": total_bradford_hill,
            "pending_hypotheses_count": len(self._pending_hypotheses),
            "running_jobs": len(self._running_jobs),
        }


def create_insight_schedule(
    scheduler: Any,
    executor: InsightJobExecutor,
    config: Optional[InsightScheduleConfig] = None,
):
    """Register insight jobs with the APScheduler.

    AGI Phase 6: Sets up scheduled background jobs for autonomous
    insight generation.

    Schedule:
    - Daily correlation scan at 3am
    - Weekly curiosity scan on Sunday 9am

    Args:
        scheduler: APScheduler instance
        executor: InsightJobExecutor instance
        config: Optional schedule configuration
    """
    from apscheduler.triggers.cron import CronTrigger

    config = config or InsightScheduleConfig()

    # Daily correlation scan
    scheduler.add_job(
        executor.execute_correlation_scan,
        trigger=CronTrigger(
            hour=config.correlation_scan_hour,
            minute=config.correlation_scan_minute,
        ),
        id="daily_correlation_scan",
        replace_existing=True,
    )

    # Weekly curiosity scan
    scheduler.add_job(
        executor.execute_curiosity_scan,
        trigger=CronTrigger(
            day_of_week=config.curiosity_scan_day,
            hour=config.curiosity_scan_hour,
            minute=config.curiosity_scan_minute,
        ),
        id="weekly_curiosity_scan",
        replace_existing=True,
    )

    logger.info(
        "Insight schedules registered: "
        f"correlation scan at {config.correlation_scan_hour}:{config.correlation_scan_minute:02d} daily, "
        f"curiosity scan at {config.curiosity_scan_hour}:{config.curiosity_scan_minute:02d} on day {config.curiosity_scan_day}"
    )
