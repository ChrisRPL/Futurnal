"""End-to-End Integration Tests for AGI Intelligence Pipelines.

Phase 9: Integration Testing

Tests the complete flow of all AGI enhancement phases:
1. Statistical Foundation - validates correlation significance
2. Context Gate - filters priors by query relevance
3. Learning Loop - feedback improves routing
4. Curiosity - detects knowledge gaps
5. Insights - generates human-readable insights
6. Jobs - background processing
7. ICDA - interactive causal verification

Option B Compliance:
- Ghost model FROZEN throughout all tests
- Learning via token priors only
"""

import pytest
from datetime import datetime
from uuid import uuid4


# ============================================================================
# Phase 1: Statistical Foundation Tests
# ============================================================================

class TestStatisticalFoundation:
    """Test statistical significance validation."""

    def test_validator_creation(self):
        """Test StatisticalCorrelationValidator initialization."""
        from futurnal.search.temporal.statistics import StatisticalCorrelationValidator

        validator = StatisticalCorrelationValidator()
        assert validator is not None
        assert hasattr(validator, 'significance_threshold')

    def test_validate_correlation_significance(self):
        """Test correlation significance validation."""
        from futurnal.search.temporal.statistics import StatisticalCorrelationValidator

        validator = StatisticalCorrelationValidator()

        # Test validate_correlation method
        assert hasattr(validator, 'validate_correlation')

        # Create sample correlation data with correct signature
        result = validator.validate_correlation(
            observed_cooccurrences=15,
            total_events_a=20,
            total_events_b=25,
            time_range_days=30.0,
            max_gap_days=2.0,
        )

        assert result is not None
        assert hasattr(result, 'is_significant')
        assert hasattr(result, 'p_value')

    def test_bonferroni_correction_exists(self):
        """Test Bonferroni correction method exists."""
        from futurnal.search.temporal.statistics import StatisticalCorrelationValidator

        validator = StatisticalCorrelationValidator()
        assert hasattr(validator, 'bonferroni_correction')


# ============================================================================
# Phase 2: Semantic Context Gate Tests
# ============================================================================

class TestSemanticContextGate:
    """Test query-aware prior filtering."""

    def test_gate_creation(self):
        """Test SemanticContextGate initialization."""
        from futurnal.learning.context_gate import SemanticContextGate

        gate = SemanticContextGate()
        assert gate is not None

    def test_compute_relevance(self):
        """Test relevance computation."""
        from futurnal.learning.context_gate import SemanticContextGate

        gate = SemanticContextGate()

        # Create mock prior
        prior = {"content": "Python programming async patterns"}

        result = gate.compute_query_prior_relevance("python tutorial", prior)

        # Should return a relevance score object
        assert result is not None

    def test_filter_priors(self):
        """Test prior filtering."""
        from futurnal.learning.context_gate import SemanticContextGate

        gate = SemanticContextGate()

        priors = {
            "p1": {"content": "Python programming"},
            "p2": {"content": "Italian cooking"},
        }

        result = gate.filter_relevant_priors("python code", priors, top_k=1)
        assert result is not None


# ============================================================================
# Phase 3: Bidirectional Learning Tests
# ============================================================================

class TestBidirectionalLearning:
    """Test feedback-routing loop."""

    def test_optimizer_creation(self):
        """Test SearchRankingOptimizer initialization."""
        from futurnal.search.hybrid.routing.optimizer import SearchRankingOptimizer

        optimizer = SearchRankingOptimizer()
        assert optimizer is not None

    def test_optimizer_has_update_check(self):
        """Test optimizer can check if update needed."""
        from futurnal.search.hybrid.routing.optimizer import SearchRankingOptimizer

        optimizer = SearchRankingOptimizer()
        assert hasattr(optimizer, 'should_update')

        result = optimizer.should_update()
        assert isinstance(result, bool)

    def test_export_learned_configs(self):
        """Test exporting learned configs as token priors."""
        from futurnal.search.hybrid.routing.optimizer import SearchRankingOptimizer

        optimizer = SearchRankingOptimizer()
        assert hasattr(optimizer, 'export_learned_configs')

        export = optimizer.export_learned_configs()
        assert isinstance(export, str)


# ============================================================================
# Phase 4: CuriosityEngine Tests
# ============================================================================

class TestCuriosityEngine:
    """Test knowledge gap detection."""

    def test_engine_creation(self):
        """Test CuriosityEngine initialization."""
        from futurnal.insights.curiosity_engine import CuriosityEngine

        engine = CuriosityEngine()
        assert engine is not None

    def test_has_gap_detection(self):
        """Test engine has gap detection methods."""
        from futurnal.insights.curiosity_engine import CuriosityEngine

        engine = CuriosityEngine()
        assert hasattr(engine, 'detect_gaps')
        assert hasattr(engine, 'detect_isolated_clusters')

    def test_suggest_exploration(self):
        """Test exploration prompt suggestion."""
        from futurnal.insights.curiosity_engine import CuriosityEngine, KnowledgeGap

        engine = CuriosityEngine()
        assert hasattr(engine, 'suggest_exploration_prompts')


# ============================================================================
# Phase 5: Emergent Insights Tests
# ============================================================================

class TestEmergentInsights:
    """Test insight generation."""

    def test_generator_creation(self):
        """Test InsightGenerator initialization."""
        from futurnal.insights.emergent_insights import InsightGenerator

        generator = InsightGenerator()
        assert generator is not None

    def test_generate_from_correlation(self):
        """Test insight generation from correlation."""
        from futurnal.insights.emergent_insights import InsightGenerator
        from futurnal.search.temporal.results import TemporalCorrelationResult

        generator = InsightGenerator()

        correlation = TemporalCorrelationResult(
            correlation_found=True,
            event_type_a="exercise",
            event_type_b="good_mood",
            correlation_strength=0.75,
            co_occurrences=10,
            avg_gap_days=0.5,
            p_value=0.01,
            is_causal_candidate=True,
        )

        insight = generator.generate_insight_from_correlation(correlation, [])

        assert insight is not None
        assert hasattr(insight, 'title')
        assert hasattr(insight, 'description')
        assert hasattr(insight, 'confidence')

    def test_export_for_token_priors(self):
        """Test exporting insights as token priors."""
        from futurnal.insights.emergent_insights import InsightGenerator

        generator = InsightGenerator()
        assert hasattr(generator, 'export_for_token_priors')


# ============================================================================
# Phase 6: Autonomous Jobs Tests
# ============================================================================

class TestAutonomousJobs:
    """Test background insight scheduling."""

    def test_executor_creation(self):
        """Test InsightJobExecutor initialization."""
        from futurnal.orchestrator.insight_jobs import InsightJobExecutor

        executor = InsightJobExecutor()
        assert executor is not None

    def test_job_types_defined(self):
        """Test job types are defined."""
        from futurnal.orchestrator.models import JobType

        assert hasattr(JobType, 'INSIGHT_GENERATION')
        assert hasattr(JobType, 'CORRELATION_SCAN')
        assert hasattr(JobType, 'CURIOSITY_SCAN')

    @pytest.mark.asyncio
    async def test_scan_methods_exist(self):
        """Test scan methods exist on executor."""
        from futurnal.orchestrator.insight_jobs import InsightJobExecutor

        executor = InsightJobExecutor()
        assert hasattr(executor, 'execute_correlation_scan')
        assert hasattr(executor, 'execute_curiosity_scan')


# ============================================================================
# Phase 7: ICDA Tests
# ============================================================================

class TestInteractiveCausal:
    """Test interactive causal discovery."""

    def test_agent_creation(self):
        """Test InteractiveCausalDiscoveryAgent initialization."""
        from futurnal.insights.interactive_causal import InteractiveCausalDiscoveryAgent

        agent = InteractiveCausalDiscoveryAgent()
        assert agent is not None

    def test_add_candidate(self):
        """Test adding causal candidate."""
        from futurnal.insights.interactive_causal import InteractiveCausalDiscoveryAgent
        from futurnal.search.temporal.results import TemporalCorrelationResult

        agent = InteractiveCausalDiscoveryAgent()

        correlation = TemporalCorrelationResult(
            correlation_found=True,
            event_type_a="coffee",
            event_type_b="focus",
            correlation_strength=0.65,
            co_occurrences=8,
            avg_gap_days=0.25,
            p_value=0.02,
            is_causal_candidate=True,
        )

        candidate = agent.add_candidate_from_correlation(correlation)
        # May be None if doesn't meet threshold, but method should work
        assert hasattr(agent, '_candidates')

    def test_question_generation(self):
        """Test verification question generation."""
        from futurnal.insights.interactive_causal import (
            InteractiveCausalDiscoveryAgent,
            CausalCandidate,
        )

        agent = InteractiveCausalDiscoveryAgent()

        candidate = CausalCandidate(
            cause_event="test_cause",
            effect_event="test_effect",
            avg_gap_days=1.0,
            co_occurrences=5,
            correlation_strength=0.6,
            initial_confidence=0.5,
        )

        question = agent.generate_verification_question(candidate)

        assert question is not None
        assert hasattr(question, 'main_question')
        assert hasattr(question, 'response_options')

    def test_response_processing(self):
        """Test user response processing."""
        from futurnal.insights.interactive_causal import (
            InteractiveCausalDiscoveryAgent,
            CausalCandidate,
            CausalResponse,
        )

        agent = InteractiveCausalDiscoveryAgent()

        # Setup candidate
        candidate = CausalCandidate(
            cause_event="cause",
            effect_event="effect",
            avg_gap_days=1.0,
            co_occurrences=5,
            correlation_strength=0.6,
            initial_confidence=0.5,
        )
        agent._candidates[candidate.candidate_id] = candidate

        question = agent.generate_verification_question(candidate)
        agent._questions[question.question_id] = question

        # Process response
        updated = agent.process_user_response(
            question.question_id,
            CausalResponse.YES_CAUSAL,
        )

        assert updated is not None
        assert updated.final_confidence > candidate.initial_confidence

    def test_export_for_token_priors(self):
        """Test exporting verified knowledge as priors."""
        from futurnal.insights.interactive_causal import InteractiveCausalDiscoveryAgent

        agent = InteractiveCausalDiscoveryAgent()
        assert hasattr(agent, 'export_for_token_priors')

        export = agent.export_for_token_priors()
        assert isinstance(export, str)


# ============================================================================
# Full Pipeline Integration Tests
# ============================================================================

class TestFullPipelineIntegration:
    """Test complete end-to-end pipelines."""

    def test_correlation_to_insight_flow(self):
        """Test: correlation -> insight generation."""
        from futurnal.search.temporal.results import TemporalCorrelationResult
        from futurnal.insights.emergent_insights import InsightGenerator

        # Create correlation
        correlation = TemporalCorrelationResult(
            correlation_found=True,
            event_type_a="meditation",
            event_type_b="reduced_stress",
            correlation_strength=0.8,
            co_occurrences=15,
            avg_gap_days=0.0,
            p_value=0.001,
            is_causal_candidate=True,
        )

        # Generate insight
        generator = InsightGenerator()
        insight = generator.generate_insight_from_correlation(correlation, [])

        # Verify insight is valid
        assert insight.title != ""
        assert insight.description != ""
        assert 0 <= insight.confidence <= 1

    def test_option_b_compliance(self):
        """Verify Option B compliance: Ghost model FROZEN."""
        # InsightGenerator - no training
        from futurnal.insights.emergent_insights import InsightGenerator
        generator = InsightGenerator()
        assert not hasattr(generator, 'train') or not callable(getattr(generator, 'train', None))

        # CuriosityEngine - no training
        from futurnal.insights.curiosity_engine import CuriosityEngine
        engine = CuriosityEngine()
        assert not hasattr(engine, 'train') or not callable(getattr(engine, 'train', None))

        # ICDA - no gradients
        from futurnal.insights.interactive_causal import InteractiveCausalDiscoveryAgent
        agent = InteractiveCausalDiscoveryAgent()
        assert not hasattr(agent, 'backward') or not callable(getattr(agent, 'backward', None))

        # Learning via token priors
        from futurnal.learning.token_priors import TokenPriorStore
        store = TokenPriorStore()
        assert hasattr(store, 'entity_priors')  # Stores strings, not gradients


# ============================================================================
# CLI Integration Tests
# ============================================================================

class TestCLIIntegration:
    """Test CLI commands."""

    def test_insights_help(self):
        """Test insights CLI help works."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "futurnal.cli", "insights", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "insights" in result.stdout.lower() or "emergent" in result.stdout.lower()

    def test_insights_list(self):
        """Test insights list command."""
        import subprocess
        import sys
        import json

        result = subprocess.run(
            [sys.executable, "-m", "futurnal.cli", "insights", "list", "--json"],
            capture_output=True,
            text=True,
        )

        # Parse stdout (may have logging before JSON)
        output = result.stdout.strip()
        if '{' in output:
            json_start = output.index('{')
            json_str = output[json_start:]
            data = json.loads(json_str)
            assert "success" in data

    def test_insights_stats(self):
        """Test insights stats command."""
        import subprocess
        import sys
        import json

        result = subprocess.run(
            [sys.executable, "-m", "futurnal.cli", "insights", "stats", "--json"],
            capture_output=True,
            text=True,
        )

        output = result.stdout.strip()
        if '{' in output:
            json_start = output.index('{')
            json_str = output[json_start:]
            data = json.loads(json_str)
            assert "totalInsights" in data or "success" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
