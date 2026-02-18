"""Tests for CausalBoundary abstraction layer.

Phase 2.5: Research Integration Sprint

Tests verify:
1. Explicit separation between correlation and causal confidence
2. Bradford-Hill validation integration
3. Option B compliance (no model updates)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from futurnal.reasoning.causal_boundary import (
    CausalBoundary,
    CausalBoundaryResult,
    ConfidenceType,
    create_causal_boundary,
)


class TestConfidenceType:
    """Tests for ConfidenceType enum."""

    def test_confidence_types_exist(self):
        """Verify all confidence types are defined."""
        assert ConfidenceType.CORRELATION == "correlation"
        assert ConfidenceType.CAUSAL == "causal"
        assert ConfidenceType.VERIFIED == "verified"


class TestCausalBoundaryResult:
    """Tests for CausalBoundaryResult dataclass."""

    def test_default_creation(self):
        """Test default result creation."""
        result = CausalBoundaryResult()
        assert result.correlation_confidence == 0.0
        assert result.causal_confidence == 0.0
        assert result.verified_confidence is None
        assert result.verification_status == "pending"

    def test_get_display_confidence_verified_takes_precedence(self):
        """Verified confidence takes precedence over all."""
        result = CausalBoundaryResult(
            correlation_confidence=0.8,
            causal_confidence=0.6,
            verified_confidence=0.9,
        )
        conf, conf_type = result.get_display_confidence()
        assert conf == 0.9
        assert conf_type == ConfidenceType.VERIFIED

    def test_get_display_confidence_causal_over_correlation(self):
        """Causal confidence takes precedence over correlation."""
        result = CausalBoundaryResult(
            correlation_confidence=0.8,
            causal_confidence=0.6,
        )
        conf, conf_type = result.get_display_confidence()
        assert conf == 0.6
        assert conf_type == ConfidenceType.CAUSAL

    def test_get_display_confidence_correlation_fallback(self):
        """Falls back to correlation if no causal validation."""
        result = CausalBoundaryResult(
            correlation_confidence=0.8,
            causal_confidence=0.0,
        )
        conf, conf_type = result.get_display_confidence()
        assert conf == 0.8
        assert conf_type == ConfidenceType.CORRELATION

    def test_to_natural_language(self):
        """Test natural language export."""
        result = CausalBoundaryResult(
            cause_event="meeting",
            effect_event="decision",
            correlation_confidence=0.7,
            causal_confidence=0.5,
            hypothesis_statement="Meetings lead to decisions",
        )
        nl = result.to_natural_language()

        assert "meeting -> decision" in nl
        assert "70%" in nl  # correlation confidence
        assert "50%" in nl  # causal confidence
        assert "Meetings lead to decisions" in nl

    def test_to_dict(self):
        """Test dictionary serialization."""
        result = CausalBoundaryResult(
            cause_event="meeting",
            effect_event="decision",
            correlation_confidence=0.7,
            causal_confidence=0.5,
        )
        d = result.to_dict()

        assert d["cause_event"] == "meeting"
        assert d["effect_event"] == "decision"
        assert d["correlation_confidence"] == 0.7
        assert d["causal_confidence"] == 0.5


class TestCausalBoundary:
    """Tests for CausalBoundary class."""

    @pytest.fixture
    def mock_correlation(self):
        """Create mock TemporalCorrelationResult."""
        corr = MagicMock()
        corr.event_type_a = "meeting"
        corr.event_type_b = "decision"
        corr.correlation_strength = 0.7
        corr.p_value = 0.03
        corr.effect_size = 1.5
        corr.co_occurrences = 10
        return corr

    @pytest.fixture
    def boundary(self):
        """Create CausalBoundary without validators."""
        return CausalBoundary()

    @pytest.mark.asyncio
    async def test_process_correlation_basic(self, boundary, mock_correlation):
        """Test basic correlation processing."""
        result = await boundary.process_correlation(mock_correlation)

        assert result.cause_event == "meeting"
        assert result.effect_event == "decision"
        assert result.correlation_confidence > 0
        assert result.processing_time_ms is not None

    @pytest.mark.asyncio
    async def test_causal_confidence_discounted_without_validator(self, boundary, mock_correlation):
        """Causal confidence should be discounted from correlation without validator."""
        result = await boundary.process_correlation(mock_correlation)

        # Without Bradford-Hill validator, causal = correlation * 0.4
        assert result.causal_confidence <= result.correlation_confidence
        assert result.causal_confidence == pytest.approx(
            result.correlation_confidence * 0.4, rel=0.1
        )
        assert result.bradford_hill_verdict == "not_validated"

    @pytest.mark.asyncio
    async def test_correlation_evidence_collected(self, boundary, mock_correlation):
        """Test that correlation evidence is collected."""
        result = await boundary.process_correlation(mock_correlation)

        assert len(result.correlation_evidence) > 0
        evidence_str = " ".join(result.correlation_evidence)
        assert "p-value" in evidence_str
        assert "Effect size" in evidence_str

    @pytest.mark.asyncio
    async def test_result_cached(self, boundary, mock_correlation):
        """Test that results are cached."""
        result1 = await boundary.process_correlation(mock_correlation)
        cached = boundary.get_cached_result("meeting", "decision")

        assert cached is not None
        assert cached.result_id == result1.result_id

    @pytest.mark.asyncio
    async def test_statistics_updated(self, boundary, mock_correlation):
        """Test that statistics are updated."""
        await boundary.process_correlation(mock_correlation)
        stats = boundary.get_statistics()

        assert stats["correlations_processed"] == 1
        assert stats["cached_results"] == 1

    def test_export_for_token_priors(self, boundary):
        """Test export as natural language."""
        boundary._results_cache["test|result"] = CausalBoundaryResult(
            cause_event="test",
            effect_event="result",
            correlation_confidence=0.7,
            causal_confidence=0.5,
        )

        export = boundary.export_for_token_priors()

        assert "test -> result" in export
        assert "correlation" in export.lower()
        assert "causal" in export.lower()

    def test_clear_cache(self, boundary):
        """Test cache clearing."""
        boundary._results_cache["key"] = CausalBoundaryResult()
        assert len(boundary._results_cache) == 1

        boundary.clear_cache()
        assert len(boundary._results_cache) == 0


class TestCausalBoundaryWithValidators:
    """Tests for CausalBoundary with mock validators."""

    @pytest.fixture
    def mock_correlation(self):
        """Create mock TemporalCorrelationResult."""
        corr = MagicMock()
        corr.event_type_a = "meeting"
        corr.event_type_b = "decision"
        corr.correlation_strength = 0.7
        corr.p_value = 0.03
        corr.effect_size = 1.5
        corr.co_occurrences = 10
        return corr

    @pytest.fixture
    def mock_hypothesis_generator(self):
        """Create mock HypothesisGenerator."""
        generator = MagicMock()
        hypothesis = MagicMock()
        hypothesis.overall_confidence = 0.8
        hypothesis.hypothesis_statement = "Meetings cause decisions"
        hypothesis.mechanism_description = "Discussion leads to resolution"
        generator.generate_hypothesis = AsyncMock(return_value=hypothesis)
        return generator

    @pytest.fixture
    def mock_bradford_hill_validator(self):
        """Create mock BradfordHillValidator."""
        validator = MagicMock()
        report = MagicMock()
        report.overall_score = 0.7
        report.verdict = MagicMock()
        report.verdict.value = "possibly_causal"
        report.summary = "Moderate evidence for causality"
        report.recommendations = ["Gather more data"]
        validator.validate = AsyncMock(return_value=report)
        return validator

    @pytest.mark.asyncio
    async def test_with_hypothesis_generator(self, mock_correlation, mock_hypothesis_generator):
        """Test integration with hypothesis generator."""
        boundary = CausalBoundary(hypothesis_generator=mock_hypothesis_generator)
        result = await boundary.process_correlation(mock_correlation)

        assert result.hypothesis_statement == "Meetings cause decisions"
        assert result.mechanism_description == "Discussion leads to resolution"
        assert result.correlation_confidence >= 0.7  # Uses hypothesis confidence

    @pytest.mark.asyncio
    async def test_with_bradford_hill_validator(
        self, mock_correlation, mock_bradford_hill_validator
    ):
        """Test integration with Bradford-Hill validator."""
        boundary = CausalBoundary(bradford_hill_validator=mock_bradford_hill_validator)
        result = await boundary.process_correlation(mock_correlation)

        assert result.bradford_hill_score == 0.7
        assert result.bradford_hill_verdict == "possibly_causal"
        assert result.causal_confidence == pytest.approx(0.7, rel=0.1)


class TestCreateCausalBoundary:
    """Tests for create_causal_boundary factory function."""

    def test_create_with_no_args(self):
        """Test factory function with no arguments."""
        # This may fail if dependencies aren't available, which is expected
        try:
            boundary = create_causal_boundary()
            assert isinstance(boundary, CausalBoundary)
        except ImportError:
            pytest.skip("Dependencies not available")

    def test_create_returns_causal_boundary(self):
        """Test that factory returns CausalBoundary instance."""
        boundary = create_causal_boundary(llm_client=None)
        assert isinstance(boundary, CausalBoundary)
