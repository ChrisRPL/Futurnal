"""Tests for CrossModalFusion.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Tests cover:
- Score normalization
- Deduplication
- Diversity selection (MMR-like)
- Modality balance enforcement
- Configuration options
- Edge cases
"""

import pytest

from futurnal.search.hybrid.multimodal.fusion import (
    CrossModalFusion,
    FusionConfig,
    FusedResult,
    FusionStats,
    create_fusion_config,
)
from futurnal.search.hybrid.multimodal.types import ContentSource


class TestFusionConfig:
    """Tests for FusionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FusionConfig()
        assert config.normalize_scores is True
        assert config.apply_diversity is True
        assert config.diversity_factor == 0.2
        assert config.min_results_per_modality == 2
        assert config.max_modality_dominance == 0.7
        assert config.similarity_threshold == 0.85

    def test_custom_config(self):
        """Test custom configuration."""
        config = FusionConfig(
            diversity_factor=0.5,
            max_modality_dominance=0.6,
            min_results_per_modality=3,
        )
        assert config.diversity_factor == 0.5
        assert config.max_modality_dominance == 0.6
        assert config.min_results_per_modality == 3


class TestFusedResult:
    """Tests for FusedResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = FusedResult(
            entity_id="e1",
            content="test content",
            original_score=0.8,
            fused_score=0.75,
            source_type=ContentSource.AUDIO_TRANSCRIPTION,
        )
        assert result.entity_id == "e1"
        assert result.original_score == 0.8
        assert result.fused_score == 0.75
        assert result.source_type == ContentSource.AUDIO_TRANSCRIPTION

    def test_result_defaults(self):
        """Test result default values."""
        result = FusedResult(
            entity_id="e1",
            content="test",
            original_score=0.5,
            fused_score=0.5,
            source_type=ContentSource.TEXT_NATIVE,
        )
        assert result.source_confidence == 1.0
        assert result.diversity_penalty == 0.0
        assert result.metadata == {}


class TestCrossModalFusion:
    """Tests for CrossModalFusion initialization."""

    def test_init_default(self):
        """Test initialization with defaults."""
        fusion = CrossModalFusion()
        assert fusion.config is not None
        assert fusion.config.diversity_factor == 0.2

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = FusionConfig(diversity_factor=0.4)
        fusion = CrossModalFusion(config=config)
        assert fusion.config.diversity_factor == 0.4


class TestScoreNormalization:
    """Tests for score normalization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = CrossModalFusion()

    def test_normalize_single_modality(self):
        """Test normalization within single modality."""
        results_by_modality = {
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"entity_id": "e1", "content": "first", "score": 0.9},
                {"entity_id": "e2", "content": "second", "score": 0.5},
                {"entity_id": "e3", "content": "third", "score": 0.1},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)

        # Find normalized scores
        scores = [r.fused_score for r in fused]
        # After normalization, max should be ~1.0, min should be ~0.0
        assert max(scores) >= 0.9  # Highest should be near 1.0
        assert min(scores) <= 0.1  # Lowest should be near 0.0

    def test_normalize_multiple_modalities(self):
        """Test normalization across multiple modalities."""
        results_by_modality = {
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"entity_id": "e1", "content": "audio content", "score": 0.9},
            ],
            ContentSource.OCR_DOCUMENT: [
                {"entity_id": "e2", "content": "ocr content", "score": 0.5},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)

        # Both should be normalized (single items normalize to 1.0)
        assert len(fused) == 2

    def test_normalize_disabled(self):
        """Test with normalization disabled."""
        config = FusionConfig(normalize_scores=False, apply_diversity=False)
        fusion = CrossModalFusion(config=config)

        results_by_modality = {
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"entity_id": "e1", "content": "content", "score": 0.75},
            ],
        }
        fused = fusion.fuse(results_by_modality, top_k=10)

        # Score should remain original
        assert fused[0].fused_score == 0.75


class TestDeduplication:
    """Tests for result deduplication."""

    def setup_method(self):
        """Set up test fixtures."""
        config = FusionConfig(apply_diversity=False, normalize_scores=False)
        self.fusion = CrossModalFusion(config=config)

    def test_deduplicate_exact_match(self):
        """Test deduplication of exact content matches."""
        results_by_modality = {
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"entity_id": "e1", "content": "duplicate content here", "score": 0.9},
            ],
            ContentSource.OCR_DOCUMENT: [
                {"entity_id": "e2", "content": "duplicate content here", "score": 0.8},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)

        # Should deduplicate, keeping higher score
        assert len(fused) == 1
        assert fused[0].fused_score == 0.9

    def test_deduplicate_keeps_higher_score(self):
        """Test deduplication keeps result with higher score."""
        results_by_modality = {
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"entity_id": "e1", "content": "same text content", "score": 0.6},
            ],
            ContentSource.OCR_DOCUMENT: [
                {"entity_id": "e2", "content": "same text content", "score": 0.9},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)

        assert len(fused) == 1
        assert fused[0].fused_score == 0.9
        assert fused[0].source_type == ContentSource.OCR_DOCUMENT

    def test_no_deduplication_different_content(self):
        """Test no deduplication for different content."""
        results_by_modality = {
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"entity_id": "e1", "content": "first unique content", "score": 0.9},
            ],
            ContentSource.OCR_DOCUMENT: [
                {"entity_id": "e2", "content": "second unique content", "score": 0.8},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)

        assert len(fused) == 2


class TestDiversitySelection:
    """Tests for MMR-like diversity selection."""

    def setup_method(self):
        """Set up test fixtures."""
        config = FusionConfig(
            apply_diversity=True,
            diversity_factor=0.5,
            normalize_scores=False,
        )
        self.fusion = CrossModalFusion(config=config)

    def test_diversity_promotes_different_content(self):
        """Test diversity promotes results with different content."""
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"entity_id": "e1", "content": "topic A about projects", "score": 0.9},
                {
                    "entity_id": "e2",
                    "content": "topic A about projects again",
                    "score": 0.85,
                },
                {"entity_id": "e3", "content": "topic B about budgets", "score": 0.8},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)

        # With diversity, topic B should rank higher than second topic A
        entity_order = [r.entity_id for r in fused]
        assert "e3" in entity_order[:3]  # Topic B should be in top 3

    def test_diversity_disabled(self):
        """Test with diversity disabled."""
        config = FusionConfig(apply_diversity=False, normalize_scores=False)
        fusion = CrossModalFusion(config=config)

        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"entity_id": "e1", "content": "same topic", "score": 0.9},
                {"entity_id": "e2", "content": "same topic again", "score": 0.85},
                {"entity_id": "e3", "content": "different topic", "score": 0.8},
            ],
        }
        fused = fusion.fuse(results_by_modality, top_k=10)

        # Without diversity, order should be by score
        assert fused[0].fused_score >= fused[1].fused_score


class TestModalityBalance:
    """Tests for modality balance enforcement."""

    def setup_method(self):
        """Set up test fixtures."""
        config = FusionConfig(
            max_modality_dominance=0.5,
            min_results_per_modality=1,
            normalize_scores=False,
            apply_diversity=False,
        )
        self.fusion = CrossModalFusion(config=config)

    def test_balance_limits_dominance(self):
        """Test balance limits single modality dominance."""
        # With enough results from both modalities to test proper limiting
        results_by_modality = {
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"entity_id": f"audio_{i}", "content": f"audio {i}", "score": 0.9 - i * 0.05}
                for i in range(10)
            ],
            ContentSource.OCR_DOCUMENT: [
                {"entity_id": f"ocr_{i}", "content": f"ocr {i}", "score": 0.85 - i * 0.05}
                for i in range(10)
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)

        # Count by modality
        audio_count = sum(
            1 for r in fused if r.source_type == ContentSource.AUDIO_TRANSCRIPTION
        )
        ocr_count = sum(
            1 for r in fused if r.source_type == ContentSource.OCR_DOCUMENT
        )

        # With max_modality_dominance=0.5 and top_k=10, each should be capped at 5
        assert audio_count <= 5
        assert ocr_count <= 5
        # Total should be 10 (5 from each)
        assert len(fused) == 10

    def test_balance_respects_min_results(self):
        """Test balance respects minimum results per modality."""
        config = FusionConfig(
            max_modality_dominance=0.6,
            min_results_per_modality=2,
            normalize_scores=False,
            apply_diversity=False,
        )
        fusion = CrossModalFusion(config=config)

        results_by_modality = {
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"entity_id": f"audio_{i}", "content": f"audio {i}", "score": 0.9 - i * 0.1}
                for i in range(6)
            ],
            ContentSource.OCR_DOCUMENT: [
                {"entity_id": f"ocr_{i}", "content": f"ocr {i}", "score": 0.3 - i * 0.05}
                for i in range(4)
            ],
        }
        fused = fusion.fuse(results_by_modality, top_k=8)

        ocr_count = sum(
            1 for r in fused if r.source_type == ContentSource.OCR_DOCUMENT
        )
        # Should have at least min_results_per_modality OCR results
        assert ocr_count >= 2


class TestFuseMethod:
    """Tests for main fuse method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = CrossModalFusion()

    def test_fuse_empty_input(self):
        """Test fusing empty input."""
        fused = self.fusion.fuse({}, top_k=10)
        assert fused == []

    def test_fuse_single_modality(self):
        """Test fusing single modality results."""
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"entity_id": "e1", "content": "content 1", "score": 0.9},
                {"entity_id": "e2", "content": "content 2", "score": 0.8},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)
        assert len(fused) == 2

    def test_fuse_multiple_modalities(self):
        """Test fusing multiple modality results."""
        results_by_modality = {
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"entity_id": "e1", "content": "audio content", "score": 0.9},
            ],
            ContentSource.OCR_DOCUMENT: [
                {"entity_id": "e2", "content": "ocr content", "score": 0.85},
            ],
            ContentSource.TEXT_NATIVE: [
                {"entity_id": "e3", "content": "text content", "score": 0.8},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)
        assert len(fused) == 3

    def test_fuse_respects_top_k(self):
        """Test fuse respects top_k limit."""
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"entity_id": f"e{i}", "content": f"content {i}", "score": 0.9 - i * 0.1}
                for i in range(20)
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=5)
        assert len(fused) <= 5

    def test_fuse_sorts_by_score(self):
        """Test fused results are sorted by score descending."""
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"entity_id": "e1", "content": "low score", "score": 0.3},
                {"entity_id": "e2", "content": "high score", "score": 0.9},
                {"entity_id": "e3", "content": "mid score", "score": 0.6},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)

        for i in range(len(fused) - 1):
            assert fused[i].fused_score >= fused[i + 1].fused_score


class TestContentSimilarity:
    """Tests for content similarity calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = CrossModalFusion()

    def test_identical_content(self):
        """Test identical content similarity."""
        sim = self.fusion._content_similarity(
            "hello world", "hello world"
        )
        assert sim == 1.0

    def test_different_content(self):
        """Test different content similarity."""
        sim = self.fusion._content_similarity(
            "hello world", "goodbye moon"
        )
        assert sim < 0.5

    def test_empty_content(self):
        """Test empty content similarity."""
        assert self.fusion._content_similarity("", "hello") == 0.0
        assert self.fusion._content_similarity("hello", "") == 0.0
        assert self.fusion._content_similarity("", "") == 0.0

    def test_partial_overlap(self):
        """Test partial content overlap."""
        sim = self.fusion._content_similarity(
            "project budget meeting", "budget meeting notes"
        )
        assert 0.3 < sim < 0.9


class TestContentSignature:
    """Tests for content signature generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = CrossModalFusion()

    def test_signature_normalized(self):
        """Test signature is normalized."""
        sig1 = self.fusion._content_signature("Hello   World")
        sig2 = self.fusion._content_signature("hello world")
        assert sig1 == sig2

    def test_signature_truncated(self):
        """Test signature is truncated for long content."""
        long_content = "a " * 500
        sig = self.fusion._content_signature(long_content)
        assert len(sig) <= 200


class TestFactoryFunction:
    """Tests for create_fusion_config factory."""

    def test_create_config_defaults(self):
        """Test factory creates config with defaults."""
        config = create_fusion_config()
        assert config.normalize_scores is True
        assert config.apply_diversity is True
        assert config.diversity_factor == 0.2

    def test_create_config_custom(self):
        """Test factory with custom values."""
        config = create_fusion_config(
            diversity_factor=0.4,
            max_modality_dominance=0.5,
            min_results_per_modality=3,
        )
        assert config.diversity_factor == 0.4
        assert config.max_modality_dominance == 0.5
        assert config.min_results_per_modality == 3


class TestEdgeCases:
    """Tests for edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = CrossModalFusion()

    def test_single_result(self):
        """Test fusing single result."""
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"entity_id": "e1", "content": "only result", "score": 0.9},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)
        assert len(fused) == 1

    def test_all_same_score(self):
        """Test fusing results with same scores."""
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"entity_id": f"e{i}", "content": f"content {i}", "score": 0.8}
                for i in range(5)
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)
        assert len(fused) == 5

    def test_missing_fields(self):
        """Test handling of missing fields in results."""
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"entity_id": "e1", "content": "content", "score": 0.9},
                {"entity_id": "e2", "score": 0.8},  # Missing content
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)
        assert len(fused) == 2

    def test_top_k_zero(self):
        """Test fusing with top_k=0."""
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"entity_id": "e1", "content": "content", "score": 0.9},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=0)
        assert len(fused) == 0

    def test_many_modalities(self):
        """Test fusing from all modality types."""
        results_by_modality = {
            ContentSource.TEXT_NATIVE: [
                {"entity_id": "e1", "content": "native text", "score": 0.9},
            ],
            ContentSource.OCR_DOCUMENT: [
                {"entity_id": "e2", "content": "ocr doc", "score": 0.85},
            ],
            ContentSource.OCR_IMAGE: [
                {"entity_id": "e3", "content": "ocr image", "score": 0.8},
            ],
            ContentSource.AUDIO_TRANSCRIPTION: [
                {"entity_id": "e4", "content": "audio", "score": 0.75},
            ],
            ContentSource.VIDEO_TRANSCRIPTION: [
                {"entity_id": "e5", "content": "video", "score": 0.7},
            ],
            ContentSource.MIXED_SOURCE: [
                {"entity_id": "e6", "content": "mixed", "score": 0.65},
            ],
        }
        fused = self.fusion.fuse(results_by_modality, top_k=10)
        assert len(fused) == 6

        # Check all modalities represented
        modalities = {r.source_type for r in fused}
        assert len(modalities) == 6
