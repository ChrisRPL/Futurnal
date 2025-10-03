"""Tests for aspirational data models."""

from datetime import datetime

import pytest

from futurnal.models.aspirational import (
    Aspiration,
    AspirationAlignment,
    AspirationCategory,
)


class TestAspiration:
    """Test Aspiration model."""

    def test_create_aspiration_with_defaults(self):
        """Test creating an aspiration with minimal fields."""
        aspiration = Aspiration(
            title="Learn Causal Inference"
        )

        assert aspiration.aspiration_id is not None
        assert aspiration.category == AspirationCategory.OUTCOME
        assert aspiration.title == "Learn Causal Inference"
        assert aspiration.description == ""
        assert aspiration.priority == 5
        assert isinstance(aspiration.created_at, datetime)
        assert aspiration.target_date is None
        assert aspiration.progress_metrics == {}
        assert aspiration.related_events == []
        assert aspiration.alignment_score == 0.0

    def test_create_aspiration_with_full_data(self):
        """Test creating an aspiration with all fields."""
        target = datetime(2025, 12, 31)
        created = datetime(2025, 1, 28)
        metrics = {"books_read": 3, "papers_reviewed": 15}

        aspiration = Aspiration(
            aspiration_id="test-asp-123",
            category=AspirationCategory.SKILL,
            title="Develop Causal Inference Expertise",
            description="Master causal reasoning frameworks",
            priority=8,
            created_at=created,
            target_date=target,
            progress_metrics=metrics,
            related_events=["event-1", "event-2"],
            alignment_score=0.65
        )

        assert aspiration.aspiration_id == "test-asp-123"
        assert aspiration.category == AspirationCategory.SKILL
        assert aspiration.title == "Develop Causal Inference Expertise"
        assert aspiration.description == "Master causal reasoning frameworks"
        assert aspiration.priority == 8
        assert aspiration.created_at == created
        assert aspiration.target_date == target
        assert aspiration.progress_metrics == metrics
        assert aspiration.related_events == ["event-1", "event-2"]
        assert aspiration.alignment_score == 0.65

    def test_aspiration_requires_title(self):
        """Test that title is required."""
        with pytest.raises(ValueError, match="title is required"):
            Aspiration(title="")

    def test_aspiration_validates_priority_range(self):
        """Test that priority must be between 1 and 10."""
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Aspiration(title="Test", priority=0)

        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Aspiration(title="Test", priority=11)

        # Valid priorities should work
        asp1 = Aspiration(title="Test", priority=1)
        asp10 = Aspiration(title="Test", priority=10)
        assert asp1.priority == 1
        assert asp10.priority == 10

    def test_aspiration_validates_alignment_score_range(self):
        """Test that alignment_score must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="alignment_score must be between 0.0 and 1.0"):
            Aspiration(title="Test", alignment_score=-0.1)

        with pytest.raises(ValueError, match="alignment_score must be between 0.0 and 1.0"):
            Aspiration(title="Test", alignment_score=1.1)

        # Valid scores should work
        asp0 = Aspiration(title="Test", alignment_score=0.0)
        asp1 = Aspiration(title="Test", alignment_score=1.0)
        assert asp0.alignment_score == 0.0
        assert asp1.alignment_score == 1.0

    def test_aspiration_category_enum(self):
        """Test all aspiration categories."""
        for category in AspirationCategory:
            aspiration = Aspiration(
                title=f"Test {category.value}",
                category=category
            )
            assert aspiration.category == category

    def test_aspiration_to_dict(self):
        """Test converting aspiration to dictionary."""
        aspiration = Aspiration(
            title="Lead High-Impact Projects",
            category=AspirationCategory.OUTCOME,
            priority=9
        )

        asp_dict = aspiration.to_dict()

        assert "aspiration_id" in asp_dict
        assert asp_dict["category"] == "outcome"
        assert asp_dict["title"] == "Lead High-Impact Projects"
        assert asp_dict["priority"] == 9
        assert isinstance(asp_dict["created_at"], str)

    def test_aspiration_from_dict(self):
        """Test creating aspiration from dictionary."""
        data = {
            "aspiration_id": "test-123",
            "category": "skill",
            "title": "Learn Python",
            "description": "Become proficient in Python",
            "priority": 7,
            "created_at": "2025-01-28T10:00:00",
            "target_date": "2025-12-31T23:59:59",
            "progress_metrics": {"courses_completed": 2},
            "related_events": ["event-1"],
            "alignment_score": 0.5
        }

        aspiration = Aspiration.from_dict(data)

        assert aspiration.aspiration_id == "test-123"
        assert aspiration.category == AspirationCategory.SKILL
        assert aspiration.title == "Learn Python"
        assert aspiration.priority == 7
        assert isinstance(aspiration.created_at, datetime)
        assert isinstance(aspiration.target_date, datetime)


class TestAspirationAlignment:
    """Test AspirationAlignment model."""

    def test_create_alignment_with_defaults(self):
        """Test creating an alignment with minimal fields."""
        alignment = AspirationAlignment(
            aspiration_id="asp-123",
            event_id="event-456"
        )

        assert alignment.alignment_id is not None
        assert alignment.aspiration_id == "asp-123"
        assert alignment.event_id == "event-456"
        assert alignment.alignment_type == "supports"
        assert alignment.contribution_score == 0.0
        assert isinstance(alignment.detected_at, datetime)
        assert alignment.confidence == 0.0
        assert alignment.explanation == ""

    def test_create_alignment_with_full_data(self):
        """Test creating an alignment with all fields."""
        detected = datetime(2025, 1, 28, 10, 30, 0)

        alignment = AspirationAlignment(
            alignment_id="align-789",
            aspiration_id="asp-123",
            event_id="event-456",
            alignment_type="opposes",
            contribution_score=-0.5,
            detected_at=detected,
            confidence=0.85,
            explanation="Reading unrelated to stated goal"
        )

        assert alignment.alignment_id == "align-789"
        assert alignment.aspiration_id == "asp-123"
        assert alignment.event_id == "event-456"
        assert alignment.alignment_type == "opposes"
        assert alignment.contribution_score == -0.5
        assert alignment.detected_at == detected
        assert alignment.confidence == 0.85
        assert alignment.explanation == "Reading unrelated to stated goal"

    def test_alignment_requires_aspiration_and_event(self):
        """Test that aspiration_id and event_id are required."""
        with pytest.raises(ValueError, match="aspiration_id is required"):
            AspirationAlignment(
                aspiration_id="",
                event_id="event-123"
            )

        with pytest.raises(ValueError, match="event_id is required"):
            AspirationAlignment(
                aspiration_id="asp-123",
                event_id=""
            )

    def test_alignment_validates_contribution_score_range(self):
        """Test that contribution_score must be between -1.0 and 1.0."""
        with pytest.raises(ValueError, match="contribution_score must be between -1.0 and 1.0"):
            AspirationAlignment(
                aspiration_id="asp-123",
                event_id="event-456",
                contribution_score=-1.1
            )

        with pytest.raises(ValueError, match="contribution_score must be between -1.0 and 1.0"):
            AspirationAlignment(
                aspiration_id="asp-123",
                event_id="event-456",
                contribution_score=1.1
            )

        # Valid scores should work
        align_neg = AspirationAlignment(
            aspiration_id="asp-123",
            event_id="event-456",
            contribution_score=-1.0
        )
        align_pos = AspirationAlignment(
            aspiration_id="asp-123",
            event_id="event-456",
            contribution_score=1.0
        )
        assert align_neg.contribution_score == -1.0
        assert align_pos.contribution_score == 1.0

    def test_alignment_validates_confidence_range(self):
        """Test that confidence must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            AspirationAlignment(
                aspiration_id="asp-123",
                event_id="event-456",
                confidence=-0.1
            )

        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            AspirationAlignment(
                aspiration_id="asp-123",
                event_id="event-456",
                confidence=1.1
            )

    def test_alignment_to_dict_and_from_dict(self):
        """Test alignment serialization round-trip."""
        alignment = AspirationAlignment(
            aspiration_id="asp-123",
            event_id="event-456",
            contribution_score=0.8,
            confidence=0.9,
            explanation="Directly supports learning goal"
        )

        # Convert to dict
        align_dict = alignment.to_dict()

        assert align_dict["aspiration_id"] == "asp-123"
        assert align_dict["event_id"] == "event-456"
        assert align_dict["contribution_score"] == 0.8
        assert align_dict["confidence"] == 0.9
        assert isinstance(align_dict["detected_at"], str)

        # Convert back
        restored = AspirationAlignment.from_dict(align_dict)

        assert restored.aspiration_id == "asp-123"
        assert restored.event_id == "event-456"
        assert restored.contribution_score == 0.8
        assert restored.confidence == 0.9
        assert isinstance(restored.detected_at, datetime)
