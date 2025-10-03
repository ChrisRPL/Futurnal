"""Tests for experiential data models."""

from datetime import datetime

import pytest

from futurnal.models.experiential import ExperientialEvent, ExperientialStream


class TestExperientialEvent:
    """Test ExperientialEvent model."""

    def test_create_event_with_defaults(self):
        """Test creating an event with minimal required fields."""
        event = ExperientialEvent(
            event_type="note_created",
            source_uri="obsidian://vault/note.md"
        )

        assert event.event_id is not None
        assert event.event_type == "note_created"
        assert event.source_uri == "obsidian://vault/note.md"
        assert isinstance(event.timestamp, datetime)
        assert event.context == {}
        assert event.related_events == []
        assert event.potential_causes == []
        assert event.potential_effects == []

    def test_create_event_with_full_data(self):
        """Test creating an event with all fields populated."""
        timestamp = datetime(2025, 1, 28, 10, 30, 0)
        context = {"file_size": 1024, "tags": ["project", "planning"]}

        event = ExperientialEvent(
            event_id="test-event-123",
            timestamp=timestamp,
            event_type="file_modified",
            source_uri="/path/to/file.md",
            context=context,
            related_events=["event-456"],
            potential_causes=["event-789"],
            potential_effects=["event-012"]
        )

        assert event.event_id == "test-event-123"
        assert event.timestamp == timestamp
        assert event.event_type == "file_modified"
        assert event.source_uri == "/path/to/file.md"
        assert event.context == context
        assert event.related_events == ["event-456"]
        assert event.potential_causes == ["event-789"]
        assert event.potential_effects == ["event-012"]

    def test_event_requires_type_and_uri(self):
        """Test that event_type and source_uri are required."""
        with pytest.raises(ValueError, match="event_type is required"):
            ExperientialEvent(
                event_type="",
                source_uri="test://uri"
            )

        with pytest.raises(ValueError, match="source_uri is required"):
            ExperientialEvent(
                event_type="test_event",
                source_uri=""
            )

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = ExperientialEvent(
            event_type="note_created",
            source_uri="test://note"
        )

        event_dict = event.to_dict()

        assert "event_id" in event_dict
        assert "timestamp" in event_dict
        assert event_dict["event_type"] == "note_created"
        assert event_dict["source_uri"] == "test://note"
        assert isinstance(event_dict["timestamp"], str)  # ISO format

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            "event_id": "test-123",
            "timestamp": "2025-01-28T10:30:00",
            "event_type": "file_created",
            "source_uri": "file:///test.md",
            "context": {"size": 100},
            "related_events": [],
            "potential_causes": [],
            "potential_effects": []
        }

        event = ExperientialEvent.from_dict(data)

        assert event.event_id == "test-123"
        assert isinstance(event.timestamp, datetime)
        assert event.event_type == "file_created"


class TestExperientialStream:
    """Test ExperientialStream model."""

    def test_create_empty_stream(self):
        """Test creating an empty event stream."""
        stream = ExperientialStream()

        assert stream.stream_id is not None
        assert stream.events == []
        assert stream.metadata == {}

    def test_add_event_maintains_temporal_order(self):
        """Test that adding events maintains chronological order."""
        stream = ExperientialStream()

        # Add events out of order
        event1 = ExperientialEvent(
            timestamp=datetime(2025, 1, 28, 12, 0, 0),
            event_type="event_1",
            source_uri="uri1"
        )
        event2 = ExperientialEvent(
            timestamp=datetime(2025, 1, 28, 10, 0, 0),
            event_type="event_2",
            source_uri="uri2"
        )
        event3 = ExperientialEvent(
            timestamp=datetime(2025, 1, 28, 11, 0, 0),
            event_type="event_3",
            source_uri="uri3"
        )

        stream.add_event(event1)
        stream.add_event(event2)
        stream.add_event(event3)

        # Verify events are sorted by timestamp
        assert len(stream.events) == 3
        assert stream.events[0].event_type == "event_2"  # 10:00
        assert stream.events[1].event_type == "event_3"  # 11:00
        assert stream.events[2].event_type == "event_1"  # 12:00

    def test_get_events_in_range(self):
        """Test filtering events by time range."""
        stream = ExperientialStream()

        # Add events at different times
        for hour in [9, 10, 11, 12, 13]:
            stream.add_event(ExperientialEvent(
                timestamp=datetime(2025, 1, 28, hour, 0, 0),
                event_type=f"event_{hour}",
                source_uri=f"uri_{hour}"
            ))

        # Query events between 10:00 and 12:00
        filtered = stream.get_events_in_range(
            start=datetime(2025, 1, 28, 10, 0, 0),
            end=datetime(2025, 1, 28, 12, 0, 0)
        )

        assert len(filtered) == 3
        assert filtered[0].event_type == "event_10"
        assert filtered[1].event_type == "event_11"
        assert filtered[2].event_type == "event_12"

    def test_get_events_in_range_with_type_filter(self):
        """Test filtering events by time range and type."""
        stream = ExperientialStream()

        # Add mixed event types
        stream.add_event(ExperientialEvent(
            timestamp=datetime(2025, 1, 28, 10, 0, 0),
            event_type="note_created",
            source_uri="uri1"
        ))
        stream.add_event(ExperientialEvent(
            timestamp=datetime(2025, 1, 28, 11, 0, 0),
            event_type="file_modified",
            source_uri="uri2"
        ))
        stream.add_event(ExperientialEvent(
            timestamp=datetime(2025, 1, 28, 12, 0, 0),
            event_type="note_created",
            source_uri="uri3"
        ))

        # Filter for note_created events
        filtered = stream.get_events_in_range(
            start=datetime(2025, 1, 28, 9, 0, 0),
            end=datetime(2025, 1, 28, 13, 0, 0),
            event_type="note_created"
        )

        assert len(filtered) == 2
        assert all(e.event_type == "note_created" for e in filtered)

    def test_stream_to_dict_and_from_dict(self):
        """Test stream serialization round-trip."""
        stream = ExperientialStream(
            stream_id="test-stream",
            metadata={"user_id": "user123"}
        )
        stream.add_event(ExperientialEvent(
            event_type="test_event",
            source_uri="test://uri"
        ))

        # Convert to dict
        stream_dict = stream.to_dict()

        assert stream_dict["stream_id"] == "test-stream"
        assert stream_dict["metadata"]["user_id"] == "user123"
        assert len(stream_dict["events"]) == 1

        # Convert back to stream
        restored = ExperientialStream.from_dict(stream_dict)

        assert restored.stream_id == "test-stream"
        assert restored.metadata["user_id"] == "user123"
        assert len(restored.events) == 1
        assert restored.events[0].event_type == "test_event"
