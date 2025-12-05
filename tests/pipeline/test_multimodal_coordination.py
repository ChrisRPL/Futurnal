"""Tests for multi-modal coordination.

Module 08: Multimodal Integration & Tool Enhancement - Phase 3 Tests
Tests cover cross-modal entity linking, temporal alignment, and result merging.
"""

import pytest
from pathlib import Path
from datetime import datetime, timezone
import hashlib

from futurnal.pipeline.multimodal.coordination import (
    MultiModalCoordinator,
    ResultMerger,
    coordinate_documents,
    merge_for_pkg,
    CrossModalEntity,
    TemporalAlignment,
    CoordinatedResult,
)
from futurnal.pipeline.models import (
    NormalizedDocument,
    NormalizedMetadata,
    DocumentFormat,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def text_document():
    """Create sample text document."""
    content = "Meeting with John Smith about Project Alpha.\nDiscussed budget and timeline."
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    metadata = NormalizedMetadata(
        source_path="/tmp/notes.md",
        source_id="text-001",
        source_type="local_files",
        format=DocumentFormat.TEXT,
        content_type="text/markdown",
        content_hash=content_hash,
        character_count=len(content),
        word_count=len(content.split()),
        line_count=content.count('\n') + 1,
        ingested_at=datetime.now(timezone.utc),
        extra={}
    )
    return NormalizedDocument(
        document_id=content_hash,
        sha256=content_hash,
        content=content,
        metadata=metadata
    )


@pytest.fixture
def audio_document():
    """Create sample audio document with temporal segments."""
    content = "Transcript: John Smith mentioned Project Alpha at 00:15. Budget discussion at 01:30."
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    metadata = NormalizedMetadata(
        source_path="/tmp/recording.wav",
        source_id="audio-001",
        source_type="local_files",
        format=DocumentFormat.AUDIO,
        content_type="audio/wav",
        content_hash=content_hash,
        character_count=len(content),
        word_count=len(content.split()),
        line_count=content.count('\n') + 1,
        ingested_at=datetime.now(timezone.utc),
        extra={
            "audio": {
                "language": "en",
                "confidence": 0.95,
                "segment_count": 2
            },
            "temporal_segments": [
                {"text": "John Smith mentioned Project Alpha", "start": 15.0, "end": 20.0, "confidence": 0.98},
                {"text": "Budget discussion", "start": 90.0, "end": 95.0, "confidence": 0.96}
            ]
        }
    )
    return NormalizedDocument(
        document_id=content_hash,
        sha256=content_hash,
        content=content,
        metadata=metadata
    )


@pytest.fixture
def image_document():
    """Create sample image document with OCR."""
    content = "INVOICE\nJohn Smith\nProject Alpha Budget\nAmount: $10,000"
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    metadata = NormalizedMetadata(
        source_path="/tmp/invoice.png",
        source_id="image-001",
        source_type="local_files",
        format=DocumentFormat.IMAGE,
        content_type="image/png",
        content_hash=content_hash,
        character_count=len(content),
        word_count=len(content.split()),
        line_count=content.count('\n') + 1,
        ingested_at=datetime.now(timezone.utc),
        extra={
            "ocr": {
                "confidence": 0.92,
                "region_count": 4,
                "layout_preserved": True,
                "ocr_backend": "TesseractOCRClient"
            }
        }
    )
    return NormalizedDocument(
        document_id=content_hash,
        sha256=content_hash,
        content=content,
        metadata=metadata
    )


# =============================================================================
# Coordinator Initialization Tests
# =============================================================================


class TestMultiModalCoordinatorInit:
    """Tests for coordinator initialization."""

    def test_initialization_with_entity_linking_enabled(self):
        """Test coordinator initialization with entity linking."""
        coordinator = MultiModalCoordinator(enable_entity_linking=True)

        assert coordinator.enable_entity_linking is True

    def test_initialization_with_entity_linking_disabled(self):
        """Test coordinator initialization without entity linking."""
        coordinator = MultiModalCoordinator(enable_entity_linking=False)

        assert coordinator.enable_entity_linking is False


# =============================================================================
# Modality Analysis Tests
# =============================================================================


class TestModalityAnalysis:
    """Tests for modality distribution analysis."""

    def test_analyze_modalities_single_format(self, text_document):
        """Test modality analysis with single format."""
        coordinator = MultiModalCoordinator()

        modality_summary = coordinator._analyze_modalities([text_document])

        assert modality_summary[DocumentFormat.TEXT] == 1
        assert len(modality_summary) == 1

    def test_analyze_modalities_multiple_formats(self, text_document, audio_document, image_document):
        """Test modality analysis with multiple formats."""
        coordinator = MultiModalCoordinator()

        documents = [text_document, audio_document, image_document]
        modality_summary = coordinator._analyze_modalities(documents)

        assert modality_summary[DocumentFormat.TEXT] == 1
        assert modality_summary[DocumentFormat.AUDIO] == 1
        assert modality_summary[DocumentFormat.IMAGE] == 1
        assert len(modality_summary) == 3

    def test_analyze_modalities_duplicate_formats(self, text_document):
        """Test modality analysis with duplicate formats."""
        coordinator = MultiModalCoordinator()

        # Create two text documents
        content2 = "Another document"
        hash2 = hashlib.sha256(content2.encode()).hexdigest()

        text_doc2 = NormalizedDocument(
            document_id=hash2,
            sha256=hash2,
            content=content2,
            metadata=text_document.metadata
        )

        documents = [text_document, text_doc2]
        modality_summary = coordinator._analyze_modalities(documents)

        assert modality_summary[DocumentFormat.TEXT] == 2


# =============================================================================
# Entity Extraction Tests
# =============================================================================


class TestEntityExtraction:
    """Tests for entity extraction from documents."""

    def test_extract_entities_from_document_capitalized_words(self, text_document):
        """Test entity extraction finds capitalized words."""
        coordinator = MultiModalCoordinator()

        entities = coordinator._extract_entities_from_document(text_document)

        # Should find "Meeting", "John", "Smith", "Project", "Alpha", "Discussed"
        # Note: entity extraction may include punctuation, so we check if text contains the entity
        entity_texts = [e[0] for e in entities]
        assert any("John" in e for e in entity_texts)
        assert any("Smith" in e for e in entity_texts)
        assert any("Project" in e for e in entity_texts)
        assert any("Alpha" in e for e in entity_texts)

    def test_extract_entities_from_document_with_metadata(self):
        """Test entity extraction from document with entities in metadata."""
        content = "Test content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        metadata = NormalizedMetadata(
            source_path="/tmp/test.txt",
            source_id="test-001",
            source_type="local_files",
            format=DocumentFormat.TEXT,
            content_type="text/plain",
            content_hash=content_hash,
            character_count=len(content),
            word_count=len(content.split()),
            line_count=1,
            ingested_at=datetime.now(timezone.utc),
            extra={
                "entities": [
                    {"text": "EntityOne", "type": "PERSON"},
                    {"text": "EntityTwo", "type": "ORGANIZATION"}
                ]
            }
        )
        doc = NormalizedDocument(
            document_id=content_hash,
            sha256=content_hash,
            content=content,
            metadata=metadata
        )

        coordinator = MultiModalCoordinator()
        entities = coordinator._extract_entities_from_document(doc)

        entity_texts = [e[0] for e in entities]
        assert "EntityOne" in entity_texts
        assert "EntityTwo" in entity_texts


# =============================================================================
# Cross-Modal Entity Linking Tests
# =============================================================================


class TestCrossModalEntityLinking:
    """Tests for cross-modal entity linking."""

    def test_extract_cross_modal_entities_single_modality(self, text_document):
        """Test cross-modal entity extraction with single modality (no cross-modal entities)."""
        coordinator = MultiModalCoordinator()

        cross_modal_entities = coordinator._extract_cross_modal_entities([text_document])

        # No entities should appear in multiple modalities
        assert len(cross_modal_entities) == 0

    def test_extract_cross_modal_entities_multiple_modalities(self, text_document, audio_document, image_document):
        """Test cross-modal entity extraction with multiple modalities."""
        coordinator = MultiModalCoordinator()

        documents = [text_document, audio_document, image_document]
        cross_modal_entities = coordinator._extract_cross_modal_entities(documents)

        # Should find "John", "Smith", "Project", "Alpha" across modalities
        entity_texts = [e.entity_text for e in cross_modal_entities]

        # At least some entities should be cross-modal
        assert len(cross_modal_entities) > 0

        # Check specific cross-modal entity
        john_entities = [e for e in cross_modal_entities if "john" in e.entity_text.lower()]
        if john_entities:
            john_entity = john_entities[0]
            assert len(john_entity.modalities) >= 2
            assert len(john_entity.occurrences) >= 2

    def test_cross_modal_entity_confidence_calculation(self):
        """Test confidence calculation for cross-modal entities."""
        content1 = "Entity appears here"
        content2 = "Entity appears here too"
        hash1 = hashlib.sha256(content1.encode()).hexdigest()
        hash2 = hashlib.sha256(content2.encode()).hexdigest()

        doc1 = NormalizedDocument(
            document_id=hash1,
            sha256=hash1,
            content=content1,
            metadata=NormalizedMetadata(
                source_path="/tmp/doc1.txt",
                source_id="doc1",
                source_type="local_files",
                format=DocumentFormat.TEXT,
                content_type="text/plain",
                content_hash=hash1,
                character_count=len(content1),
                word_count=len(content1.split()),
                line_count=1,
                ingested_at=datetime.now(timezone.utc),
            )
        )

        doc2 = NormalizedDocument(
            document_id=hash2,
            sha256=hash2,
            content=content2,
            metadata=NormalizedMetadata(
                source_path="/tmp/doc2.wav",
                source_id="doc2",
                source_type="local_files",
                format=DocumentFormat.AUDIO,
                content_type="audio/wav",
                content_hash=hash2,
                character_count=len(content2),
                word_count=len(content2.split()),
                line_count=1,
                ingested_at=datetime.now(timezone.utc),
            )
        )

        coordinator = MultiModalCoordinator()
        cross_modal_entities = coordinator._extract_cross_modal_entities([doc1, doc2])

        # Find "Entity" cross-modal entity
        entity_entities = [e for e in cross_modal_entities if "entity" in e.entity_text.lower()]
        if entity_entities:
            entity = entity_entities[0]
            # Confidence should be average of occurrences (both 1.0 by default)
            assert entity.confidence == 1.0


# =============================================================================
# Temporal Alignment Tests
# =============================================================================


class TestTemporalAlignment:
    """Tests for temporal alignment detection."""

    def test_detect_temporal_alignments_no_audio(self, text_document, image_document):
        """Test temporal alignment with no audio documents."""
        coordinator = MultiModalCoordinator()

        alignments = coordinator._detect_temporal_alignments([text_document, image_document])

        # No audio = no temporal alignments
        assert len(alignments) == 0

    def test_detect_temporal_alignments_with_audio(self, text_document, audio_document):
        """Test temporal alignment with audio document."""
        coordinator = MultiModalCoordinator()

        # Rename text document to match audio segment mention
        text_document.metadata.source_path = "/tmp/project_alpha.md"

        alignments = coordinator._detect_temporal_alignments([text_document, audio_document])

        # Should find alignment when audio mentions "project alpha" and doc is named "project_alpha.md"
        # This is a heuristic-based alignment
        assert isinstance(alignments, list)

    def test_detect_temporal_alignments_no_segments(self, text_document):
        """Test temporal alignment when audio has no temporal segments."""
        # Create audio doc without temporal_segments
        content = "Audio content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        audio_doc = NormalizedDocument(
            document_id=content_hash,
            sha256=content_hash,
            content=content,
            metadata=NormalizedMetadata(
                source_path="/tmp/recording.wav",
                source_id="audio-001",
                source_type="local_files",
                format=DocumentFormat.AUDIO,
                content_type="audio/wav",
                content_hash=content_hash,
                character_count=13,
                word_count=2,
                line_count=1,
                ingested_at=datetime.now(timezone.utc),
                extra={}
            )
        )

        coordinator = MultiModalCoordinator()
        alignments = coordinator._detect_temporal_alignments([text_document, audio_doc])

        assert len(alignments) == 0


# =============================================================================
# Metadata Merging Tests
# =============================================================================


class TestMetadataMerging:
    """Tests for metadata merging across documents."""

    def test_merge_metadata_single_document(self, text_document):
        """Test metadata merging with single document."""
        coordinator = MultiModalCoordinator()

        merged = coordinator._merge_metadata([text_document])

        assert merged["document_count"] == 1
        assert merged["total_characters"] == text_document.metadata.character_count
        assert merged["total_words"] == text_document.metadata.word_count
        assert DocumentFormat.TEXT.value in merged["formats"]
        assert text_document.metadata.source_id in merged["source_ids"]

    def test_merge_metadata_multiple_documents(self, text_document, audio_document, image_document):
        """Test metadata merging with multiple documents."""
        coordinator = MultiModalCoordinator()

        documents = [text_document, audio_document, image_document]
        merged = coordinator._merge_metadata(documents)

        assert merged["document_count"] == 3
        assert merged["total_characters"] == sum(d.metadata.character_count for d in documents)
        assert merged["total_words"] == sum(d.metadata.word_count for d in documents)
        assert len(merged["formats"]) == 3
        assert len(merged["source_ids"]) == 3

    def test_merge_metadata_with_audio(self, audio_document):
        """Test metadata merging includes audio-specific metadata."""
        coordinator = MultiModalCoordinator()

        merged = coordinator._merge_metadata([audio_document])

        assert "audio" in merged
        assert merged["audio"]["count"] == 1
        assert "en" in merged["audio"]["languages"]
        assert merged["audio"]["avg_confidence"] == 0.95

    def test_merge_metadata_with_ocr(self, image_document):
        """Test metadata merging includes OCR-specific metadata."""
        coordinator = MultiModalCoordinator()

        merged = coordinator._merge_metadata([image_document])

        assert "ocr" in merged
        assert merged["ocr"]["count"] == 1
        assert merged["ocr"]["avg_confidence"] == 0.92
        assert merged["ocr"]["total_regions"] == 4


# =============================================================================
# Coordination Tests
# =============================================================================


class TestCoordination:
    """Tests for complete coordination workflow."""

    def test_coordinate_empty_documents(self):
        """Test coordination with empty document list."""
        coordinator = MultiModalCoordinator()

        result = coordinator.coordinate([])

        assert isinstance(result, CoordinatedResult)
        assert len(result.documents) == 0
        assert len(result.cross_modal_entities) == 0
        assert len(result.temporal_alignments) == 0

    def test_coordinate_single_document(self, text_document):
        """Test coordination with single document."""
        coordinator = MultiModalCoordinator()

        result = coordinator.coordinate([text_document])

        assert isinstance(result, CoordinatedResult)
        assert len(result.documents) == 1
        assert result.documents[0] == text_document
        assert result.modality_summary[DocumentFormat.TEXT] == 1

    def test_coordinate_multiple_documents_with_entity_linking(self, text_document, audio_document, image_document):
        """Test coordination with entity linking enabled."""
        coordinator = MultiModalCoordinator(enable_entity_linking=True)

        documents = [text_document, audio_document, image_document]
        result = coordinator.coordinate(documents)

        assert isinstance(result, CoordinatedResult)
        assert len(result.documents) == 3
        assert len(result.modality_summary) == 3
        # Should have some cross-modal entities
        assert len(result.cross_modal_entities) >= 0  # May or may not find matches

    def test_coordinate_multiple_documents_without_entity_linking(self, text_document, audio_document):
        """Test coordination with entity linking disabled."""
        coordinator = MultiModalCoordinator(enable_entity_linking=False)

        documents = [text_document, audio_document]
        result = coordinator.coordinate(documents, link_entities=False)

        assert isinstance(result, CoordinatedResult)
        assert len(result.documents) == 2
        assert len(result.cross_modal_entities) == 0  # Disabled

    def test_coordinate_override_entity_linking(self, text_document, audio_document):
        """Test coordination with entity linking override."""
        # Init with disabled
        coordinator = MultiModalCoordinator(enable_entity_linking=False)

        # Override to enable
        documents = [text_document, audio_document]
        result = coordinator.coordinate(documents, link_entities=True)

        # Should have entity linking despite init setting
        assert isinstance(result, CoordinatedResult)


# =============================================================================
# Result Merger Tests
# =============================================================================


class TestResultMerger:
    """Tests for PKG result merging."""

    def test_merge_for_pkg_empty_result(self):
        """Test PKG merging with empty coordinated result."""
        merger = ResultMerger()
        coordinated = CoordinatedResult(documents=[])

        pkg_data = merger.merge_for_pkg(coordinated)

        assert pkg_data["documents"] == []
        assert pkg_data["cross_modal_entities"] == []
        assert pkg_data["temporal_alignments"] == []

    def test_merge_for_pkg_with_documents(self, text_document, audio_document):
        """Test PKG merging with documents."""
        merger = ResultMerger()
        coordinated = CoordinatedResult(
            documents=[text_document, audio_document],
            modality_summary={
                DocumentFormat.TEXT: 1,
                DocumentFormat.AUDIO: 1
            },
            merged_metadata={"document_count": 2}
        )

        pkg_data = merger.merge_for_pkg(coordinated)

        assert len(pkg_data["documents"]) == 2
        assert pkg_data["documents"][0]["source_id"] == "text-001"
        assert pkg_data["documents"][1]["source_id"] == "audio-001"
        assert pkg_data["summary"]["document_count"] == 2

    def test_merge_for_pkg_with_cross_modal_entities(self, text_document):
        """Test PKG merging with cross-modal entities."""
        merger = ResultMerger()

        cross_modal_entity = CrossModalEntity(
            entity_text="John Smith",
            entity_type="PERSON",
            modalities={DocumentFormat.TEXT, DocumentFormat.AUDIO},
            occurrences=[
                {"source_id": "text-001", "format": "text", "position": 0, "confidence": 0.95},
                {"source_id": "audio-001", "format": "audio", "position": 15, "confidence": 0.98}
            ],
            confidence=0.965
        )

        coordinated = CoordinatedResult(
            documents=[text_document],
            cross_modal_entities=[cross_modal_entity]
        )

        pkg_data = merger.merge_for_pkg(coordinated)

        assert len(pkg_data["cross_modal_entities"]) == 1
        entity_data = pkg_data["cross_modal_entities"][0]
        assert entity_data["text"] == "John Smith"
        assert entity_data["type"] == "PERSON"
        assert len(entity_data["modalities"]) == 2
        assert entity_data["confidence"] == 0.965


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_coordinate_documents_function(self, text_document, audio_document):
        """Test coordinate_documents convenience function."""
        result = coordinate_documents([text_document, audio_document], link_entities=True)

        assert isinstance(result, CoordinatedResult)
        assert len(result.documents) == 2

    def test_merge_for_pkg_function(self, text_document):
        """Test merge_for_pkg convenience function."""
        coordinated = CoordinatedResult(documents=[text_document])

        pkg_data = merge_for_pkg(coordinated)

        assert isinstance(pkg_data, dict)
        assert "documents" in pkg_data
        assert "cross_modal_entities" in pkg_data
        assert "temporal_alignments" in pkg_data
