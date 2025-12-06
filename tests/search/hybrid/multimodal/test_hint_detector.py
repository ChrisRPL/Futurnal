"""Tests for ModalityHintDetector.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Tests cover:
- Audio hint detection (voice notes, meetings, recordings)
- OCR document hint detection (scanned, PDF, handwritten)
- OCR image hint detection (photos, screenshots, whiteboard)
- Video hint detection
- Confidence thresholds
- Edge cases and no-hint queries
"""

import pytest

from futurnal.search.hybrid.multimodal.hint_detector import ModalityHintDetector
from futurnal.search.hybrid.multimodal.types import ContentSource


class TestAudioHintDetection:
    """Tests for audio transcription hint detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ModalityHintDetector()

    def test_voice_notes_explicit(self):
        """Test explicit voice notes reference."""
        hints = self.detector.detect("what's in my voice notes about budget?")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.AUDIO_TRANSCRIPTION
        assert hints[0].confidence >= 0.90
        assert "voice notes" in hints[0].hint_phrase.lower()

    def test_voice_memos(self):
        """Test voice memos reference."""
        hints = self.detector.detect("check my voice memos from yesterday")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.AUDIO_TRANSCRIPTION
        assert hints[0].confidence >= 0.90

    def test_recordings(self):
        """Test recordings reference."""
        hints = self.detector.detect("search the recordings for project updates")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.AUDIO_TRANSCRIPTION

    def test_meeting_context(self):
        """Test meeting context indicator."""
        hints = self.detector.detect("what did we discuss in the meeting?")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.AUDIO_TRANSCRIPTION
        assert hints[0].confidence >= 0.85

    def test_call_context(self):
        """Test call context indicator."""
        hints = self.detector.detect("from that call with the client")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.AUDIO_TRANSCRIPTION

    def test_what_i_said(self):
        """Test 'what I said' pattern."""
        hints = self.detector.detect("what did I say about the deadline?")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.AUDIO_TRANSCRIPTION
        assert hints[0].confidence >= 0.85

    def test_talked_about(self):
        """Test 'talked about' pattern."""
        hints = self.detector.detect("we talked about Q4 targets")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.AUDIO_TRANSCRIPTION

    def test_transcription_mention(self):
        """Test explicit transcription mention."""
        hints = self.detector.detect("find it in the transcription")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.AUDIO_TRANSCRIPTION
        assert hints[0].confidence >= 0.90

    def test_podcast_reference(self):
        """Test podcast reference."""
        hints = self.detector.detect("from that podcast episode about AI")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.AUDIO_TRANSCRIPTION


class TestOCRDocumentHintDetection:
    """Tests for OCR document hint detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ModalityHintDetector()

    def test_scanned_document_explicit(self):
        """Test explicit scanned document reference."""
        hints = self.detector.detect("from the scanned document about insurance")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.OCR_DOCUMENT
        assert hints[0].confidence >= 0.90

    def test_scanned_pdf(self):
        """Test scanned PDF reference."""
        hints = self.detector.detect("in that scanned PDF I uploaded")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.OCR_DOCUMENT
        assert hints[0].confidence >= 0.90

    def test_pdf_reference(self):
        """Test general PDF reference."""
        hints = self.detector.detect("check the pdf for the address")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.OCR_DOCUMENT
        assert hints[0].confidence >= 0.85

    def test_handwritten_notes(self):
        """Test handwritten notes reference."""
        hints = self.detector.detect("my handwritten notes from the lecture")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.OCR_DOCUMENT
        assert hints[0].confidence >= 0.85

    def test_printed_document(self):
        """Test printed document reference."""
        hints = self.detector.detect("the printed document from HR")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.OCR_DOCUMENT

    def test_physical_documents(self):
        """Test physical document type references."""
        queries = [
            "the paper from the conference",
            "that letter from the bank",
            "the form I filled out",
            "that receipt from dinner",
            "the invoice for the project",
        ]
        for query in queries:
            hints = self.detector.detect(query)
            assert len(hints) > 0, f"No hint for: {query}"
            assert hints[0].modality == ContentSource.OCR_DOCUMENT


class TestOCRImageHintDetection:
    """Tests for OCR image hint detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ModalityHintDetector()

    def test_image_reference(self):
        """Test explicit image reference."""
        hints = self.detector.detect("text from that image I took")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.OCR_IMAGE
        assert hints[0].confidence >= 0.90

    def test_photo_reference(self):
        """Test photo reference."""
        hints = self.detector.detect("in the photo of the whiteboard")
        assert len(hints) > 0
        # Could be OCR_IMAGE or OCR_DOCUMENT, both valid
        assert hints[0].modality in (
            ContentSource.OCR_IMAGE,
            ContentSource.OCR_DOCUMENT,
        )

    def test_screenshot_reference(self):
        """Test screenshot reference."""
        hints = self.detector.detect("from that screenshot I took")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.OCR_IMAGE
        assert hints[0].confidence >= 0.90

    def test_whiteboard_reference(self):
        """Test whiteboard reference."""
        hints = self.detector.detect("notes from the whiteboard")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.OCR_IMAGE

    def test_text_in_image(self):
        """Test 'text in image' pattern."""
        hints = self.detector.detect("the text in that picture")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.OCR_IMAGE
        assert hints[0].confidence >= 0.90


class TestVideoHintDetection:
    """Tests for video transcription hint detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ModalityHintDetector()

    def test_video_reference(self):
        """Test explicit video reference."""
        hints = self.detector.detect("from that video about testing")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.VIDEO_TRANSCRIPTION
        assert hints[0].confidence >= 0.85

    def test_youtube_reference(self):
        """Test YouTube reference."""
        hints = self.detector.detect("in the youtube video about React")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.VIDEO_TRANSCRIPTION


class TestNoHintQueries:
    """Tests for queries without modality hints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ModalityHintDetector()

    def test_simple_query(self):
        """Test simple query without hints."""
        hints = self.detector.detect("project deadlines")
        assert len(hints) == 0

    def test_question_without_hint(self):
        """Test question without modality hint."""
        hints = self.detector.detect("what are the Q4 targets?")
        assert len(hints) == 0

    def test_search_query(self):
        """Test generic search query."""
        hints = self.detector.detect("find information about the API")
        assert len(hints) == 0

    def test_notes_without_voice(self):
        """Test notes reference without voice qualifier."""
        # Just "notes" should NOT trigger audio hint
        hints = self.detector.detect("check my notes")
        # Should have no audio hints with high confidence
        audio_hints = [
            h for h in hints if h.modality == ContentSource.AUDIO_TRANSCRIPTION
        ]
        if audio_hints:
            assert audio_hints[0].confidence < 0.85


class TestConfidenceThresholds:
    """Tests for confidence thresholds and filtering."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ModalityHintDetector()

    def test_get_primary_modality_high_confidence(self):
        """Test get_primary_modality returns high confidence hint."""
        query = "in my voice notes about the meeting"
        primary = self.detector.get_primary_modality(query)
        assert primary == ContentSource.AUDIO_TRANSCRIPTION

    def test_get_primary_modality_no_hint(self):
        """Test get_primary_modality returns None for no hint."""
        query = "project status update"
        primary = self.detector.get_primary_modality(query)
        assert primary is None

    def test_should_filter_high_confidence(self):
        """Test should_filter_by_modality for high confidence."""
        query = "from the scanned document"
        assert self.detector.should_filter_by_modality(query) is True

    def test_should_filter_no_hint(self):
        """Test should_filter_by_modality for no hint."""
        query = "find the project plan"
        assert self.detector.should_filter_by_modality(query) is False


class TestMultipleHints:
    """Tests for queries with multiple hints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ModalityHintDetector()

    def test_multiple_audio_hints(self):
        """Test query with multiple audio hints."""
        query = "what I said in the meeting recording"
        hints = self.detector.detect(query)
        audio_hints = [
            h for h in hints if h.modality == ContentSource.AUDIO_TRANSCRIPTION
        ]
        assert len(audio_hints) >= 2

    def test_hints_sorted_by_confidence(self):
        """Test hints are sorted by confidence."""
        query = "the audio recording from the meeting conversation"
        hints = self.detector.detect(query)
        if len(hints) >= 2:
            for i in range(len(hints) - 1):
                assert hints[i].confidence >= hints[i + 1].confidence

    def test_get_all_modalities(self):
        """Test getting all detected modalities."""
        query = "check the scanned pdf and my voice notes"
        modalities = self.detector.get_all_modalities(query)
        assert ContentSource.OCR_DOCUMENT in modalities
        assert ContentSource.AUDIO_TRANSCRIPTION in modalities


class TestQueryCleaning:
    """Tests for query cleaning functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ModalityHintDetector()

    def test_extract_query_without_hints_audio(self):
        """Test removing audio hints from query."""
        query = "in my voice notes about the budget meeting"
        cleaned = self.detector.extract_query_without_hints(query)
        assert "voice notes" not in cleaned.lower()
        assert "budget" in cleaned.lower()
        assert "meeting" in cleaned.lower()

    def test_extract_query_without_hints_ocr(self):
        """Test removing OCR hints from query."""
        query = "from the scanned document show the total amount"
        cleaned = self.detector.extract_query_without_hints(query)
        assert "scanned document" not in cleaned.lower()
        assert "total amount" in cleaned.lower()

    def test_extract_query_no_hints(self):
        """Test query without hints returns original."""
        query = "project deadlines for Q4"
        cleaned = self.detector.extract_query_without_hints(query)
        assert cleaned == query


class TestConfidenceSummary:
    """Tests for confidence summary functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ModalityHintDetector()

    def test_confidence_summary_single_modality(self):
        """Test confidence summary with single modality."""
        query = "in my voice notes"
        summary = self.detector.get_confidence_summary(query)
        assert ContentSource.AUDIO_TRANSCRIPTION in summary
        assert summary[ContentSource.AUDIO_TRANSCRIPTION] >= 0.90

    def test_confidence_summary_multiple_modalities(self):
        """Test confidence summary with multiple modalities."""
        query = "from the pdf and the video recording"
        summary = self.detector.get_confidence_summary(query)
        assert len(summary) >= 2


class TestCaseInsensitivity:
    """Tests for case-insensitive pattern matching."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ModalityHintDetector()

    def test_uppercase_voice_notes(self):
        """Test uppercase input."""
        hints = self.detector.detect("IN MY VOICE NOTES")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.AUDIO_TRANSCRIPTION

    def test_mixed_case(self):
        """Test mixed case input."""
        hints = self.detector.detect("From The Scanned Document")
        assert len(hints) > 0
        assert hints[0].modality == ContentSource.OCR_DOCUMENT
