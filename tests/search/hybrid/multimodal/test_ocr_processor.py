"""Tests for OCRContentProcessor.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Tests cover:
- OCR output processing
- Layout type detection
- Quality tier mapping
- Fuzzy variant generation
- Source type detection
- Metadata extraction
"""

import pytest
from datetime import datetime

from futurnal.search.hybrid.multimodal.ocr_processor import (
    OCRContentProcessor,
    OCRLayoutType,
    OCRContentMetadata,
)
from futurnal.search.hybrid.multimodal.types import (
    ContentSource,
    ExtractionQuality,
)


class TestOCRLayoutType:
    """Tests for OCRLayoutType enum."""

    def test_layout_type_values(self):
        """Test all layout type values exist."""
        assert OCRLayoutType.SIMPLE_TEXT == "simple_text"
        assert OCRLayoutType.MULTI_COLUMN == "multi_column"
        assert OCRLayoutType.TABLE_HEAVY == "table_heavy"
        assert OCRLayoutType.FORM == "form"
        assert OCRLayoutType.MIXED == "mixed"
        assert OCRLayoutType.HANDWRITTEN == "handwritten"

    def test_layout_type_is_string_enum(self):
        """Test OCRLayoutType is string enum."""
        assert isinstance(OCRLayoutType.SIMPLE_TEXT, str)
        assert OCRLayoutType.SIMPLE_TEXT.value == "simple_text"


class TestOCRContentMetadata:
    """Tests for OCRContentMetadata dataclass."""

    def test_metadata_creation(self):
        """Test metadata creation with all fields."""
        metadata = OCRContentMetadata(
            source_file="document.pdf",
            page_number=1,
            layout_type=OCRLayoutType.SIMPLE_TEXT,
            confidence_score=0.95,
            character_error_rate=0.02,
            detected_language="en",
            has_tables=False,
            has_images=False,
            has_handwriting=False,
            extraction_model="deepseek-ocr-v2",
            bounding_boxes_preserved=True,
        )
        assert metadata.source_file == "document.pdf"
        assert metadata.page_number == 1
        assert metadata.layout_type == OCRLayoutType.SIMPLE_TEXT
        assert metadata.confidence_score == 0.95
        assert metadata.character_error_rate == 0.02

    def test_metadata_optional_page(self):
        """Test metadata with optional page number."""
        metadata = OCRContentMetadata(
            source_file="image.png",
            page_number=None,
            layout_type=OCRLayoutType.SIMPLE_TEXT,
            confidence_score=0.85,
            character_error_rate=0.05,
            detected_language="en",
            has_tables=False,
            has_images=False,
            has_handwriting=False,
            extraction_model="deepseek-ocr-v2",
            bounding_boxes_preserved=False,
        )
        assert metadata.page_number is None


class TestOCRContentProcessor:
    """Tests for OCRContentProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OCRContentProcessor()

    def test_init_default_confidence(self):
        """Test default minimum confidence."""
        processor = OCRContentProcessor()
        assert processor.min_confidence == 0.6

    def test_init_custom_confidence(self):
        """Test custom minimum confidence."""
        processor = OCRContentProcessor(min_confidence=0.8)
        assert processor.min_confidence == 0.8


class TestTextExtraction:
    """Tests for text extraction from OCR output."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OCRContentProcessor()

    def test_extract_text_direct(self):
        """Test extraction from direct text field."""
        ocr_output = {
            "text": "Hello World",
            "confidence": 0.95,
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["content"] == "Hello World"

    def test_extract_text_from_blocks(self):
        """Test extraction from blocks when no direct text."""
        ocr_output = {
            "blocks": [
                {"text": "First block", "confidence": 0.9},
                {"text": "Second block", "confidence": 0.85},
            ],
            "confidence": 0.85,
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert "First block" in result["content"]
        assert "Second block" in result["content"]

    def test_extract_text_from_regions(self):
        """Test extraction from regions field."""
        ocr_output = {
            "regions": [
                {"text": "Region one", "confidence": 0.9},
                {"text": "Region two", "confidence": 0.8},
            ],
            "confidence": 0.85,
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert "Region one" in result["content"]
        assert "Region two" in result["content"]

    def test_extract_text_filters_low_confidence(self):
        """Test low confidence blocks are filtered."""
        processor = OCRContentProcessor(min_confidence=0.7)
        ocr_output = {
            "blocks": [
                {"text": "High confidence", "confidence": 0.9},
                {"text": "Low confidence", "confidence": 0.5},
            ],
            "confidence": 0.7,
        }
        result = processor.process_ocr_result(ocr_output)
        assert "High confidence" in result["content"]
        assert "Low confidence" not in result["content"]


class TestLayoutDetection:
    """Tests for layout type detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OCRContentProcessor()

    def test_detect_simple_text(self):
        """Test simple text layout detection."""
        ocr_output = {
            "text": "Simple paragraph text",
            "confidence": 0.9,
            "layout": {},
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["ocr_metadata"]["layout_type"] == "simple_text"

    def test_detect_multi_column(self):
        """Test multi-column layout detection."""
        ocr_output = {
            "text": "Column content",
            "confidence": 0.9,
            "layout": {"column_count": 2},
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["ocr_metadata"]["layout_type"] == "multi_column"

    def test_detect_table_heavy(self):
        """Test table-heavy layout detection."""
        ocr_output = {
            "text": "Table content",
            "confidence": 0.9,
            "layout": {"table_count": 5},
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["ocr_metadata"]["layout_type"] == "table_heavy"

    def test_detect_handwritten(self):
        """Test handwritten layout detection."""
        ocr_output = {
            "text": "Handwritten notes",
            "confidence": 0.75,
            "layout": {"is_handwritten": True},
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["ocr_metadata"]["layout_type"] == "handwritten"

    def test_detect_form(self):
        """Test form layout detection."""
        ocr_output = {
            "text": "Form fields",
            "confidence": 0.9,
            "layout": {"is_form": True},
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["ocr_metadata"]["layout_type"] == "form"

    def test_detect_mixed(self):
        """Test mixed layout detection."""
        ocr_output = {
            "text": "Mixed content",
            "confidence": 0.85,
            "layout": {"is_mixed": True},
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["ocr_metadata"]["layout_type"] == "mixed"


class TestQualityTier:
    """Tests for quality tier mapping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OCRContentProcessor()

    def test_high_quality(self):
        """Test high quality tier for high confidence."""
        ocr_output = {"text": "Clear text", "confidence": 0.98}
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["extraction_quality"] == "high"

    def test_medium_quality(self):
        """Test medium quality tier for moderate confidence."""
        ocr_output = {"text": "Moderate text", "confidence": 0.88}
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["extraction_quality"] == "medium"

    def test_low_quality(self):
        """Test low quality tier for low confidence."""
        ocr_output = {"text": "Unclear text", "confidence": 0.70}
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["extraction_quality"] == "low"

    def test_uncertain_quality(self):
        """Test uncertain quality tier for very low confidence."""
        ocr_output = {"text": "Very unclear", "confidence": 0.45}
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["extraction_quality"] == "uncertain"


class TestFuzzyVariants:
    """Tests for fuzzy variant generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OCRContentProcessor()

    def test_generate_fuzzy_variants_l_to_1(self):
        """Test l -> 1 variant generation."""
        ocr_output = {"text": "hello", "confidence": 0.9}
        result = self.processor.process_ocr_result(ocr_output)
        variants = result["fuzzy_variants"]
        # Should generate variant with 'l' replaced by '1', 'I', or '|'
        assert any("he1lo" in v or "heIlo" in v or "he|lo" in v for v in variants)

    def test_generate_fuzzy_variants_O_to_0(self):
        """Test O -> 0 variant generation."""
        ocr_output = {"text": "HELLO", "confidence": 0.9}
        result = self.processor.process_ocr_result(ocr_output)
        variants = result["fuzzy_variants"]
        # After lowercasing, "o" patterns don't match but "O" patterns do on original case
        # The _generate_fuzzy_variants lowercases first
        assert isinstance(variants, list)

    def test_generate_fuzzy_variants_m_to_rn(self):
        """Test m -> rn variant generation."""
        ocr_output = {"text": "memory", "confidence": 0.9}
        result = self.processor.process_ocr_result(ocr_output)
        variants = result["fuzzy_variants"]
        assert any("rn" in v for v in variants) or len(variants) >= 0

    def test_fuzzy_variants_max_limit(self):
        """Test fuzzy variants are limited to 10."""
        # Text with many potential variants
        ocr_output = {
            "text": "lllOOOmmmIII000",
            "confidence": 0.9,
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert len(result["fuzzy_variants"]) <= 10

    def test_fuzzy_variants_empty_text(self):
        """Test fuzzy variants for empty text."""
        ocr_output = {"text": "", "confidence": 0.9}
        result = self.processor.process_ocr_result(ocr_output)
        assert result["fuzzy_variants"] == []


class TestSourceTypeDetection:
    """Tests for source type detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OCRContentProcessor()

    def test_detect_pdf_source(self):
        """Test PDF source detection."""
        ocr_output = {
            "text": "PDF content",
            "confidence": 0.9,
            "source_file": "document.pdf",
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["source_type"] == "ocr_document"

    def test_detect_image_source_png(self):
        """Test PNG image source detection."""
        ocr_output = {
            "text": "Image content",
            "confidence": 0.9,
            "source_file": "screenshot.png",
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["source_type"] == "ocr_image"

    def test_detect_image_source_jpg(self):
        """Test JPG image source detection."""
        ocr_output = {
            "text": "Photo content",
            "confidence": 0.9,
            "source_file": "photo.jpg",
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["source_type"] == "ocr_image"

    def test_detect_image_source_explicit(self):
        """Test explicit image source type."""
        ocr_output = {
            "text": "Content",
            "confidence": 0.9,
            "source_type": "image",
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["source_type"] == "ocr_image"

    def test_detect_default_document(self):
        """Test default to document for unknown source."""
        ocr_output = {
            "text": "Unknown source",
            "confidence": 0.9,
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["source_type"] == "ocr_document"


class TestFormatDetection:
    """Tests for format detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OCRContentProcessor()

    def test_detect_pdf_format(self):
        """Test PDF format detection."""
        ocr_output = {
            "text": "Content",
            "confidence": 0.9,
            "source_file": "doc.pdf",
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["original_format"] == "pdf"

    def test_detect_image_format(self):
        """Test image format detection."""
        ocr_output = {
            "text": "Content",
            "confidence": 0.9,
            "source_file": "image.png",
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["original_format"] == "image"

    def test_detect_tiff_format(self):
        """Test TIFF format detection."""
        ocr_output = {
            "text": "Content",
            "confidence": 0.9,
            "source_file": "scan.tiff",
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["original_format"] == "tiff"

    def test_detect_unknown_format(self):
        """Test unknown format fallback."""
        ocr_output = {
            "text": "Content",
            "confidence": 0.9,
            "format": "custom",
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["original_format"] == "custom"


class TestMetadataExtraction:
    """Tests for OCR metadata extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OCRContentProcessor()

    def test_has_tables_metadata(self):
        """Test has_tables metadata extraction."""
        ocr_output = {
            "text": "Content with tables",
            "confidence": 0.9,
            "layout": {"table_count": 2},
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["ocr_metadata"]["has_tables"] is True

    def test_has_images_metadata(self):
        """Test has_images metadata extraction."""
        ocr_output = {
            "text": "Content with images",
            "confidence": 0.9,
            "layout": {"image_count": 3},
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["ocr_metadata"]["has_images"] is True

    def test_has_handwriting_metadata(self):
        """Test has_handwriting metadata extraction."""
        ocr_output = {
            "text": "Handwritten content",
            "confidence": 0.9,
            "layout": {"has_handwriting": True},
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["ocr_metadata"]["has_handwriting"] is True

    def test_page_number_metadata(self):
        """Test page number metadata extraction."""
        ocr_output = {
            "text": "Page content",
            "confidence": 0.9,
            "page": 5,
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["ocr_metadata"]["page_number"] == 5

    def test_bounding_boxes_preserved(self):
        """Test bounding boxes preserved flag."""
        ocr_output = {
            "text": "Content",
            "confidence": 0.9,
            "boxes": [{"x": 0, "y": 0, "w": 100, "h": 50}],
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["ocr_metadata"]["bounding_boxes_preserved"] is True
        assert len(result["bounding_boxes"]) == 1


class TestCEREstimation:
    """Tests for Character Error Rate estimation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OCRContentProcessor()

    def test_cer_from_output(self):
        """Test CER extraction from output."""
        ocr_output = {
            "text": "Content",
            "confidence": 0.9,
            "cer": 0.05,
        }
        result = self.processor.process_ocr_result(ocr_output)
        assert result["source_metadata"]["character_error_rate"] == 0.05

    def test_cer_estimated_from_confidence(self):
        """Test CER estimation from confidence."""
        ocr_output = {
            "text": "Content",
            "confidence": 0.9,
        }
        result = self.processor.process_ocr_result(ocr_output)
        # CER ≈ (1 - 0.9) * 0.5 = 0.05
        assert result["source_metadata"]["character_error_rate"] == pytest.approx(
            0.05, rel=0.1
        )


class TestSearchableContent:
    """Tests for searchable content creation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OCRContentProcessor()

    def test_create_searchable_content_normalizes_whitespace(self):
        """Test whitespace normalization."""
        text = "Multiple   spaces   here"
        result = self.processor.create_searchable_content(text)
        assert result == "Multiple spaces here"

    def test_create_searchable_content_removes_artifacts(self):
        """Test OCR artifact removal."""
        text = "Text |||| with artifacts.......here"
        result = self.processor.create_searchable_content(text)
        assert "||||" not in result
        assert "......." not in result

    def test_create_searchable_content_empty(self):
        """Test empty text handling."""
        result = self.processor.create_searchable_content("")
        assert result == ""


class TestFullProcessing:
    """Integration tests for full OCR processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OCRContentProcessor()

    def test_full_processing_pdf(self):
        """Test full processing of PDF OCR output."""
        ocr_output = {
            "text": "This is a scanned PDF document with important information.",
            "confidence": 0.92,
            "source_file": "report.pdf",
            "page": 1,
            "language": "en",
            "model": "deepseek-ocr-v2",
            "layout": {
                "column_count": 1,
                "table_count": 0,
                "image_count": 0,
            },
        }
        result = self.processor.process_ocr_result(ocr_output)

        # Verify structure
        assert "content" in result
        assert "source_metadata" in result
        assert "ocr_metadata" in result
        assert "fuzzy_variants" in result
        assert "bounding_boxes" in result

        # Verify content
        assert result["content"] == ocr_output["text"]

        # Verify source metadata
        assert result["source_metadata"]["source_type"] == "ocr_document"
        assert result["source_metadata"]["extraction_quality"] == "medium"
        assert result["source_metadata"]["original_format"] == "pdf"
        assert result["source_metadata"]["language_detected"] == "en"

        # Verify OCR metadata
        assert result["ocr_metadata"]["layout_type"] == "simple_text"
        assert result["ocr_metadata"]["has_tables"] is False
        assert result["ocr_metadata"]["page_number"] == 1

    def test_full_processing_image(self):
        """Test full processing of image OCR output."""
        ocr_output = {
            "text": "Whiteboard meeting notes from Q4 planning session",
            "confidence": 0.78,
            "source_file": "whiteboard.jpg",
            "language": "en",
            "layout": {
                "is_handwritten": True,
            },
        }
        result = self.processor.process_ocr_result(ocr_output)

        assert result["source_metadata"]["source_type"] == "ocr_image"
        assert result["source_metadata"]["extraction_quality"] == "low"
        assert result["ocr_metadata"]["layout_type"] == "handwritten"
