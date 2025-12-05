"""Tests for ImageAdapter.

Module 08: Multimodal Integration & Tool Enhancement - Phase 2 Tests
Tests cover image normalization, OCR processing, privacy checks, and layout metadata.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from futurnal.pipeline.normalization.adapters.image import ImageAdapter
from futurnal.pipeline.models import DocumentFormat, NormalizedDocument
from futurnal.extraction.ocr_client import (
    OCRResult,
    LayoutInfo,
    TextRegion,
    BoundingBox,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def image_adapter():
    """Create ImageAdapter instance for testing."""
    return ImageAdapter()


@pytest.fixture
def mock_image_file(tmp_path):
    """Create temporary image file for testing."""
    image_file = tmp_path / "test_screenshot.png"
    # Write mock PNG header + data
    image_file.write_bytes(
        b'\x89PNG\r\n\x1a\n' + b'MOCK_IMAGE_DATA' * 1000  # ~15KB
    )
    return image_file


@pytest.fixture
def mock_ocr_result():
    """Mock OCR result from DeepSeek/Tesseract."""
    regions = [
        TextRegion(
            text="Invoice",
            bbox=BoundingBox(10, 10, 100, 30),
            confidence=0.98,
            region_type="heading"
        ),
        TextRegion(
            text="Date: January 15, 2024",
            bbox=BoundingBox(10, 40, 200, 60),
            confidence=0.96,
            region_type="paragraph"
        ),
        TextRegion(
            text="Amount: $1,234.56",
            bbox=BoundingBox(10, 70, 180, 90),
            confidence=0.97,
            region_type="paragraph"
        ),
        TextRegion(
            text="Payment due: February 15, 2024",
            bbox=BoundingBox(10, 100, 220, 120),
            confidence=0.95,
            region_type="paragraph"
        )
    ]

    layout = LayoutInfo(
        page_count=1,
        regions=regions,
        reading_order=[0, 1, 2, 3]
    )

    return OCRResult(
        text="Invoice\nDate: January 15, 2024\nAmount: $1,234.56\nPayment due: February 15, 2024",
        layout=layout,
        regions=regions,
        confidence=0.97
    )


# =============================================================================
# ImageAdapter Basic Tests
# =============================================================================


class TestImageAdapterBasics:
    """Test ImageAdapter initialization and basic properties."""

    def test_initialization(self, image_adapter):
        """Test ImageAdapter initializes correctly."""
        assert image_adapter.name == "ImageAdapter"
        assert DocumentFormat.IMAGE in image_adapter.supported_formats
        assert image_adapter.requires_unstructured_processing is False

    def test_supported_formats(self, image_adapter):
        """Test ImageAdapter supports only IMAGE format."""
        assert len(image_adapter.supported_formats) == 1
        assert image_adapter.supported_formats[0] == DocumentFormat.IMAGE

    @pytest.mark.asyncio
    async def test_validate_existing_file(self, image_adapter, mock_image_file):
        """Test validate() accepts existing image file."""
        is_valid = await image_adapter.validate(mock_image_file)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self, image_adapter, tmp_path):
        """Test validate() rejects non-existent file."""
        nonexistent = tmp_path / "nonexistent.png"
        is_valid = await image_adapter.validate(nonexistent)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_wrong_extension(self, image_adapter, tmp_path):
        """Test validate() rejects files with wrong extension."""
        wrong_ext = tmp_path / "test.txt"
        wrong_ext.write_text("not an image")
        is_valid = await image_adapter.validate(wrong_ext)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_various_image_formats(self, image_adapter, tmp_path):
        """Test validate() accepts various image formats."""
        extensions = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"]

        for ext in extensions:
            image_file = tmp_path / f"test{ext}"
            image_file.write_bytes(b"MOCK_IMAGE" * 100)
            is_valid = await image_adapter.validate(image_file)
            assert is_valid is True, f"Should accept {ext} extension"


# =============================================================================
# ImageAdapter Normalization Tests
# =============================================================================


class TestImageAdapterNormalization:
    """Test ImageAdapter normalization functionality."""

    @pytest.mark.asyncio
    async def test_normalize_success(self, image_adapter, mock_image_file, mock_ocr_result):
        """Test successful image normalization end-to-end."""
        with patch.object(image_adapter, "_get_ocr_client") as mock_get_client:
            # Mock OCR client
            mock_client = Mock()
            mock_client.extract_text.return_value = mock_ocr_result
            mock_get_client.return_value = mock_client

            # Normalize image file
            result = await image_adapter.normalize(
                file_path=mock_image_file,
                source_id="image-123",
                source_type="local_files",
                source_metadata={}
            )

            # Verify result structure
            assert isinstance(result, NormalizedDocument)
            assert result.metadata.format == DocumentFormat.IMAGE
            assert result.metadata.source_type == "local_files"
            assert result.metadata.source_id == "image-123"

            # Verify content is OCR extracted text
            assert result.content == mock_ocr_result.text
            assert len(result.content) > 0
            assert "Invoice" in result.content
            assert "January 15, 2024" in result.content

            # Verify OCR regions stored in metadata
            assert "ocr_regions" in result.metadata.extra
            regions = result.metadata.extra["ocr_regions"]
            assert len(regions) == 4

            # Verify first region structure
            assert regions[0]["text"] == "Invoice"
            assert regions[0]["bbox"]["x1"] == 10
            assert regions[0]["bbox"]["y1"] == 10
            assert regions[0]["bbox"]["x2"] == 100
            assert regions[0]["bbox"]["y2"] == 30
            assert regions[0]["confidence"] == 0.98
            assert regions[0]["type"] == "heading"

            # Verify OCR metadata
            assert "ocr" in result.metadata.extra
            assert result.metadata.extra["ocr"]["confidence"] == 0.97
            assert result.metadata.extra["ocr"]["region_count"] == 4
            assert result.metadata.extra["ocr"]["layout_preserved"] is True
            assert "ocr_backend" in result.metadata.extra["ocr"]

            # Verify layout information
            assert "layout_info" in result.metadata.extra
            assert result.metadata.extra["layout_info"]["page_count"] == 1
            assert result.metadata.extra["layout_info"]["reading_order"] == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_normalize_nonexistent_file(self, image_adapter, tmp_path):
        """Test normalization fails gracefully for non-existent file."""
        nonexistent = tmp_path / "missing.png"

        with pytest.raises(Exception, match="Image file not found"):
            await image_adapter.normalize(
                file_path=nonexistent,
                source_id="image-123",
                source_type="local_files",
                source_metadata={}
            )

    @pytest.mark.asyncio
    async def test_normalize_large_file_warning(self, image_adapter, tmp_path, mock_ocr_result):
        """Test warning logged for large image files (>10MB)."""
        # Create large mock file (>10MB)
        large_image = tmp_path / "large_screenshot.png"
        large_image.write_bytes(b"X" * (11 * 1024 * 1024))  # 11MB

        with patch.object(image_adapter, "_get_ocr_client") as mock_get_client:
            mock_client = Mock()
            mock_client.extract_text.return_value = mock_ocr_result
            mock_get_client.return_value = mock_client

            with patch("futurnal.pipeline.normalization.adapters.image.logger") as mock_logger:
                await image_adapter.normalize(
                    file_path=large_image,
                    source_id="large-image",
                    source_type="local_files",
                    source_metadata={}
                )

                # Verify warning was logged
                warning_calls = [call for call in mock_logger.warning.call_args_list]
                assert any("Large image file" in str(call) for call in warning_calls)

    @pytest.mark.asyncio
    async def test_normalize_ocr_error(self, image_adapter, mock_image_file):
        """Test normalization handles OCR errors gracefully."""
        with patch.object(image_adapter, "_get_ocr_client") as mock_get_client:
            mock_client = Mock()
            mock_client.extract_text.side_effect = Exception("OCR processing failed")
            mock_get_client.return_value = mock_client

            with pytest.raises(Exception, match="Failed to normalize image file"):
                await image_adapter.normalize(
                    file_path=mock_image_file,
                    source_id="image-123",
                    source_type="local_files",
                    source_metadata={}
                )


# =============================================================================
# OCR Region Tests
# =============================================================================


class TestImageAdapterOCRRegions:
    """Test OCR region extraction and storage."""

    @pytest.mark.asyncio
    async def test_ocr_regions_structure(self, image_adapter, mock_image_file, mock_ocr_result):
        """Test OCR regions are correctly structured."""
        with patch.object(image_adapter, "_get_ocr_client") as mock_get_client:
            mock_client = Mock()
            mock_client.extract_text.return_value = mock_ocr_result
            mock_get_client.return_value = mock_client

            result = await image_adapter.normalize(
                file_path=mock_image_file,
                source_id="image-123",
                source_type="local_files",
                source_metadata={}
            )

            regions = result.metadata.extra["ocr_regions"]

            # Verify all regions have required fields
            for region in regions:
                assert "text" in region
                assert "bbox" in region
                assert "confidence" in region
                assert "type" in region

                # Verify bbox structure
                bbox = region["bbox"]
                assert "x1" in bbox
                assert "y1" in bbox
                assert "x2" in bbox
                assert "y2" in bbox

                # Verify confidence is valid
                assert 0.0 <= region["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_ocr_regions_bounding_boxes(self, image_adapter, mock_image_file, mock_ocr_result):
        """Test OCR bounding boxes are valid."""
        with patch.object(image_adapter, "_get_ocr_client") as mock_get_client:
            mock_client = Mock()
            mock_client.extract_text.return_value = mock_ocr_result
            mock_get_client.return_value = mock_client

            result = await image_adapter.normalize(
                file_path=mock_image_file,
                source_id="image-123",
                source_type="local_files",
                source_metadata={}
            )

            regions = result.metadata.extra["ocr_regions"]

            # Verify bounding boxes are valid
            for region in regions:
                bbox = region["bbox"]
                assert bbox["x2"] > bbox["x1"], "x2 should be greater than x1"
                assert bbox["y2"] > bbox["y1"], "y2 should be greater than y1"
                assert bbox["x1"] >= 0, "x1 should be non-negative"
                assert bbox["y1"] >= 0, "y1 should be non-negative"

    @pytest.mark.asyncio
    async def test_ocr_regions_types(self, image_adapter, mock_image_file, mock_ocr_result):
        """Test OCR region types are preserved."""
        with patch.object(image_adapter, "_get_ocr_client") as mock_get_client:
            mock_client = Mock()
            mock_client.extract_text.return_value = mock_ocr_result
            mock_get_client.return_value = mock_client

            result = await image_adapter.normalize(
                file_path=mock_image_file,
                source_id="image-123",
                source_type="local_files",
                source_metadata={}
            )

            regions = result.metadata.extra["ocr_regions"]

            # Verify region types
            assert regions[0]["type"] == "heading"
            assert regions[1]["type"] == "paragraph"
            assert regions[2]["type"] == "paragraph"
            assert regions[3]["type"] == "paragraph"

    @pytest.mark.asyncio
    async def test_layout_info_reading_order(self, image_adapter, mock_image_file, mock_ocr_result):
        """Test layout reading order is preserved."""
        with patch.object(image_adapter, "_get_ocr_client") as mock_get_client:
            mock_client = Mock()
            mock_client.extract_text.return_value = mock_ocr_result
            mock_get_client.return_value = mock_client

            result = await image_adapter.normalize(
                file_path=mock_image_file,
                source_id="image-123",
                source_type="local_files",
                source_metadata={}
            )

            layout_info = result.metadata.extra["layout_info"]

            assert layout_info["page_count"] == 1
            assert layout_info["reading_order"] == [0, 1, 2, 3]


# =============================================================================
# Metadata Tests
# =============================================================================


class TestImageAdapterMetadata:
    """Test metadata extraction and enrichment."""

    @pytest.mark.asyncio
    async def test_ocr_metadata_populated(self, image_adapter, mock_image_file, mock_ocr_result):
        """Test OCR-specific metadata is populated."""
        with patch.object(image_adapter, "_get_ocr_client") as mock_get_client:
            mock_client = Mock()
            mock_client.extract_text.return_value = mock_ocr_result
            mock_get_client.return_value = mock_client

            result = await image_adapter.normalize(
                file_path=mock_image_file,
                source_id="image-123",
                source_type="local_files",
                source_metadata={"custom_key": "custom_value"}
            )

            ocr_meta = result.metadata.extra["ocr"]

            assert ocr_meta["confidence"] == 0.97
            assert ocr_meta["region_count"] == 4
            assert ocr_meta["layout_preserved"] is True
            assert "ocr_backend" in ocr_meta

            # Verify custom source metadata is preserved
            assert result.metadata.extra["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_file_size_metadata(self, image_adapter, mock_image_file, mock_ocr_result):
        """Test file size metadata is calculated."""
        with patch.object(image_adapter, "_get_ocr_client") as mock_get_client:
            mock_client = Mock()
            mock_client.extract_text.return_value = mock_ocr_result
            mock_get_client.return_value = mock_client

            result = await image_adapter.normalize(
                file_path=mock_image_file,
                source_id="image-123",
                source_type="local_files",
                source_metadata={}
            )

            # Verify file size is recorded
            assert "image_file_size_mb" in result.metadata.extra
            file_size_mb = result.metadata.extra["image_file_size_mb"]
            assert file_size_mb > 0

            # Verify against actual file size
            actual_size_mb = mock_image_file.stat().st_size / (1024 * 1024)
            assert abs(file_size_mb - actual_size_mb) < 0.01  # Within 10KB

    @pytest.mark.asyncio
    async def test_high_confidence_detection(self, image_adapter, mock_image_file, mock_ocr_result):
        """Test high confidence OCR is detected."""
        with patch.object(image_adapter, "_get_ocr_client") as mock_get_client:
            mock_client = Mock()
            mock_client.extract_text.return_value = mock_ocr_result
            mock_get_client.return_value = mock_client

            result = await image_adapter.normalize(
                file_path=mock_image_file,
                source_id="image-123",
                source_type="local_files",
                source_metadata={}
            )

            # Verify high confidence (>0.8) is recorded
            assert result.metadata.extra["ocr"]["confidence"] > 0.8

    @pytest.mark.asyncio
    async def test_low_confidence_detection(self, image_adapter, mock_image_file):
        """Test low confidence OCR is detected."""
        # Create low-confidence OCR result
        low_confidence_result = OCRResult(
            text="Blurry text",
            layout=LayoutInfo(
                page_count=1,
                regions=[
                    TextRegion(
                        text="Blurry text",
                        bbox=BoundingBox(0, 0, 100, 20),
                        confidence=0.45,
                        region_type="paragraph"
                    )
                ],
                reading_order=[0]
            ),
            regions=[
                TextRegion(
                    text="Blurry text",
                    bbox=BoundingBox(0, 0, 100, 20),
                    confidence=0.45,
                    region_type="paragraph"
                )
            ],
            confidence=0.45
        )

        with patch.object(image_adapter, "_get_ocr_client") as mock_get_client:
            mock_client = Mock()
            mock_client.extract_text.return_value = low_confidence_result
            mock_get_client.return_value = mock_client

            result = await image_adapter.normalize(
                file_path=mock_image_file,
                source_id="image-123",
                source_type="local_files",
                source_metadata={}
            )

            # Verify low confidence is recorded
            assert result.metadata.extra["ocr"]["confidence"] < 0.8


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestImageAdapterIntegration:
    """Integration tests requiring real OCR backend."""

    @pytest.mark.skipif(True, reason="Requires Tesseract OCR - manual testing only")
    @pytest.mark.asyncio
    async def test_real_ocr_processing(self, image_adapter, tmp_path):
        """Test real OCR processing (requires Tesseract or DeepSeek-OCR)."""
        # This test requires:
        # 1. Tesseract: brew install tesseract (macOS) or apt-get install tesseract-ocr
        # 2. OR DeepSeek-OCR: pip install transformers torch
        # 3. Real image file with text

        # For manual testing only
        image_file = tmp_path / "real_image.png"
        # User must provide real image file

        if not image_file.exists():
            pytest.skip("Real image file not provided")

        result = await image_adapter.normalize(
            file_path=image_file,
            source_id="integration-test",
            source_type="local_files",
            source_metadata={}
        )

        assert isinstance(result, NormalizedDocument)
        assert len(result.content) > 0
        assert len(result.metadata.extra["ocr_regions"]) > 0
