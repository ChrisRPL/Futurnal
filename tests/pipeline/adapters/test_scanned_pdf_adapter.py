"""Tests for ScannedPDFAdapter.

Module 08: Multimodal Integration & Tool Enhancement - Phase 2 Tests
Tests cover scanned PDF normalization, multi-page OCR, and page boundary preservation.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from futurnal.pipeline.normalization.adapters.scanned_pdf import ScannedPDFAdapter
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
def scanned_pdf_adapter():
    """Create ScannedPDFAdapter instance for testing."""
    return ScannedPDFAdapter()


@pytest.fixture
def mock_scanned_pdf(tmp_path):
    """Create temporary scanned PDF file for testing."""
    pdf_file = tmp_path / "scanned_invoice.pdf"
    # Write mock PDF header + data
    pdf_file.write_bytes(
        b'%PDF-1.4\n' + b'MOCK_SCANNED_PDF_DATA' * 1000  # ~20KB
    )
    return pdf_file


@pytest.fixture
def mock_pdf_images(tmp_path):
    """Mock PDF pages as images."""
    images = []
    for i in range(3):
        img_path = tmp_path / f"page_{i}.png"
        img_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'MOCK_PAGE_DATA' * 100)
        images.append(img_path)
    return images


@pytest.fixture
def mock_ocr_results_multipage():
    """Mock OCR results for multi-page PDF."""
    # Page 1 - Invoice header
    page1_regions = [
        TextRegion(
            text="INVOICE",
            bbox=BoundingBox(50, 50, 200, 80),
            confidence=0.98,
            region_type="heading"
        ),
        TextRegion(
            text="Invoice #: INV-2024-001",
            bbox=BoundingBox(50, 100, 250, 120),
            confidence=0.97,
            region_type="paragraph"
        )
    ]

    page1_result = OCRResult(
        text="INVOICE\nInvoice #: INV-2024-001",
        layout=LayoutInfo(page_count=1, regions=page1_regions, reading_order=[0, 1]),
        regions=page1_regions,
        confidence=0.975
    )

    # Page 2 - Line items
    page2_regions = [
        TextRegion(
            text="Description: Consulting Services",
            bbox=BoundingBox(50, 50, 300, 70),
            confidence=0.96,
            region_type="paragraph"
        ),
        TextRegion(
            text="Amount: $5,000.00",
            bbox=BoundingBox(50, 80, 200, 100),
            confidence=0.98,
            region_type="paragraph"
        )
    ]

    page2_result = OCRResult(
        text="Description: Consulting Services\nAmount: $5,000.00",
        layout=LayoutInfo(page_count=1, regions=page2_regions, reading_order=[0, 1]),
        regions=page2_regions,
        confidence=0.97
    )

    # Page 3 - Footer
    page3_regions = [
        TextRegion(
            text="Payment due: February 15, 2024",
            bbox=BoundingBox(50, 50, 280, 70),
            confidence=0.95,
            region_type="paragraph"
        ),
        TextRegion(
            text="Thank you for your business",
            bbox=BoundingBox(50, 80, 250, 100),
            confidence=0.96,
            region_type="paragraph"
        )
    ]

    page3_result = OCRResult(
        text="Payment due: February 15, 2024\nThank you for your business",
        layout=LayoutInfo(page_count=1, regions=page3_regions, reading_order=[0, 1]),
        regions=page3_regions,
        confidence=0.955
    )

    return [page1_result, page2_result, page3_result]


# =============================================================================
# ScannedPDFAdapter Basic Tests
# =============================================================================


class TestScannedPDFAdapterBasics:
    """Test ScannedPDFAdapter initialization and basic properties."""

    def test_initialization(self, scanned_pdf_adapter):
        """Test ScannedPDFAdapter initializes correctly."""
        assert scanned_pdf_adapter.name == "ScannedPDFAdapter"
        assert DocumentFormat.SCANNED_PDF in scanned_pdf_adapter.supported_formats
        assert scanned_pdf_adapter.requires_unstructured_processing is False

    def test_supported_formats(self, scanned_pdf_adapter):
        """Test ScannedPDFAdapter supports only SCANNED_PDF format."""
        assert len(scanned_pdf_adapter.supported_formats) == 1
        assert scanned_pdf_adapter.supported_formats[0] == DocumentFormat.SCANNED_PDF

    @pytest.mark.asyncio
    async def test_validate_existing_file(self, scanned_pdf_adapter, mock_scanned_pdf):
        """Test validate() accepts existing PDF file."""
        is_valid = await scanned_pdf_adapter.validate(mock_scanned_pdf)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self, scanned_pdf_adapter, tmp_path):
        """Test validate() rejects non-existent file."""
        nonexistent = tmp_path / "nonexistent.pdf"
        is_valid = await scanned_pdf_adapter.validate(nonexistent)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_wrong_extension(self, scanned_pdf_adapter, tmp_path):
        """Test validate() rejects files with wrong extension."""
        wrong_ext = tmp_path / "test.txt"
        wrong_ext.write_text("not a pdf")
        is_valid = await scanned_pdf_adapter.validate(wrong_ext)
        assert is_valid is False


# =============================================================================
# PDF to Images Conversion Tests
# =============================================================================


class TestPDFToImagesConversion:
    """Test PDF to images conversion functionality."""

    @pytest.mark.asyncio
    async def test_pdf_to_images_success(self, scanned_pdf_adapter, mock_scanned_pdf, tmp_path):
        """Test successful PDF to images conversion."""
        with patch("pdf2image.convert_from_path") as mock_convert:
            # Mock paths to images (pdf2image can return paths)
            mock_image_paths = [
                str(tmp_path / f"page_{i:04d}.png") for i in range(3)
            ]
            # Create the mock files
            for path_str in mock_image_paths:
                Path(path_str).write_bytes(b"MOCK_IMAGE")

            mock_convert.return_value = mock_image_paths

            images = await scanned_pdf_adapter._pdf_to_images(mock_scanned_pdf)

            # Verify conversion was called with correct parameters
            mock_convert.assert_called_once()
            call_kwargs = mock_convert.call_args.kwargs
            assert call_kwargs["dpi"] == 300
            assert call_kwargs["fmt"] == "png"

            # Verify images were created
            assert len(images) == 3
            for img_path in images:
                assert img_path.exists()
                assert img_path.suffix == ".png"

    @pytest.mark.asyncio
    async def test_pdf_to_images_nonexistent_file(self, scanned_pdf_adapter, tmp_path):
        """Test PDF to images fails for non-existent file."""
        nonexistent = tmp_path / "missing.pdf"

        # pdf2image will raise an exception when file doesn't exist
        with pytest.raises(Exception):  # pdf2image will raise PDFInfoNotInstalledError or FileNotFoundError
            await scanned_pdf_adapter._pdf_to_images(nonexistent)

    @pytest.mark.asyncio
    async def test_pdf_to_images_conversion_error(self, scanned_pdf_adapter, mock_scanned_pdf):
        """Test PDF to images handles conversion errors."""
        with patch("pdf2image.convert_from_path") as mock_convert:
            mock_convert.side_effect = Exception("PDF conversion failed")

            # Exception should propagate without wrapping
            with pytest.raises(Exception, match="PDF conversion failed"):
                await scanned_pdf_adapter._pdf_to_images(mock_scanned_pdf)


# =============================================================================
# ScannedPDFAdapter Normalization Tests
# =============================================================================


class TestScannedPDFAdapterNormalization:
    """Test ScannedPDFAdapter normalization functionality."""

    @pytest.mark.asyncio
    async def test_normalize_success_multipage(
        self,
        scanned_pdf_adapter,
        mock_scanned_pdf,
        mock_pdf_images,
        mock_ocr_results_multipage
    ):
        """Test successful multi-page PDF normalization."""
        with patch.object(scanned_pdf_adapter, "_pdf_to_images") as mock_pdf_to_images, \
             patch.object(scanned_pdf_adapter, "_get_ocr_client") as mock_get_client:

            # Mock PDF to images
            mock_pdf_to_images.return_value = mock_pdf_images

            # Mock OCR client
            mock_client = Mock()
            mock_client.extract_text.side_effect = mock_ocr_results_multipage
            mock_get_client.return_value = mock_client

            # Normalize PDF
            result = await scanned_pdf_adapter.normalize(
                file_path=mock_scanned_pdf,
                source_id="scanned-pdf-123",
                source_type="local_files",
                source_metadata={}
            )

            # Verify result structure
            assert isinstance(result, NormalizedDocument)
            assert result.metadata.format == DocumentFormat.SCANNED_PDF
            assert result.metadata.source_id == "scanned-pdf-123"

            # Verify content contains all pages with page breaks
            assert "INVOICE" in result.content
            assert "Consulting Services" in result.content
            assert "Thank you for your business" in result.content
            assert result.content.count("---PAGE BREAK---") == 2  # 3 pages = 2 breaks

            # Verify pages metadata
            assert "pages" in result.metadata.extra
            pages = result.metadata.extra["pages"]
            assert len(pages) == 3

            # Verify page 1 metadata
            assert pages[0]["page_number"] == 1
            assert pages[0]["confidence"] == 0.975
            assert pages[0]["region_count"] == 2
            assert pages[0]["text_length"] > 0

            # Verify page 2 metadata
            assert pages[1]["page_number"] == 2
            assert pages[1]["text_length"] > 0

            # Verify page 3 metadata
            assert pages[2]["page_number"] == 3
            assert pages[2]["text_length"] > 0

            # Verify overall OCR metadata
            assert "ocr" in result.metadata.extra
            assert result.metadata.extra["ocr"]["page_count"] == 3
            assert result.metadata.extra["ocr"]["total_regions"] == 6  # 2+2+2

    @pytest.mark.asyncio
    async def test_normalize_single_page(
        self,
        scanned_pdf_adapter,
        mock_scanned_pdf,
        tmp_path
    ):
        """Test single page PDF normalization."""
        # Create single page image
        single_page = [tmp_path / "page_0.png"]
        single_page[0].write_bytes(b'\x89PNG\r\n\x1a\n' + b'DATA')

        # Single page OCR result
        single_result = OCRResult(
            text="Simple document",
            layout=LayoutInfo(
                page_count=1,
                regions=[
                    TextRegion(
                        text="Simple document",
                        bbox=BoundingBox(0, 0, 100, 20),
                        confidence=0.95,
                        region_type="paragraph"
                    )
                ],
                reading_order=[0]
            ),
            regions=[
                TextRegion(
                    text="Simple document",
                    bbox=BoundingBox(0, 0, 100, 20),
                    confidence=0.95,
                    region_type="paragraph"
                )
            ],
            confidence=0.95
        )

        with patch.object(scanned_pdf_adapter, "_pdf_to_images") as mock_pdf_to_images, \
             patch.object(scanned_pdf_adapter, "_get_ocr_client") as mock_get_client:

            mock_pdf_to_images.return_value = single_page
            mock_client = Mock()
            mock_client.extract_text.return_value = single_result
            mock_get_client.return_value = mock_client

            result = await scanned_pdf_adapter.normalize(
                file_path=mock_scanned_pdf,
                source_id="single-page",
                source_type="local_files",
                source_metadata={}
            )

            # Verify single page result
            assert result.content == "Simple document"
            assert "---PAGE BREAK---" not in result.content  # No breaks for single page
            assert len(result.metadata.extra["pages"]) == 1

    @pytest.mark.asyncio
    async def test_normalize_nonexistent_file(self, scanned_pdf_adapter, tmp_path):
        """Test normalization fails for non-existent file."""
        nonexistent = tmp_path / "missing.pdf"

        with pytest.raises(Exception, match="PDF file not found"):
            await scanned_pdf_adapter.normalize(
                file_path=nonexistent,
                source_id="scanned-pdf-123",
                source_type="local_files",
                source_metadata={}
            )

    @pytest.mark.asyncio
    async def test_normalize_pdf_conversion_error(self, scanned_pdf_adapter, mock_scanned_pdf):
        """Test normalization handles PDF conversion errors."""
        with patch.object(scanned_pdf_adapter, "_pdf_to_images") as mock_pdf_to_images:
            mock_pdf_to_images.side_effect = Exception("PDF conversion failed")

            with pytest.raises(Exception, match="Failed to normalize scanned PDF"):
                await scanned_pdf_adapter.normalize(
                    file_path=mock_scanned_pdf,
                    source_id="error-pdf",
                    source_type="local_files",
                    source_metadata={}
                )

    @pytest.mark.asyncio
    async def test_normalize_ocr_error_one_page(
        self,
        scanned_pdf_adapter,
        mock_scanned_pdf,
        mock_pdf_images
    ):
        """Test normalization handles OCR errors on specific pages."""
        with patch.object(scanned_pdf_adapter, "_pdf_to_images") as mock_pdf_to_images, \
             patch.object(scanned_pdf_adapter, "_get_ocr_client") as mock_get_client:

            mock_pdf_to_images.return_value = mock_pdf_images

            # Mock OCR client that fails on page 2
            mock_client = Mock()
            mock_client.extract_text.side_effect = [
                OCRResult(
                    text="Page 1 text",
                    layout=LayoutInfo(page_count=1, regions=[], reading_order=[]),
                    regions=[],
                    confidence=0.95
                ),
                Exception("OCR failed on page 2"),
                OCRResult(
                    text="Page 3 text",
                    layout=LayoutInfo(page_count=1, regions=[], reading_order=[]),
                    regions=[],
                    confidence=0.95
                )
            ]
            mock_get_client.return_value = mock_client

            with pytest.raises(Exception, match="Failed to normalize scanned PDF"):
                await scanned_pdf_adapter.normalize(
                    file_path=mock_scanned_pdf,
                    source_id="error-pdf",
                    source_type="local_files",
                    source_metadata={}
                )

    @pytest.mark.asyncio
    async def test_normalize_large_file_warning(
        self,
        scanned_pdf_adapter,
        tmp_path,
        mock_pdf_images,
        mock_ocr_results_multipage
    ):
        """Test warning logged for large PDF files (>50MB)."""
        # Create large mock PDF (>50MB)
        large_pdf = tmp_path / "large_document.pdf"
        large_pdf.write_bytes(b"X" * (51 * 1024 * 1024))  # 51MB

        with patch.object(scanned_pdf_adapter, "_pdf_to_images") as mock_pdf_to_images, \
             patch.object(scanned_pdf_adapter, "_get_ocr_client") as mock_get_client:

            mock_pdf_to_images.return_value = mock_pdf_images
            mock_client = Mock()
            mock_client.extract_text.side_effect = mock_ocr_results_multipage
            mock_get_client.return_value = mock_client

            with patch("futurnal.pipeline.normalization.adapters.scanned_pdf.logger") as mock_logger:
                await scanned_pdf_adapter.normalize(
                    file_path=large_pdf,
                    source_id="large-pdf",
                    source_type="local_files",
                    source_metadata={}
                )

                # Verify warning was logged
                warning_calls = [call for call in mock_logger.warning.call_args_list]
                assert any("Large PDF file" in str(call) for call in warning_calls)


# =============================================================================
# Page Metadata Tests
# =============================================================================


class TestScannedPDFPageMetadata:
    """Test per-page metadata extraction."""

    @pytest.mark.asyncio
    async def test_page_metadata_structure(
        self,
        scanned_pdf_adapter,
        mock_scanned_pdf,
        mock_pdf_images,
        mock_ocr_results_multipage
    ):
        """Test per-page metadata is correctly structured."""
        with patch.object(scanned_pdf_adapter, "_pdf_to_images") as mock_pdf_to_images, \
             patch.object(scanned_pdf_adapter, "_get_ocr_client") as mock_get_client:

            mock_pdf_to_images.return_value = mock_pdf_images
            mock_client = Mock()
            mock_client.extract_text.side_effect = mock_ocr_results_multipage
            mock_get_client.return_value = mock_client

            result = await scanned_pdf_adapter.normalize(
                file_path=mock_scanned_pdf,
                source_id="scanned-pdf-123",
                source_type="local_files",
                source_metadata={}
            )

            pages = result.metadata.extra["pages"]

            # Verify all pages have required fields
            for page in pages:
                assert "page_number" in page
                assert "text_length" in page
                assert "confidence" in page
                assert "region_count" in page

    @pytest.mark.asyncio
    async def test_page_boundaries_preserved(
        self,
        scanned_pdf_adapter,
        mock_scanned_pdf,
        mock_pdf_images,
        mock_ocr_results_multipage
    ):
        """Test page boundaries are preserved in merged text."""
        with patch.object(scanned_pdf_adapter, "_pdf_to_images") as mock_pdf_to_images, \
             patch.object(scanned_pdf_adapter, "_get_ocr_client") as mock_get_client:

            mock_pdf_to_images.return_value = mock_pdf_images
            mock_client = Mock()
            mock_client.extract_text.side_effect = mock_ocr_results_multipage
            mock_get_client.return_value = mock_client

            result = await scanned_pdf_adapter.normalize(
                file_path=mock_scanned_pdf,
                source_id="scanned-pdf-123",
                source_type="local_files",
                source_metadata={}
            )

            # Verify page breaks are in correct positions
            content = result.content
            page_sections = content.split("---PAGE BREAK---")

            assert len(page_sections) == 3
            assert "INVOICE" in page_sections[0]
            assert "Consulting Services" in page_sections[1]
            assert "Thank you for your business" in page_sections[2]

    @pytest.mark.asyncio
    async def test_average_confidence_calculation(
        self,
        scanned_pdf_adapter,
        mock_scanned_pdf,
        mock_pdf_images,
        mock_ocr_results_multipage
    ):
        """Test average confidence is calculated correctly."""
        with patch.object(scanned_pdf_adapter, "_pdf_to_images") as mock_pdf_to_images, \
             patch.object(scanned_pdf_adapter, "_get_ocr_client") as mock_get_client:

            mock_pdf_to_images.return_value = mock_pdf_images
            mock_client = Mock()
            mock_client.extract_text.side_effect = mock_ocr_results_multipage
            mock_get_client.return_value = mock_client

            result = await scanned_pdf_adapter.normalize(
                file_path=mock_scanned_pdf,
                source_id="scanned-pdf-123",
                source_type="local_files",
                source_metadata={}
            )

            # Calculate expected average
            confidences = [0.975, 0.97, 0.955]
            expected_avg = sum(confidences) / len(confidences)

            # Verify average confidence
            avg_confidence = result.metadata.extra["ocr"]["average_confidence"]
            assert abs(avg_confidence - expected_avg) < 0.001


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestScannedPDFCleanup:
    """Test temporary file cleanup."""

    @pytest.mark.asyncio
    async def test_temporary_images_cleanup_success(
        self,
        scanned_pdf_adapter,
        mock_scanned_pdf,
        mock_pdf_images,
        mock_ocr_results_multipage
    ):
        """Test temporary images are cleaned up after successful processing."""
        with patch.object(scanned_pdf_adapter, "_pdf_to_images") as mock_pdf_to_images, \
             patch.object(scanned_pdf_adapter, "_get_ocr_client") as mock_get_client:

            mock_pdf_to_images.return_value = mock_pdf_images
            mock_client = Mock()
            mock_client.extract_text.side_effect = mock_ocr_results_multipage
            mock_get_client.return_value = mock_client

            await scanned_pdf_adapter.normalize(
                file_path=mock_scanned_pdf,
                source_id="scanned-pdf-123",
                source_type="local_files",
                source_metadata={}
            )

            # Verify temporary images were deleted
            for img_path in mock_pdf_images:
                assert not img_path.exists()

    @pytest.mark.asyncio
    async def test_temporary_images_cleanup_on_error(
        self,
        scanned_pdf_adapter,
        mock_scanned_pdf,
        mock_pdf_images
    ):
        """Test temporary images are cleaned up even on OCR error."""
        with patch.object(scanned_pdf_adapter, "_pdf_to_images") as mock_pdf_to_images, \
             patch.object(scanned_pdf_adapter, "_get_ocr_client") as mock_get_client:

            mock_pdf_to_images.return_value = mock_pdf_images
            mock_client = Mock()
            mock_client.extract_text.side_effect = Exception("OCR failed")
            mock_get_client.return_value = mock_client

            with pytest.raises(Exception):
                await scanned_pdf_adapter.normalize(
                    file_path=mock_scanned_pdf,
                    source_id="error-pdf",
                    source_type="local_files",
                    source_metadata={}
                )

            # Verify temporary images were deleted despite error
            for img_path in mock_pdf_images:
                assert not img_path.exists()


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestScannedPDFAdapterIntegration:
    """Integration tests requiring real OCR backend and pdf2image."""

    @pytest.mark.skipif(True, reason="Requires pdf2image and Tesseract - manual testing only")
    @pytest.mark.asyncio
    async def test_real_scanned_pdf_processing(self, scanned_pdf_adapter, tmp_path):
        """Test real scanned PDF processing (requires pdf2image + Tesseract)."""
        # This test requires:
        # 1. pdf2image: pip install pdf2image
        # 2. Poppler: brew install poppler (macOS) or apt-get install poppler-utils
        # 3. Tesseract: brew install tesseract (macOS) or apt-get install tesseract-ocr
        # 4. Real scanned PDF file

        # For manual testing only
        pdf_file = tmp_path / "real_scanned.pdf"
        # User must provide real scanned PDF file

        if not pdf_file.exists():
            pytest.skip("Real scanned PDF file not provided")

        result = await scanned_pdf_adapter.normalize(
            file_path=pdf_file,
            source_id="integration-test",
            source_type="local_files",
            source_metadata={}
        )

        assert isinstance(result, NormalizedDocument)
        assert len(result.content) > 0
        assert len(result.metadata.extra["pages"]) > 0
