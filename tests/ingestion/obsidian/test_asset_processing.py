"""Tests for asset processing pipeline functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from futurnal.ingestion.obsidian.assets import AssetPipeline
from futurnal.ingestion.obsidian.asset_processor import (
    AssetTextExtractor,
    AssetTextExtractorConfig,
    AssetProcessingPipeline,
    UNSTRUCTURED_AVAILABLE
)


class TestAssetPipeline:
    """Test cases for AssetPipeline integration."""

    @pytest.fixture
    def temp_vault(self):
        """Create a temporary vault with test assets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)

            # Create directory structure
            (vault_root / "assets").mkdir()
            (vault_root / "images").mkdir()

            # Create test files
            (vault_root / "assets" / "diagram.png").write_bytes(b"fake PNG content")
            (vault_root / "images" / "photo.jpg").write_bytes(b"fake JPEG content")
            (vault_root / "document.pdf").write_bytes(b"fake PDF content")

            yield vault_root

    @pytest.fixture
    def asset_pipeline(self, temp_vault):
        """Create AssetPipeline with temporary vault."""
        return AssetPipeline(temp_vault, "test_vault")

    def test_basic_pipeline_initialization(self, temp_vault):
        """Test basic pipeline initialization."""
        pipeline = AssetPipeline(temp_vault, "test_vault")

        assert pipeline.vault_root == temp_vault
        assert pipeline.vault_id == "test_vault"
        assert pipeline.detector is not None
        assert pipeline.resolver is not None
        assert pipeline.hasher is not None
        assert pipeline.registry is not None

    def test_process_assets_with_existing_files(self, asset_pipeline, temp_vault):
        """Test processing assets that exist in the vault."""
        content = """
        Here's a diagram: ![[diagram.png]]
        And a photo: ![Photo](images/photo.jpg)
        Missing file: ![[missing.png]]
        """

        source_file = temp_vault / "note.md"
        assets = asset_pipeline.process_assets(content, source_file)

        assert len(assets) == 3

        # Check existing assets
        diagram_asset = next(a for a in assets if a.target == "diagram.png")
        assert not diagram_asset.is_broken
        assert diagram_asset.resolved_path.resolve() == (temp_vault / "assets" / "diagram.png").resolve()
        assert diagram_asset.content_hash is not None

        photo_asset = next(a for a in assets if a.target == "images/photo.jpg")
        assert not photo_asset.is_broken
        assert photo_asset.resolved_path.resolve() == (temp_vault / "images" / "photo.jpg").resolve()
        assert photo_asset.content_hash is not None

        # Check missing asset
        missing_asset = next(a for a in assets if a.target == "missing.png")
        assert missing_asset.is_broken
        assert missing_asset.resolved_path is None

    def test_asset_deduplication(self, asset_pipeline, temp_vault):
        """Test that assets with same content are deduplicated."""
        # Create two files with identical content
        (temp_vault / "image1.png").write_bytes(b"identical content")
        (temp_vault / "image2.png").write_bytes(b"identical content")

        content1 = "First note: ![[image1.png]]"
        content2 = "Second note: ![[image2.png]]"

        source_file = temp_vault / "note.md"

        # Process first asset
        assets1 = asset_pipeline.process_assets(content1, source_file)
        assert len(assets1) == 1
        hash1 = assets1[0].content_hash

        # Process second asset (should be detected as duplicate)
        assets2 = asset_pipeline.process_assets(content2, source_file)
        assert len(assets2) == 1
        hash2 = assets2[0].content_hash

        # Hashes should be identical
        assert hash1 == hash2

        # Registry should show deduplication
        registry_stats = asset_pipeline.registry.get_stats()
        assert registry_stats["total_assets"] == 1  # Only one unique hash stored

    def test_statistics_tracking(self, asset_pipeline, temp_vault):
        """Test that pipeline statistics are tracked correctly."""
        content = """
        Asset 1: ![[diagram.png]]
        Asset 2: ![Photo](images/photo.jpg)
        """

        source_file = temp_vault / "note.md"
        assets = asset_pipeline.process_assets(content, source_file)

        stats = asset_pipeline.get_statistics()

        assert stats["vault_id"] == "test_vault"
        assert stats["vault_root"] == str(temp_vault)
        assert "supported_extensions" in stats
        assert "registry_stats" in stats

        registry_stats = stats["registry_stats"]
        assert registry_stats["total_assets"] >= len(assets)


@pytest.mark.skipif(not UNSTRUCTURED_AVAILABLE, reason="Unstructured.io not available")
class TestAssetTextExtractor:
    """Test cases for AssetTextExtractor (requires Unstructured.io)."""

    @pytest.fixture
    def extractor_config(self):
        """Create test extractor configuration."""
        return AssetTextExtractorConfig(
            enable_image_ocr=True,
            enable_pdf_extraction=True,
            ocr_languages="eng",
            max_file_size_mb=10,
            processing_timeout_seconds=30
        )

    @pytest.fixture
    def text_extractor(self, extractor_config):
        """Create AssetTextExtractor with test configuration."""
        return AssetTextExtractor(extractor_config)

    def test_config_initialization(self):
        """Test configuration initialization."""
        config = AssetTextExtractorConfig(
            enable_image_ocr=False,
            ocr_languages="fra",
            max_file_size_mb=5
        )

        assert config.enable_image_ocr is False
        assert config.ocr_languages == "fra"
        assert config.max_file_size_mb == 5
        assert config.max_file_size_bytes == 5 * 1024 * 1024

    def test_extractor_initialization(self, text_extractor):
        """Test text extractor initialization."""
        assert text_extractor.config is not None
        assert text_extractor.images_processed == 0
        assert text_extractor.pdfs_processed == 0

    @patch('futurnal.ingestion.obsidian.asset_processor.partition_image')
    def test_extract_from_image_mock(self, mock_partition, text_extractor, temp_vault):
        """Test image text extraction with mocked Unstructured.io."""
        # Mock the partition_image response
        mock_elements = [Mock(), Mock()]
        mock_elements[0].__str__ = Mock(return_value="Hello")
        mock_elements[1].__str__ = Mock(return_value="World")
        mock_partition.return_value = mock_elements

        # Create test asset
        image_file = temp_vault / "test.png"
        image_file.write_bytes(b"fake image content")

        from futurnal.ingestion.obsidian.assets import ObsidianAsset
        asset = ObsidianAsset(
            target="test.png",
            resolved_path=image_file,
            file_size=len(b"fake image content"),
            mime_type="image/png"
        )

        # Extract text
        result = text_extractor._extract_from_image(asset)

        assert result.success is True
        assert result.extracted_text == "Hello\nWorld"
        assert result.element_count == 2
        assert result.extraction_method == "ocr"
        assert result.processing_time_ms > 0

        # Verify mock was called correctly
        mock_partition.assert_called_once_with(
            filename=str(image_file),
            ocr_languages="eng",
            include_metadata=True,
            skip_single_word_elements=True
        )

    @patch('futurnal.ingestion.obsidian.asset_processor.partition_pdf')
    def test_extract_from_pdf_mock(self, mock_partition, text_extractor, temp_vault):
        """Test PDF text extraction with mocked Unstructured.io."""
        # Mock the partition_pdf response
        mock_elements = [Mock(), Mock(), Mock()]
        mock_elements[0].__str__ = Mock(return_value="Chapter 1")
        mock_elements[1].__str__ = Mock(return_value="This is content")
        mock_elements[2].__str__ = Mock(return_value="")  # Empty element
        mock_partition.return_value = mock_elements

        # Create test asset
        pdf_file = temp_vault / "test.pdf"
        pdf_file.write_bytes(b"fake PDF content")

        from futurnal.ingestion.obsidian.assets import ObsidianAsset
        asset = ObsidianAsset(
            target="test.pdf",
            resolved_path=pdf_file,
            file_size=len(b"fake PDF content"),
            mime_type="application/pdf"
        )

        # Extract text
        result = text_extractor._extract_from_pdf(asset)

        assert result.success is True
        assert result.extracted_text == "Chapter 1\nThis is content"  # Empty element filtered out
        assert result.element_count == 3
        assert result.extraction_method == "pdf_parse"
        assert result.processing_time_ms > 0

    def test_file_size_limit_enforcement(self, text_extractor):
        """Test that file size limits are enforced."""
        from futurnal.ingestion.obsidian.assets import ObsidianAsset

        # Create asset that exceeds size limit
        large_asset = ObsidianAsset(
            target="huge_file.png",
            resolved_path=Path("/fake/path"),
            file_size=20 * 1024 * 1024,  # 20MB (exceeds 10MB limit)
            mime_type="image/png"
        )

        result = text_extractor.extract_text(large_asset)

        assert result is not None
        assert result.success is False
        assert "exceeds limit" in result.error_message

    def test_broken_asset_handling(self, text_extractor):
        """Test handling of broken assets."""
        from futurnal.ingestion.obsidian.assets import ObsidianAsset

        broken_asset = ObsidianAsset(
            target="missing.png",
            is_broken=True
        )

        result = text_extractor.extract_text(broken_asset)
        assert result is None

    def test_unsupported_asset_type(self, text_extractor):
        """Test handling of unsupported asset types."""
        from futurnal.ingestion.obsidian.assets import ObsidianAsset

        # Create asset with unsupported type
        unsupported_asset = ObsidianAsset(
            target="video.mp4",
            resolved_path=Path("/fake/video.mp4"),
            file_size=1000,
            mime_type="video/mp4"
        )

        result = text_extractor.extract_text(unsupported_asset)
        assert result is None

    def test_statistics_tracking(self, text_extractor):
        """Test statistics tracking in text extractor."""
        initial_stats = text_extractor.get_statistics()

        assert initial_stats["images_processed"] == 0
        assert initial_stats["pdfs_processed"] == 0
        assert initial_stats["total_processed"] == 0
        assert initial_stats["success_rate"] == 0.0

        # Manually update statistics to test calculation
        text_extractor.images_processed = 5
        text_extractor.pdfs_processed = 3
        text_extractor.total_text_extracted = 1000
        text_extractor.total_processing_time_ms = 5000
        text_extractor.errors_encountered = 1

        updated_stats = text_extractor.get_statistics()

        assert updated_stats["total_processed"] == 8
        assert updated_stats["success_rate"] == 7/8  # 7 successful out of 8
        assert updated_stats["average_processing_time_ms"] == 5000/8
        assert updated_stats["average_text_per_asset"] == 1000/8

    def test_clear_statistics(self, text_extractor):
        """Test clearing of statistics."""
        # Set some statistics
        text_extractor.images_processed = 5
        text_extractor.errors_encountered = 2

        # Clear statistics
        text_extractor.clear_statistics()

        stats = text_extractor.get_statistics()
        assert stats["images_processed"] == 0
        assert stats["errors_encountered"] == 0

    @patch('futurnal.ingestion.obsidian.asset_processor.partition_image')
    def test_batch_processing(self, mock_partition, text_extractor, temp_vault):
        """Test batch processing of multiple assets."""
        from futurnal.ingestion.obsidian.assets import ObsidianAsset

        # Mock successful extraction
        mock_elements = [Mock()]
        mock_elements[0].__str__ = Mock(return_value="Extracted text")
        mock_partition.return_value = mock_elements

        # Create test assets
        assets = []
        for i in range(3):
            image_file = temp_vault / f"test{i}.png"
            image_file.write_bytes(b"fake image content")

            asset = ObsidianAsset(
                target=f"test{i}.png",
                resolved_path=image_file,
                file_size=100,
                mime_type="image/png"
            )
            assets.append(asset)

        # Process batch
        results = text_extractor.extract_batch(assets)

        assert len(results) == 3
        assert all(result.success for result in results)
        assert all(result.extracted_text == "Extracted text" for result in results)


class TestAssetProcessingPipeline:
    """Test cases for complete AssetProcessingPipeline."""

    @pytest.fixture
    def temp_vault(self):
        """Create a temporary vault with test assets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)

            # Create test files
            (vault_root / "image.png").write_bytes(b"fake PNG content")
            (vault_root / "document.pdf").write_bytes(b"fake PDF content")

            yield vault_root

    @pytest.fixture
    def processing_pipeline(self, temp_vault):
        """Create AssetProcessingPipeline with temporary vault."""
        config = AssetTextExtractorConfig(
            enable_image_ocr=True,
            enable_pdf_extraction=True,
            max_file_size_mb=5
        )
        return AssetProcessingPipeline(temp_vault, "test_vault", config)

    def test_pipeline_initialization(self, processing_pipeline, temp_vault):
        """Test pipeline initialization."""
        assert processing_pipeline.vault_root == temp_vault
        assert processing_pipeline.vault_id == "test_vault"
        assert processing_pipeline.asset_pipeline is not None
        assert processing_pipeline.text_extractor is not None

    @patch('futurnal.ingestion.obsidian.asset_processor.partition_image')
    def test_process_document_assets_integration(self, mock_partition, processing_pipeline, temp_vault):
        """Test complete document asset processing."""
        # Mock text extraction
        mock_elements = [Mock()]
        mock_elements[0].__str__ = Mock(return_value="Extracted from image")
        mock_partition.return_value = mock_elements

        content = """
        Document with assets:
        ![[image.png]]
        ![PDF](document.pdf)
        ![[missing.png]]
        """

        source_file = temp_vault / "note.md"

        # Process with text extraction enabled
        result = processing_pipeline.process_document_assets(
            content,
            source_file,
            include_text_extraction=True
        )

        assert "assets" in result
        assert "text_extractions" in result
        assert "statistics" in result

        # Check statistics
        stats = result["statistics"]
        assert stats["total_assets"] == 3
        assert stats["broken_assets"] == 1  # missing.png
        assert stats["processable_assets"] >= 1  # at least image.png

    def test_process_document_assets_no_extraction(self, processing_pipeline, temp_vault):
        """Test document processing without text extraction."""
        content = "Simple image: ![[image.png]]"
        source_file = temp_vault / "note.md"

        result = processing_pipeline.process_document_assets(
            content,
            source_file,
            include_text_extraction=False
        )

        assert len(result["text_extractions"]) == 0
        assert result["statistics"]["text_extracted_from"] == 0

    def test_comprehensive_statistics(self, processing_pipeline):
        """Test comprehensive statistics gathering."""
        stats = processing_pipeline.get_comprehensive_statistics()

        assert "asset_pipeline" in stats
        assert "text_extractor" in stats
        assert "unstructured_available" in stats

        assert isinstance(stats["unstructured_available"], bool)


class TestAssetProcessorWithoutUnstructured:
    """Test cases for asset processor when Unstructured.io is not available."""

    @pytest.fixture
    def temp_vault(self):
        """Create a temporary vault for testing."""
        import tempfile
        import shutil
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir)

            # Create test directories
            (vault_path / "assets").mkdir()
            (vault_path / "images").mkdir()

            # Create test files
            (vault_path / "assets" / "diagram.png").write_bytes(b"fake PNG content")
            (vault_path / "images" / "photo.jpg").write_bytes(b"fake JPEG content")

            yield vault_path

    @patch('futurnal.ingestion.obsidian.asset_processor.UNSTRUCTURED_AVAILABLE', False)
    def test_graceful_degradation_without_unstructured(self):
        """Test that the system works gracefully without Unstructured.io."""
        from futurnal.ingestion.obsidian.asset_processor import AssetTextExtractor

        extractor = AssetTextExtractor()

        # Should initialize without errors
        assert extractor.config is not None

        # Text extraction should return None
        from futurnal.ingestion.obsidian.assets import ObsidianAsset
        asset = ObsidianAsset(target="test.png")
        result = extractor.extract_text(asset)

        assert result is None

    @patch('futurnal.ingestion.obsidian.asset_processor.UNSTRUCTURED_AVAILABLE', False)
    def test_pipeline_without_unstructured(self, temp_vault):
        """Test complete pipeline without Unstructured.io."""
        from futurnal.ingestion.obsidian.asset_processor import AssetProcessingPipeline

        # Create test file
        (temp_vault / "image.png").write_bytes(b"fake PNG content")

        pipeline = AssetProcessingPipeline(temp_vault, "test_vault")

        content = "Image: ![[image.png]]"
        source_file = temp_vault / "note.md"

        result = pipeline.process_document_assets(content, source_file)

        # Should detect assets but not extract text
        assert len(result["assets"]) == 1
        assert len(result["text_extractions"]) == 0
        assert result["statistics"]["text_extracted_from"] == 0


class TestErrorHandlingAndEdgeCases:
    """Test cases for error handling and edge cases."""

    @pytest.fixture
    def temp_vault(self):
        """Create a temporary vault for testing."""
        import tempfile
        import shutil
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir)

            # Create test directories
            (vault_path / "assets").mkdir()
            (vault_path / "images").mkdir()

            # Create test files
            (vault_path / "assets" / "diagram.png").write_bytes(b"fake PNG content")
            (vault_path / "images" / "photo.jpg").write_bytes(b"fake JPEG content")

            yield vault_path

    def test_malformed_content_handling(self, temp_vault):
        """Test handling of malformed content."""
        pipeline = AssetPipeline(temp_vault, "test_vault")

        malformed_content = """
        Incomplete: ![[
        Empty: ![[]]
        Invalid chars: ![[file\x00.png]]
        """

        source_file = temp_vault / "note.md"

        # Should not crash and handle gracefully
        assets = pipeline.process_assets(malformed_content, source_file)

        # Should handle malformed syntax gracefully
        assert isinstance(assets, list)

    def test_unicode_content_handling(self, temp_vault):
        """Test handling of Unicode content."""
        pipeline = AssetPipeline(temp_vault, "test_vault")

        unicode_content = """
        Unicode filename: ![[图像.png]]
        Emoji: ![[📷.jpg]]
        Mixed: ![Alt 🖼️](images/测试.png)
        """

        source_file = temp_vault / "note.md"

        # Should handle Unicode gracefully
        assets = pipeline.process_assets(unicode_content, source_file)

        assert isinstance(assets, list)
        # Assets may be marked as broken due to missing files, but should not crash

    def test_extremely_long_content(self, temp_vault):
        """Test handling of extremely long content."""
        pipeline = AssetPipeline(temp_vault, "test_vault")

        # Create very long content
        long_content = "Text " * 10000 + "![[image.png]]" + "More text " * 10000

        source_file = temp_vault / "note.md"

        # Should handle without performance issues
        assets = pipeline.process_assets(long_content, source_file)

        assert len(assets) == 1
        assert assets[0].target == "image.png"

    def test_empty_and_whitespace_content(self, temp_vault):
        """Test handling of empty and whitespace-only content."""
        pipeline = AssetPipeline(temp_vault, "test_vault")
        source_file = temp_vault / "note.md"

        # Empty content
        assets_empty = pipeline.process_assets("", source_file)
        assert len(assets_empty) == 0

        # Whitespace only
        assets_whitespace = pipeline.process_assets("   \n\t  \n  ", source_file)
        assert len(assets_whitespace) == 0

        # Only non-asset content
        assets_no_assets = pipeline.process_assets("Just regular text content.", source_file)
        assert len(assets_no_assets) == 0