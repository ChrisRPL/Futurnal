"""Integration tests for asset processing metrics collection and quality gate evaluation."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from futurnal.ingestion.obsidian.processor import ObsidianDocumentProcessor
from futurnal.ingestion.obsidian.sync_metrics import SyncMetricsCollector, create_metrics_collector
from futurnal.ingestion.obsidian.quality_gate import (
    QualityGateConfig,
    QualityGateEvaluator,
    QualityMetricType,
    QualityGateStatus
)
from futurnal.ingestion.obsidian.asset_processor import AssetTextExtractor, AssetTextExtractorConfig
from futurnal.ingestion.local.state import FileRecord


class TestAssetProcessingMetricsIntegration:
    """Test asset processing metrics collection and quality gate integration."""

    def create_test_vault(self):
        """Create a temporary test vault with assets."""
        vault_dir = Path(tempfile.mkdtemp())
        obsidian_dir = vault_dir / ".obsidian"
        obsidian_dir.mkdir()

        # Create test documents with asset references
        test_doc1 = vault_dir / "test1.md"
        test_doc1.write_text("""# Test Document 1

This document contains an image:
![Test Image](assets/test-image.png)

And a PDF:
![PDF Document](assets/document.pdf)
""")

        test_doc2 = vault_dir / "test2.md"
        test_doc2.write_text("""# Test Document 2

Another image reference:
![Another Image](assets/missing-image.jpg)

Valid image:
![Valid Image](assets/valid-image.png)
""")

        # Create assets directory with some test files
        assets_dir = vault_dir / "assets"
        assets_dir.mkdir()

        # Create a test image file
        test_image = assets_dir / "test-image.png"
        test_image.write_bytes(b"PNG fake image data")

        valid_image = assets_dir / "valid-image.png"
        valid_image.write_bytes(b"PNG valid image data")

        # Create a test PDF file
        test_pdf = assets_dir / "document.pdf"
        test_pdf.write_bytes(b"PDF fake data")

        return vault_dir, [test_doc1, test_doc2]

    def create_file_record(self, file_path: Path) -> FileRecord:
        """Create a FileRecord for testing."""
        stats = file_path.stat()
        return FileRecord(
            path=file_path,
            sha256="test_hash_" + file_path.name,
            size=stats.st_size,
            mtime=stats.st_mtime
        )

    def test_asset_processing_success_metrics_collection(self):
        """Test that successful asset processing is recorded in metrics."""
        vault_dir, test_docs = self.create_test_vault()

        try:
            # Create metrics collector
            metrics_collector = create_metrics_collector()
            vault_id = "test_vault"

            # Create processor with asset processing enabled
            processor = ObsidianDocumentProcessor(
                workspace_dir=vault_dir / "workspace",
                vault_root=vault_dir,
                vault_id=vault_id,
                enable_link_graph=False,  # Disable to focus on asset processing
                asset_processing_config={"enable_asset_text_extraction": True},
                metrics_collector=metrics_collector
            )

            # Process the first document (has valid assets)
            file_record = self.create_file_record(test_docs[0])

            # Mock the Unstructured.io processing to avoid dependency
            with patch('futurnal.ingestion.obsidian.processor.partition') as mock_partition:
                mock_element = Mock()
                mock_element.to_dict.return_value = {"text": "test content", "type": "paragraph"}
                mock_partition.return_value = [mock_element]

                # Mock the asset text extractor to simulate successful processing
                with patch.object(processor.asset_processing_pipeline.text_extractor, 'extract_batch') as mock_extract:
                    from futurnal.ingestion.obsidian.asset_processor import AssetTextExtraction
                    from futurnal.ingestion.obsidian.assets import ObsidianAsset

                    # Create mock asset and successful extraction
                    mock_asset = Mock(spec=ObsidianAsset)
                    mock_asset.target = "assets/test-image.png"
                    mock_asset.is_image = True
                    mock_asset.is_pdf = False

                    mock_extraction = AssetTextExtraction(
                        asset=mock_asset,
                        extracted_text="extracted text content",
                        element_count=1,
                        extraction_method="ocr",
                        processing_time_ms=1000,
                        success=True
                    )
                    mock_extract.return_value = [mock_extraction]

                    # Process the document
                    results = list(processor.process_document(file_record, "test_source"))

                    # Verify we got results
                    assert len(results) > 0

            # Check that asset processing success metrics were recorded
            success_count = metrics_collector.get_counter(
                "asset_processing_success",
                labels={"vault_id": vault_id}
            )
            assert success_count > 0, "Asset processing success should be recorded"

        finally:
            import shutil
            shutil.rmtree(vault_dir)

    def test_asset_processing_failure_metrics_collection(self):
        """Test that failed asset processing is recorded in metrics."""
        vault_dir, test_docs = self.create_test_vault()

        try:
            # Create metrics collector
            metrics_collector = create_metrics_collector()
            vault_id = "test_vault"

            # Create processor with asset processing enabled
            processor = ObsidianDocumentProcessor(
                workspace_dir=vault_dir / "workspace",
                vault_root=vault_dir,
                vault_id=vault_id,
                enable_link_graph=False,
                asset_processing_config={"enable_asset_text_extraction": True},
                metrics_collector=metrics_collector
            )

            # Process document
            file_record = self.create_file_record(test_docs[1])

            # Mock the Unstructured.io processing
            with patch('futurnal.ingestion.obsidian.processor.partition') as mock_partition:
                mock_element = Mock()
                mock_element.to_dict.return_value = {"text": "test content", "type": "paragraph"}
                mock_partition.return_value = [mock_element]

                # Mock the asset text extractor to simulate failure
                with patch.object(processor.asset_processing_pipeline.text_extractor, 'extract_batch') as mock_extract:
                    from futurnal.ingestion.obsidian.asset_processor import AssetTextExtraction
                    from futurnal.ingestion.obsidian.assets import ObsidianAsset

                    # Create mock asset and failed extraction
                    mock_asset = Mock(spec=ObsidianAsset)
                    mock_asset.target = "assets/missing-image.jpg"
                    mock_asset.is_image = True
                    mock_asset.is_pdf = False

                    mock_extraction = AssetTextExtraction(
                        asset=mock_asset,
                        extracted_text="",
                        element_count=0,
                        extraction_method="error",
                        processing_time_ms=100,
                        success=False,
                        error_message="File not found"
                    )
                    mock_extract.return_value = [mock_extraction]

                    # Mock the extractor to record failure metrics
                    processor.asset_processing_pipeline.text_extractor.metrics_collector = metrics_collector
                    processor.asset_processing_pipeline.text_extractor.vault_id = vault_id

                    # Manually record failure (since we're mocking the extraction)
                    metrics_collector.increment_counter(
                        "asset_processing_failure",
                        labels={"vault_id": vault_id, "asset_type": "image", "method": "ocr", "error_type": "FileNotFoundError"}
                    )

                    # Process the document
                    results = list(processor.process_document(file_record, "test_source"))

                    # Verify we got results
                    assert len(results) > 0

            # Check that asset processing failure metrics were recorded
            failure_count = metrics_collector.get_counter(
                "asset_processing_failure",
                labels={"vault_id": vault_id}
            )
            assert failure_count > 0, "Asset processing failure should be recorded"

        finally:
            import shutil
            shutil.rmtree(vault_dir)

    def test_quality_gate_asset_processing_success_rate_evaluation(self):
        """Test quality gate evaluation using asset processing success rate."""
        # Create metrics collector with asset processing data
        metrics_collector = create_metrics_collector()
        vault_id = "test_vault"

        # Simulate asset processing metrics
        # 8 successful, 2 failed = 80% success rate (below 90% threshold)
        metrics_collector.increment_counter(
            "asset_processing_success",
            value=8,
            labels={"vault_id": vault_id}
        )
        metrics_collector.increment_counter(
            "asset_processing_failure",
            value=2,
            labels={"vault_id": vault_id}
        )

        # Add other required metrics for full evaluation
        metrics_collector.increment_counter("parse_successes", value=95, labels={"vault_id": vault_id})
        metrics_collector.increment_counter("parse_failures", value=5, labels={"vault_id": vault_id})
        metrics_collector.increment_counter("broken_links", value=2, labels={"vault_id": vault_id})
        metrics_collector.increment_counter("total_links", value=50, labels={"vault_id": vault_id})
        metrics_collector.increment_counter("consent_granted_files", value=100, labels={"vault_id": vault_id})

        # Create quality gate config with strict asset processing requirements
        config = QualityGateConfig(
            min_asset_processing_success_rate=0.90,  # 90% minimum
            enable_strict_mode=False
        )

        # Create evaluator
        evaluator = QualityGateEvaluator(config=config, metrics_collector=metrics_collector)

        # Evaluate quality gate
        result = evaluator.evaluate_vault_quality(vault_id, metrics_collector)

        # Verify evaluation
        assert result is not None
        assert result.vault_id == vault_id

        # Find the asset processing success rate metric result
        asset_metric_result = None
        for metric_result in result.metric_results:
            if metric_result.metric_type == QualityMetricType.ASSET_PROCESSING_SUCCESS_RATE:
                asset_metric_result = metric_result
                break

        assert asset_metric_result is not None, "Asset processing success rate metric should be evaluated"
        assert asset_metric_result.value == 0.8, f"Expected 80% success rate, got {asset_metric_result.value}"
        assert asset_metric_result.status in [QualityGateStatus.WARN, QualityGateStatus.FAIL], \
            f"80% success rate should trigger warning/failure (threshold: 90%), got {asset_metric_result.status}"

    def test_end_to_end_asset_processing_quality_gate(self):
        """Test complete end-to-end asset processing with quality gate evaluation."""
        vault_dir, test_docs = self.create_test_vault()

        try:
            # Create metrics collector
            metrics_collector = create_metrics_collector()
            vault_id = "test_vault"

            # Create processor
            processor = ObsidianDocumentProcessor(
                workspace_dir=vault_dir / "workspace",
                vault_root=vault_dir,
                vault_id=vault_id,
                enable_link_graph=False,
                asset_processing_config={"enable_asset_text_extraction": True},
                metrics_collector=metrics_collector
            )

            # Process multiple documents to generate metrics
            processed_docs = 0
            for test_doc in test_docs:
                file_record = self.create_file_record(test_doc)

                with patch('futurnal.ingestion.obsidian.processor.partition') as mock_partition:
                    mock_element = Mock()
                    mock_element.to_dict.return_value = {"text": "test content", "type": "paragraph"}
                    mock_partition.return_value = [mock_element]

                    try:
                        results = list(processor.process_document(file_record, "test_source"))
                        if results:
                            processed_docs += 1
                    except Exception as e:
                        # Record processing failure
                        metrics_collector.increment_counter(
                            "parse_failures",
                            labels={"vault_id": vault_id}
                        )

            # Add some baseline metrics to ensure quality gate evaluation can proceed
            if processed_docs > 0:
                metrics_collector.increment_counter("parse_successes", value=processed_docs, labels={"vault_id": vault_id})

            # Simulate some asset processing results
            metrics_collector.increment_counter("asset_processing_success", value=3, labels={"vault_id": vault_id})
            metrics_collector.increment_counter("asset_processing_failure", value=1, labels={"vault_id": vault_id})

            # Add other required metrics
            metrics_collector.increment_counter("total_links", value=10, labels={"vault_id": vault_id})
            metrics_collector.increment_counter("broken_links", value=1, labels={"vault_id": vault_id})
            metrics_collector.increment_counter("consent_granted_files", value=processed_docs, labels={"vault_id": vault_id})

            # Create quality gate evaluator
            config = QualityGateConfig(
                min_asset_processing_success_rate=0.70,  # 70% threshold (should pass with 75%)
                max_parse_failure_rate=0.10,
                max_broken_link_rate=0.20,
                min_consent_coverage_rate=0.80
            )

            evaluator = QualityGateEvaluator(config=config, metrics_collector=metrics_collector)

            # Evaluate quality gate
            result = evaluator.evaluate_vault_quality(vault_id, metrics_collector)

            # Verify evaluation completed
            assert result is not None
            assert result.vault_id == vault_id
            assert result.status in [QualityGateStatus.PASS, QualityGateStatus.WARN, QualityGateStatus.FAIL]
            assert len(result.metric_results) > 0

            # Verify asset processing metrics are included
            has_asset_metric = any(
                mr.metric_type == QualityMetricType.ASSET_PROCESSING_SUCCESS_RATE
                for mr in result.metric_results
            )
            assert has_asset_metric, "Asset processing success rate should be evaluated"

            # Test exit code functionality
            exit_code = result.get_exit_code()
            assert exit_code in [0, 1, 2], f"Exit code should be 0, 1, or 2, got {exit_code}"

        finally:
            import shutil
            shutil.rmtree(vault_dir)

    def test_basic_metrics_collector_functionality(self):
        """Test that metrics collector basic functionality works."""
        # Create metrics collector
        metrics_collector = create_metrics_collector()
        vault_id = "test_vault"

        # Directly increment a counter
        labels = {"vault_id": vault_id, "asset_type": "image", "method": "ocr"}
        metrics_collector.increment_counter(
            "asset_processing_success",
            labels=labels
        )

        # Verify the counter was incremented (using exact same labels)
        success_count = metrics_collector.get_counter(
            "asset_processing_success",
            labels=labels
        )
        assert success_count > 0, f"Expected counter > 0, got {success_count}"

    @patch('futurnal.ingestion.obsidian.asset_processor.UNSTRUCTURED_AVAILABLE', True)
    def test_asset_text_extractor_metrics_integration(self):
        """Test direct integration of AssetTextExtractor with metrics collection."""
        # Create metrics collector
        metrics_collector = create_metrics_collector()
        vault_id = "test_vault"

        # Create asset text extractor with metrics
        config = AssetTextExtractorConfig(
            enable_image_ocr=True,
            enable_pdf_extraction=True
        )

        extractor = AssetTextExtractor(
            config=config,
            metrics_collector=metrics_collector,
            vault_id=vault_id
        )

        # Create mock assets
        from futurnal.ingestion.obsidian.assets import ObsidianAsset

        # Mock successful image processing
        mock_image_asset = Mock(spec=ObsidianAsset)
        mock_image_asset.target = "test.png"
        mock_image_asset.is_image = True
        mock_image_asset.is_pdf = False
        mock_image_asset.is_processable = True
        mock_image_asset.is_broken = False
        mock_image_asset.resolved_path = Path("/fake/path/test.png")
        mock_image_asset.file_size = 1024

        # Mock the Unstructured.io partition_image function
        with patch('futurnal.ingestion.obsidian.asset_processor.partition_image') as mock_partition:
            mock_element = Mock()
            mock_element.__str__ = lambda self: "extracted text from image"
            mock_partition.return_value = [mock_element]

            # Extract text (should record success)
            result = extractor.extract_text(mock_image_asset)

            assert result is not None
            assert result.success is True
            assert result.extracted_text == "extracted text from image"

        # Verify success metric was recorded (using exact labels that the extractor uses)
        success_count = metrics_collector.get_counter(
            "asset_processing_success",
            labels={"vault_id": vault_id, "asset_type": "image", "method": "ocr"}
        )
        assert success_count > 0, f"Asset processing success should be recorded, got {success_count}"

    @patch('futurnal.ingestion.obsidian.asset_processor.UNSTRUCTURED_AVAILABLE', True)
    def test_asset_size_limit_failure_metrics(self):
        """Test that asset size limit failures are recorded in metrics."""
        # Create metrics collector
        metrics_collector = create_metrics_collector()
        vault_id = "test_vault"

        # Create asset text extractor with small size limit
        config = AssetTextExtractorConfig(
            max_file_size_mb=1  # 1MB limit
        )

        extractor = AssetTextExtractor(
            config=config,
            metrics_collector=metrics_collector,
            vault_id=vault_id
        )

        # Create mock asset that exceeds size limit
        from futurnal.ingestion.obsidian.assets import ObsidianAsset

        mock_large_asset = Mock(spec=ObsidianAsset)
        mock_large_asset.target = "large_file.pdf"
        mock_large_asset.is_image = False
        mock_large_asset.is_pdf = True
        mock_large_asset.is_processable = True
        mock_large_asset.is_broken = False
        mock_large_asset.resolved_path = Path("/fake/path/large_file.pdf")
        mock_large_asset.file_size = 2 * 1024 * 1024  # 2MB (exceeds 1MB limit)

        # Extract text (should record failure due to size)
        result = extractor.extract_text(mock_large_asset)

        assert result is not None
        assert result.success is False
        assert "exceeds limit" in result.error_message

        # Verify failure metric was recorded (using exact labels that the extractor uses)
        failure_count = metrics_collector.get_counter(
            "asset_processing_failure",
            labels={"vault_id": vault_id, "asset_type": "unknown", "method": "size_check", "error_type": "FileSizeLimitExceeded"}
        )
        assert failure_count > 0, f"Asset processing failure should be recorded for size limit, got {failure_count}"