"""Integration tests for Obsidian asset processing pipeline.

These tests validate the complete asset processing pipeline with real files
and realistic vault scenarios.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from futurnal.ingestion.obsidian.normalizer import normalize_obsidian_document
from futurnal.ingestion.obsidian.processor import ObsidianDocumentProcessor
from futurnal.ingestion.obsidian.connector import ObsidianVaultConnector, ObsidianVaultSource
from futurnal.ingestion.obsidian.link_graph import ObsidianLinkGraphConstructor
from futurnal.ingestion.obsidian.asset_processor import AssetProcessingPipeline, AssetTextExtractorConfig
from futurnal.ingestion.local.state import StateStore, FileRecord


class TestAssetIntegrationFixtures:
    """Integration tests using real asset fixtures."""

    @pytest.fixture
    def realistic_vault(self):
        """Create a realistic vault structure with actual files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)

            # Create realistic directory structure
            (vault_root / "assets").mkdir()
            (vault_root / "images").mkdir()
            (vault_root / "attachments").mkdir()
            (vault_root / "notes").mkdir()
            (vault_root / "projects").mkdir()
            (vault_root / "daily").mkdir()

            # Create test image files (minimal valid formats)
            self._create_test_image_png(vault_root / "assets" / "diagram.png")
            self._create_test_image_jpg(vault_root / "images" / "photo.jpg")
            self._create_test_image_gif(vault_root / "assets" / "icon.gif")

            # Create test PDF file (minimal valid PDF)
            self._create_test_pdf(vault_root / "attachments" / "document.pdf")

            # Create markdown notes with asset references
            self._create_markdown_files(vault_root)

            yield vault_root

    def _create_test_image_png(self, path: Path):
        """Create a minimal valid PNG file."""
        # Minimal PNG file (1x1 pixel, black)
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D,  # IHDR chunk length
            0x49, 0x48, 0x44, 0x52,  # IHDR
            0x00, 0x00, 0x00, 0x01,  # Width: 1
            0x00, 0x00, 0x00, 0x01,  # Height: 1
            0x08, 0x02,              # Bit depth: 8, Color type: 2 (RGB)
            0x00, 0x00, 0x00,        # Compression, filter, interlace
            0x90, 0x77, 0x53, 0xDE,  # CRC
            0x00, 0x00, 0x00, 0x0C,  # IDAT chunk length
            0x49, 0x44, 0x41, 0x54,  # IDAT
            0x08, 0x99, 0x01, 0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x01,        # Compressed data
            0xE5, 0x27, 0xDE, 0xFC,  # CRC
            0x00, 0x00, 0x00, 0x00,  # IEND chunk length
            0x49, 0x45, 0x4E, 0x44,  # IEND
            0xAE, 0x42, 0x60, 0x82   # CRC
        ])
        path.write_bytes(png_data)

    def _create_test_image_jpg(self, path: Path):
        """Create a minimal valid JPEG file."""
        # Minimal JPEG file
        jpeg_data = bytes([
            0xFF, 0xD8, 0xFF, 0xE0,  # JPEG signature + APP0
            0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00,
            0xFF, 0xDB, 0x00, 0x43, 0x00,  # Quantization table
        ] + [0x01] * 64 + [  # Minimal quantization values
            0xFF, 0xC0, 0x00, 0x11,  # Start of frame
            0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01,
            0xFF, 0xC4, 0x00, 0x14, 0x00,  # Huffman table
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08,
            0xFF, 0xDA, 0x00, 0x08,  # Start of scan
            0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
            0xFF, 0xD9  # End of image
        ])
        path.write_bytes(jpeg_data)

    def _create_test_image_gif(self, path: Path):
        """Create a minimal valid GIF file."""
        # Minimal GIF file (1x1 pixel)
        gif_data = bytes([
            0x47, 0x49, 0x46, 0x38, 0x39, 0x61,  # GIF89a signature
            0x01, 0x00, 0x01, 0x00,              # Width: 1, Height: 1
            0x80, 0x00, 0x00,                    # Global color table flag, color resolution, sort flag
            0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF,  # Global color table (black, white)
            0x2C, 0x00, 0x00, 0x00, 0x00,        # Image descriptor
            0x01, 0x00, 0x01, 0x00, 0x00,        # Left, top, width, height, flags
            0x02, 0x02, 0x04, 0x01, 0x00,        # Image data
            0x3B                                 # Trailer
        ])
        path.write_bytes(gif_data)

    def _create_test_pdf(self, path: Path):
        """Create a minimal valid PDF file."""
        # Minimal PDF with extractable text
        pdf_content = """%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello PDF World!) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f
0000000010 00000 n
0000000053 00000 n
0000000100 00000 n
0000000178 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
272
%%EOF"""
        path.write_text(pdf_content)

    def _create_markdown_files(self, vault_root: Path):
        """Create markdown files with various asset references."""

        # Main note with multiple asset types
        main_note = vault_root / "notes" / "main_note.md"
        main_note.write_text("""---
title: "Main Note with Assets"
tags: [example, assets, test]
created: 2024-01-01
---

# Main Note

This note demonstrates various asset types embedded in Obsidian.

## Images

Here's a diagram: ![[diagram.png]]

And a photo with alt text: ![My Photo](../images/photo.jpg)

## Documents

Important document: ![[document.pdf]]

## Missing Assets

This asset doesn't exist: ![[missing_image.png]]

## Icon

Small icon: ![[icon.gif]]

## Links to Other Notes

See also: [[project_note]]
""")

        # Project note with asset references
        project_note = vault_root / "projects" / "project_note.md"
        project_note.write_text("""---
title: "Project Documentation"
project: example
---

# Project Documentation

## Architecture

System diagram: ![[../assets/diagram.png]]

## Screenshots

Application screenshot: ![App Screenshot](../images/photo.jpg)

Reference document: ![[../attachments/document.pdf]]
""")

        # Daily note with mixed content
        daily_note = vault_root / "daily" / "2024-01-01.md"
        daily_note.write_text("""# Daily Note 2024-01-01

## Today's Work

Reviewed the architecture diagram: ![[../assets/diagram.png]]

## Notes

- Updated documentation
- Fixed icon: ![[../assets/icon.gif]]
- Shared photo: ![](../images/photo.jpg)

## References

- [[../notes/main_note]]
- [[../projects/project_note]]
""")

    def test_complete_asset_pipeline_integration(self, realistic_vault):
        """Test complete asset processing pipeline with realistic data."""
        # Initialize components
        state_store = StateStore()

        # Create vault source
        vault_source = ObsidianVaultSource.from_vault_descriptor(
            vault_id="test_vault",
            name="Test Vault",
            base_path=realistic_vault
        )

        # Create document processor
        processor = ObsidianDocumentProcessor(
            workspace_dir=realistic_vault / ".futurnal",
            vault_root=realistic_vault,
            vault_id="test_vault"
        )

        # Process main note
        main_note_path = realistic_vault / "notes" / "main_note.md"
        content = main_note_path.read_text()

        # Normalize document
        normalized_doc = normalize_obsidian_document(
            content=content,
            source_path=main_note_path,
            vault_root=realistic_vault,
            vault_id="test_vault"
        )

        # Verify asset detection
        assert len(normalized_doc.metadata.assets) >= 4  # At least 4 assets detected

        # Check specific assets
        asset_targets = [asset.target for asset in normalized_doc.metadata.assets]
        assert "diagram.png" in asset_targets
        assert "../images/photo.jpg" in asset_targets
        assert "document.pdf" in asset_targets
        assert "missing_image.png" in asset_targets

        # Check asset resolution
        resolved_assets = [asset for asset in normalized_doc.metadata.assets if not asset.is_broken]
        broken_assets = [asset for asset in normalized_doc.metadata.assets if asset.is_broken]

        assert len(resolved_assets) >= 3  # At least 3 should resolve
        assert len(broken_assets) >= 1   # At least 1 should be broken (missing file)

        # Check content hashes
        for asset in resolved_assets:
            assert asset.content_hash is not None
            assert len(asset.content_hash) == 64  # SHA256 hex length

        # Test graph construction
        graph_constructor = ObsidianLinkGraphConstructor("test_vault", realistic_vault)
        note_nodes, link_relationships, tag_relationships, asset_relationships = graph_constructor.construct_graph(normalized_doc)

        # Verify graph construction
        assert len(note_nodes) >= 1
        assert len(asset_relationships) >= 3  # At least 3 valid asset relationships

        # Check asset relationships
        for asset_rel in asset_relationships:
            assert asset_rel.relationship_type == "embeds"
            assert asset_rel.note.vault_id == "test_vault"
            assert asset_rel.asset is not None

    def test_asset_deduplication_across_notes(self, realistic_vault):
        """Test that assets are deduplicated across multiple notes."""
        from futurnal.ingestion.obsidian.assets import AssetRegistry, AssetHasher

        registry = AssetRegistry("test_vault")
        hasher = AssetHasher()

        # Process multiple notes that reference the same assets
        note_paths = [
            realistic_vault / "notes" / "main_note.md",
            realistic_vault / "projects" / "project_note.md",
            realistic_vault / "daily" / "2024-01-01.md"
        ]

        all_assets = []
        for note_path in note_paths:
            content = note_path.read_text()
            normalized_doc = normalize_obsidian_document(
                content=content,
                source_path=note_path,
                vault_root=realistic_vault,
                vault_id="test_vault"
            )
            all_assets.extend(normalized_doc.metadata.assets)

        # Register assets and track deduplication
        unique_hashes = set()
        for asset in all_assets:
            if asset.resolved_path and not asset.is_broken:
                content_hash = hasher.compute_content_hash(asset.resolved_path)
                if content_hash:
                    is_new = registry.register_asset(asset.resolved_path, content_hash)
                    unique_hashes.add(content_hash)

        # Verify deduplication
        stats = registry.get_stats()
        assert stats["total_assets"] == len(unique_hashes)
        assert stats["unique_hashes"] == len(unique_hashes)

        # Diagram.png should be referenced from multiple notes but only stored once
        diagram_refs = [asset for asset in all_assets if "diagram.png" in asset.target]
        assert len(diagram_refs) >= 2  # Referenced from multiple notes

    def test_vault_connector_integration(self, realistic_vault):
        """Test complete vault connector integration with assets."""
        from futurnal.ingestion.local.state import StateStore
        from futurnal.ingestion.obsidian.descriptor import ObsidianVaultDescriptor

        # Create vault descriptor
        vault_descriptor = ObsidianVaultDescriptor(
            id="test_vault",
            name="Test Vault",
            base_path=realistic_vault
        )

        # Create vault source
        vault_source = ObsidianVaultSource.from_vault_descriptor(vault_descriptor)

        # Initialize connector
        state_store = StateStore()
        connector = ObsidianVaultConnector(
            workspace_dir=realistic_vault / ".futurnal",
            state_store=state_store
        )

        # Crawl vault
        records = connector.crawl_source(vault_source, job_id="test_job")

        # Verify crawl results
        markdown_files = [r for r in records if r.path.suffix == '.md']
        assert len(markdown_files) >= 3  # At least 3 markdown files

        # Process documents through ingestion pipeline
        elements = list(connector.ingest(vault_source, job_id="test_job"))

        # Verify element generation
        assert len(elements) >= 3  # At least one element per markdown file

        # Check that elements contain asset metadata
        for element in elements:
            assert "metadata" in element
            assert "futurnal" in element["metadata"]

            futurnal_metadata = element["metadata"]["futurnal"]
            assert "document_metadata" in futurnal_metadata

            doc_metadata = futurnal_metadata["document_metadata"]
            if doc_metadata.get("assets_count", 0) > 0:
                # This element came from a document with assets
                assert "link_graph" in futurnal_metadata
                link_graph = futurnal_metadata["link_graph"]
                assert "asset_relationships" in link_graph

    def test_asset_processing_pipeline_integration(self, realistic_vault):
        """Test AssetProcessingPipeline with realistic vault."""
        config = AssetTextExtractorConfig(
            enable_image_ocr=True,
            enable_pdf_extraction=True,
            max_file_size_mb=10
        )

        pipeline = AssetProcessingPipeline(realistic_vault, "test_vault", config)

        # Process main note
        main_note_path = realistic_vault / "notes" / "main_note.md"
        content = main_note_path.read_text()

        result = pipeline.process_document_assets(
            content,
            main_note_path,
            include_text_extraction=False  # Skip actual text extraction in tests
        )

        # Verify results structure
        assert "assets" in result
        assert "text_extractions" in result
        assert "statistics" in result

        # Check statistics
        stats = result["statistics"]
        assert stats["total_assets"] >= 4
        assert stats["broken_assets"] >= 1
        assert stats["processable_assets"] >= 3

        # Check asset data
        assets = result["assets"]
        resolved_assets = [a for a in assets if not a["is_broken"]]
        broken_assets = [a for a in assets if a["is_broken"]]

        assert len(resolved_assets) >= 3
        assert len(broken_assets) >= 1

        # Verify content hashes for resolved assets
        for asset in resolved_assets:
            assert asset["content_hash"] is not None

    def test_security_validation_integration(self, realistic_vault):
        """Test security validation in realistic scenarios."""
        # Create malicious content attempting path traversal
        malicious_note = realistic_vault / "malicious.md"
        malicious_content = """
        # Malicious Note

        Attempting path traversal: ![[../../../etc/passwd]]
        Another attempt: ![Evil](../../../../secrets.txt)
        Null byte injection: ![[image.png\x00.txt]]
        Absolute path: ![[/etc/hosts]]
        """
        malicious_note.write_text(malicious_content)

        # Process malicious note
        normalized_doc = normalize_obsidian_document(
            content=malicious_content,
            source_path=malicious_note,
            vault_root=realistic_vault,
            vault_id="test_vault"
        )

        # All malicious assets should be marked as broken
        for asset in normalized_doc.metadata.assets:
            assert asset.is_broken, f"Malicious asset should be broken: {asset.target}"

        # Clean up
        malicious_note.unlink()

    def test_performance_with_many_assets(self, realistic_vault):
        """Test performance with a document containing many assets."""
        import time

        # Create note with many asset references
        many_assets_content = "# Many Assets Note\n\n"

        # Add 50 asset references (mix of valid and invalid)
        for i in range(50):
            if i % 3 == 0:
                many_assets_content += f"Valid asset: ![[diagram.png]]\n"
            elif i % 3 == 1:
                many_assets_content += f"Invalid asset: ![[missing_{i}.png]]\n"
            else:
                many_assets_content += f"Another valid: ![Photo](images/photo.jpg)\n"

        many_assets_note = realistic_vault / "many_assets.md"
        many_assets_note.write_text(many_assets_content)

        # Measure processing time
        start_time = time.time()

        normalized_doc = normalize_obsidian_document(
            content=many_assets_content,
            source_path=many_assets_note,
            vault_root=realistic_vault,
            vault_id="test_vault"
        )

        processing_time = time.time() - start_time

        # Verify results
        assert len(normalized_doc.metadata.assets) == 50

        # Performance should be reasonable (under 1 second for 50 assets)
        assert processing_time < 1.0, f"Processing took too long: {processing_time}s"

        # Check deduplication working
        resolved_assets = [a for a in normalized_doc.metadata.assets if not a.is_broken]
        unique_paths = set(str(a.resolved_path) for a in resolved_assets if a.resolved_path)

        # Should have deduplicated to only unique paths
        assert len(unique_paths) <= 2  # Only diagram.png and photo.jpg should resolve

        # Clean up
        many_assets_note.unlink()

    def test_error_recovery_integration(self, realistic_vault):
        """Test error recovery in realistic error scenarios."""
        # Create note with mixed valid and problematic assets
        mixed_content = """
        # Mixed Content Note

        Valid asset: ![[diagram.png]]
        Missing asset: ![[definitely_missing.png]]
        Valid PDF: ![[document.pdf]]

        Permission issue will be simulated in test.
        """

        mixed_note = realistic_vault / "mixed.md"
        mixed_note.write_text(mixed_content)

        # Process note
        normalized_doc = normalize_obsidian_document(
            content=mixed_content,
            source_path=mixed_note,
            vault_root=realistic_vault,
            vault_id="test_vault"
        )

        # Should handle errors gracefully
        assert len(normalized_doc.metadata.assets) >= 3

        # Should have both successful and failed asset resolutions
        successful = [a for a in normalized_doc.metadata.assets if not a.is_broken]
        failed = [a for a in normalized_doc.metadata.assets if a.is_broken]

        assert len(successful) >= 2  # diagram.png and document.pdf should work
        assert len(failed) >= 1     # definitely_missing.png should fail

        # Construct graph (should not fail despite broken assets)
        graph_constructor = ObsidianLinkGraphConstructor("test_vault", realistic_vault)
        note_nodes, link_relationships, tag_relationships, asset_relationships = graph_constructor.construct_graph(normalized_doc)

        # Should create relationships only for valid assets
        valid_asset_relationships = [r for r in asset_relationships if not r.asset.is_broken]
        assert len(valid_asset_relationships) >= 2

        # Clean up
        mixed_note.unlink()

    def test_memory_usage_integration(self, realistic_vault):
        """Test memory usage remains reasonable during processing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process all notes in vault
        markdown_files = list(realistic_vault.rglob("*.md"))

        for md_file in markdown_files:
            content = md_file.read_text()
            normalized_doc = normalize_obsidian_document(
                content=content,
                source_path=md_file,
                vault_root=realistic_vault,
                vault_id="test_vault"
            )

            # Construct graph
            graph_constructor = ObsidianLinkGraphConstructor("test_vault", realistic_vault)
            graph_constructor.construct_graph(normalized_doc)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for small test vault)
        memory_increase_mb = memory_increase / (1024 * 1024)
        assert memory_increase_mb < 50, f"Memory usage increased by {memory_increase_mb}MB"


class TestRealWorldScenarios:
    """Test real-world scenarios and edge cases."""

    def test_obsidian_vault_structure_compatibility(self):
        """Test compatibility with real Obsidian vault structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)

            # Create typical Obsidian vault structure
            (vault_root / ".obsidian").mkdir()
            (vault_root / ".obsidian" / "config").mkdir()
            (vault_root / "Templates").mkdir()
            (vault_root / "Attachments").mkdir()
            (vault_root / "_resources").mkdir()

            # Create typical files
            (vault_root / ".obsidian" / "workspace").touch()
            (vault_root / "Templates" / "Daily Note.md").touch()
            (vault_root / "Attachments" / "image.png").write_bytes(b"fake image")

            # Create note referencing attachment
            note_content = """
            # Daily Note

            Screenshot: ![[Attachments/image.png]]
            """

            note_path = vault_root / "Daily Note.md"
            note_path.write_text(note_content)

            # Process note
            normalized_doc = normalize_obsidian_document(
                content=note_content,
                source_path=note_path,
                vault_root=vault_root,
                vault_id="real_vault"
            )

            # Should correctly resolve the attachment
            assert len(normalized_doc.metadata.assets) == 1
            asset = normalized_doc.metadata.assets[0]
            assert not asset.is_broken
            assert asset.resolved_path == vault_root / "Attachments" / "image.png"

    def test_large_vault_simulation(self):
        """Test processing efficiency with larger vault simulation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)

            # Create many directories and files
            for i in range(10):
                dir_path = vault_root / f"folder_{i}"
                dir_path.mkdir()

                # Create asset files
                (dir_path / f"image_{i}.png").write_bytes(b"fake image data")

                # Create notes referencing assets
                note_content = f"""
                # Note {i}

                Local image: ![[image_{i}.png]]
                Shared image: ![[../folder_0/image_0.png]]
                """

                (dir_path / f"note_{i}.md").write_text(note_content)

            # Process all notes
            all_assets = []
            for md_file in vault_root.rglob("*.md"):
                content = md_file.read_text()
                normalized_doc = normalize_obsidian_document(
                    content=content,
                    source_path=md_file,
                    vault_root=vault_root,
                    vault_id="large_vault"
                )
                all_assets.extend(normalized_doc.metadata.assets)

            # Verify processing completed successfully
            assert len(all_assets) == 20  # 2 assets per note * 10 notes

            # Verify most assets resolved correctly
            resolved_count = sum(1 for asset in all_assets if not asset.is_broken)
            assert resolved_count >= 15  # Most should resolve successfully

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode filenames and special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)

            # Create files with Unicode names (if filesystem supports it)
            try:
                unicode_image = vault_root / "图像.png"
                unicode_image.write_bytes(b"fake image")

                emoji_image = vault_root / "📷.jpg"
                emoji_image.write_bytes(b"fake jpeg")

                # Create note with Unicode asset references
                note_content = """
                # Unicode Test

                Chinese filename: ![[图像.png]]
                Emoji filename: ![[📷.jpg]]
                """

                note_path = vault_root / "unicode_note.md"
                note_path.write_text(note_content)

                # Process note
                normalized_doc = normalize_obsidian_document(
                    content=note_content,
                    source_path=note_path,
                    vault_root=vault_root,
                    vault_id="unicode_vault"
                )

                # Should handle Unicode filenames gracefully
                assert len(normalized_doc.metadata.assets) == 2

                # May or may not resolve depending on filesystem support
                # But should not crash
                for asset in normalized_doc.metadata.assets:
                    assert isinstance(asset.is_broken, bool)

            except (OSError, UnicodeError):
                # Filesystem doesn't support Unicode filenames - skip test
                pytest.skip("Filesystem doesn't support Unicode filenames")

    def test_case_sensitivity_scenarios(self):
        """Test case sensitivity handling across different scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)

            # Create files with specific case
            (vault_root / "Image.PNG").write_bytes(b"fake image")
            (vault_root / "Document.PDF").write_bytes(b"fake pdf")

            # Create note with different case references
            note_content = """
            # Case Test

            Different case: ![[image.png]]
            Exact case: ![[Image.PNG]]
            Mixed case: ![[DOCUMENT.pdf]]
            """

            note_path = vault_root / "case_test.md"
            note_path.write_text(note_content)

            # Process note
            normalized_doc = normalize_obsidian_document(
                content=note_content,
                source_path=note_path,
                vault_root=vault_root,
                vault_id="case_vault"
            )

            # Should handle case sensitivity based on filesystem
            assert len(normalized_doc.metadata.assets) == 3

            # At least one should resolve (the exact match)
            resolved_count = sum(1 for asset in normalized_doc.metadata.assets if not asset.is_broken)
            assert resolved_count >= 1