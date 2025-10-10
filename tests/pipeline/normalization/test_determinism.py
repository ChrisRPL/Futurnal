"""Determinism tests for normalization pipeline.

Validates that the normalization pipeline produces deterministic, byte-identical
outputs for identical inputs. This is a critical production requirement as specified
in the quality gates (12-quality-gates-testing.md).

Requirements Tested:
- ✓ Feature Requirement: "Deterministic outputs for identical inputs (idempotency)"
- ✓ Testing Strategy: "Re-run normalization to confirm identical outputs"
- ✓ Production Quality: Byte-identical SHA-256 hashes for same content

Test Coverage:
- Content hash determinism
- Document normalization idempotency
- Metadata consistency (excluding timestamps)
- Chunk hash determinism
- Cross-session consistency
"""

from __future__ import annotations

from pathlib import Path

import pytest

from futurnal.pipeline.models import compute_content_hash
from futurnal.pipeline.normalization import create_normalization_service
from tests.pipeline.normalization.test_utils import (
    assert_deterministic_hash,
    assert_documents_identical,
    assert_valid_sha256
)


# ---------------------------------------------------------------------------
# Content Hash Determinism Tests
# ---------------------------------------------------------------------------


@pytest.mark.determinism
class TestContentHashDeterminism:
    """Test suite for content hashing determinism."""

    def test_identical_content_produces_identical_hash(self):
        """Verify content hashing is deterministic.

        From requirements: "Deterministic outputs for identical inputs"
        """
        content = "Test document content for determinism validation"

        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)

        assert hash1 == hash2, "Same content must produce same hash"
        assert_valid_sha256(hash1)

    def test_hash_consistency_across_multiple_calls(self):
        """Verify hashing remains consistent across multiple calls."""
        content = "Multi-call test content with various characters: é, ñ, 中文"

        hashes = [compute_content_hash(content) for _ in range(10)]

        # All hashes must be identical
        assert len(set(hashes)) == 1, "All hashes must be identical"
        assert_valid_sha256(hashes[0])

    def test_different_content_produces_different_hash(self):
        """Verify different content produces different hashes."""
        content1 = "First document"
        content2 = "Second document"

        hash1 = compute_content_hash(content1)
        hash2 = compute_content_hash(content2)

        assert hash1 != hash2, "Different content must produce different hashes"

    def test_whitespace_affects_hash(self):
        """Verify whitespace differences affect hash (no normalization)."""
        content1 = "Content without extra spaces"
        content2 = "Content  without extra  spaces"  # Extra spaces

        hash1 = compute_content_hash(content1)
        hash2 = compute_content_hash(content2)

        assert hash1 != hash2, "Whitespace differences must affect hash"

    def test_unicode_content_hash_determinism(self):
        """Verify unicode content hashing is deterministic."""
        content = "Unicode content: 🚀 中文 العربية Ελληνικά"

        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)

        assert hash1 == hash2
        assert_valid_sha256(hash1)

    @pytest.mark.parametrize("content", [
        "",  # Empty
        " ",  # Single space
        "\n",  # Newline
        "A",  # Single char
        "A" * 10000,  # Long repeated
    ])
    def test_edge_case_content_hashing(self, content):
        """Test determinism with edge case content."""
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)

        assert hash1 == hash2
        assert_valid_sha256(hash1)


# ---------------------------------------------------------------------------
# Document Normalization Idempotency Tests
# ---------------------------------------------------------------------------


@pytest.mark.determinism
class TestDocumentNormalizationIdempotency:
    """Test suite for document normalization idempotency."""

    @pytest.mark.asyncio
    async def test_normalize_same_document_twice_produces_identical_outputs(
        self, markdown_simple
    ):
        """Verify normalizing same document twice produces identical results.

        From requirements: "Re-run normalization to confirm identical outputs"
        """
        service = create_normalization_service()

        # Normalize same file twice
        result1 = await service.normalize_document(
            file_path=markdown_simple,
            source_id="test_1",
            source_type="determinism_test"
        )

        result2 = await service.normalize_document(
            file_path=markdown_simple,
            source_id="test_2",
            source_type="determinism_test"
        )

        # Results must be identical
        assert result1.sha256 == result2.sha256
        assert result1.content == result2.content
        assert len(result1.chunks) == len(result2.chunks)

        # Verify deterministic assertions
        assert_documents_identical(result1, result2)

    @pytest.mark.asyncio
    async def test_multiple_normalizations_produce_identical_hashes(
        self, text_simple
    ):
        """Verify multiple normalizations of same file produce same hashes."""
        service = create_normalization_service()

        # Normalize same file 5 times
        results = []
        for i in range(5):
            result = await service.normalize_document(
                file_path=text_simple,
                source_id=f"test_{i}",
                source_type="determinism_test"
            )
            results.append(result)

        # All SHA-256 hashes must be identical
        sha256_hashes = [r.sha256 for r in results]
        assert len(set(sha256_hashes)) == 1, "All hashes must be identical"

        # All content hashes must be identical
        content_hashes = [r.metadata.content_hash for r in results]
        assert len(set(content_hashes)) == 1, "All content hashes must be identical"

    @pytest.mark.asyncio
    async def test_different_source_ids_same_content_identical_hash(
        self, markdown_simple
    ):
        """Verify different source IDs don't affect content hash.

        The source_id should not affect the content hash itself.
        """
        service = create_normalization_service()

        result1 = await service.normalize_document(
            file_path=markdown_simple,
            source_id="source_alpha",
            source_type="test"
        )

        result2 = await service.normalize_document(
            file_path=markdown_simple,
            source_id="source_beta",
            source_type="test"
        )

        # Content hash must be identical regardless of source_id
        assert result1.sha256 == result2.sha256
        assert result1.metadata.content_hash == result2.metadata.content_hash

    @pytest.mark.asyncio
    async def test_file_copy_produces_identical_hash(self, tmp_path, markdown_simple):
        """Verify copying a file and normalizing both produces identical hashes."""
        service = create_normalization_service()

        # Create a copy of the file
        original_content = markdown_simple.read_text(encoding="utf-8")
        copy_path = tmp_path / "copy.md"
        copy_path.write_text(original_content, encoding="utf-8")

        # Normalize both
        result_original = await service.normalize_document(
            file_path=markdown_simple,
            source_id="original",
            source_type="test"
        )

        result_copy = await service.normalize_document(
            file_path=copy_path,
            source_id="copy",
            source_type="test"
        )

        # Must produce identical hashes
        assert result_original.sha256 == result_copy.sha256
        assert_documents_identical(result_original, result_copy)


# ---------------------------------------------------------------------------
# Chunk Determinism Tests
# ---------------------------------------------------------------------------


@pytest.mark.determinism
class TestChunkDeterminism:
    """Test suite for chunk hash determinism."""

    @pytest.mark.asyncio
    async def test_chunk_hashes_are_deterministic(self, markdown_complex):
        """Verify chunk hashes are deterministic across normalizations."""
        service = create_normalization_service()

        # Normalize twice
        result1 = await service.normalize_document(
            file_path=markdown_complex,
            source_id="test_1",
            source_type="test"
        )

        result2 = await service.normalize_document(
            file_path=markdown_complex,
            source_id="test_2",
            source_type="test"
        )

        # Chunk counts must match
        assert len(result1.chunks) == len(result2.chunks)

        # All chunk hashes must be identical
        for idx, (chunk1, chunk2) in enumerate(zip(result1.chunks, result2.chunks)):
            assert chunk1.content_hash == chunk2.content_hash, (
                f"Chunk {idx} hash mismatch"
            )
            assert chunk1.content == chunk2.content, (
                f"Chunk {idx} content mismatch"
            )

    @pytest.mark.asyncio
    async def test_chunk_order_is_deterministic(self, markdown_large):
        """Verify chunk order remains consistent across normalizations."""
        service = create_normalization_service()

        # Normalize twice
        result1 = await service.normalize_document(
            file_path=markdown_large,
            source_id="test_1",
            source_type="test"
        )

        result2 = await service.normalize_document(
            file_path=markdown_large,
            source_id="test_2",
            source_type="test"
        )

        # Chunk indices must match
        for idx, (chunk1, chunk2) in enumerate(zip(result1.chunks, result2.chunks)):
            assert chunk1.chunk_index == chunk2.chunk_index == idx

    @pytest.mark.asyncio
    async def test_chunk_metadata_is_deterministic(self, markdown_complex):
        """Verify chunk metadata is deterministic (excluding timestamps)."""
        service = create_normalization_service()

        result1 = await service.normalize_document(
            file_path=markdown_complex,
            source_id="test_1",
            source_type="test"
        )

        result2 = await service.normalize_document(
            file_path=markdown_complex,
            source_id="test_2",
            source_type="test"
        )

        # Compare chunk metadata
        for chunk1, chunk2 in zip(result1.chunks, result2.chunks):
            assert chunk1.character_count == chunk2.character_count
            assert chunk1.word_count == chunk2.word_count
            assert chunk1.section_title == chunk2.section_title
            assert chunk1.heading_hierarchy == chunk2.heading_hierarchy


# ---------------------------------------------------------------------------
# Cross-Session Determinism Tests
# ---------------------------------------------------------------------------


@pytest.mark.determinism
class TestCrossSessionDeterminism:
    """Test determinism across different service instances."""

    @pytest.mark.asyncio
    async def test_different_service_instances_produce_identical_outputs(
        self, markdown_simple
    ):
        """Verify different service instances produce identical results."""
        # Create two separate service instances
        service1 = create_normalization_service()
        service2 = create_normalization_service()

        # Normalize with both services
        result1 = await service1.normalize_document(
            file_path=markdown_simple,
            source_id="test_1",
            source_type="test"
        )

        result2 = await service2.normalize_document(
            file_path=markdown_simple,
            source_id="test_2",
            source_type="test"
        )

        # Results must be identical
        assert_documents_identical(result1, result2)

    @pytest.mark.asyncio
    async def test_sequential_vs_separate_sessions(self, text_large):
        """Verify sequential normalization matches separate sessions."""
        # First session: normalize twice sequentially
        service1 = create_normalization_service()
        result1a = await service1.normalize_document(
            file_path=text_large,
            source_id="1a",
            source_type="test"
        )
        result1b = await service1.normalize_document(
            file_path=text_large,
            source_id="1b",
            source_type="test"
        )

        # Second session: new service instance
        service2 = create_normalization_service()
        result2 = await service2.normalize_document(
            file_path=text_large,
            source_id="2",
            source_type="test"
        )

        # All results must be identical
        assert result1a.sha256 == result1b.sha256 == result2.sha256
        assert_documents_identical(result1a, result2)
        assert_documents_identical(result1b, result2)


# ---------------------------------------------------------------------------
# Format-Specific Determinism Tests
# ---------------------------------------------------------------------------


@pytest.mark.determinism
class TestFormatSpecificDeterminism:
    """Test determinism across different document formats."""

    @pytest.mark.asyncio
    async def test_json_normalization_determinism(self, json_simple):
        """Verify JSON normalization is deterministic."""
        service = create_normalization_service()

        result1 = await service.normalize_document(
            file_path=json_simple,
            source_id="test_1",
            source_type="test"
        )

        result2 = await service.normalize_document(
            file_path=json_simple,
            source_id="test_2",
            source_type="test"
        )

        assert_documents_identical(result1, result2)

    @pytest.mark.asyncio
    async def test_yaml_normalization_determinism(self, yaml_simple):
        """Verify YAML normalization is deterministic."""
        service = create_normalization_service()

        result1 = await service.normalize_document(
            file_path=yaml_simple,
            source_id="test_1",
            source_type="test"
        )

        result2 = await service.normalize_document(
            file_path=yaml_simple,
            source_id="test_2",
            source_type="test"
        )

        assert_documents_identical(result1, result2)

    @pytest.mark.asyncio
    async def test_code_normalization_determinism(self, code_python):
        """Verify code file normalization is deterministic."""
        service = create_normalization_service()

        result1 = await service.normalize_document(
            file_path=code_python,
            source_id="test_1",
            source_type="test"
        )

        result2 = await service.normalize_document(
            file_path=code_python,
            source_id="test_2",
            source_type="test"
        )

        assert_documents_identical(result1, result2)

    @pytest.mark.asyncio
    async def test_csv_normalization_determinism(self, csv_simple):
        """Verify CSV normalization is deterministic."""
        service = create_normalization_service()

        result1 = await service.normalize_document(
            file_path=csv_simple,
            source_id="test_1",
            source_type="test"
        )

        result2 = await service.normalize_document(
            file_path=csv_simple,
            source_id="test_2",
            source_type="test"
        )

        assert_documents_identical(result1, result2)


# ---------------------------------------------------------------------------
# Determinism Summary Report
# ---------------------------------------------------------------------------


@pytest.mark.determinism
class TestDeterminismReport:
    """Generate determinism test summary report."""

    @pytest.mark.asyncio
    async def test_generate_determinism_report(
        self,
        markdown_simple,
        text_simple,
        json_simple,
        yaml_simple,
        code_python,
        tmp_path
    ):
        """Generate comprehensive determinism validation report."""
        service = create_normalization_service()

        test_files = {
            "markdown": markdown_simple,
            "text": text_simple,
            "json": json_simple,
            "yaml": yaml_simple,
            "code": code_python
        }

        report = {
            "test_name": "Determinism Validation Report",
            "formats_tested": [],
            "all_deterministic": True
        }

        for format_name, file_path in test_files.items():
            # Normalize twice
            result1 = await service.normalize_document(
                file_path=file_path,
                source_id=f"{format_name}_1",
                source_type="test"
            )

            result2 = await service.normalize_document(
                file_path=file_path,
                source_id=f"{format_name}_2",
                source_type="test"
            )

            # Check determinism
            is_deterministic = (result1.sha256 == result2.sha256)
            if not is_deterministic:
                report["all_deterministic"] = False

            report["formats_tested"].append({
                "format": format_name,
                "deterministic": is_deterministic,
                "hash": result1.sha256
            })

        # Print report
        print("\n" + "=" * 70)
        print("DETERMINISM VALIDATION REPORT")
        print("=" * 70)
        print(f"All Deterministic: {'✓ YES' if report['all_deterministic'] else '✗ NO'}")
        print("\nFormat Results:")
        for result in report["formats_tested"]:
            status = "✓" if result["deterministic"] else "✗"
            print(f"  {status} {result['format']}")
        print("=" * 70 + "\n")

        assert report["all_deterministic"], "All formats must be deterministic"
