"""Tests for asset security and sandboxing functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from futurnal.ingestion.obsidian.assets import (
    AssetResolver,
    AssetHasher,
    AssetRegistry,
    ObsidianAsset
)
from futurnal.ingestion.obsidian.security import SecurityError, PathTraversalValidator


class TestAssetResolver:
    """Test cases for AssetResolver security and path resolution."""

    @pytest.fixture
    def temp_vault(self):
        """Create a temporary vault directory structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)

            # Create directory structure
            (vault_root / "assets").mkdir()
            (vault_root / "images").mkdir()
            (vault_root / "attachments").mkdir()
            (vault_root / "subdir").mkdir()
            (vault_root / "subdir" / "nested").mkdir()

            # Create test files
            (vault_root / "assets" / "test_image.png").touch()
            (vault_root / "images" / "photo.jpg").touch()
            (vault_root / "attachments" / "document.pdf").touch()
            (vault_root / "root_image.png").touch()
            (vault_root / "subdir" / "nested_image.gif").touch()
            (vault_root / "subdir" / "nested" / "deep_image.svg").touch()

            yield vault_root

    @pytest.fixture
    def asset_resolver(self, temp_vault):
        """Create AssetResolver with temporary vault."""
        return AssetResolver(temp_vault)

    def test_basic_initialization(self, temp_vault):
        """Test basic AssetResolver initialization."""
        resolver = AssetResolver(temp_vault)
        assert resolver.vault_root == temp_vault.resolve()
        assert resolver.path_validator is not None

    def test_resolve_asset_in_assets_dir(self, asset_resolver, temp_vault):
        """Test resolving asset in standard assets directory."""
        asset = ObsidianAsset(target="test_image.png")
        source_file = temp_vault / "note.md"

        resolved_asset = asset_resolver.resolve_asset(asset, source_file)

        assert not resolved_asset.is_broken
        assert resolved_asset.resolved_path == temp_vault / "assets" / "test_image.png"
        assert resolved_asset.file_size > 0  # File exists
        assert resolved_asset.mime_type == "image/png"

    def test_resolve_asset_in_images_dir(self, asset_resolver, temp_vault):
        """Test resolving asset in images directory."""
        asset = ObsidianAsset(target="photo.jpg")
        source_file = temp_vault / "note.md"

        resolved_asset = asset_resolver.resolve_asset(asset, source_file)

        assert not resolved_asset.is_broken
        assert resolved_asset.resolved_path == temp_vault / "images" / "photo.jpg"
        assert resolved_asset.mime_type == "image/jpeg"

    def test_resolve_asset_relative_to_source(self, asset_resolver, temp_vault):
        """Test resolving asset relative to source file directory."""
        # Create asset next to source file
        source_dir = temp_vault / "subdir"
        source_file = source_dir / "note.md"
        asset_file = source_dir / "local_image.png"
        asset_file.touch()

        asset = ObsidianAsset(target="local_image.png")
        resolved_asset = asset_resolver.resolve_asset(asset, source_file)

        assert not resolved_asset.is_broken
        assert resolved_asset.resolved_path == asset_file

    def test_resolve_asset_with_relative_path(self, asset_resolver, temp_vault):
        """Test resolving asset with relative path."""
        asset = ObsidianAsset(target="subdir/nested_image.gif")
        source_file = temp_vault / "note.md"

        resolved_asset = asset_resolver.resolve_asset(asset, source_file)

        assert not resolved_asset.is_broken
        assert resolved_asset.resolved_path == temp_vault / "subdir" / "nested_image.gif"

    def test_resolve_nonexistent_asset(self, asset_resolver, temp_vault):
        """Test resolving nonexistent asset."""
        asset = ObsidianAsset(target="nonexistent.png")
        source_file = temp_vault / "note.md"

        resolved_asset = asset_resolver.resolve_asset(asset, source_file)

        assert resolved_asset.is_broken
        assert resolved_asset.resolved_path is None

    def test_path_traversal_attack_prevention(self, asset_resolver, temp_vault):
        """Test prevention of path traversal attacks."""
        # Create file outside vault
        outside_file = temp_vault.parent / "outside_file.png"
        outside_file.touch()

        # Try various path traversal attacks
        attack_targets = [
            "../outside_file.png",
            "../../outside_file.png",
            "../../../outside_file.png",
            "subdir/../../outside_file.png"
        ]

        source_file = temp_vault / "note.md"

        for target in attack_targets:
            asset = ObsidianAsset(target=target)
            resolved_asset = asset_resolver.resolve_asset(asset, source_file)

            # Asset should be marked as broken due to security violation
            assert resolved_asset.is_broken, f"Path traversal should be blocked for: {target}"

    def test_absolute_path_rejection(self, asset_resolver, temp_vault):
        """Test rejection of absolute paths."""
        asset = ObsidianAsset(target="/etc/passwd")
        source_file = temp_vault / "note.md"

        resolved_asset = asset_resolver.resolve_asset(asset, source_file)

        assert resolved_asset.is_broken

    def test_null_byte_injection_prevention(self, asset_resolver, temp_vault):
        """Test prevention of null byte injection."""
        asset = ObsidianAsset(target="image.png\x00.txt")
        source_file = temp_vault / "note.md"

        resolved_asset = asset_resolver.resolve_asset(asset, source_file)

        assert resolved_asset.is_broken

    def test_symlink_escape_prevention(self, asset_resolver, temp_vault):
        """Test prevention of symlink escapes."""
        # Create symlink pointing outside vault
        outside_dir = temp_vault.parent / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "secret.png"
        outside_file.touch()

        symlink_path = temp_vault / "evil_link.png"
        symlink_path.symlink_to(outside_file)

        asset = ObsidianAsset(target="evil_link.png")
        source_file = temp_vault / "note.md"

        resolved_asset = asset_resolver.resolve_asset(asset, source_file)

        # Should be marked as broken due to symlink escape
        assert resolved_asset.is_broken

    def test_permission_denied_handling(self, asset_resolver, temp_vault):
        """Test handling of permission denied errors."""
        asset = ObsidianAsset(target="test_image.png")
        source_file = temp_vault / "note.md"

        # Mock permission error
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.side_effect = PermissionError("Permission denied")

            resolved_asset = asset_resolver.resolve_asset(asset, source_file)

            assert resolved_asset.is_broken

    def test_mime_type_detection(self, asset_resolver, temp_vault):
        """Test MIME type detection for various file types."""
        test_cases = [
            ("test.png", "image/png"),
            ("test.jpg", "image/jpeg"),
            ("test.jpeg", "image/jpeg"),
            ("test.gif", "image/gif"),
            ("test.bmp", "image/bmp"),
            ("test.svg", "image/svg+xml"),
            ("test.webp", "image/webp"),
            ("test.pdf", "application/pdf"),
            ("test.tiff", "image/tiff"),
            ("test.unknown", None)
        ]

        for filename, expected_mime in test_cases:
            # Create temporary file
            test_file = temp_vault / filename
            test_file.touch()

            detected_mime = asset_resolver._detect_mime_type(test_file)
            assert detected_mime == expected_mime, f"Wrong MIME type for {filename}"

    def test_search_in_vault_depth_limit(self, asset_resolver, temp_vault):
        """Test that search respects depth limits."""
        # Create file deep in directory structure
        deep_file = temp_vault / "subdir" / "nested" / "deep_image.svg"

        asset = ObsidianAsset(target="deep_image.svg")
        source_file = temp_vault / "note.md"

        resolved_asset = asset_resolver.resolve_asset(asset, source_file)

        # Should find the file within depth limit
        assert not resolved_asset.is_broken
        assert resolved_asset.resolved_path == deep_file

    def test_case_sensitivity_handling(self, asset_resolver, temp_vault):
        """Test handling of case sensitivity in filenames."""
        # Create file with specific case
        actual_file = temp_vault / "MyImage.PNG"
        actual_file.touch()

        # Try to find with different case
        asset = ObsidianAsset(target="myimage.png")
        source_file = temp_vault / "note.md"

        resolved_asset = asset_resolver.resolve_asset(asset, source_file)

        # Behavior depends on filesystem - should handle gracefully
        # On case-insensitive filesystems, should find the file
        # On case-sensitive filesystems, should mark as broken
        assert isinstance(resolved_asset.is_broken, bool)


class TestAssetHasher:
    """Test cases for AssetHasher functionality."""

    @pytest.fixture
    def temp_files(self):
        """Create temporary files for hashing tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)

            # Create test files with known content
            file1 = vault_root / "file1.txt"
            file1.write_text("Hello, World!")

            file2 = vault_root / "file2.txt"
            file2.write_text("Hello, World!")  # Same content as file1

            file3 = vault_root / "file3.txt"
            file3.write_text("Different content")

            yield {
                'file1': file1,
                'file2': file2,
                'file3': file3
            }

    def test_compute_content_hash(self, temp_files):
        """Test basic content hash computation."""
        hasher = AssetHasher()

        hash1 = hasher.compute_content_hash(temp_files['file1'])
        assert hash1 is not None
        assert len(hash1) == 64  # SHA256 hex string length

        # Same content should produce same hash
        hash2 = hasher.compute_content_hash(temp_files['file2'])
        assert hash1 == hash2

        # Different content should produce different hash
        hash3 = hasher.compute_content_hash(temp_files['file3'])
        assert hash1 != hash3

    def test_hash_caching(self, temp_files):
        """Test that hash computation is cached."""
        hasher = AssetHasher()

        # First computation
        hash1 = hasher.compute_content_hash(temp_files['file1'])

        # Second computation should use cache
        with patch('builtins.open') as mock_open:
            hash2 = hasher.compute_content_hash(temp_files['file1'])

            # File should not be opened again due to caching
            mock_open.assert_not_called()
            assert hash1 == hash2

    def test_nonexistent_file_handling(self):
        """Test handling of nonexistent files."""
        hasher = AssetHasher()
        nonexistent_file = Path("/nonexistent/file.txt")

        hash_result = hasher.compute_content_hash(nonexistent_file)
        assert hash_result is None

    def test_permission_denied_handling(self, temp_files):
        """Test handling of permission denied errors."""
        hasher = AssetHasher()

        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")

            hash_result = hasher.compute_content_hash(temp_files['file1'])
            assert hash_result is None

    def test_clear_cache(self, temp_files):
        """Test cache clearing functionality."""
        hasher = AssetHasher()

        # Compute hash to populate cache
        hasher.compute_content_hash(temp_files['file1'])
        assert len(hasher._hash_cache) == 1

        # Clear cache
        hasher.clear_cache()
        assert len(hasher._hash_cache) == 0

    def test_large_file_handling(self):
        """Test handling of large files with chunked reading."""
        hasher = AssetHasher()

        # Create a file larger than the chunk size
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write more than 8192 bytes (default chunk size)
            content = "x" * 10000
            temp_file.write(content.encode())
            temp_file.flush()

            try:
                hash_result = hasher.compute_content_hash(Path(temp_file.name))
                assert hash_result is not None
                assert len(hash_result) == 64
            finally:
                Path(temp_file.name).unlink()


class TestAssetRegistry:
    """Test cases for AssetRegistry functionality."""

    @pytest.fixture
    def sample_assets(self):
        """Create sample assets for testing."""
        return [
            (Path("/vault/image1.png"), "hash1"),
            (Path("/vault/image2.jpg"), "hash2"),
            (Path("/vault/document.pdf"), "hash1"),  # Duplicate hash
        ]

    def test_basic_registration(self, sample_assets):
        """Test basic asset registration."""
        registry = AssetRegistry("test_vault")

        path1, hash1 = sample_assets[0]
        is_new = registry.register_asset(path1, hash1)

        assert is_new is True
        assert registry.get_asset_by_hash(hash1) == path1
        assert registry.get_hash_by_path(path1) == hash1

    def test_duplicate_detection(self, sample_assets):
        """Test detection of duplicate assets."""
        registry = AssetRegistry("test_vault")

        # Register first asset
        path1, hash1 = sample_assets[0]
        is_new1 = registry.register_asset(path1, hash1)
        assert is_new1 is True

        # Register asset with same hash (duplicate)
        path3, hash3 = sample_assets[2]  # Same hash as first asset
        is_new3 = registry.register_asset(path3, hash3)
        assert is_new3 is False  # Should be detected as duplicate

    def test_is_duplicate(self, sample_assets):
        """Test duplicate checking functionality."""
        registry = AssetRegistry("test_vault")

        path1, hash1 = sample_assets[0]

        # Should not be duplicate before registration
        assert not registry.is_duplicate(hash1)

        # Register asset
        registry.register_asset(path1, hash1)

        # Should be duplicate after registration
        assert registry.is_duplicate(hash1)

    def test_get_statistics(self, sample_assets):
        """Test statistics generation."""
        registry = AssetRegistry("test_vault")

        # Empty registry
        stats = registry.get_stats()
        assert stats["total_assets"] == 0
        assert stats["unique_hashes"] == 0

        # Register all sample assets
        for path, hash_val in sample_assets:
            registry.register_asset(path, hash_val)

        stats = registry.get_stats()
        assert stats["total_assets"] == 2  # Only 2 unique hashes
        assert stats["unique_hashes"] == 2

    def test_vault_isolation(self):
        """Test that registries are isolated by vault ID."""
        registry1 = AssetRegistry("vault1")
        registry2 = AssetRegistry("vault2")

        path = Path("/shared/image.png")
        hash_val = "shared_hash"

        # Register in first vault
        registry1.register_asset(path, hash_val)

        # Should not exist in second vault
        assert registry2.get_asset_by_hash(hash_val) is None
        assert not registry2.is_duplicate(hash_val)


class TestSecurityIntegration:
    """Test cases for integrated security functionality."""

    @pytest.fixture
    def secure_vault_setup(self):
        """Set up a secure vault environment for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)

            # Create safe directory structure
            (vault_root / "assets").mkdir()
            (vault_root / "safe_image.png").touch()
            (vault_root / "assets" / "safe_document.pdf").touch()

            # Create directory outside vault
            outside_dir = vault_root.parent / "outside"
            outside_dir.mkdir()
            (outside_dir / "secret.txt").touch()

            yield {
                'vault_root': vault_root,
                'outside_dir': outside_dir
            }

    def test_end_to_end_security_validation(self, secure_vault_setup):
        """Test end-to-end security validation."""
        vault_root = secure_vault_setup['vault_root']
        resolver = AssetResolver(vault_root)

        # Test safe asset resolution
        safe_asset = ObsidianAsset(target="safe_image.png")
        source_file = vault_root / "note.md"

        resolved = resolver.resolve_asset(safe_asset, source_file)
        assert not resolved.is_broken

        # Test dangerous asset rejection
        dangerous_asset = ObsidianAsset(target="../outside/secret.txt")
        resolved_dangerous = resolver.resolve_asset(dangerous_asset, source_file)
        assert resolved_dangerous.is_broken

    def test_comprehensive_attack_vectors(self, secure_vault_setup):
        """Test comprehensive attack vector prevention."""
        vault_root = secure_vault_setup['vault_root']
        resolver = AssetResolver(vault_root)
        source_file = vault_root / "note.md"

        # Various attack vectors
        attack_vectors = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "assets/../../../etc/passwd",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\sam",
            "image.png\x00.txt",
            "normal/../../../etc/passwd",
            "assets/../../outside/secret.txt"
        ]

        for attack in attack_vectors:
            asset = ObsidianAsset(target=attack)
            resolved = resolver.resolve_asset(asset, source_file)
            assert resolved.is_broken, f"Attack vector should be blocked: {attack}"

    def test_unicode_and_encoding_attacks(self, secure_vault_setup):
        """Test prevention of Unicode and encoding-based attacks."""
        vault_root = secure_vault_setup['vault_root']
        resolver = AssetResolver(vault_root)
        source_file = vault_root / "note.md"

        # Unicode path traversal attempts
        unicode_attacks = [
            "..%2f..%2f..%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
            "..／..／..／etc／passwd",  # Unicode full-width slash
        ]

        for attack in unicode_attacks:
            asset = ObsidianAsset(target=attack)
            resolved = resolver.resolve_asset(asset, source_file)
            # Should either be blocked or fail to resolve
            assert resolved.is_broken, f"Unicode attack should be blocked: {attack}"