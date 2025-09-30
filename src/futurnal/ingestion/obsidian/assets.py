"""Asset pipeline for processing embedded images and PDFs in Obsidian vaults.

This module provides secure detection, resolution, and processing of embedded assets
while enforcing vault sandboxing and providing content-based deduplication.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from .security import PathTraversalValidator, SecurityError

logger = logging.getLogger(__name__)


@dataclass
class ObsidianAsset:
    """Represents an embedded asset (image, PDF, etc.) in an Obsidian note."""

    # Source information
    target: str  # Original embed target from markdown
    display_text: Optional[str] = None  # Alt text for images
    is_embed: bool = True  # True for ![[]], False for ![]()

    # Resolution information
    resolved_path: Optional[Path] = None  # Absolute path within vault
    is_broken: bool = False  # True if asset file doesn't exist

    # Content information
    content_hash: Optional[str] = None  # SHA256 of file content
    file_size: Optional[int] = None  # Size in bytes
    mime_type: Optional[str] = None  # Detected MIME type

    # Position in source markdown
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None

    @property
    def is_image(self) -> bool:
        """Check if asset is an image type."""
        if not self.resolved_path:
            return False
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}
        return self.resolved_path.suffix.lower() in image_extensions

    @property
    def is_pdf(self) -> bool:
        """Check if asset is a PDF."""
        if not self.resolved_path:
            return False
        return self.resolved_path.suffix.lower() == '.pdf'

    @property
    def is_processable(self) -> bool:
        """Check if asset can be processed for text extraction."""
        return self.is_image or self.is_pdf


# Supported asset file extensions
SUPPORTED_ASSET_EXTENSIONS = {
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp',
    # Documents
    '.pdf',
    # Additional formats that might contain extractable content
    '.tiff', '.tif'
}


class AssetDetector:
    """Detects embedded assets in Obsidian markdown content."""

    # Regex patterns for asset detection
    WIKILINK_EMBED_PATTERN = re.compile(
        r'!\[\[([^\]]+?)\]\]',
        re.MULTILINE
    )

    MARKDOWN_IMAGE_PATTERN = re.compile(
        r'!\[([^\]]*)\]\(([^)"]+?)(?:\s+"[^"]*")?\)',
        re.MULTILINE
    )

    def __init__(self, supported_extensions: Optional[Set[str]] = None):
        """Initialize asset detector with configurable extensions."""
        self.supported_extensions = supported_extensions or SUPPORTED_ASSET_EXTENSIONS

    def detect_assets(self, content: str) -> List[ObsidianAsset]:
        """Detect all embedded assets in markdown content.

        Args:
            content: Markdown content to parse

        Returns:
            List of detected ObsidianAsset objects
        """
        assets = []

        # Detect wikilink embeds: ![[file.ext]]
        for match in self.WIKILINK_EMBED_PATTERN.finditer(content):
            target = match.group(1).strip()

            # Check if target looks like an asset
            if self._is_asset_target(target):
                asset = ObsidianAsset(
                    target=target,
                    is_embed=True,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                assets.append(asset)

        # Detect markdown image links: ![alt](path)
        for match in self.MARKDOWN_IMAGE_PATTERN.finditer(content):
            alt_text = match.group(1).strip() if match.group(1) else ""
            target = match.group(2)

            # Check if target looks like an asset
            if self._is_asset_target(target):
                asset = ObsidianAsset(
                    target=target,
                    display_text=alt_text,
                    is_embed=False,  # Markdown image, not wikilink embed
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                assets.append(asset)

        logger.debug(f"Detected {len(assets)} assets in content")
        return assets

    def _is_asset_target(self, target: str) -> bool:
        """Check if a link target appears to be an asset.

        Args:
            target: Link target to check

        Returns:
            True if target appears to be an asset
        """
        if not target:
            return False

        # Check for invalid characters that should be rejected
        if '\x00' in target:
            return False

        # Remove any section or block references
        clean_target = target.split('#')[0].split('^')[0]

        # Check file extension
        try:
            path = Path(clean_target)
            return path.suffix.lower() in self.supported_extensions
        except (ValueError, OSError):
            return False


class AssetResolver:
    """Resolves asset paths securely within vault boundaries."""

    def __init__(self, vault_root: Path, path_validator: Optional[PathTraversalValidator] = None):
        """Initialize asset resolver.

        Args:
            vault_root: Root directory of the vault
            path_validator: Optional custom path validator
        """
        self.vault_root = Path(vault_root).resolve()
        self.path_validator = path_validator or PathTraversalValidator(vault_root=self.vault_root)

    def resolve_asset(self, asset: ObsidianAsset, source_file_path: Path) -> ObsidianAsset:
        """Resolve asset path and validate security constraints.

        Args:
            asset: Asset to resolve
            source_file_path: Path of the source markdown file

        Returns:
            Updated asset with resolved path information
        """
        try:
            # Validate target path for security
            self.path_validator.validate_link_path(asset.target, source_file_path)

            # Resolve the path
            resolved_path = self._resolve_asset_path(asset.target, source_file_path)

            # Check if file exists and is accessible
            if resolved_path and resolved_path.exists():
                asset.resolved_path = resolved_path
                asset.is_broken = False

                # Get basic file metadata
                try:
                    stat = resolved_path.stat()
                    asset.file_size = stat.st_size
                    asset.mime_type = self._detect_mime_type(resolved_path)
                except (OSError, PermissionError) as e:
                    logger.warning(f"Cannot access asset metadata for {resolved_path}: {e}")
                    asset.is_broken = True
            else:
                asset.is_broken = True
                logger.debug(f"Asset not found: {asset.target}")

        except SecurityError as e:
            logger.warning(f"Security validation failed for asset {asset.target}: {e}")
            asset.is_broken = True
        except Exception as e:
            logger.error(f"Failed to resolve asset {asset.target}: {e}")
            asset.is_broken = True

        return asset

    def _resolve_asset_path(self, target: str, source_file_path: Path) -> Optional[Path]:
        """Resolve asset target to absolute path within vault.

        Args:
            target: Asset target from embed
            source_file_path: Source file containing the embed

        Returns:
            Resolved absolute path or None if resolution fails
        """
        try:
            # Clean target (remove section/block references)
            clean_target = target.split('#')[0].split('^')[0].strip()

            # Handle different path formats
            if clean_target.startswith('/'):
                # Absolute path within vault (rare in Obsidian)
                candidate_path = self.vault_root / clean_target.lstrip('/')
            else:
                # Relative path - try multiple resolution strategies
                source_dir = source_file_path.parent

                # Strategy 1: Relative to source file directory
                candidate_path = source_dir / clean_target
                if candidate_path.exists():
                    return candidate_path.resolve()

                # Strategy 2: Search in common asset directories
                asset_dirs = ['assets', 'attachments', 'images', 'files']
                for asset_dir in asset_dirs:
                    asset_dir_path = self.vault_root / asset_dir
                    if asset_dir_path.exists():
                        candidate_path = asset_dir_path / clean_target
                        if candidate_path.exists():
                            return candidate_path.resolve()

                # Strategy 3: Search from vault root
                candidate_path = self.vault_root / clean_target
                if candidate_path.exists():
                    return candidate_path.resolve()

                # Strategy 4: Recursive search (limited depth for performance)
                candidate_path = self._search_asset_in_vault(clean_target)
                if candidate_path:
                    return candidate_path

            # Validate the candidate path is within vault
            if candidate_path:
                resolved = candidate_path.resolve()
                if str(resolved).startswith(str(self.vault_root)):
                    return resolved

            return None

        except (OSError, ValueError) as e:
            logger.debug(f"Path resolution failed for {target}: {e}")
            return None

    def _search_asset_in_vault(self, filename: str, max_depth: int = 3) -> Optional[Path]:
        """Search for asset file within vault (limited depth for performance).

        Args:
            filename: Filename to search for
            max_depth: Maximum directory depth to search

        Returns:
            Found path or None
        """
        try:
            for depth in range(max_depth + 1):
                pattern = '*/' * depth + filename
                matches = list(self.vault_root.glob(pattern))
                if matches:
                    # Return first match
                    return matches[0].resolve()
            return None
        except (OSError, ValueError):
            return None

    def _detect_mime_type(self, file_path: Path) -> Optional[str]:
        """Detect MIME type from file extension.

        Args:
            file_path: Path to file

        Returns:
            MIME type string or None
        """
        extension = file_path.suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.svg': 'image/svg+xml',
            '.webp': 'image/webp',
            '.pdf': 'application/pdf',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        return mime_types.get(extension)


class AssetHasher:
    """Provides content-based hashing for asset deduplication."""

    def __init__(self):
        """Initialize asset hasher."""
        self._hash_cache: Dict[Path, str] = {}

    def compute_content_hash(self, file_path: Path) -> Optional[str]:
        """Compute SHA256 hash of file content.

        Args:
            file_path: Path to file

        Returns:
            Hex string of SHA256 hash or None if computation fails
        """
        # Check cache first
        if file_path in self._hash_cache:
            return self._hash_cache[file_path]

        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                while chunk := f.read(8192):
                    hasher.update(chunk)

            content_hash = hasher.hexdigest()
            self._hash_cache[file_path] = content_hash
            return content_hash

        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the hash cache."""
        self._hash_cache.clear()


class AssetRegistry:
    """Registry for tracking asset hashes and preventing duplication."""

    def __init__(self, vault_id: str):
        """Initialize asset registry for a specific vault.

        Args:
            vault_id: Unique identifier for the vault
        """
        self.vault_id = vault_id
        self._hash_to_path: Dict[str, Path] = {}
        self._path_to_hash: Dict[Path, str] = {}

    def register_asset(self, asset_path: Path, content_hash: str) -> bool:
        """Register an asset with its content hash.

        Args:
            asset_path: Path to the asset file
            content_hash: SHA256 hash of content

        Returns:
            True if asset is new, False if already registered
        """
        if content_hash in self._hash_to_path:
            # Asset already exists with this hash
            existing_path = self._hash_to_path[content_hash]
            if existing_path != asset_path:
                logger.debug(f"Duplicate asset detected: {asset_path} has same content as {existing_path}")
            return False

        # Register new asset
        self._hash_to_path[content_hash] = asset_path
        self._path_to_hash[asset_path] = content_hash
        logger.debug(f"Registered new asset: {asset_path} with hash {content_hash[:16]}...")
        return True

    def get_asset_by_hash(self, content_hash: str) -> Optional[Path]:
        """Get asset path by content hash.

        Args:
            content_hash: SHA256 hash to look up

        Returns:
            Path to asset or None if not found
        """
        return self._hash_to_path.get(content_hash)

    def get_hash_by_path(self, asset_path: Path) -> Optional[str]:
        """Get content hash by asset path.

        Args:
            asset_path: Path to look up

        Returns:
            Content hash or None if not found
        """
        return self._path_to_hash.get(asset_path)

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if an asset with this hash already exists.

        Args:
            content_hash: Hash to check

        Returns:
            True if duplicate exists
        """
        return content_hash in self._hash_to_path

    def get_stats(self) -> Dict[str, int]:
        """Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        return {
            'total_assets': len(self._hash_to_path),
            'unique_hashes': len(set(self._hash_to_path.keys())),
        }


class AssetPipeline:
    """Main asset processing pipeline that coordinates detection, resolution, and hashing."""

    def __init__(
        self,
        vault_root: Path,
        vault_id: str,
        supported_extensions: Optional[Set[str]] = None
    ):
        """Initialize asset pipeline.

        Args:
            vault_root: Root directory of the vault
            vault_id: Unique identifier for the vault
            supported_extensions: Optional set of supported file extensions
        """
        self.vault_root = Path(vault_root)
        self.vault_id = vault_id

        # Initialize components
        self.detector = AssetDetector(supported_extensions)
        self.resolver = AssetResolver(vault_root)
        self.hasher = AssetHasher()
        self.registry = AssetRegistry(vault_id)

    def process_assets(self, content: str, source_file_path: Path) -> List[ObsidianAsset]:
        """Process all assets in markdown content.

        Args:
            content: Markdown content containing asset embeds
            source_file_path: Path to the source markdown file

        Returns:
            List of processed ObsidianAsset objects
        """
        # Step 1: Detect assets
        assets = self.detector.detect_assets(content)

        if not assets:
            return []

        logger.debug(f"Processing {len(assets)} detected assets from {source_file_path}")

        processed_assets = []
        for asset in assets:
            try:
                # Step 2: Resolve path
                resolved_asset = self.resolver.resolve_asset(asset, source_file_path)

                # Step 3: Compute content hash if file exists
                if resolved_asset.resolved_path and not resolved_asset.is_broken:
                    content_hash = self.hasher.compute_content_hash(resolved_asset.resolved_path)
                    if content_hash:
                        resolved_asset.content_hash = content_hash

                        # Step 4: Register in deduplication registry
                        is_new = self.registry.register_asset(resolved_asset.resolved_path, content_hash)
                        if not is_new:
                            logger.debug(f"Asset {resolved_asset.target} is a duplicate")

                processed_assets.append(resolved_asset)

            except Exception as e:
                logger.error(f"Failed to process asset {asset.target}: {e}")
                asset.is_broken = True
                processed_assets.append(asset)

        return processed_assets

    def get_statistics(self) -> Dict[str, Union[int, Dict]]:
        """Get pipeline statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            'vault_id': self.vault_id,
            'vault_root': str(self.vault_root),
            'supported_extensions': list(self.detector.supported_extensions),
            'registry_stats': self.registry.get_stats(),
        }