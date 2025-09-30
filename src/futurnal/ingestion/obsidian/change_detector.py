"""Advanced change detection for Obsidian vault synchronization.

This module provides sophisticated change detection algorithms that build on
the existing ObsidianPathTracker to provide enhanced content analysis,
metadata change detection, and optimized rename/move detection.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..local.state import FileRecord, StateStore
from .path_tracker import ObsidianPathTracker, PathChange

logger = logging.getLogger(__name__)


@dataclass
class ContentChange:
    """Represents a detected content change in a file."""
    file_path: Path
    change_type: str  # 'content', 'metadata', 'frontmatter', 'mixed'
    old_checksum: str
    new_checksum: str
    content_checksum: Optional[str] = None  # Checksum of content without frontmatter
    metadata_checksum: Optional[str] = None  # Checksum of frontmatter only
    detected_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, any] = field(default_factory=dict)

    def has_content_change(self) -> bool:
        """Check if this represents a content change (not just metadata)."""
        return self.change_type in {'content', 'mixed'}

    def has_metadata_change(self) -> bool:
        """Check if this represents a metadata/frontmatter change."""
        return self.change_type in {'metadata', 'frontmatter', 'mixed'}

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": str(self.file_path),
            "change_type": self.change_type,
            "old_checksum": self.old_checksum,
            "new_checksum": self.new_checksum,
            "content_checksum": self.content_checksum,
            "metadata_checksum": self.metadata_checksum,
            "detected_at": self.detected_at.isoformat(),
            "has_content_change": self.has_content_change(),
            "has_metadata_change": self.has_metadata_change(),
            "metadata": self.metadata,
        }


@dataclass
class SimilarityMatch:
    """Represents a similarity match between two files."""
    old_file: FileRecord
    new_file: FileRecord
    similarity_score: float
    match_type: str  # 'content', 'name', 'path', 'composite'
    metadata: Dict[str, any] = field(default_factory=dict)


class AdvancedChangeDetector:
    """Enhanced change detection building on ObsidianPathTracker.

    Provides sophisticated algorithms for detecting content changes,
    metadata changes, and improved rename/move detection using
    multiple similarity metrics.
    """

    def __init__(
        self,
        vault_id: str,
        vault_root: Path,
        state_store: StateStore,
        path_tracker: Optional[ObsidianPathTracker] = None,
        *,
        content_similarity_threshold: float = 0.8,
        name_similarity_threshold: float = 0.7,
        composite_similarity_threshold: float = 0.75,
        max_similarity_candidates: int = 10,
        enable_content_analysis: bool = True,
        enable_metadata_analysis: bool = True,
    ):
        self.vault_id = vault_id
        self.vault_root = Path(vault_root)
        self.state_store = state_store
        self.path_tracker = path_tracker

        # Similarity thresholds
        self.content_similarity_threshold = content_similarity_threshold
        self.name_similarity_threshold = name_similarity_threshold
        self.composite_similarity_threshold = composite_similarity_threshold
        self.max_similarity_candidates = max_similarity_candidates

        # Feature flags
        self.enable_content_analysis = enable_content_analysis
        self.enable_metadata_analysis = enable_metadata_analysis

        # Caching for performance
        self._content_cache: Dict[str, Tuple[str, str]] = {}  # checksum -> (content, metadata)
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        self._cache_max_size = 1000

        # Frontmatter regex pattern
        self._frontmatter_pattern = re.compile(
            r'^---\s*\n(.*?)\n---\s*\n',
            re.DOTALL | re.MULTILINE
        )

    def detect_changes(self, current_records: List[FileRecord]) -> Tuple[List[PathChange], List[ContentChange]]:
        """Detect both path changes and content changes.

        Args:
            current_records: Current file records from vault scan

        Returns:
            Tuple of (path_changes, content_changes)
        """
        start_time = time.time()

        # Use existing path tracker for basic path change detection
        path_changes = []
        if self.path_tracker:
            path_changes = self.path_tracker.detect_path_changes(current_records)

        # Detect content changes with enhanced analysis
        content_changes = self._detect_content_changes(current_records, path_changes)

        # Detect additional path changes using advanced similarity
        additional_path_changes = self._detect_advanced_path_changes(current_records, path_changes)
        path_changes.extend(additional_path_changes)

        detection_time = time.time() - start_time
        logger.debug(
            f"Change detection completed in {detection_time:.2f}s: "
            f"{len(path_changes)} path changes, {len(content_changes)} content changes"
        )

        return path_changes, content_changes

    def _detect_content_changes(
        self,
        current_records: List[FileRecord],
        known_path_changes: List[PathChange]
    ) -> List[ContentChange]:
        """Detect content and metadata changes in files."""
        if not self.enable_content_analysis:
            return []

        content_changes = []
        path_changed_files = {change.new_path for change in known_path_changes}

        # Get stored records for comparison
        stored_records = {record.path: record for record in self.state_store.iter_all()}

        for current_record in current_records:
            # Skip non-markdown files
            if current_record.path.suffix.lower() not in {'.md', '.markdown'}:
                continue

            # Skip files that we know have path changes (they'll be handled separately)
            if current_record.path in path_changed_files:
                continue

            stored_record = stored_records.get(current_record.path)
            if not stored_record:
                continue  # New file, not a change

            # Check if file has actually changed
            if stored_record.sha256 == current_record.sha256:
                continue  # No change

            # Analyze the type of change
            change = self._analyze_file_change(current_record, stored_record)
            if change:
                content_changes.append(change)

        return content_changes

    def _analyze_file_change(self, current_record: FileRecord, stored_record: FileRecord) -> Optional[ContentChange]:
        """Analyze the type of change in a file."""
        try:
            # Read both versions of the file
            current_content = self._read_file_content(current_record.path)
            if current_content is None:
                return None

            # Try to get stored content from cache or read from backup
            stored_content = self._get_stored_content(stored_record)
            if stored_content is None:
                # Fallback: treat as general content change
                return ContentChange(
                    file_path=current_record.path,
                    change_type='content',
                    old_checksum=stored_record.sha256,
                    new_checksum=current_record.sha256,
                    metadata={"analysis": "no_stored_content_available"}
                )

            # Parse frontmatter and content for both versions
            current_frontmatter, current_body = self._parse_markdown_content(current_content)
            stored_frontmatter, stored_body = self._parse_markdown_content(stored_content)

            # Calculate separate checksums
            current_content_hash = self._calculate_checksum(current_body)
            stored_content_hash = self._calculate_checksum(stored_body)
            current_metadata_hash = self._calculate_checksum(current_frontmatter)
            stored_metadata_hash = self._calculate_checksum(stored_frontmatter)

            # Determine change type
            content_changed = current_content_hash != stored_content_hash
            metadata_changed = current_metadata_hash != stored_metadata_hash

            if content_changed and metadata_changed:
                change_type = 'mixed'
            elif content_changed:
                change_type = 'content'
            elif metadata_changed:
                change_type = 'frontmatter'
            else:
                change_type = 'metadata'  # Other metadata like file stats

            return ContentChange(
                file_path=current_record.path,
                change_type=change_type,
                old_checksum=stored_record.sha256,
                new_checksum=current_record.sha256,
                content_checksum=current_content_hash,
                metadata_checksum=current_metadata_hash,
                metadata={
                    "content_changed": content_changed,
                    "frontmatter_changed": metadata_changed,
                    "old_content_hash": stored_content_hash,
                    "old_metadata_hash": stored_metadata_hash,
                }
            )

        except Exception as e:
            logger.error(f"Failed to analyze file change for {current_record.path}: {e}")
            return None

    def _detect_advanced_path_changes(
        self,
        current_records: List[FileRecord],
        known_path_changes: List[PathChange]
    ) -> List[PathChange]:
        """Detect additional path changes using advanced similarity matching."""
        if not self.enable_content_analysis:
            return []

        # Get stored records
        stored_records = list(self.state_store.iter_all())
        current_paths = {record.path for record in current_records}
        stored_paths = {record.path for record in stored_records}
        known_changed_paths = {change.old_path for change in known_path_changes}

        # Find unmatched files (potential renames)
        unmatched_stored = [r for r in stored_records
                           if r.path not in current_paths and r.path not in known_changed_paths]
        unmatched_current = [r for r in current_records
                            if r.path not in stored_paths]

        if not unmatched_stored or not unmatched_current:
            return []

        # Find similarity matches
        matches = self._find_similarity_matches(unmatched_stored, unmatched_current)

        # Convert high-confidence matches to path changes
        additional_path_changes = []
        for match in matches:
            if match.similarity_score >= self.composite_similarity_threshold:
                path_change = PathChange(
                    vault_id=self.vault_id,
                    old_note_id=self._path_to_note_id(match.old_file.path),
                    new_note_id=self._path_to_note_id(match.new_file.path),
                    old_path=match.old_file.path,
                    new_path=match.new_file.path,
                    old_checksum=match.old_file.sha256,
                    new_checksum=match.new_file.sha256,
                    change_type=self._determine_change_type(match.old_file.path, match.new_file.path),
                )
                additional_path_changes.append(path_change)

                logger.info(
                    f"Advanced similarity detected path change: {match.old_file.path} -> {match.new_file.path} "
                    f"(similarity: {match.similarity_score:.2f}, type: {match.match_type})"
                )

        return additional_path_changes

    def _find_similarity_matches(
        self,
        stored_records: List[FileRecord],
        current_records: List[FileRecord]
    ) -> List[SimilarityMatch]:
        """Find similarity matches between stored and current records."""
        matches = []

        # Filter to markdown files for content analysis
        stored_md = [r for r in stored_records if r.path.suffix.lower() in {'.md', '.markdown'}]
        current_md = [r for r in current_records if r.path.suffix.lower() in {'.md', '.markdown'}]

        for stored_record in stored_md[:self.max_similarity_candidates]:
            best_match = None
            best_score = 0.0

            for current_record in current_md[:self.max_similarity_candidates]:
                score = self._calculate_composite_similarity(stored_record, current_record)

                if score > best_score and score >= self.composite_similarity_threshold:
                    best_score = score
                    best_match = current_record

            if best_match:
                match = SimilarityMatch(
                    old_file=stored_record,
                    new_file=best_match,
                    similarity_score=best_score,
                    match_type='composite',
                    metadata={'algorithm': 'advanced_composite'}
                )
                matches.append(match)

        return matches

    def _calculate_composite_similarity(self, record1: FileRecord, record2: FileRecord) -> float:
        """Calculate composite similarity score between two file records."""
        cache_key = (record1.sha256, record2.sha256)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # If checksums are identical, perfect match
        if record1.sha256 == record2.sha256:
            score = 1.0
        else:
            # Calculate individual similarity components
            name_sim = self._calculate_name_similarity(record1.path, record2.path)
            content_sim = self._calculate_content_similarity(record1, record2)

            # Weighted composite score
            score = (0.4 * name_sim + 0.6 * content_sim)

        # Cache the result
        if len(self._similarity_cache) < self._cache_max_size:
            self._similarity_cache[cache_key] = score

        return score

    def _calculate_name_similarity(self, path1: Path, path2: Path) -> float:
        """Calculate similarity between file names/paths."""
        # Simple implementation using Levenshtein-style similarity
        name1 = path1.stem.lower()
        name2 = path2.stem.lower()

        if name1 == name2:
            return 1.0

        # Calculate edit distance ratio
        max_len = max(len(name1), len(name2))
        if max_len == 0:
            return 1.0

        # Simple character-based similarity
        common_chars = len(set(name1) & set(name2))
        total_chars = len(set(name1) | set(name2))

        return common_chars / total_chars if total_chars > 0 else 0.0

    def _calculate_content_similarity(self, record1: FileRecord, record2: FileRecord) -> float:
        """Calculate content similarity between two file records."""
        try:
            content1 = self._read_file_content(record1.path)
            content2 = self._read_file_content(record2.path)

            if content1 is None or content2 is None:
                return 0.0

            # Parse content (ignore frontmatter for content similarity)
            _, body1 = self._parse_markdown_content(content1)
            _, body2 = self._parse_markdown_content(content2)

            # Simple word-based similarity
            words1 = set(body1.lower().split())
            words2 = set(body2.lower().split())

            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.debug(f"Failed to calculate content similarity: {e}")
            return 0.0

    def _parse_markdown_content(self, content: str) -> Tuple[str, str]:
        """Parse markdown content into frontmatter and body."""
        if not self.enable_metadata_analysis:
            return "", content

        match = self._frontmatter_pattern.match(content)
        if match:
            frontmatter = match.group(1)
            body = content[match.end():]
            return frontmatter, body
        else:
            return "", content

    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content safely."""
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.debug(f"Failed to read file {file_path}: {e}")
            return None

    def _get_stored_content(self, record: FileRecord) -> Optional[str]:
        """Get stored content for a file record."""
        # Check cache first
        cache_key = record.sha256
        if cache_key in self._content_cache:
            stored_content, _ = self._content_cache[cache_key]
            return stored_content

        # Try to read current file (might be outdated but better than nothing)
        return self._read_file_content(record.path)

    def _calculate_checksum(self, content: str) -> str:
        """Calculate SHA256 checksum of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _path_to_note_id(self, path: Path) -> str:
        """Convert file path to note ID."""
        try:
            relative_path = path.relative_to(self.vault_root)
            return str(relative_path.with_suffix(''))
        except ValueError:
            return str(path.with_suffix(''))

    def _determine_change_type(self, old_path: Path, new_path: Path) -> str:
        """Determine the type of path change."""
        old_parent = old_path.parent
        new_parent = new_path.parent
        old_name = old_path.name
        new_name = new_path.name

        if old_parent != new_parent and old_name != new_name:
            return "rename_and_move"
        elif old_parent != new_parent:
            return "move"
        elif old_name != new_name:
            return "rename"
        else:
            return "unknown"

    def clear_caches(self) -> None:
        """Clear internal caches (useful for testing or memory management)."""
        self._content_cache.clear()
        self._similarity_cache.clear()
        logger.debug("Cleared change detector caches")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        return {
            "content_cache_size": len(self._content_cache),
            "similarity_cache_size": len(self._similarity_cache),
            "content_cache_max": self._cache_max_size,
        }


def create_change_detector(
    vault_id: str,
    vault_root: Path,
    state_store: StateStore,
    path_tracker: Optional[ObsidianPathTracker] = None,
    **config
) -> AdvancedChangeDetector:
    """Factory function to create a change detector with sensible defaults."""
    return AdvancedChangeDetector(
        vault_id=vault_id,
        vault_root=vault_root,
        state_store=state_store,
        path_tracker=path_tracker,
        **config
    )