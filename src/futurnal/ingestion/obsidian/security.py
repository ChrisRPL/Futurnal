"""Security validation for Obsidian markdown normalizer.

This module provides security utilities to validate and sanitize markdown content
during normalization to prevent path traversal attacks and other security issues.
"""

import os
from pathlib import Path
from typing import Optional


class SecurityError(Exception):
    """Raised when a security validation fails."""
    pass


class PathTraversalValidator:
    """Validates paths to prevent traversal attacks."""
    
    def __init__(self, vault_root: Optional[Path] = None):
        self.vault_root = vault_root
    
    def validate_link_path(self, link_target: str, current_file_path: Optional[Path] = None) -> bool:
        """Validate that a link target doesn't attempt path traversal.
        
        Args:
            link_target: The target path from a wikilink
            current_file_path: Current file path for relative resolution
            
        Returns:
            True if path is safe, False otherwise
            
        Raises:
            SecurityError: If path contains dangerous patterns
        """
        if not link_target or not isinstance(link_target, str):
            return True
        
        # Check for obvious path traversal attempts
        if '..' in link_target:
            raise SecurityError(f"Path traversal detected in link: {link_target}")
        
        if link_target.startswith('/'):
            raise SecurityError(f"Absolute path not allowed in link: {link_target}")
        
        # Check for null bytes
        if '\x00' in link_target:
            raise SecurityError(f"Null byte detected in link: {link_target}")
        
        # Validate resolved path stays within vault if vault_root is set
        if self.vault_root and current_file_path:
            try:
                resolved = self._resolve_safe_path(link_target, current_file_path)
                if not self._is_within_vault(resolved):
                    raise SecurityError(f"Link resolves outside vault: {link_target}")
            except (OSError, ValueError) as e:
                raise SecurityError(f"Invalid path in link: {link_target}") from e
        
        return True
    
    def _resolve_safe_path(self, target: str, base_path: Path) -> Path:
        """Safely resolve a target path relative to base path."""
        # Use the parent directory of the base file, not the file itself
        base_dir = base_path.parent if base_path.is_file() else base_path
        
        # Create potential path (not resolved yet to avoid filesystem access)
        target_path = base_dir / target
        
        # Normalize the path without accessing filesystem
        parts = []
        for part in target_path.parts:
            if part == '..':
                if parts:
                    parts.pop()
            elif part != '.' and part:
                parts.append(part)
        
        if not parts:
            raise ValueError("Empty path after normalization")
        
        return Path(*parts)
    
    def _is_within_vault(self, path: Path) -> bool:
        """Check if path is within the vault root."""
        if not self.vault_root:
            return True
        
        try:
            vault_root_resolved = self.vault_root.resolve()
            path_resolved = path.resolve()
            return str(path_resolved).startswith(str(vault_root_resolved))
        except (OSError, ValueError):
            return False


class ResourceLimiter:
    """Enforces resource limits during normalization."""
    
    def __init__(
        self,
        max_content_size: int = 50 * 1024 * 1024,  # 50MB
        max_frontmatter_size: int = 1024 * 1024,   # 1MB
        max_links: int = 10000,
        max_assets: int = 1000,
        max_tags: int = 1000,
        max_callouts: int = 1000,
        max_headings: int = 1000
    ):
        self.max_content_size = max_content_size
        self.max_frontmatter_size = max_frontmatter_size
        self.max_links = max_links
        self.max_assets = max_assets
        self.max_tags = max_tags
        self.max_callouts = max_callouts
        self.max_headings = max_headings
    
    def validate_content_size(self, content: str) -> None:
        """Validate content size is within limits."""
        size = len(content.encode('utf-8'))
        if size > self.max_content_size:
            raise SecurityError(f"Content size {size} exceeds limit {self.max_content_size}")
    
    def validate_frontmatter_size(self, frontmatter: str) -> None:
        """Validate frontmatter size is within limits."""
        size = len(frontmatter.encode('utf-8'))
        if size > self.max_frontmatter_size:
            raise SecurityError(f"Frontmatter size {size} exceeds limit {self.max_frontmatter_size}")
    
    def validate_element_count(self, count: int, element_type: str) -> None:
        """Validate element count is within limits."""
        limits = {
            'links': self.max_links,
            'assets': self.max_assets,
            'tags': self.max_tags,
            'callouts': self.max_callouts,
            'headings': self.max_headings,
        }
        
        limit = limits.get(element_type)
        if limit and count > limit:
            raise SecurityError(f"{element_type} count {count} exceeds limit {limit}")


def validate_yaml_safety(yaml_content: str) -> None:
    """Validate YAML content for safety.
    
    Args:
        yaml_content: Raw YAML content to validate
        
    Raises:
        SecurityError: If YAML contains unsafe constructs
    """
    # Check for Python object instantiation
    dangerous_patterns = [
        '!!python/',
        '!!python/object',
        '!!python/name',
        '!!python/module',
        '!!python/object/apply',
        '!!python/object/new',
    ]
    
    content_lower = yaml_content.lower()
    for pattern in dangerous_patterns:
        if pattern in content_lower:
            raise SecurityError(f"Dangerous YAML construct detected: {pattern}")
    
    # Check for excessive complexity (YAML bombs)
    if yaml_content.count('&') > 100:  # Excessive anchors
        raise SecurityError("Excessive YAML anchors detected")
    
    if yaml_content.count('*') > 1000:  # Excessive references
        raise SecurityError("Excessive YAML references detected")



