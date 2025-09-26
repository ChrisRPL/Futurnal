"""Obsidian-specific ingestion utilities."""

from .descriptor import (
    ObsidianVaultDescriptor,
    VaultRegistry,
    DEFAULT_OBSIDIAN_IGNORE_RULES,
)

__all__ = [
    "ObsidianVaultDescriptor",
    "VaultRegistry",
    "DEFAULT_OBSIDIAN_IGNORE_RULES",
]


