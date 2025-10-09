"""Unit tests for FormatAdapterRegistry.

Tests registry lifecycle including registration, retrieval, fallback behavior,
and default adapter setup.
"""

from __future__ import annotations

import pytest

from futurnal.pipeline.models import DocumentFormat
from futurnal.pipeline.normalization.adapters import (
    CodeAdapter,
    EmailAdapter,
    GenericAdapter,
    HTMLAdapter,
    MarkdownAdapter,
    PDFAdapter,
    TextAdapter,
)
from futurnal.pipeline.normalization.registry import (
    AdapterNotFoundError,
    FormatAdapterRegistry,
)


# ============================================================================
# Basic Registration Tests
# ============================================================================


def test_registry_register_adapter():
    """Test basic adapter registration."""
    registry = FormatAdapterRegistry()
    adapter = MarkdownAdapter()

    registry.register(adapter)

    assert DocumentFormat.MARKDOWN in registry.list_supported_formats()
    retrieved = registry.get_adapter(DocumentFormat.MARKDOWN)
    assert retrieved.name == adapter.name


def test_registry_register_duplicate_raises_error():
    """Test registering duplicate adapter raises ValueError."""
    registry = FormatAdapterRegistry()
    adapter1 = MarkdownAdapter()
    adapter2 = MarkdownAdapter()

    registry.register(adapter1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(adapter2)


def test_registry_register_multiple_formats():
    """Test adapter supporting multiple formats."""
    registry = FormatAdapterRegistry()
    adapter = PDFAdapter()  # Supports PDF, DOCX, PPTX

    registry.register(adapter)

    # All formats should be registered
    assert DocumentFormat.PDF in registry.list_supported_formats()
    assert DocumentFormat.DOCX in registry.list_supported_formats()
    assert DocumentFormat.PPTX in registry.list_supported_formats()

    # All should return same adapter
    pdf_adapter = registry.get_adapter(DocumentFormat.PDF)
    docx_adapter = registry.get_adapter(DocumentFormat.DOCX)
    assert pdf_adapter.name == docx_adapter.name


# ============================================================================
# Retrieval Tests
# ============================================================================


def test_registry_get_adapter_success():
    """Test successful adapter retrieval."""
    registry = FormatAdapterRegistry()
    adapter = MarkdownAdapter()
    registry.register(adapter)

    retrieved = registry.get_adapter(DocumentFormat.MARKDOWN)

    assert retrieved is not None
    assert retrieved.name == "MarkdownAdapter"


def test_registry_get_adapter_not_found():
    """Test get_adapter raises error when no adapter found."""
    registry = FormatAdapterRegistry()

    with pytest.raises(AdapterNotFoundError, match="No adapter registered"):
        registry.get_adapter(DocumentFormat.MARKDOWN)


def test_registry_has_adapter():
    """Test has_adapter method."""
    registry = FormatAdapterRegistry()
    adapter = MarkdownAdapter()

    assert registry.has_adapter(DocumentFormat.MARKDOWN) is False

    registry.register(adapter)

    assert registry.has_adapter(DocumentFormat.MARKDOWN) is True


# ============================================================================
# Fallback Adapter Tests
# ============================================================================


def test_registry_fallback_adapter():
    """Test fallback adapter registration and retrieval."""
    registry = FormatAdapterRegistry()
    fallback = GenericAdapter()

    registry.register_fallback(fallback)

    # Should use fallback for unknown format
    retrieved = registry.get_adapter(DocumentFormat.UNKNOWN)
    assert retrieved.name == "GenericAdapter"


def test_registry_fallback_over_missing_specific():
    """Test fallback is used when specific adapter missing."""
    registry = FormatAdapterRegistry()
    markdown_adapter = MarkdownAdapter()
    fallback_adapter = GenericAdapter()

    registry.register(markdown_adapter)
    registry.register_fallback(fallback_adapter)

    # Registered format returns specific adapter
    assert registry.get_adapter(DocumentFormat.MARKDOWN).name == "MarkdownAdapter"

    # Unregistered format returns fallback
    assert registry.get_adapter(DocumentFormat.HTML).name == "GenericAdapter"


def test_registry_has_adapter_with_fallback():
    """Test has_adapter returns True when fallback is available."""
    registry = FormatAdapterRegistry()
    fallback = GenericAdapter()

    assert registry.has_adapter(DocumentFormat.UNKNOWN) is False

    registry.register_fallback(fallback)

    # Should return True for any format when fallback is set
    assert registry.has_adapter(DocumentFormat.UNKNOWN) is True
    assert registry.has_adapter(DocumentFormat.MARKDOWN) is True


# ============================================================================
# Registry Management Tests
# ============================================================================


def test_registry_unregister():
    """Test unregistering an adapter."""
    registry = FormatAdapterRegistry()
    adapter = MarkdownAdapter()
    registry.register(adapter)

    assert registry.has_adapter(DocumentFormat.MARKDOWN) is True

    registry.unregister(DocumentFormat.MARKDOWN)

    with pytest.raises(AdapterNotFoundError):
        registry.get_adapter(DocumentFormat.MARKDOWN)


def test_registry_unregister_not_found():
    """Test unregister raises error for non-existent format."""
    registry = FormatAdapterRegistry()

    with pytest.raises(AdapterNotFoundError):
        registry.unregister(DocumentFormat.MARKDOWN)


def test_registry_clear():
    """Test clearing all adapters."""
    registry = FormatAdapterRegistry()
    registry.register(MarkdownAdapter())
    registry.register(PDFAdapter())
    registry.register_fallback(GenericAdapter())

    assert len(registry.list_supported_formats()) > 0

    registry.clear()

    assert len(registry.list_supported_formats()) == 0
    with pytest.raises(AdapterNotFoundError):
        registry.get_adapter(DocumentFormat.MARKDOWN)


def test_registry_list_supported_formats():
    """Test listing supported formats."""
    registry = FormatAdapterRegistry()

    assert len(registry.list_supported_formats()) == 0

    registry.register(MarkdownAdapter())
    registry.register(EmailAdapter())
    registry.register(PDFAdapter())  # Registers PDF, DOCX, PPTX

    formats = registry.list_supported_formats()

    assert DocumentFormat.MARKDOWN in formats
    assert DocumentFormat.EMAIL in formats
    assert DocumentFormat.PDF in formats
    assert DocumentFormat.DOCX in formats
    assert DocumentFormat.PPTX in formats


# ============================================================================
# Adapter Info Tests
# ============================================================================


def test_registry_get_adapter_info():
    """Test get_adapter_info returns correct information."""
    registry = FormatAdapterRegistry()
    adapter = MarkdownAdapter()
    registry.register(adapter)

    info = registry.get_adapter_info(DocumentFormat.MARKDOWN)

    assert info["name"] == "MarkdownAdapter"
    assert "markdown" in info["supported_formats"]
    assert info["requires_unstructured_processing"] is False


def test_registry_get_adapter_info_not_found():
    """Test get_adapter_info raises error for non-existent format."""
    registry = FormatAdapterRegistry()

    with pytest.raises(AdapterNotFoundError):
        registry.get_adapter_info(DocumentFormat.MARKDOWN)


# ============================================================================
# Default Adapters Tests
# ============================================================================


def test_registry_register_default_adapters():
    """Test registering all default adapters."""
    registry = FormatAdapterRegistry()

    registry.register_default_adapters()

    # Check all expected formats are registered
    formats = registry.list_supported_formats()

    assert DocumentFormat.MARKDOWN in formats
    assert DocumentFormat.EMAIL in formats
    assert DocumentFormat.HTML in formats
    assert DocumentFormat.CODE in formats
    assert DocumentFormat.PDF in formats
    assert DocumentFormat.DOCX in formats
    assert DocumentFormat.PPTX in formats

    # Check fallback is registered
    assert registry.has_adapter(DocumentFormat.UNKNOWN) is True


def test_registry_default_adapters_correct_types():
    """Test default adapters are of correct type."""
    registry = FormatAdapterRegistry()
    registry.register_default_adapters()

    # Check adapter names
    assert registry.get_adapter(DocumentFormat.MARKDOWN).name == "MarkdownAdapter"
    assert registry.get_adapter(DocumentFormat.EMAIL).name == "EmailAdapter"
    assert registry.get_adapter(DocumentFormat.HTML).name == "HTMLAdapter"
    assert registry.get_adapter(DocumentFormat.CODE).name == "CodeAdapter"
    assert registry.get_adapter(DocumentFormat.PDF).name == "PDFAdapter"


def test_registry_default_adapters_fallback():
    """Test default adapters include fallback."""
    registry = FormatAdapterRegistry()
    registry.register_default_adapters()

    # Unknown format should use fallback
    adapter = registry.get_adapter(DocumentFormat.UNKNOWN)
    assert adapter.name == "GenericAdapter"


# ============================================================================
# Multiple Registry Tests
# ============================================================================


def test_multiple_registries_independent():
    """Test multiple registry instances are independent."""
    registry1 = FormatAdapterRegistry()
    registry2 = FormatAdapterRegistry()

    registry1.register(MarkdownAdapter())

    assert registry1.has_adapter(DocumentFormat.MARKDOWN) is True
    assert registry2.has_adapter(DocumentFormat.MARKDOWN) is False


# ============================================================================
# Edge Cases Tests
# ============================================================================


def test_registry_empty_initialization():
    """Test registry starts empty."""
    registry = FormatAdapterRegistry()

    assert len(registry.list_supported_formats()) == 0
    assert registry.has_adapter(DocumentFormat.MARKDOWN) is False


def test_registry_reregister_after_clear():
    """Test can re-register adapters after clear."""
    registry = FormatAdapterRegistry()
    adapter = MarkdownAdapter()

    registry.register(adapter)
    registry.clear()
    registry.register(adapter)

    assert registry.has_adapter(DocumentFormat.MARKDOWN) is True


def test_registry_adapter_count():
    """Test adapter count after various operations."""
    registry = FormatAdapterRegistry()

    # Empty registry
    assert len(registry.list_supported_formats()) == 0

    # Add single-format adapter
    registry.register(MarkdownAdapter())
    assert len(registry.list_supported_formats()) == 1

    # Add multi-format adapter (PDF covers 3 formats)
    registry.register(PDFAdapter())
    assert len(registry.list_supported_formats()) == 4  # MARKDOWN + PDF + DOCX + PPTX

    # Unregister one
    registry.unregister(DocumentFormat.MARKDOWN)
    assert len(registry.list_supported_formats()) == 3


# ============================================================================
# Integration with Adapters Tests
# ============================================================================


def test_registry_with_all_builtin_adapters():
    """Test registry works with all built-in adapter types."""
    registry = FormatAdapterRegistry()

    # Register one of each type (excluding overlapping adapters)
    # Note: TextAdapter, CodeAdapter, and GenericAdapter all support CODE format
    # So we only register CodeAdapter for CODE format
    adapters = [
        MarkdownAdapter(),
        EmailAdapter(),
        HTMLAdapter(),
        CodeAdapter(),
        PDFAdapter(),
        # Skip TextAdapter and GenericAdapter to avoid CODE format conflict
    ]

    for adapter in adapters:
        # Should not raise
        registry.register(adapter)

    # All should be retrievable
    for adapter in adapters:
        for fmt in adapter.supported_formats:
            retrieved = registry.get_adapter(fmt)
            assert retrieved.name == adapter.name


def test_registry_adapter_properties_preserved():
    """Test adapter properties are preserved through registration."""
    registry = FormatAdapterRegistry()
    adapter = HTMLAdapter()

    registry.register(adapter)
    retrieved = registry.get_adapter(DocumentFormat.HTML)

    # Check properties are preserved
    assert retrieved.requires_unstructured_processing == adapter.requires_unstructured_processing
    assert retrieved.name == adapter.name
    assert retrieved.supported_formats == adapter.supported_formats
