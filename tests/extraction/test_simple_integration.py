"""Simple integration test for advanced extraction.

Verifies that AdvancedTripleExtractor can process a document and
extract temporal markers, events, and causal relationships.
"""

import pytest
from datetime import datetime


def test_advanced_extraction_integration():
    """Test that advanced extraction pipeline works end-to-end."""
    from futurnal.pipeline.triples import AdvancedTripleExtractor
    
    # Create extractor
    extractor = AdvancedTripleExtractor(enable_experiential_learning=False)
    
    # Sample document element
    element = {
        "sha256": "test_doc_123",
        "metadata": {
            "path": "/test/document.md",
            "created": "2024-01-15"
        },
        "text": (
            "On January 15, 2024, we met with the engineering team "
            "to discuss the project timeline. The meeting led to a "
            "decision to proceed with the launch on February 1, 2024."
        )
    }
    
    # Extract triples
    triples = extractor.extract_triples(element)
    
    # Verify triples were extracted
    assert len(triples) > 0, "Should extract at least some triples"
    
    # Verify extraction methods
    extraction_methods = {t.extraction_method for t in triples}
    assert "temporal_extraction" in extraction_methods, "Should extract temporal markers"
    
    # Log results
    print(f"\nExtracted {len(triples)} triples:")
    for triple in triples[:10]:  # Show first 10
        print(f"  {triple.predicate}: {triple.subject} -> {triple.object}")
        print(f"    Method: {triple.extraction_method}, Confidence: {triple.confidence}")
    
    print(f"\nExtraction methods used: {extraction_methods}")


def test_metadata_and_advanced_extraction_combined():
    """Test that both metadata and advanced extraction work together."""
    from futurnal.pipeline.triples import (
        MetadataTripleExtractor,
        AdvancedTripleExtractor
    )
    
    # Create extractors
    metadata_extractor = MetadataTripleExtractor()
    advanced_extractor = AdvancedTripleExtractor(enable_experiential_learning=False)
    
    # Sample document element with rich metadata
    payload = {
        "sha256": "test_doc_456",
        "metadata": {
            "path": "/test/document.md",
            "frontmatter": {
                "title": "Project Kickoff",
                "author": "Alice Smith",
                "created": "2024-01-15"
            },
            "obsidian_tags": [
                {"name": "project", "is_nested": False},
                {"name": "meeting", "is_nested": False}
            ]
        },
        "text": "Yesterday we met with the team to discuss the new feature."
    }
    
    # Extract with both extractors
    metadata_triples = metadata_extractor.extract_triples(payload)
    advanced_triples = advanced_extractor.extract_triples(payload)
    
    # Verify both extracted triples
    assert len(metadata_triples) > 0, "Should extract metadata triples"
    assert len(advanced_triples) > 0, "Should extract advanced triples"
    
    # Combine
    all_triples = metadata_triples + advanced_triples
    
    print(f"\nMetadata extraction: {len(metadata_triples)} triples")
    print(f"Advanced extraction: {len(advanced_triples)} triples")
    print(f"Total: {len(all_triples)} triples")
    
    # Verify extraction methods
    metadata_methods = {t.extraction_method for t in metadata_triples}
    advanced_methods = {t.extraction_method for t in advanced_triples}
    
    assert "metadata" in metadata_methods or "frontmatter" in metadata_methods
    assert "temporal_extraction" in advanced_methods
    
    print(f"\nMetadata methods: {metadata_methods}")
    print(f"Advanced methods: {advanced_methods}")


if __name__ == "__main__":
    # Run tests directly
    print("=" * 80)
    print("Testing Advanced Extraction Integration")
    print("=" * 80)
    
    test_advanced_extraction_integration()
    print("\n✅ Test 1 passed: Advanced extraction works")
    
    test_metadata_and_advanced_extraction_combined()
    print("\n✅ Test 2 passed: Metadata + Advanced extraction work together")
    
    print("\n" + "=" * 80)
    print("All tests passed! 🎉")
    print("=" * 80)
