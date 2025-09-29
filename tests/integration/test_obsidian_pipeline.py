"""Integration tests for the complete Obsidian processing pipeline."""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from futurnal.ingestion.obsidian import (
    ObsidianDocumentProcessor,
    ObsidianVaultConnector,
    ObsidianVaultSource,
    ObsidianVaultDescriptor,
    VaultRegistry
)
from futurnal.ingestion.local.state import StateStore, FileRecord, compute_sha256
from futurnal.pipeline.triples import MetadataTripleExtractor, TripleEnrichedNormalizationSink


@pytest.fixture(autouse=True)
def mock_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the Unstructured.io partition function for testing."""
    
    def _fake_partition(*, filename: str, strategy: str, include_metadata: bool, content_type: str = None):
        path = Path(filename)
        content = path.read_text()
        
        # Create fake elements based on content
        elements = []
        
        # Split content into paragraphs for realistic element simulation
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.startswith('#'):
                # Header element
                elements.append({
                    "text": paragraph,
                    "type": "Title",
                    "metadata": {
                        "filename": filename,
                        "element_id": f"header-{i}",
                        "languages": ["en"],
                    }
                })
            elif paragraph.startswith('---'):
                # Skip frontmatter blocks
                continue
            else:
                # Regular text element
                elements.append({
                    "text": paragraph,
                    "type": "NarrativeText", 
                    "metadata": {
                        "filename": filename,
                        "element_id": f"text-{i}",
                        "languages": ["en"],
                    }
                })
        
        return elements
    
    monkeypatch.setattr("futurnal.ingestion.obsidian.processor.partition", _fake_partition)


class MockPKGWriter:
    """Mock PKG writer for testing."""
    
    def __init__(self):
        self.documents = []
        
    def write_document(self, payload: dict) -> None:
        self.documents.append(payload)
        
    def remove_document(self, sha256: str) -> None:
        self.documents = [doc for doc in self.documents if doc.get("sha256") != sha256]


class MockVectorWriter:
    """Mock vector writer for testing."""
    
    def __init__(self):
        self.embeddings = []
        
    def write_embedding(self, payload: dict) -> None:
        self.embeddings.append(payload)
        
    def remove_embedding(self, sha256: str) -> None:
        self.embeddings = [emb for emb in self.embeddings if emb.get("sha256") != sha256]


class TestObsidianDocumentProcessor:
    """Test the document processor that bridges normalizer and Unstructured.io."""
    
    def test_process_document_integration(self):
        """Test end-to-end document processing."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            vault_path = Path(temp_dir) / "vault"
            vault_path.mkdir()
            
            # Create test document
            test_doc = vault_path / "test.md"
            test_content = """---
title: "Test Document"
tags: [test, integration]
author: "Test Author"
---

# Test Document

This has a [[link]] and #tag and a task:

- [x] Completed task
- [ ] Pending task

> [!note] Test Callout
> This is a callout block.
"""
            test_doc.write_text(test_content)
            
            # Create file record
            file_stats = test_doc.stat()
            file_record = FileRecord(
                path=test_doc,
                size=file_stats.st_size,
                mtime=file_stats.st_mtime,
                sha256=compute_sha256(test_doc)
            )
            
            # Initialize processor
            processor = ObsidianDocumentProcessor(
                workspace_dir=workspace,
                vault_root=vault_path,
                vault_id="test-vault"
            )
            
            # Process document
            elements = list(processor.process_document(file_record, "test_source"))
            
            # Verify we got elements
            assert len(elements) > 0
            
            # Check element structure
            element = elements[0]
            assert "source" in element
            assert "path" in element
            assert "sha256" in element
            assert "element_path" in element
            assert element["source"] == "test_source"
            assert str(test_doc) in element["path"]
            
            # Load and verify element data
            element_path = Path(element["element_path"])
            assert element_path.exists()
            
            with open(element_path, 'r') as f:
                element_data = json.load(f)
            
            # Verify enriched metadata
            assert "metadata" in element_data
            metadata = element_data["metadata"]
            
            assert "futurnal" in metadata
            futurnal_metadata = metadata["futurnal"]
            assert "normalizer_version" in futurnal_metadata
            assert "content_checksum" in futurnal_metadata
            assert "vault_id" in futurnal_metadata
            assert futurnal_metadata["vault_id"] == "test-vault"
            
            # Verify frontmatter preservation
            assert "frontmatter" in metadata
            frontmatter = metadata["frontmatter"]
            assert frontmatter["title"] == "Test Document"
            assert frontmatter["author"] == "Test Author"
            
            # Verify Obsidian elements
            assert "obsidian_tags" in metadata
            tags = metadata["obsidian_tags"]
            tag_names = [tag["name"] for tag in tags]
            assert "test" in tag_names
            assert "integration" in tag_names
            
            assert "obsidian_links" in metadata
            links = metadata["obsidian_links"]
            assert len(links) > 0
            assert links[0]["target"] == "link"


class TestTripleExtraction:
    """Test semantic triple extraction from processed elements."""
    
    def test_metadata_triple_extraction(self):
        """Test triple extraction from element metadata."""
        
        # Create mock element with rich metadata
        element_data = {
            "text": "Test content",
            "metadata": {
                "source": "test_source",
                "path": "/test/document.md",
                "sha256": "abc123",
                "frontmatter": {
                    "title": "Test Document",
                    "author": "Test Author",
                    "tags": ["test", "demo"],
                    "created": "2023-01-01"
                },
                "obsidian_tags": [
                    {"name": "integration", "is_nested": False},
                    {"name": "work/project", "is_nested": True}
                ],
                "obsidian_links": [
                    {
                        "target": "Other Document",
                        "display_text": None,
                        "is_embed": False,
                        "section": None,
                        "block_id": None,
                        "is_broken": False
                    }
                ]
            }
        }
        
        # Extract triples
        extractor = MetadataTripleExtractor()
        triples = extractor.extract_triples(element_data)
        
        # Verify we got triples
        assert len(triples) > 0
        
        # Check specific triples by organizing them properly
        subjects_predicates = {}
        for triple in triples:
            key = (triple.subject, triple.predicate)
            if key not in subjects_predicates:
                subjects_predicates[key] = []
            subjects_predicates[key].append(triple.object)
        
        # Document type
        doc_uri = "futurnal:doc//test/document.md"
        assert "futurnal:Document" in subjects_predicates.get((doc_uri, "rdf:type"), [])
        
        # Title
        assert "Test Document" in subjects_predicates.get((doc_uri, "dc:title"), [])
        
        # Author
        author_uri = "futurnal:person/Test_Author"
        assert author_uri in subjects_predicates.get((doc_uri, "dc:creator"), [])
        assert "futurnal:Person" in subjects_predicates.get((author_uri, "rdf:type"), [])
        
        # Tags - check that both tags are present
        tag_objects = subjects_predicates.get((doc_uri, "futurnal:hasTag"), [])
        assert "futurnal:tag/integration" in tag_objects
        assert "futurnal:tag/work_project" in tag_objects
        
        # Links - spaces are replaced with underscores in URI creation
        target_uri = "futurnal:doc/Other_Document"
        assert target_uri in subjects_predicates.get((doc_uri, "futurnal:linksTo"), [])


class TestFullPipelineIntegration:
    """Test the complete pipeline integration."""
    
    def test_end_to_end_obsidian_processing(self):
        """Test complete pipeline from Obsidian vault to PKG storage."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            vault_path = Path(temp_dir) / "vault"
            vault_path.mkdir()
            
            # Create comprehensive test document
            test_doc = vault_path / "comprehensive.md"
            test_content = """---
title: "Comprehensive Test"
author: "Integration Tester"  
tags: [comprehensive, test, obsidian]
category: "documentation"
status: "draft"
created: 2023-01-01
priority: high
---

# Comprehensive Test Document

This document tests the complete [[Obsidian]] processing pipeline.

## Links and References

- Internal link: [[Project Overview]]
- Section link: [[Meeting Notes#Action Items]]
- Broken link: [[Non-existent Document]]
- Embed: ![[architecture-diagram.png]]

## Tags and Categories

This document is tagged with #comprehensive, #integration, and #testing tags.

## Callouts

> [!note] Processing Pipeline
> The document flows through: MarkdownNormalizer → Unstructured.io → Triple Extraction

> [!warning]+ Performance Note
> Large documents are processed in chunks for memory efficiency.

## Tasks

- [x] Implement normalizer
- [x] Add Unstructured.io integration
- [x] Create triple extraction
- [ ] Add advanced NLP processing
- [ ] Implement causal insight detection

## Data Table

| Component | Status | Coverage |
|-----------|--------|----------|
| Normalizer | Complete | 95% |
| Processor | Complete | 90% |
| Extractor | Complete | 85% |

This completes our integration test[^1].

[^1]: Full end-to-end processing verification.
"""
            test_doc.write_text(test_content)
            
            # Set up pipeline components
            state_store = StateStore(workspace / "state.db")
            vault_registry = VaultRegistry()
            
            # Create .obsidian directory to make it a valid vault
            obsidian_dir = vault_path / ".obsidian"
            obsidian_dir.mkdir()
            
            vault_descriptor = ObsidianVaultDescriptor.from_path(
                base_path=vault_path,
                name="test_vault"
            )
            vault_registry.add_or_update(vault_descriptor)
            
            # Create mock storage backends
            pkg_writer = MockPKGWriter()
            vector_writer = MockVectorWriter()
            
            # Set up enriched sink
            element_sink = TripleEnrichedNormalizationSink(
                pkg_writer=pkg_writer,
                vector_writer=vector_writer
            )
            
            # Create connector
            connector = ObsidianVaultConnector(
                workspace_dir=workspace,
                state_store=state_store,
                vault_registry=vault_registry,
                element_sink=element_sink
            )
            
            # Create source
            source = ObsidianVaultSource.from_vault_descriptor(
                vault_descriptor,
                name="integration_test"
            )
            
            # Process through pipeline
            elements = list(connector.ingest(source))
            
            # Verify processing results
            assert len(elements) > 0
            
            # Verify PKG storage
            assert len(pkg_writer.documents) > 0
            pkg_doc = pkg_writer.documents[0]
            
            # Check document structure
            assert "semantic_triples" in pkg_doc
            triples = pkg_doc["semantic_triples"]
            assert len(triples) > 0
            
            # Verify triple types
            predicates = {triple["predicate"] for triple in triples}
            assert "dc:title" in predicates
            assert "dc:creator" in predicates
            assert "futurnal:hasTag" in predicates
            assert "futurnal:linksTo" in predicates
            
            # Verify vector storage
            assert len(vector_writer.embeddings) > 0
            
            # Verify metadata preservation
            metadata = pkg_doc["metadata"]
            assert "frontmatter" in metadata
            assert metadata["frontmatter"]["title"] == "Comprehensive Test"
            assert metadata["frontmatter"]["status"] == "draft"
            
            print(f"✅ End-to-end test passed:")
            print(f"   📄 Processed {len(elements)} elements")
            print(f"   🔗 Extracted {len(triples)} semantic triples")
            print(f"   💾 Stored to PKG and vector database")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
