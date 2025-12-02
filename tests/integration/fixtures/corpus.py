"""Test corpus loaders and fixtures for integration testing.

Provides standardized test document loading, corpus generation, and ground truth management.
"""

import json
import random
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from futurnal.pipeline.models import NormalizedDocument, NormalizedMetadata


class CorpusLoader:
    """Load and manage test corpora for integration testing."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            # Default to local data directory relative to this file
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = data_dir

    def load_test_document(self, filename: str) -> NormalizedDocument:
        """Load a single test document by filename."""
        doc_path = self.data_dir / "sample_documents" / filename
        if not doc_path.exists():
            # Fallback to creating a synthetic one if file doesn't exist
            return self._create_synthetic_document(filename)
        
        with open(doc_path, "r") as f:
            content = f.read()
            
        return NormalizedDocument(
            document_id=str(uuid.uuid4()),
            sha256=hashlib.sha256(content.encode()).hexdigest(),
            content=content,
            metadata=NormalizedMetadata(
                source_path=filename,
                source_id=str(uuid.uuid4()),
                source_type="test_corpus",
                format="markdown",
                content_type="text/markdown",
                created_at=datetime.now(),
                character_count=len(content),
                word_count=len(content.split()),
                line_count=len(content.splitlines()),
                content_hash=hashlib.sha256(content.encode()).hexdigest()
            )
        )

    def load_test_corpus(self, num_docs: int) -> List[NormalizedDocument]:
        """Load a corpus of specified size."""
        docs = []
        for i in range(num_docs):
            docs.append(self._create_synthetic_document(f"doc_{i}.md"))
        return docs

    def load_temporally_labeled_corpus(self, num_docs: int) -> List[Dict[str, Any]]:
        """Load corpus with temporal ground truth."""
        # For Phase 1, we'll generate synthetic labeled data
        labeled_docs = []
        for i in range(num_docs):
            doc = self._create_synthetic_document(f"temporal_doc_{i}.md")
            labels = self._generate_temporal_labels(doc)
            labeled_docs.append({"document": doc, "labels": labels})
        return labeled_docs

    def load_diverse_corpus(self, num_docs: int) -> List[NormalizedDocument]:
        """Load diverse documents for schema evolution testing."""
        # Generate docs with varied entity types
        docs = []
        topics = ["technology", "history", "science", "business"]
        for i in range(num_docs):
            topic = topics[i % len(topics)]
            docs.append(self._create_synthetic_document(f"diverse_{topic}_{i}.md", topic=topic))
        return docs

    def create_malformed_document(self) -> Any:
        """Generate a malformed document for error testing."""
        # Return something that isn't a NormalizedDocument or has invalid fields
        return NormalizedDocument.model_construct(
            document_id="malformed_id",
            content="",  # Empty content
            metadata=None,  # Missing metadata
            normalization_errors=["Simulated malformed document"]
        )

    def create_ambiguous_document(self) -> NormalizedDocument:
        """Generate a document with ambiguous content."""
        content = "Maybe something happened, or maybe it didn't. It's unclear."
        return NormalizedDocument(
            document_id=str(uuid.uuid4()),
            sha256=hashlib.sha256(content.encode()).hexdigest(),
            content=content,
            metadata=NormalizedMetadata(
                source_path="ambiguous.md",
                source_id=str(uuid.uuid4()),
                source_type="test_ambiguous",
                format="markdown",
                content_type="text/markdown",
                created_at=datetime.now(),
                character_count=len(content),
                word_count=len(content.split()),
                line_count=len(content.splitlines()),
                content_hash=hashlib.sha256(content.encode()).hexdigest()
            )
        )

    def _create_synthetic_document(self, filename: str, topic: str = "general") -> NormalizedDocument:
        """Create a synthetic document with realistic content."""
        if topic == "technology":
            content = (
                f"On {datetime.now().strftime('%Y-%m-%d')}, the team deployed the new API. "
                "Performance improved by 20% after the update. "
                "The deployment caused a brief outage in the staging environment."
            )
        elif topic == "history":
            content = (
                "In 1995, the company was founded. "
                "Ten years later, they expanded to Europe. "
                "This expansion led to significant growth."
            )
        else:
            content = (
                f"This is a test document named {filename}. "
                "It contains some entities like Person A and Organization B. "
                "Event X happened before Event Y."
            )

        return NormalizedDocument(
            document_id=str(uuid.uuid4()),
            sha256=hashlib.sha256(content.encode()).hexdigest(),
            content=content,
            metadata=NormalizedMetadata(
                source_path=filename,
                source_id=str(uuid.uuid4()),
                source_type="synthetic_generator",
                format="markdown",
                content_type="text/markdown",
                created_at=datetime.now(),
                character_count=len(content),
                word_count=len(content.split()),
                line_count=len(content.splitlines()),
                content_hash=hashlib.sha256(content.encode()).hexdigest()
            )
        )

    def _generate_temporal_labels(self, doc: NormalizedDocument) -> Dict[str, Any]:
        """Generate synthetic ground truth for temporal labels."""
        # Simple heuristic labels for synthetic docs
        return {
            "has_explicit_date": "1995" in doc.content or "-" in doc.content,
            "has_relative_date": "later" in doc.content or "ago" in doc.content,
            "temporal_ordering_valid": True
        }
