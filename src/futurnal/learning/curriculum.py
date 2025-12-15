"""
Curriculum Generator for Experiential Learning

Implements SEAgent Curriculum Generator for progressive document ordering
from simple to complex, optimizing the learning progression.

Research Foundation:
- SEAgent (2508.04700v2): Curriculum Generator that generates increasingly
  diverse and challenging tasks for optimal learning

Quality Gates:
- Documents ordered by complexity: simple -> medium -> complex
- Learning value prioritization based on patterns

Option B Compliance:
- Curriculum guides document selection, NOT model parameters
- Complexity assessment done locally without cloud calls
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from futurnal.extraction.schema.evolution import Document

logger = logging.getLogger(__name__)


# Complexity weight defaults per Step 06 specification
DEFAULT_TEMPORAL_WEIGHT = 0.30  # Temporal expressions matter most for Phase 3
DEFAULT_ENTITY_WEIGHT = 0.25
DEFAULT_LENGTH_WEIGHT = 0.25
DEFAULT_REFERENCE_WEIGHT = 0.20

# Normalization targets
TARGET_DOCUMENT_LENGTH = 5000  # Characters
TARGET_ENTITY_DENSITY = 10.0  # Entities per 1000 tokens
TARGET_TEMPORAL_EXPRESSIONS = 10  # Temporal markers per document
TARGET_CROSS_REFERENCES = 5  # Cross-document references


class DocumentProtocol(Protocol):
    """Protocol for document objects."""

    content: str
    doc_id: str


@dataclass
class DocumentComplexity:
    """Complexity assessment factors per SEAgent curriculum.

    Captures multiple complexity dimensions for curriculum ordering:
    - Document length (longer = more complex)
    - Entity density (more entities = more complex)
    - Temporal expression count (more temporal = more complex, critical for Phase 3)
    - Cross-reference count (wikilinks, citations = more complex)

    Research Reference: SEAgent Section 3 - "Curriculum Generator that
    generates increasingly diverse and challenging tasks"
    """

    document_id: str
    document_length: int = 0
    entity_density: float = 0.0  # Entities per 1000 tokens
    temporal_expression_count: int = 0
    cross_reference_count: int = 0  # Wikilinks, cross-doc refs
    estimated_token_count: int = 0

    # Computed complexity score (set by generator)
    complexity_score: float = 0.0

    # Metadata
    assessed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "document_id": self.document_id,
            "document_length": self.document_length,
            "entity_density": self.entity_density,
            "temporal_expression_count": self.temporal_expression_count,
            "cross_reference_count": self.cross_reference_count,
            "estimated_token_count": self.estimated_token_count,
            "complexity_score": self.complexity_score,
            "assessed_at": self.assessed_at.isoformat(),
        }


class CurriculumGenerator:
    """SEAgent-inspired curriculum for document ordering.

    Orders documents from simple to complex for optimal learning progression.
    The curriculum ensures that the experiential learning system encounters
    manageable documents first, building up patterns before tackling complex ones.

    Research Reference:
    - SEAgent (2508.04700v2) Section 3: "Curriculum Generator that generates
      increasingly diverse and challenging tasks"

    Strategy:
    1. Assess complexity of each document using multiple factors
    2. Order: simple -> medium -> complex
    3. Track which documents have been processed
    4. Optionally adapt curriculum based on learning progress

    Example:
        >>> generator = CurriculumGenerator()
        >>> complexity = generator.assess_document_complexity(document)
        >>> ordered_docs = generator.generate_curriculum(documents)
    """

    # Regex patterns for complexity assessment
    WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
    MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    TEMPORAL_PATTERNS = [
        re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),  # YYYY-MM-DD
        re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b"),  # MM-DD-YYYY
        re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?\b", re.I),
        re.compile(r"\b(?:yesterday|today|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year))\b", re.I),
        re.compile(r"\b(?:morning|afternoon|evening|night)\b", re.I),
        re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b", re.I),  # Time expressions
        re.compile(r"\b(?:before|after|during|while|when|since|until)\b", re.I),
    ]
    ENTITY_INDICATOR_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

    def __init__(
        self,
        temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
        entity_weight: float = DEFAULT_ENTITY_WEIGHT,
        length_weight: float = DEFAULT_LENGTH_WEIGHT,
        reference_weight: float = DEFAULT_REFERENCE_WEIGHT,
    ):
        """Initialize Curriculum Generator with complexity weights.

        Args:
            temporal_weight: Weight for temporal expression count (default 0.30)
            entity_weight: Weight for entity density (default 0.25)
            length_weight: Weight for document length (default 0.25)
            reference_weight: Weight for cross-references (default 0.20)
        """
        # Normalize weights to sum to 1.0
        total = temporal_weight + entity_weight + length_weight + reference_weight
        self.temporal_weight = temporal_weight / total
        self.entity_weight = entity_weight / total
        self.length_weight = length_weight / total
        self.reference_weight = reference_weight / total

        # Track processed documents
        self.processed_document_ids: set = set()
        self.complexity_cache: Dict[str, DocumentComplexity] = {}

    def assess_document_complexity(
        self,
        document: DocumentProtocol,
        use_cache: bool = True,
    ) -> DocumentComplexity:
        """Assess complexity factors for a single document.

        Args:
            document: Document to assess (must have content and doc_id)
            use_cache: Whether to use cached complexity if available

        Returns:
            DocumentComplexity with all factors and computed score
        """
        doc_id = getattr(document, "doc_id", str(id(document)))
        content = getattr(document, "content", "")

        # Check cache
        if use_cache and doc_id in self.complexity_cache:
            return self.complexity_cache[doc_id]

        # Compute complexity factors
        document_length = len(content)
        estimated_tokens = document_length // 4  # Rough token estimate

        # Count temporal expressions
        temporal_count = 0
        for pattern in self.TEMPORAL_PATTERNS:
            temporal_count += len(pattern.findall(content))

        # Count cross-references (wikilinks + markdown links)
        wikilinks = len(self.WIKILINK_PATTERN.findall(content))
        markdown_links = len(self.MARKDOWN_LINK_PATTERN.findall(content))
        cross_refs = wikilinks + markdown_links

        # Estimate entity density
        potential_entities = len(self.ENTITY_INDICATOR_PATTERN.findall(content))
        entity_density = (potential_entities / max(estimated_tokens, 1)) * 1000

        # Create complexity object
        complexity = DocumentComplexity(
            document_id=doc_id,
            document_length=document_length,
            entity_density=entity_density,
            temporal_expression_count=temporal_count,
            cross_reference_count=cross_refs,
            estimated_token_count=estimated_tokens,
        )

        # Compute weighted complexity score
        complexity.complexity_score = self._compute_complexity_score(complexity)

        # Cache result
        self.complexity_cache[doc_id] = complexity

        return complexity

    def _compute_complexity_score(self, complexity: DocumentComplexity) -> float:
        """Compute weighted complexity score from factors.

        Args:
            complexity: DocumentComplexity with raw factors

        Returns:
            float: Normalized complexity score (0-1)
        """
        # Normalize each factor to 0-1 range
        length_normalized = min(complexity.document_length / TARGET_DOCUMENT_LENGTH, 1.0)
        entity_normalized = min(complexity.entity_density / TARGET_ENTITY_DENSITY, 1.0)
        temporal_normalized = min(
            complexity.temporal_expression_count / TARGET_TEMPORAL_EXPRESSIONS, 1.0
        )
        refs_normalized = min(complexity.cross_reference_count / TARGET_CROSS_REFERENCES, 1.0)

        # Apply weights
        score = (
            self.length_weight * length_normalized
            + self.entity_weight * entity_normalized
            + self.temporal_weight * temporal_normalized
            + self.reference_weight * refs_normalized
        )

        return score

    def generate_curriculum(
        self,
        documents: List[DocumentProtocol],
        strategy: str = "progressive",
        exclude_processed: bool = False,
    ) -> List[DocumentProtocol]:
        """Order documents by complexity for optimal learning.

        Args:
            documents: List of documents to order
            strategy: Ordering strategy:
                - "progressive": Fixed simple -> complex ordering (default)
                - "reverse": Complex -> simple (for testing)
                - "adaptive": Adjusts based on current learning success rate
            exclude_processed: If True, exclude already-processed documents

        Returns:
            List of documents ordered by complexity
        """
        if not documents:
            return []

        # Filter out processed if requested
        if exclude_processed:
            documents = [
                d for d in documents
                if getattr(d, "doc_id", str(id(d))) not in self.processed_document_ids
            ]

        # Assess complexity for all documents
        complexities = [
            (doc, self.assess_document_complexity(doc))
            for doc in documents
        ]

        # Sort based on strategy
        if strategy == "progressive":
            # Simple to complex
            sorted_docs = sorted(complexities, key=lambda x: x[1].complexity_score)
        elif strategy == "reverse":
            # Complex to simple
            sorted_docs = sorted(complexities, key=lambda x: x[1].complexity_score, reverse=True)
        elif strategy == "adaptive":
            # Default to progressive for now (adaptive would need learning state)
            sorted_docs = sorted(complexities, key=lambda x: x[1].complexity_score)
        else:
            raise ValueError(f"Unknown curriculum strategy: {strategy}")

        logger.info(
            f"Generated curriculum for {len(sorted_docs)} documents "
            f"(strategy: {strategy}, complexity range: "
            f"{sorted_docs[0][1].complexity_score:.3f} - {sorted_docs[-1][1].complexity_score:.3f})"
        )

        return [doc for doc, _ in sorted_docs]

    def get_next_batch(
        self,
        documents: List[DocumentProtocol],
        batch_size: int,
        current_quality: Optional[float] = None,
    ) -> List[DocumentProtocol]:
        """Get next batch of documents adapted to current learning state.

        Args:
            documents: Pool of documents to select from
            batch_size: Number of documents to return
            current_quality: Current quality score (for adaptive selection)

        Returns:
            List of documents for next batch
        """
        # Filter out already processed documents
        unprocessed = [
            doc for doc in documents
            if getattr(doc, "doc_id", str(id(doc))) not in self.processed_document_ids
        ]

        if not unprocessed:
            logger.info("All documents have been processed")
            return []

        # Order by complexity
        ordered = self.generate_curriculum(unprocessed, strategy="progressive")

        # If quality is low, prefer simpler documents
        if current_quality is not None and current_quality < 0.5:
            # Take from the simpler end
            batch = ordered[:batch_size]
        else:
            # Progress to more complex documents
            batch = ordered[:batch_size]

        return batch

    def mark_processed(self, document_ids: List[str]) -> None:
        """Mark documents as processed.

        Args:
            document_ids: List of document IDs that have been processed
        """
        self.processed_document_ids.update(document_ids)
        logger.debug(f"Marked {len(document_ids)} documents as processed")

    def get_complexity_distribution(
        self,
        documents: List[DocumentProtocol],
    ) -> Dict[str, Any]:
        """Get distribution of complexity scores across documents.

        Args:
            documents: Documents to analyze

        Returns:
            Dict with complexity distribution statistics
        """
        if not documents:
            return {"status": "no_documents"}

        complexities = [self.assess_document_complexity(doc) for doc in documents]
        scores = [c.complexity_score for c in complexities]

        # Categorize into buckets
        simple = sum(1 for s in scores if s < 0.33)
        medium = sum(1 for s in scores if 0.33 <= s < 0.66)
        complex_count = sum(1 for s in scores if s >= 0.66)

        return {
            "total_documents": len(documents),
            "simple_count": simple,
            "medium_count": medium,
            "complex_count": complex_count,
            "min_complexity": min(scores),
            "max_complexity": max(scores),
            "avg_complexity": sum(scores) / len(scores),
            "processed_count": len(self.processed_document_ids),
            "remaining_count": len(documents) - len(
                [d for d in documents
                 if getattr(d, "doc_id", str(id(d))) in self.processed_document_ids]
            ),
        }

    def reset(self) -> None:
        """Reset curriculum state (clear processed tracking and cache)."""
        count = len(self.processed_document_ids)
        self.processed_document_ids.clear()
        self.complexity_cache.clear()
        logger.info(f"Reset curriculum generator (cleared {count} processed documents)")
