Summary: Enable hybrid search across multimodal content with source-aware retrieval for OCR-extracted documents, audio transcriptions, and mixed-source queries.

# 07 · Multimodal Query Handling

## Purpose
Enable hybrid search to handle queries about content indexed from multiple modalities (text, scanned documents, audio transcriptions) with source-aware retrieval strategies that account for modality-specific characteristics.

**Criticality**: HIGH - Ensures search quality across all ingested content types

## Scope
- OCR-extracted content retrieval (DeepSeek-OCR integration)
- Audio transcription search (Whisper V3 integration)
- Multimodal routing via NVIDIA Orchestrator-8B
- Source confidence scoring and ranking
- Cross-modal relevance fusion
- Modality hint detection in queries

## Requirements Alignment
- **Option B Requirement**: "Local-first processing" with multimodal support
- **Quality Target**: >80% relevance for OCR content, >75% for audio
- **Integration**: Leverages existing PKG metadata from ingestion pipeline

## Research Foundation

### Multimodal RAG Patterns
- **Source-Aware Retrieval**: Different retrieval strategies per content source
- **Confidence-Weighted Ranking**: Weight results by extraction confidence
- **Cross-Modal Fusion**: Combine results from different modalities coherently

### Key Papers
- **CausalRAG (ACL 2025)**: Multi-source evidence integration for causal reasoning
- **DeepSearch**: Multimodal deep search with adaptive retrieval

---

## Component Design

### 1. Content Source Types

```python
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import re


class ContentSource(str, Enum):
    """Types of content sources in PKG."""
    TEXT_NATIVE = "text_native"           # Direct text files (Markdown, etc.)
    OCR_DOCUMENT = "ocr_document"         # Scanned documents via DeepSeek-OCR
    OCR_IMAGE = "ocr_image"               # Images with text via DeepSeek-OCR
    AUDIO_TRANSCRIPTION = "audio_transcription"  # Whisper V3 transcriptions
    VIDEO_TRANSCRIPTION = "video_transcription"  # Video audio track transcriptions
    MIXED_SOURCE = "mixed_source"         # Composite documents


class ExtractionQuality(str, Enum):
    """Quality tier of extracted content."""
    HIGH = "high"           # Confidence >0.95
    MEDIUM = "medium"       # Confidence 0.80-0.95
    LOW = "low"            # Confidence 0.60-0.80
    UNCERTAIN = "uncertain" # Confidence <0.60


@dataclass
class SourceMetadata:
    """Metadata about content source for retrieval optimization."""
    source_type: ContentSource
    extraction_confidence: float          # 0.0-1.0
    extraction_quality: ExtractionQuality
    extractor_version: str               # e.g., "deepseek-ocr-v2", "whisper-v3"
    extraction_timestamp: datetime
    original_format: str                 # e.g., "pdf", "mp3", "png"
    language_detected: str               # ISO code
    word_error_rate: Optional[float]     # For audio (WER estimate)
    character_error_rate: Optional[float] # For OCR (CER estimate)
    layout_complexity: Optional[str]     # For documents: simple/complex/table-heavy
    audio_quality: Optional[str]         # For audio: clean/noisy/mixed
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def retrieval_boost(self) -> float:
        """Calculate retrieval score boost based on source quality."""
        base_boost = {
            ContentSource.TEXT_NATIVE: 1.0,
            ContentSource.OCR_DOCUMENT: 0.9,
            ContentSource.OCR_IMAGE: 0.85,
            ContentSource.AUDIO_TRANSCRIPTION: 0.85,
            ContentSource.VIDEO_TRANSCRIPTION: 0.8,
            ContentSource.MIXED_SOURCE: 0.75
        }.get(self.source_type, 0.7)

        # Adjust by extraction confidence
        confidence_factor = 0.5 + (self.extraction_confidence * 0.5)

        return base_boost * confidence_factor
```

---

### 2. Modality Hint Detector

```python
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModalityHint:
    """Detected modality hint from query."""
    modality: ContentSource
    confidence: float
    hint_phrase: str
    query_position: Tuple[int, int]  # Start, end in query


class ModalityHintDetector:
    """
    Detects modality hints in user queries.

    Examples:
    - "in my voice notes" → AUDIO_TRANSCRIPTION
    - "from the scanned document" → OCR_DOCUMENT
    - "in that PDF I uploaded" → OCR_DOCUMENT or TEXT_NATIVE
    - "what I said in the meeting" → AUDIO_TRANSCRIPTION

    Integration Points:
    - QueryRouter: Provides modality hints for routing decisions
    - MultimodalQueryHandler: Configures source filtering
    """

    MODALITY_PATTERNS = {
        ContentSource.AUDIO_TRANSCRIPTION: [
            (r"(?:in|from)\s+(?:my|the)\s+(?:voice\s+)?(?:notes?|memos?|recordings?)", 0.95),
            (r"(?:what|when)\s+(?:I|we)\s+(?:said|mentioned|discussed)", 0.85),
            (r"(?:in|during|from)\s+(?:the|that|my)\s+(?:meeting|call|conversation|interview)", 0.90),
            (r"(?:audio|recording|podcast|episode)", 0.80),
            (r"(?:I|we)\s+(?:talked|spoke)\s+about", 0.85),
            (r"transcri(?:pt|ption|bed)", 0.90),
        ],
        ContentSource.OCR_DOCUMENT: [
            (r"(?:in|from)\s+(?:the|that)\s+(?:scanned|uploaded)\s+(?:document|pdf|page)", 0.95),
            (r"(?:in|from)\s+(?:the|that)\s+(?:pdf|scan)", 0.90),
            (r"(?:handwritten|printed)\s+(?:notes?|document)", 0.85),
            (r"(?:the|that)\s+(?:paper|letter|form|receipt)", 0.80),
            (r"(?:scanned|digitized)", 0.85),
        ],
        ContentSource.OCR_IMAGE: [
            (r"(?:in|from)\s+(?:the|that)\s+(?:image|photo|picture|screenshot)", 0.90),
            (r"(?:text|words)\s+(?:in|on|from)\s+(?:the|that)\s+(?:image|photo)", 0.95),
            (r"(?:whiteboard|blackboard|sign|poster)", 0.85),
            (r"(?:screenshot|screen\s*shot|capture)", 0.90),
        ],
        ContentSource.VIDEO_TRANSCRIPTION: [
            (r"(?:in|from)\s+(?:the|that)\s+video", 0.90),
            (r"(?:youtube|vimeo|video\s+recording)", 0.85),
            (r"(?:watched|viewed)\s+(?:video|clip)", 0.80),
        ],
    }

    def __init__(self):
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[ContentSource, List[Tuple[re.Pattern, float]]]:
        """Pre-compile regex patterns for efficiency."""
        compiled = {}
        for source, patterns in self.MODALITY_PATTERNS.items():
            compiled[source] = [
                (re.compile(pattern, re.IGNORECASE), conf)
                for pattern, conf in patterns
            ]
        return compiled

    def detect(self, query: str) -> List[ModalityHint]:
        """
        Detect modality hints in query.

        Returns:
            List of detected hints, sorted by confidence
        """
        hints = []

        for source, patterns in self._compiled_patterns.items():
            for pattern, base_confidence in patterns:
                matches = pattern.finditer(query)
                for match in matches:
                    hints.append(ModalityHint(
                        modality=source,
                        confidence=base_confidence,
                        hint_phrase=match.group(0),
                        query_position=(match.start(), match.end())
                    ))

        # Sort by confidence descending
        hints.sort(key=lambda h: h.confidence, reverse=True)

        return hints

    def get_primary_modality(self, query: str) -> Optional[ContentSource]:
        """Get the most likely intended modality, if any."""
        hints = self.detect(query)
        if hints and hints[0].confidence >= 0.80:
            return hints[0].modality
        return None

    def should_filter_by_modality(self, query: str) -> bool:
        """Determine if query should filter by modality."""
        hints = self.detect(query)
        return len(hints) > 0 and hints[0].confidence >= 0.75
```

---

### 3. Multimodal Query Handler

```python
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RetrievalMode(str, Enum):
    """Retrieval mode based on modality analysis."""
    ALL_SOURCES = "all_sources"           # Search all content
    SINGLE_MODALITY = "single_modality"   # Filter to specific modality
    PRIORITIZED = "prioritized"           # Search all but boost specific modality
    CROSS_MODAL = "cross_modal"           # Explicitly cross-modal query


@dataclass
class MultimodalQueryPlan:
    """Query execution plan for multimodal search."""
    retrieval_mode: RetrievalMode
    target_modalities: List[ContentSource]
    modality_weights: Dict[ContentSource, float]
    apply_confidence_weighting: bool
    fuzzy_matching_boost: float           # For OCR content
    semantic_priority: float              # For audio content
    cross_modal_fusion: bool


class MultimodalQueryHandler:
    """
    Handles queries across multimodal content.

    Strategies:
    1. Native text: Standard hybrid retrieval
    2. OCR content: Enhanced fuzzy matching for OCR errors
    3. Audio content: Semantic similarity priority (handles transcription variance)
    4. Mixed sources: Unified ranking with source weighting

    Integration Points:
    - ModalityHintDetector: Query analysis for modality hints
    - SourceMetadata: Content source information from PKG
    - HybridSearchAPI: Executes multimodal search plans
    """

    def __init__(
        self,
        hint_detector: ModalityHintDetector,
        pkg_client: "PKGClient",
        embedding_router: "QueryEmbeddingRouter"
    ):
        self.hint_detector = hint_detector
        self.pkg = pkg_client
        self.embedding_router = embedding_router

        # Default modality weights for unified search
        self.default_weights = {
            ContentSource.TEXT_NATIVE: 1.0,
            ContentSource.OCR_DOCUMENT: 0.95,
            ContentSource.OCR_IMAGE: 0.90,
            ContentSource.AUDIO_TRANSCRIPTION: 0.90,
            ContentSource.VIDEO_TRANSCRIPTION: 0.85,
            ContentSource.MIXED_SOURCE: 0.80
        }

    def analyze_query(self, query: str) -> MultimodalQueryPlan:
        """
        Analyze query and create multimodal execution plan.

        Args:
            query: User search query

        Returns:
            MultimodalQueryPlan with retrieval configuration
        """
        hints = self.hint_detector.detect(query)

        # No modality hints - search all sources
        if not hints:
            return MultimodalQueryPlan(
                retrieval_mode=RetrievalMode.ALL_SOURCES,
                target_modalities=list(ContentSource),
                modality_weights=self.default_weights,
                apply_confidence_weighting=True,
                fuzzy_matching_boost=1.0,
                semantic_priority=1.0,
                cross_modal_fusion=False
            )

        primary_hint = hints[0]

        # Strong modality hint - filter to that modality
        if primary_hint.confidence >= 0.90:
            return MultimodalQueryPlan(
                retrieval_mode=RetrievalMode.SINGLE_MODALITY,
                target_modalities=[primary_hint.modality],
                modality_weights={primary_hint.modality: 1.0},
                apply_confidence_weighting=True,
                fuzzy_matching_boost=self._get_fuzzy_boost(primary_hint.modality),
                semantic_priority=self._get_semantic_priority(primary_hint.modality),
                cross_modal_fusion=False
            )

        # Moderate hint - prioritize but don't exclude
        if primary_hint.confidence >= 0.75:
            weights = self.default_weights.copy()
            weights[primary_hint.modality] = 1.5  # Boost hinted modality

            return MultimodalQueryPlan(
                retrieval_mode=RetrievalMode.PRIORITIZED,
                target_modalities=list(ContentSource),
                modality_weights=weights,
                apply_confidence_weighting=True,
                fuzzy_matching_boost=self._get_fuzzy_boost(primary_hint.modality),
                semantic_priority=self._get_semantic_priority(primary_hint.modality),
                cross_modal_fusion=False
            )

        # Weak hints - search all with slight prioritization
        return MultimodalQueryPlan(
            retrieval_mode=RetrievalMode.ALL_SOURCES,
            target_modalities=list(ContentSource),
            modality_weights=self.default_weights,
            apply_confidence_weighting=True,
            fuzzy_matching_boost=1.0,
            semantic_priority=1.0,
            cross_modal_fusion=True
        )

    def _get_fuzzy_boost(self, modality: ContentSource) -> float:
        """Get fuzzy matching boost for modality."""
        # OCR content benefits from fuzzy matching
        fuzzy_boosts = {
            ContentSource.OCR_DOCUMENT: 1.3,
            ContentSource.OCR_IMAGE: 1.4,
            ContentSource.AUDIO_TRANSCRIPTION: 1.1,
        }
        return fuzzy_boosts.get(modality, 1.0)

    def _get_semantic_priority(self, modality: ContentSource) -> float:
        """Get semantic search priority for modality."""
        # Audio content benefits from semantic search
        semantic_priorities = {
            ContentSource.AUDIO_TRANSCRIPTION: 1.4,
            ContentSource.VIDEO_TRANSCRIPTION: 1.3,
            ContentSource.OCR_DOCUMENT: 1.1,
        }
        return semantic_priorities.get(modality, 1.0)

    async def execute(
        self,
        query: str,
        plan: MultimodalQueryPlan,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute multimodal search.

        Args:
            query: Search query
            plan: Execution plan from analyze_query
            top_k: Number of results to return

        Returns:
            List of search results with source metadata
        """
        # Build source filter
        source_filter = self._build_source_filter(plan)

        # Execute retrieval based on mode
        if plan.retrieval_mode == RetrievalMode.SINGLE_MODALITY:
            results = await self._single_modality_search(
                query, plan, source_filter, top_k
            )
        elif plan.retrieval_mode == RetrievalMode.PRIORITIZED:
            results = await self._prioritized_search(
                query, plan, source_filter, top_k
            )
        else:
            results = await self._all_sources_search(
                query, plan, top_k
            )

        # Apply source-aware ranking
        ranked_results = self._apply_source_ranking(results, plan)

        return ranked_results[:top_k]

    async def _single_modality_search(
        self,
        query: str,
        plan: MultimodalQueryPlan,
        source_filter: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search within single modality."""
        modality = plan.target_modalities[0]

        # Configure search based on modality
        if modality in [ContentSource.OCR_DOCUMENT, ContentSource.OCR_IMAGE]:
            # Enhanced fuzzy matching for OCR
            return await self._ocr_optimized_search(
                query, source_filter, top_k, plan.fuzzy_matching_boost
            )

        elif modality in [ContentSource.AUDIO_TRANSCRIPTION, ContentSource.VIDEO_TRANSCRIPTION]:
            # Semantic-first for transcriptions
            return await self._transcription_optimized_search(
                query, source_filter, top_k, plan.semantic_priority
            )

        else:
            # Standard hybrid search
            return await self._standard_hybrid_search(
                query, source_filter, top_k
            )

    async def _ocr_optimized_search(
        self,
        query: str,
        source_filter: Dict[str, Any],
        top_k: int,
        fuzzy_boost: float
    ) -> List[Dict[str, Any]]:
        """
        OCR-optimized search with fuzzy matching.

        OCR content may have character recognition errors:
        - "receipt" might be "reciept" or "recei pt"
        - Handles common OCR error patterns
        """
        # Standard embedding search
        query_embedding = await self.embedding_router.embed_query(query)

        results = await self.pkg.vector_search(
            embedding=query_embedding,
            filters=source_filter,
            top_k=top_k * 2  # Over-fetch for re-ranking
        )

        # Apply fuzzy text matching as secondary signal
        fuzzy_results = await self.pkg.fuzzy_text_search(
            query=query,
            filters=source_filter,
            max_edit_distance=2,  # Allow 2 character edits
            top_k=top_k
        )

        # Merge results with fuzzy boost
        merged = self._merge_with_boost(
            results, fuzzy_results, fuzzy_boost
        )

        return merged

    async def _transcription_optimized_search(
        self,
        query: str,
        source_filter: Dict[str, Any],
        top_k: int,
        semantic_priority: float
    ) -> List[Dict[str, Any]]:
        """
        Transcription-optimized search with semantic priority.

        Audio transcriptions may have:
        - Word substitutions (homophones)
        - Missing words
        - Speaker diarization noise

        Semantic search handles these better than exact matching.
        """
        # Embed with semantic-focused model
        query_embedding = await self.embedding_router.embed_query(
            query,
            optimize_for="semantic_similarity"
        )

        # Heavy weight on semantic search
        semantic_results = await self.pkg.vector_search(
            embedding=query_embedding,
            filters=source_filter,
            top_k=top_k
        )

        # Light keyword search for high-confidence terms
        keyword_results = await self.pkg.keyword_search(
            query=query,
            filters=source_filter,
            top_k=top_k // 2
        )

        # Merge with semantic priority
        return self._merge_with_semantic_priority(
            semantic_results, keyword_results, semantic_priority
        )

    async def _prioritized_search(
        self,
        query: str,
        plan: MultimodalQueryPlan,
        source_filter: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search all sources with modality prioritization."""
        query_embedding = await self.embedding_router.embed_query(query)

        # Search all sources
        all_results = await self.pkg.vector_search(
            embedding=query_embedding,
            filters=None,  # No source filter
            top_k=top_k * 3
        )

        # Apply modality weights
        for result in all_results:
            source = result.get("source_metadata", {}).get("source_type")
            if source:
                weight = plan.modality_weights.get(
                    ContentSource(source),
                    self.default_weights.get(ContentSource(source), 0.8)
                )
                result["weighted_score"] = result.get("score", 0) * weight
            else:
                result["weighted_score"] = result.get("score", 0)

        # Re-sort by weighted score
        all_results.sort(key=lambda r: r["weighted_score"], reverse=True)

        return all_results

    async def _all_sources_search(
        self,
        query: str,
        plan: MultimodalQueryPlan,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Standard search across all sources."""
        query_embedding = await self.embedding_router.embed_query(query)

        results = await self.pkg.vector_search(
            embedding=query_embedding,
            filters=None,
            top_k=top_k * 2
        )

        return results

    def _apply_source_ranking(
        self,
        results: List[Dict[str, Any]],
        plan: MultimodalQueryPlan
    ) -> List[Dict[str, Any]]:
        """Apply source-aware ranking adjustments."""
        for result in results:
            source_meta = result.get("source_metadata", {})

            # Apply extraction confidence weighting
            if plan.apply_confidence_weighting:
                confidence = source_meta.get("extraction_confidence", 1.0)
                result["adjusted_score"] = (
                    result.get("weighted_score", result.get("score", 0)) *
                    (0.7 + confidence * 0.3)  # 70% base + 30% from confidence
                )
            else:
                result["adjusted_score"] = result.get(
                    "weighted_score", result.get("score", 0)
                )

        # Sort by adjusted score
        results.sort(key=lambda r: r["adjusted_score"], reverse=True)

        return results

    def _build_source_filter(
        self,
        plan: MultimodalQueryPlan
    ) -> Optional[Dict[str, Any]]:
        """Build PKG filter for source types."""
        if plan.retrieval_mode == RetrievalMode.ALL_SOURCES:
            return None

        return {
            "source_type": {
                "$in": [m.value for m in plan.target_modalities]
            }
        }

    def _merge_with_boost(
        self,
        primary: List[Dict[str, Any]],
        boosted: List[Dict[str, Any]],
        boost_factor: float
    ) -> List[Dict[str, Any]]:
        """Merge result sets with boost for secondary results."""
        # Create ID -> result mapping
        result_map = {r["id"]: r for r in primary}

        # Add boosted results, merging scores
        for r in boosted:
            rid = r["id"]
            if rid in result_map:
                # Merge: add boosted score
                result_map[rid]["score"] += r.get("score", 0) * boost_factor
            else:
                r["score"] = r.get("score", 0) * boost_factor
                result_map[rid] = r

        merged = list(result_map.values())
        merged.sort(key=lambda r: r["score"], reverse=True)

        return merged

    def _merge_with_semantic_priority(
        self,
        semantic: List[Dict[str, Any]],
        keyword: List[Dict[str, Any]],
        priority_factor: float
    ) -> List[Dict[str, Any]]:
        """Merge with strong semantic priority."""
        result_map = {}

        # Semantic results with priority boost
        for i, r in enumerate(semantic):
            r["combined_score"] = r.get("score", 0) * priority_factor
            r["semantic_rank"] = i
            result_map[r["id"]] = r

        # Keyword results as secondary signal
        for i, r in enumerate(keyword):
            rid = r["id"]
            if rid in result_map:
                # Small boost for keyword match
                result_map[rid]["combined_score"] += r.get("score", 0) * 0.3
            else:
                r["combined_score"] = r.get("score", 0) * 0.5
                r["semantic_rank"] = len(semantic) + i
                result_map[rid] = r

        merged = list(result_map.values())
        merged.sort(key=lambda r: r["combined_score"], reverse=True)

        return merged
```

---

### 4. OCR Content Integration

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class OCRLayoutType(str, Enum):
    """Document layout types from OCR processing."""
    SIMPLE_TEXT = "simple_text"       # Paragraphs, simple structure
    MULTI_COLUMN = "multi_column"     # Newspaper-style columns
    TABLE_HEAVY = "table_heavy"       # Many tables
    FORM = "form"                     # Form fields
    MIXED = "mixed"                   # Mixed content types
    HANDWRITTEN = "handwritten"       # Handwritten notes


@dataclass
class OCRContentMetadata:
    """Metadata specific to OCR-extracted content."""
    source_file: str
    page_number: Optional[int]
    layout_type: OCRLayoutType
    confidence_score: float           # Overall OCR confidence
    character_error_rate: float       # Estimated CER
    detected_language: str
    has_tables: bool
    has_images: bool
    has_handwriting: bool
    extraction_model: str             # "deepseek-ocr-v2"
    bounding_boxes_preserved: bool    # Spatial info available


class OCRContentProcessor:
    """
    Processes OCR-extracted content for optimal retrieval.

    Integration with DeepSeek-OCR output:
    - Preserves layout information for spatial queries
    - Handles table structures
    - Manages confidence scores per region
    - Enables fuzzy matching for OCR errors

    Integration Points:
    - MultimodalQueryHandler: OCR-specific search strategies
    - PKGClient: Stores OCR metadata with content
    """

    # Common OCR error patterns for fuzzy matching
    OCR_ERROR_PATTERNS = {
        'l': ['1', 'I', '|'],
        'O': ['0', 'Q'],
        'm': ['rn', 'nn'],
        'w': ['vv'],
        'cl': ['d'],
        'rn': ['m'],
        ' ': [''],  # Missing spaces
    }

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence

    def process_ocr_result(
        self,
        ocr_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process DeepSeek-OCR output for PKG storage.

        Args:
            ocr_output: Raw output from DeepSeek-OCR

        Returns:
            Processed content with metadata for PKG storage
        """
        # Extract text content
        text_content = self._extract_text(ocr_output)

        # Build metadata
        metadata = OCRContentMetadata(
            source_file=ocr_output.get("source_file", ""),
            page_number=ocr_output.get("page", None),
            layout_type=self._detect_layout_type(ocr_output),
            confidence_score=ocr_output.get("confidence", 0.0),
            character_error_rate=self._estimate_cer(ocr_output),
            detected_language=ocr_output.get("language", "en"),
            has_tables=self._has_tables(ocr_output),
            has_images=self._has_images(ocr_output),
            has_handwriting=ocr_output.get("has_handwriting", False),
            extraction_model="deepseek-ocr-v2",
            bounding_boxes_preserved=True
        )

        # Generate fuzzy variants for error tolerance
        fuzzy_variants = self._generate_fuzzy_variants(text_content)

        return {
            "content": text_content,
            "source_metadata": {
                "source_type": ContentSource.OCR_DOCUMENT.value,
                "extraction_confidence": metadata.confidence_score,
                "extraction_quality": self._quality_tier(metadata.confidence_score).value,
                "extractor_version": metadata.extraction_model,
                "original_format": self._detect_format(ocr_output),
                "character_error_rate": metadata.character_error_rate,
                "layout_complexity": metadata.layout_type.value,
            },
            "ocr_metadata": {
                "layout_type": metadata.layout_type.value,
                "has_tables": metadata.has_tables,
                "has_images": metadata.has_images,
                "has_handwriting": metadata.has_handwriting,
                "page_number": metadata.page_number,
            },
            "fuzzy_variants": fuzzy_variants,
            "bounding_boxes": ocr_output.get("boxes", [])
        }

    def _extract_text(self, ocr_output: Dict[str, Any]) -> str:
        """Extract clean text from OCR output."""
        if "text" in ocr_output:
            return ocr_output["text"]

        # Reconstruct from blocks if needed
        blocks = ocr_output.get("blocks", [])
        text_parts = []

        for block in blocks:
            if block.get("confidence", 0) >= self.min_confidence:
                text_parts.append(block.get("text", ""))

        return "\n".join(text_parts)

    def _detect_layout_type(self, ocr_output: Dict[str, Any]) -> OCRLayoutType:
        """Detect document layout type."""
        layout_info = ocr_output.get("layout", {})

        if layout_info.get("is_handwritten", False):
            return OCRLayoutType.HANDWRITTEN

        if layout_info.get("table_count", 0) > 2:
            return OCRLayoutType.TABLE_HEAVY

        if layout_info.get("column_count", 1) > 1:
            return OCRLayoutType.MULTI_COLUMN

        if layout_info.get("is_form", False):
            return OCRLayoutType.FORM

        if layout_info.get("is_mixed", False):
            return OCRLayoutType.MIXED

        return OCRLayoutType.SIMPLE_TEXT

    def _estimate_cer(self, ocr_output: Dict[str, Any]) -> float:
        """Estimate Character Error Rate."""
        # Use provided CER if available
        if "cer" in ocr_output:
            return ocr_output["cer"]

        # Estimate from confidence
        confidence = ocr_output.get("confidence", 0.9)
        # Rough estimation: CER ≈ (1 - confidence) * 0.5
        return (1 - confidence) * 0.5

    def _has_tables(self, ocr_output: Dict[str, Any]) -> bool:
        """Check if document contains tables."""
        return ocr_output.get("layout", {}).get("table_count", 0) > 0

    def _has_images(self, ocr_output: Dict[str, Any]) -> bool:
        """Check if document contains images."""
        return ocr_output.get("layout", {}).get("image_count", 0) > 0

    def _quality_tier(self, confidence: float) -> ExtractionQuality:
        """Map confidence to quality tier."""
        if confidence >= 0.95:
            return ExtractionQuality.HIGH
        elif confidence >= 0.80:
            return ExtractionQuality.MEDIUM
        elif confidence >= 0.60:
            return ExtractionQuality.LOW
        else:
            return ExtractionQuality.UNCERTAIN

    def _detect_format(self, ocr_output: Dict[str, Any]) -> str:
        """Detect original file format."""
        source = ocr_output.get("source_file", "")
        if source.lower().endswith(".pdf"):
            return "pdf"
        elif source.lower().endswith((".png", ".jpg", ".jpeg")):
            return "image"
        elif source.lower().endswith(".tiff"):
            return "tiff"
        return "unknown"

    def _generate_fuzzy_variants(self, text: str) -> List[str]:
        """Generate common OCR error variants for fuzzy matching."""
        variants = []

        # Generate variants based on common OCR errors
        for original, replacements in self.OCR_ERROR_PATTERNS.items():
            if original in text:
                for replacement in replacements:
                    variant = text.replace(original, replacement, 1)
                    if variant != text and variant not in variants:
                        variants.append(variant)

        # Limit variants to prevent explosion
        return variants[:10]
```

---

### 5. Audio Transcription Integration

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class AudioQuality(str, Enum):
    """Audio quality classification."""
    STUDIO = "studio"           # Professional recording
    CLEAN = "clean"            # Good quality, minimal noise
    NOISY = "noisy"            # Background noise present
    MIXED = "mixed"            # Variable quality
    POOR = "poor"              # Significant quality issues


@dataclass
class TranscriptionMetadata:
    """Metadata for audio transcription content."""
    source_file: str
    duration_seconds: float
    audio_quality: AudioQuality
    word_error_rate: float           # Estimated WER
    speaker_count: int               # Number of speakers detected
    language: str
    model_version: str               # "whisper-v3-turbo"
    has_timestamps: bool
    has_speaker_labels: bool
    confidence_scores: List[float]   # Per-segment confidence


class TranscriptionProcessor:
    """
    Processes Whisper V3 transcriptions for optimal retrieval.

    Handles transcription-specific challenges:
    - Speaker diarization noise
    - Homophone confusion
    - Filler word handling
    - Timestamp alignment

    Integration Points:
    - MultimodalQueryHandler: Transcription-specific search
    - PKGClient: Stores transcription metadata
    """

    # Common transcription confusions for semantic handling
    HOMOPHONE_GROUPS = [
        ["their", "there", "they're"],
        ["to", "too", "two"],
        ["its", "it's"],
        ["your", "you're"],
        ["hear", "here"],
        ["know", "no"],
        ["write", "right"],
        ["weather", "whether"],
    ]

    # Filler words to handle specially
    FILLER_WORDS = [
        "um", "uh", "like", "you know", "basically",
        "actually", "literally", "honestly", "right"
    ]

    def __init__(self):
        self.homophone_map = self._build_homophone_map()

    def _build_homophone_map(self) -> Dict[str, List[str]]:
        """Build homophone lookup for search expansion."""
        hmap = {}
        for group in self.HOMOPHONE_GROUPS:
            for word in group:
                hmap[word] = [w for w in group if w != word]
        return hmap

    def process_transcription(
        self,
        whisper_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process Whisper V3 output for PKG storage.

        Args:
            whisper_output: Raw Whisper output

        Returns:
            Processed content with metadata
        """
        # Extract and clean text
        text_content = self._extract_clean_text(whisper_output)

        # Build metadata
        metadata = self._build_metadata(whisper_output)

        # Generate search-optimized content
        searchable_content = self._create_searchable_content(
            text_content, whisper_output
        )

        return {
            "content": text_content,
            "searchable_content": searchable_content,
            "source_metadata": {
                "source_type": ContentSource.AUDIO_TRANSCRIPTION.value,
                "extraction_confidence": self._avg_confidence(whisper_output),
                "extraction_quality": self._quality_tier(metadata.word_error_rate).value,
                "extractor_version": "whisper-v3-turbo",
                "original_format": self._detect_format(whisper_output),
                "word_error_rate": metadata.word_error_rate,
                "audio_quality": metadata.audio_quality.value,
            },
            "transcription_metadata": {
                "duration_seconds": metadata.duration_seconds,
                "speaker_count": metadata.speaker_count,
                "has_timestamps": metadata.has_timestamps,
                "has_speaker_labels": metadata.has_speaker_labels,
                "language": metadata.language,
            },
            "segments": self._extract_segments(whisper_output),
            "speakers": self._extract_speakers(whisper_output)
        }

    def _extract_clean_text(self, whisper_output: Dict[str, Any]) -> str:
        """Extract clean text from Whisper output."""
        if "text" in whisper_output:
            text = whisper_output["text"]
        else:
            segments = whisper_output.get("segments", [])
            text = " ".join(s.get("text", "") for s in segments)

        # Clean up common transcription artifacts
        text = self._clean_transcription(text)

        return text

    def _clean_transcription(self, text: str) -> str:
        """Clean transcription artifacts."""
        import re

        # Remove repeated filler words
        for filler in self.FILLER_WORDS:
            pattern = rf'\b({filler})\s+\1\b'
            text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _build_metadata(self, whisper_output: Dict[str, Any]) -> TranscriptionMetadata:
        """Build transcription metadata."""
        segments = whisper_output.get("segments", [])

        # Calculate duration
        if segments:
            duration = segments[-1].get("end", 0)
        else:
            duration = whisper_output.get("duration", 0)

        # Get confidence scores
        confidences = [
            s.get("confidence", s.get("avg_logprob", -0.5))
            for s in segments
        ]

        # Estimate WER from confidence
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
        estimated_wer = self._estimate_wer_from_confidence(avg_conf)

        return TranscriptionMetadata(
            source_file=whisper_output.get("source_file", ""),
            duration_seconds=duration,
            audio_quality=self._classify_audio_quality(whisper_output),
            word_error_rate=estimated_wer,
            speaker_count=len(set(s.get("speaker", "") for s in segments if s.get("speaker"))),
            language=whisper_output.get("language", "en"),
            model_version="whisper-v3-turbo",
            has_timestamps=any("start" in s for s in segments),
            has_speaker_labels=any("speaker" in s for s in segments),
            confidence_scores=confidences
        )

    def _estimate_wer_from_confidence(self, avg_confidence: float) -> float:
        """Estimate Word Error Rate from average confidence."""
        # Log prob to confidence mapping
        if avg_confidence < 0:  # Log prob
            confidence = min(1.0, max(0.0, 1.0 + (avg_confidence / 4)))
        else:
            confidence = avg_confidence

        # WER estimation: lower confidence = higher WER
        return max(0.01, (1 - confidence) * 0.4)

    def _classify_audio_quality(self, whisper_output: Dict[str, Any]) -> AudioQuality:
        """Classify audio quality."""
        quality_info = whisper_output.get("audio_quality", {})

        if quality_info.get("snr", 0) > 30:
            return AudioQuality.STUDIO
        elif quality_info.get("snr", 0) > 20:
            return AudioQuality.CLEAN
        elif quality_info.get("snr", 0) > 10:
            return AudioQuality.NOISY
        elif quality_info.get("variable", False):
            return AudioQuality.MIXED
        elif quality_info.get("snr", 0) <= 10:
            return AudioQuality.POOR

        return AudioQuality.CLEAN  # Default

    def _create_searchable_content(
        self,
        text: str,
        whisper_output: Dict[str, Any]
    ) -> str:
        """Create search-optimized content with expansions."""
        words = text.lower().split()
        expanded_words = []

        for word in words:
            expanded_words.append(word)
            # Add homophone variants
            if word in self.homophone_map:
                expanded_words.extend(self.homophone_map[word])

        return " ".join(expanded_words)

    def _extract_segments(self, whisper_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract timestamped segments."""
        segments = whisper_output.get("segments", [])
        return [
            {
                "start": s.get("start"),
                "end": s.get("end"),
                "text": s.get("text", ""),
                "speaker": s.get("speaker"),
                "confidence": s.get("confidence", s.get("avg_logprob"))
            }
            for s in segments
        ]

    def _extract_speakers(self, whisper_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract speaker information."""
        segments = whisper_output.get("segments", [])
        speakers = {}

        for s in segments:
            speaker = s.get("speaker")
            if speaker:
                if speaker not in speakers:
                    speakers[speaker] = {
                        "id": speaker,
                        "segments": [],
                        "total_duration": 0
                    }
                speakers[speaker]["segments"].append(s.get("id", len(speakers[speaker]["segments"])))
                speakers[speaker]["total_duration"] += s.get("end", 0) - s.get("start", 0)

        return list(speakers.values())

    def _quality_tier(self, wer: float) -> ExtractionQuality:
        """Map WER to quality tier."""
        if wer < 0.05:
            return ExtractionQuality.HIGH
        elif wer < 0.15:
            return ExtractionQuality.MEDIUM
        elif wer < 0.30:
            return ExtractionQuality.LOW
        else:
            return ExtractionQuality.UNCERTAIN

    def _avg_confidence(self, whisper_output: Dict[str, Any]) -> float:
        """Calculate average confidence score."""
        segments = whisper_output.get("segments", [])
        if not segments:
            return 0.8

        confidences = [
            s.get("confidence", 0.8) if s.get("confidence", 0) > 0
            else 0.8 + s.get("avg_logprob", -0.2) / 4
            for s in segments
        ]

        return sum(confidences) / len(confidences)

    def _detect_format(self, whisper_output: Dict[str, Any]) -> str:
        """Detect original audio format."""
        source = whisper_output.get("source_file", "")
        for ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]:
            if source.lower().endswith(ext):
                return ext[1:]
        return "audio"

    def expand_query_for_transcription(self, query: str) -> str:
        """
        Expand query with homophone variants.

        This improves recall when searching transcriptions
        that may have homophone errors.
        """
        words = query.lower().split()
        expanded = []

        for word in words:
            expanded.append(word)
            if word in self.homophone_map:
                expanded.extend(f"({alt})" for alt in self.homophone_map[word][:2])

        return " ".join(expanded)
```

---

### 6. Cross-Modal Fusion

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class FusionConfig:
    """Configuration for cross-modal result fusion."""
    normalize_scores: bool = True
    apply_diversity: bool = True
    diversity_factor: float = 0.2
    min_results_per_modality: int = 2
    max_modality_dominance: float = 0.7  # Max % from single modality


class CrossModalFusion:
    """
    Fuses results from multiple modalities into unified ranking.

    Handles:
    - Score normalization across modalities
    - Diversity injection to prevent modality dominance
    - Confidence-weighted combination
    - Source-aware deduplication

    Integration Points:
    - MultimodalQueryHandler: Final result fusion
    - HybridSearchAPI: Unified result presentation
    """

    def __init__(self, config: FusionConfig = None):
        self.config = config or FusionConfig()

    def fuse(
        self,
        results_by_modality: Dict[ContentSource, List[Dict[str, Any]]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple modalities.

        Args:
            results_by_modality: Results grouped by source modality
            top_k: Number of final results

        Returns:
            Fused and ranked result list
        """
        # Normalize scores within each modality
        if self.config.normalize_scores:
            results_by_modality = self._normalize_scores(results_by_modality)

        # Flatten to single list
        all_results = []
        for modality, results in results_by_modality.items():
            for r in results:
                r["_modality"] = modality
                all_results.append(r)

        # Deduplicate by content
        deduped = self._deduplicate(all_results)

        # Sort by score
        deduped.sort(key=lambda r: r.get("normalized_score", r.get("score", 0)), reverse=True)

        # Apply diversity if enabled
        if self.config.apply_diversity:
            deduped = self._apply_diversity(deduped, top_k)

        # Enforce max modality dominance
        final = self._enforce_modality_balance(deduped, top_k)

        return final[:top_k]

    def _normalize_scores(
        self,
        results_by_modality: Dict[ContentSource, List[Dict[str, Any]]]
    ) -> Dict[ContentSource, List[Dict[str, Any]]]:
        """Normalize scores to 0-1 range within each modality."""
        normalized = {}

        for modality, results in results_by_modality.items():
            if not results:
                normalized[modality] = []
                continue

            scores = [r.get("score", 0) for r in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            normalized_results = []
            for r in results:
                r_copy = r.copy()
                if score_range > 0:
                    r_copy["normalized_score"] = (r.get("score", 0) - min_score) / score_range
                else:
                    r_copy["normalized_score"] = 1.0
                normalized_results.append(r_copy)

            normalized[modality] = normalized_results

        return normalized

    def _deduplicate(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate results by content similarity."""
        seen_ids = set()
        deduped = []

        for r in results:
            rid = r.get("id", r.get("content_hash", str(r.get("content", ""))[:50]))
            if rid not in seen_ids:
                seen_ids.add(rid)
                deduped.append(r)

        return deduped

    def _apply_diversity(
        self,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Apply diversity injection using MMR-like approach.

        Prevents results that are too similar from dominating.
        """
        if len(results) <= top_k:
            return results

        selected = [results[0]]  # Start with top result
        remaining = results[1:]

        while len(selected) < top_k and remaining:
            best_idx = 0
            best_score = -float('inf')

            for i, candidate in enumerate(remaining):
                relevance = candidate.get("normalized_score", 0)

                # Diversity: penalty for similarity to selected
                diversity_penalty = self._calculate_diversity_penalty(
                    candidate, selected
                )

                combined = relevance - (self.config.diversity_factor * diversity_penalty)

                if combined > best_score:
                    best_score = combined
                    best_idx = i

            selected.append(remaining[best_idx])
            remaining.pop(best_idx)

        return selected + remaining

    def _calculate_diversity_penalty(
        self,
        candidate: Dict[str, Any],
        selected: List[Dict[str, Any]]
    ) -> float:
        """Calculate diversity penalty based on modality and content similarity."""
        penalty = 0.0

        candidate_modality = candidate.get("_modality")

        for s in selected:
            # Same modality penalty
            if s.get("_modality") == candidate_modality:
                penalty += 0.3

            # Content similarity penalty (simplified)
            if self._content_similar(candidate, s):
                penalty += 0.5

        return penalty / len(selected) if selected else 0.0

    def _content_similar(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any]
    ) -> bool:
        """Check if two results have similar content."""
        # Simple overlap check
        content_a = set(a.get("content", "").lower().split()[:20])
        content_b = set(b.get("content", "").lower().split()[:20])

        if not content_a or not content_b:
            return False

        overlap = len(content_a & content_b) / len(content_a | content_b)
        return overlap > 0.5

    def _enforce_modality_balance(
        self,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Ensure no single modality dominates results."""
        # Count by modality in top_k
        modality_counts: Dict[ContentSource, int] = {}

        for r in results[:top_k]:
            mod = r.get("_modality")
            modality_counts[mod] = modality_counts.get(mod, 0) + 1

        # Check dominance
        max_allowed = int(top_k * self.config.max_modality_dominance)

        # If any modality exceeds limit, rebalance
        for modality, count in modality_counts.items():
            if count > max_allowed:
                # Need to swap some results
                results = self._rebalance_modality(
                    results, modality, max_allowed, top_k
                )
                break

        return results

    def _rebalance_modality(
        self,
        results: List[Dict[str, Any]],
        dominant_modality: ContentSource,
        max_count: int,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rebalance to reduce modality dominance."""
        # Separate by modality
        dominant = [r for r in results if r.get("_modality") == dominant_modality]
        others = [r for r in results if r.get("_modality") != dominant_modality]

        # Keep top max_count from dominant, interleave with others
        kept_dominant = dominant[:max_count]
        remaining_dominant = dominant[max_count:]

        # Interleave
        rebalanced = []
        dom_idx = 0
        other_idx = 0

        for i in range(top_k):
            if dom_idx < len(kept_dominant) and (
                other_idx >= len(others) or
                kept_dominant[dom_idx].get("normalized_score", 0) >
                others[other_idx].get("normalized_score", 0)
            ):
                rebalanced.append(kept_dominant[dom_idx])
                dom_idx += 1
            elif other_idx < len(others):
                rebalanced.append(others[other_idx])
                other_idx += 1

        # Add remaining
        rebalanced.extend(kept_dominant[dom_idx:])
        rebalanced.extend(others[other_idx:])
        rebalanced.extend(remaining_dominant)

        return rebalanced
```

---

## Testing Strategy

### Unit Tests

```python
class TestModalityHintDetector:
    """Tests for modality hint detection."""

    def test_audio_hint_detection(self):
        """Test detection of audio-related hints."""
        detector = ModalityHintDetector()

        queries = [
            ("what did I say in my voice notes", ContentSource.AUDIO_TRANSCRIPTION, 0.9),
            ("from the meeting recording", ContentSource.AUDIO_TRANSCRIPTION, 0.9),
            ("in my podcast episode", ContentSource.AUDIO_TRANSCRIPTION, 0.8),
        ]

        for query, expected_source, min_confidence in queries:
            hints = detector.detect(query)
            assert len(hints) > 0, f"No hints for: {query}"
            assert hints[0].modality == expected_source
            assert hints[0].confidence >= min_confidence

    def test_ocr_hint_detection(self):
        """Test detection of OCR-related hints."""
        detector = ModalityHintDetector()

        queries = [
            ("from the scanned document", ContentSource.OCR_DOCUMENT, 0.9),
            ("in that PDF I uploaded", ContentSource.OCR_DOCUMENT, 0.9),
            ("the handwritten notes", ContentSource.OCR_DOCUMENT, 0.8),
        ]

        for query, expected_source, min_confidence in queries:
            hints = detector.detect(query)
            assert len(hints) > 0
            assert hints[0].modality == expected_source
            assert hints[0].confidence >= min_confidence

    def test_no_hint_query(self):
        """Test queries without modality hints."""
        detector = ModalityHintDetector()

        query = "what meetings do I have tomorrow"
        hints = detector.detect(query)

        # May have weak hints or none
        if hints:
            assert hints[0].confidence < 0.75


class TestMultimodalQueryHandler:
    """Tests for multimodal query handling."""

    def test_plan_generation_with_hint(self):
        """Test query plan generation with modality hint."""
        handler = MultimodalQueryHandler(
            hint_detector=ModalityHintDetector(),
            pkg_client=MockPKGClient(),
            embedding_router=MockEmbeddingRouter()
        )

        plan = handler.analyze_query("in my voice notes about meetings")

        assert plan.retrieval_mode == RetrievalMode.SINGLE_MODALITY
        assert ContentSource.AUDIO_TRANSCRIPTION in plan.target_modalities

    def test_plan_generation_no_hint(self):
        """Test query plan without modality hint."""
        handler = MultimodalQueryHandler(
            hint_detector=ModalityHintDetector(),
            pkg_client=MockPKGClient(),
            embedding_router=MockEmbeddingRouter()
        )

        plan = handler.analyze_query("project deadlines")

        assert plan.retrieval_mode == RetrievalMode.ALL_SOURCES
        assert len(plan.target_modalities) == len(ContentSource)


class TestOCRContentProcessor:
    """Tests for OCR content processing."""

    def test_ocr_result_processing(self):
        """Test processing of OCR output."""
        processor = OCRContentProcessor()

        ocr_output = {
            "text": "Invoice #12345",
            "confidence": 0.92,
            "source_file": "invoice.pdf",
            "layout": {"is_form": True}
        }

        result = processor.process_ocr_result(ocr_output)

        assert result["content"] == "Invoice #12345"
        assert result["source_metadata"]["source_type"] == ContentSource.OCR_DOCUMENT.value
        assert result["source_metadata"]["extraction_confidence"] == 0.92

    def test_fuzzy_variant_generation(self):
        """Test fuzzy variant generation for OCR errors."""
        processor = OCRContentProcessor()

        variants = processor._generate_fuzzy_variants("hello world")

        # Should generate variants for common OCR errors
        assert len(variants) > 0


class TestTranscriptionProcessor:
    """Tests for transcription processing."""

    def test_transcription_processing(self):
        """Test processing of Whisper output."""
        processor = TranscriptionProcessor()

        whisper_output = {
            "text": "Hello, this is a test recording.",
            "segments": [
                {"start": 0, "end": 2, "text": "Hello,", "confidence": 0.95},
                {"start": 2, "end": 5, "text": "this is a test recording.", "confidence": 0.92}
            ],
            "language": "en"
        }

        result = processor.process_transcription(whisper_output)

        assert "Hello" in result["content"]
        assert result["source_metadata"]["source_type"] == ContentSource.AUDIO_TRANSCRIPTION.value

    def test_homophone_expansion(self):
        """Test query expansion for homophones."""
        processor = TranscriptionProcessor()

        expanded = processor.expand_query_for_transcription("their meeting notes")

        assert "there" in expanded or "(there)" in expanded
        assert "they're" in expanded or "(they're)" in expanded
```

### Integration Tests

```python
class TestMultimodalIntegration:
    """Integration tests for multimodal search."""

    @pytest.mark.integration
    async def test_ocr_search_end_to_end(self):
        """Test OCR content search end-to-end."""
        api = create_hybrid_search_api()

        # Index OCR content
        ocr_content = {
            "text": "Project Budget Report Q4 2024",
            "confidence": 0.95,
            "source_file": "budget.pdf"
        }
        await api.index_ocr_content(ocr_content)

        # Search with OCR hint
        results = await api.search(
            "budget report from the PDF",
            top_k=5
        )

        assert len(results) > 0
        assert "budget" in results[0]["content"].lower()

    @pytest.mark.integration
    async def test_transcription_search_end_to_end(self):
        """Test audio transcription search end-to-end."""
        api = create_hybrid_search_api()

        # Index transcription
        transcription = {
            "text": "We discussed the quarterly targets in the meeting.",
            "segments": [{"start": 0, "end": 5, "text": "We discussed the quarterly targets in the meeting."}]
        }
        await api.index_transcription(transcription)

        # Search with audio hint
        results = await api.search(
            "what did we discuss in the meeting about targets",
            top_k=5
        )

        assert len(results) > 0
        assert "quarterly targets" in results[0]["content"].lower()

    @pytest.mark.integration
    async def test_cross_modal_fusion(self):
        """Test cross-modal result fusion."""
        api = create_hybrid_search_api()

        # Index content from multiple modalities
        await api.index_text("Meeting notes: Project deadline is Friday")
        await api.index_ocr_content({"text": "Project Schedule: Friday deadline", "confidence": 0.9})
        await api.index_transcription({"text": "The project deadline is on Friday"})

        # Search without modality hint
        results = await api.search("project deadline", top_k=10)

        # Should have results from multiple modalities
        modalities = set(r.get("source_type") for r in results)
        assert len(modalities) >= 2  # At least 2 different sources
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| OCR Content Relevance | >80% | Precision@5 for OCR-specific queries |
| Audio Content Relevance | >75% | Precision@5 for audio-specific queries |
| Modality Hint Accuracy | >90% | Correct modality detection rate |
| Cross-Modal Fusion Quality | >0.7 MRR | MRR for mixed-source queries |
| Query Latency | <1.2s | P95 latency for multimodal queries |

---

## Dependencies

### Internal Dependencies
- ModalityHintDetector: Query analysis
- MultimodalQueryHandler: Execution orchestration
- OCRContentProcessor: DeepSeek-OCR integration
- TranscriptionProcessor: Whisper V3 integration
- CrossModalFusion: Result combination

### External Dependencies
- DeepSeek-OCR output format
- Whisper V3 output format
- PKGClient for storage/retrieval
- QueryEmbeddingRouter for embeddings

---

## Option B Compliance

- **Ghost Model Frozen**: OCR/Whisper models used for extraction only
- **Local-First Processing**: All multimodal processing runs on-device
- **Quality Gates**: Relevance targets validated per modality
- **Schema Evolution**: Source metadata schema supports version evolution

---

**This module enables comprehensive multimodal search across all ingested content types.**
