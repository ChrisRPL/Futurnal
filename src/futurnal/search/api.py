"""Hybrid Search API Factory.

Provides unified HybridSearchAPI class and factory function for integration testing.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/06-integration-testing.md

Option B Compliance:
- Ghost model frozen (Ollama inference only)
- Experiential learning via quality feedback
- Temporal-first design
- Local-first processing
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from futurnal.search.config import SearchConfig
from futurnal.search.hybrid import (
    HybridSearchConfig,
    QueryRouter,
    SchemaAwareRetrieval,
    SearchQualityFeedback,
    QueryTemplateDatabase,
)
from futurnal.search.hybrid.performance import (
    MultiLayerCache,
    CacheLayer,
    PerformanceProfiler,
)
from futurnal.search.audit_events import (
    log_search_executed,
    log_search_failed,
    log_content_indexed,
    log_multimodal_search_executed,
    log_ocr_content_indexed,
    log_transcription_indexed,
    log_feedback_recorded,
)

if TYPE_CHECKING:
    from futurnal.pkg.client import PKGClient
    from futurnal.privacy.audit import AuditLogger
    from futurnal.embeddings.service import MultiModelEmbeddingService
    from futurnal.search.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result."""

    id: str
    content: str
    score: float
    confidence: float = 1.0
    timestamp: Optional[str] = None
    entity_type: Optional[str] = None
    source_type: Optional[str] = None
    causal_chain: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "entity_type": self.entity_type,
            "source_type": self.source_type,
            "causal_chain": self.causal_chain,
            **self.metadata,
        }


class HybridSearchAPI:
    """Unified Hybrid Search API.

    Wraps all search modules for end-to-end integration:
    - Module 01: Temporal Query Engine
    - Module 02: Causal Chain Retrieval
    - Module 03: Schema-Aware Retrieval
    - Module 04: Query Routing & Orchestration
    - Module 05: Performance & Caching
    - Module 07: Multimodal Query Handling

    Example:
        >>> api = create_hybrid_search_api()
        >>> results = await api.search("What happened yesterday?", top_k=10)
    """

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        pkg_client: Optional["PKGClient"] = None,
        pkg_manager: Optional[Any] = None,
        multimodal_enabled: bool = False,
        experiential_learning: bool = False,
        caching_enabled: bool = True,
        audit_logger: Optional["AuditLogger"] = None,
        embedding_service: Optional["MultiModelEmbeddingService"] = None,
        graphrag_enabled: bool = True,
    ):
        """Initialize HybridSearchAPI.

        Args:
            config: Search configuration
            pkg_client: PKG client (driver) for graph queries (legacy)
            pkg_manager: PKGDatabaseManager for graph queries (preferred)
            multimodal_enabled: Enable multimodal content search
            experiential_learning: Enable GRPO feedback recording
            caching_enabled: Enable multi-layer caching
            audit_logger: Optional audit logger for search event tracking
            embedding_service: Optional embedding service for GraphRAG
            graphrag_enabled: Enable GraphRAG pipeline (requires embedding_service)
        """
        self.config = config or SearchConfig()
        self.pkg = pkg_client
        self._pkg_manager = pkg_manager
        self.multimodal_enabled = multimodal_enabled
        self.experiential_learning = experiential_learning
        self.caching_enabled = caching_enabled
        self._audit = audit_logger
        self._embedding_service = embedding_service
        self._graphrag_enabled = graphrag_enabled

        # Initialize components
        self._init_components()

        # Answer generation (Step 02)
        self._answer_generator: Optional["AnswerGenerator"] = None
        self._answer_generation_enabled: bool = True

        # Tracking for tests
        self.last_strategy: Optional[str] = None
        self.last_embedding_model: Optional[str] = None

    def _init_components(self) -> None:
        """Initialize search components."""
        # Module 05: Caching
        if self.caching_enabled:
            self.cache = MultiLayerCache()
        else:
            self.cache = None

        # Module 04: Query routing
        self.router: Optional[QueryRouter] = None
        self._init_router()

        # Module 03: Schema-aware retrieval (GraphRAG)
        self.schema_retrieval: Optional[SchemaAwareRetrieval] = None
        if self._graphrag_enabled:
            self._init_schema_aware_retrieval()

        # Experiential learning
        if self.experiential_learning:
            self.quality_feedback = SearchQualityFeedback()
            self.template_database = QueryTemplateDatabase()
        else:
            self.quality_feedback = None
            self.template_database = None

        # Performance profiling
        self.profiler = PerformanceProfiler()

        # Multimodal handler
        self.multimodal_handler = None
        self.ocr_processor = None
        self.transcription_processor = None
        if self.multimodal_enabled:
            self._init_multimodal()

        # Schema manager stub
        self.schema_manager = None

    def _init_multimodal(self) -> None:
        """Initialize multimodal components."""
        try:
            from futurnal.search.hybrid.multimodal import (
                MultimodalQueryHandler,
                OCRContentProcessor,
                TranscriptionProcessor,
            )

            self.multimodal_handler = MultimodalQueryHandler(
                pkg_client=self.pkg,
            )
            self.ocr_processor = OCRContentProcessor()
            self.transcription_processor = TranscriptionProcessor()
            logger.info("Multimodal search components initialized")
        except Exception as e:
            logger.warning(f"Could not init multimodal components: {e}")
            self.multimodal_handler = None

    def _init_router(self) -> None:
        """Initialize query router with available backends."""
        try:
            self.router = QueryRouter()
        except Exception as e:
            logger.warning(f"Could not init QueryRouter: {e}")
            self.router = None

    def _init_schema_aware_retrieval(self) -> None:
        """Initialize SchemaAwareRetrieval for GraphRAG pipeline.

        Per GFM-RAG paper (2502.01113v1):
        - Vector search via ChromaDB (SchemaVersionedEmbeddingStore)
        - Graph traversal via Neo4j (TemporalGraphQueries)
        - Intent-based embedding via QueryEmbeddingRouter

        Gracefully falls back to None if dependencies unavailable.
        """
        try:
            # Import dependencies for GraphRAG
            from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore
            from futurnal.embeddings.integration import TemporalAwareVectorWriter
            from futurnal.search.hybrid.query_router import QueryEmbeddingRouter
            from futurnal.search.temporal.engine import TemporalQueryEngine
            from futurnal.search.causal.retrieval import CausalChainRetrieval

            # Check if embedding service is available
            if self._embedding_service is None:
                logger.info(
                    "GraphRAG: No embedding service provided, "
                    "attempting lazy initialization"
                )
                # Attempt lazy initialization of embedding service
                try:
                    from futurnal.embeddings.service import MultiModelEmbeddingService
                    from futurnal.embeddings.config import EmbeddingServiceConfig

                    self._embedding_service = MultiModelEmbeddingService(
                        config=EmbeddingServiceConfig()
                    )
                    logger.info("GraphRAG: Lazily initialized embedding service")
                except Exception as e:
                    logger.warning(f"GraphRAG: Could not init embedding service: {e}")
                    return

            # Initialize schema-versioned embedding store (ChromaDB wrapper)
            try:
                embedding_store = SchemaVersionedEmbeddingStore(
                    config=self._embedding_service.config,
                )
                logger.debug("GraphRAG: Initialized SchemaVersionedEmbeddingStore")
            except Exception as e:
                logger.warning(f"GraphRAG: Could not init embedding store: {e}")
                return

            # Initialize query embedding router
            try:
                query_router = QueryEmbeddingRouter(
                    embedding_service=self._embedding_service,
                )
                logger.debug("GraphRAG: Initialized QueryEmbeddingRouter")
            except Exception as e:
                logger.warning(f"GraphRAG: Could not init query router: {e}")
                query_router = None

            # Initialize temporal engine if PKG client available
            temporal_engine = None
            causal_retrieval = None
            pkg_queries = None

            if self._pkg_manager is not None:
                try:
                    from futurnal.pkg.queries.temporal import TemporalGraphQueries

                    pkg_queries = TemporalGraphQueries(db_manager=self._pkg_manager)

                    temporal_engine = TemporalQueryEngine(
                        pkg_queries=pkg_queries,
                    )
                    logger.debug("GraphRAG: Initialized TemporalQueryEngine")

                    causal_retrieval = CausalChainRetrieval(
                        pkg_queries=pkg_queries,
                    )
                    logger.debug("GraphRAG: Initialized CausalChainRetrieval")
                except Exception as e:
                    logger.warning(f"GraphRAG: Could not init PKG queries: {e}")

            # Initialize SchemaAwareRetrieval with all dependencies
            # pkg_queries is required - if not available, we can't use SchemaAwareRetrieval
            if pkg_queries is None:
                logger.warning(
                    "GraphRAG: pkg_queries not available, cannot initialize SchemaAwareRetrieval"
                )
                return

            self.schema_retrieval = SchemaAwareRetrieval(
                pkg_queries=pkg_queries,
                embedding_store=embedding_store,
                embedding_router=query_router,
                temporal_engine=temporal_engine,
                causal_retrieval=causal_retrieval,
                config=HybridSearchConfig(),
                audit_logger=self._audit,
            )

            logger.info(
                f"GraphRAG: Initialized SchemaAwareRetrieval "
                f"(temporal={temporal_engine is not None}, "
                f"causal={causal_retrieval is not None})"
            )

        except Exception as e:
            logger.warning(f"GraphRAG: Could not initialize SchemaAwareRetrieval: {e}")
            self.schema_retrieval = None

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute hybrid search.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional search filters

        Returns:
            List of search results with metadata
        """
        start_time = time.time()
        intent_type = "exploratory"  # Default
        cache_hit = False
        fallback_used = False

        # Check cache
        if self.cache:
            cached, hit = self.cache.get(CacheLayer.QUERY_RESULT, query)
            if hit:
                cache_hit = True
                # Audit log cache hit
                if self._audit:
                    latency_ms = (time.time() - start_time) * 1000
                    log_search_executed(
                        self._audit,
                        search_type="hybrid",
                        intent=intent_type,
                        result_count=len(cached) if cached else 0,
                        latency_ms=latency_ms,
                        cache_hit=True,
                        filters_applied=filters,
                    )
                return cached

        results: List[Dict[str, Any]] = []

        try:
            # Check for multimodal hints first
            if self.multimodal_handler:
                modality_summary = self.multimodal_handler.get_modality_summary(query)
                if modality_summary:
                    # Has modality hints - use multimodal search
                    results = await self._multimodal_search(query, top_k, filters)
                    self.last_strategy = "multimodal"
                    intent_type = "multimodal"
                    if results:
                        # Cache and return multimodal results
                        if self.cache:
                            self.cache.set(CacheLayer.QUERY_RESULT, query, results)
                        # Audit log multimodal search
                        if self._audit:
                            latency_ms = (time.time() - start_time) * 1000
                            log_multimodal_search_executed(
                                self._audit,
                                modalities=list(modality_summary.keys()) if isinstance(modality_summary, dict) else ["text"],
                                result_count=len(results),
                                latency_ms=latency_ms,
                            )
                        return results

            # Phase 2.5 P0 Fix: Check relationship intent FIRST (before router)
            # Relationship queries have explicit patterns that LLM routing may miss
            simple_intent = self._detect_simple_intent(query)
            if simple_intent == "relationship":
                logger.info(f"Relationship query detected, bypassing router")
                intent_type = "relationship"
                self.last_strategy = intent_type
                results = await self._relationship_search(query, top_k, filters)
            elif self.router:
                # Standard routing for non-relationship queries
                plan = self.router.route_query(query)
                intent_type = plan.intent.value if plan and plan.intent else "exploratory"
                self.last_strategy = intent_type

                # Execute based on intent
                results = await self._execute_by_intent(query, plan, top_k, filters)
            else:
                # Fallback with simple intent detection
                intent_type = simple_intent  # Reuse already-detected intent
                self.last_strategy = intent_type
                results = await self._execute_by_intent_type(query, intent_type, top_k, filters)

        except Exception as e:
            logger.error(f"Search error: {e}")
            # Audit log failure
            if self._audit:
                latency_ms = (time.time() - start_time) * 1000
                log_search_failed(
                    self._audit,
                    search_type="hybrid",
                    intent=intent_type,
                    error_type="search_exception",
                    latency_ms=latency_ms,
                    fallback_attempted=True,
                )
            # Fallback search
            fallback_used = True
            results = await self._basic_search(query, top_k)

        # Cache results
        if self.cache and results:
            self.cache.set(CacheLayer.QUERY_RESULT, query, results)

        # Record latency with proper signature
        latency_ms = (time.time() - start_time) * 1000
        try:
            self.profiler.record_query(
                query_id=query[:50],
                total_ms=latency_ms,
                components={"search": latency_ms},
                strategy="hybrid",
                intent_type=intent_type,
            )
        except Exception:
            pass  # Profiler recording is optional

        # Audit log successful search (query content is NOT logged)
        if self._audit:
            log_search_executed(
                self._audit,
                search_type="hybrid",
                intent=intent_type,
                result_count=len(results),
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                fallback_used=fallback_used,
                filters_applied=filters,
            )

        return results

    async def _get_answer_generator(self) -> "AnswerGenerator":
        """Lazy initialization of answer generator.

        Returns:
            Initialized AnswerGenerator instance
        """
        if self._answer_generator is None:
            from futurnal.search.answer_generator import AnswerGenerator

            self._answer_generator = AnswerGenerator()
            await self._answer_generator.initialize()
            logger.info("Answer generator initialized")
        return self._answer_generator

    async def search_with_answer(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        generate_answer: bool = True,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search with optional LLM answer generation.

        Combines GraphRAG retrieval with LLM-powered answer synthesis.

        Research Foundation:
        - CausalRAG (ACL 2025): Causal-aware generation
        - LLM-Enhanced Symbolic (2501.01246v1): Hybrid approach

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional search filters
            generate_answer: Whether to generate synthesized answer
            model: Optional model override for answer generation

        Returns:
            {
                'answer': str,           # Synthesized answer (if generate_answer=True)
                'results': List[Dict],   # Raw search results
                'sources': List[str],    # Source documents used
                'intent': Dict,          # Query intent
                'execution_time_ms': float,
            }
        """
        start = time.time()

        # Step 1: Execute search (existing GraphRAG pipeline)
        results = await self.search(query, top_k=top_k, filters=filters)

        response: Dict[str, Any] = {
            "results": results,
            "sources": [
                r.get("metadata", {}).get("source")
                for r in results
                if r.get("metadata", {}).get("source")
            ],
            "intent": {
                "primary": self.last_strategy or "exploratory",
            },
            "answer": None,
        }

        # Step 2: Generate answer if requested and results exist
        if generate_answer and results and self._answer_generation_enabled:
            try:
                generator = await self._get_answer_generator()

                # Collect graph context from results
                graph_context = self._aggregate_graph_context(results)

                generated = await generator.generate_answer(
                    query=query,
                    context=results,
                    graph_context=graph_context,
                    model=model,
                )
                response["answer"] = generated.answer
                response["sources"] = generated.sources
                response["model"] = generated.model
                response["generation_time_ms"] = generated.generation_time_ms

                logger.info(
                    "Search with answer completed: %d results, answer generated",
                    len(results),
                )

            except Exception as e:
                logger.warning(f"Answer generation failed: {e}")
                # Graceful degradation - return results without answer

        response["execution_time_ms"] = (time.time() - start) * 1000
        return response

    def _aggregate_graph_context(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate graph context from all results.

        Collects relationships and entities from GraphRAG results
        for use in answer generation.

        Args:
            results: Search results with potential graph_context

        Returns:
            Aggregated graph context with relationships and entities
        """
        relationships: List[Dict[str, Any]] = []
        related_entities: List[Dict[str, Any]] = []

        for r in results:
            gc = r.get("graph_context")
            if gc:
                relationships.extend(gc.get("relationships", []))
                related_entities.extend(gc.get("related_entities", []))

        return {
            "relationships": relationships[:10],  # Limit for prompt size
            "related_entities": related_entities[:10],
        }

    def _detect_simple_intent(self, query: str) -> str:
        """Simple rule-based intent detection fallback.

        Phase 2.5 P0 Fix: Added relationship intent detection for
        "How is X connected to Y?" queries.
        """
        import re

        query_lower = query.lower()

        # Phase 2.5 P0 Fix: Check relationship patterns FIRST
        # These are queries asking about connections between entities
        # IMPORTANT: Patterns must handle various phrasings:
        # - "How is X connected to Y?"
        # - "How are X related with Y?"  (note: "with" not just "to")
        # - "How my X are related to Y?" (note: "my" before subject)
        relationship_patterns = [
            r'\bconnected\s+(to|with)\b',         # "connected to" or "connected with"
            r'\brelated\s+(to|with)\b',           # "related to" or "related with"
            r'\blinks?\s+(between|to)\b',
            r'\brelationship\s+(between|with)\b',
            r'\bhow\s+.*\s+(connected|related)\b',  # More flexible: "how ... connected/related"
            r'\bwhat\s+(connects|links)\b',
            r'\bare\s+.*\s+related\b',            # "are X related" anywhere
            r'\bare\s+.*\s+connected\b',          # "are X connected" anywhere
        ]

        for pattern in relationship_patterns:
            if re.search(pattern, query_lower):
                return "relationship"

        if any(w in query_lower for w in ["when", "yesterday", "last week", "today", "between"]):
            return "temporal"
        elif any(w in query_lower for w in ["why", "cause", "led to", "because", "reason"]):
            return "causal"
        elif any(w in query_lower for w in ["def ", "function", "class ", "code", "module"]):
            return "code"
        elif any(w in query_lower for w in ["what is", "who is", "where is"]):
            return "factual"
        else:
            return "exploratory"

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entity/document names from relationship query.

        Phase 2.5 P0 Fix: Extracts the two entities being asked about
        in a relationship query.

        Examples:
            "How are MMM subtasks connected to Literature papers?" -> ["MMM subtasks", "Literature papers"]
            "How my subtasks from MMM project are related with literature papers?" -> ["subtasks from MMM project", "literature papers"]
        """
        import re

        query_lower = query.lower()
        entities = []

        # Pattern 1: "How is/are X connected/related to/with Y"
        pattern = r'how\s+(?:is|are)\s+(.+?)\s+(?:connected|related)\s+(?:to|with)\s+(.+?)(?:\?|$)'
        match = re.search(pattern, query_lower)
        if match:
            entities = [match.group(1).strip(), match.group(2).strip()]
            return self._clean_entity_names(entities)

        # Pattern 2: "How my X are connected/related to/with Y" (handles "my" before subject)
        pattern = r'how\s+(?:my|the)?\s*(.+?)\s+(?:is|are)\s+(?:connected|related)\s+(?:to|with)\s+(.+?)(?:\?|$)'
        match = re.search(pattern, query_lower)
        if match:
            entities = [match.group(1).strip(), match.group(2).strip()]
            return self._clean_entity_names(entities)

        # Pattern 3: "relationship between X and Y"
        pattern = r'relationship\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)'
        match = re.search(pattern, query_lower)
        if match:
            return self._clean_entity_names([match.group(1).strip(), match.group(2).strip()])

        # Pattern 4: "X connected/related to/with Y" (without "how")
        pattern = r'(.+?)\s+(?:connected|related)\s+(?:to|with)\s+(.+?)(?:\?|$)'
        match = re.search(pattern, query_lower)
        if match:
            return self._clean_entity_names([match.group(1).strip(), match.group(2).strip()])

        # Fallback: split on "to" or "with"
        for separator in [" to ", " with "]:
            if separator in query_lower:
                parts = query_lower.split(separator, 1)
                if len(parts) == 2:
                    # Extract last noun phrase from first part
                    first_words = parts[0].split()
                    first = first_words[-3:] if len(first_words) >= 3 else first_words
                    second = parts[1].split()[:3]  # First 3 words after separator
                    return self._clean_entity_names([" ".join(first), " ".join(second).rstrip("?")])

        return entities

    def _clean_entity_names(self, entities: List[str]) -> List[str]:
        """Clean up extracted entity names by removing common prefixes."""
        import re
        cleaned = []
        for entity in entities:
            # Remove common prefixes
            entity = re.sub(r'^(my|the|a|an|your|their)\s+', '', entity)
            # Remove trailing punctuation
            entity = entity.rstrip('?.!')
            cleaned.append(entity.strip())
        return cleaned

    async def _graphrag_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        intent: str = "exploratory",
    ) -> List[Dict[str, Any]]:
        """Execute GraphRAG hybrid search per GFM-RAG paper.

        Pipeline:
        1. Semantic retrieval (ChromaDB via SchemaVersionedEmbeddingStore)
        2. Graph traversal (Neo4j via TemporalGraphQueries)
        3. Context fusion and re-ranking

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional search filters
            intent: Search intent (temporal, causal, exploratory, lookup)

        Returns:
            List of search results with graph context

        Raises:
            Falls back to legacy search on any error
        """
        if self.schema_retrieval is None:
            logger.debug("GraphRAG: SchemaAwareRetrieval not available, using legacy search")
            return []

        try:
            # Execute hybrid search via SchemaAwareRetrieval
            # This combines vector similarity + graph expansion
            hybrid_results = self.schema_retrieval.hybrid_search(
                query=query,
                intent=intent,
                top_k=top_k,
            )

            # Convert HybridSearchResult to API response format
            results: List[Dict[str, Any]] = []
            for r in hybrid_results:
                # Use source_type from metadata, default to "text"
                source_type = r.metadata.get("source_type", "text") if r.metadata else "text"
                result_dict: Dict[str, Any] = {
                    "id": r.entity_id,
                    "content": r.content,
                    "score": r.combined_score,
                    "confidence": r.vector_score,  # Vector similarity as confidence
                    "timestamp": r.metadata.get("timestamp") if r.metadata else None,
                    "entity_type": r.entity_type,
                    "source_type": source_type,
                    "metadata": {
                        "vector_score": r.vector_score,
                        "graph_score": r.graph_score,
                        "schema_version": r.schema_version,
                        "retrieval_source": r.source,  # "vector"/"graph"/"hybrid" - retrieval strategy
                        "graph_enhanced": True,  # Flag for frontend to show graph badge
                        **(r.metadata if r.metadata else {}),  # Preserve actual document source
                    },
                }

                # Include graph context if available (serialize Pydantic model)
                if hasattr(r, 'graph_context') and r.graph_context:
                    if hasattr(r.graph_context, 'model_dump'):
                        result_dict["graph_context"] = r.graph_context.model_dump()
                    elif hasattr(r.graph_context, 'dict'):
                        result_dict["graph_context"] = r.graph_context.dict()
                    else:
                        result_dict["graph_context"] = str(r.graph_context)

                results.append(result_dict)

            logger.info(
                f"GraphRAG search completed: {len(results)} results "
                f"(intent={intent}, query_len={len(query)})"
            )

            return results

        except Exception as e:
            logger.warning(f"GraphRAG search failed: {e}, falling back to legacy")
            return []

    async def _execute_by_intent_type(
        self,
        query: str,
        intent_type: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute query based on intent type string."""
        if intent_type == "temporal":
            return await self._temporal_search(query, top_k, filters)
        elif intent_type == "causal":
            return await self._causal_search(query, top_k, filters)
        elif intent_type == "relationship":
            # Phase 2.5 P0 Fix: Relationship query handling
            return await self._relationship_search(query, top_k, filters)
        elif intent_type == "code":
            self.last_embedding_model = "codebert"
            return await self._code_search(query, top_k, filters)
        else:
            return await self._general_search(query, top_k, filters)

    async def _execute_by_intent(
        self,
        query: str,
        plan: Any,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute query based on classified intent from QueryPlan."""
        # QueryPlan has .intent which is a QueryIntent enum
        intent_type = plan.intent.value if plan and hasattr(plan, 'intent') and plan.intent else "exploratory"

        # Phase 2.5 P0 Fix: Also check for relationship intent via simple detection
        # QueryRouter may not detect relationship intent, so we double-check
        if intent_type == "exploratory":
            simple_intent = self._detect_simple_intent(query)
            if simple_intent == "relationship":
                intent_type = "relationship"

        if intent_type == "temporal":
            return await self._temporal_search(query, top_k, filters)
        elif intent_type == "causal":
            return await self._causal_search(query, top_k, filters)
        elif intent_type == "relationship":
            return await self._relationship_search(query, top_k, filters)
        elif intent_type == "code":
            self.last_embedding_model = "codebert"
            return await self._code_search(query, top_k, filters)
        else:
            return await self._general_search(query, top_k, filters)

    async def _temporal_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute temporal-aware search using GraphRAG or legacy fallback.

        Per GFM-RAG paper:
        1. Try GraphRAG with temporal intent (uses TemporalQueryEngine)
        2. Fall back to legacy temporal search if GraphRAG unavailable

        Args:
            query: Natural language query with temporal expressions
            top_k: Number of results to return
            filters: Optional search filters

        Returns:
            List of temporally-filtered search results
        """
        # Try GraphRAG with temporal intent first
        graphrag_results = await self._graphrag_search(query, top_k, filters, intent="temporal")
        if graphrag_results:
            self.last_embedding_model = "graphrag-temporal"
            return graphrag_results

        # Fall back to legacy temporal search
        logger.debug("GraphRAG returned no results, falling back to legacy temporal search")
        return await self._legacy_temporal_search(query, top_k, filters)

    async def _legacy_temporal_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Legacy temporal-aware search - fallback when GraphRAG unavailable.

        Parses temporal expressions like "last week" and filters documents
        by their timestamps within the specified time range.
        """
        from datetime import datetime, timedelta
        from futurnal.extraction.temporal.markers import TemporalMarkerExtractor

        workspace = Path.home() / ".futurnal" / "workspace"
        parsed_dir = workspace / "parsed"
        imap_dir = workspace / "imap"

        # Extract temporal markers from query
        extractor = TemporalMarkerExtractor()
        markers = extractor.extract_temporal_markers(query)

        # Determine date range from temporal markers
        start_date = None
        end_date = datetime.now()

        if markers:
            ref_time = markers[0].timestamp
            query_lower = query.lower()

            if "week" in query_lower:
                start_date = ref_time - timedelta(days=3)
                end_date = ref_time + timedelta(days=4)
            elif "month" in query_lower:
                start_date = ref_time - timedelta(days=15)
                end_date = ref_time + timedelta(days=16)
            elif "year" in query_lower:
                start_date = ref_time - timedelta(days=180)
                end_date = ref_time + timedelta(days=180)
            else:
                start_date = ref_time - timedelta(days=1)
                end_date = ref_time + timedelta(days=2)

        results: List[Dict[str, Any]] = []
        temporal_words = {"yesterday", "today", "tomorrow", "last", "next", "this", "week", "month", "year", "ago"}
        query_terms = [w for w in query.lower().split() if w not in temporal_words]

        # Search parsed documents
        if parsed_dir.exists():
            for json_file in parsed_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                    text = data.get("text", "") or data.get("content", "")

                    doc_timestamp = data.get("ingested_at")
                    if doc_timestamp and start_date:
                        try:
                            if isinstance(doc_timestamp, str):
                                doc_time = datetime.fromisoformat(doc_timestamp.replace("Z", "+00:00"))
                                if doc_time.tzinfo:
                                    doc_time = doc_time.replace(tzinfo=None)
                            else:
                                doc_time = doc_timestamp

                            if not (start_date <= doc_time <= end_date):
                                continue
                        except Exception:
                            pass

                    score = 0.5
                    if query_terms:
                        text_lower = text.lower()
                        matches = sum(1 for term in query_terms if term in text_lower)
                        score += (matches / len(query_terms)) * 0.5

                    if score > 0.4:
                        metadata = data.get("metadata", {})
                        snippet = text[:300] + "..." if len(text) > 300 else text

                        results.append({
                            "id": data.get("element_id", json_file.stem),
                            "content": snippet,
                            "score": score,
                            "confidence": 0.8,
                            "timestamp": doc_timestamp,
                            "entity_type": "Event" if "event" in text.lower() else "Document",
                            "source_type": "text",
                            "metadata": {
                                "source": metadata.get("source", ""),
                                "path": metadata.get("path"),
                                "temporal_match": True,
                            },
                        })
                except Exception as e:
                    logger.debug(f"Error reading {json_file}: {e}")
                    continue

        # Search IMAP emails
        if imap_dir.exists():
            for json_file in imap_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                    content = data.get("content", "")
                    metadata = data.get("metadata", {})

                    doc_date = metadata.get("date")
                    if doc_date and start_date:
                        try:
                            doc_time = datetime.fromisoformat(doc_date.replace("Z", "+00:00"))
                            if doc_time.tzinfo:
                                doc_time = doc_time.replace(tzinfo=None)
                            if not (start_date <= doc_time <= end_date):
                                continue
                        except Exception:
                            pass

                    score = 0.5
                    if query_terms:
                        text_lower = content.lower()
                        matches = sum(1 for term in query_terms if term in text_lower)
                        score += (matches / len(query_terms)) * 0.5

                    if score > 0.4:
                        snippet = content[:300] + "..." if len(content) > 300 else content
                        results.append({
                            "id": data.get("sha256", json_file.stem),
                            "content": snippet,
                            "score": score,
                            "confidence": 0.8,
                            "timestamp": doc_date,
                            "entity_type": "Email",
                            "source_type": "text",
                            "metadata": {"source": metadata.get("source", ""), "subject": metadata.get("subject", ""), "temporal_match": True},
                        })
                except Exception as e:
                    logger.debug(f"Error reading {json_file}: {e}")
                    continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def _causal_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute causal chain search using GraphRAG.

        Per CausalRAG paper (ACL 2025):
        Uses CausalChainRetrieval to find cause-effect relationships
        in the personal knowledge graph.

        Args:
            query: Natural language query about causality
            top_k: Number of results to return
            filters: Optional search filters

        Returns:
            List of results with causal chain information
        """
        # Try GraphRAG with causal intent
        graphrag_results = await self._graphrag_search(query, top_k, filters, intent="causal")
        if graphrag_results:
            self.last_embedding_model = "graphrag-causal"
            return graphrag_results

        # Fallback: Return empty results with system message when GraphRAG unavailable
        logger.warning("GraphRAG causal search unavailable - returning system notification")
        return [{
            "id": "causal_search_unavailable",
            "content": (
                "Causal analysis requires the knowledge graph to be connected. "
                "Please ensure Neo4j is running and the PKG database is initialized. "
                "You can check status with 'futurnal health check'."
            ),
            "score": 0.0,
            "confidence": 0.0,
            "entity_type": "SystemMessage",
            "source_type": "system",
            "metadata": {
                "is_system_message": True,
                "action_required": "Connect to PKG database",
            },
        }]

    async def _relationship_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute relationship query search.

        Phase 2.5 P0 Fix: Handles "How is X connected to Y?" queries
        by traversing the knowledge graph WITHOUT temporal filtering.

        Uses query_entity_connections() to find all paths between entities.

        Args:
            query: Natural language query about relationships
            top_k: Number of results to return
            filters: Optional search filters

        Returns:
            List of results showing relationship paths
        """
        # Extract entities from query
        entities = self._extract_entities_from_query(query)

        if len(entities) < 2:
            logger.warning(f"Could not extract 2 entities from query: {query}")
            # Fall back to general search
            return await self._general_search(query, top_k, filters)

        entity_a, entity_b = entities[0], entities[1]
        logger.info(f"Relationship search: '{entity_a}' <-> '{entity_b}'")

        # Try to use PKG queries if available
        if self._pkg_manager is not None:
            try:
                from futurnal.pkg.queries.temporal import TemporalGraphQueries

                pkg_queries = TemporalGraphQueries(db_manager=self._pkg_manager)

                # Query entity connections (NO temporal filtering)
                connections = pkg_queries.query_entity_connections(
                    entity_a=entity_a,
                    entity_b=entity_b,
                    max_hops=3,
                    limit=top_k * 2,  # Get more, then dedupe
                )

                if connections:
                    return self._format_relationship_results(
                        connections=connections,
                        entity_a=entity_a,
                        entity_b=entity_b,
                        query=query,
                        top_k=top_k,
                    )
                else:
                    logger.info(f"No direct connections found between '{entity_a}' and '{entity_b}'")

            except Exception as e:
                logger.warning(f"Relationship search via PKG failed: {e}")

        # Fallback to GraphRAG with exploratory intent
        logger.debug("Falling back to GraphRAG for relationship query")
        graphrag_results = await self._graphrag_search(query, top_k, filters, intent="exploratory")
        if graphrag_results:
            return graphrag_results

        # Final fallback to keyword search
        return await self._legacy_keyword_search(query, top_k, filters)

    def _format_relationship_results(
        self,
        connections: List[Dict[str, Any]],
        entity_a: str,
        entity_b: str,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Format relationship query results for display.

        Converts graph path data into user-friendly results.
        """
        results: List[Dict[str, Any]] = []
        seen_paths = set()

        for conn in connections:
            path = conn.get("path", [])
            rels = conn.get("relationships", [])
            hops = conn.get("total_hops", len(path) - 1)

            # Create path description
            path_names = [node.get("name", "?") for node in path]
            path_str = " → ".join(path_names)

            # Skip duplicates
            if path_str in seen_paths:
                continue
            seen_paths.add(path_str)

            # Calculate score based on path length (shorter = better)
            score = 1.0 / (1 + hops * 0.2)

            # Build content description
            content_parts = [f"**Connection Path**: {path_str}"]
            if rels:
                content_parts.append(f"**Relationships**: {' → '.join(rels)}")
            content_parts.append(f"**Hops**: {hops}")

            # Add node details
            for i, node in enumerate(path):
                node_name = node.get("name", "Unknown")
                node_labels = node.get("labels", [])
                node_type = node_labels[0] if node_labels else "Node"
                content_parts.append(f"  {i+1}. [{node_type}] {node_name}")

            results.append({
                "id": f"path_{hash(path_str)}",
                "content": "\n".join(content_parts),
                "score": score,
                "confidence": 0.9,
                "entity_type": "RelationshipPath",
                "source_type": "graph",
                "metadata": {
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "path": path_names,
                    "relationships": rels,
                    "hops": hops,
                    "is_relationship_result": True,
                },
            })

        # Sort by score (shorter paths first)
        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Formatted {len(results)} relationship results")
        return results[:top_k]

    async def _code_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute code-specific search.

        Uses GraphRAG with code intent for semantic code search.
        Falls back to legacy keyword search on code files if GraphRAG unavailable.
        """
        # Try GraphRAG with code intent
        graphrag_results = await self._graphrag_search(query, top_k, filters, intent="code")
        if graphrag_results:
            self.last_embedding_model = "graphrag-code"
            return graphrag_results

        # Fallback to legacy keyword search filtered for code files
        logger.debug("GraphRAG code search unavailable, falling back to keyword search")
        return await self._legacy_keyword_search(query, top_k, filters)

    async def _general_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute general hybrid search using GraphRAG or legacy fallback.

        Per GFM-RAG paper:
        1. Try semantic retrieval via ChromaDB + graph traversal via Neo4j
        2. Fall back to keyword matching if GraphRAG unavailable

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional search filters

        Returns:
            List of search results
        """
        # Try GraphRAG first
        graphrag_results = await self._graphrag_search(query, top_k, filters, intent="exploratory")
        if graphrag_results:
            self.last_embedding_model = "graphrag"
            return graphrag_results

        # Fall back to legacy keyword search
        logger.debug("GraphRAG returned no results, falling back to legacy keyword search")
        return await self._legacy_keyword_search(query, top_k, filters)

    async def _legacy_keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Legacy keyword-based search over workspace documents.

        Fallback when GraphRAG infrastructure is unavailable.
        Uses term frequency scoring for relevance ranking.
        """
        workspace = Path.home() / ".futurnal" / "workspace"
        parsed_dir = workspace / "parsed"
        imap_dir = workspace / "imap"

        results: List[Dict[str, Any]] = []
        query_terms = query.lower().split()

        if not query_terms:
            return []

        # Search parsed documents (Obsidian, GitHub, etc.)
        if parsed_dir.exists():
            for json_file in parsed_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                    text = data.get("text", "") or data.get("content", "")
                    if not text:
                        continue

                    # Calculate term frequency score
                    text_lower = text.lower()
                    matches = sum(1 for term in query_terms if term in text_lower)

                    if matches > 0:
                        score = matches / len(query_terms)

                        # Extract metadata
                        metadata = data.get("metadata", {})
                        extra = metadata.get("extra", {})

                        # Determine entity type
                        source = metadata.get("source", "")
                        entity_type = "Document"
                        if source.endswith((".py", ".rs", ".ts", ".js", ".tsx", ".jsx")):
                            entity_type = "Code"
                        elif source.endswith(".md"):
                            entity_type = "Document"

                        # Get best label
                        label = (
                            extra.get("title")
                            or metadata.get("filename", "")
                            or json_file.stem[:50]
                        )

                        # Truncate content for snippet
                        snippet = text[:300] + "..." if len(text) > 300 else text

                        results.append({
                            "id": data.get("element_id", json_file.stem),
                            "content": snippet,
                            "score": score,
                            "confidence": min(0.5 + (score * 0.5), 1.0),
                            "timestamp": data.get("ingested_at"),
                            "entity_type": entity_type,
                            "source_type": "text",
                            "metadata": {
                                "source": source,
                                "label": label,
                                "path": metadata.get("path"),
                            },
                        })
                except Exception as e:
                    logger.debug(f"Error reading {json_file}: {e}")
                    continue

        # Search IMAP emails
        if imap_dir.exists():
            for json_file in imap_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                    content = data.get("content", "")
                    if not content:
                        continue

                    # Calculate term frequency score
                    text_lower = content.lower()
                    matches = sum(1 for term in query_terms if term in text_lower)

                    if matches > 0:
                        score = matches / len(query_terms)
                        metadata = data.get("metadata", {})

                        # Truncate content for snippet
                        snippet = content[:300] + "..." if len(content) > 300 else content

                        results.append({
                            "id": data.get("sha256", json_file.stem),
                            "content": snippet,
                            "score": score,
                            "confidence": min(0.5 + (score * 0.5), 1.0),
                            "timestamp": metadata.get("date"),
                            "entity_type": "Email",
                            "source_type": "text",
                            "metadata": {
                                "source": metadata.get("source", ""),
                                "label": metadata.get("subject", "Email"),
                                "sender": metadata.get("sender"),
                                "recipient": metadata.get("recipient"),
                            },
                        })
                except Exception as e:
                    logger.debug(f"Error reading {json_file}: {e}")
                    continue

        # Sort by score descending and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def _basic_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Basic fallback search."""
        return await self._general_search(query, top_k, None)

    async def _multimodal_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute multimodal-aware search.

        Uses modality hints to route to appropriate content sources
        (OCR documents, audio transcriptions, etc.).

        Args:
            query: Natural language query with modality hints
            top_k: Number of results to return
            filters: Optional search filters

        Returns:
            List of search results from specified modalities
        """
        if not self.multimodal_handler:
            return await self._general_search(query, top_k, filters)

        try:
            # Analyze query for modality hints
            plan = self.multimodal_handler.analyze_query(query)

            # Execute multimodal search
            multimodal_results = await self.multimodal_handler.execute(
                query=query,
                plan=plan,
                top_k=top_k,
            )

            # Convert to standard result format
            results: List[Dict[str, Any]] = []
            for r in multimodal_results:
                results.append({
                    "id": r.entity_id,
                    "content": r.content,
                    "score": r.score,
                    "confidence": r.source_confidence,
                    "source_type": r.source_type.value,
                    "retrieval_boost": r.retrieval_boost,
                    **r.metadata,
                })

            return results

        except Exception as e:
            logger.warning(f"Multimodal search failed, falling back: {e}")
            return await self._general_search(query, top_k, filters)

    async def record_feedback(
        self,
        query: str,
        clicked_result_id: Optional[str],
        feedback_type: str = "click",
        result_count: int = 0,
        clicked_position: Optional[int] = None,
    ) -> None:
        """Record search quality feedback for GRPO.

        Args:
            query: Original query
            clicked_result_id: ID of clicked result
            feedback_type: Type of feedback (click, no_results, refinement)
            result_count: Number of results in original search
            clicked_position: Position of clicked result (if applicable)
        """
        if self.quality_feedback:
            # Use record_signal which is the correct method
            signal_value = 1.0 if feedback_type == "click" else -0.5
            self.quality_feedback.record_signal(
                query_id=query[:50],
                signal_type=feedback_type,
                signal_value=signal_value,
                context={"result_id": clicked_result_id} if clicked_result_id else None,
            )

        # Audit log feedback (query content is NOT logged)
        if self._audit:
            log_feedback_recorded(
                self._audit,
                feedback_type=feedback_type,
                search_type="hybrid",
                result_count=result_count,
                clicked_position=clicked_position,
            )

    async def index_ocr_content(self, ocr_output: Dict[str, Any]) -> str:
        """Index OCR-extracted content with source metadata.

        Processes OCR output using OCRContentProcessor to extract
        text, source metadata, and fuzzy variants for search.

        Args:
            ocr_output: Raw OCR output with text, confidence, source_file,
                layout information, and optional bounding boxes

        Returns:
            ID of indexed content

        Example:
            content_id = await api.index_ocr_content({
                "text": "Scanned document content",
                "confidence": 0.92,
                "source_file": "document.pdf",
                "language": "en",
            })
        """
        # Process with OCR processor if available
        if self.ocr_processor:
            processed = self.ocr_processor.process_ocr_result(ocr_output)
            content_text = processed["content"]
            source_metadata = processed["source_metadata"]
            fuzzy_variants = processed.get("fuzzy_variants", [])
        else:
            content_text = ocr_output.get("text", "")
            source_metadata = {"source_type": "ocr_document"}
            fuzzy_variants = []

        content_id = f"ocr_{hash(content_text)}"

        # Store in PKG with source metadata (actual storage is placeholder)
        # In production, this would call self.pkg.store_content() with:
        # - content: content_text
        # - source_metadata: source_metadata
        # - fuzzy_variants: fuzzy_variants for search expansion
        logger.info(
            f"Indexed OCR content: {content_id}",
            extra={
                "source_type": source_metadata.get("source_type"),
                "confidence": source_metadata.get("extraction_confidence"),
            },
        )

        # Audit log OCR indexing (content is NOT logged)
        if self._audit:
            from hashlib import sha256
            source_file = ocr_output.get("source_file", "unknown")
            source_file_hash = sha256(str(source_file).encode()).hexdigest()[:32]
            log_ocr_content_indexed(
                self._audit,
                content_id=content_id,
                confidence=ocr_output.get("confidence", 0.0),
                source_file_hash=source_file_hash,
                page_count=ocr_output.get("page_count", 1),
            )

        return content_id

    async def index_transcription(self, whisper_output: Dict[str, Any]) -> str:
        """Index audio transcription with source metadata.

        Processes Whisper output using TranscriptionProcessor to extract
        text, speaker info, and homophone-expanded searchable content.

        Args:
            whisper_output: Raw Whisper output with text, segments,
                language, speaker labels, and confidence scores

        Returns:
            ID of indexed content

        Example:
            content_id = await api.index_transcription({
                "text": "Meeting discussion about project",
                "segments": [{"text": "...", "start": 0, "end": 5}],
                "language": "en",
            })
        """
        # Process with transcription processor if available
        if self.transcription_processor:
            processed = self.transcription_processor.process_transcription(
                whisper_output
            )
            content_text = processed["content"]
            searchable_content = processed["searchable_content"]
            source_metadata = processed["source_metadata"]
            segments = processed.get("segments", [])
            speakers = processed.get("speakers", [])
        else:
            content_text = whisper_output.get("text", "")
            searchable_content = content_text
            source_metadata = {"source_type": "audio_transcription"}
            segments = whisper_output.get("segments", [])
            speakers = []

        content_id = f"audio_{hash(content_text)}"

        # Calculate duration from segments
        duration_seconds = 0.0
        if segments:
            last_segment = segments[-1]
            duration_seconds = last_segment.get("end", 0.0)
        elif self.transcription_processor:
            duration_seconds = processed.get("transcription_metadata", {}).get(
                "duration_seconds", 0.0
            )

        # Store in PKG with source metadata (actual storage is placeholder)
        # In production, this would call self.pkg.store_content() with:
        # - content: content_text
        # - searchable_content: searchable_content (homophone-expanded)
        # - source_metadata: source_metadata
        # - segments: segments for timestamp-based retrieval
        # - speakers: speakers for speaker-based filtering
        logger.info(
            f"Indexed transcription: {content_id}",
            extra={
                "source_type": source_metadata.get("source_type"),
                "duration": duration_seconds,
                "speaker_count": len(speakers),
            },
        )

        # Audit log transcription indexing (content is NOT logged)
        if self._audit:
            log_transcription_indexed(
                self._audit,
                content_id=content_id,
                duration_seconds=duration_seconds,
                speaker_count=len(speakers),
                language=whisper_output.get("language", "en"),
            )

        return content_id

    async def index_text(self, text: str) -> str:
        """Index text content.

        Args:
            text: Text content

        Returns:
            ID of indexed content
        """
        content_id = f"text_{hash(text)}"
        logger.info(f"Indexed text: {content_id}")

        # Audit log text indexing (content is NOT logged)
        if self._audit:
            log_content_indexed(
                self._audit,
                content_id=content_id,
                content_type="text",
                source_type="text",
                size_bytes=len(text.encode("utf-8")),
            )

        return content_id


def create_hybrid_search_api(
    multimodal_enabled: bool = False,
    experiential_learning: bool = False,
    caching_enabled: bool = True,
    config: Optional[SearchConfig] = None,
    audit_logger: Optional["AuditLogger"] = None,
    embedding_service: Optional["MultiModelEmbeddingService"] = None,
    graphrag_enabled: bool = True,
    pkg_client: Optional[Any] = None,
) -> HybridSearchAPI:
    """Create HybridSearchAPI instance with GraphRAG support.

    Factory function for integration tests and production use.

    Args:
        multimodal_enabled: Enable multimodal content search
        experiential_learning: Enable GRPO feedback
        caching_enabled: Enable caching
        config: Optional search config
        audit_logger: Optional audit logger for search event tracking
        embedding_service: Optional embedding service for GraphRAG
        graphrag_enabled: Enable GraphRAG pipeline (semantic + graph search)
        pkg_client: Optional PKG client for graph queries (attempts auto-connect if None)

    Returns:
        Configured HybridSearchAPI instance

    Example:
        # Basic usage (attempts lazy GraphRAG initialization)
        api = create_hybrid_search_api()

        # With explicit embedding service
        from futurnal.embeddings.service import MultiModelEmbeddingService
        service = MultiModelEmbeddingService()
        api = create_hybrid_search_api(embedding_service=service)

        # Disable GraphRAG (legacy keyword search only)
        api = create_hybrid_search_api(graphrag_enabled=False)
    """
    # Try to connect to PKG if graphrag enabled and no client provided
    pkg_manager = None
    if graphrag_enabled and pkg_client is None:
        try:
            from futurnal.pkg.database.manager import PKGDatabaseManager
            from futurnal.configuration.settings import bootstrap_settings

            settings = bootstrap_settings()
            if settings and settings.workspace and settings.workspace.storage:
                pkg_manager = PKGDatabaseManager(settings.workspace.storage)
                # connect() returns the driver
                pkg_client = pkg_manager.connect()
                logger.info("GraphRAG: Connected to PKG database")
        except Exception as e:
            logger.debug(f"GraphRAG: Could not connect to PKG database: {e}")
            # Continues without PKG - GraphRAG will be limited to vector search

    return HybridSearchAPI(
        config=config,
        pkg_client=pkg_client,
        pkg_manager=pkg_manager,
        multimodal_enabled=multimodal_enabled,
        experiential_learning=experiential_learning,
        caching_enabled=caching_enabled,
        audit_logger=audit_logger,
        embedding_service=embedding_service,
        graphrag_enabled=graphrag_enabled,
    )
