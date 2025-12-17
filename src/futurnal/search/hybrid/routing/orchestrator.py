"""Query Router Orchestrator.

Main orchestrator for query routing with LLM-based intent classification,
multi-strategy execution, and result fusion.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Responsibilities:
1. Classify query intent via LLM (<100ms with Ollama)
2. Build execution plan with strategy weights
3. Execute multi-strategy retrieval
4. Fuse results with weighted combination
5. Record feedback for GRPO learning
6. AGI Phase 3: Dynamic weight optimization via SearchRankingOptimizer

Integration Points:
- TemporalQueryEngine: For temporal queries
- CausalChainRetrieval: For causal queries
- SchemaAwareRetrieval: For hybrid/lookup queries
- SearchQualityFeedback: GRPO integration
- QueryTemplateDatabase: Query understanding templates
- SearchRankingOptimizer: AGI Phase 3 bidirectional learning

Option B Compliance:
- Ghost model FROZEN: LLM classification only
- Experiential learning via SearchQualityFeedback
- Temporal-first design: TEMPORAL queries prioritized
- Dynamic weights via SearchRankingOptimizer (not model updates)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from futurnal.search.hybrid.types import QueryIntent, QueryPlan, QueryResult
from futurnal.search.hybrid.routing.intent_classifier import (
    IntentClassifierLLM,
    get_intent_classifier,
)

if TYPE_CHECKING:
    from futurnal.search.temporal.engine import TemporalQueryEngine
    from futurnal.search.causal.retrieval import CausalChainRetrieval
    from futurnal.search.hybrid.retrieval import SchemaAwareRetrieval
    from futurnal.search.hybrid.routing.feedback import SearchQualityFeedback
    from futurnal.search.hybrid.routing.optimizer import SearchRankingOptimizer
    from futurnal.search.hybrid.routing.templates import (
        QueryTemplate,
        QueryTemplateDatabase,
    )

logger = logging.getLogger(__name__)


class QueryRouter:
    """Routes queries to appropriate retrieval strategies.

    Main orchestrator that:
    1. Classifies intent via LLM
    2. Builds execution plan with strategy weights
    3. Executes multi-strategy retrieval
    4. Fuses results
    5. Records feedback for GRPO learning

    Strategy Configuration by Intent:
    - TEMPORAL: primary=temporal_query (0.7), secondary=hybrid_retrieval (0.3)
    - CAUSAL: primary=causal_chain (0.6), secondary=temporal_query (0.4)
    - LOOKUP: primary=hybrid_retrieval (1.0)
    - EXPLORATORY: primary=hybrid_retrieval (0.6), secondary=temporal_query (0.4)

    Example:
        router = QueryRouter(
            temporal_engine=temporal_engine,
            causal_retrieval=causal_retrieval,
            schema_retrieval=schema_retrieval,
        )

        result = router.route_and_execute("What happened in January 2024?")
        print(result.intent)  # QueryIntent.TEMPORAL
        print(result.entities)  # Retrieved entities
    """

    # Strategy configuration per intent type
    STRATEGY_CONFIGS: Dict[QueryIntent, Dict[str, Any]] = {
        QueryIntent.TEMPORAL: {
            "primary_strategy": "temporal_query",
            "secondary_strategy": "hybrid_retrieval",
            "weights": {"temporal": 0.7, "hybrid": 0.3},
            "estimated_latency_ms": 300,
        },
        QueryIntent.CAUSAL: {
            "primary_strategy": "causal_chain",
            "secondary_strategy": "temporal_query",
            "weights": {"causal": 0.6, "temporal": 0.4},
            "estimated_latency_ms": 500,
        },
        QueryIntent.LOOKUP: {
            "primary_strategy": "hybrid_retrieval",
            "secondary_strategy": None,
            "weights": {"hybrid": 1.0},
            "estimated_latency_ms": 200,
        },
        QueryIntent.EXPLORATORY: {
            "primary_strategy": "hybrid_retrieval",
            "secondary_strategy": "temporal_query",
            "weights": {"hybrid": 0.6, "temporal": 0.4},
            "estimated_latency_ms": 400,
        },
    }

    def __init__(
        self,
        temporal_engine: Optional["TemporalQueryEngine"] = None,
        causal_retrieval: Optional["CausalChainRetrieval"] = None,
        schema_retrieval: Optional["SchemaAwareRetrieval"] = None,
        intent_classifier: Optional[IntentClassifierLLM] = None,
        grpo_feedback: Optional["SearchQualityFeedback"] = None,
        template_db: Optional["QueryTemplateDatabase"] = None,
        optimizer: Optional["SearchRankingOptimizer"] = None,
    ):
        """Initialize QueryRouter.

        Args:
            temporal_engine: TemporalQueryEngine for temporal strategies
            causal_retrieval: CausalChainRetrieval for causal strategies
            schema_retrieval: SchemaAwareRetrieval for hybrid strategies
            intent_classifier: Intent classifier (auto-created if None)
            grpo_feedback: GRPO feedback collector (optional)
            template_db: Query template database (optional)
            optimizer: AGI Phase 3 - SearchRankingOptimizer for bidirectional learning
        """
        self.temporal = temporal_engine
        self.causal = causal_retrieval
        self.schema = schema_retrieval
        self.classifier = intent_classifier or get_intent_classifier()
        self.grpo_feedback = grpo_feedback
        self.template_db = template_db
        self.optimizer = optimizer

        # AGI Phase 3: Connect optimizer to feedback system for bidirectional learning
        if self.grpo_feedback and self.optimizer:
            self.grpo_feedback.set_optimizer(self.optimizer)
            self.grpo_feedback.set_query_router(self)
            logger.info("Bidirectional learning loop connected")

        logger.info(
            f"QueryRouter initialized "
            f"(optimizer: {'connected' if optimizer else 'not connected'})"
        )

    def route_query(self, query: str) -> QueryPlan:
        """Route query to appropriate strategies.

        Steps:
        1. Classify intent via LLM (<100ms with Ollama)
        2. Apply query understanding template (if available)
        3. Build execution plan with strategy weights
        4. Record for GRPO feedback

        Args:
            query: User query string

        Returns:
            QueryPlan with routing decisions
        """
        start_time = time.time()

        # Step 1: Classify intent
        intent_result = self.classifier.classify_intent(query)
        intent = QueryIntent(intent_result["intent"])
        confidence = intent_result["confidence"]

        logger.debug(
            f"Intent classified: {intent.value} "
            f"(confidence: {confidence:.2f})"
        )

        # Step 2: Apply query template if available
        enhanced_query = query
        if self.template_db:
            template = self.template_db.select_template(intent)
            enhanced_query = self._apply_query_template(query, template)

        # Step 3: Build execution plan
        plan = self._build_execution_plan(
            query=enhanced_query,
            original_query=query,
            intent=intent,
            confidence=confidence,
        )

        # Step 4: Record for GRPO feedback
        if self.grpo_feedback:
            self.grpo_feedback.record_query(plan)

        routing_latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Query routed: {intent.value} -> {plan.primary_strategy} "
            f"(latency: {routing_latency_ms:.1f}ms)"
        )

        return plan

    def execute_plan(self, plan: QueryPlan) -> QueryResult:
        """Execute query plan across strategies.

        Runs primary and secondary strategies, fuses results.

        Args:
            plan: QueryPlan from route_query()

        Returns:
            QueryResult with fused results
        """
        start_time = time.time()

        results: List[Dict[str, Any]] = []
        sources: Set[str] = set()

        # Execute primary strategy
        try:
            primary_results = self._execute_strategy(
                plan.primary_strategy,
                plan.original_query,
            )
            results.extend(primary_results)
            sources.add(plan.primary_strategy)
            logger.debug(
                f"Primary strategy {plan.primary_strategy}: "
                f"{len(primary_results)} results"
            )
        except Exception as e:
            logger.warning(f"Primary strategy {plan.primary_strategy} failed: {e}")

        # Execute secondary strategy if present
        if plan.secondary_strategy:
            try:
                secondary_results = self._execute_strategy(
                    plan.secondary_strategy,
                    plan.original_query,
                )
                results.extend(secondary_results)
                sources.add(plan.secondary_strategy)
                logger.debug(
                    f"Secondary strategy {plan.secondary_strategy}: "
                    f"{len(secondary_results)} results"
                )
            except Exception as e:
                logger.warning(
                    f"Secondary strategy {plan.secondary_strategy} failed: {e}"
                )

        # Fuse results with weights
        fused = self._fuse_results(results, plan.weights, plan.intent)

        latency_ms = int((time.time() - start_time) * 1000)

        result = QueryResult(
            result_id=str(uuid.uuid4()),
            query_id=plan.query_id,
            entities=fused.get("entities", []),
            relationships=fused.get("relationships", []),
            temporal_context=fused.get("temporal_context"),
            causal_chain=fused.get("causal_chain"),
            relevance_scores=fused.get("scores", {}),
            provenance=fused.get("sources", []),
            latency_ms=latency_ms,
        )

        logger.info(
            f"Plan executed: {len(result.entities)} entities, "
            f"{len(result.relationships)} relationships "
            f"(latency: {latency_ms}ms)"
        )

        return result

    def route_and_execute(self, query: str) -> QueryResult:
        """Convenience method: route + execute in one call.

        Args:
            query: User query string

        Returns:
            QueryResult with fused results
        """
        plan = self.route_query(query)
        return self.execute_plan(plan)

    def _build_execution_plan(
        self,
        query: str,
        original_query: str,
        intent: QueryIntent,
        confidence: float,
    ) -> QueryPlan:
        """Build execution plan based on intent.

        Args:
            query: Enhanced query (after template application)
            original_query: Original user query
            intent: Classified intent
            confidence: Classification confidence

        Returns:
            QueryPlan with strategy configuration
        """
        config = self.STRATEGY_CONFIGS[intent]

        return QueryPlan(
            query_id=str(uuid.uuid4()),
            original_query=original_query,
            intent=intent,
            intent_confidence=confidence,
            primary_strategy=config["primary_strategy"],
            secondary_strategy=config["secondary_strategy"],
            weights=config["weights"],
            estimated_latency_ms=config["estimated_latency_ms"],
            created_at=datetime.utcnow(),
        )

    def _execute_strategy(
        self,
        strategy: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Execute a single retrieval strategy.

        Args:
            strategy: Strategy name
            query: Query string

        Returns:
            List of result dictionaries
        """
        if strategy == "temporal_query":
            if self.temporal is None:
                logger.warning("TemporalQueryEngine not configured")
                return []
            return self._execute_temporal_strategy(query)

        elif strategy == "causal_chain":
            if self.causal is None:
                logger.warning("CausalChainRetrieval not configured")
                return []
            return self._execute_causal_strategy(query)

        elif strategy == "hybrid_retrieval":
            if self.schema is None:
                logger.warning("SchemaAwareRetrieval not configured")
                return []
            return self._execute_hybrid_strategy(query)

        else:
            logger.warning(f"Unknown strategy: {strategy}")
            return []

    def _execute_temporal_strategy(self, query: str) -> List[Dict[str, Any]]:
        """Execute temporal query strategy.

        Args:
            query: Query string

        Returns:
            Temporal query results
        """
        try:
            # Use TemporalQueryEngine's search method
            # The engine returns TemporalSearchResult or similar
            from futurnal.search.temporal.types import (
                TemporalQuery,
                TemporalQueryType,
            )

            # Create a time range query for exploration
            temporal_query = TemporalQuery(
                query_type=TemporalQueryType.TIME_RANGE,
                query_text=query,
            )

            result = self.temporal.query(temporal_query)

            # Convert to standard format
            entities = []
            if hasattr(result, "events"):
                for event in result.events:
                    entities.append({
                        "id": event.event_id if hasattr(event, "event_id") else str(uuid.uuid4()),
                        "type": "Event",
                        "content": str(event),
                        "score": event.score if hasattr(event, "score") else 0.8,
                        "source_strategy": "temporal_query",
                    })

            return entities

        except Exception as e:
            logger.error(f"Temporal strategy execution failed: {e}")
            return []

    def _execute_causal_strategy(self, query: str) -> List[Dict[str, Any]]:
        """Execute causal chain strategy.

        Args:
            query: Query string

        Returns:
            Causal chain results
        """
        try:
            # Use CausalChainRetrieval's search method
            from futurnal.search.causal.types import (
                CausalQuery,
                CausalQueryType,
            )

            causal_query = CausalQuery(
                query_type=CausalQueryType.FIND_CAUSES,
                query_text=query,
            )

            result = self.causal.query(causal_query)

            # Convert to standard format
            entities = []
            if hasattr(result, "causes"):
                for cause in result.causes:
                    entities.append({
                        "id": cause.event_id if hasattr(cause, "event_id") else str(uuid.uuid4()),
                        "type": "CausalEvent",
                        "content": str(cause),
                        "score": cause.confidence if hasattr(cause, "confidence") else 0.7,
                        "source_strategy": "causal_chain",
                    })

            return entities

        except Exception as e:
            logger.error(f"Causal strategy execution failed: {e}")
            return []

    def _execute_hybrid_strategy(self, query: str) -> List[Dict[str, Any]]:
        """Execute hybrid retrieval strategy.

        Args:
            query: Query string

        Returns:
            Hybrid search results
        """
        try:
            # Use SchemaAwareRetrieval's hybrid_search method
            results = self.schema.hybrid_search(
                query=query,
                intent="exploratory",
                top_k=20,
            )

            # Convert HybridSearchResult to standard format
            entities = []
            for result in results:
                entities.append({
                    "id": result.entity_id,
                    "type": result.entity_type,
                    "content": result.content,
                    "score": result.combined_score,
                    "vector_score": result.vector_score,
                    "graph_score": result.graph_score,
                    "source_strategy": "hybrid_retrieval",
                })

            return entities

        except Exception as e:
            logger.error(f"Hybrid strategy execution failed: {e}")
            return []

    def _fuse_results(
        self,
        results: List[Dict[str, Any]],
        weights: Dict[str, float],
        intent: QueryIntent,
    ) -> Dict[str, Any]:
        """Fuse results from multiple strategies.

        Applies weighted combination and deduplication.

        Args:
            results: Results from all strategies
            weights: Strategy weights
            intent: Query intent for context

        Returns:
            Fused result dictionary
        """
        # Deduplicate by entity ID
        seen_ids: Set[str] = set()
        fused_entities: List[Dict[str, Any]] = []
        fused_relationships: List[Dict[str, Any]] = []
        all_sources: Set[str] = set()

        for result in results:
            entity_id = result.get("id")
            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)

                # Apply strategy weight to score
                strategy = result.get("source_strategy", "unknown")
                weight = self._get_strategy_weight(strategy, weights)
                original_score = result.get("score", 0.5)
                weighted_score = original_score * weight

                result["weighted_score"] = weighted_score
                fused_entities.append(result)

                # Track sources for provenance
                source_id = result.get("source_id")
                if source_id:
                    all_sources.add(source_id)

        # Sort by weighted score
        fused_entities.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)

        # Build context based on intent
        temporal_context = None
        causal_chain = None

        if intent == QueryIntent.TEMPORAL:
            temporal_context = {
                "query_type": "temporal",
                "entities_count": len(fused_entities),
            }
        elif intent == QueryIntent.CAUSAL:
            causal_chain = [
                {"entity_id": e["id"], "score": e.get("weighted_score", 0)}
                for e in fused_entities[:10]
            ]

        return {
            "entities": fused_entities[:20],  # Top 20
            "relationships": fused_relationships,
            "temporal_context": temporal_context,
            "causal_chain": causal_chain,
            "scores": {
                "combined_relevance": (
                    sum(e.get("weighted_score", 0) for e in fused_entities[:10]) / 10
                    if fused_entities else 0
                ),
            },
            "sources": list(all_sources),
        }

    def _get_strategy_weight(
        self,
        strategy: str,
        weights: Dict[str, float],
    ) -> float:
        """Get weight for a strategy.

        Args:
            strategy: Strategy name
            weights: Weight configuration

        Returns:
            Weight value (0-1)
        """
        # Map strategy names to weight keys
        strategy_key_map = {
            "temporal_query": "temporal",
            "causal_chain": "causal",
            "hybrid_retrieval": "hybrid",
        }

        key = strategy_key_map.get(strategy, strategy)
        return weights.get(key, 0.5)

    def _apply_query_template(
        self,
        query: str,
        template: Optional["QueryTemplate"],
    ) -> str:
        """Apply query understanding template.

        Enhances query with structured understanding hints.

        Args:
            query: Original query
            template: Query template

        Returns:
            Enhanced query (or original if no template)
        """
        if template is None:
            return query

        # Template provides structured understanding hints
        # For now, just return the original query
        # In future, templates could augment the query with context
        return query

    def update_strategy_config(
        self,
        intent: QueryIntent,
        config: Dict[str, Any],
    ):
        """Update strategy configuration for an intent.

        Allows runtime tuning of strategy weights.

        Args:
            intent: Intent to update
            config: New configuration
        """
        if intent in self.STRATEGY_CONFIGS:
            self.STRATEGY_CONFIGS[intent].update(config)
            logger.info(f"Updated strategy config for {intent.value}")

    def get_strategy_config(self, intent: QueryIntent) -> Dict[str, Any]:
        """Get current strategy configuration for an intent.

        Args:
            intent: Intent to query

        Returns:
            Strategy configuration dictionary
        """
        return self.STRATEGY_CONFIGS.get(intent, {})

    # =========================================================================
    # AGI Phase 3: Bidirectional Learning Methods
    # =========================================================================

    def get_strategy_effectiveness(self, intent: QueryIntent) -> Dict[str, float]:
        """Get effectiveness scores for strategies of an intent.

        AGI Phase 3: Returns learned effectiveness from optimizer.

        Args:
            intent: Query intent

        Returns:
            Dictionary mapping strategy key to effectiveness score
        """
        if not self.optimizer:
            return {}
        return self.optimizer.get_strategy_effectiveness(intent)

    def export_learned_configs(self) -> str:
        """Export learned routing configurations as JSON.

        AGI Phase 3: Exports the learned weights for analysis or persistence.

        Returns:
            JSON string of learned configurations
        """
        if not self.optimizer:
            return "{}"
        return self.optimizer.export_learned_configs()

    def reset_learned_weights(self):
        """Reset learned weights to defaults.

        AGI Phase 3: Allows resetting the learning state.
        """
        if self.optimizer:
            self.optimizer.reset()
            logger.info("Learned weights reset to defaults")
