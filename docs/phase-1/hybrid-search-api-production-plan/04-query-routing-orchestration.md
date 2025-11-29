Summary: Implement query routing with intent classification, multi-strategy composition, result ranking, and context assembly.

# 04 · Query Routing & Orchestration

## Purpose
Implement intelligent query routing that classifies user intent and orchestrates appropriate retrieval strategies (temporal, causal, exploratory, lookup) with result ranking and context assembly.

**Criticality**: HIGH - Ensures queries route to optimal strategies

## Scope
- Intent classification (lookup, exploratory, temporal, causal)
- Multi-strategy composition
- Result ranking and fusion
- Context assembly for answers
- Query plan optimization

## Requirements Alignment
- **Option B Requirement**: "Route queries to appropriate retrieval strategies"
- **Performance Target**: Intent classification in <100ms
- **Enables**: Optimal retrieval for diverse query types

## Component Design

```python
class QueryRouter:
    """
    Routes queries to appropriate retrieval strategies.

    Classifies intent and composes multi-strategy plans.
    """

    def __init__(
        self,
        temporal_engine,
        causal_retrieval,
        schema_retrieval,
        llm
    ):
        self.temporal = temporal_engine
        self.causal = causal_retrieval
        self.schema = schema_retrieval
        self.llm = llm

    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route query to appropriate strategies.

        Returns execution plan with strategies and weights.
        """
        # Classify intent
        intent = self._classify_intent(query)

        # Build execution plan
        if intent == "temporal":
            plan = {
                "primary_strategy": "temporal_query",
                "secondary_strategy": "hybrid_retrieval",
                "weights": {"temporal": 0.7, "hybrid": 0.3}
            }
        elif intent == "causal":
            plan = {
                "primary_strategy": "causal_chain",
                "secondary_strategy": "temporal_query",
                "weights": {"causal": 0.6, "temporal": 0.4}
            }
        elif intent == "lookup":
            plan = {
                "primary_strategy": "hybrid_retrieval",
                "weights": {"hybrid": 1.0}
            }
        else:  # exploratory
            plan = {
                "primary_strategy": "hybrid_retrieval",
                "secondary_strategy": "temporal_query",
                "weights": {"hybrid": 0.6, "temporal": 0.4}
            }

        return plan

    def _classify_intent(self, query: str) -> str:
        """
        Classify query intent using LLM.

        Returns: "lookup", "exploratory", "temporal", "causal"
        """
        prompt = f"""Classify the query intent:

Query: "{query}"

Intent types:
- lookup: Specific entity or fact lookup
- exploratory: Broad exploration or discovery
- temporal: Time-based query (when, before, after)
- causal: Causation query (why, what caused, what led to)

Intent:"""

        response = self.llm.generate(prompt, max_tokens=10)
        intent = response.strip().lower()

        return intent if intent in ["lookup", "exploratory", "temporal", "causal"] else "exploratory"
```

## Success Metrics

- ✅ Intent classification >85% accuracy
- ✅ Query routing <100ms latency
- ✅ Multi-strategy composition functional
- ✅ Result ranking quality high

## Dependencies

- Temporal query engine (01)
- Causal chain retrieval (02)
- Schema-aware retrieval (03)
- LLM for intent classification

**This module orchestrates optimal retrieval strategies per query.**
