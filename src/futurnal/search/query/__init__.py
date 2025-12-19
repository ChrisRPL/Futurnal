"""
Query Decomposition and Processing.

Implements complex query handling:
- Multi-clause query decomposition
- Query planning and optimization
- Sub-query execution and fusion

Based on GraphSearch and agentic RAG papers.
"""

from .decomposer import (
    QueryDecomposer,
    SubQuery,
    QueryPlan,
    DecompositionStrategy,
)
from .planner import (
    QueryPlanner,
    ExecutionPlan,
    PlanNode,
)
from .executor import (
    QueryExecutor,
    SubQueryResult,
    FusedResult,
    QueryPipeline,
)

__all__ = [
    "QueryDecomposer",
    "SubQuery",
    "QueryPlan",
    "DecompositionStrategy",
    "QueryPlanner",
    "ExecutionPlan",
    "PlanNode",
    "QueryExecutor",
    "SubQueryResult",
    "FusedResult",
    "QueryPipeline",
]
