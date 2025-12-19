"""
Community Detection for Knowledge Graph.

Implements community detection algorithms for hierarchical knowledge organization:
- Louvain algorithm for community detection
- Leiden algorithm for improved modularity
- Hierarchical community structure

Based on GraphRAG and Youtu-GraphRAG papers.
"""

from .detection import (
    CommunityDetector,
    LouvainDetector,
    LeidenDetector,
    DuallyPerceivedDetector,
)
from .hierarchy import (
    CommunityHierarchy,
    HierarchyBuilder,
    CommunityLevel,
)
from .summarization import (
    CommunitySummarizer,
    HierarchicalSummarizer,
    SummaryCache,
)

__all__ = [
    "CommunityDetector",
    "LouvainDetector",
    "LeidenDetector",
    "DuallyPerceivedDetector",
    "CommunityHierarchy",
    "HierarchyBuilder",
    "CommunityLevel",
    "CommunitySummarizer",
    "HierarchicalSummarizer",
    "SummaryCache",
]
