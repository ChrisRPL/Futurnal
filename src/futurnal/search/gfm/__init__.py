"""
Graph Foundation Model (GFM) for Retrieval-Augmented Generation.

Implements a query-dependent Graph Neural Network that enables:
- Single-step multi-hop retrieval
- Cross-dataset generalization
- Efficient document ranking via graph structure

Based on GFM-RAG paper (2502.01113v1).
"""

from .model import GraphFoundationModel
from .retriever import GFMRetriever
from .kg_index import KGIndexBuilder
from .training import GFMTrainer

__all__ = [
    "GraphFoundationModel",
    "GFMRetriever",
    "KGIndexBuilder",
    "GFMTrainer",
]
