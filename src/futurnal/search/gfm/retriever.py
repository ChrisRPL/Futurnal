"""
GFM Retriever - Integration layer for Graph Foundation Model retrieval.

Provides a high-level interface for:
- Query processing
- Document retrieval via GFM
- Result ranking and filtering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
from torch import Tensor

from .model import GraphFoundationModel, GFMConfig, GFMEmbedder, GraphBatch
from .kg_index import KGIndex, KGIndexBuilder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from GFM retrieval."""
    document_id: str
    relevance_score: float
    entity_path: List[str] = field(default_factory=list)
    reasoning_chain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GFMRetrieverConfig:
    """Configuration for GFM Retriever."""
    # Model config
    model_config: GFMConfig = field(default_factory=GFMConfig)

    # Retrieval settings
    top_k: int = 10
    min_relevance: float = 0.3
    use_reranking: bool = True

    # Multi-hop settings
    max_hops: int = 3
    hop_decay: float = 0.8

    # Caching
    cache_embeddings: bool = True
    cache_size: int = 10000


class GFMRetriever:
    """
    High-level retriever using Graph Foundation Model.

    Integrates with Futurnal's existing search infrastructure to provide
    GNN-enhanced retrieval with multi-hop reasoning.
    """

    def __init__(
        self,
        config: Optional[GFMRetrieverConfig] = None,
        kg_index: Optional[KGIndex] = None,
        model_path: Optional[Path] = None
    ):
        self.config = config or GFMRetrieverConfig()
        self.kg_index = kg_index

        # Initialize model
        self.model = GraphFoundationModel(self.config.model_config)
        if model_path and model_path.exists():
            self._load_model(model_path)

        # Initialize embedder
        self.embedder = GFMEmbedder()

        # Embedding cache
        self._query_cache: Dict[str, Tensor] = {}

    def _load_model(self, path: Path) -> None:
        """Load pre-trained model weights."""
        try:
            state_dict = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded GFM model from {path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")

    def set_index(self, kg_index: KGIndex) -> None:
        """Set or update the KG index."""
        self.kg_index = kg_index

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents relevant to a query using GFM.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filter_doc_ids: Optional list of doc IDs to filter results

        Returns:
            List of RetrievalResult ordered by relevance
        """
        if self.kg_index is None:
            logger.warning("No KG index set, returning empty results")
            return []

        top_k = top_k or self.config.top_k

        # Get query embedding (with caching)
        query_embedding = self._get_query_embedding(query)

        # Build graph batch
        graph_batch = self._create_graph_batch(query_embedding)

        # Run GFM retrieval
        self.model.eval()
        with torch.no_grad():
            raw_results = self.model.retrieve(graph_batch, top_k=top_k * 2)

        # Process and filter results
        results = []
        seen_docs = set()

        for doc_id, score in raw_results:
            if score < self.config.min_relevance:
                continue

            if filter_doc_ids and doc_id not in filter_doc_ids:
                continue

            if doc_id in seen_docs:
                continue

            seen_docs.add(doc_id)

            # Get entity path for explanation
            entity_path = self._get_entity_path(doc_id)

            result = RetrievalResult(
                document_id=doc_id,
                relevance_score=score,
                entity_path=entity_path,
                metadata={"method": "gfm", "hops": len(entity_path)}
            )
            results.append(result)

            if len(results) >= top_k:
                break

        # Optional reranking
        if self.config.use_reranking and len(results) > 1:
            results = self._rerank_results(query, results)

        return results

    def retrieve_multi_hop(
        self,
        query: str,
        seed_entities: Optional[List[str]] = None,
        max_hops: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Perform multi-hop retrieval starting from seed entities.

        Args:
            query: Natural language query
            seed_entities: Starting entities for traversal
            max_hops: Maximum hop distance

        Returns:
            Retrieved documents with reasoning paths
        """
        max_hops = max_hops or self.config.max_hops

        if self.kg_index is None:
            return []

        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Find seed nodes if not provided
        if seed_entities is None:
            seed_entities = self._identify_seed_entities(query)

        # Multi-hop traversal
        visited = set()
        results = []
        hop_decay = 1.0

        for hop in range(max_hops):
            if not seed_entities:
                break

            # Get neighbors for current seeds
            next_seeds = []
            for entity_id in seed_entities:
                if entity_id in visited:
                    continue
                visited.add(entity_id)

                # Get entity's documents
                if entity_id in self.kg_index.entity_to_idx:
                    node_idx = self.kg_index.entity_to_idx[entity_id]
                    for doc_id in self.kg_index.node_to_docs.get(node_idx, []):
                        # Score based on query relevance and hop distance
                        entity = self.kg_index.entities[entity_id]
                        if entity.embedding is not None:
                            sim = torch.cosine_similarity(
                                query_embedding.squeeze(),
                                entity.embedding,
                                dim=0
                            ).item()
                        else:
                            sim = 0.5

                        score = sim * hop_decay

                        if score >= self.config.min_relevance:
                            results.append(RetrievalResult(
                                document_id=doc_id,
                                relevance_score=score,
                                entity_path=[entity_id],
                                metadata={"hop": hop, "seed_entity": entity_id}
                            ))

                # Get neighbors for next hop
                neighbors = self._get_neighbors(entity_id)
                next_seeds.extend(neighbors)

            seed_entities = list(set(next_seeds) - visited)
            hop_decay *= self.config.hop_decay

        # Sort by score
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Deduplicate by document
        seen = set()
        unique_results = []
        for r in results:
            if r.document_id not in seen:
                seen.add(r.document_id)
                unique_results.append(r)

        return unique_results[:self.config.top_k]

    def _get_query_embedding(self, query: str) -> Tensor:
        """Get query embedding with caching."""
        if self.config.cache_embeddings and query in self._query_cache:
            return self._query_cache[query]

        embedding = self.embedder.encode_query(query)

        if self.config.cache_embeddings:
            if len(self._query_cache) >= self.config.cache_size:
                # Simple LRU: remove random entry
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[query] = embedding

        return embedding

    def _create_graph_batch(self, query_embedding: Tensor) -> GraphBatch:
        """Create a GraphBatch from the current KG index."""
        if self.kg_index is None or self.kg_index.num_entities == 0:
            # Empty graph
            return GraphBatch(
                node_features=torch.zeros(1, self.config.model_config.hidden_dim),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_type=torch.zeros(0, dtype=torch.long),
                query_embedding=query_embedding
            )

        # Stack entity embeddings
        node_features = []
        for i in range(self.kg_index.num_entities):
            entity_id = self.kg_index.idx_to_entity[i]
            entity = self.kg_index.entities[entity_id]
            if entity.embedding is not None:
                node_features.append(entity.embedding)
            else:
                node_features.append(torch.zeros(self.config.model_config.hidden_dim))

        node_features = torch.stack(node_features)

        # Create node to doc mapping
        node_to_doc = {}
        for node_idx, doc_ids in self.kg_index.node_to_docs.items():
            for doc_id in doc_ids:
                node_to_doc[node_idx] = doc_id  # Take first doc if multiple

        return GraphBatch(
            node_features=node_features,
            edge_index=self.kg_index.edge_index,
            edge_type=self.kg_index.edge_type,
            query_embedding=query_embedding,
            node_to_doc=node_to_doc
        )

    def _get_entity_path(self, doc_id: str) -> List[str]:
        """Get the entity path that leads to a document."""
        if self.kg_index is None:
            return []

        path = []
        for entity_id, entity in self.kg_index.entities.items():
            if doc_id in entity.source_docs:
                path.append(entity_id)

        return path[:5]  # Limit path length

    def _get_neighbors(self, entity_id: str) -> List[str]:
        """Get neighboring entities in the graph."""
        if self.kg_index is None or entity_id not in self.kg_index.entity_to_idx:
            return []

        node_idx = self.kg_index.entity_to_idx[entity_id]
        neighbors = []

        # Find edges from/to this node
        if self.kg_index.edge_index is not None:
            src, dst = self.kg_index.edge_index
            mask = (src == node_idx) | (dst == node_idx)
            neighbor_indices = torch.unique(
                torch.cat([src[mask], dst[mask]])
            ).tolist()

            for idx in neighbor_indices:
                if idx != node_idx and idx in self.kg_index.idx_to_entity:
                    neighbors.append(self.kg_index.idx_to_entity[idx])

        return neighbors

    def _identify_seed_entities(self, query: str) -> List[str]:
        """Identify relevant seed entities from query."""
        if self.kg_index is None:
            return []

        query_embedding = self._get_query_embedding(query).squeeze()
        scores = []

        for entity_id, entity in self.kg_index.entities.items():
            if entity.embedding is not None:
                sim = torch.cosine_similarity(
                    query_embedding, entity.embedding, dim=0
                ).item()
                scores.append((entity_id, sim))

        # Sort and return top entities
        scores.sort(key=lambda x: x[1], reverse=True)
        return [entity_id for entity_id, _ in scores[:10]]

    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank results using additional signals."""
        # Simple reranking based on entity path diversity
        for result in results:
            # Boost results with longer entity paths (more connected)
            path_bonus = min(len(result.entity_path) * 0.05, 0.2)
            result.relevance_score += path_bonus

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results


class HybridGFMRetriever:
    """
    Hybrid retriever combining GFM with vector search.

    Fuses results from:
    1. GFM graph-based retrieval
    2. Traditional vector similarity search
    """

    def __init__(
        self,
        gfm_retriever: GFMRetriever,
        vector_retriever: Any,  # ChromaDB or similar
        gfm_weight: float = 0.6,
        vector_weight: float = 0.4
    ):
        self.gfm_retriever = gfm_retriever
        self.vector_retriever = vector_retriever
        self.gfm_weight = gfm_weight
        self.vector_weight = vector_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining GFM and vector search.

        Args:
            query: Natural language query
            top_k: Number of results

        Returns:
            Fused and ranked results
        """
        # Get GFM results
        gfm_results = self.gfm_retriever.retrieve(query, top_k=top_k * 2)

        # Get vector results
        vector_results = self._get_vector_results(query, top_k * 2)

        # Fuse results
        scores = {}
        metadata = {}

        for result in gfm_results:
            doc_id = result.document_id
            scores[doc_id] = scores.get(doc_id, 0) + result.relevance_score * self.gfm_weight
            if doc_id not in metadata:
                metadata[doc_id] = {"gfm_score": result.relevance_score, "entity_path": result.entity_path}
            else:
                metadata[doc_id]["gfm_score"] = result.relevance_score
                metadata[doc_id]["entity_path"] = result.entity_path

        for doc_id, score in vector_results:
            scores[doc_id] = scores.get(doc_id, 0) + score * self.vector_weight
            if doc_id not in metadata:
                metadata[doc_id] = {"vector_score": score}
            else:
                metadata[doc_id]["vector_score"] = score

        # Sort and return
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                document_id=doc_id,
                relevance_score=score,
                entity_path=metadata.get(doc_id, {}).get("entity_path", []),
                metadata=metadata.get(doc_id, {})
            )
            for doc_id, score in sorted_results[:top_k]
        ]

    def _get_vector_results(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Get results from vector retriever."""
        try:
            if hasattr(self.vector_retriever, "query"):
                results = self.vector_retriever.query(query, n_results=top_k)
                # Parse results based on ChromaDB format
                if isinstance(results, dict) and "ids" in results:
                    ids = results["ids"][0] if results["ids"] else []
                    distances = results.get("distances", [[]])[0]
                    # Convert distances to similarity scores
                    scores = [1.0 - d for d in distances] if distances else []
                    return list(zip(ids, scores))
            return []
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")
            return []
