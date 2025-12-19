"""
Graph Foundation Model - Query-dependent GNN for RAG.

Implements the core GFM architecture from GFM-RAG paper:
- Query-conditioned message passing
- Relational interaction modeling
- Multi-hop reasoning in single step
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Fundamental relational interactions in knowledge graphs."""
    HEAD_TO_TAIL = "h2t"  # Subject to object
    TAIL_TO_HEAD = "t2h"  # Object to subject
    HEAD_TO_HEAD = "h2h"  # Co-subject relations
    TAIL_TO_TAIL = "t2t"  # Co-object relations


@dataclass
class GFMConfig:
    """Configuration for Graph Foundation Model."""

    # Model dimensions
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1

    # Graph structure
    num_relation_types: int = 100
    max_nodes: int = 10000

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Inference
    top_k_entities: int = 50
    relevance_threshold: float = 0.5


@dataclass
class GraphBatch:
    """Batched graph data for GFM processing."""

    # Node features [num_nodes, hidden_dim]
    node_features: Tensor

    # Edge information
    edge_index: Tensor  # [2, num_edges]
    edge_type: Tensor   # [num_edges]

    # Query information
    query_embedding: Tensor  # [batch_size, hidden_dim]

    # Mapping
    node_to_doc: Dict[int, str] = field(default_factory=dict)
    batch_idx: Optional[Tensor] = None


class QueryConditionedAttention(nn.Module):
    """
    Query-conditioned multi-head attention for message passing.

    Implements attention where the query influences both key and value
    computations, enabling query-dependent graph reasoning.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query projection (conditioned on both node and query)
        self.q_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        node_features: Tensor,
        query_embedding: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        relation_embeddings: Tensor
    ) -> Tensor:
        """
        Apply query-conditioned attention for message passing.

        Args:
            node_features: [num_nodes, hidden_dim]
            query_embedding: [1, hidden_dim] or [batch_size, hidden_dim]
            edge_index: [2, num_edges]
            edge_type: [num_edges]
            relation_embeddings: [num_relations, hidden_dim]

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        num_nodes = node_features.size(0)

        # Expand query to all nodes
        if query_embedding.size(0) == 1:
            query_expanded = query_embedding.expand(num_nodes, -1)
        else:
            query_expanded = query_embedding

        # Concatenate node features with query for conditioning
        node_query = torch.cat([node_features, query_expanded], dim=-1)

        # Compute queries and keys with query conditioning
        q = self.q_proj(node_query)
        k = self.k_proj(node_query)
        v = self.v_proj(node_features)

        # Reshape for multi-head attention
        q = q.view(num_nodes, self.num_heads, self.head_dim)
        k = k.view(num_nodes, self.num_heads, self.head_dim)
        v = v.view(num_nodes, self.num_heads, self.head_dim)

        # Compute attention scores along edges
        src, dst = edge_index

        # Get relation-specific bias
        rel_emb = relation_embeddings[edge_type]  # [num_edges, hidden_dim]
        rel_bias = rel_emb.view(-1, self.num_heads, self.head_dim)

        # Attention scores: q[dst] * (k[src] + rel_bias)
        q_dst = q[dst]  # [num_edges, num_heads, head_dim]
        k_src = k[src] + rel_bias

        attn_scores = (q_dst * k_src).sum(dim=-1) / self.scale  # [num_edges, num_heads]

        # Softmax over incoming edges for each node
        attn_weights = self._sparse_softmax(attn_scores, dst, num_nodes)
        attn_weights = self.dropout(attn_weights)

        # Message passing: aggregate values weighted by attention
        v_src = v[src]  # [num_edges, num_heads, head_dim]
        messages = attn_weights.unsqueeze(-1) * v_src  # [num_edges, num_heads, head_dim]

        # Aggregate messages to destination nodes
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=node_features.device)
        dst_expanded = dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.head_dim)
        out.scatter_add_(0, dst_expanded, messages)

        # Reshape and project output
        out = out.view(num_nodes, self.hidden_dim)
        out = self.out_proj(out)

        return out

    def _sparse_softmax(self, scores: Tensor, indices: Tensor, num_nodes: int) -> Tensor:
        """Compute softmax over sparse edge scores grouped by destination node."""
        # Subtract max for numerical stability
        max_scores = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        max_scores.scatter_reduce_(0, indices.unsqueeze(-1).expand_as(scores), scores, reduce="amax")
        scores = scores - max_scores[indices]

        # Exp and sum
        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        sum_exp.scatter_add_(0, indices.unsqueeze(-1).expand_as(exp_scores), exp_scores)

        # Normalize
        return exp_scores / (sum_exp[indices] + 1e-8)


class RelationalInteractionLayer(nn.Module):
    """
    Models four fundamental relational interactions.

    Based on ULTRA's relation graph approach:
    - h2t: Head-to-tail (direct relation)
    - t2h: Tail-to-head (inverse relation)
    - h2h: Head-to-head (co-subject)
    - t2t: Tail-to-tail (co-object)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()

        self.interaction_transforms = nn.ModuleDict({
            rt.value: nn.Linear(hidden_dim, hidden_dim)
            for rt in RelationType
        })

        self.combine = nn.Linear(hidden_dim * 4, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        relation_embeddings: Tensor,
        edge_index: Tensor,
        edge_type: Tensor
    ) -> Tensor:
        """
        Compute relation-aware node representations.

        Args:
            relation_embeddings: [num_relations, hidden_dim]
            edge_index: [2, num_edges]
            edge_type: [num_edges]

        Returns:
            Updated relation embeddings [num_relations, hidden_dim]
        """
        num_relations = relation_embeddings.size(0)
        hidden_dim = relation_embeddings.size(1)
        device = relation_embeddings.device

        # Compute all four interaction types
        interactions = []
        for rt in RelationType:
            transformed = self.interaction_transforms[rt.value](relation_embeddings)
            interactions.append(transformed)

        # Combine interactions
        combined = torch.cat(interactions, dim=-1)
        out = self.combine(combined)

        # Residual connection and layer norm
        out = self.layer_norm(relation_embeddings + out)

        return out


class GFMLayer(nn.Module):
    """Single layer of the Graph Foundation Model."""

    def __init__(self, config: GFMConfig):
        super().__init__()

        self.attention = QueryConditionedAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )

        self.relational = RelationalInteractionLayer(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        node_features: Tensor,
        query_embedding: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        relation_embeddings: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply one GFM layer.

        Returns:
            Updated (node_features, relation_embeddings)
        """
        # Message passing with query conditioning
        attn_out = self.attention(
            node_features, query_embedding, edge_index, edge_type, relation_embeddings
        )
        node_features = self.norm1(node_features + attn_out)

        # Feed-forward
        ffn_out = self.ffn(node_features)
        node_features = self.norm2(node_features + ffn_out)

        # Update relation embeddings
        relation_embeddings = self.relational(relation_embeddings, edge_index, edge_type)

        return node_features, relation_embeddings


class GraphFoundationModel(nn.Module):
    """
    Graph Foundation Model for Retrieval-Augmented Generation.

    Enables single-step multi-hop reasoning over knowledge graphs
    with query-dependent message passing.
    """

    def __init__(self, config: GFMConfig):
        super().__init__()
        self.config = config

        # Embedding layers
        self.node_embedding = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.relation_embedding = nn.Embedding(config.num_relation_types, config.hidden_dim)
        self.query_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

        # GFM layers
        self.layers = nn.ModuleList([
            GFMLayer(config) for _ in range(config.num_layers)
        ])

        # Scoring head
        self.scoring_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        query_embedding: Tensor,
        return_node_scores: bool = True
    ) -> Dict[str, Tensor]:
        """
        Forward pass of GFM.

        Args:
            node_features: Initial node features [num_nodes, hidden_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Relation types for edges [num_edges]
            query_embedding: Query vector [batch_size, hidden_dim]
            return_node_scores: Whether to compute relevance scores

        Returns:
            Dictionary with:
            - node_embeddings: Final node representations
            - relation_embeddings: Updated relation embeddings
            - node_scores: Relevance scores per node (if requested)
        """
        # Project inputs
        node_features = self.node_embedding(node_features)
        query_embedding = self.query_projection(query_embedding)
        relation_embeddings = self.relation_embedding.weight

        # Apply GFM layers
        for layer in self.layers:
            node_features, relation_embeddings = layer(
                node_features, query_embedding, edge_index, edge_type, relation_embeddings
            )

        result = {
            "node_embeddings": node_features,
            "relation_embeddings": relation_embeddings
        }

        if return_node_scores:
            # Compute relevance scores: how relevant is each node to the query?
            query_expanded = query_embedding.expand(node_features.size(0), -1)
            combined = torch.cat([node_features, query_expanded], dim=-1)
            scores = self.scoring_head(combined).squeeze(-1)
            result["node_scores"] = torch.sigmoid(scores)

        return result

    def retrieve(
        self,
        graph_batch: GraphBatch,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k relevant documents for a query.

        Args:
            graph_batch: Batched graph data with query
            top_k: Number of documents to retrieve

        Returns:
            List of (document_id, relevance_score) tuples
        """
        top_k = top_k or self.config.top_k_entities

        with torch.no_grad():
            output = self.forward(
                node_features=graph_batch.node_features,
                edge_index=graph_batch.edge_index,
                edge_type=graph_batch.edge_type,
                query_embedding=graph_batch.query_embedding
            )

        scores = output["node_scores"]

        # Get top-k nodes
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))

        # Map to documents
        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            if idx in graph_batch.node_to_doc:
                doc_id = graph_batch.node_to_doc[idx]
                results.append((doc_id, score))

        return results


class GFMEmbedder:
    """
    Utility class to create embeddings compatible with GFM.

    Converts text queries and entities to embeddings using
    a frozen text encoder.
    """

    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(encoder_name)
            self.hidden_dim = self.encoder.get_sentence_embedding_dimension()
        except ImportError:
            logger.warning("sentence-transformers not available, using random embeddings")
            self.encoder = None
            self.hidden_dim = 256

    def encode_query(self, query: str) -> Tensor:
        """Encode a text query to a tensor."""
        if self.encoder is not None:
            embedding = self.encoder.encode(query, convert_to_tensor=True)
            return embedding.unsqueeze(0)
        else:
            return torch.randn(1, self.hidden_dim)

    def encode_entities(self, entities: List[str]) -> Tensor:
        """Encode a list of entity names to tensors."""
        if self.encoder is not None:
            embeddings = self.encoder.encode(entities, convert_to_tensor=True)
            return embeddings
        else:
            return torch.randn(len(entities), self.hidden_dim)
