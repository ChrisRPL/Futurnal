"""
GFM Training - Training utilities for Graph Foundation Model.

Supports:
- Supervised training with relevance labels
- Contrastive learning
- Cross-dataset generalization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .model import GraphFoundationModel, GFMConfig, GraphBatch
from .kg_index import KGIndex

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for GFM training."""
    # Training params
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 100

    # Loss settings
    contrastive_temp: float = 0.07
    margin: float = 0.3

    # Evaluation
    eval_every: int = 100
    save_every: int = 500

    # Paths
    checkpoint_dir: Path = Path("checkpoints/gfm")


@dataclass
class TrainingSample:
    """Single training sample for GFM."""
    query: str
    query_embedding: Tensor
    positive_entities: List[str]  # Relevant entities
    negative_entities: List[str]  # Non-relevant entities
    document_ids: List[str]


class GFMDataset(Dataset):
    """Dataset for GFM training."""

    def __init__(
        self,
        samples: List[TrainingSample],
        kg_index: KGIndex,
        max_negatives: int = 10
    ):
        self.samples = samples
        self.kg_index = kg_index
        self.max_negatives = max_negatives

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Get positive node indices
        pos_indices = []
        for entity_id in sample.positive_entities:
            if entity_id in self.kg_index.entity_to_idx:
                pos_indices.append(self.kg_index.entity_to_idx[entity_id])

        # Get negative node indices
        neg_indices = []
        for entity_id in sample.negative_entities[:self.max_negatives]:
            if entity_id in self.kg_index.entity_to_idx:
                neg_indices.append(self.kg_index.entity_to_idx[entity_id])

        return {
            "query_embedding": sample.query_embedding,
            "positive_indices": torch.tensor(pos_indices, dtype=torch.long),
            "negative_indices": torch.tensor(neg_indices, dtype=torch.long),
            "query": sample.query
        }


class GFMTrainer:
    """
    Trainer for Graph Foundation Model.

    Supports multiple training objectives:
    1. Binary classification: Is entity relevant to query?
    2. Contrastive learning: Push relevant closer, irrelevant farther
    3. Ranking loss: Order entities by relevance
    """

    def __init__(
        self,
        model: GraphFoundationModel,
        kg_index: KGIndex,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.kg_index = kg_index
        self.config = config or TrainingConfig()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.margin_loss = nn.MarginRankingLoss(margin=self.config.margin)

        # Training state
        self.global_step = 0
        self.best_metrics = {}

    def train(
        self,
        train_samples: List[TrainingSample],
        val_samples: Optional[List[TrainingSample]] = None
    ) -> Dict[str, float]:
        """
        Train the GFM model.

        Args:
            train_samples: Training samples
            val_samples: Validation samples

        Returns:
            Final training metrics
        """
        # Create datasets
        train_dataset = GFMDataset(train_samples, self.kg_index)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

        val_loader = None
        if val_samples:
            val_dataset = GFMDataset(val_samples, self.kg_index)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                collate_fn=self._collate_fn
            )

        # Prepare node features
        node_features = self._get_node_features()

        # Training loop
        self.model.train()
        metrics_history = []

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                loss = self._train_step(batch, node_features)
                epoch_loss += loss
                num_batches += 1
                self.global_step += 1

                if self.global_step % self.config.eval_every == 0 and val_loader:
                    val_metrics = self.evaluate(val_loader, node_features)
                    logger.info(f"Step {self.global_step}: {val_metrics}")

                if self.global_step % self.config.save_every == 0:
                    self._save_checkpoint()

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")
            metrics_history.append({"epoch": epoch, "loss": avg_loss})

        # Final evaluation
        if val_loader:
            final_metrics = self.evaluate(val_loader, node_features)
            logger.info(f"Final metrics: {final_metrics}")
            return final_metrics

        return {"loss": metrics_history[-1]["loss"] if metrics_history else 0.0}

    def _train_step(
        self,
        batch: Dict[str, Any],
        node_features: Tensor
    ) -> float:
        """Single training step."""
        self.optimizer.zero_grad()

        # Create graph batch
        graph_batch = GraphBatch(
            node_features=node_features,
            edge_index=self.kg_index.edge_index,
            edge_type=self.kg_index.edge_type,
            query_embedding=batch["query_embedding"]
        )

        # Forward pass
        output = self.model(
            node_features=graph_batch.node_features,
            edge_index=graph_batch.edge_index,
            edge_type=graph_batch.edge_type,
            query_embedding=graph_batch.query_embedding
        )

        # Compute loss
        loss = self._compute_loss(
            output["node_scores"],
            batch["positive_indices"],
            batch["negative_indices"]
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _compute_loss(
        self,
        node_scores: Tensor,
        positive_indices: Tensor,
        negative_indices: Tensor
    ) -> Tensor:
        """Compute combined training loss."""
        losses = []

        # Binary cross-entropy loss
        if len(positive_indices) > 0 and len(negative_indices) > 0:
            pos_scores = node_scores[positive_indices]
            neg_scores = node_scores[negative_indices]

            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)

            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([pos_labels, neg_labels])

            bce = self.bce_loss(all_scores, all_labels)
            losses.append(bce)

        # Margin ranking loss
        if len(positive_indices) > 0 and len(negative_indices) > 0:
            pos_score_mean = node_scores[positive_indices].mean()
            neg_score_mean = node_scores[negative_indices].mean()

            margin = self.margin_loss(
                pos_score_mean.unsqueeze(0),
                neg_score_mean.unsqueeze(0),
                torch.ones(1, device=node_scores.device)
            )
            losses.append(margin)

        # Contrastive loss
        if len(positive_indices) > 0 and len(negative_indices) > 0:
            pos_emb = self.model.layers[-1].attention.out_proj.weight[positive_indices].mean(0)
            neg_emb = self.model.layers[-1].attention.out_proj.weight[negative_indices].mean(0)

            # InfoNCE-style loss
            similarity = F.cosine_similarity(
                pos_emb.unsqueeze(0),
                neg_emb.unsqueeze(0)
            ) / self.config.contrastive_temp

            contrastive = -torch.log(
                torch.exp(-similarity) / (torch.exp(-similarity) + 1)
            )
            losses.append(contrastive * 0.1)

        return sum(losses) if losses else torch.tensor(0.0, requires_grad=True)

    def evaluate(
        self,
        val_loader: DataLoader,
        node_features: Tensor
    ) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()

        total_correct = 0
        total_samples = 0
        total_mrr = 0.0

        with torch.no_grad():
            for batch in val_loader:
                graph_batch = GraphBatch(
                    node_features=node_features,
                    edge_index=self.kg_index.edge_index,
                    edge_type=self.kg_index.edge_type,
                    query_embedding=batch["query_embedding"]
                )

                output = self.model(
                    node_features=graph_batch.node_features,
                    edge_index=graph_batch.edge_index,
                    edge_type=graph_batch.edge_type,
                    query_embedding=graph_batch.query_embedding
                )

                scores = output["node_scores"]

                # Accuracy: do positive entities score higher than negatives?
                pos_indices = batch["positive_indices"]
                neg_indices = batch["negative_indices"]

                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    pos_mean = scores[pos_indices].mean()
                    neg_mean = scores[neg_indices].mean()
                    total_correct += int(pos_mean > neg_mean)
                    total_samples += 1

                    # MRR
                    all_indices = torch.cat([pos_indices, neg_indices])
                    all_scores = scores[all_indices]
                    sorted_indices = torch.argsort(all_scores, descending=True)
                    pos_count = len(pos_indices)
                    for rank, idx in enumerate(sorted_indices):
                        if idx < pos_count:
                            total_mrr += 1.0 / (rank + 1)
                            break

        self.model.train()

        accuracy = total_correct / max(total_samples, 1)
        mrr = total_mrr / max(total_samples, 1)

        return {"accuracy": accuracy, "mrr": mrr}

    def _get_node_features(self) -> Tensor:
        """Get node features from KG index."""
        features = []
        for i in range(self.kg_index.num_entities):
            entity_id = self.kg_index.idx_to_entity[i]
            entity = self.kg_index.entities[entity_id]
            if entity.embedding is not None:
                features.append(entity.embedding)
            else:
                features.append(torch.zeros(self.model.config.hidden_dim))

        return torch.stack(features) if features else torch.zeros(1, self.model.config.hidden_dim)

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate function for DataLoader."""
        # Stack query embeddings
        query_embeddings = torch.stack([b["query_embedding"].squeeze(0) for b in batch])

        # Flatten indices
        all_pos = []
        all_neg = []
        for b in batch:
            all_pos.extend(b["positive_indices"].tolist())
            all_neg.extend(b["negative_indices"].tolist())

        return {
            "query_embedding": query_embeddings.mean(0, keepdim=True),
            "positive_indices": torch.tensor(all_pos, dtype=torch.long) if all_pos else torch.tensor([], dtype=torch.long),
            "negative_indices": torch.tensor(all_neg, dtype=torch.long) if all_neg else torch.tensor([], dtype=torch.long),
            "queries": [b["query"] for b in batch]
        }

    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.checkpoint_dir / f"gfm_step_{self.global_step}.pt"
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved checkpoint to {path}")


def create_training_samples_from_pkg(
    neo4j_driver,
    query_samples: List[Dict[str, Any]]
) -> List[TrainingSample]:
    """
    Create training samples from PKG and query examples.

    Args:
        neo4j_driver: Neo4j driver
        query_samples: List of {query, relevant_docs, irrelevant_docs}

    Returns:
        Training samples for GFM
    """
    from .model import GFMEmbedder
    embedder = GFMEmbedder()

    samples = []
    for qs in query_samples:
        query = qs["query"]
        query_embedding = embedder.encode_query(query)

        # Get entities for relevant docs
        pos_entities = []
        for doc_path in qs.get("relevant_docs", []):
            entities = _get_entities_for_doc(neo4j_driver, doc_path)
            pos_entities.extend(entities)

        # Get entities for irrelevant docs
        neg_entities = []
        for doc_path in qs.get("irrelevant_docs", []):
            entities = _get_entities_for_doc(neo4j_driver, doc_path)
            neg_entities.extend(entities)

        if pos_entities:
            samples.append(TrainingSample(
                query=query,
                query_embedding=query_embedding.squeeze(0),
                positive_entities=list(set(pos_entities)),
                negative_entities=list(set(neg_entities)),
                document_ids=qs.get("relevant_docs", [])
            ))

    return samples


def _get_entities_for_doc(neo4j_driver, doc_path: str) -> List[str]:
    """Get entity IDs associated with a document."""
    query = """
    MATCH (d:DocumentNode {path: $path})-[:EXTRACTED_FROM|DISCOVERED_IN]-(e)
    RETURN e.id as entity_id
    """
    entities = []
    with neo4j_driver.session() as session:
        result = session.run(query, path=doc_path)
        for record in result:
            entities.append(record["entity_id"])
    return entities
