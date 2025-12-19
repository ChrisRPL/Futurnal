"""
Knowledge Graph Index Builder for GFM-RAG.

Constructs a KG-index from documents by:
1. Extracting entities and relationships
2. Building graph structure
3. Creating node-to-document mappings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    entity_type: str
    description: Optional[str] = None
    embedding: Optional[Tensor] = None
    source_docs: Set[str] = field(default_factory=set)


@dataclass
class Relation:
    """Represents a relation between entities."""
    id: str
    head_id: str
    tail_id: str
    relation_type: str
    confidence: float = 1.0
    source_doc: Optional[str] = None


@dataclass
class KGIndex:
    """
    Knowledge Graph Index structure.

    Stores the graph in a format suitable for GFM processing.
    """
    # Entity storage
    entities: Dict[str, Entity] = field(default_factory=dict)

    # Relations
    relations: List[Relation] = field(default_factory=list)

    # Graph structure (for PyTorch)
    edge_index: Optional[Tensor] = None
    edge_type: Optional[Tensor] = None

    # Mappings
    entity_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_entity: Dict[int, str] = field(default_factory=dict)
    relation_type_to_idx: Dict[str, int] = field(default_factory=dict)
    node_to_docs: Dict[int, Set[str]] = field(default_factory=lambda: defaultdict(set))

    # Metadata
    num_entities: int = 0
    num_relations: int = 0
    num_relation_types: int = 0


class KGIndexBuilder:
    """
    Builds a KG-Index from documents.

    Integrates with Futurnal's existing PKG (Neo4j) to:
    1. Extract existing graph structure
    2. Convert to tensor format for GFM
    3. Maintain document-entity mappings
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        use_existing_embeddings: bool = True
    ):
        self.hidden_dim = hidden_dim
        self.use_existing_embeddings = use_existing_embeddings
        self._embedder = None

    @property
    def embedder(self):
        """Lazy-load embedder."""
        if self._embedder is None:
            from .model import GFMEmbedder
            self._embedder = GFMEmbedder()
        return self._embedder

    def build_from_documents(
        self,
        documents: List[Dict[str, Any]],
        extract_entities: bool = True
    ) -> KGIndex:
        """
        Build KG-Index from a list of documents.

        Args:
            documents: List of document dictionaries with 'id', 'content', 'entities', etc.
            extract_entities: Whether to extract new entities if not provided

        Returns:
            Constructed KGIndex
        """
        kg_index = KGIndex()

        # Process each document
        for doc in documents:
            doc_id = doc.get("id", doc.get("path", str(id(doc))))

            # Get or extract entities
            entities = doc.get("entities", [])
            if extract_entities and not entities:
                entities = self._extract_entities(doc.get("content", ""))

            # Add entities to index
            for ent in entities:
                entity_id = ent.get("id", f"{doc_id}:{ent.get('name', '')}")

                if entity_id not in kg_index.entities:
                    entity = Entity(
                        id=entity_id,
                        name=ent.get("name", ""),
                        entity_type=ent.get("type", "unknown"),
                        description=ent.get("description"),
                        source_docs={doc_id}
                    )
                    kg_index.entities[entity_id] = entity
                else:
                    kg_index.entities[entity_id].source_docs.add(doc_id)

            # Get or extract relations
            relations = doc.get("relations", [])
            if extract_entities and not relations:
                relations = self._extract_relations(doc.get("content", ""), entities)

            # Add relations
            for rel in relations:
                relation = Relation(
                    id=f"{doc_id}:{rel.get('head')}:{rel.get('type')}:{rel.get('tail')}",
                    head_id=rel.get("head"),
                    tail_id=rel.get("tail"),
                    relation_type=rel.get("type", "related_to"),
                    confidence=rel.get("confidence", 1.0),
                    source_doc=doc_id
                )
                kg_index.relations.append(relation)

        # Build tensor representations
        self._build_tensors(kg_index)

        return kg_index

    def build_from_neo4j(
        self,
        neo4j_driver,
        query_filter: Optional[str] = None
    ) -> KGIndex:
        """
        Build KG-Index from existing Neo4j PKG.

        Args:
            neo4j_driver: Neo4j driver instance
            query_filter: Optional Cypher WHERE clause to filter nodes

        Returns:
            Constructed KGIndex
        """
        kg_index = KGIndex()

        # Query entities
        entity_query = """
        MATCH (n)
        WHERE n:PersonNode OR n:OrganizationNode OR n:ConceptNode
            OR n:DocumentNode OR n:EventNode
        """
        if query_filter:
            entity_query += f" AND {query_filter}"
        entity_query += """
        RETURN n.id as id, labels(n)[0] as type, n.name as name,
               n.description as description
        """

        with neo4j_driver.session() as session:
            result = session.run(entity_query)
            for record in result:
                entity = Entity(
                    id=record["id"],
                    name=record["name"] or record["id"],
                    entity_type=record["type"],
                    description=record["description"]
                )
                kg_index.entities[entity.id] = entity

        # Query relationships
        relation_query = """
        MATCH (h)-[r]->(t)
        WHERE (h:PersonNode OR h:OrganizationNode OR h:ConceptNode
               OR h:DocumentNode OR h:EventNode)
          AND (t:PersonNode OR t:OrganizationNode OR t:ConceptNode
               OR t:DocumentNode OR t:EventNode)
        RETURN h.id as head_id, type(r) as rel_type, t.id as tail_id,
               r.confidence as confidence, r.source_path as source_doc
        """

        with neo4j_driver.session() as session:
            result = session.run(relation_query)
            for record in result:
                if record["head_id"] in kg_index.entities and record["tail_id"] in kg_index.entities:
                    relation = Relation(
                        id=f"{record['head_id']}:{record['rel_type']}:{record['tail_id']}",
                        head_id=record["head_id"],
                        tail_id=record["tail_id"],
                        relation_type=record["rel_type"],
                        confidence=record["confidence"] or 1.0,
                        source_doc=record["source_doc"]
                    )
                    kg_index.relations.append(relation)

        # Build document mappings from DocumentNode relationships
        doc_query = """
        MATCH (d:DocumentNode)-[r:EXTRACTED_FROM|DISCOVERED_IN]-(e)
        RETURN d.path as doc_path, e.id as entity_id
        """

        with neo4j_driver.session() as session:
            result = session.run(doc_query)
            for record in result:
                entity_id = record["entity_id"]
                doc_path = record["doc_path"]
                if entity_id in kg_index.entities:
                    kg_index.entities[entity_id].source_docs.add(doc_path)

        # Build tensor representations
        self._build_tensors(kg_index)

        return kg_index

    def _build_tensors(self, kg_index: KGIndex) -> None:
        """Convert graph to tensor format for GFM."""
        # Create entity index mapping
        for idx, entity_id in enumerate(kg_index.entities.keys()):
            kg_index.entity_to_idx[entity_id] = idx
            kg_index.idx_to_entity[idx] = entity_id

        kg_index.num_entities = len(kg_index.entities)

        # Create relation type mapping
        relation_types = set(r.relation_type for r in kg_index.relations)
        for idx, rel_type in enumerate(relation_types):
            kg_index.relation_type_to_idx[rel_type] = idx

        kg_index.num_relation_types = len(relation_types)

        # Build edge tensors
        edge_src = []
        edge_dst = []
        edge_types = []

        for relation in kg_index.relations:
            if relation.head_id in kg_index.entity_to_idx and relation.tail_id in kg_index.entity_to_idx:
                head_idx = kg_index.entity_to_idx[relation.head_id]
                tail_idx = kg_index.entity_to_idx[relation.tail_id]
                rel_type_idx = kg_index.relation_type_to_idx[relation.relation_type]

                edge_src.append(head_idx)
                edge_dst.append(tail_idx)
                edge_types.append(rel_type_idx)

                # Also add reverse edge for bidirectional message passing
                edge_src.append(tail_idx)
                edge_dst.append(head_idx)
                edge_types.append(rel_type_idx + kg_index.num_relation_types)  # Offset for inverse

        if edge_src:
            kg_index.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            kg_index.edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            # Empty graph fallback
            kg_index.edge_index = torch.zeros((2, 0), dtype=torch.long)
            kg_index.edge_type = torch.zeros(0, dtype=torch.long)

        kg_index.num_relations = len(kg_index.relations)

        # Build node-to-docs mapping
        for entity_id, entity in kg_index.entities.items():
            node_idx = kg_index.entity_to_idx[entity_id]
            for doc_id in entity.source_docs:
                kg_index.node_to_docs[node_idx].add(doc_id)

        # Compute entity embeddings
        entity_names = [kg_index.entities[kg_index.idx_to_entity[i]].name
                       for i in range(kg_index.num_entities)]
        if entity_names:
            embeddings = self.embedder.encode_entities(entity_names)
            for entity_id, entity in kg_index.entities.items():
                idx = kg_index.entity_to_idx[entity_id]
                entity.embedding = embeddings[idx]

    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from document content using simple NER."""
        # Simple entity extraction - can be enhanced with LLM
        entities = []

        # Use simple heuristics for now
        import re

        # Find capitalized phrases (potential named entities)
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(pattern, content)

        seen = set()
        for match in matches:
            if match not in seen and len(match) > 2:
                entities.append({
                    "name": match,
                    "type": "concept",  # Default type
                    "id": match.lower().replace(" ", "_")
                })
                seen.add(match)

        return entities[:50]  # Limit to prevent explosion

    def _extract_relations(
        self,
        content: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relations between entities from content."""
        relations = []

        # Simple co-occurrence based relation extraction
        entity_names = [e.get("name", "") for e in entities]

        # Find sentences with multiple entities
        sentences = content.split(".")
        for sentence in sentences:
            present_entities = [e for e in entity_names if e in sentence]
            if len(present_entities) >= 2:
                # Create relations between co-occurring entities
                for i, head in enumerate(present_entities[:-1]):
                    for tail in present_entities[i + 1:]:
                        relations.append({
                            "head": head.lower().replace(" ", "_"),
                            "tail": tail.lower().replace(" ", "_"),
                            "type": "related_to",
                            "confidence": 0.7
                        })

        return relations[:100]  # Limit relations

    def update_index(
        self,
        kg_index: KGIndex,
        new_documents: List[Dict[str, Any]]
    ) -> KGIndex:
        """
        Incrementally update an existing KG-Index with new documents.

        Args:
            kg_index: Existing index to update
            new_documents: New documents to add

        Returns:
            Updated KGIndex
        """
        # Process new documents and merge
        new_index = self.build_from_documents(new_documents)

        # Merge entities
        for entity_id, entity in new_index.entities.items():
            if entity_id in kg_index.entities:
                kg_index.entities[entity_id].source_docs.update(entity.source_docs)
            else:
                kg_index.entities[entity_id] = entity

        # Merge relations
        existing_rel_ids = {r.id for r in kg_index.relations}
        for relation in new_index.relations:
            if relation.id not in existing_rel_ids:
                kg_index.relations.append(relation)

        # Rebuild tensors
        self._build_tensors(kg_index)

        return kg_index
