"""
Link Prediction and Knowledge Base Completion.

Predicts missing links in the knowledge graph using:
- Graph structure features (node similarity, path-based)
- Embedding-based prediction
- Rule-based inference (from learned rules)
- LLM-assisted prediction

Research Foundation:
- LeSR (2501.01246v1): LLM-enhanced symbolic reasoning for KBC
- GFM-RAG: Graph embeddings for entity/relation prediction
- TransE, RotatE: Knowledge graph embedding methods

Option B Compliance:
- Uses pre-computed embeddings (no training during inference)
- Rule-based predictions from learned rules
- LLM generates candidates, doesn't train
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from enum import Enum
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


class PredictionMethod(str, Enum):
    """Methods for link prediction."""
    STRUCTURAL = "structural"  # Based on graph structure only
    EMBEDDING = "embedding"  # Based on embeddings
    RULE_BASED = "rule_based"  # Based on learned rules
    LLM_ASSISTED = "llm_assisted"  # Using LLM
    HYBRID = "hybrid"  # Combination of methods


class LinkType(str, Enum):
    """Types of predicted links."""
    ENTITY_ENTITY = "entity_entity"  # Relation between entities
    ENTITY_CONCEPT = "entity_concept"  # Entity belongs to concept
    TEMPORAL = "temporal"  # Temporal relationship
    CAUSAL = "causal"  # Causal relationship


@dataclass
class PredictedLink:
    """A predicted link in the knowledge graph."""
    head_id: str
    head_name: str
    relation: str
    tail_id: str
    tail_name: str

    # Confidence scores
    confidence: float = 0.0
    method_scores: Dict[PredictionMethod, float] = field(default_factory=dict)

    # Metadata
    link_type: LinkType = LinkType.ENTITY_ENTITY
    prediction_method: PredictionMethod = PredictionMethod.HYBRID
    evidence: List[str] = field(default_factory=list)
    predicted_at: datetime = field(default_factory=datetime.utcnow)

    # For validation
    rule_id: Optional[str] = None  # If rule-based
    similar_existing_links: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head_id": self.head_id,
            "head_name": self.head_name,
            "relation": self.relation,
            "tail_id": self.tail_id,
            "tail_name": self.tail_name,
            "confidence": self.confidence,
            "link_type": self.link_type.value,
            "prediction_method": self.prediction_method.value,
            "evidence": self.evidence,
        }

    @property
    def triple(self) -> Tuple[str, str, str]:
        """Return as (head, relation, tail) triple."""
        return (self.head_id, self.relation, self.tail_id)


@dataclass
class CompletionCandidate:
    """A candidate for knowledge base completion."""
    head_id: str
    relation: str
    tail_id: str
    score: float
    method: PredictionMethod
    supporting_paths: List[List[str]] = field(default_factory=list)


class LinkPredictor:
    """
    Predicts missing links in the knowledge graph.

    Combines multiple prediction methods:
    1. Structural features (common neighbors, path-based)
    2. Embedding similarity
    3. Rule-based inference
    4. LLM-assisted prediction
    """

    def __init__(
        self,
        neo4j_driver: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        rule_store: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        min_confidence: float = 0.3
    ):
        """Initialize link predictor.

        Args:
            neo4j_driver: Neo4j driver for graph queries
            embedding_model: Model for embedding-based prediction
            rule_store: Store of learned extraction rules
            llm_client: LLM client for assisted prediction
            min_confidence: Minimum confidence threshold
        """
        self.driver = neo4j_driver
        self.embedding_model = embedding_model
        self.rule_store = rule_store
        self.llm_client = llm_client
        self.min_confidence = min_confidence

        # Cache
        self._node_embeddings: Dict[str, List[float]] = {}
        self._relation_embeddings: Dict[str, List[float]] = {}

    async def predict_links(
        self,
        entity_id: str,
        relation: Optional[str] = None,
        max_predictions: int = 10,
        method: PredictionMethod = PredictionMethod.HYBRID
    ) -> List[PredictedLink]:
        """Predict missing links for an entity.

        Args:
            entity_id: Entity to predict links for
            relation: Optional specific relation to predict
            max_predictions: Maximum predictions to return
            method: Prediction method to use

        Returns:
            List of predicted links
        """
        predictions = []

        # Get entity info
        entity_name = await self._get_entity_name(entity_id)

        if method == PredictionMethod.STRUCTURAL:
            predictions = await self._predict_structural(entity_id, relation, max_predictions * 2)
        elif method == PredictionMethod.EMBEDDING:
            predictions = await self._predict_embedding(entity_id, relation, max_predictions * 2)
        elif method == PredictionMethod.RULE_BASED:
            predictions = await self._predict_rule_based(entity_id, relation, max_predictions * 2)
        elif method == PredictionMethod.LLM_ASSISTED:
            predictions = await self._predict_llm(entity_id, entity_name, relation, max_predictions * 2)
        else:  # HYBRID
            # Combine all methods
            structural = await self._predict_structural(entity_id, relation, max_predictions)
            embedding = await self._predict_embedding(entity_id, relation, max_predictions)
            rule_based = await self._predict_rule_based(entity_id, relation, max_predictions)

            # Merge and re-score
            predictions = self._merge_predictions(
                structural, embedding, rule_based,
                weights={"structural": 0.3, "embedding": 0.3, "rule_based": 0.4}
            )

        # Filter by confidence
        predictions = [p for p in predictions if p.confidence >= self.min_confidence]

        # Sort and limit
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:max_predictions]

    def predict_links_sync(
        self,
        entity_id: str,
        relation: Optional[str] = None,
        max_predictions: int = 10,
        method: PredictionMethod = PredictionMethod.HYBRID
    ) -> List[PredictedLink]:
        """Synchronous version of predict_links."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.predict_links(entity_id, relation, max_predictions, method)
        )

    async def _predict_structural(
        self,
        entity_id: str,
        relation: Optional[str],
        limit: int
    ) -> List[PredictedLink]:
        """Predict links using structural features."""
        predictions = []

        if not self.driver:
            return predictions

        # Common neighbors approach
        cn_predictions = await self._common_neighbors(entity_id, relation, limit)
        predictions.extend(cn_predictions)

        # Jaccard similarity
        jac_predictions = await self._jaccard_similarity(entity_id, relation, limit)
        predictions.extend(jac_predictions)

        # Adamic-Adar index
        aa_predictions = await self._adamic_adar(entity_id, relation, limit)
        predictions.extend(aa_predictions)

        # Deduplicate
        seen = set()
        unique = []
        for p in predictions:
            key = p.triple
            if key not in seen:
                seen.add(key)
                p.prediction_method = PredictionMethod.STRUCTURAL
                unique.append(p)

        return unique

    async def _common_neighbors(
        self,
        entity_id: str,
        relation: Optional[str],
        limit: int
    ) -> List[PredictedLink]:
        """Predict based on common neighbors."""
        predictions = []

        rel_filter = f":{relation}" if relation else ""

        query = f"""
        MATCH (a {{id: $entity_id}})-[{rel_filter}]-(common)-[{rel_filter}]-(candidate)
        WHERE a <> candidate
        AND NOT (a)-[]-(candidate)
        WITH candidate, count(DISTINCT common) as cn_count, collect(DISTINCT common.name) as neighbors
        ORDER BY cn_count DESC
        LIMIT $limit
        RETURN candidate.id as id, candidate.name as name, cn_count, neighbors
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, entity_id=entity_id, limit=limit)
                for record in result:
                    # Score based on common neighbors count
                    score = min(1.0, record["cn_count"] / 10.0)

                    predictions.append(PredictedLink(
                        head_id=entity_id,
                        head_name=await self._get_entity_name(entity_id),
                        relation=relation or "RELATED_TO",
                        tail_id=record["id"],
                        tail_name=record["name"] or record["id"],
                        confidence=score,
                        evidence=[f"Common neighbors: {', '.join(record['neighbors'][:3])}"],
                    ))
        except Exception as e:
            logger.warning(f"Common neighbors query failed: {e}")

        return predictions

    async def _jaccard_similarity(
        self,
        entity_id: str,
        relation: Optional[str],
        limit: int
    ) -> List[PredictedLink]:
        """Predict based on Jaccard similarity of neighborhoods."""
        predictions = []

        rel_filter = f":{relation}" if relation else ""

        query = f"""
        MATCH (a {{id: $entity_id}})-[{rel_filter}]-(neighbor)
        WITH a, collect(DISTINCT neighbor) as a_neighbors
        MATCH (candidate)-[{rel_filter}]-(neighbor2)
        WHERE candidate <> a AND NOT (a)-[]-(candidate)
        WITH a, a_neighbors, candidate, collect(DISTINCT neighbor2) as c_neighbors
        WITH candidate,
             size([x IN a_neighbors WHERE x IN c_neighbors]) as intersection,
             size(a_neighbors + [x IN c_neighbors WHERE NOT x IN a_neighbors]) as union_size
        WHERE union_size > 0
        WITH candidate, toFloat(intersection) / toFloat(union_size) as jaccard
        ORDER BY jaccard DESC
        LIMIT $limit
        RETURN candidate.id as id, candidate.name as name, jaccard
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, entity_id=entity_id, limit=limit)
                for record in result:
                    predictions.append(PredictedLink(
                        head_id=entity_id,
                        head_name=await self._get_entity_name(entity_id),
                        relation=relation or "SIMILAR_TO",
                        tail_id=record["id"],
                        tail_name=record["name"] or record["id"],
                        confidence=record["jaccard"],
                        evidence=[f"Jaccard similarity: {record['jaccard']:.2%}"],
                    ))
        except Exception as e:
            logger.warning(f"Jaccard query failed: {e}")

        return predictions

    async def _adamic_adar(
        self,
        entity_id: str,
        relation: Optional[str],
        limit: int
    ) -> List[PredictedLink]:
        """Predict using Adamic-Adar index."""
        predictions = []

        rel_filter = f":{relation}" if relation else ""

        query = f"""
        MATCH (a {{id: $entity_id}})-[{rel_filter}]-(common)-[{rel_filter}]-(candidate)
        WHERE a <> candidate AND NOT (a)-[]-(candidate)
        WITH candidate, common
        MATCH (common)-[]-()
        WITH candidate, common, count(*) as degree
        WHERE degree > 1
        WITH candidate, sum(1.0 / log(degree)) as aa_score
        ORDER BY aa_score DESC
        LIMIT $limit
        RETURN candidate.id as id, candidate.name as name, aa_score
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, entity_id=entity_id, limit=limit)
                max_score = 1.0
                for i, record in enumerate(result):
                    if i == 0:
                        max_score = max(record["aa_score"], 1.0)

                    # Normalize score
                    score = min(1.0, record["aa_score"] / max_score)

                    predictions.append(PredictedLink(
                        head_id=entity_id,
                        head_name=await self._get_entity_name(entity_id),
                        relation=relation or "RELATED_TO",
                        tail_id=record["id"],
                        tail_name=record["name"] or record["id"],
                        confidence=score,
                        evidence=[f"Adamic-Adar score: {record['aa_score']:.2f}"],
                    ))
        except Exception as e:
            logger.warning(f"Adamic-Adar query failed: {e}")

        return predictions

    async def _predict_embedding(
        self,
        entity_id: str,
        relation: Optional[str],
        limit: int
    ) -> List[PredictedLink]:
        """Predict using embedding similarity."""
        predictions = []

        if not self.embedding_model:
            return predictions

        # Get entity embedding
        entity_emb = await self._get_entity_embedding(entity_id)
        if entity_emb is None:
            return predictions

        # For TransE-style: tail = head + relation
        # For RotatE-style: tail = head * relation
        # Here we use simple similarity

        # Get candidate embeddings
        candidates = await self._get_candidate_entities(entity_id, limit * 2)

        for candidate_id in candidates:
            candidate_emb = await self._get_entity_embedding(candidate_id)
            if candidate_emb is None:
                continue

            # Compute similarity
            similarity = self._cosine_similarity(entity_emb, candidate_emb)

            if similarity > 0.3:  # Threshold
                predictions.append(PredictedLink(
                    head_id=entity_id,
                    head_name=await self._get_entity_name(entity_id),
                    relation=relation or "SIMILAR_TO",
                    tail_id=candidate_id,
                    tail_name=await self._get_entity_name(candidate_id),
                    confidence=similarity,
                    prediction_method=PredictionMethod.EMBEDDING,
                    evidence=[f"Embedding similarity: {similarity:.2%}"],
                ))

        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:limit]

    async def _predict_rule_based(
        self,
        entity_id: str,
        relation: Optional[str],
        limit: int
    ) -> List[PredictedLink]:
        """Predict using learned rules."""
        predictions = []

        if not self.rule_store:
            return predictions

        # Get applicable rules
        active_rules = self.rule_store.get_active_rules() if hasattr(self.rule_store, 'get_active_rules') else []

        for rule in active_rules:
            if relation and rule.conclusion.relation_type != relation:
                continue

            # Check if entity matches rule pattern
            matches = self._check_rule_match(entity_id, rule)

            for match in matches:
                predictions.append(PredictedLink(
                    head_id=entity_id,
                    head_name=await self._get_entity_name(entity_id),
                    relation=rule.conclusion.relation_type,
                    tail_id=match["target_id"],
                    tail_name=match.get("target_name", match["target_id"]),
                    confidence=rule.confidence,
                    prediction_method=PredictionMethod.RULE_BASED,
                    rule_id=rule.id,
                    evidence=[f"Rule: {rule.name}", f"Rule confidence: {rule.confidence:.2%}"],
                ))

        return predictions[:limit]

    def _check_rule_match(self, entity_id: str, rule: Any) -> List[Dict[str, Any]]:
        """Check if entity matches a rule pattern."""
        matches = []

        if not self.driver or not rule.pattern:
            return matches

        # Build query from rule pattern
        try:
            query = f"""
            MATCH (start {{id: $entity_id}})
            {rule.pattern.to_cypher().replace('MATCH', '')}
            WHERE NOT (start)-[:{rule.conclusion.relation_type}]->(target)
            RETURN target.id as target_id, target.name as target_name
            LIMIT 10
            """

            with self.driver.session() as session:
                result = session.run(query, entity_id=entity_id)
                for record in result:
                    matches.append({
                        "target_id": record["target_id"],
                        "target_name": record["target_name"],
                    })
        except Exception as e:
            logger.debug(f"Rule match check failed: {e}")

        return matches

    async def _predict_llm(
        self,
        entity_id: str,
        entity_name: str,
        relation: Optional[str],
        limit: int
    ) -> List[PredictedLink]:
        """Predict using LLM assistance."""
        predictions = []

        if not self.llm_client:
            return predictions

        # Get entity context
        context = await self._get_entity_context(entity_id)

        prompt = f"""Given the following entity and its context in a personal knowledge graph:

Entity: {entity_name}

Context:
{context}

Predict {limit} entities that might be related to "{entity_name}" {f'via the "{relation}" relationship' if relation else ''}.

For each prediction, provide:
1. Entity name
2. Relationship type
3. Confidence (0.0-1.0)
4. Reasoning

Format: entity_name | relationship | confidence | reasoning
"""

        try:
            if hasattr(self.llm_client, "generate"):
                response = await self.llm_client.generate(prompt)
            elif hasattr(self.llm_client, "chat"):
                response = await self.llm_client.chat([{"role": "user", "content": prompt}])
                response = response.get("content", "")
            else:
                return predictions

            # Parse response
            lines = response.strip().split("\n")
            for line in lines:
                if "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        try:
                            confidence = float(parts[2])
                            predictions.append(PredictedLink(
                                head_id=entity_id,
                                head_name=entity_name,
                                relation=parts[1] if len(parts) > 1 else "RELATED_TO",
                                tail_id=parts[0].lower().replace(" ", "_"),
                                tail_name=parts[0],
                                confidence=min(1.0, max(0.0, confidence)),
                                prediction_method=PredictionMethod.LLM_ASSISTED,
                                evidence=[parts[3] if len(parts) > 3 else "LLM prediction"],
                            ))
                        except ValueError:
                            continue
        except Exception as e:
            logger.warning(f"LLM prediction failed: {e}")

        return predictions[:limit]

    def _merge_predictions(
        self,
        *prediction_lists: List[PredictedLink],
        weights: Dict[str, float] = None
    ) -> List[PredictedLink]:
        """Merge predictions from multiple methods."""
        weights = weights or {}

        # Group by (head, relation, tail)
        grouped: Dict[Tuple, List[PredictedLink]] = defaultdict(list)

        for preds in prediction_lists:
            for pred in preds:
                key = pred.triple
                grouped[key].append(pred)

        # Merge scores
        merged = []
        for key, preds in grouped.items():
            # Combine confidences
            total_weight = 0.0
            weighted_score = 0.0

            method_scores = {}
            evidence = []

            for pred in preds:
                method = pred.prediction_method.value
                weight = weights.get(method, 0.33)

                method_scores[pred.prediction_method] = pred.confidence
                weighted_score += pred.confidence * weight
                total_weight += weight
                evidence.extend(pred.evidence)

            final_score = weighted_score / total_weight if total_weight > 0 else 0

            # Use first prediction as template
            merged_pred = preds[0]
            merged_pred.confidence = final_score
            merged_pred.method_scores = method_scores
            merged_pred.prediction_method = PredictionMethod.HYBRID
            merged_pred.evidence = list(set(evidence))  # Deduplicate

            merged.append(merged_pred)

        return merged

    async def _get_entity_name(self, entity_id: str) -> str:
        """Get entity name from ID."""
        if not self.driver:
            return entity_id

        query = "MATCH (n {id: $id}) RETURN n.name as name"
        try:
            with self.driver.session() as session:
                result = session.run(query, id=entity_id)
                record = result.single()
                if record and record["name"]:
                    return record["name"]
        except Exception:
            pass

        return entity_id

    async def _get_entity_context(self, entity_id: str) -> str:
        """Get context about an entity from the graph."""
        if not self.driver:
            return ""

        query = """
        MATCH (n {id: $id})
        OPTIONAL MATCH (n)-[r]->(m)
        WITH n, collect(type(r) + ' -> ' + coalesce(m.name, m.id)) as outgoing
        OPTIONAL MATCH (n)<-[r2]-(m2)
        WITH n, outgoing, collect(coalesce(m2.name, m2.id) + ' -> ' + type(r2)) as incoming
        RETURN n.name as name, n.description as desc, outgoing, incoming
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, id=entity_id)
                record = result.single()
                if record:
                    lines = [f"Name: {record['name']}"]
                    if record["desc"]:
                        lines.append(f"Description: {record['desc']}")
                    if record["outgoing"]:
                        lines.append(f"Outgoing relations: {', '.join(record['outgoing'][:5])}")
                    if record["incoming"]:
                        lines.append(f"Incoming relations: {', '.join(record['incoming'][:5])}")
                    return "\n".join(lines)
        except Exception:
            pass

        return ""

    async def _get_entity_embedding(self, entity_id: str) -> Optional[List[float]]:
        """Get entity embedding."""
        if entity_id in self._node_embeddings:
            return self._node_embeddings[entity_id]

        if not self.embedding_model:
            return None

        # Get entity text for embedding
        name = await self._get_entity_name(entity_id)

        if hasattr(self.embedding_model, "encode"):
            embedding = self.embedding_model.encode(name).tolist()
            self._node_embeddings[entity_id] = embedding
            return embedding

        return None

    async def _get_candidate_entities(self, entity_id: str, limit: int) -> List[str]:
        """Get candidate entities that might be related."""
        candidates = []

        if not self.driver:
            return candidates

        query = """
        MATCH (a {id: $entity_id})-[*1..2]-(candidate)
        WHERE candidate <> a AND NOT (a)-[]-(candidate)
        RETURN DISTINCT candidate.id as id
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, entity_id=entity_id, limit=limit)
                for record in result:
                    candidates.append(record["id"])
        except Exception as e:
            logger.warning(f"Candidate query failed: {e}")

        return candidates

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


class KnowledgeBaseCompleter:
    """
    Completes the knowledge base by identifying and filling gaps.

    Workflow:
    1. Identify missing links
    2. Generate predictions
    3. Validate predictions
    4. Optionally add to graph
    """

    def __init__(
        self,
        link_predictor: Optional[LinkPredictor] = None,
        neo4j_driver: Optional[Any] = None,
        validation_threshold: float = 0.7,
        auto_add: bool = False
    ):
        """Initialize completer.

        Args:
            link_predictor: Predictor for missing links
            neo4j_driver: Neo4j driver for graph operations
            validation_threshold: Minimum confidence for auto-adding
            auto_add: Whether to automatically add high-confidence predictions
        """
        self.predictor = link_predictor or LinkPredictor(neo4j_driver=neo4j_driver)
        self.driver = neo4j_driver
        self.validation_threshold = validation_threshold
        self.auto_add = auto_add

        # Completion statistics
        self.stats = {
            "predictions_generated": 0,
            "predictions_validated": 0,
            "links_added": 0,
        }

    async def complete(
        self,
        entity_ids: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        max_predictions_per_entity: int = 5
    ) -> List[PredictedLink]:
        """Complete knowledge base for specified entities.

        Args:
            entity_ids: Entities to complete (all if None)
            relation_types: Relations to predict (all if None)
            max_predictions_per_entity: Max predictions per entity

        Returns:
            List of all predicted links
        """
        all_predictions = []

        # Get entities to process
        if entity_ids is None:
            entity_ids = await self._get_all_entity_ids()

        for entity_id in entity_ids:
            for relation in (relation_types or [None]):
                predictions = await self.predictor.predict_links(
                    entity_id,
                    relation=relation,
                    max_predictions=max_predictions_per_entity,
                )

                self.stats["predictions_generated"] += len(predictions)

                # Validate predictions
                for pred in predictions:
                    is_valid = await self._validate_prediction(pred)
                    self.stats["predictions_validated"] += 1

                    if is_valid:
                        all_predictions.append(pred)

                        # Auto-add if enabled and high confidence
                        if self.auto_add and pred.confidence >= self.validation_threshold:
                            await self._add_link(pred)
                            self.stats["links_added"] += 1

        return all_predictions

    async def identify_gaps(
        self,
        entity_types: Optional[List[str]] = None,
        expected_relations: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Identify gaps in the knowledge base.

        Args:
            entity_types: Types of entities to check
            expected_relations: Expected relations for each type

        Returns:
            List of identified gaps
        """
        gaps = []

        if not self.driver:
            return gaps

        # Check for entities missing expected relations
        if expected_relations:
            for entity_type, relations in expected_relations.items():
                for relation in relations:
                    query = f"""
                    MATCH (n:{entity_type})
                    WHERE NOT (n)-[:{relation}]->()
                    RETURN n.id as id, n.name as name
                    LIMIT 100
                    """

                    try:
                        with self.driver.session() as session:
                            result = session.run(query)
                            for record in result:
                                gaps.append({
                                    "entity_id": record["id"],
                                    "entity_name": record["name"],
                                    "missing_relation": relation,
                                    "entity_type": entity_type,
                                })
                    except Exception as e:
                        logger.warning(f"Gap identification query failed: {e}")

        return gaps

    async def _get_all_entity_ids(self) -> List[str]:
        """Get all entity IDs from the graph."""
        entity_ids = []

        if not self.driver:
            return entity_ids

        query = """
        MATCH (n)
        WHERE n.id IS NOT NULL
        RETURN n.id as id
        LIMIT 1000
        """

        try:
            with self.driver.session() as session:
                result = session.run(query)
                for record in result:
                    entity_ids.append(record["id"])
        except Exception as e:
            logger.warning(f"Entity query failed: {e}")

        return entity_ids

    async def _validate_prediction(self, prediction: PredictedLink) -> bool:
        """Validate a predicted link."""
        # Check if link already exists
        if not self.driver:
            return prediction.confidence >= self.validation_threshold

        query = """
        MATCH (a {id: $head_id})-[r]->(b {id: $tail_id})
        RETURN count(r) > 0 as exists
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    head_id=prediction.head_id,
                    tail_id=prediction.tail_id
                )
                record = result.single()
                if record and record["exists"]:
                    return False  # Link already exists
        except Exception:
            pass

        return prediction.confidence >= self.validation_threshold

    async def _add_link(self, prediction: PredictedLink) -> bool:
        """Add a predicted link to the graph."""
        if not self.driver:
            return False

        query = f"""
        MATCH (a {{id: $head_id}}), (b {{id: $tail_id}})
        CREATE (a)-[r:{prediction.relation} {{
            confidence: $confidence,
            predicted: true,
            predicted_at: datetime()
        }}]->(b)
        RETURN r
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    head_id=prediction.head_id,
                    tail_id=prediction.tail_id,
                    confidence=prediction.confidence
                )
                return result.single() is not None
        except Exception as e:
            logger.warning(f"Failed to add predicted link: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get completion statistics."""
        return self.stats.copy()
