"""
Schema Discovery Engine

Discover new schema elements (entity and relationship types)
from document patterns using LLM analysis.

Research Foundation:
- AutoSchemaKG (2505.23628v1): Autonomous schema induction
- Target: >90% semantic alignment with manual schemas
- Multi-phase extraction: Entity-Entity → Entity-Event → Event-Event
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from futurnal.extraction.schema.models import EntityType, RelationshipType, SchemaDiscovery

if TYPE_CHECKING:
    from futurnal.extraction.schema.evolution import Document

logger = logging.getLogger(__name__)

# NER label to entity type mapping (per AutoSchemaKG approach)
NER_TYPE_HINTS: Dict[str, str] = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "DATE": "TemporalMarker",
    "TIME": "TemporalMarker",
    "EVENT": "Event",
    "PRODUCT": "Product",
    "WORK_OF_ART": "Document",
    "FAC": "Facility",
    "NORP": "Group",
    "MONEY": "Value",
    "QUANTITY": "Value",
    "PERCENT": "Value",
    "CARDINAL": "Value",
    "ORDINAL": "Value",
    "LAW": "Document",
}


class SchemaDiscoveryEngine:
    """
    Discover new schema elements from document patterns.

    Uses NLP analysis (spaCy) and semantic clustering (ChromaDB)
    to identify entity and relationship patterns.

    Research Foundation:
    - AutoSchemaKG: Multi-phase discovery
      Phase 1: Entity-Entity relationships
      Phase 2: Entity-Event relationships
      Phase 3: Event-Event relationships (causal candidates)
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        discovery_threshold: float = 0.75,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
    ):
        """
        Initialize discovery engine.

        Args:
            llm: LLM interface for pattern analysis (optional)
            discovery_threshold: Min confidence to propose discovery (default 0.75)
            similarity_threshold: Min similarity for clustering (default 0.85)
            min_cluster_size: Min examples needed to propose type (default 3)
        """
        self.llm = llm
        self.discovery_threshold = discovery_threshold
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

        # Lazy-loaded NLP model
        self._nlp = None
        self._embedding_client = None

    @property
    def nlp(self):
        """
        Lazy load spaCy model.

        Attempts to load en_core_web_md for word vectors,
        falls back to en_core_web_sm if unavailable.
        """
        if self._nlp is None:
            try:
                import spacy

                try:
                    self._nlp = spacy.load("en_core_web_md")
                    logger.info("Loaded spaCy model: en_core_web_md")
                except OSError:
                    self._nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy model: en_core_web_sm (fallback)")
            except ImportError:
                logger.error("spaCy not installed. Run: pip install spacy")
                raise ImportError(
                    "spaCy is required for schema discovery. "
                    "Install with: pip install spacy && python -m spacy download en_core_web_sm"
                )
        return self._nlp

    def discover_entity_patterns(
        self, documents: List["Document"]
    ) -> List[SchemaDiscovery]:
        """
        Discover entity patterns from documents using NLP and clustering.

        Per AutoSchemaKG approach:
        1. Extract noun phrases and named entities using spaCy
        2. Cluster by semantic similarity using ChromaDB embeddings
        3. Propose entity types for each cluster
        4. Filter by confidence threshold

        Args:
            documents: Documents to analyze for entity patterns

        Returns:
            List[SchemaDiscovery]: Discovered entity patterns meeting confidence threshold
        """
        if not documents:
            logger.warning("No documents provided for entity pattern discovery")
            return []

        discoveries: List[SchemaDiscovery] = []

        # Step 1: Extract noun phrases and named entities
        noun_phrases = self._extract_noun_phrases(documents)

        if not noun_phrases:
            logger.info("No noun phrases extracted from documents")
            return []

        # Step 2: Cluster by semantic similarity
        clusters = self._cluster_by_similarity(noun_phrases)

        if not clusters:
            logger.info("No clusters formed from noun phrases")
            return []

        # Step 3: Propose entity types for each cluster
        for cluster in clusters:
            discovery = self._propose_entity_type(cluster)

            # Step 4: Filter by confidence threshold
            if discovery.confidence >= self.discovery_threshold:
                discoveries.append(discovery)
                logger.debug(
                    f"Discovered entity type: {discovery.name} "
                    f"(confidence: {discovery.confidence:.3f}, examples: {len(discovery.examples)})"
                )

        # Sort by confidence descending
        discoveries.sort(key=lambda d: d.confidence, reverse=True)

        logger.info(
            f"Discovered {len(discoveries)} entity patterns from {len(documents)} documents "
            f"({len(clusters)} clusters, threshold: {self.discovery_threshold})"
        )

        return discoveries

    def discover_relationship_patterns(
        self,
        documents: List["Document"],
        entity_types: Dict[str, EntityType],
    ) -> List[SchemaDiscovery]:
        """
        Discover relationship patterns between known entities.

        Per AutoSchemaKG approach:
        1. Extract sentences with multiple entities
        2. Identify connecting verbs using dependency parsing
        3. Cluster similar relationship patterns
        4. Propose relationship types with subject/object constraints

        Args:
            documents: Documents to analyze
            entity_types: Known entity types to find relationships between

        Returns:
            List[SchemaDiscovery]: Discovered relationship patterns
        """
        if not documents:
            logger.warning("No documents provided for relationship discovery")
            return []

        discoveries: List[SchemaDiscovery] = []
        relationship_patterns: List[Dict[str, Any]] = []

        # Build lookup for entity type names
        known_type_names = set(entity_types.keys())

        for doc in documents:
            try:
                spacy_doc = self.nlp(doc.content)

                for sent in spacy_doc.sents:
                    # Find entities in sentence
                    entities_in_sent: List[Dict[str, Any]] = []

                    for ent in sent.ents:
                        # Match to known entity types via NER label
                        matched_type = self._match_entity_to_type(ent, known_type_names)
                        if matched_type:
                            entities_in_sent.append(
                                {
                                    "text": ent.text,
                                    "type": matched_type,
                                    "start": ent.start,
                                    "end": ent.end,
                                    "label": ent.label_,
                                }
                            )

                    # Find relationships between entity pairs
                    if len(entities_in_sent) >= 2:
                        for i, ent1 in enumerate(entities_in_sent):
                            for ent2 in entities_in_sent[i + 1 :]:
                                # Extract connecting phrase between entities
                                connecting = self._extract_connecting_phrase(
                                    sent, ent1, ent2
                                )
                                if connecting:
                                    relationship_patterns.append(
                                        {
                                            "subject_type": ent1["type"],
                                            "object_type": ent2["type"],
                                            "predicate": connecting["verb"],
                                            "context": sent.text[:500],
                                            "doc_id": doc.doc_id,
                                            "subject_text": ent1["text"],
                                            "object_text": ent2["text"],
                                        }
                                    )

            except Exception as e:
                logger.warning(f"Error processing document {doc.doc_id}: {e}")
                continue

        if not relationship_patterns:
            logger.info("No relationship patterns found in documents")
            return []

        # Cluster similar relationship patterns by predicate
        pattern_clusters = self._cluster_relationship_patterns(relationship_patterns)

        # Propose relationship types for each cluster
        for cluster in pattern_clusters:
            if len(cluster) >= self.min_cluster_size:
                discovery = self._propose_relationship_type(cluster)
                if discovery.confidence >= self.discovery_threshold:
                    discoveries.append(discovery)

        # Sort by confidence descending
        discoveries.sort(key=lambda d: d.confidence, reverse=True)

        logger.info(
            f"Discovered {len(discoveries)} relationship patterns from {len(documents)} documents"
        )

        return discoveries

    def _match_entity_to_type(
        self, ent, known_types: Set[str]
    ) -> Optional[str]:
        """
        Match a spaCy entity to a known entity type.

        Args:
            ent: spaCy entity
            known_types: Set of known entity type names

        Returns:
            Matched type name or None
        """
        # First try direct NER label mapping
        type_hint = NER_TYPE_HINTS.get(ent.label_)
        if type_hint and type_hint in known_types:
            return type_hint

        # Fallback mappings
        label_to_type = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "EVENT": "Event",
            "DATE": "TemporalMarker",
            "TIME": "TemporalMarker",
        }

        mapped = label_to_type.get(ent.label_)
        if mapped and mapped in known_types:
            return mapped

        # If no match, return generic Concept if available
        if "Concept" in known_types:
            return "Concept"

        return None

    def _extract_connecting_phrase(
        self, sent, ent1: Dict[str, Any], ent2: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """
        Extract verb phrase connecting two entities using dependency parsing.

        Args:
            sent: spaCy sentence
            ent1: First entity info
            ent2: Second entity info

        Returns:
            Dict with verb info or None if no connection found
        """
        # Determine token range between entities
        start_idx = min(ent1["end"], ent2["end"])
        end_idx = max(ent1["start"], ent2["start"])

        # Look for main verb in connecting region
        for token in sent:
            if start_idx <= token.i < end_idx:
                if token.pos_ == "VERB":
                    return {
                        "verb": token.lemma_.lower(),
                        "full_text": token.text,
                        "dep": token.dep_,
                    }

        # Fallback: look for ROOT verb in sentence
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return {
                    "verb": token.lemma_.lower(),
                    "full_text": token.text,
                    "dep": token.dep_,
                }

        # Last resort: any verb in sentence
        for token in sent:
            if token.pos_ == "VERB":
                return {
                    "verb": token.lemma_.lower(),
                    "full_text": token.text,
                    "dep": token.dep_,
                }

        return None

    def _cluster_relationship_patterns(
        self, patterns: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster relationship patterns by predicate similarity.

        Args:
            patterns: Relationship patterns to cluster

        Returns:
            Clustered patterns
        """
        # Group by predicate (verb lemma)
        predicate_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for pattern in patterns:
            predicate = pattern["predicate"]
            predicate_groups[predicate].append(pattern)

        # Filter groups meeting minimum size
        clusters = [
            group
            for group in predicate_groups.values()
            if len(group) >= self.min_cluster_size
        ]

        return clusters

    def _propose_relationship_type(
        self, cluster: List[Dict[str, Any]]
    ) -> SchemaDiscovery:
        """
        Propose relationship type from clustered patterns.

        Args:
            cluster: Clustered relationship patterns

        Returns:
            SchemaDiscovery for relationship type
        """
        if not cluster:
            return SchemaDiscovery(
                element_type="relationship",
                name="unknown_relation",
                description="Unknown relationship",
                examples=[],
                confidence=0.0,
                source_documents=[],
            )

        # Get dominant predicate
        predicates = [p["predicate"] for p in cluster]
        predicate_counts = Counter(predicates)
        dominant_predicate = predicate_counts.most_common(1)[0][0]

        # Collect subject and object types
        subject_types = list(set(p["subject_type"] for p in cluster))
        object_types = list(set(p["object_type"] for p in cluster))

        # Generate examples
        examples = []
        for p in cluster[:5]:
            example = f"{p['subject_text']} {dominant_predicate} {p['object_text']}"
            examples.append(example)

        # Collect document IDs
        doc_ids = list(set(p["doc_id"] for p in cluster))[:10]

        # Calculate confidence
        cluster_size_factor = min(len(cluster) / 10, 1.0)
        type_consistency = len(set((p["subject_type"], p["object_type"]) for p in cluster))
        type_consistency_factor = 1.0 / max(type_consistency, 1)
        doc_diversity = min(len(doc_ids) / 3, 1.0)

        confidence = 0.4 * cluster_size_factor + 0.3 * type_consistency_factor + 0.3 * doc_diversity

        # Create relationship name (verb with subject/object hint)
        rel_name = dominant_predicate.replace(" ", "_")

        return SchemaDiscovery(
            element_type="relationship",
            name=rel_name,
            description=f"Relationship '{dominant_predicate}' between {subject_types} and {object_types}",
            examples=examples,
            confidence=round(confidence, 3),
            source_documents=doc_ids,
        )

    def _extract_noun_phrases(self, documents: List["Document"]) -> List[Dict[str, Any]]:
        """
        Extract noun phrases and named entities from documents using spaCy.

        Per AutoSchemaKG approach:
        - Extract noun chunks (compound noun phrases)
        - Extract named entities with NER labels
        - Include context for LLM validation

        Args:
            documents: Documents to extract from

        Returns:
            List[Dict]: Extracted phrases with metadata:
                - text: The noun phrase (lowercased)
                - root: Head noun of the phrase
                - label: NER label or "NOUN" for general nouns
                - doc_id: Source document ID
                - context: Surrounding sentence for validation
        """
        noun_phrases: List[Dict[str, Any]] = []

        for doc in documents:
            try:
                spacy_doc = self.nlp(doc.content)

                # Extract noun chunks (compound noun phrases)
                for chunk in spacy_doc.noun_chunks:
                    # Skip very short or very long chunks
                    if len(chunk.text.strip()) < 2 or len(chunk.text) > 100:
                        continue

                    # Get sentence context
                    try:
                        context = chunk.sent.text if hasattr(chunk, "sent") else ""
                    except Exception:
                        context = ""

                    noun_phrases.append(
                        {
                            "text": chunk.text.lower().strip(),
                            "root": chunk.root.text.lower(),
                            "label": chunk.root.ent_type_ or "NOUN",
                            "doc_id": doc.doc_id,
                            "context": context[:500],  # Limit context length
                        }
                    )

                # Extract named entities (more specific)
                for ent in spacy_doc.ents:
                    # Skip entities already captured in noun chunks
                    if len(ent.text.strip()) < 2:
                        continue

                    # Get sentence context
                    try:
                        context = ent.sent.text if hasattr(ent, "sent") else ""
                    except Exception:
                        context = ""

                    noun_phrases.append(
                        {
                            "text": ent.text.lower().strip(),
                            "root": ent.text.lower().strip(),
                            "label": ent.label_,  # PERSON, ORG, GPE, etc.
                            "doc_id": doc.doc_id,
                            "context": context[:500],
                        }
                    )

            except Exception as e:
                logger.warning(f"Error processing document {doc.doc_id}: {e}")
                continue

        logger.info(f"Extracted {len(noun_phrases)} noun phrases from {len(documents)} documents")
        return noun_phrases

    def _cluster_by_similarity(
        self, phrases: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster phrases by semantic similarity using ChromaDB embeddings.

        Per AutoSchemaKG approach:
        - Embed all phrases for semantic comparison
        - Find connected components above similarity threshold
        - Return clusters for type proposal

        Args:
            phrases: Phrases with metadata to cluster

        Returns:
            List[List[Dict]]: Clustered phrases grouped by similarity
        """
        if not phrases:
            return []

        # Deduplicate phrases by text
        unique_phrases: Dict[str, Dict[str, Any]] = {}
        for phrase in phrases:
            text = phrase["text"]
            if text not in unique_phrases:
                unique_phrases[text] = phrase
            else:
                # Merge doc_ids for duplicate texts
                existing = unique_phrases[text]
                if existing["doc_id"] != phrase["doc_id"]:
                    # Track multi-document presence (increases confidence)
                    if "doc_ids" not in existing:
                        existing["doc_ids"] = {existing["doc_id"]}
                    existing["doc_ids"].add(phrase["doc_id"])

        phrase_list = list(unique_phrases.values())

        if len(phrase_list) < self.min_cluster_size:
            # Not enough unique phrases to cluster
            return []

        try:
            import chromadb
            from chromadb.config import Settings

            # Initialize ephemeral ChromaDB client for clustering
            client = chromadb.Client(Settings(anonymized_telemetry=False))

            # Create temporary collection for clustering
            collection_name = f"schema_discovery_{hashlib.md5(str(len(phrases)).encode()).hexdigest()[:8]}"
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass

            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # Add phrases to collection
            ids = []
            documents = []
            metadatas = []

            for i, phrase in enumerate(phrase_list):
                phrase_id = f"phrase_{i}"
                ids.append(phrase_id)
                documents.append(phrase["text"])
                metadatas.append(
                    {
                        "label": phrase.get("label", "NOUN"),
                        "doc_id": phrase.get("doc_id", ""),
                        "root": phrase.get("root", ""),
                        "idx": str(i),
                    }
                )

            collection.add(ids=ids, documents=documents, metadatas=metadatas)

            # Cluster using query-based similarity (connected components approach)
            clusters: List[List[Dict[str, Any]]] = []
            clustered_indices: Set[int] = set()

            for i, phrase in enumerate(phrase_list):
                if i in clustered_indices:
                    continue

                # Query for similar phrases
                results = collection.query(
                    query_texts=[phrase["text"]],
                    n_results=min(50, len(phrase_list)),
                    include=["distances", "metadatas"],
                )

                # Build cluster from similar results
                cluster: List[Dict[str, Any]] = []
                for j, (doc_id, distance) in enumerate(
                    zip(results["ids"][0], results["distances"][0])
                ):
                    # ChromaDB returns L2 distance for cosine space
                    # Convert to similarity: similarity = 1 - distance
                    # For cosine distance normalized vectors: distance in [0, 2]
                    similarity = 1 - (distance / 2)

                    if similarity >= self.similarity_threshold:
                        idx = int(results["metadatas"][0][j]["idx"])
                        if idx not in clustered_indices:
                            cluster.append(phrase_list[idx])
                            clustered_indices.add(idx)

                if len(cluster) >= self.min_cluster_size:
                    clusters.append(cluster)

            # Clean up temporary collection
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass

            logger.info(
                f"Clustered {len(phrase_list)} unique phrases into {len(clusters)} clusters"
            )
            return clusters

        except ImportError:
            logger.error("ChromaDB not installed. Run: pip install chromadb")
            # Fallback: group by NER label only
            return self._fallback_cluster_by_label(phrase_list)
        except Exception as e:
            logger.warning(f"ChromaDB clustering failed: {e}. Using fallback clustering.")
            return self._fallback_cluster_by_label(phrase_list)

    def _fallback_cluster_by_label(
        self, phrases: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Fallback clustering by NER label when ChromaDB is unavailable.

        Args:
            phrases: Phrases to cluster

        Returns:
            List[List[Dict]]: Phrases grouped by NER label
        """
        label_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for phrase in phrases:
            label = phrase.get("label", "NOUN")
            label_groups[label].append(phrase)

        # Return groups meeting minimum size
        return [
            group
            for group in label_groups.values()
            if len(group) >= self.min_cluster_size
        ]

    def _propose_entity_type(
        self, cluster: List[Dict[str, Any]]
    ) -> SchemaDiscovery:
        """
        Propose entity type from clustered examples using NER hints and LLM validation.

        Per AutoSchemaKG approach:
        1. Analyze cluster for common NER labels
        2. Use majority label as type hint
        3. LLM validates and refines type definition (if available)
        4. Calculate confidence from cluster coherence

        Args:
            cluster: Clustered phrase dictionaries with metadata

        Returns:
            SchemaDiscovery: Proposed entity type with confidence
        """
        if not cluster:
            return SchemaDiscovery(
                element_type="entity",
                name="UnknownType",
                description="No examples available",
                examples=[],
                confidence=0.0,
                source_documents=[],
            )

        # Count NER labels in cluster (majority voting)
        label_counts = Counter(p.get("label", "NOUN") for p in cluster)
        dominant_label, dominant_count = label_counts.most_common(1)[0]

        # Extract unique examples (deduplicated)
        examples = list(set(p["text"] for p in cluster))[:10]

        # Collect unique document IDs
        doc_ids: Set[str] = set()
        for p in cluster:
            doc_ids.add(p.get("doc_id", ""))
            if "doc_ids" in p:
                doc_ids.update(p["doc_ids"])
        source_documents = list(doc_ids)[:20]

        # Map NER label to entity type hint
        suggested_type = NER_TYPE_HINTS.get(dominant_label)

        # Calculate base confidence from cluster properties
        # Higher confidence when:
        # - More examples in cluster
        # - More documents represented
        # - Consistent NER labels
        label_consistency = dominant_count / len(cluster)
        cluster_size_factor = min(len(cluster) / 20, 1.0)
        doc_diversity_factor = min(len(source_documents) / 5, 1.0)

        base_confidence = (
            0.4 * label_consistency
            + 0.3 * cluster_size_factor
            + 0.3 * doc_diversity_factor
        )

        # If LLM available and type is ambiguous (NOUN or unknown), use LLM for refinement
        if self.llm and (not suggested_type or dominant_label == "NOUN"):
            llm_result = self._llm_propose_entity_type(examples, cluster)
            if llm_result:
                suggested_type = llm_result.get("type_name", "Concept")
                description = llm_result.get(
                    "description", f"Entity type discovered from {len(cluster)} examples"
                )
                # Boost confidence if LLM validated
                confidence = min(base_confidence + 0.15, 0.95)
            else:
                # Default to Concept for unrecognized noun phrases
                suggested_type = "Concept"
                description = f"Abstract concept or topic (from {len(cluster)} examples)"
                confidence = base_confidence
        else:
            # Use NER-based type
            if suggested_type:
                description = f"{suggested_type} entity type discovered from NER analysis"
                confidence = min(base_confidence + 0.1, 0.95)  # NER types get slight boost
            else:
                suggested_type = "Concept"
                description = f"Abstract concept or topic (from {len(cluster)} examples)"
                confidence = base_confidence

        return SchemaDiscovery(
            element_type="entity",
            name=suggested_type,
            description=description,
            examples=examples[:5],
            confidence=round(confidence, 3),
            source_documents=source_documents[:10],
        )

    def _llm_propose_entity_type(
        self, examples: List[str], cluster: List[Dict[str, Any]]
    ) -> Optional[Dict[str, str]]:
        """
        Use LLM to propose entity type for ambiguous clusters.

        Args:
            examples: Example phrases from cluster
            cluster: Full cluster with metadata

        Returns:
            Dict with type_name and description, or None if LLM unavailable/failed
        """
        if not self.llm:
            return None

        # Collect context samples
        contexts = [p.get("context", "") for p in cluster[:3] if p.get("context")]

        prompt = f"""Analyze these noun phrases and propose an entity type name.

Examples: {', '.join(examples[:10])}

Context samples:
{chr(10).join(f'- "{c}"' for c in contexts[:3] if c)}

Based on these examples, what entity type do they represent?
Consider common types like: Person, Organization, Project, Event, Concept, Document, Location, Product, Tool, Technology

Respond in JSON format:
{{"type_name": "EntityTypeName", "description": "Brief description of what this type represents"}}
"""
        try:
            # Support different LLM interface patterns
            if hasattr(self.llm, "extract"):
                response = self.llm.extract(prompt)
            elif hasattr(self.llm, "generate"):
                response = self.llm.generate(prompt)
            elif hasattr(self.llm, "complete"):
                response = self.llm.complete(prompt)
            elif callable(self.llm):
                response = self.llm(prompt)
            else:
                logger.warning("LLM interface not recognized")
                return None

            # Parse JSON response
            import json

            # Handle string or object response
            if isinstance(response, str):
                # Try to extract JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    data = json.loads(json_str)
                else:
                    return None
            elif isinstance(response, dict):
                data = response
            else:
                return None

            return {
                "type_name": data.get("type_name", "Concept"),
                "description": data.get("description", ""),
            }

        except Exception as e:
            logger.warning(f"LLM entity type proposal failed: {e}")
            return None
