"""
Unit Tests for Schema Discovery Engine

Tests for NLP extraction, semantic clustering, type proposal,
and pattern discovery mechanisms.

Research Foundation:
- AutoSchemaKG (2505.23628v1): Autonomous schema induction
- Target: >90% semantic alignment with manual schemas
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from futurnal.extraction.schema.discovery import SchemaDiscoveryEngine, NER_TYPE_HINTS
from futurnal.extraction.schema.evolution import Document
from futurnal.extraction.schema.models import SchemaDiscovery, EntityType


class TestSchemaDiscoveryEngineInit:
    """Test discovery engine initialization."""

    def test_default_initialization(self):
        """Test engine initializes with defaults."""
        engine = SchemaDiscoveryEngine()

        assert engine.discovery_threshold == 0.75
        assert engine.similarity_threshold == 0.85
        assert engine.min_cluster_size == 3
        assert engine.llm is None
        assert engine._nlp is None

    def test_custom_initialization(self):
        """Test engine accepts custom parameters."""
        mock_llm = Mock()
        engine = SchemaDiscoveryEngine(
            llm=mock_llm,
            discovery_threshold=0.8,
            similarity_threshold=0.9,
            min_cluster_size=5,
        )

        assert engine.llm == mock_llm
        assert engine.discovery_threshold == 0.8
        assert engine.similarity_threshold == 0.9
        assert engine.min_cluster_size == 5


class TestNLPExtraction:
    """Test NLP-based noun phrase extraction."""

    @pytest.fixture
    def engine(self):
        """Create engine with mocked spaCy."""
        return SchemaDiscoveryEngine()

    def test_extract_noun_phrases_empty_documents(self, engine):
        """Test extraction handles empty document list."""
        phrases = engine._extract_noun_phrases([])
        assert phrases == []

    def test_extract_noun_phrases_basic(self):
        """Test basic noun phrase extraction with mocked spaCy."""
        engine = SchemaDiscoveryEngine()

        # Create mock spaCy doc
        mock_chunk = MagicMock()
        mock_chunk.text = "Alice Smith"
        mock_chunk.root.text = "Smith"
        mock_chunk.root.ent_type_ = "PERSON"
        mock_sent = MagicMock()
        mock_sent.text = "Alice Smith works at Acme."
        mock_chunk.sent = mock_sent

        mock_ent = MagicMock()
        mock_ent.text = "Acme Corp"
        mock_ent.label_ = "ORG"
        mock_ent.sent = mock_sent

        mock_doc = MagicMock()
        mock_doc.noun_chunks = [mock_chunk]
        mock_doc.ents = [mock_ent]

        # Create mock nlp that returns the mock doc
        mock_nlp = MagicMock(return_value=mock_doc)
        engine._nlp = mock_nlp

        documents = [Document("Alice Smith works at Acme Corp.", "doc1")]
        phrases = engine._extract_noun_phrases(documents)

        assert len(phrases) >= 1
        assert mock_nlp.called

    def test_ner_type_hints_coverage(self):
        """Test NER type hints cover common labels."""
        expected_labels = ["PERSON", "ORG", "GPE", "DATE", "TIME", "EVENT"]

        for label in expected_labels:
            assert label in NER_TYPE_HINTS, f"Missing NER hint for {label}"


class TestSemanticClustering:
    """Test semantic clustering functionality."""

    def test_cluster_empty_phrases(self):
        """Test clustering handles empty input."""
        engine = SchemaDiscoveryEngine()
        clusters = engine._cluster_by_similarity([])
        assert clusters == []

    def test_cluster_insufficient_phrases(self):
        """Test clustering returns empty for insufficient phrases."""
        engine = SchemaDiscoveryEngine(min_cluster_size=3)
        phrases = [
            {"text": "alice", "label": "PERSON", "doc_id": "d1", "root": "alice"},
        ]
        clusters = engine._cluster_by_similarity(phrases)
        assert clusters == []

    def test_fallback_cluster_by_label(self):
        """Test fallback clustering groups by NER label."""
        engine = SchemaDiscoveryEngine(min_cluster_size=2)

        phrases = [
            {"text": "alice", "label": "PERSON", "doc_id": "d1", "root": "alice"},
            {"text": "bob", "label": "PERSON", "doc_id": "d2", "root": "bob"},
            {"text": "acme", "label": "ORG", "doc_id": "d3", "root": "acme"},
            {"text": "techcorp", "label": "ORG", "doc_id": "d4", "root": "techcorp"},
        ]

        clusters = engine._fallback_cluster_by_label(phrases)

        # Should have 2 clusters (PERSON and ORG)
        assert len(clusters) == 2

        # Each cluster should have 2 items
        for cluster in clusters:
            assert len(cluster) >= 2


class TestTypeProposal:
    """Test entity type proposal from clusters."""

    def test_propose_entity_type_empty_cluster(self):
        """Test proposal handles empty cluster."""
        engine = SchemaDiscoveryEngine()
        discovery = engine._propose_entity_type([])

        assert discovery.element_type == "entity"
        assert discovery.name == "UnknownType"
        assert discovery.confidence == 0.0

    def test_propose_entity_type_person_cluster(self):
        """Test proposal for PERSON-labeled cluster."""
        engine = SchemaDiscoveryEngine()

        cluster = [
            {"text": "john smith", "label": "PERSON", "doc_id": "d1", "root": "john", "context": "John Smith works..."},
            {"text": "alice jones", "label": "PERSON", "doc_id": "d2", "root": "alice", "context": "Alice Jones..."},
            {"text": "bob wilson", "label": "PERSON", "doc_id": "d3", "root": "bob", "context": "Bob Wilson..."},
        ]

        discovery = engine._propose_entity_type(cluster)

        assert discovery.element_type == "entity"
        assert discovery.name == "Person"
        assert discovery.confidence > 0.5
        assert len(discovery.examples) > 0

    def test_propose_entity_type_org_cluster(self):
        """Test proposal for ORG-labeled cluster."""
        engine = SchemaDiscoveryEngine()

        cluster = [
            {"text": "acme corp", "label": "ORG", "doc_id": "d1", "root": "acme", "context": ""},
            {"text": "techstartup", "label": "ORG", "doc_id": "d2", "root": "tech", "context": ""},
            {"text": "bigco inc", "label": "ORG", "doc_id": "d3", "root": "bigco", "context": ""},
        ]

        discovery = engine._propose_entity_type(cluster)

        assert discovery.name == "Organization"
        assert discovery.confidence > 0.5

    def test_propose_entity_type_noun_cluster(self):
        """Test proposal for generic NOUN cluster defaults to Concept."""
        engine = SchemaDiscoveryEngine()

        cluster = [
            {"text": "machine learning", "label": "NOUN", "doc_id": "d1", "root": "learning", "context": ""},
            {"text": "deep learning", "label": "NOUN", "doc_id": "d2", "root": "learning", "context": ""},
            {"text": "ai research", "label": "NOUN", "doc_id": "d3", "root": "research", "context": ""},
        ]

        discovery = engine._propose_entity_type(cluster)

        # Without LLM, NOUN clusters default to Concept
        assert discovery.name == "Concept"

    def test_confidence_increases_with_cluster_size(self):
        """Test confidence increases with larger clusters."""
        engine = SchemaDiscoveryEngine()

        small_cluster = [
            {"text": f"person{i}", "label": "PERSON", "doc_id": f"d{i}", "root": "person", "context": ""}
            for i in range(3)
        ]

        large_cluster = [
            {"text": f"person{i}", "label": "PERSON", "doc_id": f"d{i}", "root": "person", "context": ""}
            for i in range(20)
        ]

        small_discovery = engine._propose_entity_type(small_cluster)
        large_discovery = engine._propose_entity_type(large_cluster)

        assert large_discovery.confidence >= small_discovery.confidence


class TestEntityPatternDiscovery:
    """Test complete entity pattern discovery pipeline."""

    def test_discover_entity_patterns_empty(self):
        """Test discovery handles empty document list."""
        engine = SchemaDiscoveryEngine()
        discoveries = engine.discover_entity_patterns([])
        assert discoveries == []

    @patch("futurnal.extraction.schema.discovery.SchemaDiscoveryEngine._extract_noun_phrases")
    def test_discover_entity_patterns_no_phrases(self, mock_extract):
        """Test discovery handles no extracted phrases."""
        mock_extract.return_value = []

        engine = SchemaDiscoveryEngine()
        documents = [Document("Test content", "doc1")]
        discoveries = engine.discover_entity_patterns(documents)

        assert discoveries == []

    def test_discover_entity_patterns_pipeline(self):
        """Test full discovery pipeline with mocked methods."""
        engine = SchemaDiscoveryEngine(discovery_threshold=0.5)

        # Mock internal methods
        engine._extract_noun_phrases = MagicMock(return_value=[
            {"text": "alice", "label": "PERSON", "doc_id": "d1", "root": "alice", "context": ""},
            {"text": "bob", "label": "PERSON", "doc_id": "d2", "root": "bob", "context": ""},
            {"text": "carol", "label": "PERSON", "doc_id": "d3", "root": "carol", "context": ""},
        ])

        engine._cluster_by_similarity = MagicMock(return_value=[
            [
                {"text": "alice", "label": "PERSON", "doc_id": "d1", "root": "alice", "context": ""},
                {"text": "bob", "label": "PERSON", "doc_id": "d2", "root": "bob", "context": ""},
                {"text": "carol", "label": "PERSON", "doc_id": "d3", "root": "carol", "context": ""},
            ]
        ])

        documents = [
            Document("Alice works here.", "d1"),
            Document("Bob manages.", "d2"),
            Document("Carol leads.", "d3"),
        ]

        discoveries = engine.discover_entity_patterns(documents)

        assert len(discoveries) >= 1
        assert all(d.element_type == "entity" for d in discoveries)
        engine._extract_noun_phrases.assert_called_once()
        engine._cluster_by_similarity.assert_called_once()


class TestRelationshipPatternDiscovery:
    """Test relationship pattern discovery."""

    def test_discover_relationship_patterns_empty(self):
        """Test discovery handles empty documents."""
        engine = SchemaDiscoveryEngine()
        discoveries = engine.discover_relationship_patterns([], {})
        assert discoveries == []

    def test_match_entity_to_type(self):
        """Test entity-to-type matching."""
        engine = SchemaDiscoveryEngine()
        known_types = {"Person", "Organization", "Location", "Concept"}

        # Create mock entity
        mock_ent = MagicMock()
        mock_ent.label_ = "PERSON"

        matched = engine._match_entity_to_type(mock_ent, known_types)
        assert matched == "Person"

    def test_match_entity_to_type_unknown(self):
        """Test entity matching with unknown label falls back to Concept."""
        engine = SchemaDiscoveryEngine()
        known_types = {"Person", "Concept"}

        mock_ent = MagicMock()
        mock_ent.label_ = "UNKNOWN"

        matched = engine._match_entity_to_type(mock_ent, known_types)
        assert matched == "Concept"

    def test_cluster_relationship_patterns(self):
        """Test relationship pattern clustering by predicate."""
        engine = SchemaDiscoveryEngine(min_cluster_size=2)

        patterns = [
            {"predicate": "work", "subject_type": "Person", "object_type": "Organization"},
            {"predicate": "work", "subject_type": "Person", "object_type": "Organization"},
            {"predicate": "create", "subject_type": "Person", "object_type": "Document"},
            {"predicate": "create", "subject_type": "Organization", "object_type": "Product"},
        ]

        clusters = engine._cluster_relationship_patterns(patterns)

        # Should have 2 clusters (work and create)
        assert len(clusters) == 2

    def test_propose_relationship_type(self):
        """Test relationship type proposal from cluster."""
        engine = SchemaDiscoveryEngine()

        cluster = [
            {
                "predicate": "work",
                "subject_type": "Person",
                "object_type": "Organization",
                "subject_text": "Alice",
                "object_text": "Acme",
                "doc_id": "d1",
            },
            {
                "predicate": "work",
                "subject_type": "Person",
                "object_type": "Organization",
                "subject_text": "Bob",
                "object_text": "TechCorp",
                "doc_id": "d2",
            },
            {
                "predicate": "work",
                "subject_type": "Person",
                "object_type": "Organization",
                "subject_text": "Carol",
                "object_text": "StartupX",
                "doc_id": "d3",
            },
        ]

        discovery = engine._propose_relationship_type(cluster)

        assert discovery.element_type == "relationship"
        assert discovery.name == "work"
        assert discovery.confidence > 0
        assert len(discovery.examples) > 0


class TestLLMIntegration:
    """Test LLM integration for type proposal."""

    def test_llm_propose_entity_type_no_llm(self):
        """Test LLM proposal returns None without LLM."""
        engine = SchemaDiscoveryEngine(llm=None)

        result = engine._llm_propose_entity_type(["example"], [])
        assert result is None

    def test_llm_propose_entity_type_with_mock_llm(self):
        """Test LLM proposal with mock LLM."""
        mock_llm = MagicMock()
        mock_llm.extract.return_value = '{"type_name": "Project", "description": "Software project"}'

        engine = SchemaDiscoveryEngine(llm=mock_llm)

        result = engine._llm_propose_entity_type(
            ["project alpha", "project beta"],
            [{"text": "project alpha", "context": "Working on project alpha"}],
        )

        assert result is not None
        assert result["type_name"] == "Project"

    def test_llm_handles_invalid_response(self):
        """Test LLM proposal handles invalid JSON gracefully."""
        mock_llm = MagicMock()
        mock_llm.extract.return_value = "invalid json response"

        engine = SchemaDiscoveryEngine(llm=mock_llm)

        result = engine._llm_propose_entity_type(["example"], [])
        assert result is None


class TestQualityGates:
    """Test quality gate compliance."""

    def test_no_hardcoded_entity_types_in_discovery(self):
        """Ensure no hardcoded entity type lists in discovery engine.

        Per .cursor/rules/schema-evolution.mdc:
        NO hardcoded entity types - autonomous schema evolution
        """
        import ast
        import inspect

        # Get source code of SchemaDiscoveryEngine
        source = inspect.getsource(SchemaDiscoveryEngine)
        tree = ast.parse(source)

        # Look for hardcoded lists at class level
        hardcoded_lists = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name_upper = target.id.upper()
                        if "ENTITY_TYPE" in name_upper and isinstance(node.value, ast.List):
                            hardcoded_lists.append(target.id)

        assert len(hardcoded_lists) == 0, f"Found hardcoded type lists: {hardcoded_lists}"

    def test_discovery_threshold_enforced(self):
        """Test discoveries below threshold are filtered."""
        engine = SchemaDiscoveryEngine(discovery_threshold=0.8)

        # Create low-confidence cluster
        cluster = [
            {"text": "x", "label": "NOUN", "doc_id": "d1", "root": "x", "context": ""},
        ]

        discovery = engine._propose_entity_type(cluster)

        # Low confidence discovery should be filtered in pipeline
        assert discovery.confidence < engine.discovery_threshold or discovery.name == "UnknownType"

    def test_min_cluster_size_enforced(self):
        """Test minimum cluster size is respected."""
        engine = SchemaDiscoveryEngine(min_cluster_size=5)

        phrases = [
            {"text": f"item{i}", "label": "NOUN", "doc_id": f"d{i}", "root": "item", "context": ""}
            for i in range(3)  # Less than min_cluster_size
        ]

        clusters = engine._fallback_cluster_by_label(phrases)

        # Should return empty - cluster size is below minimum
        assert len(clusters) == 0


class TestHardcodedTypesRegression:
    """
    Regression tests for NO hardcoded entity types.

    Per .cursor/rules/schema-evolution.mdc:
    "NO hardcoded entity types - autonomous schema evolution"

    These tests verify that production code does not contain
    hardcoded entity/relationship type lists.
    """

    def test_no_hardcoded_entity_types_discovery_module(self):
        """Verify discovery.py has no hardcoded entity type lists."""
        import ast
        import inspect
        from futurnal.extraction.schema import discovery

        source = inspect.getsource(discovery)
        tree = ast.parse(source)

        # Allowed exceptions: NER_TYPE_HINTS is a mapping, not a list of types
        allowed_names = {"NER_TYPE_HINTS"}

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        name_upper = name.upper()
                        # Check for suspicious patterns
                        if any(pattern in name_upper for pattern in ["ENTITY_TYPES", "ENTITY_LIST", "TYPE_LIST", "KNOWN_TYPES"]):
                            if name not in allowed_names:
                                violations.append(name)

        assert len(violations) == 0, f"Hardcoded type lists found in discovery.py: {violations}"

    def test_no_hardcoded_entity_types_evolution_module(self):
        """Verify evolution.py has no hardcoded entity type lists."""
        import ast
        import inspect
        from futurnal.extraction.schema import evolution

        source = inspect.getsource(evolution)
        tree = ast.parse(source)

        # Find class-level or module-level list assignments with entity type patterns
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name_upper = target.id.upper()
                        # Check for hardcoded type lists
                        if any(pattern in name_upper for pattern in ["ENTITY_TYPES", "ENTITY_LIST", "TYPE_LIST", "HARDCODED"]):
                            if isinstance(node.value, (ast.List, ast.Set)):
                                violations.append(target.id)

        assert len(violations) == 0, f"Hardcoded type lists found in evolution.py: {violations}"

    def test_no_hardcoded_entity_types_refinement_module(self):
        """Verify refinement.py has no hardcoded entity type lists."""
        import ast
        import inspect
        from futurnal.extraction.schema import refinement

        source = inspect.getsource(refinement)
        tree = ast.parse(source)

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name_upper = target.id.upper()
                        if any(pattern in name_upper for pattern in ["ENTITY_TYPES", "ENTITY_LIST", "TYPE_LIST", "HARDCODED"]):
                            if isinstance(node.value, (ast.List, ast.Set)):
                                violations.append(target.id)

        assert len(violations) == 0, f"Hardcoded type lists found in refinement.py: {violations}"

    def test_no_hardcoded_types_in_schema_module(self):
        """Comprehensive check: no hardcoded type lists across entire schema module."""
        import ast
        import os

        schema_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        schema_dir = os.path.join(schema_dir, "src", "futurnal", "extraction", "schema")

        # Allowed files/patterns (mappings for NER labels are OK)
        allowed_patterns = {"NER_TYPE_HINTS", "TEMPORAL_KEYWORDS", "CAUSAL_KEYWORDS"}

        violations = {}

        for filename in os.listdir(schema_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                filepath = os.path.join(schema_dir, filename)
                with open(filepath, "r") as f:
                    try:
                        source = f.read()
                        tree = ast.parse(source)

                        for node in ast.walk(tree):
                            if isinstance(node, ast.Assign):
                                for target in node.targets:
                                    if isinstance(target, ast.Name):
                                        name = target.id
                                        name_upper = name.upper()
                                        if any(p in name_upper for p in ["ENTITY_TYPES", "ENTITY_LIST", "TYPE_LIST"]):
                                            if name not in allowed_patterns:
                                                if isinstance(node.value, (ast.List, ast.Set)):
                                                    if filename not in violations:
                                                        violations[filename] = []
                                                    violations[filename].append(name)
                    except SyntaxError:
                        continue

        assert len(violations) == 0, f"Hardcoded type lists found: {violations}"

    def test_seed_schema_is_configurable(self):
        """Verify seed schema is configurable, not hardcoded in discovery/evolution."""
        import ast
        import inspect
        from futurnal.extraction.schema import evolution

        source = inspect.getsource(evolution.SchemaEvolutionEngine.__init__)

        # Should accept seed_schema as parameter
        assert "seed_schema" in source, "Evolution engine should accept seed_schema parameter"


class TestResearchAlignment:
    """Test alignment with research papers."""

    def test_autoschemakg_multi_phase_support(self):
        """Test engine supports AutoSchemaKG multi-phase discovery.

        AutoSchemaKG (2505.23628v1):
        - Phase 1: Entity-Entity relationships
        - Phase 2: Entity-Event relationships
        - Phase 3: Event-Event relationships (causal candidates)
        """
        engine = SchemaDiscoveryEngine()

        # Engine should support both entity and relationship discovery
        assert hasattr(engine, "discover_entity_patterns")
        assert hasattr(engine, "discover_relationship_patterns")

        # Discovery methods should return SchemaDiscovery objects
        assert callable(engine.discover_entity_patterns)
        assert callable(engine.discover_relationship_patterns)

    def test_reflection_compatible_output(self):
        """Test discovery output is compatible with reflection mechanism."""
        engine = SchemaDiscoveryEngine()

        # Create mock discovery
        cluster = [
            {"text": "alice", "label": "PERSON", "doc_id": "d1", "root": "alice", "context": ""},
            {"text": "bob", "label": "PERSON", "doc_id": "d2", "root": "bob", "context": ""},
            {"text": "carol", "label": "PERSON", "doc_id": "d3", "root": "carol", "context": ""},
        ]

        discovery = engine._propose_entity_type(cluster)

        # Discovery should have all required fields for refinement
        assert hasattr(discovery, "element_type")
        assert hasattr(discovery, "name")
        assert hasattr(discovery, "description")
        assert hasattr(discovery, "examples")
        assert hasattr(discovery, "confidence")
        assert hasattr(discovery, "source_documents")
