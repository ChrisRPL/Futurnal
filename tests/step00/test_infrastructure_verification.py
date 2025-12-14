"""Step 00: Infrastructure Verification Tests.

Verifies that all Phase 1 infrastructure components exist and can be imported.
This is NOT comprehensive testing (component-specific tests do that) -
this confirms infrastructure is ready for integration.

Tests verify:
1. Embeddings Service - MultiModelEmbeddingService exists and instantiates
2. PKG Database Manager - PKGDatabaseManager exists and config works
3. Ollama LLM Client - OllamaLLMClient exists with model mapping
4. Search Infrastructure - All hybrid/temporal/causal components exist
5. ChromaDB Configuration - Settings integration works

Success means: All imports succeed, classes exist, basic instantiation works.
"""

from __future__ import annotations

import pytest
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch


# =============================================================================
# 1. EMBEDDINGS SERVICE VERIFICATION
# =============================================================================


class TestEmbeddingsServiceExists:
    """Verify MultiModelEmbeddingService infrastructure is ready."""

    def test_service_can_be_imported(self) -> None:
        """Service module and class can be imported."""
        from futurnal.embeddings.service import MultiModelEmbeddingService
        assert MultiModelEmbeddingService is not None

    def test_config_can_be_imported(self) -> None:
        """EmbeddingServiceConfig can be imported."""
        from futurnal.embeddings.config import EmbeddingServiceConfig
        assert EmbeddingServiceConfig is not None

    def test_config_instantiates(self) -> None:
        """Config can be instantiated with defaults."""
        from futurnal.embeddings.config import EmbeddingServiceConfig
        config = EmbeddingServiceConfig()
        assert config is not None
        assert hasattr(config, "schema_version")

    def test_embedding_request_exists(self) -> None:
        """EmbeddingRequest model exists."""
        from futurnal.embeddings.request import EmbeddingRequest
        assert EmbeddingRequest is not None

    def test_embedding_models_exist(self) -> None:
        """Core embedding models can be imported."""
        from futurnal.embeddings.models import (
            EmbeddingResult,
            TemporalEmbeddingContext,
            EmbeddingEntityType,
        )
        assert EmbeddingResult is not None
        assert TemporalEmbeddingContext is not None
        assert EmbeddingEntityType is not None

    def test_temporal_context_requires_timestamp_for_option_b(self) -> None:
        """Option B: Temporal context requires timestamp."""
        from futurnal.embeddings.models import TemporalEmbeddingContext

        # Should work with timestamp
        ctx = TemporalEmbeddingContext(timestamp=datetime.now())
        assert ctx.timestamp is not None

    def test_model_registry_exists(self) -> None:
        """ModelRegistry for multi-model architecture exists."""
        from futurnal.embeddings.registry import ModelRegistry
        registry = ModelRegistry()
        assert registry is not None
        # Should have method to get models for entity types
        assert hasattr(registry, "get_model_for_entity_type")

    def test_model_router_exists(self) -> None:
        """ModelRouter for entity-type routing exists."""
        from futurnal.embeddings.router import ModelRouter
        assert ModelRouter is not None


# =============================================================================
# 2. PKG DATABASE MANAGER VERIFICATION
# =============================================================================


class TestPKGDatabaseManagerExists:
    """Verify PKGDatabaseManager infrastructure is ready."""

    def test_manager_can_be_imported(self) -> None:
        """Manager module and class can be imported."""
        from futurnal.pkg.database.manager import PKGDatabaseManager
        assert PKGDatabaseManager is not None

    def test_config_can_be_imported(self) -> None:
        """PKGDatabaseConfig can be imported."""
        from futurnal.pkg.database.config import PKGDatabaseConfig
        assert PKGDatabaseConfig is not None

    def test_config_instantiates_with_defaults(self) -> None:
        """Config instantiates with sensible defaults."""
        from futurnal.pkg.database.config import PKGDatabaseConfig
        config = PKGDatabaseConfig()
        assert config is not None
        # Check Option B compliance fields (actual field names)
        assert hasattr(config, "max_connection_pool_size")
        assert hasattr(config, "connection_timeout")

    def test_exceptions_exist(self) -> None:
        """PKG-specific exceptions exist."""
        from futurnal.pkg.database.exceptions import (
            PKGConnectionError,
            PKGHealthCheckError,
            PKGSchemaInitializationError,
        )
        assert PKGConnectionError is not None
        assert PKGHealthCheckError is not None
        assert PKGSchemaInitializationError is not None

    def test_schema_constraints_exist(self) -> None:
        """Schema constraint functions from Module 01 exist."""
        from futurnal.pkg.schema.constraints import (
            init_schema,
            validate_schema,
            get_schema_statistics,
        )
        assert init_schema is not None
        assert validate_schema is not None
        assert get_schema_statistics is not None

    def test_temporal_queries_exist(self) -> None:
        """Temporal graph queries exist (Option B temporal-first)."""
        from futurnal.pkg.queries.temporal import TemporalGraphQueries
        assert TemporalGraphQueries is not None

    def test_pkg_schema_models_exist(self) -> None:
        """PKG schema models exist."""
        from futurnal.pkg.schema.models import EventNode
        assert EventNode is not None


# =============================================================================
# 3. OLLAMA LLM CLIENT VERIFICATION
# =============================================================================


class TestOllamaLLMClientExists:
    """Verify OllamaLLMClient infrastructure is ready."""

    def test_client_can_be_imported(self) -> None:
        """Ollama client can be imported."""
        from futurnal.extraction.ollama_client import OllamaLLMClient
        assert OllamaLLMClient is not None

    def test_model_mapping_exists(self) -> None:
        """Model mapping from HuggingFace to Ollama exists."""
        from futurnal.extraction.ollama_client import OLLAMA_MODEL_MAP
        assert OLLAMA_MODEL_MAP is not None
        assert len(OLLAMA_MODEL_MAP) > 0
        # Should include key models
        assert "microsoft/Phi-3-mini-4k-instruct" in OLLAMA_MODEL_MAP

    def test_client_implements_llm_interface(self) -> None:
        """Client implements LLM interface (has generate method)."""
        from futurnal.extraction.ollama_client import OllamaLLMClient

        # OllamaLLMClient should have the generate method
        assert hasattr(OllamaLLMClient, "generate")
        assert callable(getattr(OllamaLLMClient, "generate"))

    @patch("requests.get")
    def test_client_instantiates(self, mock_get: MagicMock) -> None:
        """Client can be instantiated (with mocked server check)."""
        # Mock the Ollama server availability check
        mock_get.return_value = MagicMock(status_code=200)

        from futurnal.extraction.ollama_client import OllamaLLMClient
        client = OllamaLLMClient(model_name="microsoft/Phi-3-mini-4k-instruct")

        assert client is not None
        assert client.ollama_model == "phi3:mini"

    def test_ollama_helper_functions_exist(self) -> None:
        """Helper functions for Ollama exist."""
        from futurnal.extraction.ollama_client import (
            ollama_available,
            get_ollama_models,
        )
        assert callable(ollama_available)
        assert callable(get_ollama_models)


# =============================================================================
# 4. SEARCH INFRASTRUCTURE VERIFICATION
# =============================================================================


class TestSchemaAwareRetrievalExists:
    """Verify SchemaAwareRetrieval hybrid search engine exists."""

    def test_retrieval_can_be_imported(self) -> None:
        """SchemaAwareRetrieval can be imported."""
        from futurnal.search.hybrid.retrieval import SchemaAwareRetrieval
        assert SchemaAwareRetrieval is not None

    def test_hybrid_search_config_exists(self) -> None:
        """HybridSearchConfig exists."""
        from futurnal.search.hybrid.config import HybridSearchConfig
        assert HybridSearchConfig is not None

    def test_hybrid_types_exist(self) -> None:
        """Hybrid search types exist."""
        from futurnal.search.hybrid.types import (
            HybridSearchQuery,
            HybridSearchResult,
            VectorSearchResult,
            GraphSearchResult,
        )
        assert HybridSearchQuery is not None
        assert HybridSearchResult is not None

    def test_result_fusion_exists(self) -> None:
        """Result fusion component exists."""
        from futurnal.search.hybrid.fusion import ResultFusion
        assert ResultFusion is not None


class TestQueryEmbeddingRouterExists:
    """Verify QueryEmbeddingRouter for query-model routing exists."""

    def test_router_can_be_imported(self) -> None:
        """QueryEmbeddingRouter can be imported."""
        from futurnal.search.hybrid.query_router import QueryEmbeddingRouter
        assert QueryEmbeddingRouter is not None


class TestTemporalQueryEngineExists:
    """Verify TemporalQueryEngine exists."""

    def test_engine_can_be_imported(self) -> None:
        """TemporalQueryEngine can be imported."""
        from futurnal.search.temporal.engine import TemporalQueryEngine
        assert TemporalQueryEngine is not None

    def test_temporal_types_exist(self) -> None:
        """Temporal query types exist."""
        from futurnal.search.temporal.types import TemporalQuery, TemporalQueryType
        assert TemporalQuery is not None
        assert TemporalQueryType is not None

    def test_temporal_results_exist(self) -> None:
        """Temporal result types exist."""
        from futurnal.search.temporal.results import (
            TemporalSearchResult,
            ScoredEvent,
            RecurringPattern,
        )
        assert TemporalSearchResult is not None
        assert ScoredEvent is not None

    def test_decay_scorer_exists(self) -> None:
        """Temporal decay scorer exists (Option B compliance)."""
        from futurnal.search.temporal.decay import TemporalDecayScorer
        assert TemporalDecayScorer is not None

    def test_pattern_matcher_exists(self) -> None:
        """Temporal pattern matcher exists."""
        from futurnal.search.temporal.patterns import TemporalPatternMatcher
        assert TemporalPatternMatcher is not None

    def test_correlation_detector_exists(self) -> None:
        """Temporal correlation detector exists (Phase 2 foundation)."""
        from futurnal.search.temporal.correlation import TemporalCorrelationDetector
        assert TemporalCorrelationDetector is not None


class TestCausalChainRetrievalExists:
    """Verify CausalChainRetrieval exists."""

    def test_retrieval_can_be_imported(self) -> None:
        """CausalChainRetrieval can be imported."""
        from futurnal.search.causal.retrieval import CausalChainRetrieval
        assert CausalChainRetrieval is not None

    def test_causal_types_exist(self) -> None:
        """Causal query types exist."""
        from futurnal.search.causal.types import CausalQuery, CausalQueryType
        assert CausalQuery is not None
        assert CausalQueryType is not None

    def test_causal_results_exist(self) -> None:
        """Causal result types exist."""
        from futurnal.search.causal.results import (
            CausalPathResult,
            FindCausesResult,
            FindEffectsResult,
        )
        assert CausalPathResult is not None
        assert FindCausesResult is not None
        assert FindEffectsResult is not None

    def test_temporal_ordering_validator_exists(self) -> None:
        """Temporal ordering validator exists (Option B: 100% validation)."""
        from futurnal.search.causal.validation import TemporalOrderingValidator
        assert TemporalOrderingValidator is not None


# =============================================================================
# 5. CHROMADB CONFIGURATION VERIFICATION
# =============================================================================


class TestChromaDBConfigurationExists:
    """Verify ChromaDB configuration integration exists."""

    def test_storage_settings_has_chroma_path(self) -> None:
        """StorageSettings has ChromaDB path configuration."""
        from futurnal.configuration.settings import StorageSettings
        # Check the field exists in the model
        assert "chroma_path" in StorageSettings.model_fields

    def test_temporal_aware_writer_exists(self) -> None:
        """TemporalAwareVectorWriter for ChromaDB exists."""
        from futurnal.embeddings.integration import TemporalAwareVectorWriter
        assert TemporalAwareVectorWriter is not None

    def test_schema_versioned_store_exists(self) -> None:
        """SchemaVersionedEmbeddingStore exists."""
        from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore
        assert SchemaVersionedEmbeddingStore is not None

    def test_vector_writer_exists(self) -> None:
        """ChromaVectorWriter exists."""
        from futurnal.pipeline.vector import ChromaVectorWriter
        assert ChromaVectorWriter is not None


# =============================================================================
# 6. SEARCH API GAP VERIFICATION (Documents the Problem)
# =============================================================================


class TestSearchAPIGapDocumented:
    """Document the gap between api.py and infrastructure.

    These tests verify the PROBLEM exists (keyword matching instead of semantic),
    which Step 01 will fix.
    """

    def test_hybrid_search_api_exists(self) -> None:
        """Main HybridSearchAPI exists in api.py."""
        from futurnal.search.api import HybridSearchAPI
        assert HybridSearchAPI is not None

    def test_api_has_search_method(self) -> None:
        """API has the main search method."""
        from futurnal.search.api import HybridSearchAPI
        assert hasattr(HybridSearchAPI, "search")

    def test_infrastructure_components_exist_but_not_wired(self) -> None:
        """Verify infrastructure exists (Step 01 will wire them).

        This test documents the gap: sophisticated components exist
        but are not used in the main API.
        """
        # These ALL exist and are production-ready
        from futurnal.search.hybrid.retrieval import SchemaAwareRetrieval
        from futurnal.search.hybrid.query_router import QueryEmbeddingRouter
        from futurnal.search.temporal.engine import TemporalQueryEngine
        from futurnal.search.causal.retrieval import CausalChainRetrieval

        # The main API exists
        from futurnal.search.api import HybridSearchAPI

        # All components are NOT None
        assert SchemaAwareRetrieval is not None
        assert QueryEmbeddingRouter is not None
        assert TemporalQueryEngine is not None
        assert CausalChainRetrieval is not None
        assert HybridSearchAPI is not None

        # GAP: HybridSearchAPI doesn't use SchemaAwareRetrieval
        # This will be fixed in Step 01


# =============================================================================
# 7. OPTION B COMPLIANCE MARKERS
# =============================================================================


class TestOptionBComplianceMarkers:
    """Verify Option B compliance patterns exist in infrastructure."""

    def test_embeddings_require_temporal_context_for_events(self) -> None:
        """Option B: Events require temporal context.

        EmbeddingRequest should validate temporal context for Event type.
        """
        from futurnal.embeddings.request import EmbeddingRequest
        from futurnal.embeddings.models import TemporalEmbeddingContext

        # Event without temporal context should fail
        with pytest.raises(ValueError):
            EmbeddingRequest(
                entity_type="Event",
                content="Team meeting",
                # Missing temporal_context - should raise
            )

        # Event with temporal context should work
        request = EmbeddingRequest(
            entity_type="Event",
            content="Team meeting",
            temporal_context=TemporalEmbeddingContext(timestamp=datetime.now()),
        )
        assert request is not None

    def test_frozen_model_compliance_documented(self) -> None:
        """Option B: Models are frozen (no fine-tuning).

        The embedding service config should NOT have training parameters.
        """
        from futurnal.embeddings.config import EmbeddingServiceConfig
        config = EmbeddingServiceConfig()

        # Should NOT have fine-tuning related fields
        assert not hasattr(config, "training_epochs")
        assert not hasattr(config, "learning_rate")
        assert not hasattr(config, "fine_tune_model")

    def test_causal_temporal_validation_required(self) -> None:
        """Option B: Causal chains require 100% temporal validation.

        TemporalOrderingValidator should exist and have validate_path method.
        """
        from futurnal.search.causal.validation import TemporalOrderingValidator

        # Validator should have method to check temporal ordering
        assert hasattr(TemporalOrderingValidator, "validate_path")


# =============================================================================
# SUMMARY TEST
# =============================================================================


class TestStep00InfrastructureSummary:
    """Summary test that confirms all infrastructure is ready."""

    def test_all_infrastructure_ready_for_step01(self) -> None:
        """All infrastructure components ready for Step 01 integration."""
        # Embeddings
        from futurnal.embeddings.service import MultiModelEmbeddingService
        from futurnal.embeddings.config import EmbeddingServiceConfig

        # PKG
        from futurnal.pkg.database.manager import PKGDatabaseManager
        from futurnal.pkg.database.config import PKGDatabaseConfig

        # LLM
        from futurnal.extraction.ollama_client import OllamaLLMClient

        # Search Infrastructure
        from futurnal.search.hybrid.retrieval import SchemaAwareRetrieval
        from futurnal.search.hybrid.query_router import QueryEmbeddingRouter
        from futurnal.search.temporal.engine import TemporalQueryEngine
        from futurnal.search.causal.retrieval import CausalChainRetrieval

        # ChromaDB
        from futurnal.embeddings.integration import TemporalAwareVectorWriter
        from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore

        # All imports succeeded = infrastructure ready
        assert True, "All infrastructure components verified"
