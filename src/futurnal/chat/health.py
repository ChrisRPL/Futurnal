"""Chat Intelligence Health Check.

Provides diagnostic information about the intelligence infrastructure
to help users understand why conversations might not be working as expected.

Research Foundation:
- Transparency for trust-building with users
- Debug visibility for Phase 1 completion validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: str  # "connected", "disconnected", "error", "not_initialized"
    details: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntelligenceHealthReport:
    """Complete health report for intelligence infrastructure."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_status: str = "unknown"
    components: List[ComponentHealth] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Check if all critical components are connected."""
        critical = ["pkg_database", "graphrag", "ollama", "causal_intelligence"]
        for comp in self.components:
            if comp.name in critical and comp.status != "connected":
                return False
        return True

    @property
    def intelligence_capabilities(self) -> Dict[str, bool]:
        """Get a summary of which intelligence capabilities are available."""
        capabilities = {
            "knowledge_graph": False,
            "semantic_search": False,
            "causal_discovery": False,
            "hypothesis_generation": False,
            "link_prediction": False,
            "reflective_reasoning": False,
            "community_detection": False,
            "advanced_agents": False,
        }

        for comp in self.components:
            if comp.status == "connected":
                if comp.name == "pkg_database":
                    capabilities["knowledge_graph"] = True
                elif comp.name == "graphrag":
                    capabilities["semantic_search"] = True
                elif comp.name == "causal_intelligence":
                    capabilities["causal_discovery"] = True
                    capabilities["hypothesis_generation"] = True
                elif comp.name == "link_prediction":
                    capabilities["link_prediction"] = True
                elif comp.name == "reflective_reasoning":
                    capabilities["reflective_reasoning"] = True
                elif comp.name == "community_detection":
                    capabilities["community_detection"] = True
                elif comp.name == "agent_capabilities":
                    capabilities["advanced_agents"] = True

        return capabilities

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status,
            "is_healthy": self.is_healthy,
            "intelligence_capabilities": self.intelligence_capabilities,
            "components": [
                {
                    "name": c.name,
                    "status": c.status,
                    "details": c.details,
                    "metrics": c.metrics,
                }
                for c in self.components
            ],
            "recommendations": self.recommendations,
        }


async def check_intelligence_health() -> IntelligenceHealthReport:
    """Check health of all intelligence infrastructure components.

    Returns comprehensive health report showing:
    - PKG database connection status
    - GraphRAG pipeline status
    - Ollama LLM availability
    - ChromaDB vector store status
    - Experiential learning status
    - Autonomous loop status

    Example:
        >>> report = await check_intelligence_health()
        >>> if not report.is_healthy:
        ...     print("Issues found:", report.recommendations)
    """
    report = IntelligenceHealthReport()

    # Check PKG Database (Neo4j)
    pkg_health = await _check_pkg_database()
    report.components.append(pkg_health)

    # Check GraphRAG Pipeline
    graphrag_health = await _check_graphrag()
    report.components.append(graphrag_health)

    # Check Ollama LLM
    ollama_health = await _check_ollama()
    report.components.append(ollama_health)

    # Check ChromaDB
    chroma_health = await _check_chromadb()
    report.components.append(chroma_health)

    # Check Experiential Learning
    learning_health = await _check_experiential_learning()
    report.components.append(learning_health)

    # Check Autonomous Loop
    loop_health = await _check_autonomous_loop()
    report.components.append(loop_health)

    # Check Causal Intelligence Pipeline
    causal_health = await _check_causal_intelligence()
    report.components.append(causal_health)

    # Check Link Prediction
    link_health = await _check_link_prediction()
    report.components.append(link_health)

    # Check Reflective Reasoning (AHPO)
    reasoning_health = await _check_reflective_reasoning()
    report.components.append(reasoning_health)

    # Check Agent Capabilities
    agent_health = await _check_agent_capabilities()
    report.components.append(agent_health)

    # Check Community Detection
    community_health = await _check_community_detection()
    report.components.append(community_health)

    # Determine overall status
    connected_count = sum(1 for c in report.components if c.status == "connected")
    total_count = len(report.components)

    if connected_count == total_count:
        report.overall_status = "healthy"
    elif connected_count >= 3:  # PKG, GraphRAG, Ollama minimum
        report.overall_status = "degraded"
    else:
        report.overall_status = "unhealthy"

    # Generate recommendations
    report.recommendations = _generate_recommendations(report.components)

    return report


async def _check_pkg_database() -> ComponentHealth:
    """Check PKG (Neo4j) database connection."""
    try:
        from futurnal.configuration.settings import bootstrap_settings
        from futurnal.pkg.database.manager import PKGDatabaseManager

        settings = bootstrap_settings()
        if settings and settings.workspace and settings.workspace.storage:
            manager = PKGDatabaseManager(settings.workspace.storage)
            driver = manager.connect()

            if driver:
                # Get statistics
                stats = manager.get_statistics()
                return ComponentHealth(
                    name="pkg_database",
                    status="connected",
                    details="Neo4j PKG database connected",
                    metrics={
                        "node_count": stats.get("node_count", 0),
                        "relationship_count": stats.get("relationship_count", 0),
                    },
                )

        return ComponentHealth(
            name="pkg_database",
            status="disconnected",
            details="No storage settings configured",
        )

    except Exception as e:
        logger.debug(f"PKG health check failed: {e}")
        return ComponentHealth(
            name="pkg_database",
            status="error",
            details=str(e),
        )


async def _check_graphrag() -> ComponentHealth:
    """Check GraphRAG pipeline status."""
    try:
        from futurnal.search.api import create_hybrid_search_api

        api = create_hybrid_search_api(graphrag_enabled=True)

        if api.schema_retrieval is not None:
            return ComponentHealth(
                name="graphrag",
                status="connected",
                details="GraphRAG pipeline initialized with SchemaAwareRetrieval",
                metrics={
                    "has_temporal_engine": api.schema_retrieval._temporal_engine is not None,
                    "has_causal_retrieval": api.schema_retrieval._causal_retrieval is not None,
                },
            )
        else:
            return ComponentHealth(
                name="graphrag",
                status="degraded",
                details="GraphRAG fallback to keyword search (PKG not connected)",
            )

    except Exception as e:
        logger.debug(f"GraphRAG health check failed: {e}")
        return ComponentHealth(
            name="graphrag",
            status="error",
            details=str(e),
        )


async def _check_ollama() -> ComponentHealth:
    """Check Ollama LLM availability."""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:11434/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    model_names = [m.get("name", "") for m in models]

                    return ComponentHealth(
                        name="ollama",
                        status="connected",
                        details=f"Ollama running with {len(models)} models",
                        metrics={
                            "models_available": model_names[:5],
                            "model_count": len(models),
                        },
                    )
                else:
                    return ComponentHealth(
                        name="ollama",
                        status="error",
                        details=f"Ollama returned status {response.status}",
                    )

    except Exception as e:
        logger.debug(f"Ollama health check failed: {e}")
        return ComponentHealth(
            name="ollama",
            status="disconnected",
            details="Ollama not running on localhost:11434. Start with 'ollama serve'",
        )


async def _check_chromadb() -> ComponentHealth:
    """Check ChromaDB vector store status."""
    try:
        from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore
        from futurnal.embeddings.config import EmbeddingServiceConfig

        config = EmbeddingServiceConfig()
        store = SchemaVersionedEmbeddingStore(config=config)

        # Check if we can access the collection
        collection_count = store.count() if hasattr(store, 'count') else 0

        return ComponentHealth(
            name="chromadb",
            status="connected",
            details="ChromaDB vector store initialized",
            metrics={
                "embedding_count": collection_count,
            },
        )

    except Exception as e:
        logger.debug(f"ChromaDB health check failed: {e}")
        return ComponentHealth(
            name="chromadb",
            status="error",
            details=str(e),
        )


async def _check_experiential_learning() -> ComponentHealth:
    """Check experiential learning pipeline status."""
    try:
        from futurnal.learning.integration import get_persistent_pipeline

        pipeline = get_persistent_pipeline()

        if pipeline:
            return ComponentHealth(
                name="experiential_learning",
                status="connected",
                details="Experiential learning pipeline loaded",
                metrics={
                    "documents_processed": pipeline.state.total_documents_processed,
                    "entity_priors": len(pipeline.token_store.entity_priors),
                    "relation_priors": len(pipeline.token_store.relation_priors),
                    "temporal_priors": len(pipeline.token_store.temporal_priors),
                    "quality_improvement": f"{pipeline.state.overall_quality_improvement:.1f}%",
                },
            )
        else:
            return ComponentHealth(
                name="experiential_learning",
                status="not_initialized",
                details="No learning data yet - process documents to start learning",
            )

    except Exception as e:
        logger.debug(f"Experiential learning health check failed: {e}")
        return ComponentHealth(
            name="experiential_learning",
            status="error",
            details=str(e),
        )


async def _check_autonomous_loop() -> ComponentHealth:
    """Check autonomous analysis loop status."""
    try:
        # Check if insight jobs module is available
        from futurnal.orchestrator.insight_jobs import InsightJobExecutor
        from futurnal.insights.curiosity_engine import CuriosityEngine
        from futurnal.insights.emergent_insights import InsightGenerator

        # These can be imported, meaning the modules exist
        return ComponentHealth(
            name="autonomous_loop",
            status="connected",
            details="Autonomous analysis components available",
            metrics={
                "curiosity_engine": "available",
                "insight_generator": "available",
                "insight_executor": "available",
            },
        )

    except Exception as e:
        logger.debug(f"Autonomous loop health check failed: {e}")
        return ComponentHealth(
            name="autonomous_loop",
            status="error",
            details=str(e),
        )


async def _check_causal_intelligence() -> ComponentHealth:
    """Check causal intelligence pipeline status."""
    try:
        from futurnal.insights.hypothesis_generation import HypothesisPipeline
        from futurnal.insights.interactive_causal import InteractiveCausalDiscoveryAgent
        from futurnal.search.temporal.correlation import TemporalCorrelationDetector

        metrics = {
            "hypothesis_pipeline": "available",
            "icda_agent": "available",
            "correlation_detector": "available",
        }

        # Check if Bradford-Hill validator is available
        try:
            from futurnal.search.causal.bradford_hill import BradfordHillValidator
            metrics["bradford_hill_validator"] = "available"
        except ImportError:
            metrics["bradford_hill_validator"] = "not_available"

        return ComponentHealth(
            name="causal_intelligence",
            status="connected",
            details="Causal discovery pipeline components available",
            metrics=metrics,
        )

    except Exception as e:
        logger.debug(f"Causal intelligence health check failed: {e}")
        return ComponentHealth(
            name="causal_intelligence",
            status="error",
            details=str(e),
        )


async def _check_link_prediction() -> ComponentHealth:
    """Check link prediction and knowledge completion status."""
    try:
        from futurnal.extraction.link_prediction import (
            LinkPredictor,
            KnowledgeBaseCompleter,
        )

        return ComponentHealth(
            name="link_prediction",
            status="connected",
            details="Link prediction and KB completion available",
            metrics={
                "link_predictor": "available",
                "kb_completer": "available",
            },
        )

    except Exception as e:
        logger.debug(f"Link prediction health check failed: {e}")
        return ComponentHealth(
            name="link_prediction",
            status="error",
            details=str(e),
        )


async def _check_reflective_reasoning() -> ComponentHealth:
    """Check MM-HELIX reflective reasoning with AHPO status."""
    try:
        from futurnal.learning.reflective_reasoning import (
            ReflectiveReasoner,
            AdaptivePolicyOptimizer,
        )

        return ComponentHealth(
            name="reflective_reasoning",
            status="connected",
            details="AHPO reflective reasoning pipeline available",
            metrics={
                "reflective_reasoner": "available",
                "adaptive_policy_optimizer": "available",
            },
        )

    except Exception as e:
        logger.debug(f"Reflective reasoning health check failed: {e}")
        return ComponentHealth(
            name="reflective_reasoning",
            status="error",
            details=str(e),
        )


async def _check_agent_capabilities() -> ComponentHealth:
    """Check advanced agent capabilities (web browsing, prediction, research)."""
    try:
        capabilities = {}

        # Check web browsing agent
        try:
            from futurnal.agents.web_browser import WebBrowsingAgent
            capabilities["web_browsing"] = "available"
        except ImportError:
            capabilities["web_browsing"] = "not_available"

        # Check future prediction engine
        try:
            from futurnal.agents.future_prediction import FuturePredictionEngine
            capabilities["future_prediction"] = "available"
        except ImportError:
            capabilities["future_prediction"] = "not_available"

        # Check personalized research agent
        try:
            from futurnal.agents.deep_research import PersonalizedResearchAgent
            capabilities["personalized_research"] = "available"
        except ImportError:
            capabilities["personalized_research"] = "not_available"

        # Check multi-agent orchestrator
        try:
            from futurnal.agents.multi_agent import MultiAgentOrchestrator
            capabilities["multi_agent"] = "available"
        except ImportError:
            capabilities["multi_agent"] = "not_available"

        available_count = sum(1 for v in capabilities.values() if v == "available")
        status = "connected" if available_count >= 2 else "degraded" if available_count >= 1 else "not_initialized"

        return ComponentHealth(
            name="agent_capabilities",
            status=status,
            details=f"{available_count}/4 advanced agent capabilities available",
            metrics=capabilities,
        )

    except Exception as e:
        logger.debug(f"Agent capabilities health check failed: {e}")
        return ComponentHealth(
            name="agent_capabilities",
            status="error",
            details=str(e),
        )


async def _check_community_detection() -> ComponentHealth:
    """Check community detection for hierarchical knowledge organization."""
    try:
        from futurnal.search.community.detection import (
            CommunityDetector,
            LouvainDetector,
            LeidenDetector,
            DuallyPerceivedDetector,
        )

        return ComponentHealth(
            name="community_detection",
            status="connected",
            details="Community detection available (Louvain, Leiden, DuallyPerceived)",
            metrics={
                "community_detector": "available",
                "louvain_detector": "available",
                "leiden_detector": "available",
                "dually_perceived_detector": "available",
            },
        )

    except Exception as e:
        logger.debug(f"Community detection health check failed: {e}")
        return ComponentHealth(
            name="community_detection",
            status="error",
            details=str(e),
        )


def _generate_recommendations(components: List[ComponentHealth]) -> List[str]:
    """Generate actionable recommendations based on component health."""
    recommendations = []

    for comp in components:
        if comp.status == "disconnected":
            if comp.name == "pkg_database":
                recommendations.append(
                    "Start Neo4j: Run 'neo4j start' or configure connection in ~/.futurnal/config.yaml"
                )
            elif comp.name == "ollama":
                recommendations.append(
                    "Start Ollama: Run 'ollama serve' and ensure a model is pulled (e.g., 'ollama pull llama3.1:8b')"
                )
            elif comp.name == "graphrag":
                recommendations.append(
                    "GraphRAG requires PKG database - ensure Neo4j is running"
                )

        elif comp.status == "not_initialized":
            if comp.name == "experiential_learning":
                recommendations.append(
                    "Process documents to enable learning: 'futurnal sources obsidian vault scan <vault>'"
                )
            elif comp.name == "agent_capabilities":
                recommendations.append(
                    "Advanced agents (web browsing, research, prediction) will activate after initial knowledge graph population"
                )

        elif comp.status == "degraded":
            if comp.name == "agent_capabilities":
                recommendations.append(
                    "Some advanced agents are available. Full capabilities require all agent modules to be initialized."
                )

        elif comp.status == "error":
            if comp.name == "causal_intelligence":
                recommendations.append(
                    "Causal intelligence error - ensure hypothesis_generation.py and interactive_causal.py exist"
                )
            elif comp.name == "link_prediction":
                recommendations.append(
                    "Link prediction error - ensure link_prediction.py is properly configured"
                )
            elif comp.name == "reflective_reasoning":
                recommendations.append(
                    "Reflective reasoning error - ensure reflective_reasoning.py is properly configured"
                )
            elif comp.name == "community_detection":
                recommendations.append(
                    "Community detection error - ensure community/detection.py exists"
                )
            else:
                recommendations.append(
                    f"Fix {comp.name}: {comp.details}"
                )

    if not recommendations:
        recommendations.append("All systems operational - full intelligence infrastructure is ready")
        recommendations.append("Causal discovery, hypothesis generation, and advanced agents are all available")

    return recommendations
