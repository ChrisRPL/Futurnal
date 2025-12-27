"""
Futurnal Agents Module.

Specialized agents for various tasks:
- Web browsing and information retrieval
- Research synthesis
- Future prediction
- Multi-agent collaboration
"""

from .web_browser import (
    WebBrowsingAgent,
    BrowsingResult,
    BrowsingAction,
    WebPage,
    SourceReliability,
)
from .future_prediction import (
    FuturePredictionEngine,
    FuturePrediction,
    PredictionType,
    ConfidenceLevel,
    ScenarioAnalysis,
)
from .deep_research import (
    PersonalizedResearchAgent,
    ResearchResult,
    UserProfile,
    ExpertiseLevel,
    ResearchDepth,
)
from .multi_agent import (
    MultiAgentOrchestrator,
    Agent,
    AgentRole,
    AgentTask,
    AgentMessage,
    MessageType,
)

# Phase 2E: AgentFlow Architecture
from .memory_buffer import (
    EvolvingMemoryBuffer,
    MemoryEntry,
    MemoryEntryType,
    MemoryPriority,
    get_memory_buffer,
)
from .correlation_planner import (
    CorrelationPlanner,
    CorrelationHypothesis,
    HypothesisType,
    HypothesisStatus,
    QueryPlan,
    get_correlation_planner,
)
from .correlation_verifier import (
    CorrelationVerifier,
    VerificationResult,
    VerificationReport,
    EvidenceItem,
    BradfordHillCriterion,
    get_correlation_verifier,
)

__all__ = [
    # Web Browser
    "WebBrowsingAgent",
    "BrowsingResult",
    "BrowsingAction",
    "WebPage",
    "SourceReliability",
    # Future Prediction
    "FuturePredictionEngine",
    "FuturePrediction",
    "PredictionType",
    "ConfidenceLevel",
    "ScenarioAnalysis",
    # Deep Research
    "PersonalizedResearchAgent",
    "ResearchResult",
    "UserProfile",
    "ExpertiseLevel",
    "ResearchDepth",
    # Multi-Agent
    "MultiAgentOrchestrator",
    "Agent",
    "AgentRole",
    "AgentTask",
    "AgentMessage",
    "MessageType",
    # Phase 2E: AgentFlow Architecture
    "EvolvingMemoryBuffer",
    "MemoryEntry",
    "MemoryEntryType",
    "MemoryPriority",
    "get_memory_buffer",
    "CorrelationPlanner",
    "CorrelationHypothesis",
    "HypothesisType",
    "HypothesisStatus",
    "QueryPlan",
    "get_correlation_planner",
    "CorrelationVerifier",
    "VerificationResult",
    "VerificationReport",
    "EvidenceItem",
    "BradfordHillCriterion",
    "get_correlation_verifier",
]
