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
]
