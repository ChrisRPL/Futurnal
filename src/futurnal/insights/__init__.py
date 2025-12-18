"""Phase 2 (Analyst): Emergent Insights and Curiosity Engine.

This module will implement the Ghost's proactive intelligence capabilities,
surfacing autonomous discoveries and knowledge gaps to users.

Phase 2 Capabilities (TO BE IMPLEMENTED):
- **Emergent Insights Generation**: Convert detected patterns into human-readable
  insights with supporting evidence (e.g., "A pattern has been detected: 75% of
  your project proposals written on Monday are accepted...")
- **Curiosity Engine**: Identify significant gaps in the user's knowledge network
  that warrant exploration (e.g., "15 notes reference 'Project Titan' but none
  are linked to your aspiration 'Lead high-impact projects'")
- **Insight Prioritization**: Rank insights by relevance, novelty, and alignment
  with user's Aspirational Self
- **Evidence Collection**: Gather supporting examples and counterexamples for
  each insight

Current Status: SCAFFOLDING ONLY
All classes are stubs with TODO markers. Phase 2 implementation will enable
proactive, autonomous insight generation.
"""

from .emergent_insights import EmergentInsight, InsightGenerator
from .curiosity_engine import CuriosityEngine, KnowledgeGap
from .insight_storage import UserInsight, InsightStorageService, get_insight_service

__all__ = [
    "EmergentInsight",
    "InsightGenerator",
    "CuriosityEngine",
    "KnowledgeGap",
    "UserInsight",
    "InsightStorageService",
    "get_insight_service",
]
