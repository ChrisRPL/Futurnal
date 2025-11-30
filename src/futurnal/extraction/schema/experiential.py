"""
Experiential Learning Module

Implements Training-Free GRPO (Group Relative Policy Optimization) for 
Ghost→Animal evolution via experiential knowledge accumulation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional, Protocol, Any, Dict

from futurnal.extraction.schema.models import (
    ExperientialKnowledge,
    SemanticAdvantage,
)


class LLMClient(Protocol):
    """Protocol for LLM interactions."""
    
    def extract(self, prompt: str) -> Any:
        """
        Run extraction on a prompt.
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            Extraction result (type depends on implementation, usually an object or dict).
        """
        ...


class ExtractionResult(Protocol):
    """Protocol for extraction results."""
    
    content: str
    confidence: float


class TrainingFreeGRPO:
    """
    Lightweight evolution without parameter updates.

    Key Innovation: Experiential knowledge = token priors (not parameters)
    """

    def __init__(
        self,
        llm: LLMClient,
        knowledge_capacity: int = 20,  # Keep top N patterns
        rollout_group_size: int = 4
    ):
        self.llm = llm
        self.experiential_knowledge: List[ExperientialKnowledge] = []
        self.knowledge_capacity = knowledge_capacity
        self.rollout_group_size = rollout_group_size

    def generate_rollouts(
        self,
        document: Any,  # Typed as Any for now, should be Document
        base_prompt: str
    ) -> List[Any]:
        """
        Generate K rollouts for same document.

        Ghost model runs K times on same input → K different extractions
        """
        rollouts = []
        for _ in range(self.rollout_group_size):
            # Add experiential knowledge as context (token priors)
            prompt = self._build_prompt_with_experience(base_prompt, document)

            # Run Ghost model (no parameter changes)
            result = self.llm.extract(prompt)
            rollouts.append(result)

        return rollouts

    def extract_semantic_advantages(
        self,
        rollouts: List[Any],
        ground_truth: Optional[Any] = None
    ) -> List[SemanticAdvantage]:
        """
        LLM introspection: compare rollouts to discover better approaches.

        Training-Free GRPO innovation: Use LLM to generate advantages
        """
        advantages = []

        # Rank rollouts by quality
        ranked = self._rank_rollouts_by_quality(rollouts, ground_truth)

        # Compare best vs worst
        # We compare the top half against the bottom half
        half_size = len(ranked) // 2
        for i in range(half_size):
            better = ranked[i]
            worse = ranked[-(i+1)]

            # Ask LLM: "Why is extraction A better than extraction B?"
            advantage = self._llm_introspect_advantage(better, worse)
            if advantage and advantage.confidence > 0.7:
                advantages.append(advantage)

        return advantages

    def update_experiential_knowledge(
        self,
        advantages: List[SemanticAdvantage]
    ):
        """
        Update knowledge base from semantic advantages.

        Key: Store as natural language (token priors), not parameters
        """
        for advantage in advantages:
            knowledge = ExperientialKnowledge(
                pattern_id=self._generate_pattern_id(),
                description=advantage.better_approach,
                context=advantage.reasoning,
                confidence=advantage.confidence,
                examples=[],
                created_at=datetime.utcnow()
            )

            self._add_to_knowledge_base(knowledge)

        # Prune to capacity
        self._prune_knowledge_base()

    def _build_prompt_with_experience(
        self,
        base_prompt: str,
        document: Any
    ) -> str:
        """
        Inject experiential knowledge as token priors.

        This is how Ghost + Experience = Animal
        """
        if not self.experiential_knowledge:
            return base_prompt

        experience_context = "## Learned Patterns\n\n"
        # Sort by confidence/success and take top 5
        sorted_knowledge = sorted(
            self.experiential_knowledge, 
            key=lambda k: k.confidence, 
            reverse=True
        )
        
        for knowledge in sorted_knowledge[:5]:  # Top 5
            experience_context += f"- {knowledge.description}\n"
            experience_context += f"  Context: {knowledge.context}\n"
            total_runs = knowledge.success_count + knowledge.failure_count
            if total_runs > 0:
                experience_context += f"  Success rate: {knowledge.success_count}/{total_runs}\n\n"
            else:
                experience_context += "  Success rate: New\n\n"

        return f"{experience_context}\n\n{base_prompt}"

    def _rank_rollouts_by_quality(
        self, 
        rollouts: List[Any], 
        ground_truth: Optional[Any] = None
    ) -> List[Any]:
        """
        Rank rollouts by quality.
        
        In a real implementation, this would use a reward model or 
        heuristic evaluation. For now, we'll assume the LLM result 
        has a confidence score or we use a placeholder ranking.
        """
        # Placeholder: sort by confidence if available, otherwise assume input order
        # This needs to be fleshed out with actual metrics
        def get_score(r):
            return getattr(r, 'confidence', 0.0)
            
        return sorted(rollouts, key=get_score, reverse=True)

    def _llm_introspect_advantage(
        self, 
        better: Any, 
        worse: Any
    ) -> Optional[SemanticAdvantage]:
        """
        Ask LLM to explain why one extraction is better than another.
        """
        # In a real implementation, this would call the LLM with a specific prompt
        # For now, we'll return a mock or None if we can't actually call the LLM here
        # This method is expected to be mocked in tests or implemented with actual LLM calls
        
        # We can't implement the actual LLM call without the prompt template
        # and the specific LLM interface.
        # Returning None to indicate this needs to be implemented/mocked
        return None

    def _generate_pattern_id(self, advantage: Optional[SemanticAdvantage] = None) -> str:
        """Generate a unique ID for a pattern."""
        return str(uuid.uuid4())

    def _add_to_knowledge_base(self, knowledge: ExperientialKnowledge):
        """Add knowledge to the list."""
        self.experiential_knowledge.append(knowledge)

    def _prune_knowledge_base(self):
        """Prune knowledge base to capacity."""
        if len(self.experiential_knowledge) > self.knowledge_capacity:
            # Sort by confidence and keep top N
            self.experiential_knowledge.sort(key=lambda k: k.confidence, reverse=True)
            self.experiential_knowledge = self.experiential_knowledge[:self.knowledge_capacity]


class WorldStateModel:
    """Track extraction quality trajectory and identify patterns."""

    def __init__(self):
        self.quality_history: List[Dict[str, float]] = []
        self.success_patterns: List[str] = []
        self.failure_patterns: List[str] = []

    def assess_extraction_trajectory(
        self,
        recent_extractions: List[Any]  # List of ExtractionResult
    ) -> Dict[str, float]:
        """
        Assess if quality is improving over time.

        Metrics:
        - Precision trend
        - Confidence trend
        - Success rate trend
        """
        if len(recent_extractions) < 10:
            return {"insufficient_data": 1.0}  # Using 1.0 as boolean True equivalent for float dict

        metrics = {
            "current_precision": self._compute_precision(recent_extractions[-10:]),
            "previous_precision": self._compute_precision(recent_extractions[-20:-10]),
            "improvement": 0.0
        }

        metrics["improvement"] = metrics["current_precision"] - metrics["previous_precision"]

        return metrics

    def generate_curriculum(
        self,
        pending_documents: List[Any]  # List of Document
    ) -> List[Any]:
        """
        Order documents by learning value.

        Priority:
        1. Documents with patterns similar to recent failures
        2. Documents with novel patterns
        3. Documents similar to recent successes
        """
        scored = []
        for doc in pending_documents:
            score = self._compute_learning_value(doc)
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored]

    def _compute_precision(self, extractions: List[Any]) -> float:
        """Compute average precision/confidence for a batch of extractions."""
        if not extractions:
            return 0.0
        
        total_score = sum(getattr(e, 'confidence', 0.0) for e in extractions)
        return total_score / len(extractions)

    def _compute_learning_value(self, doc: Any) -> float:
        """
        Compute learning value for a document.
        
        Placeholder implementation.
        """
        # In a real implementation, this would analyze document content
        # against failure/success patterns.
        return 0.5
