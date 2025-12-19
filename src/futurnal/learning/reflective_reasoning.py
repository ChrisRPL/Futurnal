"""
MM-HELIX AHPO - Adaptive Hybrid Policy Optimization for Reflective Reasoning.

Implements reflective reasoning that:
- Evaluates reasoning quality
- Adapts reasoning strategies based on feedback
- Balances exploration vs exploitation
- Stores successful patterns as priors

Research Foundation:
- MM-HELIX (2510.08540v1): Reflective reasoning with AHPO
- Training-Free GRPO: Policy optimization without gradient updates
- Self-reflection in LLM reasoning

Key Features:
- Multi-step reflective reasoning
- Adaptive strategy selection
- Quality-based strategy weighting
- Experience-based improvement

Option B Compliance:
- No model parameter updates
- Uses token priors for experience storage
- All learning is experiential (natural language)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import math
import random

logger = logging.getLogger(__name__)


class ReasoningStrategy(str, Enum):
    """Available reasoning strategies."""
    DIRECT = "direct"  # Direct answer without much reasoning
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    TREE_OF_THOUGHT = "tree_of_thought"  # Multiple paths with selection
    SELF_CONSISTENCY = "self_consistency"  # Multiple answers with voting
    REFLECTION = "reflection"  # Reason, evaluate, refine
    DECOMPOSITION = "decomposition"  # Break into sub-problems


class ReasoningOutcome(str, Enum):
    """Outcomes of reasoning attempts."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_id: str
    strategy: ReasoningStrategy
    input_text: str
    output_text: str
    confidence: float = 0.0
    execution_time_ms: float = 0.0

    # Evaluation
    quality_score: float = 0.0
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning attempt."""
    trace_id: str
    query: str
    steps: List[ReasoningStep] = field(default_factory=list)

    # Overall result
    final_answer: str = ""
    outcome: ReasoningOutcome = ReasoningOutcome.FAILURE
    overall_confidence: float = 0.0
    overall_quality: float = 0.0

    # Strategy used
    primary_strategy: ReasoningStrategy = ReasoningStrategy.DIRECT
    strategies_tried: List[ReasoningStrategy] = field(default_factory=list)

    # Timing
    total_time_ms: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StrategyWeight:
    """Weight for a reasoning strategy."""
    strategy: ReasoningStrategy
    weight: float = 1.0
    successes: int = 0
    failures: int = 0
    avg_quality: float = 0.5

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5


class AdaptivePolicyOptimizer:
    """
    Adaptive Hybrid Policy Optimization for strategy selection.

    Uses Thompson Sampling / UCB-style exploration-exploitation
    to select reasoning strategies based on past performance.
    """

    def __init__(
        self,
        exploration_factor: float = 2.0,
        min_weight: float = 0.1,
        decay_factor: float = 0.99
    ):
        """Initialize AHPO.

        Args:
            exploration_factor: UCB exploration parameter
            min_weight: Minimum weight for any strategy
            decay_factor: Weight decay for old experiences
        """
        self.exploration_factor = exploration_factor
        self.min_weight = min_weight
        self.decay_factor = decay_factor

        # Strategy weights
        self.weights: Dict[ReasoningStrategy, StrategyWeight] = {
            strategy: StrategyWeight(strategy=strategy)
            for strategy in ReasoningStrategy
        }

        # History for context-aware selection
        self.query_type_history: Dict[str, Dict[ReasoningStrategy, StrategyWeight]] = {}

    def select_strategy(
        self,
        query_type: Optional[str] = None,
        available_strategies: Optional[List[ReasoningStrategy]] = None
    ) -> ReasoningStrategy:
        """Select best strategy using UCB-style selection.

        Args:
            query_type: Type of query for context-aware selection
            available_strategies: Strategies to choose from

        Returns:
            Selected strategy
        """
        strategies = available_strategies or list(ReasoningStrategy)

        # Use query-type specific weights if available
        if query_type and query_type in self.query_type_history:
            weights = self.query_type_history[query_type]
        else:
            weights = self.weights

        # Calculate UCB scores
        total_trials = sum(w.successes + w.failures for w in weights.values())
        total_trials = max(1, total_trials)

        scores = {}
        for strategy in strategies:
            if strategy not in weights:
                weights[strategy] = StrategyWeight(strategy=strategy)

            w = weights[strategy]
            trials = w.successes + w.failures

            if trials == 0:
                # High score for unexplored strategies
                scores[strategy] = 1.0 + self.exploration_factor
            else:
                # UCB formula
                exploitation = w.avg_quality * w.success_rate
                exploration = self.exploration_factor * math.sqrt(
                    math.log(total_trials) / trials
                )
                scores[strategy] = exploitation + exploration

        # Select strategy with highest score
        best_strategy = max(scores, key=scores.get)
        return best_strategy

    def update(
        self,
        strategy: ReasoningStrategy,
        outcome: ReasoningOutcome,
        quality: float,
        query_type: Optional[str] = None
    ):
        """Update strategy weights based on outcome.

        Args:
            strategy: Strategy that was used
            outcome: Outcome of the reasoning
            quality: Quality score (0-1)
            query_type: Type of query for context-aware learning
        """
        # Update global weights
        self._update_weight(self.weights, strategy, outcome, quality)

        # Update query-type specific weights
        if query_type:
            if query_type not in self.query_type_history:
                self.query_type_history[query_type] = {
                    s: StrategyWeight(strategy=s) for s in ReasoningStrategy
                }
            self._update_weight(
                self.query_type_history[query_type],
                strategy, outcome, quality
            )

    def _update_weight(
        self,
        weights: Dict[ReasoningStrategy, StrategyWeight],
        strategy: ReasoningStrategy,
        outcome: ReasoningOutcome,
        quality: float
    ):
        """Update a specific weight dictionary."""
        if strategy not in weights:
            weights[strategy] = StrategyWeight(strategy=strategy)

        w = weights[strategy]

        # Update counts
        if outcome == ReasoningOutcome.SUCCESS:
            w.successes += 1
        elif outcome in [ReasoningOutcome.FAILURE, ReasoningOutcome.ERROR]:
            w.failures += 1

        # Update average quality with exponential moving average
        alpha = 0.3  # Learning rate for quality updates
        w.avg_quality = (1 - alpha) * w.avg_quality + alpha * quality

        # Update weight
        w.weight = max(
            self.min_weight,
            w.weight * self.decay_factor + (1 - self.decay_factor) * quality
        )

    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics about strategy performance."""
        return {
            strategy.value: {
                "weight": w.weight,
                "success_rate": w.success_rate,
                "avg_quality": w.avg_quality,
                "trials": w.successes + w.failures,
            }
            for strategy, w in self.weights.items()
        }


class ReflectiveReasoner:
    """
    Implements reflective reasoning with adaptive strategy selection.

    Uses MM-HELIX-style reflection loop:
    1. Generate initial reasoning
    2. Evaluate reasoning quality
    3. Reflect on weaknesses
    4. Refine if needed
    5. Learn from outcome
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        token_prior_store: Optional[Any] = None,
        optimizer: Optional[AdaptivePolicyOptimizer] = None,
        max_reflection_steps: int = 3
    ):
        """Initialize reflective reasoner.

        Args:
            llm_client: LLM for reasoning
            token_prior_store: Store for experiential knowledge
            optimizer: AHPO optimizer for strategy selection
            max_reflection_steps: Maximum reflection iterations
        """
        self.llm_client = llm_client
        self.token_store = token_prior_store
        self.optimizer = optimizer or AdaptivePolicyOptimizer()
        self.max_reflection_steps = max_reflection_steps

        # Reasoning history
        self.traces: List[ReasoningTrace] = []
        self.max_traces = 100

    async def reason(
        self,
        query: str,
        context: Optional[str] = None,
        preferred_strategy: Optional[ReasoningStrategy] = None
    ) -> ReasoningTrace:
        """Execute reflective reasoning on a query.

        Args:
            query: The query to reason about
            context: Optional context
            preferred_strategy: Optional strategy override

        Returns:
            ReasoningTrace with complete reasoning process
        """
        import time
        from uuid import uuid4

        start_time = time.time()

        # Initialize trace
        trace = ReasoningTrace(
            trace_id=str(uuid4()),
            query=query,
        )

        # Select strategy
        query_type = self._classify_query(query)
        strategy = preferred_strategy or self.optimizer.select_strategy(query_type)
        trace.primary_strategy = strategy
        trace.strategies_tried.append(strategy)

        # Execute reasoning with selected strategy
        initial_step = await self._execute_strategy(query, strategy, context)
        trace.steps.append(initial_step)

        # Reflection loop
        current_answer = initial_step.output_text
        current_quality = initial_step.quality_score

        for reflection_idx in range(self.max_reflection_steps):
            if current_quality >= 0.8:  # Good enough, no reflection needed
                break

            # Evaluate and reflect
            evaluation = await self._evaluate_reasoning(query, current_answer, context)

            if evaluation["is_satisfactory"]:
                break

            # Reflect on weaknesses
            reflection = await self._reflect(
                query, current_answer, evaluation["weaknesses"], context
            )

            # Try to improve
            improved_step = await self._improve_reasoning(
                query, current_answer, reflection, context
            )
            trace.steps.append(improved_step)

            if improved_step.quality_score > current_quality:
                current_answer = improved_step.output_text
                current_quality = improved_step.quality_score

        # Finalize trace
        trace.final_answer = current_answer
        trace.overall_quality = current_quality
        trace.overall_confidence = self._calculate_confidence(trace)
        trace.outcome = (
            ReasoningOutcome.SUCCESS if current_quality >= 0.6
            else ReasoningOutcome.PARTIAL_SUCCESS if current_quality >= 0.3
            else ReasoningOutcome.FAILURE
        )
        trace.total_time_ms = (time.time() - start_time) * 1000

        # Update optimizer
        self.optimizer.update(
            strategy=strategy,
            outcome=trace.outcome,
            quality=trace.overall_quality,
            query_type=query_type
        )

        # Store trace
        self._store_trace(trace)

        # Learn from experience
        if self.token_store and trace.outcome == ReasoningOutcome.SUCCESS:
            await self._store_experiential_knowledge(trace)

        return trace

    def reason_sync(
        self,
        query: str,
        context: Optional[str] = None,
        preferred_strategy: Optional[ReasoningStrategy] = None
    ) -> ReasoningTrace:
        """Synchronous version of reason."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.reason(query, context, preferred_strategy)
        )

    async def _execute_strategy(
        self,
        query: str,
        strategy: ReasoningStrategy,
        context: Optional[str]
    ) -> ReasoningStep:
        """Execute a specific reasoning strategy."""
        from uuid import uuid4
        import time

        start_time = time.time()

        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            output = await self._chain_of_thought(query, context)
        elif strategy == ReasoningStrategy.REFLECTION:
            output = await self._reflective_reasoning(query, context)
        elif strategy == ReasoningStrategy.DECOMPOSITION:
            output = await self._decomposition_reasoning(query, context)
        elif strategy == ReasoningStrategy.SELF_CONSISTENCY:
            output = await self._self_consistency(query, context)
        else:  # DIRECT
            output = await self._direct_reasoning(query, context)

        execution_time = (time.time() - start_time) * 1000

        # Evaluate quality
        quality = await self._estimate_quality(query, output, context)

        return ReasoningStep(
            step_id=str(uuid4()),
            strategy=strategy,
            input_text=query,
            output_text=output,
            quality_score=quality,
            execution_time_ms=execution_time,
        )

    async def _direct_reasoning(
        self,
        query: str,
        context: Optional[str]
    ) -> str:
        """Direct reasoning without elaborate process."""
        prompt = f"""Answer the following question directly:

Question: {query}
{f'Context: {context}' if context else ''}

Provide a clear, concise answer."""

        return await self._call_llm(prompt)

    async def _chain_of_thought(
        self,
        query: str,
        context: Optional[str]
    ) -> str:
        """Chain of thought reasoning."""
        prompt = f"""Think through this step by step:

Question: {query}
{f'Context: {context}' if context else ''}

Let me think through this step by step:
1."""

        response = await self._call_llm(prompt)

        # Extract final answer
        if "therefore" in response.lower():
            parts = response.lower().split("therefore")
            return parts[-1].strip()

        return response

    async def _reflective_reasoning(
        self,
        query: str,
        context: Optional[str]
    ) -> str:
        """Reflective reasoning with self-evaluation."""
        # Initial reasoning
        initial = await self._direct_reasoning(query, context)

        # Self-evaluation
        eval_prompt = f"""Evaluate this answer:

Question: {query}
Answer: {initial}

Is this answer correct and complete? What could be improved?"""

        evaluation = await self._call_llm(eval_prompt)

        # Refinement if needed
        if "improve" in evaluation.lower() or "incorrect" in evaluation.lower():
            refine_prompt = f"""Based on this evaluation, provide an improved answer:

Question: {query}
Original answer: {initial}
Evaluation: {evaluation}

Improved answer:"""

            return await self._call_llm(refine_prompt)

        return initial

    async def _decomposition_reasoning(
        self,
        query: str,
        context: Optional[str]
    ) -> str:
        """Break down into sub-problems."""
        # Decompose
        decompose_prompt = f"""Break this question into smaller sub-questions:

Question: {query}
{f'Context: {context}' if context else ''}

Sub-questions:
1."""

        sub_questions = await self._call_llm(decompose_prompt)

        # Answer sub-questions
        answers_prompt = f"""Answer each sub-question:

Main question: {query}
Sub-questions:
{sub_questions}

Answers for each:"""

        sub_answers = await self._call_llm(answers_prompt)

        # Synthesize
        synthesize_prompt = f"""Synthesize a final answer:

Main question: {query}
Sub-answers:
{sub_answers}

Final answer:"""

        return await self._call_llm(synthesize_prompt)

    async def _self_consistency(
        self,
        query: str,
        context: Optional[str],
        num_samples: int = 3
    ) -> str:
        """Generate multiple answers and vote."""
        answers = []

        for _ in range(num_samples):
            answer = await self._direct_reasoning(query, context)
            answers.append(answer)

        # Simple majority voting (or semantic similarity in production)
        if len(set(answers)) == 1:
            return answers[0]

        # Pick most common or first if no clear winner
        from collections import Counter
        counter = Counter(answers)
        most_common = counter.most_common(1)[0][0]
        return most_common

    async def _evaluate_reasoning(
        self,
        query: str,
        answer: str,
        context: Optional[str]
    ) -> Dict[str, Any]:
        """Evaluate the quality of reasoning."""
        prompt = f"""Evaluate this answer quality:

Question: {query}
Answer: {answer}
{f'Context: {context}' if context else ''}

Rate from 1-10 and list any weaknesses.
Format:
SCORE: X/10
WEAKNESSES: <list weaknesses>
SATISFACTORY: yes/no"""

        response = await self._call_llm(prompt)

        # Parse response
        score = 5
        weaknesses = []
        satisfactory = False

        for line in response.split("\n"):
            if "SCORE:" in line.upper():
                try:
                    score = int(line.split(":")[1].strip().split("/")[0])
                except (ValueError, IndexError):
                    pass
            elif "WEAKNESSES:" in line.upper():
                weaknesses = [w.strip() for w in line.split(":")[1].split(",")]
            elif "SATISFACTORY:" in line.upper():
                satisfactory = "yes" in line.lower()

        return {
            "score": score,
            "weaknesses": weaknesses,
            "is_satisfactory": satisfactory or score >= 7,
        }

    async def _reflect(
        self,
        query: str,
        answer: str,
        weaknesses: List[str],
        context: Optional[str]
    ) -> str:
        """Reflect on weaknesses and suggest improvements."""
        prompt = f"""Reflect on how to improve this answer:

Question: {query}
Current answer: {answer}
Weaknesses identified: {', '.join(weaknesses)}

How can we address these weaknesses? What additional reasoning is needed?"""

        return await self._call_llm(prompt)

    async def _improve_reasoning(
        self,
        query: str,
        current_answer: str,
        reflection: str,
        context: Optional[str]
    ) -> ReasoningStep:
        """Generate improved reasoning based on reflection."""
        from uuid import uuid4
        import time

        start_time = time.time()

        prompt = f"""Based on the reflection, provide an improved answer:

Question: {query}
Current answer: {current_answer}
Reflection: {reflection}
{f'Context: {context}' if context else ''}

Improved answer:"""

        improved = await self._call_llm(prompt)
        quality = await self._estimate_quality(query, improved, context)

        return ReasoningStep(
            step_id=str(uuid4()),
            strategy=ReasoningStrategy.REFLECTION,
            input_text=query,
            output_text=improved,
            quality_score=quality,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    async def _estimate_quality(
        self,
        query: str,
        answer: str,
        context: Optional[str]
    ) -> float:
        """Estimate quality of an answer."""
        # Simple heuristics if no LLM
        if not self.llm_client:
            # Length-based heuristic
            if len(answer) < 10:
                return 0.2
            elif len(answer) > 500:
                return 0.6
            return 0.5

        # LLM-based quality estimation
        prompt = f"""Rate the quality of this answer from 0.0 to 1.0:

Question: {query}
Answer: {answer}

Just respond with a number between 0.0 and 1.0."""

        response = await self._call_llm(prompt)

        try:
            import re
            numbers = re.findall(r"0\.\d+|1\.0|0|1", response)
            if numbers:
                return float(numbers[0])
        except (ValueError, IndexError):
            pass

        return 0.5

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM."""
        if not self.llm_client:
            return "LLM not available"

        try:
            if hasattr(self.llm_client, "generate"):
                return await self.llm_client.generate(prompt)
            elif hasattr(self.llm_client, "chat"):
                response = await self.llm_client.chat([{"role": "user", "content": prompt}])
                return response.get("content", "")
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")

        return ""

    def _classify_query(self, query: str) -> str:
        """Classify query type for context-aware strategy selection."""
        query_lower = query.lower()

        if any(w in query_lower for w in ["why", "cause", "reason"]):
            return "causal"
        elif any(w in query_lower for w in ["how", "steps", "process"]):
            return "procedural"
        elif any(w in query_lower for w in ["what", "define", "explain"]):
            return "definitional"
        elif any(w in query_lower for w in ["compare", "difference", "versus"]):
            return "comparative"
        elif any(w in query_lower for w in ["predict", "will", "future"]):
            return "predictive"

        return "general"

    def _calculate_confidence(self, trace: ReasoningTrace) -> float:
        """Calculate overall confidence for a trace."""
        if not trace.steps:
            return 0.0

        # Average quality across steps
        avg_quality = sum(s.quality_score for s in trace.steps) / len(trace.steps)

        # Bonus for successful reflections
        if len(trace.steps) > 1:
            improvement = trace.steps[-1].quality_score - trace.steps[0].quality_score
            if improvement > 0:
                avg_quality += 0.1

        return min(1.0, avg_quality)

    def _store_trace(self, trace: ReasoningTrace):
        """Store trace in history."""
        self.traces.append(trace)
        if len(self.traces) > self.max_traces:
            self.traces = self.traces[-self.max_traces:]

    async def _store_experiential_knowledge(self, trace: ReasoningTrace):
        """Store successful reasoning as experiential knowledge."""
        if not self.token_store:
            return

        try:
            # Store as natural language prior
            description = (
                f"For queries like '{trace.query[:50]}...', "
                f"the {trace.primary_strategy.value} strategy worked well "
                f"(quality: {trace.overall_quality:.0%}). "
                f"Key insight: {trace.final_answer[:100]}..."
            )

            # Use token store API if available
            if hasattr(self.token_store, "add_reasoning_pattern"):
                self.token_store.add_reasoning_pattern(
                    pattern_type="successful_reasoning",
                    description=description,
                    confidence=trace.overall_confidence,
                )
        except Exception as e:
            logger.warning(f"Failed to store experiential knowledge: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning statistics."""
        total = len(self.traces)
        if total == 0:
            return {"total_traces": 0}

        successes = sum(
            1 for t in self.traces
            if t.outcome == ReasoningOutcome.SUCCESS
        )

        avg_quality = sum(t.overall_quality for t in self.traces) / total
        avg_steps = sum(len(t.steps) for t in self.traces) / total
        avg_time = sum(t.total_time_ms for t in self.traces) / total

        return {
            "total_traces": total,
            "success_rate": successes / total,
            "avg_quality": avg_quality,
            "avg_steps": avg_steps,
            "avg_time_ms": avg_time,
            "strategy_stats": self.optimizer.get_strategy_stats(),
        }
