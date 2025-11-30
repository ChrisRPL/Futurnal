"""
Tests for Experiential Learning Module
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from futurnal.extraction.schema.models import (
    ExperientialKnowledge,
    SemanticAdvantage,
)
from futurnal.extraction.schema.experiential import (
    TrainingFreeGRPO,
    WorldStateModel,
)


class MockExtractionResult:
    def __init__(self, content: str, confidence: float):
        self.content = content
        self.confidence = confidence


class MockDocument:
    def __init__(self, content: str, doc_id: str):
        self.content = content
        self.doc_id = doc_id


class TestExperientialModels:
    def test_experiential_knowledge_creation(self):
        """Test creating ExperientialKnowledge model."""
        knowledge = ExperientialKnowledge(
            pattern_id="test-id",
            description="Better way",
            context="Context",
            confidence=0.8,
            examples=["ex1"]
        )
        assert knowledge.pattern_id == "test-id"
        assert knowledge.confidence == 0.8
        assert knowledge.created_at is not None

    def test_semantic_advantage_creation(self):
        """Test creating SemanticAdvantage model."""
        advantage = SemanticAdvantage(
            better_approach="A",
            worse_approach="B",
            reasoning="Because A is better",
            confidence=0.9
        )
        assert advantage.better_approach == "A"
        assert advantage.confidence == 0.9


class TestTrainingFreeGRPO:
    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.extract.return_value = MockExtractionResult("result", 0.8)
        return llm

    def test_initialization(self, mock_llm):
        """Test initialization of TrainingFreeGRPO."""
        grpo = TrainingFreeGRPO(mock_llm, knowledge_capacity=10, rollout_group_size=2)
        assert grpo.knowledge_capacity == 10
        assert grpo.rollout_group_size == 2
        assert len(grpo.experiential_knowledge) == 0

    def test_generate_rollouts(self, mock_llm):
        """Test generating rollouts."""
        grpo = TrainingFreeGRPO(mock_llm, rollout_group_size=3)
        document = MockDocument("content", "doc1")
        
        rollouts = grpo.generate_rollouts(document, "Extract entities")
        
        assert len(rollouts) == 3
        assert mock_llm.extract.call_count == 3

    def test_update_experiential_knowledge(self, mock_llm):
        """Test updating knowledge base."""
        grpo = TrainingFreeGRPO(mock_llm, knowledge_capacity=2)
        
        advantages = [
            SemanticAdvantage(
                better_approach="Appr1", 
                worse_approach="Appr2", 
                reasoning="R1", 
                confidence=0.9
            ),
            SemanticAdvantage(
                better_approach="Appr3", 
                worse_approach="Appr4", 
                reasoning="R2", 
                confidence=0.8
            ),
            SemanticAdvantage(
                better_approach="Appr5", 
                worse_approach="Appr6", 
                reasoning="R3", 
                confidence=0.7
            )
        ]
        
        grpo.update_experiential_knowledge(advantages)
        
        # Should be pruned to capacity 2
        assert len(grpo.experiential_knowledge) == 2
        # Should keep highest confidence
        confidences = [k.confidence for k in grpo.experiential_knowledge]
        assert 0.9 in confidences
        assert 0.8 in confidences
        assert 0.7 not in confidences

    def test_prompt_construction_with_knowledge(self, mock_llm):
        """Test prompt construction with injected knowledge."""
        grpo = TrainingFreeGRPO(mock_llm)
        
        # Add some knowledge
        knowledge = ExperientialKnowledge(
            pattern_id="p1",
            description="Use specific format",
            context="When extracting dates",
            confidence=0.9,
            success_count=5,
            failure_count=1
        )
        grpo.experiential_knowledge.append(knowledge)
        
        prompt = grpo._build_prompt_with_experience("Base prompt", MockDocument("c", "d"))
        
        assert "## Learned Patterns" in prompt
        assert "Use specific format" in prompt
        assert "Success rate: 5/6" in prompt
        assert "Base prompt" in prompt

    def test_extract_semantic_advantages(self, mock_llm):
        """Test extracting semantic advantages."""
        grpo = TrainingFreeGRPO(mock_llm)
        
        # Mock _llm_introspect_advantage since it's not fully implemented
        grpo._llm_introspect_advantage = Mock(return_value=SemanticAdvantage(
            better_approach="Better",
            worse_approach="Worse",
            reasoning="Reason",
            confidence=0.8
        ))
        
        rollouts = [
            MockExtractionResult("Best", 0.9),
            MockExtractionResult("Good", 0.8),
            MockExtractionResult("Bad", 0.4),
            MockExtractionResult("Worst", 0.2)
        ]
        
        advantages = grpo.extract_semantic_advantages(rollouts)
        
        assert len(advantages) > 0
        assert advantages[0].better_approach == "Better"


class TestWorldStateModel:
    def test_assess_extraction_trajectory(self):
        """Test assessing trajectory."""
        model = WorldStateModel()
        
        # Not enough data
        metrics = model.assess_extraction_trajectory([])
        assert "insufficient_data" in metrics
        
        # Create dummy history
        history = [MockExtractionResult("r", 0.5 + i*0.01) for i in range(20)]
        
        metrics = model.assess_extraction_trajectory(history)
        assert metrics["improvement"] > 0

    def test_generate_curriculum(self):
        """Test curriculum generation."""
        model = WorldStateModel()
        docs = [
            MockDocument("d1", "1"),
            MockDocument("d2", "2")
        ]
        
        # Mock _compute_learning_value
        model._compute_learning_value = Mock(side_effect=[0.2, 0.8])
        
        ordered = model.generate_curriculum(docs)
        
        assert ordered[0].doc_id == "2"
        assert ordered[1].doc_id == "1"
