"""Causal structure data models for Module 05.

This module defines data structures for causal extraction:
- Event type classification (seed types, extensible via schema evolution)
- Causal relationship types (semantic differentiation)
- Causal candidate structure (Phase 1 preparation, Phase 3 validation)
- Bradford Hill criteria (8 criteria for causal inference)

Implementation follows production plan:
docs/phase-1/entity-relationship-extraction-production-plan/05-causal-structure.md

Option B Compliance:
- No hardcoded event types (seed schema only, discoverable)
- Temporal-first design (causality requires temporal ordering)
- Phase 1 scope: structure preparation, NOT validation
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event type classification for causal analysis.
    
    Seed event types for initial classification. Schema evolution
    will discover additional event types from user data.
    
    Option B Compliance:
    - Seed schema only (not hardcoded forever)
    - Discovery mechanism via schema evolution
    - Extensible for user-specific event patterns
    """
    
    MEETING = "meeting"                    # Meetings, discussions, gatherings
    DECISION = "decision"                  # Decisions, choices, determinations
    PUBLICATION = "publication"            # Publications, releases, announcements
    COMMUNICATION = "communication"        # Communications, messages, exchanges
    ACTION = "action"                      # Actions, activities, tasks
    STATE_CHANGE = "state_change"         # State changes, transitions, updates
    UNKNOWN = "unknown"                    # Unclassified events


class CausalRelationshipType(str, Enum):
    """Semantic types of causal relationships.
    
    Differentiates causal mechanisms for Phase 3 validation:
    - Direct causation (CAUSES, TRIGGERS)
    - Indirect causation (LEADS_TO, CONTRIBUTES_TO)
    - Enabling/blocking (ENABLES, PREVENTS)
    
    Each type supports different Bradford Hill validation strategies.
    """
    
    CAUSES = "causes"                      # Strong causal claim (A causes B)
    ENABLES = "enables"                    # Prerequisite relationship (A enables B)
    PREVENTS = "prevents"                  # Blocking relationship (A prevents B)
    TRIGGERS = "triggers"                  # Immediate causation (A triggers B)
    LEADS_TO = "leads_to"                 # Indirect causation (A leads to B)
    CONTRIBUTES_TO = "contributes_to"     # Partial causation (A contributes to B)


class CausalCandidate(BaseModel):
    """Event-event relationship flagged for Phase 3 causal validation.
    
    Phase 1 Scope (this implementation):
    - Extract event pairs with temporal ordering
    - Identify causal language in text
    - Flag candidates with confidence scores
    - Prepare Bradford Hill criteria structure
    
    Phase 3 Scope (future):
    - Validate Bradford Hill criteria
    - Interactive hypothesis testing
    - Counterfactual reasoning
    - Causal graph construction
    
    Success Metrics:
    - Causal candidates flagged with confidence >0.6
    - Temporal ordering validated for all candidates (100%)
    - Bradford Hill criteria structure prepared
    """
    
    id: str = Field(..., description="Unique identifier for this candidate")
    cause_event_id: str = Field(..., description="ID of the cause event")
    effect_event_id: str = Field(..., description="ID of the effect event")
    relationship_type: CausalRelationshipType = Field(
        ...,
        description="Type of causal relationship"
    )
    
    # Temporal validation (required for causality)
    temporal_gap: timedelta = Field(
        ...,
        description="Time between cause and effect events"
    )
    temporal_ordering_valid: bool = Field(
        ...,
        description="True if cause precedes effect (temporal precedence)"
    )
    
    # Evidence and confidence
    causal_evidence: str = Field(
        ...,
        description="Text evidence supporting the causal relationship"
    )
    causal_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in causal relationship (0.0-1.0)"
    )
    
    # Bradford Hill criteria (Phase 1: structure only, Phase 3: validation)
    temporality_satisfied: bool = Field(
        ...,
        description="Bradford Hill criterion 1: Does cause precede effect?"
    )
    strength: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Bradford Hill criterion 2: Strength of association (Phase 3)"
    )
    consistency: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Bradford Hill criterion 4: Replicability across contexts (Phase 3)"
    )
    plausibility: Optional[str] = Field(
        None,
        description="Bradford Hill criterion 5: Mechanistic explanation (Phase 3)"
    )
    
    # Phase 3 validation status
    is_validated: bool = Field(
        default=False,
        description="True if validated in Phase 3, False during Phase 1 prep"
    )
    validation_method: Optional[str] = Field(
        None,
        description="Method used for Phase 3 validation"
    )
    
    # Provenance
    source_document: str = Field(..., description="Document ID where relationship was found")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Extraction timestamp"
    )
    
    model_config = {
        "json_encoders": {
            timedelta: lambda v: v.total_seconds(),
            datetime: lambda v: v.isoformat(),
        }
    }


class BradfordHillCriteria(BaseModel):
    """Bradford Hill criteria for causal inference validation.
    
    The 9 Bradford Hill criteria (1965) for establishing causation:
    1. Temporality (required) - Does cause precede effect?
    2. Strength - How strong is the association?
    3. Dose-response - More cause → more effect?
    4. Consistency - Replicable across contexts?
    5. Plausibility - Mechanistic explanation exists?
    6. Coherence - Fits existing knowledge?
    7. Experiment - Can we test it?
    8. Analogy - Similar to known causal patterns?
    9. Specificity - Specific cause → specific effect?
    
    Phase 1 Scope (this implementation):
    - Temporality: Validated (we have timestamps)
    - Other criteria: Structure prepared, fields nullable
    
    Phase 3 Scope (future):
    - Interactive validation with user
    - LLM-assisted causal reasoning
    - Integration with PKG for evidence gathering
    
    Reference:
    Hill, A. B. (1965). "The Environment and Disease: Association or Causation?"
    Proceedings of the Royal Society of Medicine, 58(5), 295-300.
    """
    
    # Criterion 1: Temporality (REQUIRED for causation)
    temporality: bool = Field(
        ...,
        description="Does cause precede effect? (Required criterion)"
    )
    
    # Criterion 2: Strength of association
    strength: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="How strong is the relationship? (Phase 3)"
    )
    
    # Criterion 3: Dose-response relationship
    dose_response: Optional[bool] = Field(
        None,
        description="Does more cause lead to more effect? (Phase 3)"
    )
    
    # Criterion 4: Consistency
    consistency: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Is the relationship replicable across contexts? (Phase 3)"
    )
    
    # Criterion 5: Plausibility
    plausibility: Optional[str] = Field(
        None,
        description="What is the mechanistic explanation? (Phase 3)"
    )
    
    # Criterion 6: Coherence
    coherence: Optional[bool] = Field(
        None,
        description="Does this fit with existing knowledge? (Phase 3)"
    )
    
    # Criterion 7: Experiment
    experiment_possible: Optional[bool] = Field(
        None,
        description="Can this relationship be tested experimentally? (Phase 3)"
    )
    
    # Criterion 8: Analogy
    analogy: Optional[str] = Field(
        None,
        description="Similar to known causal patterns? (Phase 3)"
    )
    
    # Criterion 9: Specificity
    specificity: Optional[bool] = Field(
        None,
        description="Specific cause → specific effect? (Phase 3)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "temporality": True,
                "strength": None,  # Phase 3
                "dose_response": None,  # Phase 3
                "consistency": None,  # Phase 3
                "plausibility": None,  # Phase 3
                "coherence": None,  # Phase 3
                "experiment_possible": None,  # Phase 3
                "analogy": None,  # Phase 3
                "specificity": None,  # Phase 3
            }
        }
    }
