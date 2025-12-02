"""Bradford Hill criteria preparation for Module 05.

This module implements preparation of causal candidates for Phase 3 validation:
- Transforms CausalCandidate into BradfordHillCriteria structure
- Validates temporality (cause precedes effect)
- Prepares structure for Phase 3 interactive validation

Implementation follows production plan:
docs/phase-1/entity-relationship-extraction-production-plan/05-causal-structure.md

Phase 1 Scope (this implementation):
- Temporality: Validated and populated
- Other criteria: Structure prepared, fields nullable

Phase 3 Scope (future):
- Interactive validation with user
- LLM-assisted causal reasoning
- Integration with PKG for evidence gathering

Bradford Hill Criteria (1965):
1. Temporality (required) - Does cause precede effect?
2. Strength - How strong is the association?
3. Dose-response - More cause → more effect?
4. Consistency - Replicable across contexts?
5. Plausibility - Mechanistic explanation exists?
6. Coherence - Fits existing knowledge?
7. Experiment - Can we test it?
8. Analogy - Similar to known causal patterns?
9. Specificity - Specific cause → specific effect?

Reference:
Hill, A. B. (1965). "The Environment and Disease: Association or Causation?"
Proceedings of the Royal Society of Medicine, 58(5), 295-300.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from futurnal.extraction.causal.models import (
    BradfordHillCriteria,
    CausalCandidate,
)


class BradfordHillPreparation:
    """Prepare causal candidates for Phase 3 Bradford Hill validation.
    
    This class transforms CausalCandidate objects into BradfordHillCriteria
    structures ready for Phase 3 interactive validation.
    
    Phase 1 Responsibilities:
    - Validate and populate temporality (the only REQUIRED criterion)
    - Prepare structure with nullable fields for other criteria
    - Ensure data readiness for Phase 3 validation workflow
    
    Phase 3 will populate the remaining criteria through:
    - Interactive user validation
    - LLM-assisted causal reasoning
    - PKG evidence gathering
    - Statistical analysis (where applicable)
    
    Important:
    - Temporality is the ONLY criterion validated in Phase 1
    - All other criteria are prepared but left as None
    - This follows the production plan scope separation
    """
    
    def prepare_for_validation(
        self,
        candidate: CausalCandidate,
        pkg_context: Optional[Dict[str, Any]] = None
    ) -> BradfordHillCriteria:
        """Prepare BradfordHillCriteria from CausalCandidate.
        
        Phase 1 Implementation:
        - Temporality: Populated from candidate.temporal_ordering_valid
        - All other criteria: None (Phase 3 will populate)
        
        Args:
            candidate: Causal candidate to prepare
            pkg_context: Optional PKG context for future extensibility
                        (not used in Phase 1, reserved for Phase 3)
            
        Returns:
            BradfordHillCriteria with temporality validated
            
        Example:
            >>> prep = BradfordHillPreparation()
            >>> candidate = CausalCandidate(
            ...     temporal_ordering_valid=True,
            ...     temporality_satisfied=True,
            ...     ...
            ... )
            >>> criteria = prep.prepare_for_validation(candidate)
            >>> assert criteria.temporality is True
            >>> assert criteria.strength is None  # Phase 3
        """
        # Validate temporality from candidate
        # This is the ONLY required criterion in Phase 1
        temporality = candidate.temporal_ordering_valid and candidate.temporality_satisfied
        
        # Create Bradford Hill criteria structure
        # Phase 1: Only temporality is populated
        # Phase 3: Will populate other criteria through validation workflow
        criteria = BradfordHillCriteria(
            temporality=temporality,
            # All other criteria prepared but null (Phase 3 scope)
            strength=None,
            dose_response=None,
            consistency=None,
            plausibility=None,
            coherence=None,
            experiment_possible=None,
            analogy=None,
            specificity=None
        )
        
        return criteria
    
    def validate_temporality(self, candidate: CausalCandidate) -> bool:
        """Validate temporality criterion explicitly.
        
        Helper method to check if temporality is satisfied.
        
        Args:
            candidate: Causal candidate to validate
            
        Returns:
            True if cause precedes effect (temporality satisfied)
        """
        return candidate.temporal_ordering_valid and candidate.temporality_satisfied
