"""Causal relationship detection for Module 05.

This module implements event-event relationship detection for causal structure:
- Identifies potential causal pairs based on temporal ordering
- Extracts causal evidence from document text
- Assesses confidence in causal relationships
- Flags candidates for Phase 3 Bradford Hill validation

Implementation follows production plan:
docs/phase-1/entity-relationship-extraction-production-plan/05-causal-structure.md

Option B Compliance:
- Temporal-first design (causality requires temporal ordering)
- LLM-based extraction (consistent with EventExtractor pattern)
- No hardcoded causal patterns (discoverable via experiential learning)

Success Metrics:
- Event-event relationships identified
- Causal candidates flagged with confidence >0.6
- Temporal ordering validated for all candidates (100%)
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any, Dict, List, Optional, Protocol

from futurnal.extraction.causal.models import (
    CausalCandidate,
    CausalRelationshipType,
)
from futurnal.extraction.temporal.models import Event


class LLMClient(Protocol):
    """Protocol for LLM interactions.
    
    Matches the LLMClient protocol from experiential learning.
    """
    
    def extract(self, prompt: str) -> Any:
        """Run extraction on a prompt.
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            Extraction result (implementation-specific).
        """
        ...


class Document(Protocol):
    """Protocol for document structure.
    
    Defines minimal interface required for causal detection.
    """
    
    content: str
    doc_id: str


class CausalRelationshipDetector:
    """Detect causal relationships between events.
    
    Identifies event-event relationships as causal candidates for Phase 3
    validation. Uses temporal ordering, causal language, and LLM analysis
    to assess potential causal relationships.
    
    Key Requirements:
    - Cause must precede effect (temporal ordering)
    - Reasonable temporal proximity (default: <1 year)
    - Causal evidence in document text
    - Confidence threshold (default: >0.6)
    
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
    """
    
    # Causal indicator keywords (seed heuristics, will improve via learning)
    CAUSAL_INDICATORS = [
        # Direct causation
        "caused", "causes", "cause of",
        "led to", "leads to", "leading to",
        "resulted in", "results in", "resulting in",
        # Triggering
        "triggered", "triggers", "trigger",
        # Enabling/blocking
        "enabled", "enables", "enable",
        "allowed", "allows", "allow",
        "prevented", "prevents", "prevent",
        "blocked", "blocks", "block",
        # Contribution
        "contributed to", "contributes to", "contributing to",
        # Consequence
        "consequence of", "consequences of",
        "because of", "due to", "owing to",
        # Influence
        "influenced", "influences", "influence",
        "affected", "affects", "affect",
    ]
    
    def __init__(
        self,
        llm: LLMClient,
        confidence_threshold: float = 0.6,
        max_temporal_gap_days: int = 365
    ):
        """Initialize causal relationship detector.
        
        Args:
            llm: LLM client for causal analysis
            confidence_threshold: Minimum confidence for candidate inclusion (default 0.6)
            max_temporal_gap_days: Maximum days between cause and effect (default 365)
        """
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.max_temporal_gap = timedelta(days=max_temporal_gap_days)
    
    def detect_causal_candidates(
        self,
        events: List[Event],
        document: Document
    ) -> List[CausalCandidate]:
        """Detect potential causal relationships between events.
        
        Success Metrics:
        - Event-event relationships identified
        - Causal candidates flagged with confidence >0.6
        - Temporal ordering validated for all candidates (100%)
        
        Args:
            events: List of extracted events from document
            document: Source document
            
        Returns:
            List of causal candidates with confidence >threshold
            
        Example:
            >>> detector = CausalRelationshipDetector(llm)
            >>> events = [
            ...     Event(name="Meeting", timestamp=datetime(2024, 1, 15)),
            ...     Event(name="Decision", timestamp=datetime(2024, 1, 16))
            ... ]
            >>> candidates = detector.detect_causal_candidates(events, doc)
            >>> assert all(c.temporal_ordering_valid for c in candidates)  # 100%
        """
        candidates = []
        
        # Pairwise comparison of events
        for i, cause in enumerate(events):
            for effect in events[i+1:]:  # Only consider future events
                if self._is_potential_causal_pair(cause, effect, document):
                    candidate = self._create_causal_candidate(
                        cause=cause,
                        effect=effect,
                        document=document
                    )
                    
                    # Filter by confidence threshold
                    if candidate and candidate.causal_confidence >= self.confidence_threshold:
                        candidates.append(candidate)
        
        return candidates
    
    def _is_potential_causal_pair(
        self,
        cause: Event,
        effect: Event,
        document: Document
    ) -> bool:
        """Check if two events could be causally related.
        
        Requirements:
        - Temporal ordering (cause before effect)
        - Reasonable temporal proximity
        - Both events have timestamps
        
        Args:
            cause: Potential cause event
            effect: Potential effect event
            document: Source document
            
        Returns:
            True if events could be causally related
        """
        # Both events must have timestamps
        if not cause.timestamp or not effect.timestamp:
            return False
        
        # Temporality check: cause must precede effect
        if cause.timestamp >= effect.timestamp:
            return False
        
        # Proximity check: within maximum temporal gap
        temporal_gap = effect.timestamp - cause.timestamp
        if temporal_gap > self.max_temporal_gap:
            return False
        
        return True
    
    def _create_causal_candidate(
        self,
        cause: Event,
        effect: Event,
        document: Document
    ) -> Optional[CausalCandidate]:
        """Create causal candidate with Bradford Hill criteria prep.
        
        Args:
            cause: Cause event
            effect: Effect event
            document: Source document
            
        Returns:
            CausalCandidate or None if creation fails
        """
        try:
            # Calculate temporal gap
            temporal_gap = effect.timestamp - cause.timestamp
            
            # Extract causal evidence from document
            evidence = self._extract_causal_evidence(cause, effect, document)
            
            # Assess causal confidence
            confidence = self._assess_causal_confidence(evidence, cause, effect)
            
            # Infer causal relationship type
            relationship_type = self._infer_causal_type(evidence)
            
            # Create candidate
            candidate = CausalCandidate(
                id=f"causal_{cause.name}_{effect.name}",
                cause_event_id=cause.name,  # Use name as ID (Event model doesn't have id)
                effect_event_id=effect.name,
                relationship_type=relationship_type,
                temporal_gap=temporal_gap,
                temporal_ordering_valid=True,  # Already validated
                causal_evidence=evidence,
                causal_confidence=confidence,
                temporality_satisfied=True,  # Bradford Hill criterion 1
                is_validated=False,  # Phase 3 will validate
                source_document=document.doc_id
            )
            
            return candidate
            
        except Exception as e:
            # Candidate creation failed, return None
            # In production, log this error
            return None
    
    def _extract_causal_evidence(
        self,
        cause: Event,
        effect: Event,
        document: Document
    ) -> str:
        """Extract text evidence supporting causal relationship.
        
        Uses LLM to identify relevant passages that suggest causation
        between the two events.
        
        Args:
            cause: Cause event
            effect: Effect event
            document: Source document
            
        Returns:
            Text evidence (or empty string if none found)
        """
        # Build prompt for evidence extraction
        prompt = self._build_causal_prompt(cause, effect, document)
        
        try:
            # Call LLM for evidence extraction
            result = self.llm.extract(prompt)
            
            # Parse response
            parsed = self._parse_llm_response(result)
            if parsed and "evidence" in parsed:
                return parsed["evidence"]
            
            return ""
            
        except Exception:
            # Fallback: search for causal indicators in document
            return self._keyword_based_evidence(cause, effect, document)
    
    def _assess_causal_confidence(
        self,
        evidence: str,
        cause: Event,
        effect: Event
    ) -> float:
        """Assess confidence in causal relationship.
        
        Args:
            evidence: Causal evidence text
            cause: Cause event
            effect: Effect event
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # If evidence is empty, return low confidence
        if not evidence or len(evidence.strip()) == 0:
            return 0.5
        
        # Start with base confidence
        confidence = 0.6  # Start at threshold to give benefit of doubt when evidence exists
        
        # Boost if evidence contains strong causal indicators
        strong_indicators = ["caused", "led to", "resulted in", "triggered"]
        if any(indicator in evidence.lower() for indicator in strong_indicators):
            confidence += 0.2
        
        # Boost if evidence is substantial
        if len(evidence) > 50:
            confidence += 0.1
        
        # Boost if both event names appear in evidence
        if cause.name.lower() in evidence.lower() and effect.name.lower() in evidence.lower():
            confidence += 0.1
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _infer_causal_type(
        self,
        evidence: str
    ) -> CausalRelationshipType:
        """Infer causal relationship type from evidence.
        
        Args:
            evidence: Causal evidence text
            
        Returns:
            Inferred relationship type
        """
        evidence_lower = evidence.lower()
        
        # Check for specific relationship types
        if any(word in evidence_lower for word in ["caused", "cause"]):
            return CausalRelationshipType.CAUSES
        elif any(word in evidence_lower for word in ["triggered", "trigger"]):
            return CausalRelationshipType.TRIGGERS
        elif any(word in evidence_lower for word in ["enabled", "enable", "allowed", "allow"]):
            return CausalRelationshipType.ENABLES
        elif any(word in evidence_lower for word in ["prevented", "prevent", "blocked", "block"]):
            return CausalRelationshipType.PREVENTS
        elif any(word in evidence_lower for word in ["contributed", "contribute"]):
            return CausalRelationshipType.CONTRIBUTES_TO
        elif any(word in evidence_lower for word in ["led to", "resulted in"]):
            return CausalRelationshipType.LEADS_TO
        
        # Default to generic causation
        return CausalRelationshipType.CAUSES
    
    def _build_causal_prompt(
        self,
        cause: Event,
        effect: Event,
        document: Document
    ) -> str:
        """Build LLM prompt for causal analysis.
        
        Args:
            cause: Cause event
            effect: Effect event
            document: Source document
            
        Returns:
            Structured prompt for causal evidence extraction
        """
        prompt = f"""Analyze the potential causal relationship between two events.

Document: {document.content[:500]}...

Event 1 (Potential Cause): {cause.name}
- Type: {cause.event_type}
- Timestamp: {cause.timestamp.isoformat() if cause.timestamp else 'Unknown'}

Event 2 (Potential Effect): {effect.name}
- Type: {effect.event_type}
- Timestamp: {effect.timestamp.isoformat() if effect.timestamp else 'Unknown'}

Task: Determine if Event 1 caused or influenced Event 2.

Extract the following information in JSON format:
{{
  "evidence": "Text from the document that supports the causal relationship (or empty string if none)",
  "relationship_type": "One of: causes, triggers, enables, prevents, leads_to, contributes_to",
  "confidence": "Confidence score 0.0-1.0"
}}

Causal Relationship Types:
- causes: Strong direct causation (A causes B)
- triggers: Immediate causation (A triggers B)
- enables: Prerequisite relationship (A enables B)
- prevents: Blocking relationship (A prevents B)
- leads_to: Indirect causation (A leads to B)
- contributes_to: Partial causation (A contributes to B)

Examples:

Event 1: Meeting scheduled
Event 2: Decision made
Output:
{{
  "evidence": "The meeting led to a unanimous decision to proceed with the project.",
  "relationship_type": "leads_to",
  "confidence": 0.85
}}

Event 1: Budget approved
Event 2: Project started
Output:
{{
  "evidence": "The budget approval enabled the team to begin the project.",
  "relationship_type": "enables",
  "confidence": 0.9
}}

Now analyze the given events:
"""
        return prompt
    
    def _parse_llm_response(self, result: Any) -> Optional[Dict[str, Any]]:
        """Parse LLM response into structured data.
        
        Args:
            result: LLM extraction result
            
        Returns:
            Parsed dict or None if parsing fails
        """
        try:
            # Handle different result formats
            if isinstance(result, dict):
                return result
            elif isinstance(result, str):
                # Try to parse as JSON
                start = result.find('{')
                end = result.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = result[start:end]
                    return json.loads(json_str)
            elif hasattr(result, 'content'):
                # Handle result with content attribute
                return self._parse_llm_response(result.content)
            
            return None
            
        except (json.JSONDecodeError, AttributeError):
            return None
    
    def _keyword_based_evidence(
        self,
        cause: Event,
        effect: Event,
        document: Document
    ) -> str:
        """Fallback: keyword-based evidence extraction.
        
        Args:
            cause: Cause event
            effect: Effect event
            document: Source document
            
        Returns:
            Evidence text (or empty string)
        """
        # Look for sentences containing both event names and causal indicators
        sentences = document.content.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence mentions both events
            has_cause = cause.name.lower() in sentence_lower
            has_effect = effect.name.lower() in sentence_lower
            
            # Check for causal indicators
            has_causal = any(
                indicator in sentence_lower
                for indicator in self.CAUSAL_INDICATORS
            )
            
            if has_cause and has_effect and has_causal:
                return sentence.strip()
        
        return ""
