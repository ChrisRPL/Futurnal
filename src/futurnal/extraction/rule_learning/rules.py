"""
Extraction Rules - LLM-enhanced rule generation and validation.

Implements the core LeSR approach:
1. Pattern-based rule proposal
2. LLM-powered rule refinement
3. Validation against ground truth
4. Continuous rule evolution
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

from .patterns import SubgraphPattern, PatternMatch

logger = logging.getLogger(__name__)


@dataclass
class RuleCondition:
    """A condition in an extraction rule."""
    variable: str
    property_name: str
    operator: str  # "equals", "contains", "matches", "exists"
    value: Optional[Any] = None


@dataclass
class RuleConclusion:
    """The conclusion (action) of an extraction rule."""
    action: str  # "create_relation", "create_entity", "set_property"
    head_variable: Optional[str] = None
    tail_variable: Optional[str] = None
    relation_type: Optional[str] = None
    entity_type: Optional[str] = None
    property_name: Optional[str] = None
    property_value: Optional[str] = None


@dataclass
class ExtractionRule:
    """
    An extraction rule for knowledge base completion.

    Format: IF pattern THEN conclusion
    Example: IF Person-WORKS_AT->Organization AND Organization-LOCATED_IN->City
            THEN Person-LIVES_IN->City (with confidence 0.7)
    """
    id: str
    name: str
    description: str

    # Rule structure
    pattern: SubgraphPattern
    conditions: List[RuleCondition] = field(default_factory=list)
    conclusion: RuleConclusion = field(default_factory=RuleConclusion)

    # Rule quality metrics
    support: int = 0  # Number of positive examples
    confidence: float = 0.0  # Precision on validation set
    coverage: float = 0.0  # Recall on validation set
    lift: float = 1.0  # Improvement over random

    # Metadata
    source: str = "mined"  # "mined", "llm_proposed", "user_defined"
    created_at: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None
    is_active: bool = True

    # Validation history
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def to_cypher_create(self) -> str:
        """Generate Cypher to apply this rule."""
        if self.conclusion.action == "create_relation":
            return f"""
            {self.pattern.to_cypher()}
            WHERE NOT ({self.conclusion.head_variable})-[:{self.conclusion.relation_type}]->({self.conclusion.tail_variable})
            CREATE ({self.conclusion.head_variable})-[r:{self.conclusion.relation_type} {{
                confidence: {self.confidence},
                rule_id: '{self.id}',
                inferred: true
            }}]->({self.conclusion.tail_variable})
            RETURN count(r) as created
            """
        return ""

    def to_natural_language(self) -> str:
        """Convert rule to natural language description."""
        pattern_desc = self._describe_pattern()
        conclusion_desc = self._describe_conclusion()
        return f"IF {pattern_desc} THEN {conclusion_desc}"

    def _describe_pattern(self) -> str:
        """Describe pattern in natural language."""
        parts = []
        for edge in self.pattern.edges:
            src_type = self.pattern.nodes[edge.source].node_type
            tgt_type = self.pattern.nodes[edge.target].node_type
            parts.append(f"{src_type} {edge.relation_type} {tgt_type}")
        return " AND ".join(parts)

    def _describe_conclusion(self) -> str:
        """Describe conclusion in natural language."""
        if self.conclusion.action == "create_relation":
            return f"create {self.conclusion.relation_type} relation"
        elif self.conclusion.action == "create_entity":
            return f"create {self.conclusion.entity_type} entity"
        elif self.conclusion.action == "set_property":
            return f"set {self.conclusion.property_name}"
        return "unknown action"

    def get_signature(self) -> str:
        """Get unique signature for deduplication."""
        sig = f"{self.pattern.get_signature()}|{self.conclusion.action}"
        if self.conclusion.relation_type:
            sig += f"|{self.conclusion.relation_type}"
        return hashlib.md5(sig.encode()).hexdigest()


class RuleProposer:
    """
    Proposes extraction rules using LLM-enhanced reasoning.

    Combines:
    1. Pattern mining from existing graph
    2. LLM-based rule generation
    3. Semantic validation
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        max_rules_per_pattern: int = 3
    ):
        self.llm_client = llm_client
        self.max_rules_per_pattern = max_rules_per_pattern

    async def propose_rules(
        self,
        patterns: List[SubgraphPattern],
        existing_relations: List[str],
        context: Optional[str] = None
    ) -> List[ExtractionRule]:
        """
        Propose extraction rules based on patterns.

        Args:
            patterns: Frequent subgraph patterns
            existing_relations: Known relation types in the graph
            context: Optional domain context for better proposals

        Returns:
            List of proposed extraction rules
        """
        rules = []

        for pattern in patterns:
            if pattern.support < 2:
                continue

            # Generate rule proposals using LLM
            proposed = await self._propose_for_pattern(
                pattern, existing_relations, context
            )
            rules.extend(proposed)

        # Deduplicate
        seen = set()
        unique_rules = []
        for rule in rules:
            sig = rule.get_signature()
            if sig not in seen:
                seen.add(sig)
                unique_rules.append(rule)

        logger.info(f"Proposed {len(unique_rules)} unique rules from {len(patterns)} patterns")
        return unique_rules

    async def _propose_for_pattern(
        self,
        pattern: SubgraphPattern,
        existing_relations: List[str],
        context: Optional[str]
    ) -> List[ExtractionRule]:
        """Propose rules for a single pattern."""
        if self.llm_client:
            return await self._propose_with_llm(pattern, existing_relations, context)
        else:
            return self._propose_heuristic(pattern, existing_relations)

    async def _propose_with_llm(
        self,
        pattern: SubgraphPattern,
        existing_relations: List[str],
        context: Optional[str]
    ) -> List[ExtractionRule]:
        """Use LLM to propose rules."""
        prompt = self._build_proposal_prompt(pattern, existing_relations, context)

        try:
            if hasattr(self.llm_client, "generate"):
                response = await self.llm_client.generate(prompt)
            elif hasattr(self.llm_client, "chat"):
                response = await self.llm_client.chat([{"role": "user", "content": prompt}])
                response = response.get("content", "")
            else:
                return self._propose_heuristic(pattern, existing_relations)

            return self._parse_llm_proposals(response, pattern)

        except Exception as e:
            logger.warning(f"LLM rule proposal failed: {e}")
            return self._propose_heuristic(pattern, existing_relations)

    def _build_proposal_prompt(
        self,
        pattern: SubgraphPattern,
        existing_relations: List[str],
        context: Optional[str]
    ) -> str:
        """Build prompt for LLM rule proposal."""
        pattern_desc = []
        for edge in pattern.edges:
            src = pattern.nodes[edge.source]
            tgt = pattern.nodes[edge.target]
            pattern_desc.append(f"({src.node_type})-[{edge.relation_type}]->({tgt.node_type})")

        prompt = f"""You are a knowledge graph expert. Given a frequent pattern in a knowledge graph, propose extraction rules that could infer new relationships.

## Pattern (appears {pattern.support} times):
{' AND '.join(pattern_desc)}

## Available Relationship Types:
{', '.join(existing_relations[:30])}

"""
        if context:
            prompt += f"## Domain Context:\n{context}\n\n"

        prompt += """## Task:
Propose 1-3 extraction rules that could infer NEW relationships based on this pattern.

For each rule, provide:
1. RELATION: The new relationship type to create
2. HEAD: Which pattern variable should be the head (source)
3. TAIL: Which pattern variable should be the tail (target)
4. CONFIDENCE: Estimated confidence (0.0-1.0)
5. REASONING: Brief explanation

## Output Format (JSON):
```json
[
  {
    "relation": "LIVES_IN",
    "head": "h",
    "tail": "t",
    "confidence": 0.7,
    "reasoning": "People who work at organizations often live in the same city"
  }
]
```
"""
        return prompt

    def _parse_llm_proposals(
        self,
        response: str,
        pattern: SubgraphPattern
    ) -> List[ExtractionRule]:
        """Parse LLM response into rules."""
        rules = []

        # Extract JSON from response
        try:
            # Find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                proposals = json.loads(json_str)

                for i, prop in enumerate(proposals):
                    rule = ExtractionRule(
                        id=f"llm_{pattern.id}_{i}",
                        name=f"Rule: {prop.get('relation', 'unknown')}",
                        description=prop.get("reasoning", "LLM-generated rule"),
                        pattern=pattern,
                        conclusion=RuleConclusion(
                            action="create_relation",
                            head_variable=prop.get("head", "h"),
                            tail_variable=prop.get("tail", "t"),
                            relation_type=prop.get("relation", "RELATED_TO")
                        ),
                        confidence=float(prop.get("confidence", 0.5)),
                        support=pattern.support,
                        source="llm_proposed"
                    )
                    rules.append(rule)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM proposals: {e}")

        return rules

    def _propose_heuristic(
        self,
        pattern: SubgraphPattern,
        existing_relations: List[str]
    ) -> List[ExtractionRule]:
        """Propose rules using simple heuristics."""
        rules = []

        # Heuristic: If A->B and B->C, propose A->C
        if len(pattern.edges) >= 2:
            first_edge = pattern.edges[0]
            last_edge = pattern.edges[-1]

            # Check if we can create a transitive relation
            if first_edge.target == last_edge.source:
                head_type = pattern.nodes[first_edge.source].node_type
                tail_type = pattern.nodes[last_edge.target].node_type

                # Propose a combined relation
                combined_rel = f"{first_edge.relation_type}_{last_edge.relation_type}"

                rule = ExtractionRule(
                    id=f"heuristic_{pattern.id}_transitive",
                    name=f"Transitive: {combined_rel}",
                    description=f"If {head_type} {first_edge.relation_type} and then {last_edge.relation_type}, infer direct relation",
                    pattern=pattern,
                    conclusion=RuleConclusion(
                        action="create_relation",
                        head_variable=first_edge.source,
                        tail_variable=last_edge.target,
                        relation_type=combined_rel
                    ),
                    confidence=0.5,
                    support=pattern.support,
                    source="mined"
                )
                rules.append(rule)

        return rules


class RuleValidator:
    """
    Validates extraction rules against the knowledge graph.

    Computes precision, recall, and other quality metrics.
    """

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def validate(
        self,
        rule: ExtractionRule,
        sample_size: int = 100
    ) -> Dict[str, float]:
        """
        Validate a rule and compute quality metrics.

        Args:
            rule: Rule to validate
            sample_size: Number of samples to evaluate

        Returns:
            Dictionary of metrics
        """
        # Get pattern matches
        matches = self._get_pattern_matches(rule.pattern, sample_size)

        if not matches:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

        # Check which matches have the conclusion relation
        tp = 0  # True positives: rule predicts and relation exists
        fp = 0  # False positives: rule predicts but relation doesn't exist
        fn = 0  # False negatives: relation exists but different pattern

        for match in matches:
            head_id = match.get(rule.conclusion.head_variable)
            tail_id = match.get(rule.conclusion.tail_variable)

            if head_id and tail_id:
                has_relation = self._check_relation(
                    head_id, rule.conclusion.relation_type, tail_id
                )
                if has_relation:
                    tp += 1
                else:
                    fp += 1

        # Estimate false negatives (relations that exist without the pattern)
        fn = self._estimate_false_negatives(rule, sample_size)

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Update rule
        rule.true_positives = tp
        rule.false_positives = fp
        rule.false_negatives = fn
        rule.confidence = precision
        rule.coverage = recall
        rule.last_validated = datetime.now()

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fp,
            "true_positives": tp,
            "false_positives": fp
        }

    def _get_pattern_matches(
        self,
        pattern: SubgraphPattern,
        limit: int
    ) -> List[Dict[str, str]]:
        """Get matches for a pattern."""
        cypher = pattern.to_cypher()
        return_vars = list(pattern.nodes.keys())
        cypher += f" RETURN {', '.join([f'{v}.id as {v}' for v in return_vars])}"
        cypher += f" LIMIT {limit}"

        matches = []
        with self.driver.session() as session:
            result = session.run(cypher)
            for record in result:
                matches.append({v: record[v] for v in return_vars})

        return matches

    def _check_relation(
        self,
        head_id: str,
        relation_type: str,
        tail_id: str
    ) -> bool:
        """Check if a relation exists."""
        query = f"""
        MATCH (h {{id: $head_id}})-[r:{relation_type}]->(t {{id: $tail_id}})
        RETURN count(r) > 0 as exists
        """
        with self.driver.session() as session:
            result = session.run(query, head_id=head_id, tail_id=tail_id)
            record = result.single()
            return record["exists"] if record else False

    def _estimate_false_negatives(
        self,
        rule: ExtractionRule,
        sample_size: int
    ) -> int:
        """Estimate false negatives for a rule."""
        # Count relations that exist but don't match the pattern
        if not rule.conclusion.relation_type:
            return 0

        query = f"""
        MATCH (h)-[r:{rule.conclusion.relation_type}]->(t)
        RETURN count(r) as total
        """
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            total_relations = record["total"] if record else 0

        # Rough estimate: fn = total - tp
        return max(0, total_relations - rule.true_positives)


@dataclass
class RuleStore:
    """
    Persistent storage for extraction rules.

    Supports:
    - Rule persistence to file
    - Rule versioning
    - Active/inactive management
    """
    rules: Dict[str, ExtractionRule] = field(default_factory=dict)
    storage_path: Optional[Path] = None

    def add(self, rule: ExtractionRule) -> None:
        """Add a rule to the store."""
        self.rules[rule.id] = rule

    def get(self, rule_id: str) -> Optional[ExtractionRule]:
        """Get a rule by ID."""
        return self.rules.get(rule_id)

    def get_active_rules(self, min_confidence: float = 0.3) -> List[ExtractionRule]:
        """Get all active rules above confidence threshold."""
        return [
            r for r in self.rules.values()
            if r.is_active and r.confidence >= min_confidence
        ]

    def get_rules_for_pattern(self, pattern_signature: str) -> List[ExtractionRule]:
        """Get rules matching a pattern signature."""
        return [
            r for r in self.rules.values()
            if r.pattern.get_signature() == pattern_signature
        ]

    def deactivate(self, rule_id: str) -> None:
        """Deactivate a rule."""
        if rule_id in self.rules:
            self.rules[rule_id].is_active = False

    def save(self, path: Optional[Path] = None) -> None:
        """Save rules to file."""
        path = path or self.storage_path
        if not path:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for rule in self.rules.values():
            data.append({
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "pattern_signature": rule.pattern.get_signature(),
                "conclusion": {
                    "action": rule.conclusion.action,
                    "head_variable": rule.conclusion.head_variable,
                    "tail_variable": rule.conclusion.tail_variable,
                    "relation_type": rule.conclusion.relation_type
                },
                "confidence": rule.confidence,
                "support": rule.support,
                "source": rule.source,
                "is_active": rule.is_active,
                "created_at": rule.created_at.isoformat(),
                "true_positives": rule.true_positives,
                "false_positives": rule.false_positives
            })

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(data)} rules to {path}")

    def load(self, path: Optional[Path] = None) -> None:
        """Load rules from file."""
        path = path or self.storage_path
        if not path or not path.exists():
            return

        with open(path) as f:
            data = json.load(f)

        for item in data:
            # Reconstruct rule (simplified - pattern needs to be rebuilt)
            rule = ExtractionRule(
                id=item["id"],
                name=item["name"],
                description=item["description"],
                pattern=SubgraphPattern(id=item.get("pattern_signature", "")),
                conclusion=RuleConclusion(**item["conclusion"]),
                confidence=item["confidence"],
                support=item["support"],
                source=item["source"],
                is_active=item["is_active"],
                created_at=datetime.fromisoformat(item["created_at"]),
                true_positives=item.get("true_positives", 0),
                false_positives=item.get("false_positives", 0)
            )
            self.rules[rule.id] = rule

        logger.info(f"Loaded {len(self.rules)} rules from {path}")
