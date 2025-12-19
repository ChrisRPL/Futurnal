"""
Rule Learner - Main orchestrator for LeSR-style rule learning.

Coordinates:
- Pattern extraction
- Rule proposal
- Validation
- Continuous improvement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from .patterns import PatternExtractor, SubgraphPattern
from .rules import ExtractionRule, RuleProposer, RuleValidator, RuleStore

logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """Configuration for rule learning."""
    # Pattern mining
    min_pattern_support: int = 3
    max_pattern_size: int = 4
    max_patterns: int = 500

    # Rule generation
    max_rules_per_pattern: int = 2
    min_rule_confidence: float = 0.3
    min_rule_support: int = 5

    # Validation
    validation_sample_size: int = 100
    revalidation_interval_hours: int = 24

    # Storage
    rules_path: Optional[Path] = None

    # Learning schedule
    learning_interval_hours: int = 6
    max_new_rules_per_cycle: int = 50


@dataclass
class LearningMetrics:
    """Metrics from a learning cycle."""
    cycle_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Pattern metrics
    patterns_extracted: int = 0
    patterns_above_support: int = 0

    # Rule metrics
    rules_proposed: int = 0
    rules_validated: int = 0
    rules_accepted: int = 0
    rules_rejected: int = 0

    # Quality metrics
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1: float = 0.0

    # Application metrics
    facts_inferred: int = 0


class RuleLearner:
    """
    Main orchestrator for LeSR-style rule learning.

    Implements continuous learning loop:
    1. Extract patterns from knowledge graph
    2. Propose rules using LLM
    3. Validate rules
    4. Apply high-confidence rules
    5. Track and improve over time
    """

    def __init__(
        self,
        neo4j_driver,
        config: Optional[LearningConfig] = None,
        llm_client: Optional[Any] = None
    ):
        self.driver = neo4j_driver
        self.config = config or LearningConfig()
        self.llm_client = llm_client

        # Components
        self.pattern_extractor = PatternExtractor(
            min_support=self.config.min_pattern_support,
            max_pattern_size=self.config.max_pattern_size,
            max_patterns=self.config.max_patterns
        )
        self.rule_proposer = RuleProposer(
            llm_client=llm_client,
            max_rules_per_pattern=self.config.max_rules_per_pattern
        )
        self.rule_validator = RuleValidator(neo4j_driver)
        self.rule_store = RuleStore(storage_path=self.config.rules_path)

        # Load existing rules
        if self.config.rules_path:
            self.rule_store.load()

        # Learning history
        self.metrics_history: List[LearningMetrics] = []
        self.last_learning_cycle: Optional[datetime] = None

    async def learn(
        self,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> LearningMetrics:
        """
        Run a complete learning cycle.

        Args:
            node_labels: Filter for node types
            relationship_types: Filter for relationship types
            context: Domain context for LLM

        Returns:
            Metrics from the learning cycle
        """
        metrics = LearningMetrics(
            cycle_id=f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            started_at=datetime.now()
        )

        logger.info(f"Starting learning cycle {metrics.cycle_id}")

        # Step 1: Extract patterns
        logger.info("Step 1: Extracting patterns...")
        patterns = self.pattern_extractor.extract_from_neo4j(
            self.driver, node_labels, relationship_types
        )
        metrics.patterns_extracted = len(patterns)
        metrics.patterns_above_support = len([
            p for p in patterns if p.support >= self.config.min_pattern_support
        ])

        # Step 2: Get existing relation types
        existing_relations = self._get_existing_relation_types()

        # Step 3: Propose rules
        logger.info("Step 2: Proposing rules...")
        proposed_rules = await self.rule_proposer.propose_rules(
            patterns, existing_relations, context
        )
        metrics.rules_proposed = len(proposed_rules)

        # Step 4: Validate rules
        logger.info("Step 3: Validating rules...")
        accepted_rules = []
        precision_sum = 0.0
        recall_sum = 0.0

        for rule in proposed_rules[:self.config.max_new_rules_per_cycle]:
            validation = self.rule_validator.validate(
                rule, self.config.validation_sample_size
            )
            metrics.rules_validated += 1

            precision_sum += validation["precision"]
            recall_sum += validation["recall"]

            if (validation["precision"] >= self.config.min_rule_confidence and
                validation["support"] >= self.config.min_rule_support):
                accepted_rules.append(rule)
                self.rule_store.add(rule)
                metrics.rules_accepted += 1
            else:
                metrics.rules_rejected += 1

        # Compute averages
        if metrics.rules_validated > 0:
            metrics.avg_precision = precision_sum / metrics.rules_validated
            metrics.avg_recall = recall_sum / metrics.rules_validated
            metrics.avg_f1 = (
                2 * metrics.avg_precision * metrics.avg_recall /
                (metrics.avg_precision + metrics.avg_recall)
                if (metrics.avg_precision + metrics.avg_recall) > 0 else 0.0
            )

        # Step 5: Apply high-confidence rules
        logger.info("Step 4: Applying accepted rules...")
        for rule in accepted_rules:
            if rule.confidence >= 0.7:  # Only apply very confident rules
                inferred = await self._apply_rule(rule)
                metrics.facts_inferred += inferred

        # Save rules
        if self.config.rules_path:
            self.rule_store.save()

        metrics.completed_at = datetime.now()
        self.metrics_history.append(metrics)
        self.last_learning_cycle = datetime.now()

        logger.info(
            f"Learning cycle completed: {metrics.rules_accepted} rules accepted, "
            f"{metrics.facts_inferred} facts inferred"
        )

        return metrics

    def learn_sync(
        self,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> LearningMetrics:
        """Synchronous version of learn."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.learn(node_labels, relationship_types, context)
        )

    async def revalidate_rules(self) -> Dict[str, Any]:
        """
        Revalidate existing rules and update confidence scores.

        Returns:
            Summary of revalidation results
        """
        results = {
            "total_rules": len(self.rule_store.rules),
            "revalidated": 0,
            "improved": 0,
            "degraded": 0,
            "deactivated": 0
        }

        for rule in list(self.rule_store.rules.values()):
            if not rule.is_active:
                continue

            # Check if revalidation is needed
            if rule.last_validated:
                hours_since = (datetime.now() - rule.last_validated).total_seconds() / 3600
                if hours_since < self.config.revalidation_interval_hours:
                    continue

            old_confidence = rule.confidence
            validation = self.rule_validator.validate(rule, self.config.validation_sample_size)
            results["revalidated"] += 1

            if validation["precision"] > old_confidence + 0.05:
                results["improved"] += 1
            elif validation["precision"] < old_confidence - 0.1:
                results["degraded"] += 1

            # Deactivate rules that dropped below threshold
            if validation["precision"] < self.config.min_rule_confidence:
                rule.is_active = False
                results["deactivated"] += 1

        # Save updated rules
        if self.config.rules_path:
            self.rule_store.save()

        return results

    async def _apply_rule(self, rule: ExtractionRule) -> int:
        """Apply a rule to create new facts."""
        if rule.conclusion.action != "create_relation":
            return 0

        cypher = rule.to_cypher_create()
        if not cypher:
            return 0

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                record = result.single()
                created = record["created"] if record else 0
                logger.info(f"Rule {rule.id} created {created} new relations")
                return created
        except Exception as e:
            logger.warning(f"Failed to apply rule {rule.id}: {e}")
            return 0

    def _get_existing_relation_types(self) -> List[str]:
        """Get all existing relationship types in the graph."""
        query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        rel_types = []

        try:
            with self.driver.session() as session:
                result = session.run(query)
                for record in result:
                    rel_types.append(record["relationshipType"])
        except Exception as e:
            logger.warning(f"Failed to get relation types: {e}")

        return rel_types

    def get_rule_recommendations(
        self,
        entity_id: str,
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get rule-based recommendations for an entity.

        Args:
            entity_id: Entity to get recommendations for
            max_recommendations: Max recommendations to return

        Returns:
            List of recommended relations with confidence
        """
        recommendations = []
        active_rules = self.rule_store.get_active_rules()

        for rule in active_rules:
            if rule.confidence < 0.5:
                continue

            # Check if entity matches rule pattern
            matches = self._check_entity_matches_rule(entity_id, rule)
            for match in matches[:max_recommendations - len(recommendations)]:
                recommendations.append({
                    "rule_id": rule.id,
                    "rule_name": rule.name,
                    "relation_type": rule.conclusion.relation_type,
                    "target_entity": match,
                    "confidence": rule.confidence,
                    "reasoning": rule.description
                })

            if len(recommendations) >= max_recommendations:
                break

        return recommendations

    def _check_entity_matches_rule(
        self,
        entity_id: str,
        rule: ExtractionRule
    ) -> List[str]:
        """Check if an entity matches a rule and return potential targets."""
        if not rule.pattern.edges:
            return []

        # Build query to find potential targets
        first_node = list(rule.pattern.nodes.keys())[0]
        target_node = rule.conclusion.tail_variable

        query = f"""
        MATCH (start {{id: $entity_id}})
        {rule.pattern.to_cypher().replace('MATCH', '')}
        WHERE {first_node} = start
        AND NOT (start)-[:{rule.conclusion.relation_type}]->({target_node})
        RETURN DISTINCT {target_node}.id as target_id
        LIMIT 10
        """

        targets = []
        try:
            with self.driver.session() as session:
                result = session.run(query, entity_id=entity_id)
                for record in result:
                    if record["target_id"]:
                        targets.append(record["target_id"])
        except Exception as e:
            logger.debug(f"Rule match check failed: {e}")

        return targets

    def should_run_learning(self) -> bool:
        """Check if it's time to run a learning cycle."""
        if self.last_learning_cycle is None:
            return True

        hours_since = (datetime.now() - self.last_learning_cycle).total_seconds() / 3600
        return hours_since >= self.config.learning_interval_hours

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress."""
        total_rules = len(self.rule_store.rules)
        active_rules = len(self.rule_store.get_active_rules())

        avg_confidence = 0.0
        if active_rules > 0:
            confidences = [r.confidence for r in self.rule_store.get_active_rules()]
            avg_confidence = sum(confidences) / len(confidences)

        total_facts = sum(m.facts_inferred for m in self.metrics_history)

        return {
            "total_rules": total_rules,
            "active_rules": active_rules,
            "avg_confidence": avg_confidence,
            "learning_cycles": len(self.metrics_history),
            "total_facts_inferred": total_facts,
            "last_cycle": self.last_learning_cycle.isoformat() if self.last_learning_cycle else None
        }
