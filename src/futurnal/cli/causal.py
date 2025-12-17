"""Causal Chain CLI commands for exploring cause-effect relationships.

Step 08: Frontend Intelligence Integration - Phase 2

Provides commands for:
- Finding causes of an event (what led to X?)
- Finding effects of an event (what resulted from X?)
- Finding causal paths between events (how did A lead to B?)

Research Foundation:
- Youtu-GraphRAG: Multi-hop causal reasoning
- CausalRAG: Causal-aware retrieval
"""

import json
import logging
from datetime import datetime
from typing import Optional

from typer import Typer, Argument, Option

logger = logging.getLogger(__name__)

causal_app = Typer(help="Causal chain exploration commands")


def _get_causal_retrieval():
    """Factory function to get CausalChainRetrieval instance.

    Lazy initialization to avoid circular imports.
    """
    from futurnal.pkg.queries.temporal import TemporalGraphQueries
    from futurnal.search.causal.retrieval import CausalChainRetrieval
    from futurnal.pkg.db import get_pkg_database

    db = get_pkg_database()
    pkg_queries = TemporalGraphQueries(db)
    return CausalChainRetrieval(pkg_queries)


@causal_app.command("causes")
def find_causes(
    event_id: str = Argument(..., help="ID of the target event"),
    max_hops: int = Option(3, "--max-hops", "-h", help="Maximum causal hops to traverse (1-10)"),
    min_confidence: float = Option(0.6, "--min-confidence", "-c", help="Minimum confidence threshold"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Find what caused an event.

    Example: "What led to this decision?"

    Examples:
        futurnal causal causes event_123
        futurnal causal causes event_123 --max-hops 5 --json
    """
    try:
        retrieval = _get_causal_retrieval()
        result = retrieval.find_causes(
            event_id=event_id,
            max_hops=max_hops,
            min_confidence=min_confidence,
        )

        if output_json:
            output = {
                "success": True,
                "targetEventId": result.target_event_id,
                "targetEvent": result.target_event.model_dump() if result.target_event else None,
                "causes": [
                    {
                        "causeId": c.cause_id,
                        "causeName": c.cause_name,
                        "causeTimestamp": c.cause_timestamp.isoformat() if c.cause_timestamp else None,
                        "distance": c.distance,
                        "confidenceScores": c.confidence_scores,
                        "aggregateConfidence": c.aggregate_confidence,
                        "temporalOrderingValid": c.temporal_ordering_valid,
                    }
                    for c in result.causes
                ],
                "maxHopsRequested": result.max_hops_requested,
                "minConfidenceRequested": result.min_confidence_requested,
                "queryTimeMs": result.query_time_ms,
            }
            print(json.dumps(output))
        else:
            print(f"Causes of event '{event_id}':")
            print(f"Query time: {result.query_time_ms:.1f}ms")
            print("-" * 40)

            if not result.causes:
                print("No causes found within specified constraints.")
            else:
                for cause in result.causes:
                    valid_marker = "+" if cause.temporal_ordering_valid else "-"
                    print(
                        f"[{valid_marker}] {cause.cause_name} "
                        f"(distance: {cause.distance}, conf: {cause.aggregate_confidence:.2f})"
                    )

    except Exception as e:
        logger.exception("Find causes failed")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)


@causal_app.command("effects")
def find_effects(
    event_id: str = Argument(..., help="ID of the source event"),
    max_hops: int = Option(3, "--max-hops", "-h", help="Maximum causal hops to traverse (1-10)"),
    min_confidence: float = Option(0.6, "--min-confidence", "-c", help="Minimum confidence threshold"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Find what resulted from an event.

    Example: "What resulted from this meeting?"

    Examples:
        futurnal causal effects event_123
        futurnal causal effects event_123 --max-hops 5 --json
    """
    try:
        retrieval = _get_causal_retrieval()
        result = retrieval.find_effects(
            event_id=event_id,
            max_hops=max_hops,
            min_confidence=min_confidence,
        )

        if output_json:
            output = {
                "success": True,
                "sourceEventId": result.source_event_id,
                "sourceEvent": result.source_event.model_dump() if result.source_event else None,
                "effects": [
                    {
                        "effectId": e.effect_id,
                        "effectName": e.effect_name,
                        "effectTimestamp": e.effect_timestamp.isoformat() if e.effect_timestamp else None,
                        "distance": e.distance,
                        "confidenceScores": e.confidence_scores,
                        "aggregateConfidence": e.aggregate_confidence,
                        "temporalOrderingValid": e.temporal_ordering_valid,
                    }
                    for e in result.effects
                ],
                "maxHopsRequested": result.max_hops_requested,
                "minConfidenceRequested": result.min_confidence_requested,
                "queryTimeMs": result.query_time_ms,
            }
            print(json.dumps(output))
        else:
            print(f"Effects of event '{event_id}':")
            print(f"Query time: {result.query_time_ms:.1f}ms")
            print("-" * 40)

            if not result.effects:
                print("No effects found within specified constraints.")
            else:
                for effect in result.effects:
                    valid_marker = "+" if effect.temporal_ordering_valid else "-"
                    print(
                        f"[{valid_marker}] {effect.effect_name} "
                        f"(distance: {effect.distance}, conf: {effect.aggregate_confidence:.2f})"
                    )

    except Exception as e:
        logger.exception("Find effects failed")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)


@causal_app.command("path")
def find_causal_path(
    start_id: str = Argument(..., help="ID of the starting event"),
    end_id: str = Argument(..., help="ID of the ending event"),
    max_hops: int = Option(5, "--max-hops", "-h", help="Maximum path length (1-10)"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Find causal path from one event to another.

    Example: "How did event A lead to event B?"

    Examples:
        futurnal causal path event_1 event_2
        futurnal causal path event_1 event_2 --max-hops 7 --json
    """
    try:
        retrieval = _get_causal_retrieval()
        result = retrieval.find_causal_path(
            start_event_id=start_id,
            end_event_id=end_id,
            max_hops=max_hops,
        )

        if output_json:
            path_data = None
            if result.path:
                path_data = {
                    "startEventId": result.path.start_event_id,
                    "endEventId": result.path.end_event_id,
                    "path": result.path.path,
                    "causalConfidence": result.path.causal_confidence,
                    "confidenceScores": result.path.confidence_scores,
                    "temporalOrderingValid": result.path.temporal_ordering_valid,
                    "causalEvidence": result.path.causal_evidence,
                }

            output = {
                "success": True,
                "pathFound": result.path_found,
                "path": path_data,
                "startEventId": result.start_event_id,
                "endEventId": result.end_event_id,
                "maxHopsRequested": result.max_hops_requested,
                "queryTimeMs": result.query_time_ms,
            }
            print(json.dumps(output))
        else:
            print(f"Causal path from '{start_id}' to '{end_id}':")
            print(f"Query time: {result.query_time_ms:.1f}ms")
            print("-" * 40)

            if not result.path_found:
                print(f"No causal path found within {max_hops} hops.")
            else:
                path = result.path
                print(f"Path found with {len(path.path)} events:")
                print(" -> ".join(path.path))
                print(f"\nConfidence: {path.causal_confidence:.2f}")
                print(f"Temporal ordering valid: {path.temporal_ordering_valid}")

                if path.causal_evidence:
                    print("\nEvidence:")
                    for i, ev in enumerate(path.causal_evidence):
                        if ev:
                            print(f"  {i+1}. {ev}")

    except Exception as e:
        logger.exception("Find causal path failed")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)
