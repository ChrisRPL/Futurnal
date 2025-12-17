"""Learning Progress CLI commands for experiential learning visualization.

Step 08: Frontend Intelligence Integration - Phase 6

Provides commands for:
- Getting learning progress metrics
- Getting pattern learning stats
- Getting quality gates status
- Recording document learning events

Research Foundation:
- RLHI: Reinforcement Learning from Human Interactions
- AgentFlow: Learning from user feedback
- Option B: Ghost frozen, learning via token priors
"""

import json
import logging
from typing import Optional, List

from typer import Typer, Option, Argument

logger = logging.getLogger(__name__)

learning_app = Typer(help="Experiential learning progress commands")


def _get_persistent_pipeline():
    """Get the persistent ExperientialLearningPipeline singleton.

    Uses the singleton pattern to ensure state is preserved across CLI calls.
    Returns None if module not available.
    """
    try:
        from futurnal.learning.integration import get_persistent_pipeline
        return get_persistent_pipeline()
    except ImportError as e:
        logger.warning(f"Learning module not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Learning pipeline initialization failed: {e}")
        return None


@learning_app.command("progress")
def get_learning_progress(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get learning progress metrics and quality gates status.

    Examples:
        futurnal learning progress
        futurnal learning progress --json
    """
    try:
        # Get persistent learning pipeline (preserves state across calls)
        pipeline = _get_persistent_pipeline()

        if pipeline is not None:
            # Get stats from actual pipeline
            state = pipeline.state
            documents_processed = state.total_documents_processed
            success_rate = state.overall_success_rate

            # Quality progression
            if state.total_documents_processed > 0:
                quality_before = state.cumulative_quality_before / state.total_documents_processed
                quality_after = state.cumulative_quality_after / state.total_documents_processed
            else:
                quality_before = 0.0
                quality_after = 0.0
            improvement = quality_after - quality_before

            # Pattern learning stats from token store
            entity_priors = len(pipeline.token_store.entity_priors)
            relation_priors = len(pipeline.token_store.relation_priors)
            temporal_priors = len(pipeline.token_store.temporal_priors)

            # Quality gates
            ghost_frozen = True  # Always true per Option B
            improvement_threshold = 0.05
            meets_threshold = state.passes_quality_gate
        else:
            # Default/placeholder values when pipeline not available
            documents_processed = 0
            success_rate = 0.0
            quality_before = 0.0
            quality_after = 0.0
            improvement = 0.0
            entity_priors = 0
            relation_priors = 0
            temporal_priors = 0
            ghost_frozen = True
            improvement_threshold = 0.05
            meets_threshold = False

        if output_json:
            output = {
                "success": True,
                "documentsProcessed": documents_processed,
                "successRate": success_rate,
                "qualityProgression": {
                    "before": quality_before,
                    "after": quality_after,
                    "improvement": improvement,
                },
                "patternLearning": {
                    "entityPriors": entity_priors,
                    "relationPriors": relation_priors,
                    "temporalPriors": temporal_priors,
                },
                "qualityGates": {
                    "ghostFrozen": ghost_frozen,
                    "improvementThreshold": improvement_threshold,
                    "meetsThreshold": meets_threshold,
                },
            }
            print(json.dumps(output))
        else:
            print("Learning Progress:")
            print("-" * 40)
            print(f"\nDocuments Processed: {documents_processed}")
            print(f"Success Rate: {success_rate * 100:.1f}%")

            print("\nQuality Progression:")
            print(f"  Before: {quality_before:.2f}")
            print(f"  After: {quality_after:.2f}")
            print(f"  Improvement: {improvement:+.2f}")

            print("\nPattern Learning:")
            print(f"  Entity Priors: {entity_priors}")
            print(f"  Relation Priors: {relation_priors}")
            print(f"  Temporal Priors: {temporal_priors}")

            print("\nQuality Gates:")
            print(f"  Ghost Model Frozen: {ghost_frozen}")
            print(f"  Improvement Threshold: {improvement_threshold * 100:.0f}%")
            print(f"  Meets Threshold: {meets_threshold}")

    except Exception as e:
        logger.exception("Get learning progress failed")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)


@learning_app.command("record")
def record_document_learning(
    document_id: str = Argument(..., help="Unique identifier for the document"),
    content: str = Option("", "--content", "-c", help="Extracted text content"),
    source: str = Option("chat", "--source", "-s", help="Source of the document (chat, ingestion, etc.)"),
    content_type: str = Option("text", "--type", "-t", help="Content type (text, image, audio, document)"),
    success: bool = Option(True, "--success/--failure", help="Whether extraction was successful"),
    quality: Optional[float] = Option(None, "--quality", "-q", help="Quality score (0-1)"),
    entities: Optional[str] = Option(None, "--entities", "-e", help="Comma-separated entity types"),
    relations: Optional[str] = Option(None, "--relations", "-r", help="Comma-separated relation types"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Record a document processing event to the learning pipeline.

    Updates learning state and token priors, then persists to disk.
    Used by frontend to track chat attachments and processed documents.

    Examples:
        futurnal learning record doc_123 --content "Meeting notes..." --type text
        futurnal learning record img_456 --type image --entities "Image,Screenshot"
        futurnal learning record audio_789 --type audio --source chat --json
    """
    try:
        pipeline = _get_persistent_pipeline()

        if pipeline is None:
            error_msg = "Learning pipeline not available"
            if output_json:
                print(json.dumps({"success": False, "error": error_msg}))
            else:
                print(f"Error: {error_msg}")
            raise SystemExit(1)

        # Parse entity and relation types
        entity_types: Optional[List[str]] = None
        if entities:
            entity_types = [e.strip() for e in entities.split(",") if e.strip()]

        relation_types: Optional[List[str]] = None
        if relations:
            relation_types = [r.strip() for r in relations.split(",") if r.strip()]

        # Record document
        result = pipeline.record_document(
            document_id=document_id,
            content=content,
            source=source,
            content_type=content_type,
            success=success,
            quality_score=quality,
            entity_types=entity_types,
            relation_types=relation_types,
        )

        if output_json:
            output = {
                "success": True,
                "documentId": result["document_id"],
                "qualityScore": result["quality_score"],
                "totalDocuments": result["total_documents"],
                "overallSuccessRate": result["overall_success_rate"],
                "entityPriors": result["entity_priors"],
                "relationPriors": result["relation_priors"],
            }
            print(json.dumps(output))
        else:
            print(f"Recorded document: {document_id}")
            print(f"  Quality Score: {result['quality_score']:.2f}")
            print(f"  Total Documents: {result['total_documents']}")
            print(f"  Success Rate: {result['overall_success_rate'] * 100:.1f}%")
            print(f"  Entity Priors: {result['entity_priors']}")
            print(f"  Relation Priors: {result['relation_priors']}")

    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Record document learning failed")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)


@learning_app.command("reset")
def reset_learning(
    confirm: bool = Option(False, "--yes", "-y", help="Confirm reset without prompt"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Reset all learning state and token priors.

    WARNING: This clears all learned patterns and statistics.

    Examples:
        futurnal learning reset --yes
        futurnal learning reset --json --yes
    """
    try:
        if not confirm:
            print("This will reset all learning state and token priors.")
            print("Use --yes to confirm.")
            return

        pipeline = _get_persistent_pipeline()

        if pipeline is None:
            error_msg = "Learning pipeline not available"
            if output_json:
                print(json.dumps({"success": False, "error": error_msg}))
            else:
                print(f"Error: {error_msg}")
            raise SystemExit(1)

        # Reset learning
        result = pipeline.reset_learning()

        # Save the reset state
        pipeline.save_state()

        if output_json:
            output = {
                "success": True,
                "trajectoriesCleared": result["trajectories_cleared"],
                "priorsCleared": result["priors_cleared"],
                "stateReset": result["learning_state_reset"],
            }
            print(json.dumps(output))
        else:
            print("Learning state reset:")
            print(f"  Trajectories Cleared: {result['trajectories_cleared']}")
            print(f"  Priors Cleared: {result['priors_cleared']}")
            print(f"  State Reset: {result['learning_state_reset']}")

    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Reset learning failed")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)
