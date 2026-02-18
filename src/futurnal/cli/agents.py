"""AgentFlow CLI commands for Phase 2E.

Provides commands for:
- Memory buffer management
- Hypothesis generation and investigation
- Correlation verification
- AgentFlow status monitoring
"""

import json
import logging
from typing import Optional

from typer import Typer, Option, Argument

logger = logging.getLogger(__name__)

agents_app = Typer(help="AgentFlow analysis and correlation commands")


def _get_memory_buffer():
    """Get EvolvingMemoryBuffer singleton."""
    try:
        from futurnal.agents.memory_buffer import get_memory_buffer
        return get_memory_buffer()
    except ImportError as e:
        logger.warning(f"Memory buffer not available: {e}")
        return None


def _get_correlation_planner():
    """Get CorrelationPlanner singleton."""
    try:
        from futurnal.agents.correlation_planner import get_correlation_planner
        return get_correlation_planner()
    except ImportError as e:
        logger.warning(f"Correlation planner not available: {e}")
        return None


def _get_correlation_verifier():
    """Get CorrelationVerifier singleton."""
    try:
        from futurnal.agents.correlation_verifier import get_correlation_verifier
        return get_correlation_verifier()
    except ImportError as e:
        logger.warning(f"Correlation verifier not available: {e}")
        return None


# ============================================================================
# Memory Buffer Commands
# ============================================================================

@agents_app.command("memory-stats")
def memory_stats(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get memory buffer statistics.

    Examples:
        futurnal agents memory-stats
        futurnal agents memory-stats --json
    """
    try:
        buffer = _get_memory_buffer()
        if buffer is None:
            raise RuntimeError("Memory buffer not available")

        stats = buffer.get_stats()
        stats["success"] = True

        if output_json:
            print(json.dumps(stats))
        else:
            print("\nMemory Buffer Statistics")
            print("-" * 40)
            print(f"Total entries: {stats['totalEntries']}/{stats['maxEntries']}")
            print(f"Utilization: {stats['utilization']:.0%}")
            print("\nBy Type:")
            for entry_type, count in stats.get("byType", {}).items():
                print(f"  {entry_type}: {count}")
            print("\nBy Priority:")
            for priority, count in stats.get("byPriority", {}).items():
                print(f"  {priority}: {count}")

    except Exception as e:
        logger.error(f"Memory stats failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")


@agents_app.command("memory-recent")
def memory_recent(
    limit: int = Option(10, "--limit", "-n", help="Number of entries to show"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get recent memory entries.

    Examples:
        futurnal agents memory-recent
        futurnal agents memory-recent --limit 5 --json
    """
    try:
        buffer = _get_memory_buffer()
        if buffer is None:
            raise RuntimeError("Memory buffer not available")

        entries = buffer.get_recent(limit)

        if output_json:
            print(json.dumps({
                "success": True,
                "entries": [e.to_dict() for e in entries],
                "count": len(entries),
            }))
        else:
            print(f"\nRecent Memory Entries ({len(entries)} shown)")
            print("-" * 60)
            for entry in entries:
                ts = entry.timestamp.strftime("%Y-%m-%d %H:%M")
                print(f"[{ts}] ({entry.entry_type.value}) {entry.content[:60]}...")

    except Exception as e:
        logger.error(f"Memory recent failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "entries": [], "error": str(e)}))
        else:
            print(f"Error: {e}")


@agents_app.command("memory-search")
def memory_search(
    query: str = Argument(..., help="Search query"),
    limit: int = Option(5, "--limit", "-n", help="Maximum results"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Search memory buffer for relevant entries.

    Examples:
        futurnal agents memory-search "productivity patterns"
        futurnal agents memory-search "monday meetings" --limit 3 --json
    """
    try:
        buffer = _get_memory_buffer()
        if buffer is None:
            raise RuntimeError("Memory buffer not available")

        entries = buffer.get_relevant_context(query, max_entries=limit)

        if output_json:
            print(json.dumps({
                "success": True,
                "query": query,
                "entries": [e.to_dict() for e in entries],
                "count": len(entries),
            }))
        else:
            print(f"\nSearch Results for '{query}' ({len(entries)} found)")
            print("-" * 60)
            for entry in entries:
                ts = entry.timestamp.strftime("%Y-%m-%d")
                print(f"[{ts}] ({entry.priority.value}) {entry.content[:80]}...")

    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "entries": [], "error": str(e)}))
        else:
            print(f"Error: {e}")


@agents_app.command("memory-clear")
def memory_clear(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Clear all memory entries.

    Examples:
        futurnal agents memory-clear
        futurnal agents memory-clear --json
    """
    try:
        buffer = _get_memory_buffer()
        if buffer is None:
            raise RuntimeError("Memory buffer not available")

        count = buffer.clear()

        if output_json:
            print(json.dumps({"success": True, "clearedCount": count}))
        else:
            print(f"Cleared {count} memory entries")

    except Exception as e:
        logger.error(f"Memory clear failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "clearedCount": 0, "error": str(e)}))
        else:
            print(f"Error: {e}")


# ============================================================================
# Hypothesis Commands
# ============================================================================

@agents_app.command("hypotheses")
def list_hypotheses(
    status: Optional[str] = Option(None, "--status", "-s", help="Filter by status"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """List active correlation hypotheses.

    Examples:
        futurnal agents hypotheses
        futurnal agents hypotheses --status investigating --json
    """
    try:
        planner = _get_correlation_planner()
        if planner is None:
            raise RuntimeError("Correlation planner not available")

        hypotheses = planner.get_active_hypotheses()

        # Filter by status if specified
        if status:
            from futurnal.agents.correlation_planner import HypothesisStatus
            try:
                status_enum = HypothesisStatus(status)
                hypotheses = [h for h in hypotheses if h.status == status_enum]
            except ValueError:
                valid = ", ".join(s.value for s in HypothesisStatus)
                raise ValueError(f"Invalid status '{status}'. Valid: {valid}")

        if output_json:
            print(json.dumps({
                "success": True,
                "hypotheses": [h.to_dict() for h in hypotheses],
                "count": len(hypotheses),
            }))
        else:
            print(f"\nActive Hypotheses ({len(hypotheses)} total)")
            print("-" * 60)
            for h in hypotheses:
                conf = f"{h.confidence:.0%}"
                print(f"[{h.status.value}] ({conf}) {h.description[:50]}...")

    except Exception as e:
        logger.error(f"List hypotheses failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "hypotheses": [], "error": str(e)}))
        else:
            print(f"Error: {e}")


@agents_app.command("generate-hypotheses")
def generate_hypotheses(
    event_types: str = Argument(..., help="Comma-separated event types"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Generate correlation hypotheses from event types.

    Examples:
        futurnal agents generate-hypotheses "note_created,meeting_noted,decision_made"
        futurnal agents generate-hypotheses "commit_pushed,task_completed" --json
    """
    try:
        planner = _get_correlation_planner()
        if planner is None:
            raise RuntimeError("Correlation planner not available")

        types = [t.strip() for t in event_types.split(",") if t.strip()]
        if len(types) < 2:
            raise ValueError("Need at least 2 event types to generate hypotheses")

        hypotheses = planner.generate_hypotheses(types)

        if output_json:
            print(json.dumps({
                "success": True,
                "hypotheses": [h.to_dict() for h in hypotheses],
                "count": len(hypotheses),
            }))
        else:
            print(f"\nGenerated {len(hypotheses)} hypotheses")
            print("-" * 60)
            for h in hypotheses:
                print(f"- [{h.hypothesis_type.value}] {h.description}")

    except Exception as e:
        logger.error(f"Generate hypotheses failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "hypotheses": [], "error": str(e)}))
        else:
            print(f"Error: {e}")


@agents_app.command("investigate")
def investigate_hypothesis(
    hypothesis_id: str = Argument(..., help="Hypothesis ID to investigate"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Design a query plan to investigate a hypothesis.

    Examples:
        futurnal agents investigate <hypothesis_id>
        futurnal agents investigate <hypothesis_id> --json
    """
    try:
        planner = _get_correlation_planner()
        if planner is None:
            raise RuntimeError("Correlation planner not available")

        hypothesis = planner.get_hypothesis(hypothesis_id)
        if not hypothesis:
            raise ValueError(f"Hypothesis not found: {hypothesis_id}")

        plan = planner.design_query_strategy(hypothesis)

        if output_json:
            print(json.dumps({
                "success": True,
                "hypothesisId": hypothesis_id,
                "queryPlan": plan.to_dict(),
            }))
        else:
            print(f"\nQuery Plan for: {hypothesis.description[:50]}...")
            print("-" * 60)
            print(f"Queries to execute: {len(plan.queries)}")
            for i, query in enumerate(plan.queries, 1):
                print(f"  {i}. {query.get('description', query.get('type', 'unknown'))}")
            print(f"\nExpected results:")
            for result in plan.expected_results:
                print(f"  - {result}")
            print(f"\nCompletion criteria: {plan.completion_criteria}")

    except Exception as e:
        logger.error(f"Investigate failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")


# ============================================================================
# Verification Commands
# ============================================================================

@agents_app.command("verify")
def verify_hypothesis(
    hypothesis_id: str = Argument(..., help="Hypothesis ID to verify"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Verify a hypothesis with current evidence.

    Examples:
        futurnal agents verify <hypothesis_id>
        futurnal agents verify <hypothesis_id> --json
    """
    try:
        planner = _get_correlation_planner()
        verifier = _get_correlation_verifier()

        if planner is None:
            raise RuntimeError("Correlation planner not available")
        if verifier is None:
            raise RuntimeError("Correlation verifier not available")

        hypothesis = planner.get_hypothesis(hypothesis_id)
        if not hypothesis:
            raise ValueError(f"Hypothesis not found: {hypothesis_id}")

        # Get existing evidence from hypothesis
        from futurnal.agents.correlation_verifier import EvidenceItem
        evidence = []
        for e in hypothesis.evidence_for:
            evidence.append(EvidenceItem(description=e, is_supporting=True))
        for e in hypothesis.evidence_against:
            evidence.append(EvidenceItem(description=e, is_supporting=False))

        report = verifier.verify_evidence(hypothesis, evidence)

        if output_json:
            print(json.dumps({
                "success": True,
                "hypothesisId": hypothesis_id,
                "report": report.to_dict(),
            }))
        else:
            print(f"\nVerification Report")
            print("-" * 60)
            print(report.to_natural_language())

    except Exception as e:
        logger.error(f"Verify failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")


@agents_app.command("verification-history")
def verification_history(
    hypothesis_id: str = Argument(..., help="Hypothesis ID"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get verification history for a hypothesis.

    Examples:
        futurnal agents verification-history <hypothesis_id>
        futurnal agents verification-history <hypothesis_id> --json
    """
    try:
        verifier = _get_correlation_verifier()
        if verifier is None:
            raise RuntimeError("Correlation verifier not available")

        history = verifier.get_verification_history(hypothesis_id)

        if output_json:
            print(json.dumps({
                "success": True,
                "hypothesisId": hypothesis_id,
                "history": [r.to_dict() for r in history],
                "count": len(history),
            }))
        else:
            print(f"\nVerification History ({len(history)} records)")
            print("-" * 60)
            for report in history:
                ts = report.created_at.strftime("%Y-%m-%d %H:%M")
                print(f"[{ts}] {report.result.value.upper()} ({report.confidence:.0%})")

    except Exception as e:
        logger.error(f"Verification history failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "history": [], "error": str(e)}))
        else:
            print(f"Error: {e}")


# ============================================================================
# Status Commands
# ============================================================================

@agents_app.command("status")
def agentflow_status(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get AgentFlow system status.

    Examples:
        futurnal agents status
        futurnal agents status --json
    """
    try:
        buffer = _get_memory_buffer()
        planner = _get_correlation_planner()
        verifier = _get_correlation_verifier()

        status = {
            "success": True,
            "memory_buffer": {
                "available": buffer is not None,
                "stats": buffer.get_stats() if buffer else None,
            },
            "correlation_planner": {
                "available": planner is not None,
                "active_hypotheses": len(planner.get_active_hypotheses()) if planner else 0,
            },
            "correlation_verifier": {
                "available": verifier is not None,
                "verified_count": len(verifier._verification_history) if verifier else 0,
            },
        }

        if output_json:
            print(json.dumps(status))
        else:
            print("\nAgentFlow Status")
            print("-" * 40)
            print(f"Memory Buffer: {'OK' if buffer else 'NOT AVAILABLE'}")
            if buffer:
                stats = buffer.get_stats()
                print(f"  Entries: {stats['total_entries']}/{stats['max_entries']}")
            print(f"\nCorrelation Planner: {'OK' if planner else 'NOT AVAILABLE'}")
            if planner:
                print(f"  Active hypotheses: {len(planner.get_active_hypotheses())}")
            print(f"\nCorrelation Verifier: {'OK' if verifier else 'NOT AVAILABLE'}")
            if verifier:
                print(f"  Verified hypotheses: {len(verifier._verification_history)}")

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")


@agents_app.command("export-priors")
def export_token_priors(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Export AgentFlow state as natural language for token priors.

    This exports the memory buffer and correlation state as natural
    language text suitable for use as context in LLM prompts.

    Examples:
        futurnal agents export-priors
        futurnal agents export-priors --json
    """
    try:
        buffer = _get_memory_buffer()
        planner = _get_correlation_planner()
        verifier = _get_correlation_verifier()

        parts = []

        if buffer:
            parts.append(buffer.export_for_token_priors())

        if planner:
            parts.append(planner.export_for_token_priors())

        if verifier:
            parts.append(verifier.export_for_token_priors())

        content = "\n\n---\n\n".join(parts) if parts else "No AgentFlow state available."

        if output_json:
            print(json.dumps({
                "success": True,
                "content": content,
            }))
        else:
            print("\nAgentFlow Token Priors Export")
            print("=" * 60)
            print(content)

    except Exception as e:
        logger.error(f"Export priors failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "content": "", "error": str(e)}))
        else:
            print(f"Error: {e}")
