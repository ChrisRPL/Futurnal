"""Chat CLI commands.

Provides conversational interface for the Futurnal desktop app.

Commands:
    futurnal chat send <session_id> "message" --json
    futurnal chat history <session_id> --json
    futurnal chat sessions --json
    futurnal chat clear <session_id>
    futurnal chat delete <session_id>

Step 03: Chat Interface & Conversational AI
Research Foundation:
- ProPerSim (2509.21730v1): Proactive + personalized AI
- Causal-Copilot (2504.13263v2): Natural language causal exploration
"""

import asyncio
import json
import logging
import sys
from typing import Optional

import typer

logger = logging.getLogger(__name__)

chat_app = typer.Typer(help="Chat commands for conversational knowledge exploration")


def _output_json(data: dict) -> None:
    """Output JSON to stdout for IPC."""
    print(json.dumps(data, indent=2, default=str))


def _output_error(message: str, as_json: bool = False) -> None:
    """Output error message."""
    if as_json:
        _output_json({"error": message, "success": False})
    else:
        typer.echo(f"Error: {message}", err=True)
    sys.exit(1)


@chat_app.command("send")
def send_message(
    session_id: str = typer.Argument(..., help="Session identifier"),
    message: str = typer.Argument(..., help="Message to send"),
    context_entity_id: Optional[str] = typer.Option(
        None, "--context", "-c", help="Entity ID for focused context ('Ask about this')"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="LLM model override"
    ),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Send a chat message and get a response.

    Research Foundation:
    - ProPerSim: Multi-turn context carryover
    - Causal-Copilot: Confidence scoring, source citations

    Example:
        futurnal chat send session-1 "What is Python?" --json
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()

        async def run():
            await service.initialize()
            try:
                response = await service.chat(
                    session_id=session_id,
                    message=message,
                    context_entity_id=context_entity_id,
                    model=model,
                )
                return response
            finally:
                await service.close()

        response = asyncio.run(run())

        if output_json:
            _output_json({
                "success": True,
                "sessionId": session_id,
                "response": {
                    "role": response.role,
                    "content": response.content,
                    "sources": response.sources,
                    "entityRefs": response.entity_refs,
                    "confidence": response.confidence,
                    "timestamp": response.timestamp.isoformat(),
                },
            })
        else:
            typer.echo(f"\nAssistant: {response.content}")
            if response.sources:
                typer.echo(f"\nSources: {', '.join(response.sources)}")
            typer.echo(f"\nConfidence: {response.confidence:.0%}")

    except Exception as e:
        logger.exception("Chat send failed")
        _output_error(str(e), as_json=output_json)


@chat_app.command("stream")
def stream_message(
    session_id: str = typer.Argument(..., help="Session identifier"),
    message: str = typer.Argument(..., help="Message to send"),
    context_entity_id: Optional[str] = typer.Option(
        None, "--context", "-c", help="Entity ID for focused context"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="LLM model override"
    ),
) -> None:
    """Stream a chat response (for real-time UI).

    Outputs tokens as they are generated, followed by JSON metadata.

    Example:
        futurnal chat stream session-1 "Explain knowledge graphs"
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()

        async def run():
            await service.initialize()
            try:
                async for chunk in service.stream_chat(
                    session_id=session_id,
                    message=message,
                    context_entity_id=context_entity_id,
                    model=model,
                ):
                    # Print each chunk without newline for streaming
                    print(chunk, end="", flush=True)

                # Final newline and metadata
                print()
                print("---STREAM_END---")
            finally:
                await service.close()

        asyncio.run(run())

    except Exception as e:
        logger.exception("Chat stream failed")
        print(f"\n---ERROR---\n{e}")
        sys.exit(1)


@chat_app.command("history")
def get_history(
    session_id: str = typer.Argument(..., help="Session identifier"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum messages to return"),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Get conversation history for a session.

    Example:
        futurnal chat history session-1 --limit 20 --json
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()
        messages = service.get_session_history(session_id)

        # Limit messages
        messages = messages[-limit:] if len(messages) > limit else messages

        if output_json:
            _output_json({
                "success": True,
                "sessionId": session_id,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "sources": m.sources,
                        "entityRefs": m.entity_refs,
                        "confidence": m.confidence,
                        "timestamp": m.timestamp.isoformat(),
                    }
                    for m in messages
                ],
                "total": len(messages),
            })
        else:
            if not messages:
                typer.echo("No messages in session")
                return

            for msg in messages:
                role = "User" if msg.role == "user" else "Assistant"
                typer.echo(f"\n[{msg.timestamp.strftime('%H:%M')}] {role}:")
                typer.echo(msg.content[:500])
                if msg.sources:
                    typer.echo(f"  Sources: {', '.join(msg.sources)}")

    except Exception as e:
        logger.exception("Chat history failed")
        _output_error(str(e), as_json=output_json)


@chat_app.command("sessions")
def list_sessions(
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum sessions to return"),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """List all chat sessions.

    Example:
        futurnal chat sessions --limit 10 --json
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()
        sessions = service.list_sessions()

        # Limit sessions
        sessions = sessions[:limit]

        if output_json:
            _output_json({
                "success": True,
                "sessions": sessions,
                "total": len(sessions),
            })
        else:
            if not sessions:
                typer.echo("No chat sessions found")
                return

            for s in sessions:
                typer.echo(f"\n{s['id']}")
                typer.echo(f"  Messages: {s['message_count']}")
                typer.echo(f"  Updated: {s['updated_at']}")
                if s['preview']:
                    typer.echo(f"  Preview: {s['preview'][:60]}...")

    except Exception as e:
        logger.exception("Chat sessions failed")
        _output_error(str(e), as_json=output_json)


@chat_app.command("clear")
def clear_session(
    session_id: str = typer.Argument(..., help="Session identifier"),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Clear messages from a session (keep the session).

    Example:
        futurnal chat clear session-1
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()
        service.clear_session(session_id)

        if output_json:
            _output_json({
                "success": True,
                "sessionId": session_id,
                "message": "Session cleared",
            })
        else:
            typer.echo(f"Session '{session_id}' cleared")

    except Exception as e:
        logger.exception("Chat clear failed")
        _output_error(str(e), as_json=output_json)


@chat_app.command("delete")
def delete_session(
    session_id: str = typer.Argument(..., help="Session identifier"),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Delete a chat session.

    Example:
        futurnal chat delete session-1
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()
        deleted = service.delete_session(session_id)

        if output_json:
            _output_json({
                "success": deleted,
                "sessionId": session_id,
                "message": "Session deleted" if deleted else "Session not found",
            })
        else:
            if deleted:
                typer.echo(f"Session '{session_id}' deleted")
            else:
                typer.echo(f"Session '{session_id}' not found")

    except Exception as e:
        logger.exception("Chat delete failed")
        _output_error(str(e), as_json=output_json)


@chat_app.command("status")
def check_status(
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Check intelligence infrastructure health.

    Shows status of all components required for intelligent conversations:
    - PKG Database (Neo4j)
    - GraphRAG Pipeline
    - Ollama LLM
    - ChromaDB Vector Store
    - Experiential Learning
    - Autonomous Analysis Loop

    Example:
        futurnal chat status --json
    """
    try:
        from futurnal.chat.health import check_intelligence_health

        async def run():
            return await check_intelligence_health()

        report = asyncio.run(run())

        if output_json:
            _output_json({
                "success": True,
                **report.to_dict(),
            })
        else:
            # Pretty print for terminal
            status_icon = {
                "connected": "[OK]",
                "disconnected": "[X]",
                "error": "[!]",
                "degraded": "[~]",
                "not_initialized": "[-]",
            }

            typer.echo(f"\nIntelligence Infrastructure Status: {report.overall_status.upper()}")
            typer.echo("=" * 50)

            for comp in report.components:
                icon = status_icon.get(comp.status, "[?]")
                typer.echo(f"\n{icon} {comp.name}")
                typer.echo(f"    Status: {comp.status}")
                if comp.details:
                    typer.echo(f"    Details: {comp.details}")
                if comp.metrics:
                    for key, value in comp.metrics.items():
                        typer.echo(f"    {key}: {value}")

            if report.recommendations:
                typer.echo("\nRecommendations:")
                for rec in report.recommendations:
                    typer.echo(f"  - {rec}")

    except Exception as e:
        logger.exception("Health check failed")
        _output_error(str(e), as_json=output_json)


# ============================================================
# Causal Discovery CLI Commands (Phase 3)
# ============================================================


@chat_app.command("insights")
def get_causal_insights(
    session_id: str = typer.Argument(..., help="Session identifier"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Topic to focus insights on"),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Get causal insights for a conversation.

    Returns correlations, hypotheses, and pending verifications
    relevant to the current session context.

    Example:
        futurnal chat insights session-1 --topic "productivity"
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()

        async def run():
            await service.initialize()
            try:
                return await service.get_causal_insights(session_id, topic)
            finally:
                await service.close()

        insights = asyncio.run(run())

        if output_json:
            _output_json({
                "success": True,
                "sessionId": session_id,
                "insights": insights,
            })
        else:
            typer.echo(f"\nCausal Insights for session: {session_id}")
            typer.echo("=" * 50)

            if insights.get("correlations"):
                typer.echo("\nDetected Correlations:")
                for corr in insights["correlations"]:
                    typer.echo(f"  - {corr.get('description', str(corr))}")

            if insights.get("hypotheses"):
                typer.echo("\nCausal Hypotheses:")
                for hyp in insights["hypotheses"]:
                    typer.echo(f"  - {hyp.get('hypothesis_text', str(hyp))}")

            if insights.get("pending_verifications"):
                typer.echo("\nPending Verification Questions:")
                for ver in insights["pending_verifications"]:
                    typer.echo(f"  ? {ver.get('main_question', str(ver))}")

            if not any(insights.values()):
                typer.echo("\nNo causal insights available yet.")

    except Exception as e:
        logger.exception("Get insights failed")
        _output_error(str(e), as_json=output_json)


@chat_app.command("ask-causal")
def ask_causal(
    session_id: str = typer.Argument(..., help="Session identifier"),
    cause: str = typer.Argument(..., help="Potential cause event/entity"),
    effect: str = typer.Argument(..., help="Potential effect event/entity"),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Ask about a specific causal relationship.

    Analyzes whether cause might lead to effect based on PKG patterns.

    Example:
        futurnal chat ask-causal session-1 "morning exercise" "productivity"
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()

        async def run():
            await service.initialize()
            try:
                return await service.ask_causal(session_id, cause, effect)
            finally:
                await service.close()

        response = asyncio.run(run())

        if output_json:
            _output_json({
                "success": True,
                "sessionId": session_id,
                "cause": cause,
                "effect": effect,
                "response": {
                    "content": response.content,
                    "confidence": response.confidence,
                    "sources": response.sources,
                },
            })
        else:
            typer.echo(f"\nCausal Analysis: '{cause}' -> '{effect}'")
            typer.echo("=" * 50)
            typer.echo(response.content)
            typer.echo(f"\nConfidence: {response.confidence:.0%}")

    except Exception as e:
        logger.exception("Causal query failed")
        _output_error(str(e), as_json=output_json)


@chat_app.command("explore-chain")
def explore_causal_chain(
    session_id: str = typer.Argument(..., help="Session identifier"),
    start_event: str = typer.Argument(..., help="Starting event/entity"),
    end_event: Optional[str] = typer.Option(None, "--to", "-t", help="Target event/entity"),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Explore causal chains from a starting event.

    Traces the chain of effects from start_event, optionally to end_event.

    Example:
        futurnal chat explore-chain session-1 "project deadline" --to "stress"
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()

        async def run():
            await service.initialize()
            try:
                return await service.explore_causal_chain(session_id, start_event, end_event)
            finally:
                await service.close()

        response = asyncio.run(run())

        if output_json:
            _output_json({
                "success": True,
                "sessionId": session_id,
                "startEvent": start_event,
                "endEvent": end_event,
                "response": {
                    "content": response.content,
                    "confidence": response.confidence,
                    "sources": response.sources,
                },
            })
        else:
            if end_event:
                typer.echo(f"\nCausal Chain: '{start_event}' -> ... -> '{end_event}'")
            else:
                typer.echo(f"\nCausal Chain from: '{start_event}'")
            typer.echo("=" * 50)
            typer.echo(response.content)
            typer.echo(f"\nConfidence: {response.confidence:.0%}")

    except Exception as e:
        logger.exception("Causal chain exploration failed")
        _output_error(str(e), as_json=output_json)


@chat_app.command("pending-icda")
def get_pending_icda_questions(
    limit: int = typer.Option(5, "--limit", "-l", help="Maximum questions to return"),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Get pending ICDA verification questions.

    Returns questions that need user verification to validate causal hypotheses.

    Example:
        futurnal chat pending-icda --limit 3
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()

        async def run():
            await service.initialize()
            try:
                return await service.get_pending_icda_questions(max_items=limit)
            finally:
                await service.close()

        questions = asyncio.run(run())

        if output_json:
            _output_json({
                "success": True,
                "questions": questions,
                "count": len(questions),
            })
        else:
            if not questions:
                typer.echo("\nNo pending verification questions.")
                return

            typer.echo(f"\nPending ICDA Verification Questions ({len(questions)})")
            typer.echo("=" * 50)

            for i, q in enumerate(questions, 1):
                typer.echo(f"\n[{i}] {q.get('main_question', 'Unknown question')}")
                if q.get('hypothesis_text'):
                    typer.echo(f"    Hypothesis: {q['hypothesis_text'][:80]}...")
                typer.echo(f"    Question ID: {q.get('question_id', 'N/A')}")

    except Exception as e:
        logger.exception("Get pending ICDA failed")
        _output_error(str(e), as_json=output_json)


@chat_app.command("respond-icda")
def respond_to_icda(
    session_id: str = typer.Argument(..., help="Session identifier"),
    question_id: str = typer.Argument(..., help="Question ID to respond to"),
    response: str = typer.Argument(..., help="Your response (yes/no/partial/unsure)"),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Respond to an ICDA verification question.

    Valid responses: yes, no, partial, reverse, unsure

    Example:
        futurnal chat respond-icda session-1 q123 "yes"
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()

        async def run():
            await service.initialize()
            try:
                return await service.respond_to_icda(session_id, question_id, response)
            finally:
                await service.close()

        result = asyncio.run(run())

        if output_json:
            _output_json({
                "success": True,
                "sessionId": session_id,
                "questionId": question_id,
                "userResponse": response,
                "acknowledgment": result.content,
            })
        else:
            typer.echo(f"\n{result.content}")

    except Exception as e:
        logger.exception("ICDA response failed")
        _output_error(str(e), as_json=output_json)


@chat_app.command("insight-stats")
def get_insight_statistics(
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Get causal insight statistics.

    Shows statistics about insight surfacing and causal discovery.

    Example:
        futurnal chat insight-stats --json
    """
    try:
        from futurnal.chat import ChatService

        service = ChatService()
        stats = service.get_insight_statistics()

        if output_json:
            _output_json({
                "success": True,
                "statistics": stats,
            })
        else:
            typer.echo("\nCausal Insight Statistics")
            typer.echo("=" * 50)
            typer.echo(f"\nSurfaced Insights: {stats.get('surfaced_insights_count', 0)}")
            typer.echo(f"Insight Surfacing: {'Enabled' if stats.get('insight_surfacing_enabled') else 'Disabled'}")

            components = stats.get("components_available", {})
            typer.echo("\nComponents:")
            for name, available in components.items():
                status = "[OK]" if available else "[X]"
                typer.echo(f"  {status} {name}")

            if stats.get("executor_statistics"):
                exec_stats = stats["executor_statistics"]
                typer.echo("\nExecutor Statistics:")
                typer.echo(f"  Total Jobs: {exec_stats.get('total_jobs', 0)}")
                typer.echo(f"  Correlations Found: {exec_stats.get('total_correlations_found', 0)}")
                typer.echo(f"  Hypotheses Generated: {exec_stats.get('total_hypotheses_generated', 0)}")
                typer.echo(f"  Hypotheses Validated: {exec_stats.get('total_hypotheses_validated', 0)}")

    except Exception as e:
        logger.exception("Get insight stats failed")
        _output_error(str(e), as_json=output_json)
