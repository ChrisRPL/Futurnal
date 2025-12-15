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
