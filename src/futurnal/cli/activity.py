"""Activity Stream CLI commands.

Step 08: Frontend Intelligence Integration - Phase 3

Provides unified activity stream combining:
- Audit logs (system events)
- Search history
- Chat sessions
- Ingestion events
- Schema evolution

Research Foundation:
- AgentFlow: Activity tracking patterns
- RLHI: User interaction history
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from typer import Typer, Option

logger = logging.getLogger(__name__)

activity_app = Typer(help="Activity stream commands")


def _get_workspace_dir() -> Path:
    """Get Futurnal workspace directory."""
    import os
    return Path(os.path.expanduser("~/.futurnal"))


def _read_audit_logs(
    workspace: Path,
    limit: int,
    date_from: Optional[datetime],
    date_to: Optional[datetime],
) -> List[Dict[str, Any]]:
    """Read events from audit logs."""
    events = []
    audit_path = workspace / "audit" / "audit.log"

    if not audit_path.exists():
        return events

    try:
        with open(audit_path, 'r') as f:
            lines = f.readlines()

        # Process most recent first
        for line in reversed(lines[-1000:]):  # Cap at 1000 lines for performance
            if len(events) >= limit:
                break

            try:
                entry = json.loads(line.strip())
                timestamp_str = entry.get("timestamp")
                if not timestamp_str:
                    continue

                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

                # Apply date filters
                if date_from and timestamp < date_from:
                    continue
                if date_to and timestamp > date_to:
                    continue

                # Map to activity event
                event_type = _map_action_to_type(entry.get("action", ""))
                event = {
                    "id": entry.get("job_id", f"audit-{timestamp.timestamp()}"),
                    "type": event_type,
                    "category": _get_event_category(event_type),
                    "title": _format_audit_title(entry),
                    "description": entry.get("redacted_path") or entry.get("metadata", {}).get("message"),
                    "timestamp": timestamp.isoformat(),
                    "relatedEntityIds": [],
                    "metadata": {
                        "source": entry.get("source"),
                        "action": entry.get("action"),
                        "status": entry.get("status"),
                    },
                }
                events.append(event)

            except (json.JSONDecodeError, ValueError) as e:
                continue

    except Exception as e:
        logger.warning(f"Failed to read audit logs: {e}")

    return events


def _map_action_to_type(action: str) -> str:
    """Map audit action to activity type."""
    action_lower = action.lower()

    if "search" in action_lower:
        return "search"
    elif "chat" in action_lower:
        return "chat"
    elif "ingest" in action_lower or "process" in action_lower:
        return "document"
    elif "consent" in action_lower:
        return "schema"
    elif "entity" in action_lower or "graph" in action_lower:
        return "entity"
    elif "learn" in action_lower or "pattern" in action_lower:
        return "insight"
    else:
        return "document"


def _get_event_category(event_type: str) -> str:
    """Get category for event type."""
    if event_type in ("search", "chat"):
        return "interaction"
    elif event_type in ("insight", "schema"):
        return "learning"
    else:
        return "data"


def _format_audit_title(entry: Dict[str, Any]) -> str:
    """Format audit entry as activity title."""
    action = entry.get("action", "Unknown action")
    source = entry.get("source", "system")
    status = entry.get("status", "")

    # Clean up action for display
    action = action.replace("_", " ").replace(":", " - ").title()

    if status:
        return f"{action} ({status})"
    return action


@activity_app.command("list")
def list_activities(
    limit: int = Option(50, "--limit", "-l", help="Maximum number of activities to return"),
    offset: int = Option(0, "--offset", "-o", help="Offset for pagination"),
    event_types: Optional[str] = Option(None, "--types", "-t", help="Comma-separated event types filter"),
    date_from: Optional[str] = Option(None, "--from", "-f", help="Start date (ISO format)"),
    date_to: Optional[str] = Option(None, "--to", help="End date (ISO format)"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """List activity events from all sources.

    Combines audit logs, search history, chat sessions, and more.

    Examples:
        futurnal activity list
        futurnal activity list --limit 20 --types search,chat
        futurnal activity list --from 2024-01-01 --json
    """
    try:
        workspace = _get_workspace_dir()

        # Parse date filters
        parsed_from = datetime.fromisoformat(date_from) if date_from else None
        parsed_to = datetime.fromisoformat(date_to) if date_to else None

        # Parse event types filter
        type_filter = event_types.split(",") if event_types else None

        # Collect events from all sources
        all_events = []

        # 1. Audit logs
        audit_events = _read_audit_logs(
            workspace,
            limit=limit * 2,  # Fetch more, filter later
            date_from=parsed_from,
            date_to=parsed_to,
        )
        all_events.extend(audit_events)

        # 2. Additional sources could be added here
        # - Search history from searchStore
        # - Chat sessions from chatStore
        # - etc.

        # Sort by timestamp (newest first)
        all_events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

        # Apply type filter
        if type_filter:
            all_events = [e for e in all_events if e.get("type") in type_filter]

        # Apply pagination
        paginated = all_events[offset:offset + limit]

        if output_json:
            output = {
                "success": True,
                "events": paginated,
                "total": len(all_events),
                "limit": limit,
                "offset": offset,
            }
            print(json.dumps(output))
        else:
            print(f"Activity Stream ({len(paginated)} of {len(all_events)} events)")
            print("-" * 50)

            for event in paginated:
                ts = event.get("timestamp", "")
                try:
                    ts_display = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
                except:
                    ts_display = ts[:16]

                type_icon = {
                    "search": "[S]",
                    "chat": "[C]",
                    "document": "[D]",
                    "entity": "[E]",
                    "insight": "[I]",
                    "schema": "[+]",
                }.get(event.get("type", ""), "[?]")

                print(f"{type_icon} {ts_display} | {event.get('title', 'Unknown')}")

    except Exception as e:
        logger.exception("Failed to list activities")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)


@activity_app.command("recent")
def recent_activities(
    limit: int = Option(10, "--limit", "-l", help="Number of recent activities"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get most recent activities (shortcut for list --limit N).

    Examples:
        futurnal activity recent
        futurnal activity recent --limit 5 --json
    """
    # Just call list with appropriate defaults
    list_activities(
        limit=limit,
        offset=0,
        event_types=None,
        date_from=None,
        date_to=None,
        output_json=output_json,
    )
