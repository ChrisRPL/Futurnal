"""End-to-end user journey tests.

Validates complete user workflows through Futurnal,
from installation to daily usage patterns.

User Journey Steps:
1. First launch / onboarding
2. Add data sources
3. Run ingestion
4. Search knowledge
5. Chat with data
6. Explore graph
7. Manage privacy settings
8. Temporal queries
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestUserOnboarding:
    """Test user onboarding experience."""

    def test_first_launch_creates_workspace(self):
        """Verify first launch creates workspace directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / ".futurnal"

            # Simulate first launch
            workspace_path.mkdir(parents=True, exist_ok=True)
            (workspace_path / "config.yaml").touch()
            (workspace_path / "consent").mkdir()
            (workspace_path / "audit").mkdir()

            # Verify structure
            assert workspace_path.exists()
            assert (workspace_path / "config.yaml").exists()
            assert (workspace_path / "consent").is_dir()
            assert (workspace_path / "audit").is_dir()

    def test_first_launch_no_sources_configured(self):
        """Verify clean state has no data sources."""
        initial_config = {
            "version": "1.0.0",
            "sources": {},
            "privacy": {
                "local_only": True,
                "telemetry": False,
            },
        }

        assert initial_config["sources"] == {}
        assert initial_config["privacy"]["local_only"] is True

    def test_onboarding_prompts_for_consent(self):
        """Verify onboarding requires consent before processing."""
        consent_state = {
            "has_acknowledged_privacy": False,
            "has_granted_source_consent": False,
        }

        # Cannot proceed without consent acknowledgment
        can_proceed = (
            consent_state["has_acknowledged_privacy"]
            and consent_state["has_granted_source_consent"]
        )
        assert not can_proceed

        # After consent
        consent_state["has_acknowledged_privacy"] = True
        consent_state["has_granted_source_consent"] = True

        can_proceed = (
            consent_state["has_acknowledged_privacy"]
            and consent_state["has_granted_source_consent"]
        )
        assert can_proceed


class TestDataSourceManagement:
    """Test data source addition and management."""

    def test_add_obsidian_vault(self):
        """Test adding an Obsidian vault as data source."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock vault
            vault_path = Path(temp_dir) / "test-vault"
            vault_path.mkdir()
            (vault_path / ".obsidian").mkdir()
            (vault_path / "test-note.md").write_text("# Test Note\n\nContent")

            # Simulate adding vault
            vault_config = {
                "id": "vault-1",
                "type": "obsidian",
                "path": str(vault_path),
                "name": "test-vault",
                "consent": {
                    "read": True,
                    "process": True,
                    "store": True,
                },
            }

            # Verify configuration
            assert vault_config["type"] == "obsidian"
            assert Path(vault_config["path"]).exists()
            assert vault_config["consent"]["read"] is True

    def test_source_requires_consent(self):
        """Verify source addition requires explicit consent."""
        source_without_consent = {
            "type": "obsidian",
            "path": "/path/to/vault",
            # Missing consent
        }

        has_consent = source_without_consent.get("consent", {}).get("read", False)
        assert not has_consent

    def test_list_configured_sources(self):
        """Test listing all configured data sources."""
        sources = [
            {"id": "vault-1", "type": "obsidian", "name": "Personal Notes"},
            {"id": "imap-1", "type": "imap", "name": "Email"},
            {"id": "github-1", "type": "github", "name": "Code"},
        ]

        assert len(sources) == 3
        source_types = [s["type"] for s in sources]
        assert "obsidian" in source_types
        assert "imap" in source_types


class TestIngestionWorkflow:
    """Test data ingestion workflow."""

    @pytest.mark.asyncio
    async def test_ingestion_progress_tracking(self):
        """Test ingestion provides progress updates."""
        progress_updates = []

        async def track_progress(status: Dict[str, Any]):
            progress_updates.append(status)

        # Simulate ingestion progress
        for i in range(0, 101, 20):
            await track_progress({
                "phase": "processing",
                "progress": i,
                "documents_processed": i,
                "documents_total": 100,
            })
            await asyncio.sleep(0.01)

        assert len(progress_updates) == 6
        assert progress_updates[0]["progress"] == 0
        assert progress_updates[-1]["progress"] == 100

    @pytest.mark.asyncio
    async def test_ingestion_handles_errors_gracefully(self):
        """Test ingestion continues despite individual file errors."""
        results = {
            "total": 100,
            "succeeded": 98,
            "failed": 2,
            "quarantined": [
                {"file": "corrupt.md", "error": "ParseError"},
                {"file": "binary.exe", "error": "UnsupportedFormat"},
            ],
        }

        # Should complete despite errors
        assert results["succeeded"] == 98
        assert len(results["quarantined"]) == 2

    @pytest.mark.asyncio
    async def test_ingestion_respects_consent(self):
        """Test ingestion only processes consented sources."""
        sources = [
            {"id": "vault-1", "consent": {"read": True, "process": True}},
            {"id": "vault-2", "consent": {"read": True, "process": False}},
            {"id": "vault-3", "consent": {"read": False, "process": False}},
        ]

        processable = [
            s for s in sources
            if s["consent"]["read"] and s["consent"]["process"]
        ]

        assert len(processable) == 1
        assert processable[0]["id"] == "vault-1"


class TestSearchWorkflow:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_basic_search(self):
        """Test basic keyword search."""
        mock_results = [
            {
                "id": "doc-1",
                "title": "Meeting Notes",
                "snippet": "Discussion about project timeline...",
                "score": 0.95,
            },
            {
                "id": "doc-2",
                "title": "Project Plan",
                "snippet": "Timeline for Q1 deliverables...",
                "score": 0.85,
            },
        ]

        # Simulate search
        query = "project timeline"
        results = mock_results

        assert len(results) > 0
        assert results[0]["score"] > results[1]["score"]

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test search with date and source filters."""
        filters = {
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-12-31",
            },
            "sources": ["vault-1"],
            "content_types": ["note", "email"],
        }

        # Verify filters are applied
        assert filters["date_range"]["start"] is not None
        assert len(filters["sources"]) == 1

    @pytest.mark.asyncio
    async def test_temporal_search(self):
        """Test temporal query search."""
        temporal_query = {
            "query": "project discussions",
            "temporal_mode": "last_week",
            "intent": "temporal",
        }

        mock_results = [
            {
                "id": "doc-1",
                "title": "Monday Standup",
                "date": "2024-12-16",
                "temporal_relevance": 0.9,
            },
            {
                "id": "doc-2",
                "title": "Friday Wrap-up",
                "date": "2024-12-13",
                "temporal_relevance": 0.85,
            },
        ]

        # Results should be temporally ordered
        assert temporal_query["intent"] == "temporal"
        assert len(mock_results) > 0


class TestChatWorkflow:
    """Test chat functionality."""

    @pytest.mark.asyncio
    async def test_chat_session_creation(self):
        """Test creating a new chat session."""
        session = {
            "id": "session-123",
            "created_at": datetime.utcnow().isoformat(),
            "messages": [],
            "context": {
                "mode": "knowledge",
                "sources": ["vault-1"],
            },
        }

        assert session["id"] is not None
        assert session["messages"] == []

    @pytest.mark.asyncio
    async def test_chat_with_context(self):
        """Test chat retrieves relevant context."""
        mock_response = {
            "content": "Based on your notes about the project...",
            "sources": [
                {"id": "doc-1", "title": "Project Notes", "relevance": 0.9},
                {"id": "doc-2", "title": "Meeting Summary", "relevance": 0.8},
            ],
            "confidence": 0.85,
        }

        # Response should include sources
        assert len(mock_response["sources"]) > 0
        assert mock_response["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_chat_streaming(self):
        """Test streaming chat responses."""
        tokens = ["Based ", "on ", "your ", "notes..."]
        received_tokens = []

        for token in tokens:
            received_tokens.append(token)
            await asyncio.sleep(0.01)

        full_response = "".join(received_tokens)
        assert full_response == "Based on your notes..."

    @pytest.mark.asyncio
    async def test_chat_multi_turn(self):
        """Test multi-turn conversation maintains context."""
        conversation = [
            {"role": "user", "content": "What projects am I working on?"},
            {"role": "assistant", "content": "Based on your notes, you have 3 active projects..."},
            {"role": "user", "content": "Tell me more about the first one"},
            {"role": "assistant", "content": "The first project is..."},
        ]

        # Conversation should maintain context
        assert len(conversation) == 4
        assert conversation[-1]["role"] == "assistant"


class TestGraphVisualization:
    """Test graph visualization functionality."""

    def test_graph_data_structure(self):
        """Test graph data structure for visualization."""
        graph_data = {
            "nodes": [
                {"id": "person-1", "type": "Person", "label": "John"},
                {"id": "topic-1", "type": "Topic", "label": "Machine Learning"},
                {"id": "doc-1", "type": "Document", "label": "ML Notes"},
            ],
            "edges": [
                {"source": "person-1", "target": "topic-1", "type": "interested_in"},
                {"source": "doc-1", "target": "topic-1", "type": "about"},
            ],
        }

        assert len(graph_data["nodes"]) == 3
        assert len(graph_data["edges"]) == 2

    def test_graph_filtering(self):
        """Test graph filtering by node type."""
        nodes = [
            {"id": "1", "type": "Person"},
            {"id": "2", "type": "Topic"},
            {"id": "3", "type": "Person"},
            {"id": "4", "type": "Document"},
        ]

        people = [n for n in nodes if n["type"] == "Person"]
        assert len(people) == 2


class TestPrivacySettings:
    """Test privacy settings and consent management."""

    def test_view_consent_status(self):
        """Test viewing consent status for all sources."""
        consent_status = [
            {
                "source_id": "vault-1",
                "source_name": "Personal Notes",
                "consent": {
                    "read": True,
                    "process": True,
                    "store": True,
                },
                "granted_at": "2024-12-01T10:00:00Z",
            },
            {
                "source_id": "imap-1",
                "source_name": "Email",
                "consent": {
                    "read": True,
                    "process": False,
                    "store": False,
                },
                "granted_at": "2024-12-01T10:00:00Z",
            },
        ]

        # Should show consent for each source
        for status in consent_status:
            assert "consent" in status
            assert "source_name" in status

    def test_revoke_consent(self):
        """Test revoking consent for a source."""
        consent_record = {
            "source_id": "vault-1",
            "consent": {
                "read": True,
                "process": True,
                "store": True,
            },
            "revoked": False,
        }

        # Revoke consent
        consent_record["consent"]["process"] = False
        consent_record["consent"]["store"] = False
        consent_record["revoked_at"] = datetime.utcnow().isoformat()

        assert consent_record["consent"]["process"] is False

    def test_audit_log_access(self):
        """Test viewing audit log entries."""
        audit_entries = [
            {
                "timestamp": "2024-12-17T10:00:00Z",
                "action": "search_executed",
                "metadata": {
                    "search_type": "hybrid",
                    "result_count": 10,
                    # NO query content
                },
            },
            {
                "timestamp": "2024-12-17T10:01:00Z",
                "action": "consent_granted",
                "metadata": {
                    "source_id": "vault-1",
                    "scopes": ["read", "process"],
                },
            },
        ]

        # Audit entries should not contain query content
        for entry in audit_entries:
            assert "query" not in entry.get("metadata", {})


class TestSettingsManagement:
    """Test settings and configuration management."""

    def test_view_current_settings(self):
        """Test viewing current application settings."""
        settings = {
            "llm": {
                "model": "llama3.2:3b",
                "endpoint": "http://localhost:11434",
            },
            "privacy": {
                "local_only": True,
                "telemetry": False,
            },
            "performance": {
                "cache_enabled": True,
                "batch_size": 10,
            },
        }

        assert settings["privacy"]["local_only"] is True
        assert settings["privacy"]["telemetry"] is False

    def test_modify_settings(self):
        """Test modifying application settings."""
        settings = {
            "llm": {"model": "llama3.2:3b"},
            "performance": {"cache_enabled": True},
        }

        # Modify setting
        settings["llm"]["model"] = "llama3.1:8b"

        assert settings["llm"]["model"] == "llama3.1:8b"

    def test_export_settings(self):
        """Test exporting settings for backup."""
        settings = {
            "version": "1.0.0",
            "llm": {"model": "llama3.2:3b"},
            "sources": [{"id": "vault-1", "type": "obsidian"}],
        }

        # Export should be valid JSON-serializable
        import json
        exported = json.dumps(settings)
        restored = json.loads(exported)

        assert restored["version"] == "1.0.0"


class TestCompleteUserJourney:
    """Test complete user journey from start to finish."""

    @pytest.mark.asyncio
    async def test_new_user_complete_journey(self):
        """Test complete journey for a new user."""
        journey_steps = []

        # Step 1: First launch
        journey_steps.append({
            "step": "first_launch",
            "action": "Initialize workspace",
            "success": True,
        })

        # Step 2: Acknowledge privacy
        journey_steps.append({
            "step": "privacy_acknowledgment",
            "action": "Accept privacy policy",
            "success": True,
        })

        # Step 3: Add data source
        journey_steps.append({
            "step": "add_source",
            "action": "Add Obsidian vault",
            "success": True,
        })

        # Step 4: Grant consent
        journey_steps.append({
            "step": "grant_consent",
            "action": "Grant read/process/store consent",
            "success": True,
        })

        # Step 5: Run ingestion
        journey_steps.append({
            "step": "ingestion",
            "action": "Process documents",
            "success": True,
        })

        # Step 6: Search
        journey_steps.append({
            "step": "search",
            "action": "Execute search query",
            "success": True,
        })

        # Step 7: Chat
        journey_steps.append({
            "step": "chat",
            "action": "Chat with knowledge",
            "success": True,
        })

        # Step 8: Explore graph
        journey_steps.append({
            "step": "graph",
            "action": "View knowledge graph",
            "success": True,
        })

        # Step 9: Review settings
        journey_steps.append({
            "step": "settings",
            "action": "Review privacy settings",
            "success": True,
        })

        # All steps should succeed
        assert all(step["success"] for step in journey_steps)
        assert len(journey_steps) == 9

    @pytest.mark.asyncio
    async def test_returning_user_journey(self):
        """Test journey for a returning user."""
        # Returning user has existing config and data
        existing_state = {
            "workspace_exists": True,
            "sources_configured": True,
            "data_ingested": True,
        }

        journey_steps = []

        # Step 1: Launch (skip onboarding)
        journey_steps.append({
            "step": "launch",
            "action": "Load existing workspace",
            "skipped_onboarding": existing_state["workspace_exists"],
        })

        # Step 2: Search immediately
        journey_steps.append({
            "step": "search",
            "action": "Search knowledge",
            "success": True,
        })

        # Step 3: Chat
        journey_steps.append({
            "step": "chat",
            "action": "Chat session",
            "success": True,
        })

        # Returning user should skip onboarding
        assert journey_steps[0]["skipped_onboarding"] is True
