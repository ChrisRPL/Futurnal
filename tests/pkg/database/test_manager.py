"""Tests for PKG Database Manager.

Tests PKGDatabaseManager lifecycle management including:
- Connection establishment and retry
- Schema initialization
- Session context manager
- Health checks
- Disconnect cleanup

Uses testcontainers for real Neo4j instances - no mocks.

Follows production plan testing strategy:
docs/phase-1/pkg-graph-storage-production-plan/02-database-setup.md
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from futurnal.configuration.settings import StorageSettings
from futurnal.pkg.database.config import PKGDatabaseConfig
from futurnal.pkg.database.exceptions import (
    PKGConnectionError,
    PKGSchemaInitializationError,
)
from futurnal.pkg.database.manager import PKGDatabaseManager

# Import test markers from conftest
from tests.pkg.conftest import (
    requires_docker,
    requires_neo4j,
    requires_testcontainers,
)


@pytest.fixture
def storage_settings(neo4j_container) -> StorageSettings:
    """Create StorageSettings from Neo4j test container."""
    return StorageSettings(
        neo4j_uri=neo4j_container.get_connection_url(),
        neo4j_username="neo4j",
        neo4j_password=SecretStr("testpassword"),
        neo4j_encrypted=False,
        chroma_path="/tmp/chroma",
    )


@pytest.fixture
def invalid_storage_settings() -> StorageSettings:
    """Create StorageSettings with invalid credentials."""
    return StorageSettings(
        neo4j_uri="bolt://localhost:9999",  # Non-existent
        neo4j_username="wrong",
        neo4j_password=SecretStr("wrong"),
        neo4j_encrypted=False,
        chroma_path="/tmp/chroma",
    )


@pytest.fixture
def pkg_config() -> PKGDatabaseConfig:
    """Create PKGDatabaseConfig with fast retry settings for tests."""
    return PKGDatabaseConfig(
        max_connection_retries=2,
        retry_delay_seconds=0.1,  # Fast retries for testing
        connection_timeout=5.0,
    )


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestPKGDatabaseManagerConnection:
    """Test database connection management."""

    def test_connect_success(self, storage_settings, pkg_config, neo4j_container):
        """Connection to Neo4j succeeds with valid credentials."""
        manager = PKGDatabaseManager(storage_settings, pkg_config)

        try:
            driver = manager.connect()
            assert driver is not None
            assert manager.is_connected
        finally:
            manager.disconnect()

    def test_context_manager(self, storage_settings, pkg_config, neo4j_container):
        """Context manager connects on enter and disconnects on exit."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            assert manager.is_connected
            driver = manager.get_driver()
            assert driver is not None

        # After exit, should be disconnected
        assert not manager.is_connected
        assert manager.get_driver() is None

    def test_connect_failure_invalid_uri(self, invalid_storage_settings, pkg_config):
        """Connection fails gracefully with invalid URI."""
        manager = PKGDatabaseManager(invalid_storage_settings, pkg_config)

        with pytest.raises(PKGConnectionError) as exc_info:
            manager.connect()

        assert "Failed to connect" in str(exc_info.value)
        assert exc_info.value.attempts == pkg_config.max_connection_retries
        assert not manager.is_connected

    def test_disconnect_when_not_connected(self, storage_settings, pkg_config):
        """Disconnect is safe to call when not connected."""
        manager = PKGDatabaseManager(storage_settings, pkg_config)

        # Should not raise
        manager.disconnect()
        assert not manager.is_connected

    def test_disconnect_cleanup(self, storage_settings, pkg_config, neo4j_container):
        """Disconnect properly closes driver and resources."""
        manager = PKGDatabaseManager(storage_settings, pkg_config)
        manager.connect()
        assert manager.is_connected

        manager.disconnect()
        assert not manager.is_connected
        assert manager.get_driver() is None


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestPKGDatabaseManagerSchema:
    """Test schema initialization."""

    def test_initialize_schema(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Schema initialization creates all constraints/indices."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            results = manager.initialize_schema()

            # Should have created multiple schema elements
            assert len(results) > 0

            # All should be successful
            assert all(results.values()), f"Failed: {[k for k, v in results.items() if not v]}"

    def test_initialize_schema_idempotent(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Schema initialization is idempotent (can run multiple times)."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            results1 = manager.initialize_schema()
            results2 = manager.initialize_schema()

            # Both should succeed
            assert all(results1.values())
            assert all(results2.values())
            # Same number of elements
            assert len(results1) == len(results2)

    def test_initialize_schema_not_connected(self, storage_settings, pkg_config):
        """Schema initialization fails when not connected."""
        manager = PKGDatabaseManager(storage_settings, pkg_config)

        with pytest.raises(PKGConnectionError):
            manager.initialize_schema()


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestPKGDatabaseManagerSession:
    """Test session management."""

    def test_session_context_manager(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Session context manager provides working session."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            with manager.session() as session:
                # Create a test node
                session.run("CREATE (n:TestNode {name: 'test'}) RETURN n")

            # Verify in new session
            with manager.session() as session:
                result = session.run(
                    "MATCH (n:TestNode {name: 'test'}) RETURN n.name as name"
                )
                record = result.single()
                assert record["name"] == "test"

    def test_session_not_connected(self, storage_settings, pkg_config):
        """Session fails when not connected."""
        manager = PKGDatabaseManager(storage_settings, pkg_config)

        with pytest.raises(PKGConnectionError):
            with manager.session():
                pass


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestPKGDatabaseManagerHealthCheck:
    """Test health check functionality."""

    def test_health_check_healthy(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Health check returns (True, message) when healthy."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            manager.initialize_schema()

            is_healthy, message = manager.health_check()
            assert is_healthy is True
            assert "Healthy" in message

    def test_health_check_not_connected(self, storage_settings, pkg_config):
        """Health check returns (False, message) when not connected."""
        manager = PKGDatabaseManager(storage_settings, pkg_config)

        is_healthy, message = manager.health_check()
        assert is_healthy is False
        assert "Not connected" in message


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestPKGDatabaseManagerStatistics:
    """Test database statistics."""

    def test_get_statistics(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """get_statistics returns database statistics."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            manager.initialize_schema()

            # Create some test data
            with manager.session() as session:
                session.run(
                    """
                    CREATE (p:Person {id: 'p1', name: 'Test Person'})
                    CREATE (e:Event {id: 'e1', name: 'Test Event', timestamp: datetime()})
                    """
                )

            stats = manager.get_statistics()

            # Should have node counts
            assert "node_counts" in stats
            assert stats["node_counts"]["Person"] == 1
            assert stats["node_counts"]["Event"] == 1

    def test_get_statistics_not_connected(self, storage_settings, pkg_config):
        """get_statistics fails when not connected."""
        manager = PKGDatabaseManager(storage_settings, pkg_config)

        with pytest.raises(PKGConnectionError):
            manager.get_statistics()
