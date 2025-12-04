"""Tests for ACID Semantics Validation.

Validates that Neo4j provides proper ACID guarantees:
- Atomicity: Transactions are all-or-nothing
- Consistency: Constraint violations prevent commits
- Isolation: Concurrent transactions don't interfere
- Durability: Committed data persists after disconnect

Uses testcontainers for real Neo4j instances - no mocks.

Follows production plan testing strategy:
docs/phase-1/pkg-graph-storage-production-plan/02-database-setup.md
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from pydantic import SecretStr

# Import ConstraintError with fallback for stub environment
try:
    from neo4j.exceptions import ConstraintError
except (ImportError, ModuleNotFoundError):
    class ConstraintError(Exception):
        """Stub ConstraintError for testing without neo4j."""
        pass

from futurnal.configuration.settings import StorageSettings
from futurnal.pkg.database.config import PKGDatabaseConfig
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
def pkg_config() -> PKGDatabaseConfig:
    """Create PKGDatabaseConfig for tests."""
    return PKGDatabaseConfig(
        max_connection_retries=2,
        retry_delay_seconds=0.1,
    )


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestAtomicity:
    """Test transaction atomicity (all-or-nothing)."""

    def test_atomicity_rollback(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Transaction rollback undoes ALL operations."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            manager.initialize_schema()

            # Start a transaction, create nodes, then rollback
            with manager.session() as session:
                tx = session.begin_transaction()
                try:
                    tx.run("CREATE (n:Person {id: 'rollback_test', name: 'Will Not Exist'})")
                    tx.run("CREATE (n:Organization {id: 'rollback_org', name: 'Also Gone'})")
                    # Explicit rollback
                    tx.rollback()
                except Exception:
                    tx.rollback()
                    raise

            # Verify nodes were NOT created
            with manager.session() as session:
                result = session.run(
                    "MATCH (n) WHERE n.id IN ['rollback_test', 'rollback_org'] RETURN count(n) as count"
                )
                count = result.single()["count"]
                assert count == 0, "Rolled back nodes should not exist"

    def test_atomicity_commit(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Committed transaction persists all changes."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            manager.initialize_schema()

            # Create nodes in a committed transaction
            with manager.session() as session:
                tx = session.begin_transaction()
                try:
                    tx.run("CREATE (n:Person {id: 'commit_test_1', name: 'Exists 1'})")
                    tx.run("CREATE (n:Person {id: 'commit_test_2', name: 'Exists 2'})")
                    tx.commit()
                except Exception:
                    tx.rollback()
                    raise

            # Verify nodes exist after commit
            with manager.session() as session:
                result = session.run(
                    "MATCH (n:Person) WHERE n.id STARTS WITH 'commit_test' RETURN count(n) as count"
                )
                count = result.single()["count"]
                assert count == 2, "Committed nodes should exist"

    def test_atomicity_partial_failure(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """If any operation in a transaction fails, all are rolled back."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            manager.initialize_schema()

            # First, create a node to cause constraint violation
            with manager.session() as session:
                session.run("CREATE (n:Person {id: 'existing_person', name: 'Original'})")

            # Try to create multiple nodes where one violates constraint
            with manager.session() as session:
                tx = session.begin_transaction()
                try:
                    # This should succeed
                    tx.run("CREATE (n:Person {id: 'new_person_1', name: 'New 1'})")
                    # This will violate unique constraint
                    tx.run("CREATE (n:Person {id: 'existing_person', name: 'Duplicate'})")
                    tx.commit()
                    pytest.fail("Should have raised ConstraintError")
                except ConstraintError:
                    tx.rollback()

            # Verify neither new node exists (atomic rollback)
            with manager.session() as session:
                result = session.run(
                    "MATCH (n:Person {id: 'new_person_1'}) RETURN count(n) as count"
                )
                count = result.single()["count"]
                assert count == 0, "New node should not exist after atomic rollback"


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestConsistency:
    """Test database consistency (constraints enforced)."""

    def test_consistency_constraint_violation(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Constraint violations abort transaction."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            manager.initialize_schema()

            # Create initial Person
            with manager.session() as session:
                session.run("CREATE (n:Person {id: 'person_1', name: 'First Person'})")

            # Try to create duplicate - should fail
            with manager.session() as session:
                with pytest.raises(ConstraintError):
                    session.run("CREATE (n:Person {id: 'person_1', name: 'Duplicate'})")

            # Verify original still exists and unchanged
            with manager.session() as session:
                result = session.run(
                    "MATCH (n:Person {id: 'person_1'}) RETURN n.name as name"
                )
                name = result.single()["name"]
                assert name == "First Person"

    def test_consistency_after_failed_operation(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Database remains consistent after failed operations."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            manager.initialize_schema()

            # Create initial data
            with manager.session() as session:
                session.run(
                    """
                    CREATE (p1:Person {id: 'consistent_1', name: 'Person 1'})
                    CREATE (p2:Person {id: 'consistent_2', name: 'Person 2'})
                    """
                )

            # Attempt various failing operations
            with manager.session() as session:
                try:
                    session.run("CREATE (n:Person {id: 'consistent_1'})")  # Duplicate
                except ConstraintError:
                    pass  # Expected

            # Verify database is still consistent
            with manager.session() as session:
                result = session.run(
                    "MATCH (n:Person) WHERE n.id STARTS WITH 'consistent' RETURN count(n) as count"
                )
                count = result.single()["count"]
                assert count == 2, "Database should have exactly 2 persons"


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestIsolation:
    """Test transaction isolation (concurrent transactions don't interfere)."""

    def test_isolation_concurrent_writes(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Concurrent transactions don't interfere with each other."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            manager.initialize_schema()

            # Use threading to run concurrent transactions
            results = []
            errors = []

            def create_node(node_id: int):
                try:
                    with manager.session() as session:
                        session.run(
                            "CREATE (n:Person {id: $id, name: $name})",
                            id=f"concurrent_{node_id}",
                            name=f"Person {node_id}",
                        )
                    results.append(node_id)
                except Exception as e:
                    errors.append((node_id, e))

            # Create 10 nodes concurrently
            threads = []
            for i in range(10):
                t = threading.Thread(target=create_node, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # All should succeed
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 10

            # Verify all nodes exist
            with manager.session() as session:
                result = session.run(
                    "MATCH (n:Person) WHERE n.id STARTS WITH 'concurrent_' RETURN count(n) as count"
                )
                count = result.single()["count"]
                assert count == 10, "All concurrent nodes should exist"

    def test_isolation_read_committed(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Uncommitted changes are not visible to other transactions."""
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            manager.initialize_schema()

            # Start a transaction but don't commit
            session1 = manager.get_driver().session()
            tx1 = session1.begin_transaction()
            tx1.run("CREATE (n:Person {id: 'uncommitted', name: 'Not Visible'})")
            # tx1 is not committed

            try:
                # Another session should not see the uncommitted node
                with manager.session() as session2:
                    result = session2.run(
                        "MATCH (n:Person {id: 'uncommitted'}) RETURN count(n) as count"
                    )
                    count = result.single()["count"]
                    assert count == 0, "Uncommitted node should not be visible"
            finally:
                tx1.rollback()
                session1.close()


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestDurability:
    """Test data durability (committed data persists)."""

    def test_durability_after_disconnect(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Data persists after connection close/reopen."""
        # First connection - create data
        manager1 = PKGDatabaseManager(storage_settings, pkg_config)
        manager1.connect()
        manager1.initialize_schema()

        with manager1.session() as session:
            session.run(
                "CREATE (n:Person {id: 'durable', name: 'Persists After Disconnect'})"
            )

        # Disconnect
        manager1.disconnect()
        assert not manager1.is_connected

        # Second connection - verify data exists
        manager2 = PKGDatabaseManager(storage_settings, pkg_config)
        manager2.connect()

        try:
            with manager2.session() as session:
                result = session.run(
                    "MATCH (n:Person {id: 'durable'}) RETURN n.name as name"
                )
                record = result.single()
                assert record is not None, "Durable node should exist"
                assert record["name"] == "Persists After Disconnect"
        finally:
            manager2.disconnect()

    def test_durability_after_multiple_operations(
        self, storage_settings, pkg_config, neo4j_container, clean_database
    ):
        """Multiple operations persist correctly."""
        # Create data with multiple operations
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            manager.initialize_schema()

            with manager.session() as session:
                session.run("CREATE (n:Person {id: 'd1', name: 'Person 1'})")
                session.run("CREATE (n:Person {id: 'd2', name: 'Person 2'})")
                session.run(
                    """
                    MATCH (p1:Person {id: 'd1'}), (p2:Person {id: 'd2'})
                    CREATE (p1)-[:KNOWS]->(p2)
                    """
                )

        # Reconnect and verify
        with PKGDatabaseManager(storage_settings, pkg_config) as manager:
            with manager.session() as session:
                result = session.run(
                    """
                    MATCH (p1:Person {id: 'd1'})-[:KNOWS]->(p2:Person {id: 'd2'})
                    RETURN p1.name as name1, p2.name as name2
                    """
                )
                record = result.single()
                assert record is not None, "Relationship should persist"
                assert record["name1"] == "Person 1"
                assert record["name2"] == "Person 2"
