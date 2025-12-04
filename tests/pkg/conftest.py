"""PKG Test Fixtures.

Provides Neo4j test fixtures using testcontainers for real database behavior.
Follows no-mockups.mdc rule: tests execute against real Neo4j, not mocks.

Usage:
    def test_something(neo4j_driver):
        with neo4j_driver.session() as session:
            session.run("CREATE (n:Test) RETURN n")
"""

from __future__ import annotations

import logging
import os
from typing import Generator

import pytest

logger = logging.getLogger(__name__)

# Check if testcontainers is available
try:
    from testcontainers.neo4j import Neo4jContainer
    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    logger.warning(
        "testcontainers not available. Install with: pip install testcontainers[neo4j]"
    )

# Check if neo4j driver is available
try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning(
        "neo4j driver not available. Install with: pip install neo4j"
    )


# ---------------------------------------------------------------------------
# Skip Markers
# ---------------------------------------------------------------------------


requires_neo4j = pytest.mark.skipif(
    not NEO4J_AVAILABLE,
    reason="neo4j driver not installed"
)

requires_testcontainers = pytest.mark.skipif(
    not TESTCONTAINERS_AVAILABLE,
    reason="testcontainers not installed"
)

requires_docker = pytest.mark.skipif(
    not os.environ.get("DOCKER_HOST") and not os.path.exists("/var/run/docker.sock"),
    reason="Docker not available"
)


# ---------------------------------------------------------------------------
# Neo4j Container Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def neo4j_container() -> Generator:
    """Provide a Neo4j container for the test session.

    Uses testcontainers to spin up a real Neo4j instance.
    The container is shared across all tests in the session for efficiency.

    Yields:
        Neo4jContainer instance with connection details
    """
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers not available")

    # Use Neo4j 5.x community edition
    container = Neo4jContainer("neo4j:5.15.0-community")
    container.with_env("NEO4J_AUTH", "neo4j/testpassword")
    container.with_env("NEO4J_PLUGINS", '["apoc"]')

    try:
        container.start()
        logger.info(f"Started Neo4j container: {container.get_connection_url()}")
        yield container
    finally:
        container.stop()
        logger.info("Stopped Neo4j container")


@pytest.fixture(scope="session")
def neo4j_driver(neo4j_container) -> Generator[Driver, None, None]:
    """Provide a Neo4j driver connected to the test container.

    Args:
        neo4j_container: The Neo4j container fixture

    Yields:
        Neo4j Driver instance
    """
    if not NEO4J_AVAILABLE:
        pytest.skip("neo4j driver not available")

    driver = GraphDatabase.driver(
        neo4j_container.get_connection_url(),
        auth=("neo4j", "testpassword")
    )

    # Verify connection
    driver.verify_connectivity()
    logger.info("Connected to Neo4j test container")

    yield driver

    driver.close()
    logger.info("Closed Neo4j driver")


@pytest.fixture(scope="function")
def neo4j_session(neo4j_driver):
    """Provide a fresh Neo4j session for each test.

    Cleans up all data after each test to ensure isolation.

    Args:
        neo4j_driver: The Neo4j driver fixture

    Yields:
        Neo4j Session instance
    """
    session = neo4j_driver.session()

    yield session

    # Clean up all data after test
    session.run("MATCH (n) DETACH DELETE n")
    session.close()


@pytest.fixture(scope="function")
def clean_database(neo4j_driver):
    """Clean the database before and after each test.

    Use this fixture when you need a completely clean database state.

    Args:
        neo4j_driver: The Neo4j driver fixture
    """
    with neo4j_driver.session() as session:
        # Clean before test
        session.run("MATCH (n) DETACH DELETE n")

    yield

    with neo4j_driver.session() as session:
        # Clean after test
        session.run("MATCH (n) DETACH DELETE n")


# ---------------------------------------------------------------------------
# Schema Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def initialized_schema(neo4j_driver, clean_database):
    """Provide a database with initialized PKG schema.

    Creates all constraints and indices before the test.

    Args:
        neo4j_driver: The Neo4j driver fixture
        clean_database: Ensures clean state

    Yields:
        The Neo4j driver with schema initialized
    """
    from futurnal.pkg.schema.constraints import init_schema

    init_schema(neo4j_driver)
    yield neo4j_driver


@pytest.fixture(scope="function")
def schema_version_manager(neo4j_driver, clean_database):
    """Provide a SchemaVersionManager instance.

    Args:
        neo4j_driver: The Neo4j driver fixture
        clean_database: Ensures clean state

    Yields:
        SchemaVersionManager instance
    """
    from futurnal.pkg.schema.migration import SchemaVersionManager

    manager = SchemaVersionManager(neo4j_driver)
    yield manager


# ---------------------------------------------------------------------------
# Model Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_person():
    """Provide a sample PersonNode for testing."""
    from futurnal.pkg.schema.models import PersonNode

    return PersonNode(
        name="John Doe",
        aliases=["Johnny", "JD"],
        discovery_count=5,
        confidence=0.95,
        first_seen_document="doc_123",
    )


@pytest.fixture
def sample_organization():
    """Provide a sample OrganizationNode for testing."""
    from futurnal.pkg.schema.models import OrganizationNode

    return OrganizationNode(
        name="Acme Corp",
        type="company",
        aliases=["Acme", "ACME Corporation"],
        confidence=0.9,
    )


@pytest.fixture
def sample_concept():
    """Provide a sample ConceptNode for testing."""
    from futurnal.pkg.schema.models import ConceptNode

    return ConceptNode(
        name="Machine Learning",
        description="A branch of artificial intelligence",
        category="field",
        aliases=["ML", "machine learning"],
        confidence=1.0,
    )


@pytest.fixture
def sample_document():
    """Provide a sample DocumentNode for testing."""
    from futurnal.pkg.schema.models import DocumentNode
    from datetime import datetime

    return DocumentNode(
        source_id="vault/notes/test.md",
        source_type="obsidian_vault",
        content_hash="abc123def456",
        format="markdown",
        modified_at=datetime(2024, 1, 15, 10, 30, 0),
    )


@pytest.fixture
def sample_event():
    """Provide a sample EventNode for testing."""
    from futurnal.pkg.schema.models import EventNode
    from datetime import datetime, timedelta

    return EventNode(
        name="Team Meeting",
        event_type="meeting",
        description="Weekly sync meeting",
        timestamp=datetime(2024, 1, 15, 14, 0, 0),
        duration=timedelta(hours=1),
        location="Conference Room A",
        source_document="doc_456",
        extraction_method="explicit",
    )


@pytest.fixture
def sample_events_pair():
    """Provide a pair of EventNodes with temporal ordering."""
    from futurnal.pkg.schema.models import EventNode
    from datetime import datetime, timedelta

    event1 = EventNode(
        name="Planning Meeting",
        event_type="meeting",
        description="Initial planning",
        timestamp=datetime(2024, 1, 15, 9, 0, 0),
        duration=timedelta(hours=1),
        source_document="doc_001",
    )

    event2 = EventNode(
        name="Decision Made",
        event_type="decision",
        description="Final decision",
        timestamp=datetime(2024, 1, 15, 16, 0, 0),
        source_document="doc_002",
    )

    return event1, event2
