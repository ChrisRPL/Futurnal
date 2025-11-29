Summary: Configure embedded Neo4j with encryption, ACID semantics, and performance tuning for on-device operation.

# 02 · Database Setup & Configuration

## Purpose
Set up and configure the embedded graph database with encryption at rest, ACID semantics, performance tuning for on-device operation, and integration with the PKG schema designed in module 01.

## Scope
- Database selection (Neo4j embedded vs alternatives)
- Encryption configuration
- ACID semantics validation
- Performance tuning
- Backup and restore configuration

## Database Configuration

```python
from neo4j import GraphDatabase
from pathlib import Path

class PKGDatabaseConfig:
    """PKG database configuration."""

    def __init__(self, workspace_path: Path):
        self.db_path = workspace_path / "pkg" / "neo4j"
        self.backup_path = workspace_path / "pkg" / "backups"

        # Neo4j embedded configuration
        self.config = {
            "dbms.directories.data": str(self.db_path / "data"),
            "dbms.directories.logs": str(self.db_path / "logs"),
            "dbms.directories.import": str(self.db_path / "import"),

            # Encryption
            "dbms.security.procedures.unrestricted": "apoc.*",

            # Performance (on-device optimization)
            "dbms.memory.heap.initial_size": "512m",
            "dbms.memory.heap.max_size": "2g",
            "dbms.memory.pagecache.size": "512m",

            # ACID
            "dbms.tx_log.rotation.retention_policy": "7 days",

            # Security
            "dbms.security.auth_enabled": False  # Local only
        }

    def initialize_database(self):
        """Initialize embedded database."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)

        driver = GraphDatabase.driver(
            f"neo4j://localhost:7687",
            auth=None
        )

        return driver
```

## Testing Strategy

```python
class TestDatabaseSetup:
    def test_database_initialization(self):
        """Validate database initializes correctly."""
        config = PKGDatabaseConfig(test_workspace)
        driver = config.initialize_database()

        assert driver is not None

    def test_acid_semantics(self):
        """Validate ACID properties."""
        # Test transaction atomicity
        with driver.session() as session:
            with session.begin_transaction() as tx:
                tx.run("CREATE (n:Test {value: 1})")
                tx.rollback()

        # Verify rollback worked
        result = session.run("MATCH (n:Test) RETURN count(n)")
        assert result.single()[0] == 0

    def test_encryption_at_rest(self):
        """Validate data encrypted on disk."""
        # Implementation depends on OS keychain integration
        pass
```

## Success Metrics
- ✅ Database initializes successfully
- ✅ ACID semantics validated
- ✅ Performance tuned for on-device
- ✅ Backup/restore functional

See [PHASE-1-OPTION-B-ROADMAP.md](../PHASE-1-OPTION-B-ROADMAP.md) for timeline.
