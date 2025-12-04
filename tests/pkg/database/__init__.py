"""PKG Database Module Tests.

Tests for PKG database setup, configuration, and lifecycle management.

Test modules:
- test_config.py: Configuration validation and environment overrides
- test_manager.py: Database lifecycle and connection management
- test_acid.py: ACID semantics validation
- test_backup.py: Backup and restore operations

Uses testcontainers for real Neo4j instances - no mocks.
"""
