"""Performance benchmark tests for Futurnal.

These tests validate production readiness performance targets:
- Search latency < 1s
- Chat response < 3s
- Ingestion throughput > 5 docs/sec
- Memory usage < 2GB
- Cold start time < 10s

Run with: pytest tests/performance/ -v -m performance
"""
