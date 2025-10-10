"""Test fixtures for normalization pipeline quality gates testing.

This package provides comprehensive test data including:
- Format samples for all 16 DocumentFormat types
- Edge case data (empty, large, corrupted, unicode)
- Utilities for generating test data
"""

from .format_samples import *  # noqa: F401, F403
from .edge_case_data import *  # noqa: F401, F403
