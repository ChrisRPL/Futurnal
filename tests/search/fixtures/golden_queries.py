"""Golden Query Set for Hybrid Search Relevance Testing.

Provides curated test queries with expected results for:
- Relevance metrics validation (MRR, precision@5)
- Multimodal content testing (OCR, audio)
- Query type categorization (temporal, causal, exploratory, factual)

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/06-integration-testing.md

Option B Compliance:
- Real test data, no mockups
- Covers temporal, causal, exploratory, factual query types
- Includes multimodal modality filters
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GoldenQuery:
    """A curated test query with expected results."""

    query: str
    expected_id: str
    expected_ids: List[str] = field(default_factory=list)
    query_type: str = "general"  # temporal, causal, exploratory, factual, code
    modality: Optional[str] = None  # ocr, audio, text, None for any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure expected_ids includes expected_id."""
        if self.expected_id and self.expected_id not in self.expected_ids:
            self.expected_ids = [self.expected_id] + self.expected_ids


# ---------------------------------------------------------------------------
# Golden Query Sets
# ---------------------------------------------------------------------------

TEMPORAL_QUERIES: List[GoldenQuery] = [
    GoldenQuery(
        query="What happened between January and March 2024?",
        expected_id="event_q1_review_2024",
        expected_ids=["event_q1_review_2024", "event_project_kickoff_2024"],
        query_type="temporal",
        metadata={"time_range": "2024-01/2024-03"},
    ),
    GoldenQuery(
        query="What happened last week?",
        expected_id="event_recent_meeting",
        query_type="temporal",
    ),
    GoldenQuery(
        query="What meetings did I have in January?",
        expected_id="event_jan_meeting_1",
        expected_ids=["event_jan_meeting_1", "event_jan_meeting_2"],
        query_type="temporal",
    ),
    GoldenQuery(
        query="Events from yesterday",
        expected_id="event_yesterday_1",
        query_type="temporal",
    ),
    GoldenQuery(
        query="What happened after the project launch?",
        expected_id="event_post_launch_review",
        query_type="temporal",
    ),
]

CAUSAL_QUERIES: List[GoldenQuery] = [
    GoldenQuery(
        query="What led to the product launch decision?",
        expected_id="decision_product_launch",
        expected_ids=["decision_product_launch", "event_market_analysis"],
        query_type="causal",
    ),
    GoldenQuery(
        query="Why did the server crash?",
        expected_id="event_server_crash",
        expected_ids=["event_server_crash", "event_memory_leak"],
        query_type="causal",
    ),
    GoldenQuery(
        query="What caused the delay in delivery?",
        expected_id="event_delivery_delay",
        query_type="causal",
    ),
    GoldenQuery(
        query="What were the consequences of the budget cut?",
        expected_id="decision_budget_cut",
        expected_ids=["decision_budget_cut", "event_team_reduction"],
        query_type="causal",
    ),
]

EXPLORATORY_QUERIES: List[GoldenQuery] = [
    GoldenQuery(
        query="Tell me about machine learning projects",
        expected_id="project_ml_classifier",
        expected_ids=["project_ml_classifier", "project_nlp_pipeline", "doc_ml_guide"],
        query_type="exploratory",
    ),
    GoldenQuery(
        query="What do we know about customer feedback?",
        expected_id="doc_customer_feedback_summary",
        query_type="exploratory",
    ),
    GoldenQuery(
        query="Information about the authentication system",
        expected_id="doc_auth_architecture",
        expected_ids=["doc_auth_architecture", "code_auth_module"],
        query_type="exploratory",
    ),
]

FACTUAL_QUERIES: List[GoldenQuery] = [
    GoldenQuery(
        query="What is the project deadline for Alpha?",
        expected_id="project_alpha",
        query_type="factual",
        metadata={"expected_confidence": 0.85},
    ),
    GoldenQuery(
        query="Who is the project manager?",
        expected_id="person_pm_john",
        query_type="factual",
    ),
    GoldenQuery(
        query="What is the database connection string?",
        expected_id="config_db_connection",
        query_type="factual",
    ),
]

CODE_QUERIES: List[GoldenQuery] = [
    GoldenQuery(
        query="How does the authentication module work?",
        expected_id="code_auth_module",
        expected_ids=["code_auth_module", "code_jwt_handler"],
        query_type="code",
    ),
    GoldenQuery(
        query="def authenticate_user(token):",
        expected_id="code_auth_function",
        query_type="code",
    ),
    GoldenQuery(
        query="function calculateTotal(items)",
        expected_id="code_calc_function",
        query_type="code",
    ),
]

OCR_QUERIES: List[GoldenQuery] = [
    GoldenQuery(
        query="invoice from the PDF",
        expected_id="ocr_invoice_12345",
        expected_ids=["ocr_invoice_12345", "ocr_receipt_001"],
        query_type="factual",
        modality="ocr",
    ),
    GoldenQuery(
        query="receipt from the document",
        expected_id="ocr_receipt_001",
        query_type="factual",
        modality="ocr",
    ),
    GoldenQuery(
        query="in the scanned document about contracts",
        expected_id="ocr_contract_draft",
        query_type="exploratory",
        modality="ocr",
    ),
    GoldenQuery(
        query="handwritten notes from the meeting",
        expected_id="ocr_handwritten_notes",
        query_type="temporal",
        modality="ocr",
    ),
    GoldenQuery(
        query="text in that screenshot",
        expected_id="ocr_screenshot_error",
        query_type="factual",
        modality="ocr",
    ),
]

AUDIO_QUERIES: List[GoldenQuery] = [
    GoldenQuery(
        query="what did we discuss in the meeting about revenue",
        expected_id="audio_meeting_q4_revenue",
        expected_ids=["audio_meeting_q4_revenue", "audio_call_budget"],
        query_type="exploratory",
        modality="audio",
    ),
    GoldenQuery(
        query="in my voice notes about the project",
        expected_id="audio_voice_note_project",
        query_type="exploratory",
        modality="audio",
    ),
    GoldenQuery(
        query="what I said about the deadline",
        expected_id="audio_deadline_discussion",
        query_type="temporal",
        modality="audio",
    ),
    GoldenQuery(
        query="from the interview recording",
        expected_id="audio_interview_candidate",
        query_type="exploratory",
        modality="audio",
    ),
]


# ---------------------------------------------------------------------------
# Query Set Loading Functions
# ---------------------------------------------------------------------------


def load_golden_query_set(
    modality: Optional[str] = None,
    query_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load golden query set with optional filtering.

    Args:
        modality: Filter by content modality (ocr, audio, text, None for all)
        query_type: Filter by query type (temporal, causal, exploratory, factual, code)

    Returns:
        List of query dicts with 'query', 'expected_id', 'expected_ids' keys
    """
    # Select base query sets
    if modality == "ocr":
        queries = OCR_QUERIES
    elif modality == "audio":
        queries = AUDIO_QUERIES
    else:
        # All queries
        queries = (
            TEMPORAL_QUERIES
            + CAUSAL_QUERIES
            + EXPLORATORY_QUERIES
            + FACTUAL_QUERIES
            + CODE_QUERIES
        )

    # Filter by query type if specified
    if query_type:
        queries = [q for q in queries if q.query_type == query_type]

    # Convert to dict format for test compatibility
    return [
        {
            "query": q.query,
            "expected_id": q.expected_id,
            "expected_ids": q.expected_ids,
            "query_type": q.query_type,
            "modality": q.modality,
            "metadata": q.metadata,
        }
        for q in queries
    ]


def generate_benchmark_queries(n: int = 100) -> List[str]:
    """Generate queries for performance benchmarking.

    Args:
        n: Number of queries to generate

    Returns:
        List of query strings for latency testing
    """
    base_queries = [
        "What happened yesterday?",
        "Tell me about projects",
        "Why did this fail?",
        "What is the deadline?",
        "How does authentication work?",
        "Meeting notes from last week",
        "Find documents about budgets",
        "What caused the issue?",
        "Show me the code for login",
        "Events in January",
        "Project status update",
        "Who is responsible for this?",
        "What are the next steps?",
        "Find invoices from Q4",
        "Search notes about clients",
    ]

    # Generate n queries by cycling through base queries with variations
    queries = []
    for i in range(n):
        base = base_queries[i % len(base_queries)]
        # Add slight variations to avoid pure cache hits
        if i >= len(base_queries):
            base = f"{base} (variant {i // len(base_queries)})"
        queries.append(base)

    return queries


def get_all_golden_queries() -> List[GoldenQuery]:
    """Get all golden queries as GoldenQuery objects."""
    return (
        TEMPORAL_QUERIES
        + CAUSAL_QUERIES
        + EXPLORATORY_QUERIES
        + FACTUAL_QUERIES
        + CODE_QUERIES
        + OCR_QUERIES
        + AUDIO_QUERIES
    )


def get_query_type_distribution() -> Dict[str, int]:
    """Get distribution of queries by type."""
    all_queries = get_all_golden_queries()
    distribution: Dict[str, int] = {}
    for q in all_queries:
        distribution[q.query_type] = distribution.get(q.query_type, 0) + 1
    return distribution
