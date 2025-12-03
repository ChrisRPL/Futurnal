"""Real test corpus with ground truth labels for extraction validation.

This module provides a comprehensive test corpus with manually curated
ground truth labels for validating extraction accuracy WITHOUT mocks.

Test corpus includes:
- Temporal markers and relationships
- Entities (Person, Organization, Concept)
- Events and causal relationships
- Multi-document scenarios for learning validation
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class GroundTruthEntity:
    """Ground truth entity annotation."""
    text: str
    entity_type: str  # Person, Organization, Concept, Event
    start_char: int
    end_char: int
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruthRelationship:
    """Ground truth relationship annotation."""
    subject: str
    predicate: str
    object: str
    relationship_type: str
    confidence: float = 1.0  # Ground truth is 100% confident
    temporal_type: Optional[str] = None  # BEFORE, AFTER, DURING, etc.


@dataclass
class GroundTruthTemporal:
    """Ground truth temporal marker annotation."""
    text: str
    timestamp: Optional[datetime]
    start_char: int
    end_char: int
    temporal_type: str  # explicit, relative, metadata
    confidence: float = 1.0


@dataclass
class GroundTruthEvent:
    """Ground truth event annotation."""
    text: str
    event_type: str  # meeting, decision, publication, etc.
    timestamp: datetime
    participants: List[str]
    confidence: float = 1.0


@dataclass
class TestDocument:
    """Test document with ground truth labels."""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    entities: List[GroundTruthEntity]
    relationships: List[GroundTruthRelationship]
    temporal_markers: List[GroundTruthTemporal]
    events: List[GroundTruthEvent]
    difficulty: str = "medium"  # easy, medium, hard
    domain: str = "general"  # general, technical, medical, etc.


# ==============================================================================
# TEST CORPUS: Temporal Extraction
# ==============================================================================

TEMPORAL_TEST_CORPUS = [
    TestDocument(
        doc_id="temp_001",
        content="""# Project Meeting Notes

On January 15, 2024, we held our quarterly planning meeting. The team discussed
the Q1 2024 roadmap and set key milestones.

## Action Items
- Review proposal (deadline: 2024-01-20 at 2:30 PM)
- Schedule follow-up meeting for next week
- Send summary email by tomorrow evening

## Timeline
After reviewing the proposal, we'll make a decision. The implementation will
begin in 2 weeks and should be completed by March 15, 2024.""",
        metadata={
            "created": "2024-01-15T14:00:00Z",
            "modified": "2024-01-15T16:30:00Z"
        },
        entities=[
            GroundTruthEntity("team", "Organization", 77, 81),
            GroundTruthEntity("proposal", "Concept", 190, 198),
        ],
        relationships=[
            GroundTruthRelationship(
                "team", "discussed", "Q1 2024 roadmap",
                relationship_type="Action",
                temporal_type="DURING"
            ),
        ],
        temporal_markers=[
            GroundTruthTemporal(
                "January 15, 2024",
                datetime(2024, 1, 15),
                11, 28,
                "explicit"
            ),
            GroundTruthTemporal(
                "2024-01-20 at 2:30 PM",
                datetime(2024, 1, 20, 14, 30),
                211, 232,
                "explicit"
            ),
            GroundTruthTemporal(
                "next week",
                datetime(2024, 1, 22),  # Approximate
                268, 277,
                "relative"
            ),
            GroundTruthTemporal(
                "tomorrow evening",
                datetime(2024, 1, 16, 18, 0),  # Approximate
                302, 318,
                "relative"
            ),
            GroundTruthTemporal(
                "in 2 weeks",
                datetime(2024, 1, 29),
                417, 427,
                "relative"
            ),
            GroundTruthTemporal(
                "March 15, 2024",
                datetime(2024, 3, 15),
                453, 467,
                "explicit"
            ),
        ],
        events=[
            GroundTruthEvent(
                "quarterly planning meeting",
                "meeting",
                datetime(2024, 1, 15, 14, 0),
                ["team"],
            ),
        ],
        difficulty="easy"
    ),

    TestDocument(
        doc_id="temp_002",
        content="""Research Meeting - NLP Progress

Yesterday we made significant progress on the entity recognition model.
The team trained a new version that improved F1 score from 0.82 to 0.87.

Three months ago, we started this project. Last week, we completed the
data annotation phase. Now we're ready to move into production.

The paper was published on 2023-12-01. Our implementation followed
two weeks later. The results will be presented at the conference
next month.""",
        metadata={
            "created": "2024-01-16T09:00:00Z",
            "author": "Research Team"
        },
        entities=[
            GroundTruthEntity("entity recognition model", "Concept", 83, 109),
            GroundTruthEntity("team", "Organization", 115, 119),
        ],
        relationships=[],
        temporal_markers=[
            GroundTruthTemporal(
                "Yesterday",
                datetime(2024, 1, 15),  # Relative to creation date
                29, 38,
                "relative"
            ),
            GroundTruthTemporal(
                "Three months ago",
                datetime(2023, 10, 16),  # 3 months before Jan 16
                198, 214,
                "relative"
            ),
            GroundTruthTemporal(
                "Last week",
                datetime(2024, 1, 9),  # Week before Jan 16
                242, 251,
                "relative"
            ),
            GroundTruthTemporal(
                "2023-12-01",
                datetime(2023, 12, 1),
                351, 361,
                "explicit"
            ),
            GroundTruthTemporal(
                "two weeks later",
                datetime(2023, 12, 15),  # 2 weeks after paper
                392, 407,
                "relative"
            ),
            GroundTruthTemporal(
                "next month",
                datetime(2024, 2, 16),  # Month after creation
                458, 468,
                "relative"
            ),
        ],
        events=[
            GroundTruthEvent(
                "Research Meeting - NLP Progress",
                "meeting",
                datetime(2024, 1, 16, 9, 0),
                ["Research Team"],
            ),
            GroundTruthEvent(
                "paper was published",
                "publication",
                datetime(2023, 12, 1),
                [],
            ),
        ],
        difficulty="medium"
    ),

    TestDocument(
        doc_id="temp_003",
        content="""# Email Thread: Budget Approval

From: john.doe@company.com
Date: Fri, 12 Jan 2024 08:30:00 +0000
Subject: Budget Q1 2024

Hi team,

Following our meeting last Tuesday (Jan 9th), I'm sending the revised
budget proposal. Please review by Monday morning (Jan 15th at 9 AM).

The board meeting is scheduled for Wednesday, January 17, 2024 at 2:00 PM.
We need approval before the end of this month to start Q1 initiatives.

Best,
John""",
        metadata={
            "email_date": "Fri, 12 Jan 2024 08:30:00 +0000",
            "from": "john.doe@company.com",
            "subject": "Budget Q1 2024"
        },
        entities=[
            GroundTruthEntity("john.doe@company.com", "Person", 29, 51),
            GroundTruthEntity("team", "Organization", 109, 113),
            GroundTruthEntity("budget proposal", "Concept", 186, 201),
            GroundTruthEntity("board", "Organization", 270, 275),
        ],
        relationships=[],
        temporal_markers=[
            GroundTruthTemporal(
                "Fri, 12 Jan 2024 08:30:00 +0000",
                datetime(2024, 1, 12, 8, 30),
                59, 90,
                "explicit"
            ),
            GroundTruthTemporal(
                "last Tuesday (Jan 9th)",
                datetime(2024, 1, 9),
                129, 151,
                "relative"
            ),
            GroundTruthTemporal(
                "Monday morning (Jan 15th at 9 AM)",
                datetime(2024, 1, 15, 9, 0),
                222, 255,
                "explicit"
            ),
            GroundTruthTemporal(
                "Wednesday, January 17, 2024 at 2:00 PM",
                datetime(2024, 1, 17, 14, 0),
                295, 334,
                "explicit"
            ),
            GroundTruthTemporal(
                "end of this month",
                datetime(2024, 1, 31),
                362, 379,
                "relative"
            ),
        ],
        events=[
            GroundTruthEvent(
                "meeting last Tuesday",
                "meeting",
                datetime(2024, 1, 9),
                ["team", "john.doe@company.com"],
            ),
            GroundTruthEvent(
                "board meeting",
                "meeting",
                datetime(2024, 1, 17, 14, 0),
                ["board"],
            ),
        ],
        difficulty="medium"
    ),
]


# ==============================================================================
# TEST CORPUS: Entity-Relationship Extraction
# ==============================================================================

ENTITY_RELATIONSHIP_TEST_CORPUS = [
    TestDocument(
        doc_id="er_001",
        content="""# Team Structure

Alice Johnson is the Engineering Manager at TechCorp. She leads a team of
15 engineers working on the CloudPlatform project. Bob Smith, the CTO,
oversees all technical initiatives.

Alice reports to Bob. The team uses Agile methodology and meets daily
for standups. They recently adopted Kubernetes for container orchestration.""",
        metadata={"domain": "organizational"},
        entities=[
            GroundTruthEntity("Alice Johnson", "Person", 18, 31),
            GroundTruthEntity("Engineering Manager", "Concept", 39, 58),
            GroundTruthEntity("TechCorp", "Organization", 62, 70),
            GroundTruthEntity("CloudPlatform", "Concept", 120, 133),
            GroundTruthEntity("Bob Smith", "Person", 144, 153),
            GroundTruthEntity("CTO", "Concept", 159, 162),
            GroundTruthEntity("Agile methodology", "Concept", 236, 253),
            GroundTruthEntity("Kubernetes", "Concept", 312, 322),
        ],
        relationships=[
            GroundTruthRelationship(
                "Alice Johnson", "works_at", "TechCorp",
                "Employment"
            ),
            GroundTruthRelationship(
                "Alice Johnson", "has_role", "Engineering Manager",
                "Role"
            ),
            GroundTruthRelationship(
                "Alice Johnson", "leads", "team",
                "Management"
            ),
            GroundTruthRelationship(
                "Bob Smith", "has_role", "CTO",
                "Role"
            ),
            GroundTruthRelationship(
                "Alice Johnson", "reports_to", "Bob Smith",
                "Management"
            ),
            GroundTruthRelationship(
                "team", "uses", "Agile methodology",
                "Practice"
            ),
            GroundTruthRelationship(
                "team", "adopted", "Kubernetes",
                "Technology"
            ),
        ],
        temporal_markers=[],
        events=[],
        difficulty="easy"
    ),

    TestDocument(
        doc_id="er_002",
        content="""Research Collaboration

Dr. Maria Garcia from Stanford University collaborated with Prof. James Chen
at MIT on breakthrough research in quantum computing. Their paper, "Quantum
Error Correction Using Topological Codes", was published in Nature Physics.

The research was funded by the National Science Foundation (NSF) and received
the Best Paper Award at the International Conference on Quantum Computing 2023.

Dr. Garcia's team at the Stanford Quantum Lab includes 8 PhD students and
2 postdoctoral researchers. Prof. Chen directs the MIT Quantum Information
Processing Group.""",
        metadata={"domain": "academic"},
        entities=[
            GroundTruthEntity("Dr. Maria Garcia", "Person", 25, 41),
            GroundTruthEntity("Stanford University", "Organization", 47, 66),
            GroundTruthEntity("Prof. James Chen", "Person", 84, 100),
            GroundTruthEntity("MIT", "Organization", 104, 107),
            GroundTruthEntity("quantum computing", "Concept", 135, 152),
            GroundTruthEntity("Quantum Error Correction Using Topological Codes", "Concept", 172, 220),
            GroundTruthEntity("Nature Physics", "Organization", 241, 255),
            GroundTruthEntity("National Science Foundation", "Organization", 286, 314),
            GroundTruthEntity("NSF", "Organization", 316, 319),
            GroundTruthEntity("Stanford Quantum Lab", "Organization", 445, 465),
            GroundTruthEntity("MIT Quantum Information Processing Group", "Organization", 540, 580),
        ],
        relationships=[
            GroundTruthRelationship(
                "Dr. Maria Garcia", "affiliated_with", "Stanford University",
                "Affiliation"
            ),
            GroundTruthRelationship(
                "Prof. James Chen", "affiliated_with", "MIT",
                "Affiliation"
            ),
            GroundTruthRelationship(
                "Dr. Maria Garcia", "collaborated_with", "Prof. James Chen",
                "Collaboration"
            ),
            GroundTruthRelationship(
                "Dr. Maria Garcia", "authored", "Quantum Error Correction Using Topological Codes",
                "Authorship"
            ),
            GroundTruthRelationship(
                "Prof. James Chen", "authored", "Quantum Error Correction Using Topological Codes",
                "Authorship"
            ),
            GroundTruthRelationship(
                "Quantum Error Correction Using Topological Codes", "published_in", "Nature Physics",
                "Publication"
            ),
            GroundTruthRelationship(
                "research", "funded_by", "National Science Foundation",
                "Funding"
            ),
            GroundTruthRelationship(
                "Dr. Maria Garcia", "leads", "Stanford Quantum Lab",
                "Management"
            ),
            GroundTruthRelationship(
                "Prof. James Chen", "directs", "MIT Quantum Information Processing Group",
                "Management"
            ),
        ],
        temporal_markers=[],
        events=[
            GroundTruthEvent(
                "paper published",
                "publication",
                datetime(2023, 1, 1),  # Approximate
                ["Dr. Maria Garcia", "Prof. James Chen"],
            ),
        ],
        difficulty="hard"
    ),
]


# ==============================================================================
# TEST CORPUS: Causal Relationships
# ==============================================================================

CAUSAL_TEST_CORPUS = [
    TestDocument(
        doc_id="causal_001",
        content="""Project Timeline Analysis

On March 1st, 2024, we decided to adopt microservices architecture. This decision
led to a complete redesign of the system, which began on March 15th, 2024.

The redesign caused significant delays in the release schedule. As a result,
we had to push the launch date from May 1st to July 15th, 2024.

However, the new architecture enabled better scalability, which triggered
increased customer adoption starting August 2024.""",
        metadata={},
        entities=[],
        relationships=[
            GroundTruthRelationship(
                "adopt microservices", "CAUSES", "complete redesign",
                "Causal",
                temporal_type="CAUSES"
            ),
            GroundTruthRelationship(
                "redesign", "CAUSES", "delays in release",
                "Causal",
                temporal_type="CAUSES"
            ),
            GroundTruthRelationship(
                "new architecture", "ENABLES", "better scalability",
                "Causal",
                temporal_type="ENABLES"
            ),
            GroundTruthRelationship(
                "better scalability", "TRIGGERS", "increased adoption",
                "Causal",
                temporal_type="TRIGGERS"
            ),
        ],
        temporal_markers=[
            GroundTruthTemporal(
                "March 1st, 2024",
                datetime(2024, 3, 1),
                31, 47,
                "explicit"
            ),
            GroundTruthTemporal(
                "March 15th, 2024",
                datetime(2024, 3, 15),
                148, 164,
                "explicit"
            ),
            GroundTruthTemporal(
                "May 1st",
                datetime(2024, 5, 1),
                260, 267,
                "explicit"
            ),
            GroundTruthTemporal(
                "July 15th, 2024",
                datetime(2024, 7, 15),
                271, 286,
                "explicit"
            ),
            GroundTruthTemporal(
                "August 2024",
                datetime(2024, 8, 1),
                390, 401,
                "explicit"
            ),
        ],
        events=[
            GroundTruthEvent(
                "decided to adopt microservices",
                "decision",
                datetime(2024, 3, 1),
                [],
            ),
            GroundTruthEvent(
                "complete redesign began",
                "action",
                datetime(2024, 3, 15),
                [],
            ),
            GroundTruthEvent(
                "increased customer adoption",
                "state_change",
                datetime(2024, 8, 1),
                [],
            ),
        ],
        difficulty="medium"
    ),
]


# ==============================================================================
# Corpus Loading Utilities
# ==============================================================================

def load_corpus(corpus_name: str) -> List[TestDocument]:
    """Load test corpus by name.

    Args:
        corpus_name: One of "temporal", "entity_relationship", "causal", "all"

    Returns:
        List of test documents with ground truth
    """
    if corpus_name == "temporal":
        return TEMPORAL_TEST_CORPUS
    elif corpus_name == "entity_relationship":
        return ENTITY_RELATIONSHIP_TEST_CORPUS
    elif corpus_name == "causal":
        return CAUSAL_TEST_CORPUS
    elif corpus_name == "all":
        return (
            TEMPORAL_TEST_CORPUS +
            ENTITY_RELATIONSHIP_TEST_CORPUS +
            CAUSAL_TEST_CORPUS
        )
    else:
        raise ValueError(f"Unknown corpus: {corpus_name}")


def filter_corpus(
    corpus: List[TestDocument],
    difficulty: Optional[str] = None,
    domain: Optional[str] = None
) -> List[TestDocument]:
    """Filter corpus by difficulty and/or domain.

    Args:
        corpus: Test corpus to filter
        difficulty: Filter by difficulty (easy, medium, hard)
        domain: Filter by domain (general, technical, academic, etc.)

    Returns:
        Filtered corpus
    """
    filtered = corpus

    if difficulty:
        filtered = [doc for doc in filtered if doc.difficulty == difficulty]

    if domain:
        filtered = [doc for doc in filtered if doc.domain == domain]

    return filtered


def get_corpus_stats(corpus: List[TestDocument]) -> Dict[str, Any]:
    """Get statistics about test corpus.

    Returns:
        Dict with counts of documents, entities, relationships, etc.
    """
    return {
        "num_documents": len(corpus),
        "num_entities": sum(len(doc.entities) for doc in corpus),
        "num_relationships": sum(len(doc.relationships) for doc in corpus),
        "num_temporal_markers": sum(len(doc.temporal_markers) for doc in corpus),
        "num_events": sum(len(doc.events) for doc in corpus),
        "difficulties": {
            d: len([doc for doc in corpus if doc.difficulty == d])
            for d in ["easy", "medium", "hard"]
        },
        "domains": list(set(doc.domain for doc in corpus)),
    }
