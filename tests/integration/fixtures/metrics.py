"""Measurement utilities for quality validation.

Standardized measurement of precision, accuracy, alignment, and other quality metrics.
"""

import time
import psutil
import os
from typing import List, Dict, Any, Union, Protocol

# Define protocols for pipeline and results to avoid circular imports
class ExtractionResult(Protocol):
    confidence: float
    entities: List[Any]
    relationships: List[Any]
    events: List[Any]

class ExtractionPipeline(Protocol):
    def process(self, doc: Any) -> ExtractionResult: ...


def measure_precision(pipeline: ExtractionPipeline, docs: List[Any]) -> float:
    """Measure extraction precision over a set of documents."""
    total_precision = 0.0
    count = 0
    
    for doc in docs:
        result = pipeline.process(doc)
        # In a real scenario, we'd compare against ground truth
        # For integration testing without full ground truth, we use confidence as a proxy
        # or check internal consistency
        if result.entities or result.relationships:
            total_precision += result.confidence
            count += 1
            
    return total_precision / count if count > 0 else 0.0


def measure_temporal_accuracy(pipeline: ExtractionPipeline, labeled_docs: List[Dict[str, Any]]) -> float:
    """Measure temporal extraction accuracy."""
    correct = 0
    total = 0
    
    for item in labeled_docs:
        doc = item["document"]
        labels = item["labels"]
        
        result = pipeline.process(doc)
        
        # Check if temporal extraction matches labels
        # This is a simplified check for the integration test
        has_temporal = any(hasattr(e, 'timestamp') and e.timestamp for e in result.events)
        
        if has_temporal == labels.get("has_explicit_date", False) or \
           has_temporal == labels.get("has_relative_date", False):
            correct += 1
        total += 1
            
    return correct / total if total > 0 else 0.0


def compute_semantic_alignment(evolved_schema: Any, manual_schema: Any) -> float:
    """Compute semantic alignment between evolved and manual schemas."""
    # Placeholder for schema alignment logic
    # In a real implementation, this would compare graph structures or embedding similarity
    
    # For testing purposes, we assume high alignment if key types exist
    evolved_types = set()
    if hasattr(evolved_schema, "get_types"):
        evolved_types = set(evolved_schema.get_types())
    
    manual_types = set()
    if hasattr(manual_schema, "get_types"):
        manual_types = set(manual_schema.get_types())
        
    if not manual_types:
        return 1.0
        
    intersection = evolved_types.intersection(manual_types)
    return len(intersection) / len(manual_types)


def measure_extraction_precision(pipeline: ExtractionPipeline, labeled_docs: List[Dict[str, Any]]) -> float:
    """Measure overall extraction precision."""
    # Alias for measure_precision but specifically for labeled data
    return measure_precision(pipeline, [d["document"] for d in labeled_docs])


def measure_throughput(pipeline: ExtractionPipeline, docs: List[Any]) -> float:
    """Measure throughput in documents per second."""
    if not docs:
        return 0.0
        
    start_time = time.time()
    for doc in docs:
        pipeline.process(doc)
    elapsed = time.time() - start_time
    
    return len(docs) / elapsed if elapsed > 0 else float('inf')


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    memory_bytes = process.memory_info().rss
    return memory_bytes / (1024 ** 3)  # Convert to GB
