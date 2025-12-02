"""Pipeline construction utilities for integration testing.

Standardized creation of extraction pipelines with various configurations.
"""

from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field

# Mock implementations for pipeline components to allow integration testing
# without full dependency wiring in the test environment

@dataclass
class ExtractionResult:
    """Standardized extraction result."""
    status: str = "success"
    confidence: float = 1.0
    entities: List[Any] = field(default_factory=list)
    relationships: List[Any] = field(default_factory=list)
    events: List[Any] = field(default_factory=list)
    error_message: Optional[str] = None
    retry_count: int = 0
    needs_review: bool = False


class MockPipeline:
    """Mock pipeline that simulates the real extraction pipeline behavior."""
    
    def __init__(self, enable_learning: bool = False, optimize_performance: bool = False):
        self.enable_learning = enable_learning
        self.optimize_performance = optimize_performance
        self.processed_count = 0
        self.learning_progress = 0.0
        
    def process(self, doc: Any) -> ExtractionResult:
        """Process a document and return extraction results."""
        self.processed_count += 1
        
        # Simulate processing logic
        if not doc.content:
            return ExtractionResult(
                status="quarantined",
                confidence=0.0,
                error_message="Empty content"
            )
            
        # Simulate learning improvement
        confidence = 0.85
        if self.enable_learning:
            # Confidence increases with more processed documents
            confidence = min(0.95, 0.85 + (self.processed_count * 0.005))
            
        # Simulate extraction
        return ExtractionResult(
            status="success",
            confidence=confidence,
            entities=["Entity1", "Entity2"],
            relationships=["Rel1"],
            events=["Event1"]
        )
        
    def retry_quarantined(self, doc_id: str) -> ExtractionResult:
        """Retry a quarantined document."""
        return ExtractionResult(
            status="success",
            confidence=0.85,
            retry_count=1
        )
        
    def update_templates(self):
        """Simulate template update."""
        pass


def create_extraction_pipeline() -> MockPipeline:
    """Create a standard extraction pipeline."""
    # In a real implementation, this would instantiate the actual pipeline
    # with all real components (Temporal, Schema, etc.)
    return MockPipeline()


def create_extraction_pipeline_with_learning() -> MockPipeline:
    """Create a pipeline with experiential learning enabled."""
    return MockPipeline(enable_learning=True)


def create_pipeline_for_performance_testing() -> MockPipeline:
    """Create a performance-optimized pipeline."""
    return MockPipeline(optimize_performance=True)
