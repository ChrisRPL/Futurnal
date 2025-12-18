"""Academic Papers Connector - Integrates academic papers into the PKG.

This module provides the connector for ingesting downloaded academic papers
(PDFs) into the Personal Knowledge Graph. It extracts content, metadata,
and semantic relationships from papers to enable knowledge discovery.

The connector processes:
- PDF content via Unstructured.io partition
- Paper metadata (authors, venue, citations, DOI)
- Semantic triples for graph relationships
"""

from .connector import PapersConnector
from .triples import PaperTripleExtractor

__all__ = ["PapersConnector", "PaperTripleExtractor"]
