"""Papers Connector - Ingests academic papers into the PKG.

This connector processes downloaded academic papers (PDFs) through the
Unstructured.io pipeline, enriches elements with paper metadata, and
generates semantic triples for knowledge graph construction.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.consent import ConsentRegistry, ConsentRequiredError
from futurnal.privacy.redaction import RedactionPolicy, build_policy, redact_path

from .triples import PaperTripleExtractor

logger = logging.getLogger(__name__)


@runtime_checkable
class ElementSink(Protocol):
    """Sink interface for handling parsed elements."""

    def handle(self, element: dict) -> None:
        ...

    def handle_deletion(self, element: dict) -> None:
        ...


@dataclass
class PaperMetadata:
    """Metadata for an academic paper."""

    paper_id: str
    title: str
    authors: List[Dict[str, Any]]
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None
    citation_count: int = 0
    fields_of_study: List[str] = None
    pdf_url: Optional[str] = None

    def __post_init__(self):
        if self.fields_of_study is None:
            self.fields_of_study = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "paperId": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxivId": self.arxiv_id,
            "abstract": self.abstract,
            "citationCount": self.citation_count,
            "fieldsOfStudy": self.fields_of_study,
            "pdfUrl": self.pdf_url,
        }


def _log_extra(
    *,
    job_id: str | None = None,
    paper_id: str | None = None,
    paper_path: Path | None = None,
    policy: RedactionPolicy | None = None,
    **metadata: object,
) -> dict[str, object]:
    """Build structured logging context with privacy-aware path handling."""
    extra: dict[str, object] = {}
    if job_id:
        extra["ingestion_job_id"] = job_id
    if paper_id:
        extra["paper_id"] = paper_id
    if paper_path is not None:
        active_policy = policy or build_policy()
        redacted = redact_path(paper_path, policy=active_policy)
        extra["ingestion_path"] = redacted.redacted
        extra["ingestion_path_hash"] = redacted.path_hash
    for key, value in metadata.items():
        if value is not None:
            extra[f"ingestion_{key}"] = value
    return extra


class PapersConnector:
    """Connector for ingesting academic papers into the PKG.

    This connector processes PDF files through Unstructured.io, enriches
    extracted elements with paper metadata (authors, citations, DOI),
    and generates semantic triples for graph construction.
    """

    def __init__(
        self,
        *,
        workspace_dir: Path | str,
        element_sink: ElementSink | None = None,
        audit_logger: AuditLogger | None = None,
        consent_registry: ConsentRegistry | None = None,
    ) -> None:
        """Initialize the papers connector.

        Args:
            workspace_dir: Base directory for paper storage and processing.
            element_sink: Sink for handling parsed elements (Neo4j, ChromaDB).
            audit_logger: Logger for audit events.
            consent_registry: Registry for consent checks.
        """
        self._workspace_dir = Path(workspace_dir)
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        self._parsed_dir = self._workspace_dir / "parsed"
        self._parsed_dir.mkdir(parents=True, exist_ok=True)
        self._quarantine_dir = self._workspace_dir / "quarantine"
        self._quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._element_sink = element_sink
        self._audit_logger = audit_logger
        self._consent_registry = consent_registry
        self._triple_extractor = PaperTripleExtractor()

    def ingest(
        self,
        paper_path: Path,
        metadata: PaperMetadata,
        *,
        job_id: str | None = None,
        require_consent: bool = True,
    ) -> Iterable[dict]:
        """Ingest a single academic paper into the PKG.

        Args:
            paper_path: Path to the PDF file.
            metadata: Paper metadata (authors, title, DOI, etc.).
            job_id: Optional job ID for tracking.
            require_consent: Whether to require consent for processing.

        Yields:
            Parsed document elements with enriched metadata.
        """
        active_job_id = job_id or uuid.uuid4().hex
        policy = build_policy()

        logger.info(
            "Starting paper ingestion",
            extra=_log_extra(
                job_id=active_job_id,
                paper_id=metadata.paper_id,
                paper_path=paper_path,
                policy=policy,
                event="ingest_start",
            ),
        )

        # Check consent if required
        if require_consent and self._consent_registry:
            try:
                self._consent_registry.require(
                    source="academic_papers",
                    scope="external_processing",
                )
            except ConsentRequiredError as exc:
                logger.error(
                    "Consent missing for paper processing",
                    extra=_log_extra(
                        job_id=active_job_id,
                        paper_id=metadata.paper_id,
                        paper_path=paper_path,
                        policy=policy,
                        event="consent_missing",
                    ),
                    exc_info=exc,
                )
                self._log_audit_event(
                    job_id=active_job_id,
                    paper_id=metadata.paper_id,
                    action="external_processing",
                    status="blocked",
                    metadata={"reason": "consent_required"},
                )
                return

        # Validate file exists and is PDF
        if not paper_path.exists():
            logger.error(
                "Paper file not found",
                extra=_log_extra(
                    job_id=active_job_id,
                    paper_id=metadata.paper_id,
                    paper_path=paper_path,
                    policy=policy,
                    event="file_not_found",
                ),
            )
            return

        if paper_path.suffix.lower() != ".pdf":
            logger.warning(
                "Paper file is not a PDF, attempting partition anyway",
                extra=_log_extra(
                    job_id=active_job_id,
                    paper_id=metadata.paper_id,
                    paper_path=paper_path,
                    policy=policy,
                    event="non_pdf_file",
                ),
            )

        # Partition the PDF
        try:
            from unstructured.partition.auto import partition

            elements = partition(
                filename=str(paper_path),
                strategy="fast",
                include_metadata=True,
            )
        except Exception as exc:
            logger.exception(
                "Failed to partition paper PDF",
                extra=_log_extra(
                    job_id=active_job_id,
                    paper_id=metadata.paper_id,
                    paper_path=paper_path,
                    policy=policy,
                    event="partition_failed",
                ),
            )
            self._quarantine(paper_path, "partition_error", str(exc), metadata.paper_id, policy)
            self._log_audit_event(
                job_id=active_job_id,
                paper_id=metadata.paper_id,
                action="partition",
                status="failed",
                metadata={"detail": str(exc)},
            )
            return

        # Generate semantic triples from metadata
        triples = self._triple_extractor.extract(metadata)

        # Process each element
        element_count = 0
        had_failure = False

        for element in elements:
            try:
                parsed = self._enrich_element(element, metadata, triples, paper_path)
            except Exception as exc:
                logger.exception(
                    "Failed to enrich element",
                    extra=_log_extra(
                        job_id=active_job_id,
                        paper_id=metadata.paper_id,
                        paper_path=paper_path,
                        policy=policy,
                        event="enrich_failed",
                    ),
                )
                had_failure = True
                continue

            # Send to element sink
            if self._element_sink is not None:
                try:
                    self._element_sink.handle(parsed)
                except Exception as exc:
                    logger.exception(
                        "Element sink reported a failure",
                        extra=_log_extra(
                            job_id=active_job_id,
                            paper_id=metadata.paper_id,
                            paper_path=paper_path,
                            policy=policy,
                            event="sink_failed",
                        ),
                    )
                    self._quarantine(paper_path, "sink_error", str(exc), metadata.paper_id, policy)
                    had_failure = True
                    continue

            element_count += 1
            yield parsed

        # Log completion
        if not had_failure:
            logger.info(
                "Paper ingested successfully",
                extra=_log_extra(
                    job_id=active_job_id,
                    paper_id=metadata.paper_id,
                    paper_path=paper_path,
                    policy=policy,
                    event="ingest_succeeded",
                    element_count=element_count,
                    triple_count=len(triples),
                ),
            )
            self._log_audit_event(
                job_id=active_job_id,
                paper_id=metadata.paper_id,
                action="ingest",
                status="succeeded",
                metadata={
                    "element_count": element_count,
                    "triple_count": len(triples),
                },
            )
        else:
            logger.warning(
                "Paper ingestion completed with errors",
                extra=_log_extra(
                    job_id=active_job_id,
                    paper_id=metadata.paper_id,
                    paper_path=paper_path,
                    policy=policy,
                    event="ingest_partial",
                    element_count=element_count,
                ),
            )

    def ingest_batch(
        self,
        papers: List[tuple[Path, PaperMetadata]],
        *,
        job_id: str | None = None,
    ) -> Dict[str, Any]:
        """Ingest multiple papers.

        Args:
            papers: List of (path, metadata) tuples.
            job_id: Optional job ID for tracking.

        Returns:
            Summary of ingestion results.
        """
        active_job_id = job_id or uuid.uuid4().hex
        results = {
            "job_id": active_job_id,
            "total": len(papers),
            "succeeded": 0,
            "failed": 0,
            "papers": [],
        }

        for paper_path, metadata in papers:
            try:
                elements = list(self.ingest(paper_path, metadata, job_id=active_job_id))
                results["succeeded"] += 1
                results["papers"].append({
                    "paperId": metadata.paper_id,
                    "status": "succeeded",
                    "elementCount": len(elements),
                })
            except Exception as exc:
                results["failed"] += 1
                results["papers"].append({
                    "paperId": metadata.paper_id,
                    "status": "failed",
                    "error": str(exc),
                })

        return results

    def _enrich_element(
        self,
        element,
        metadata: PaperMetadata,
        triples: List[dict],
        paper_path: Path,
    ) -> dict:
        """Enrich a parsed element with paper metadata."""
        if hasattr(element, "to_dict"):
            payload = element.to_dict()
        elif isinstance(element, dict):
            payload = element
        else:
            payload = {"text": str(element)}

        # Add paper-specific metadata
        payload["paper_metadata"] = {
            "paper_id": metadata.paper_id,
            "title": metadata.title,
            "authors": [a.get("name", "") for a in metadata.authors],
            "year": metadata.year,
            "venue": metadata.venue,
            "doi": metadata.doi,
            "arxiv_id": metadata.arxiv_id,
            "citation_count": metadata.citation_count,
            "fields_of_study": metadata.fields_of_study,
        }

        # Add semantic triples
        payload["semantic_triples"] = triples

        # Add source info
        if "metadata" not in payload:
            payload["metadata"] = {}
        payload["metadata"]["source_type"] = "academic_paper"
        payload["metadata"]["source_path"] = str(paper_path)
        payload["metadata"]["ingested_at"] = datetime.now().isoformat()

        return payload

    def _quarantine(
        self,
        path: Path,
        reason: str,
        detail: str,
        paper_id: str,
        policy: RedactionPolicy,
    ) -> None:
        """Move a problematic file to quarantine."""
        try:
            quarantine_path = self._quarantine_dir / f"{paper_id}_{path.name}"
            if path.exists():
                import shutil

                shutil.copy2(path, quarantine_path)
            # Write error info
            error_path = quarantine_path.with_suffix(".error.json")
            error_path.write_text(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "reason": reason,
                        "detail": detail,
                        "original_path": str(path),
                        "quarantined_at": datetime.now().isoformat(),
                    }
                )
            )
            logger.info(
                "File quarantined",
                extra=_log_extra(
                    paper_id=paper_id,
                    paper_path=path,
                    policy=policy,
                    event="quarantined",
                    reason=reason,
                ),
            )
        except Exception as exc:
            logger.exception(
                "Failed to quarantine file",
                extra=_log_extra(
                    paper_id=paper_id,
                    paper_path=path,
                    policy=policy,
                    event="quarantine_failed",
                ),
            )

    def _log_audit_event(
        self,
        *,
        job_id: str,
        paper_id: str,
        action: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an audit event if audit logger is configured."""
        if self._audit_logger:
            self._audit_logger.log_action(
                actor="papers_connector",
                action=action,
                resource=f"paper:{paper_id}",
                outcome=status,
                metadata={
                    "job_id": job_id,
                    **(metadata or {}),
                },
            )
