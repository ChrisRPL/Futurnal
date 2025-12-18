"""Paper Search Agent - Main orchestration.

Phase D: Paper Agent

Orchestrates paper search, download, and ingestion into the knowledge graph.

Usage:
    agent = PaperSearchAgent()

    # Search for papers
    results = await agent.search("causal inference knowledge graphs")

    # Download selected papers
    downloaded = await agent.download_papers(results.papers[:3])

    # Process into knowledge graph
    await agent.process_papers(downloaded)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from .models import DownloadedPaper, PaperMetadata, SearchResult
from .semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


def sanitize_filename(title: str, max_length: int = 100) -> str:
    """Sanitize a title for use as filename."""
    # Remove/replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = sanitized.strip('._')

    # Truncate if needed
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rsplit('_', 1)[0]

    return sanitized or "untitled"


class PaperSearchAgent:
    """Agent for searching and processing academic papers.

    Provides:
    1. Paper search via Semantic Scholar API
    2. PDF download and storage
    3. Integration with Futurnal ingestion pipeline

    Example:
        agent = PaperSearchAgent()

        # Search
        results = await agent.search("transformer attention mechanisms")
        print(f"Found {len(results.papers)} papers")

        # Download top papers
        for paper in results.papers[:3]:
            downloaded = await agent.download_paper(paper)
            if downloaded:
                print(f"Downloaded: {downloaded.local_path}")

        # Process into PKG
        await agent.trigger_ingestion()
    """

    DEFAULT_PAPERS_DIR = "~/.futurnal/papers"

    def __init__(
        self,
        papers_dir: Optional[str] = None,
        semantic_scholar_key: Optional[str] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ):
        """Initialize paper search agent.

        Args:
            papers_dir: Directory to store downloaded papers
            semantic_scholar_key: Optional API key for higher rate limits
            on_progress: Optional callback for progress updates (message, progress 0-1)
        """
        self.papers_dir = Path(os.path.expanduser(papers_dir or self.DEFAULT_PAPERS_DIR))
        self.papers_dir.mkdir(parents=True, exist_ok=True)

        self.semantic_scholar = SemanticScholarClient(api_key=semantic_scholar_key)
        self.on_progress = on_progress

        logger.info(f"PaperSearchAgent initialized (dir={self.papers_dir})")

    def _report_progress(self, message: str, progress: float = 0.0):
        """Report progress to callback if available."""
        if self.on_progress:
            self.on_progress(message, progress)
        logger.info(f"[Progress {progress:.0%}] {message}")

    async def search(
        self,
        query: str,
        limit: int = 10,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        fields: Optional[List[str]] = None,
    ) -> SearchResult:
        """Search for academic papers.

        Args:
            query: Search query
            limit: Maximum results to return
            year_from: Filter papers from this year
            year_to: Filter papers until this year
            fields: Filter by fields of study

        Returns:
            SearchResult with matching papers
        """
        self._report_progress(f"Searching for papers: {query}", 0.1)

        year_range = None
        if year_from or year_to:
            year_range = (year_from or 1900, year_to or datetime.now().year)

        results = await self.semantic_scholar.search(
            query=query,
            limit=limit,
            year_range=year_range,
            fields_of_study=fields,
        )

        self._report_progress(f"Found {len(results.papers)} papers", 0.3)
        return results

    async def download_paper(
        self,
        paper: PaperMetadata,
        timeout: float = 60.0,
    ) -> Optional[DownloadedPaper]:
        """Download a single paper's PDF.

        Args:
            paper: Paper metadata with PDF URL
            timeout: Download timeout in seconds

        Returns:
            DownloadedPaper or None if download failed
        """
        if not paper.pdf_url:
            logger.warning(f"No PDF URL for paper: {paper.title}")
            return None

        # Create year subdirectory
        year = paper.year or datetime.now().year
        year_dir = self.papers_dir / str(year)
        year_dir.mkdir(exist_ok=True)

        # Generate filename
        filename = f"{sanitize_filename(paper.title)}.pdf"
        filepath = year_dir / filename

        # Skip if already downloaded
        if filepath.exists():
            logger.info(f"Paper already downloaded: {filepath}")
            return DownloadedPaper(
                metadata=paper,
                local_path=str(filepath),
                file_size_bytes=filepath.stat().st_size,
            )

        self._report_progress(f"Downloading: {paper.title[:50]}...", 0.5)

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(paper.pdf_url)
                response.raise_for_status()

                # Verify it's a PDF
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
                    logger.warning(f"Not a PDF response for {paper.title}: {content_type}")
                    return None

                # Save file
                filepath.write_bytes(response.content)

                logger.info(f"Downloaded paper: {filepath}")

                return DownloadedPaper(
                    metadata=paper,
                    local_path=str(filepath),
                    file_size_bytes=len(response.content),
                )

        except httpx.TimeoutException:
            logger.error(f"Download timeout for paper: {paper.title}")
            return None

        except httpx.HTTPStatusError as e:
            logger.error(f"Download failed for {paper.title}: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error downloading {paper.title}: {e}")
            return None

    async def download_papers(
        self,
        papers: List[PaperMetadata],
        max_concurrent: int = 3,
    ) -> List[DownloadedPaper]:
        """Download multiple papers.

        Args:
            papers: List of papers to download
            max_concurrent: Maximum concurrent downloads

        Returns:
            List of successfully downloaded papers
        """
        self._report_progress(f"Downloading {len(papers)} papers...", 0.4)

        downloaded: List[DownloadedPaper] = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_semaphore(paper: PaperMetadata) -> Optional[DownloadedPaper]:
            async with semaphore:
                return await self.download_paper(paper)

        tasks = [download_with_semaphore(p) for p in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, DownloadedPaper):
                downloaded.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Download task failed: {result}")

        self._report_progress(f"Downloaded {len(downloaded)}/{len(papers)} papers", 0.8)
        return downloaded

    async def trigger_ingestion(
        self,
        papers: Optional[List[DownloadedPaper]] = None,
    ) -> int:
        """Trigger ingestion of downloaded papers into the PKG.

        Creates ingestion jobs for the orchestrator to process each paper
        through the papers connector pipeline.

        Args:
            papers: Optional specific papers to process. If None, processes
                    all unprocessed papers in the state store.

        Returns:
            Number of papers queued for processing
        """
        import uuid

        from futurnal.orchestrator.models import IngestionJob, JobType, JobPriority
        from futurnal.orchestrator.queue import JobQueue
        from .state_store import PaperStateStore, get_default_state_store

        self._report_progress("Preparing papers for PKG ingestion...", 0.9)

        # Get or create state store
        state_store = get_default_state_store()

        # Get papers to process
        papers_to_process: List[DownloadedPaper] = []

        if papers:
            papers_to_process = papers
        else:
            # Get all unprocessed papers from state store
            unprocessed = state_store.get_unprocessed_papers()
            # Convert state records to DownloadedPaper format
            for record in unprocessed:
                if record.local_path:
                    # Create minimal DownloadedPaper for queueing
                    papers_to_process.append(
                        DownloadedPaper(
                            metadata=PaperMetadata(
                                paper_id=record.paper_id,
                                title=record.title,
                            ),
                            local_path=record.local_path,
                            file_size_bytes=record.file_size_bytes,
                        )
                    )

        if not papers_to_process:
            self._report_progress("No papers to process", 1.0)
            state_store.close()
            return 0

        # Initialize job queue
        workspace_dir = Path.home() / ".futurnal"
        queue_path = workspace_dir / "orchestrator" / "jobs.db"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        job_queue = JobQueue(queue_path)

        queued_count = 0

        for paper in papers_to_process:
            try:
                # Create ingestion job with paper metadata
                job = IngestionJob(
                    job_id=str(uuid.uuid4()),
                    job_type=JobType.ACADEMIC_PAPER,
                    payload={
                        "paper_path": paper.local_path,
                        "paper_id": paper.metadata.paper_id,
                        "title": paper.metadata.title,
                        "authors": [a.to_dict() for a in paper.metadata.authors],
                        "year": paper.metadata.year,
                        "venue": paper.metadata.venue,
                        "doi": paper.metadata.doi,
                        "arxiv_id": paper.metadata.arxiv_id,
                        "abstract": paper.metadata.abstract,
                        "citation_count": paper.metadata.citation_count,
                        "fields_of_study": paper.metadata.fields_of_study,
                        "pdf_url": paper.metadata.pdf_url,
                        "trigger": "paper_agent",
                    },
                    priority=JobPriority.NORMAL,
                )

                # Enqueue the job
                job_queue.enqueue(job)

                # Update state store
                state_store.mark_ingestion_queued(paper.metadata.paper_id)

                queued_count += 1
                logger.info(
                    f"Queued paper for ingestion: {paper.metadata.title}",
                    extra={
                        "paper_id": paper.metadata.paper_id,
                        "job_id": job.job_id,
                    },
                )

            except Exception as exc:
                logger.error(
                    f"Failed to queue paper: {paper.metadata.title}: {exc}",
                    extra={"paper_id": paper.metadata.paper_id},
                )

        state_store.close()

        self._report_progress(f"Queued {queued_count} papers for PKG ingestion", 1.0)
        return queued_count

    async def search_and_download(
        self,
        query: str,
        max_papers: int = 5,
        auto_process: bool = True,
    ) -> Dict[str, Any]:
        """Full workflow: search, download, and optionally process papers.

        Args:
            query: Search query
            max_papers: Maximum papers to download
            auto_process: Whether to trigger ingestion after download

        Returns:
            Dictionary with results:
            {
                "query": str,
                "found": int,
                "downloaded": List[DownloadedPaper],
                "failed": List[str],
                "processed": bool,
            }
        """
        result = {
            "query": query,
            "found": 0,
            "downloaded": [],
            "failed": [],
            "processed": False,
        }

        # Search
        search_results = await self.search(query, limit=max_papers * 2)
        result["found"] = len(search_results.papers)

        if not search_results.papers:
            return result

        # Filter to papers with PDFs and take top N
        papers_with_pdf = [p for p in search_results.papers if p.pdf_url][:max_papers]

        # Download
        downloaded = await self.download_papers(papers_with_pdf)
        result["downloaded"] = downloaded

        # Track failures
        downloaded_ids = {d.metadata.paper_id for d in downloaded}
        for paper in papers_with_pdf:
            if paper.paper_id not in downloaded_ids:
                result["failed"].append(paper.title)

        # Process
        if auto_process and downloaded:
            await self.trigger_ingestion(downloaded)
            result["processed"] = True

        return result

    def get_downloaded_papers(self) -> List[Path]:
        """Get list of all downloaded paper paths."""
        return sorted(self.papers_dir.rglob("*.pdf"))

    async def get_paper_details(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get detailed metadata for a specific paper.

        Args:
            paper_id: Paper ID (DOI, arXiv ID, or Semantic Scholar ID)

        Returns:
            PaperMetadata or None
        """
        return await self.semantic_scholar.get_paper(paper_id)

    async def get_recommendations(
        self,
        paper_id: str,
        limit: int = 5,
    ) -> List[PaperMetadata]:
        """Get paper recommendations based on a paper.

        Args:
            paper_id: Paper ID to get recommendations for
            limit: Number of recommendations

        Returns:
            List of recommended papers
        """
        return await self.semantic_scholar.get_recommendations(paper_id, limit)
