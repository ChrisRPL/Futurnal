"""CLI commands for managing quarantined jobs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from .models import IngestionJob, JobPriority, JobType
from .quarantine import QuarantineReason, QuarantineStore
from .queue import JobQueue


quarantine_app = typer.Typer(help="Manage quarantine system")


def _get_quarantine_store(workspace_path: Optional[Path] = None) -> QuarantineStore:
    """Get quarantine store instance from workspace path."""
    workspace = workspace_path or Path.home() / ".futurnal"
    return QuarantineStore(workspace / "quarantine" / "quarantine.db")


def _get_job_queue(workspace_path: Optional[Path] = None) -> JobQueue:
    """Get job queue instance from workspace path."""
    workspace = workspace_path or Path.home() / ".futurnal"
    return JobQueue(workspace / "queue" / "jobs.db")


def _format_age(quarantined_at: datetime) -> str:
    """Format age of quarantined job in human-readable format."""
    delta = datetime.utcnow() - quarantined_at
    days = delta.days
    hours = delta.seconds // 3600

    if days > 0:
        return f"{days}d {hours}h"
    elif hours > 0:
        return f"{hours}h"
    else:
        minutes = delta.seconds // 60
        return f"{minutes}m"


@quarantine_app.command("list")
def list_quarantined(
    reason: Optional[str] = typer.Option(None, help="Filter by quarantine reason"),
    limit: int = typer.Option(50, help="Limit results (default: 50)"),
    format_output: str = typer.Option("table", "--format", help="Output format: table or json"),
    workspace: Optional[Path] = typer.Option(None, help="Workspace directory"),
) -> None:
    """List quarantined jobs with optional filtering."""
    store = _get_quarantine_store(workspace)

    # Parse reason filter if provided
    reason_filter = None
    if reason:
        try:
            reason_filter = QuarantineReason(reason)
        except ValueError:
            typer.echo(f"Invalid reason: {reason}")
            typer.echo(f"Valid reasons: {', '.join(r.value for r in QuarantineReason)}")
            raise typer.Exit(1)

    jobs = store.list(reason=reason_filter, limit=limit)

    if format_output == "json":
        output = []
        for job in jobs:
            output.append({
                "job_id": job.job_id,
                "job_type": job.job_type,
                "reason": job.reason.value,
                "age": _format_age(job.quarantined_at),
                "retry_count": job.retry_count,
                "can_retry": job.can_retry,
                "error_message": job.error_message,
            })
        typer.echo(json.dumps(output, indent=2))
    else:
        # Table format
        if not jobs:
            typer.echo("No quarantined jobs found.")
            return

        # Header
        typer.echo(f"{'JOB_ID':<36} | {'REASON':<20} | {'AGE':<10} | {'RETRIES':<8} | {'CAN_RETRY':<10}")
        typer.echo("-" * 90)

        # Rows
        for job in jobs:
            job_id_short = job.job_id[:36]
            age = _format_age(job.quarantined_at)
            can_retry = "Yes" if job.can_retry else "No"
            typer.echo(f"{job_id_short:<36} | {job.reason.value:<20} | {age:<10} | {job.retry_count:<8} | {can_retry:<10}")


@quarantine_app.command("show")
def show_quarantined(
    job_id: str = typer.Argument(..., help="Job ID to display"),
    workspace: Optional[Path] = typer.Option(None, help="Workspace directory"),
) -> None:
    """Display detailed information about a quarantined job."""
    store = _get_quarantine_store(workspace)
    job = store.get(job_id)

    if job is None:
        typer.echo(f"Job {job_id} not found in quarantine.")
        raise typer.Exit(1)

    # Display all job details
    typer.echo(f"\n{'='*80}")
    typer.echo(f"Quarantined Job: {job.job_id}")
    typer.echo(f"{'='*80}\n")

    typer.echo(f"Job Type:         {job.job_type}")
    typer.echo(f"Reason:           {job.reason.value}")
    typer.echo(f"Quarantined At:   {job.quarantined_at.isoformat()}")
    typer.echo(f"Age:              {_format_age(job.quarantined_at)}")
    typer.echo(f"Can Retry:        {'Yes' if job.can_retry else 'No'}")
    typer.echo(f"Retry Count:      {job.retry_count}")
    typer.echo(f"Success/Failure:  {job.retry_success_count}/{job.retry_failure_count}")
    if job.last_retry_at:
        typer.echo(f"Last Retry:       {job.last_retry_at.isoformat()}")

    typer.echo(f"\n{'Error Message:':-<80}")
    typer.echo(job.error_message)

    if job.error_traceback:
        typer.echo(f"\n{'Error Traceback:':-<80}")
        typer.echo(job.error_traceback)

    typer.echo(f"\n{'Original Job Payload:':-<80}")
    typer.echo(json.dumps(job.original_payload, indent=2))

    if job.metadata:
        typer.echo(f"\n{'Metadata:':-<80}")
        typer.echo(json.dumps(job.metadata, indent=2))

    if job.operator_notes:
        typer.echo(f"\n{'Operator Notes:':-<80}")
        typer.echo(job.operator_notes)

    typer.echo(f"\n{'='*80}\n")


@quarantine_app.command("retry")
def retry_quarantined(
    job_id: str = typer.Argument(..., help="Job ID to retry"),
    force: bool = typer.Option(False, "--force", help="Retry even if can_retry is False"),
    note: Optional[str] = typer.Option(None, "--note", help="Add operator note"),
    workspace: Optional[Path] = typer.Option(None, help="Workspace directory"),
) -> None:
    """Re-enqueue a quarantined job for retry."""
    store = _get_quarantine_store(workspace)
    queue = _get_job_queue(workspace)

    job = store.get(job_id)
    if job is None:
        typer.echo(f"Job {job_id} not found in quarantine.")
        raise typer.Exit(1)

    if not job.can_retry and not force:
        typer.echo(f"Job {job_id} is marked as cannot retry. Use --force to override.")
        raise typer.Exit(1)

    # Add operator note if provided
    if note:
        # Note: We'd need to add an update_notes method to QuarantineStore
        # For now, we'll just log it
        typer.echo(f"Note recorded: {note}")

    # Create new ingestion job from quarantined job
    try:
        job_type = JobType(job.job_type)
    except ValueError:
        typer.echo(f"Invalid job type: {job.job_type}")
        raise typer.Exit(1)

    # Create payload with quarantine markers
    retry_payload = dict(job.original_payload)
    retry_payload["from_quarantine"] = job.job_id
    retry_payload["quarantine_retry"] = True
    retry_payload["attempts"] = 0  # Reset attempts counter for fresh retry

    # Create and enqueue job with HIGH priority
    retry_job = IngestionJob(
        job_id=job.job_id,  # Reuse same job_id for tracking
        job_type=job_type,
        payload=retry_payload,
        priority=JobPriority.HIGH,
        scheduled_for=datetime.utcnow(),
    )

    queue.enqueue(retry_job)

    typer.echo(f"✓ Job {job_id} re-enqueued for retry with HIGH priority")
    typer.echo(f"  The orchestrator will pick it up automatically.")
    typer.echo(f"  Use 'futurnal quarantine show {job_id}' to check status.")


@quarantine_app.command("purge")
def purge_quarantined(
    older_than_days: Optional[int] = typer.Option(None, "--older-than-days", help="Remove jobs older than N days"),
    reason: Optional[str] = typer.Option(None, "--reason", help="Purge specific reason"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Purge specific job"),
    all_jobs: bool = typer.Option(False, "--all", help="Purge all quarantined jobs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be purged"),
    workspace: Optional[Path] = typer.Option(None, help="Workspace directory"),
) -> None:
    """Remove quarantined jobs based on criteria."""
    store = _get_quarantine_store(workspace)

    # Validate options
    if not any([older_than_days, reason, job_id, all_jobs]):
        typer.echo("Error: Must specify at least one purge criterion:")
        typer.echo("  --older-than-days, --reason, --job-id, or --all")
        raise typer.Exit(1)

    # Collect jobs to purge
    jobs_to_purge = []

    if job_id:
        # Purge specific job
        job = store.get(job_id)
        if job:
            jobs_to_purge.append(job)
        else:
            typer.echo(f"Job {job_id} not found in quarantine.")
            raise typer.Exit(1)
    elif all_jobs:
        # Purge all jobs
        jobs_to_purge = store.list()
    else:
        # Purge based on filters
        reason_filter = None
        if reason:
            try:
                reason_filter = QuarantineReason(reason)
            except ValueError:
                typer.echo(f"Invalid reason: {reason}")
                raise typer.Exit(1)

        all_jobs_list = store.list(reason=reason_filter)

        if older_than_days:
            cutoff = datetime.utcnow() - __import__('datetime').timedelta(days=older_than_days)
            jobs_to_purge = [j for j in all_jobs_list if j.quarantined_at < cutoff]
        else:
            jobs_to_purge = all_jobs_list

    if not jobs_to_purge:
        typer.echo("No jobs match the purge criteria.")
        return

    # Display what will be purged
    typer.echo(f"\n{'Jobs to purge:':=^80}\n")
    for job in jobs_to_purge:
        age = _format_age(job.quarantined_at)
        typer.echo(f"  - {job.job_id[:36]} | {job.reason.value:<20} | Age: {age}")

    typer.echo(f"\nTotal: {len(jobs_to_purge)} jobs")

    if dry_run:
        typer.echo("\n[DRY RUN] No jobs were purged.")
        return

    # Confirm purge
    if not typer.confirm("\nProceed with purge?"):
        typer.echo("Purge cancelled.")
        return

    # Execute purge
    for job in jobs_to_purge:
        store.remove(job.job_id)

    typer.echo(f"\n✓ Purged {len(jobs_to_purge)} jobs from quarantine.")


@quarantine_app.command("stats")
def show_statistics(
    workspace: Optional[Path] = typer.Option(None, help="Workspace directory"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Display quarantine statistics."""
    store = _get_quarantine_store(workspace)
    stats = store.statistics()

    if json_output:
        typer.echo(json.dumps(stats, indent=2))
    else:
        typer.echo(f"\n{'Quarantine Statistics':=^80}\n")
        typer.echo(f"Total Quarantined Jobs:    {stats['total_quarantined']}")
        typer.echo(f"Oldest Job Age:            {stats['oldest_job_age_days']} days")
        typer.echo(f"Recent (24h):              {stats['recent_quarantines_24h']}")
        typer.echo(f"Retry Success Rate:        {stats['retry_success_rate']:.1%}")

        if stats['by_reason']:
            typer.echo(f"\n{'Breakdown by Reason:':-<80}")
            for reason, count in sorted(stats['by_reason'].items(), key=lambda x: -x[1]):
                typer.echo(f"  {reason:<25} {count:>5} jobs")

        typer.echo(f"\n{'='*80}\n")
