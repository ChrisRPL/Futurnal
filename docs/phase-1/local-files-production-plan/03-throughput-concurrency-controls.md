Summary: Plan for configuring and validating throughput and concurrency controls.

# Task · Throughput & Concurrency Controls

## Objectives
- Give operators control over ingestion concurrency, batch sizing, and scan intervals.
- Ensure system sustains ≥5 MB/s throughput without starving other workloads or flooding the queue.
- Harden FSEvents-driven updates to avoid runaway scheduling under heavy churn.

## Deliverables
- Configuration options (CLI flags, config file, or environment variables) for:
  - Worker concurrency (# of parallel ingestion jobs)
  - Batch size / throttling per job
  - Periodic scan interval fallback if file watcher unavailable
- Telemetry counters reporting configured vs. effective concurrency and measured throughput.
- Guardrails to prevent too many simultaneous jobs (e.g., resource caps, adaptive backoff).
- Documentation guiding users on tuning for their hardware.

## Work Breakdown
1. **Config Interface**
   - Extend config schema (`LocalIngestionConfig` or new orchestrator settings) to include concurrency knobs.
   - Update CLI to expose flags (e.g., `--max-workers`, `--scan-interval`), with validation.
2. **Scheduler Changes**
   - Update worker loop to respect configured max concurrency; consider using `asyncio.Semaphore`.
   - Implement batching for files within a job to reduce overhead while measuring impact on latency.
3. **Telemetry Enhancements**
   - Record actual throughput, active workers, queue depth, and backpressure events.
   - Surface metrics via CLI telemetry command and summary JSON.
4. **Watcher Resilience**
   - Rate-limit file watcher enqueue calls; batch changes before queuing jobs.
   - Fallback to timed scans if watcher errors occur.
5. **Performance Validation**
   - Run benchmarks across configurations; document recommended defaults for typical vault sizes.
   - Ensure adjustments do not conflict with privacy or resource constraints (CPU, IO).

## Open Questions
- Should concurrency be source-specific or global? (e.g., some directories may tolerate higher parallelism.)
- How do we surface warnings when user-defined settings are too aggressive (e.g., saturating disk)?
- Do we need automated tuning (adaptive concurrency) or presets based on hardware detection?


