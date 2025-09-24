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
- Update `DEVELOPMENT_GUIDE.md` with links to the new tuning guidance.

## Work Breakdown
1. **Config Interface**
   - Extend config schema (`LocalIngestionConfig` or new orchestrator settings) to include concurrency knobs.
   - Update CLI to expose flags (e.g., `--max-workers`, `--scan-interval`), with validation.
   - Persist CLI-configured defaults and surface them in telemetry metadata.
2. **Scheduler Changes**
   - Update worker loop to respect configured max concurrency; consider using `asyncio.Semaphore`.
   - Implement batching for files within a job to reduce overhead while measuring impact on latency.
3. **Telemetry Enhancements**
   - Record actual throughput, active workers, queue depth, and backpressure events.
   - Surface metrics via CLI telemetry command and summary JSON.
4. **Watcher Resilience**
   - Rate-limit file watcher enqueue calls; batch changes before queuing jobs.
   - Fallback to timed scans if watcher errors occur.
5. **Operator Guidance**
   - Document CLI usage, example configurations, and telemetry interpretation for tuning decisions.
   - Capture recommended presets for representative hardware profiles (e.g., M2 Air vs. M2 Max).
5. **Performance Validation**
   - Run benchmarks across configurations; document recommended defaults for typical vault sizes.
   - Ensure adjustments do not conflict with privacy or resource constraints (CPU, IO).
   - Capture deviations between requested and effective concurrency in the telemetry summary.
   - Publish results to operator playbooks and update `DEVELOPMENT_GUIDE.md` pointers.

## Implementation Summary
- `LocalIngestionSource` now accepts optional `max_workers`, `max_files_per_batch`, `scan_interval_seconds`, and `watcher_debounce_seconds` fields that validate ranges before persisting the configuration.
- `futurnal local-sources register` exposes matching flags so operators can set these defaults without editing JSON by hand.
- The ingestion orchestrator clamps requested worker counts to hardware-aware ceilings, enforces concurrency via `asyncio.Semaphore`, and debounces file-system triggers per source.
- Batch limits short-circuit directory walks once the configured `max_files_per_batch` is reached; subsequent jobs resume processing remaining files.
- Telemetry entries now attach configured vs. active worker counts, queue depth, and observed throughput (bytes per second) so discrepancies are easy to spot.
- Fallback interval scans are registered automatically when `scan_interval_seconds` is provided, ensuring ingestion proceeds even if FSEvents drops events.

## Tuning Guidance
- **Light Hardware (e.g., M2 Air 8-core, ≤16 GB RAM):** Start with `--max-workers 2`, `--max-files-per-batch 25`, and `--watcher-debounce-seconds 0.5` to reduce CPU spikes. Use `--scan-interval-seconds 120` as a safety net.
- **Balanced Hardware (e.g., M3 Pro / Ryzen 7):** Configure `--max-workers 4`, `--max-files-per-batch 100`, and leave debounce at default (0) unless directories churn heavily. Set fallback scans around 60 seconds when operating on network drives.
- **High-End Hardware (e.g., Mac Studio M2 Max, ≥64 GB RAM):** Increase to `--max-workers 6` but monitor telemetry for thermal throttling. Batches of 250 files keep throughput high without overwhelming the PKG writers.
- **Telemetry Interpretation:** After runs, execute `futurnal local-sources telemetry --raw` and inspect `metadata.active_workers` vs. `metadata.configured_workers`. A large gap indicates the semaphore or hardware guardrails are limiting concurrency; adjust `max_workers` or investigate system load. Track `metadata.effective_throughput_bps` against the ≥5 MB/s target, and compare queue depth to ensure jobs are not starving downstream pipelines.
- **Watcher Health:** If telemetry shows frequent fallback scans or operators observe missed updates, lower `watcher_debounce_seconds` and shorten `scan_interval_seconds` incrementally until events stabilize.
- **Batch Sizing:** Use smaller `max_files_per_batch` values when low-latency updates are preferred (e.g., PKG nodes powering live dashboards). Larger batches reduce overhead for bulk backfills but should be paired with higher concurrency only when storage bandwidth allows.

## Open Questions
- Should concurrency be source-specific or global? (e.g., some directories may tolerate higher parallelism.)
- How do we surface warnings when user-defined settings are too aggressive (e.g., saturating disk)?
- Do we need automated tuning (adaptive concurrency) or presets based on hardware detection?


