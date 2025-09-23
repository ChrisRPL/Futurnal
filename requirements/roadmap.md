Summary: Provides the phased MVP rollout plan, success metrics, and milestone checkpoints for Futurnal.

# Implementation Roadmap

## Overview
Futurnal progresses through three phases—Archivist, Analyst, Guide—each expanding on capabilities delivered previously. Timelines reflect the initial 15-month plan and should be revisited quarterly.

## Phase 1 · The Archivist (Months 1–4)
- **Goals:** Establish ingestion pipeline, PKG construction, and hybrid search UI.
- **Key Features:**
  - Connectors for local files, Obsidian vaults, IMAP email, GitHub repositories.
  - Automated entity extraction and PKG population.
  - Unified search interface combining semantic and graph queries.
  - Interactive graph visualization for manual exploration.
- **Milestones:**
  - Stable ingestion for three data sources with <1% parsing failures.
  - Query latency under 1 second on reference hardware.
  - Beta testers actively retrieving insights via search.
- **KPIs:** Daily active users (DAU), average queries per session, number of data sources connected.

## Phase 2 · The Analyst (Months 5–9)
- **Goals:** Deliver proactive insight generation and user notifications.
- **Key Features:**
  - Emergent Insights dashboard aggregating correlations and clusters.
  - Scheduled insight detection jobs leveraging graph analytics (community detection, centrality).
  - Insight ranking model tuned via user feedback and Aspirational Self context.
  - Notification system for newly surfaced insights.
- **Milestones:**
  - Insight detection pipeline running at least daily with ≥90% uptime.
  - Initial ranking heuristics yielding >50% user "useful" ratings.
  - Closed-beta release of Analyst dashboards.
- **KPIs:** Insight click-through rate (CTR), qualitative feedback scores, dismiss vs. explore ratio.

## Phase 3 · The Guide (Months 10–15)
- **Goals:** Enable causal exploration conversations and goal alignment workflows.
- **Key Features:**
  - Conversational interface integrated with PKG context retrieval.
  - LLM-driven hypothesis generation based on identified correlations.
  - Guided investigation prompts to evaluate confounders and causal plausibility.
  - Full Aspirational Self integration with proactive alignment alerts.
- **Milestones:**
  - Conversational loop validated with at least three user-tested causal scenarios.
  - Hypothesis library grounded in causal inference best practices.
  - Production-ready Guide experience for premium tier.
- **KPIs:** Session length in Guide mode, percentage of users defining aspirations, free-to-pro conversion rate.

## Cross-Phase Enablers
- **Privacy & Security:** Implement consent management, audit logs, and encryption from Phase 1.
- **Performance Observability:** Build profiling and telemetry tools early to monitor ingestion, query, and insight pipelines.
- **Feedback Loops:** Integrate in-app feedback mechanisms for insight relevance, search quality, and conversational guidance.

## Risk Mitigation Plan
- **Insight Accuracy:** Maintain human-in-the-loop review for initial rollouts; flag low-confidence insights automatically.
- **User Trust:** Provide transparent messaging about data handling and optional cloud escalations.
- **Scalability:** Design ingestion and graph pipelines for parallel execution to accommodate growing data volumes.

