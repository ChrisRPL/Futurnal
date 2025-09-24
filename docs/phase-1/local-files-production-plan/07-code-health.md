Summary: Plan for code health improvements and residual technical debt for the Local Files Connector.

# Task · Code Health & Clean-up

## Objectives
- Address outstanding technical debt (deprecated APIs, logging gaps, documentation cross-links).
- Improve maintainability and reviewer experience before declaring Feature 1 complete.

## Deliverables
- Migration of Pydantic validators to V2 `@field_validator` pattern.
- Comprehensive docstrings for key modules (connector, orchestrator, queue, telemetry) and inline comments clarifying tricky logic.
- Enhanced logging with structured context and consistent levels.
- Updated prompts/checklists referencing this production plan for code reviews.

## Work Breakdown
1. **Pydantic Migration**
   - Replace `@validator` usage with `@field_validator`; adjust imports and tests.
   - Ensure validation error messages remain user-friendly.
2. **Documentation Improvements**
   - Add/refresh module docstrings and README sections explaining architecture hooks.
   - Reference this production plan from `DEVELOPMENT_GUIDE.md` and feature docs.
3. **Logging Cleanup**
   - Standardize log format (include job/source identifiers, correlation IDs where available).
   - Ensure warning/error logs provide actionable detail without leaking sensitive data.
4. **Review Prompts**
   - Update `prompts/phase-1-archivist.md` or create supplementary checklist ensuring reviewers verify telemetry, privacy, and test coverage.
5. **Lint & Typing**
   - Run static analysis (mypy/ruff) and address any outstanding findings.
   - Decide on minimum coverage thresholds and add badges/reporting if helpful.

## Open Questions
- Should we enforce stricter type hints (e.g., `mypy --strict`) for ingestion modules?
- Are there opportunities to refactor large functions (e.g., `_process_job`) for clarity before GA?
- Do we need a contributor guide specific to ingestion components?


