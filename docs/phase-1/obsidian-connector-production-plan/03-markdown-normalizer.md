Summary: Specify markdown parsing/normalization for Obsidian notes, frontmatter, callouts, and dataview.

# 03 · Markdown Normalizer

## Purpose
Normalize Obsidian markdown to a consistent intermediate form suitable for Unstructured.io processing and PKG enrichment while preserving round-trip fidelity.

## Parsing Strategy
- Use `markdown-it-py` plus plugins equivalent to callouts, tables, task lists
- Extract YAML frontmatter via a robust parser; map known keys to PKG properties
- Preserve Obsidian callouts (`> [!note]`), tags (`#tag`), and code blocks

## Normalization
- Standardize line endings and whitespace
- Resolve relative links and embeds to absolute paths within vault sandbox
- Produce a structured representation: blocks, headings, links, tags, frontmatter

## Output Contract
- Emit normalized text for vector embedding
- Emit structured metadata for graph triple generation
- Attach provenance and checksum for deduplication

## Acceptance Criteria
- Supports complex markdown (tables, callouts, fenced code, footnotes)
- Frontmatter mapped deterministically; unknown keys preserved as `extra`

## Test Plan
- Unit: frontmatter parsing, callout recognition, tag extraction
- Integration: compare normalized output across diverse fixtures


