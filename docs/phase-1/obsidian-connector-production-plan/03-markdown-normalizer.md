Summary: Specify markdown parsing/normalization for extracting thought patterns from Obsidian notes.

# 03 · Markdown Normalizer

## Purpose
Normalize Obsidian markdown to extract the user's thought patterns and conceptual relationships while preparing content for Ghost's experiential learning. This component transforms user's markdown notes into structured understanding suitable for PKG enrichment, preserving both the semantic content and the mental model revealed through Obsidian's formatting conventions. The normalizer enables the Ghost to learn from how the user structures thoughts, connects concepts, and categorizes knowledge.

## Parsing Strategy
- Use `markdown-it-py` plus plugins equivalent to callouts, tables, task lists
- Extract YAML frontmatter via a robust parser; map known keys to PKG properties
- Preserve Obsidian callouts (`> [!note]`), tags (`#tag`), and code blocks

## Normalization
- Standardize line endings and whitespace
- Resolve relative links and embeds to absolute paths within vault sandbox
- Produce a structured representation: blocks, headings, links, tags, frontmatter

## Output Contract
- Emit normalized text for Ghost's semantic understanding (vector embeddings)
- Emit structured metadata revealing user's thought patterns for experiential graph construction
- Preserve conceptual relationships (wikilinks, tags) for Ghost's mental model learning
- Attach temporal and provenance metadata enabling Ghost to learn from content evolution

## Acceptance Criteria
- Supports complex markdown (tables, callouts, fenced code, footnotes)
- Frontmatter mapped deterministically; unknown keys preserved as `extra`

## Test Plan
- Unit: frontmatter parsing, callout recognition, tag extraction
- Integration: compare normalized output across diverse fixtures


