Summary: Define pipeline for handling Obsidian embedded assets (images, PDFs) securely.

# 05 · Asset Pipeline

## Purpose
Process embedded images and PDFs referenced in notes, deduplicate by content hash, and route through Unstructured.io where appropriate.

## Flow
1. Detect embeds `![[file.ext]]` and markdown image links
2. Resolve to absolute path under vault root; deny traversal outside sandbox
3. Compute content hash; store once per vault
4. For PDFs/images, extract text with Unstructured when configured
5. Link assets to notes with `embeds` relationship, including bounding metadata

## Security & Privacy
- Enforce vault sandboxing and deny symlink escapes
- Respect redact lists; avoid logging sensitive filenames

## Acceptance Criteria
- Assets deduplicated; ingestion idempotent
- Text extraction optional and auditable

## Test Plan
- Unit: path resolution and sandbox enforcement
- Integration: mixed embeds with duplicates and symlinks


