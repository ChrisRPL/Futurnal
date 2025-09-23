Summary: Prompt script to develop and validate Phase 1 Archivist capabilities—data ingestion, PKG construction, and hybrid search UI.

# Phase 1 — Archivist Development Prompts

## Architecture & Ingestion Setup
Prompt:
"You are implementing Futurnal's Phase 1 Archivist features. Outline the ingestion pipeline that connects to local directories, Obsidian vaults, IMAP email, and GitHub repositories. Specify libraries, data flow diagrams, and error handling strategies that guarantee <1% parsing failure."

## Entity Extraction Pipeline
Prompt:
"Design the prompt templates and batching strategy for a local 8B quantized model that converts normalized text chunks into semantic triples (Subject, Predicate, Object). Include confidence scoring, provenance tracking, and incremental updates when source documents change."

## Personal Knowledge Graph Bootstrapping
Prompt:
"Describe how to construct, version, and query the initial Personal Knowledge Graph using embedded Neo4j. Provide Cypher examples for inserting triples, retrieving context for search, and pruning outdated nodes." 

## Hybrid Search Experience
Prompt:
"Create the UX flow for Futurnal's Archivist search interface. Detail how semantic vector retrieval and graph traversal results are merged, ranked, and displayed to the user. Include fallback behaviors when either subsystem returns low-confidence results."

## Performance & Privacy Validation
Prompt:
"List the performance benchmarks, profiling tools, and privacy checks required before shipping Phase 1. Include methods to verify sub-second query latency and confirm no raw user content leaves the device without consent." 

