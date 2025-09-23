Summary: Details Futurnal's layered technical architecture including on-device inference, personal knowledge graph, and causal insight engine.

# System Architecture Overview

## Architectural Principles
- **Privacy by Design:** All raw personal data remains on-device by default.
- **Hybrid Intelligence:** Combine efficient local models with optional, structured cloud escalations.
- **Dynamic Knowledge Representation:** Maintain a continuously updated Personal Knowledge Graph (PKG) augmented with vectors.
- **Causal Insight Enablement:** Enable proactive detection of patterns and guided causal investigations.

## Layer 1 — On-Device Foundation
- **Local LLM Serving:** Utilize Ollama or `llama.cpp` to host quantized compact models (e.g., Llama-3.1 8B, Mistral 7B) for ingestion, entity extraction, and RAG queries.
- **Hardware Optimization:** Tailor performance to Apple Silicon and mid/high-tier GPUs (≥12 GB VRAM) with clear user guidance.
- **Offline-first Experience:** Ensure core search and retrieval operate without network connectivity.
- **Cloud Escalation Path:** Escalate anonymized, structured graph queries to advanced cloud models (GPT-4o, Claude 3.5 Sonnet) only with explicit user consent for complex reasoning tasks.

## Layer 2 — Dynamic Personal Knowledge Graph
- **Ingestion & Transformation:** Use Unstructured.io to parse diverse file types into normalized documents; apply LangChain text splitters for chunking.
- **Entity & Relationship Extraction:** Run the local LLM with tailored prompts to produce semantic triples (Subject, Predicate, Object).
- **Graph Storage:** Persist triples in an embedded Neo4j instance (or equivalent) for rich traversal via Cypher queries.
- **Vector Enrichment:** Store embeddings in ChromaDB or Weaviate to enable hybrid search strategies that merge semantic similarity with graph context.
- **Dynamic Updates:** Continuously reconcile changes from data sources to keep the PKG current and versioned.

## Layer 3 — Proactive Insight & Causal Exploration
- **GraphRAG Context Retrieval:** For user queries, combine vector results with graph traversals to assemble context packages, reducing hallucinations and supporting multi-hop reasoning.
- **Emergent Insight Detection:** Periodically run graph analytics—community detection, centrality measures—to uncover significant correlations and thematic clusters.
- **LLM-Guided Hypothesis Generation:** Prompt the model with causal inference principles to generate candidate explanations for identified patterns.
- **Guided Investigation Workflow:** Query the PKG to validate or refute hypotheses by examining timelines, confounders, and linked entities; present conversational walkthroughs.

## Privacy & Personalization Enhancements
- **User Control:** Provide per-source permission settings and transparent audit trails of model accesses.
- **Federated Learning Roadmap:** Plan for opt-in, privacy-preserving model updates that aggregate encrypted gradient signals without accessing raw data.

## Frontend & Interaction Layer
- **Application Shell:** Build cross-platform desktop clients using Electron or Tauri with a dark-mode-first interface.
- **Visualization Tools:** Offer interactive graph exploration and insight dashboards tailored to each phase (Archivist, Analyst, Guide).
- **Notification & Prompting:** Surface proactive alerts and causal exploration prompts while preserving user context.

## Technical Dependencies (Initial Stack)
- Model Serving: Ollama / `llama.cpp`
- Data Parsing: Unstructured.io
- Orchestration: LangChain, LangGraph
- Graph Store: Neo4j (embedded)
- Vector Store: ChromaDB or Weaviate
- Frontend: Electron or Tauri

## Risks & Mitigations
- **Causal Misinterpretation:** Mitigate by presenting hypotheses as exploratory, not factual; emphasize user-led validation.
- **Hardware Variability:** Provide profiling tools and feature toggles for resource-constrained environments.
- **Data Quality Variance:** Implement cleansing filters to ignore toxic or irrelevant content when ingesting community data (e.g., GitHub issues).

