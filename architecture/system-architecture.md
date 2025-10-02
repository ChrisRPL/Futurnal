Summary: Details Futurnal's layered technical architecture including on-device inference, personal knowledge graph, and causal insight engine.

# System Architecture Overview

## Architectural Principles
- **Privacy-Enabled AI Evolution:** All raw experiential data remains on-device, enabling the deep trust required for AI personalization and continuous learning.
- **Adaptive Intelligence Architecture:** Combine efficient local models with optional cloud consultations, allowing AI capabilities to scale as intelligence evolves.
- **Experiential Memory System:** Maintain a continuously updated Personal Knowledge Graph (PKG) that serves as the AI's evolving memory of user experience.
- **Intelligence Development Framework:** Enable progressive AI evolution from basic pattern recognition to sophisticated causal reasoning about personal dynamics.

## Layer 1 — AI Foundation & Evolution Platform
- **Personal AI Core:** Utilize Ollama or `llama.cpp` to host the user's personal AI model (e.g., quantized Llama-3.1 8B, Mistral 7B) which evolves from generic capabilities toward experiential intelligence through continuous learning from user data.
- **Evolution-Ready Hardware:** Tailor performance to Apple Silicon and mid/high-tier GPUs (≥12 GB VRAM) with architecture designed to scale as AI capabilities become more sophisticated.
- **Autonomous Operation:** Ensure the evolving AI operates independently without network dependency for core experiential understanding and reasoning.
- **Intelligence Consultation:** For advanced reasoning beyond current AI capabilities, enable anonymous consultation with cloud models using structured queries (never raw data), gradually reducing dependency as local intelligence evolves.

## Layer 2 — Experiential Memory & Learning System
- **Experience Ingestion:** Use Unstructured.io to transform diverse experiential data into normalized understanding; apply intelligent chunking that preserves experiential context.
- **Pattern Learning:** The evolving AI analyzes experiential data with increasingly sophisticated prompts to extract meaningful patterns, relationships, and personal dynamics.
- **Memory Architecture:** Persist experiential understanding in embedded Neo4j as a living memory system that grows more sophisticated over time.
- **Contextual Understanding:** Combine embeddings in ChromaDB/Weaviate with graph memory to enable the AI's developing contextual intelligence.
- **Continuous Learning:** Dynamically update experiential memory as new data arrives, enabling the AI to develop increasingly nuanced understanding of user patterns.

## Layer 3 — Intelligent Reasoning & Sophisticated Analysis
- **Contextual Intelligence:** For user interactions, the AI combines its experiential memory with learned patterns to provide responses that demonstrate genuine understanding of personal context.
- **Autonomous Pattern Recognition:** The AI continuously develops more sophisticated analytical capabilities, autonomously discovering correlations and thematic insights in experiential data.
- **Reasoning Development:** The AI evolves its hypothesis generation capabilities, learning to apply increasingly sophisticated reasoning principles to personal patterns.
- **Collaborative Investigation:** Guide users through evidence-based exploration of personal dynamics, with the AI demonstrating growing sophistication in understanding temporal patterns, contextual factors, and personal causality.

## Privacy & Intelligence Evolution
- **Evolutionary Trust:** Provide granular control over experiential data access, enabling users to guide their AI's development while maintaining complete privacy sovereignty.
- **Collective Intelligence Growth:** Future opt-in federated learning will enable AI models to benefit from collective experiential patterns while preserving individual privacy, accelerating the evolution toward sophisticated personalized intelligence.

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

## Evolution Risks & Safeguards
- **Intelligence Development Guardrails:** Ensure AI reasoning remains grounded in evidence and presents insights as collaborative discoveries rather than authoritative judgments.
- **Adaptive Performance:** Provide intelligent resource management that adapts to hardware capabilities while maintaining AI development trajectory.
- **Experiential Quality:** Implement intelligent filtering to focus AI learning on meaningful experiential patterns while excluding noise or harmful content.

