## Feature · Agentic Entity & Relationship Extraction (v2.0: Deep Research Agent for PKG)

This feature implements a State-of-the-Art (SOTA) agentic workflow to convert raw, normalized documents into a structural **Personal Knowledge Graph (PKG)**. Unlike traditional one-pass NLP extraction, this pipeline employs an **Iterative Deep Research** paradigm, leveraging Graph Retrieval-Augmented Generation (GraphRAG) techniques to ensure high precision, multi-hop capability, and continuous improvement via learned feedback.

The resulting knowledge base is formally defined as a **Graph Knowledge Database (GKB) $G = \{E, R, K\}$**, encompassing extracted entities ($E$), structural relations ($R$), and associated chunk-level textual context ($K$).

### I. Goal & SOTA Context

| Attribute | v1.0 (Previous) | v2.0 (SOTA Update) |
| :--- | :--- | :--- |
| **Core Paradigm** | Simple LLM Triple Extraction | **Agentic Iterative Reasoning** (Deep Research Agents - DRAs) |
| **Data Structure** | Semantic Triples (S, P, O) | **Graph Knowledge Database (GKB)** $G = \{E, R, K\}$ |
| **Accuracy Mechanism** | Confidence Score & Rule-based Filtering | **Context Refinement** & **Self-Refine** / **LLM-as-Judge** |
| **Future Proofing** | Manual Feedback Loop | **Experience-Driven Closed-Loop Learning** (MUSE/ACE) |
| **Efficiency** | Quantized LLM inference | Quantized LLM + **KV-Cache Compression** |

### II. Technical Requirements (For Claude Code Implementation)

#### 1. Agent Architecture and Inference

1.  **Agent Orchestration:** The core pipeline must be implemented as an iterative agent, utilizing a **ReAct loop** (Reasoning and Acting) structure (or similar tool-integrated loop), ensuring the ability to perform multi-step reasoning before producing the final knowledge output.
2.  **LLM Engine & Efficiency:** The Agent must default to highly efficient **quantized LLMs** (e.g., Llama-3.1 8B, Qwen3-8B) to meet the **on-device inference default** requirement. Implement support for inference optimizations like **KV-Cache compression** or tiling techniques to handle long input contexts effectively.
3.  **Tool Call Format:** All interaction steps (e.g., invoking the sub-query module, extracting triples) must follow a predictable, structured format (e.g., JSON or XML tags) suitable for agent parsing and execution.

#### 2. Data Structures and Formats

*   **Input:** Normalized document chunks, each associated with provenance (source, timestamp, chunk hash).
*   **Intermediate Output (GKB Triples):** Triples must include the extracted entity names, the explicit relationship, and critical metadata:
    *   `(Subject, Predicate, Object, {Provenance: chunk\_ID, Confidence: score, Context\_Snippet: text})`
*   **Memory Structure (for RL/SFT):** The system must generate data points capturing the trajectory of extraction (successful or failed attempts) to enable future tuning, akin to generating memory items defined by **Title, Description, and Content** insights.

### III. Core Agentic Pipeline (Multi-Stage Extraction)

The feature execution uses a multi-stage process inspired by **GraphRAG** principles:

#### Stage 1: Initial Context & Query Decomposition (Required for Complex Documents)

1.  **Initial Retrieval (RAG Baseline):** Retrieve initial text chunks ($K_{initial}$) based on source document embeddings (using models like **OpenAI text-embedding-3-small** or other SOTA encoders like **BGE-M3**).
2.  **Query Decomposition (PQD):** For documents requiring deep analysis (simulating multi-hop reasoning or Open-Ended Deep Research (OEDR)), the LLM agent must decompose the implicit extraction goal $Q_{Extraction}$ into an ordered sequence of focused sub-queries, $\{q_1, q_2, \ldots, q_n\}$.
    *   *Implementation Note:* Utilize a specialized prompt template (PQD) to force the LLM to output sub-queries in a structured, executable list, similar to examples in GraphRAG implementations.

#### Stage 2: Dual-Channel Extraction and Refinement

1.  **Iterative Retrieval:** Execute each sub-query $q_i$ against both the textual document corpus (Semantic Channel) and the growing PKG (Relational Channel - if prior data exists).
2.  **Context Refinement:** After each retrieval, apply a process to filter redundancy and highlight relevance (Context Refinement $C'_{q_i} = PCR(q_i, C_{q_i})$). This prevents context overload and improves factuality, addressing a critical weakness of long-context LLMs.
3.  **Knowledge Generation (The Extractor):** The LLM processes the refined context $C'_{q_i}$ and generates:
    *   **Relational Output:** Structured triples, potentially using unresolved placeholders (e.g., `Entity#1`) if multi-hop traversal is ongoing.
    *   **Semantic Output:** Descriptive entities/facts (Name, Properties, Description - $e = \{e_{name}, e_{prop}, e_{desc}\}$) and the source chunk context $K$.

#### Stage 3: Post-Processing, Normalization, and Validation

1.  **Entity Normalization (Hybrid Approach):** Apply Named Entity Recognition (**NER**) combined with LLM logic (the existing `spaCy/LLM hybrid`) to resolve coreference, map aliases, and normalize entities.
2.  **Robustness Enhancement (Fuzzing/Abstention):** Implement logic inspired by synthetic data generation techniques to enhance robustness:
    *   **Fuzzy Entity Matching:** Substitute specific names with generic descriptions (e.g., “Albert Einstein” becomes “a famous physicist”).
    *   **Fuzzy Static Data:** Replace specific numbers/dates with broader ranges/vague descriptions (e.g., “1992” becomes “early 1990s”).
3.  **Confidence Scoring and Filtering:** Calculate confidence based on two signals: model score (intrinsic confidence) and external validation (e.g., **LLM-as-Judge** verification). Reject or flag triples that fall below a configurable quality threshold.

#### Stage 4: Feedback and Memory Capture (Self-Evolution)

1.  **Feedback Capture:** Log decisions (acceptance/rejection of triples) and successful/failed trajectories.
2.  **Reflector Component:** Implement a mechanism (conceptually acting as the **Reflector** in the ACE framework) that critiques past extraction **trajectories** to extract high-level lessons or insights.
3.  **Delta Update Generation:** The Reflector generates structured, reusable **Memory Items** (delta contexts) capturing reusable strategies or common pitfalls. This data should be logged in a structure (e.g., JSON list of memory items) suitable for future **Supervised Fine-Tuning (SFT)** or **Reinforcement Learning (RL)**, such as **Search-R1** or **DeepSeek-R1**, which incentivize reasoning capability.

---

### IV. Useful Links and Referenced Papers

| Citation | Title/Description | Link/Reference |
| :--- | :--- | :--- |
| **** | **Graph Knowledge Database (GKB)** Structure & **Context Refinement** (Critical for GraphRAG) | Excerpts from "2509.22009v2.pdf" |
| **** | **Entity Tree Construction** and **Fuzzing** (Ensuring robustness and synthetic data principles) | Excerpts from "2509.25189v1.pdf" |
| **** | **Self-Refine / Chain-of-Verification** (Mechanisms to reduce hallucination and improve factuality) | Madaan et al. 2023 (`Self-refine`); Dhuliawala et al. 2023 (`Chain-of-verification`) |
| **** | **Reinforcement Learning for Search/Reasoning** (SOTA methods for training agents to use tools like extraction) | Jin et al. 2025 (`Search-R1`); Guo et al. 2025 (`DeepSeek-R1`) |
| **** | **Agentic Memory Management (ACE Framework)** (Iterative refinement and incremental delta updates for learning) | Excerpts from "https://arxiv.org/pdf/2510.04618" |
| **** | **LLM Model Selection** (Reference for quantized model size) | Meta Llama 3.1 Model Card; Existing Feature Description |
| **** | **Retrieval-Augmented Generation (RAG)** (Foundational paradigm) | Lewis et al. 2020 (RAG) |
| **** | **Multi-hop Question Answering** (Underlying complexity addressed by Query Decomposition) | Musique; HotpotQA |
| **** | **LoRA** (Low-Rank Adaptation, applicable if light tuning is required) | Hu et al. 2022 (LoRA) |