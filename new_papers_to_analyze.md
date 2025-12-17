Research Proposal: "Awakening the Ghost"
Context: This proposal directly addresses the "Critical Failures" identified in the Dec 2025 Investment Report, specifically the missing logic in 
TemporalCorrelationDetector
 ("Causal Engine") and 
CuriosityEngine
.

Executive Summary
Futurnal has built a robust "Body" (Ingestion, Privacy, RAG) but lacks the "Brain" required to evolve from an Archivist to an Analyst. To avoid the "Wrapper Trap" and solve the "Horoscope Problem," we must ground our implementation in cutting-edge research.

We propose four core research domains to operationalize the "Vaporware" components.

Domain 1: Causal Discovery from Unstructured Event Logs
Target Component: 
TemporalCorrelationDetector
 (currently a shell) Problem: Determining real causality from noisy personal data without generating spurious correlations ("Horoscopes").

Recommended Papers (2024-2025)
"Event-CausNet: Unlocking Causal Knowledge from Text with LLMs" (2025)

Relevance: Directly addresses extracting quantitative causal features from unstructured textual event logs (like Obsidian notes or Journal entries). This is the exact bridge needed between the ExperientialStream and the Causal Engine.
Application: Use this to parse user journals into valid causal events before correlation analysis.
"ACCESS: A Benchmark for Abstract Causal Event Discovery" (2025)

Relevance: Highlights the failure modes of standard LLMs in causal discovery.
Application: Use their evaluation metrics to validate our 
TemporalCorrelationDetector
 so we can prove to investors that our insights are not hallucinations.
"ICDA: Interactive Causal Discovery through Large Language Model Agents" (2024)

Relevance: Proposes an agent that asks the user to verify causal hypotheses ("Did X cause Y?").
Application: Implement this loop to handle low-confidence correlations. Instead of guessing, the Ghost asks: "I noticed you slept poorly. Was that because of the late coding session?"
Domain 2: Intrinsic Motivation & Curiosity
Target Component: 
CuriosityEngine
 (currently a shell) Problem: The agent is reactive, not proactive. It needs a mathematical framework for "curiosity" to explore knowledge gaps without user prompting.

Recommended Papers (2024)
"Curiosity-driven Autotelic AI Agents" (Oudeyer, 2024)

Relevance: Defines mechanisms for agents to invent their own goals.
Application: Implement an "Entropy Reduction" goal for the Ghost. It should experience "pain" (negative reward) when there are disconnected clusters in the Personal Knowledge Graph (PKG), driving it to ask questions that bridge those gaps.
"Intrinsically-Motivated Humans and Agents in Open-World Exploration" (2025)

Relevance: Correlates information gain with exploration progress.
Application: Use Information Gain as the scoring metric for the 
suggest_exploration_prompts
 method in 
CuriosityEngine
, replacing the current random/placeholder logic.
"DyMemR: Dynamic Memory Enhancement for TKG Reasoning" (2024)

Relevance: Inspired by human memory loss/retention.
Application: Give the Curiosity Engine a "Forgetting Curve." If a user hasn't accessed a memory in 6 months, the Ghost should be "curious" about whether it's still relevant.
Domain 3: Temporal Knowledge Graphs (PKG)
Target Component: Futurnal.pkg (Foundation) Problem: Personal data is not static; facts change (e.g., job title, skills).

Recommended Papers
"Time-Aware Personal Knowledge Graphs" (2024)

Relevance: Specifically targets the problem of capturing lifespan events.
Application: Adopt their schema for "validity intervals" to handle facts that are true only for a specific period, preventing the RAG system from retrieving outdated context.
"Selective TKG Reasoning with CEHis" (2024)

Relevance: Allows the model to abstain when uncertain.
Application: Critical for trust. The Ghost should say "I don't know" rather than hallucinating a temporal connection.
Domain 4: Privacy & Collective Intelligence (Phase 4)
Target Component: Future "Collective Intelligence" Problem: How to learn from all users without reading any user's data.

Recommended Papers
"OpenFedLLM: Training LLMs on Decentralized Private Data" (2024)

Relevance: Framework for federated instruction tuning.
Application: This is the blueprint for Phase 4. We can fine-tune a "Global Ghost" on the causal learnings of local Ghosts without ever uploading the raw journal entries.
"PrE-Text: Training Language Models on Private Federated Data" (2024)

Relevance: generating differentially private synthetic data.
Application: Generate "Synthetic Lives" to train the Causal Engine before deploying it to real users, ensuring safety and robustness.