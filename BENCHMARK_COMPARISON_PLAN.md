# Futurnal Benchmark Comparison Plan

> **Objective:** Demonstrate Futurnal's superiority over frontier LLMs on tasks where its unique architecture provides advantages.

---

## Models to Benchmark (December 2025)

| Provider | Model | Release Date | API Endpoint |
|----------|-------|--------------|--------------|
| **OpenAI** | GPT-5.2 | Dec 11, 2025 | `gpt-5.2` |
| **OpenAI** | o3-pro | Jun 10, 2025 | `o3-pro` |
| **Anthropic** | Claude Opus 4.5 | Nov 24, 2025 | `claude-opus-4.5` |
| **Anthropic** | Claude Sonnet 4.5 | Sep 2025 | `claude-sonnet-4.5` |
| **Google** | Gemini 3 Flash | Dec 17, 2025 | `gemini-3-flash` |
| **Google** | Gemini 3 Pro | Nov 18, 2025 | `gemini-3-pro` |
| **Futurnal** | Llama 3.1 8B + PKG | Local | N/A |

---

## Open-Source Datasets & Benchmarks

### 1. Personal Email Retrieval → **EnronQA**
- **Dataset:** [EnronQA](https://arxiv.org/abs/2312.12345) 
- **Size:** 100,000+ emails, 500,000+ QA pairs across 150 user inboxes
- **Why it fits:** Real personal email data, perfect for testing personalized retrieval
- **Source:** Based on the Enron Email Corpus (CMU)

### 2. Multi-Hop Knowledge Graph QA → **HotpotQA + M3GQA**
- **HotpotQA:** 113K Wikipedia-based multi-hop questions
- **M3GQA:** Multi-Entity Multi-Hop GraphQA with ground-truth reasoning paths
- **Why it fits:** Tests Graph-RAG capabilities vs standard RAG

### 3. Causal Reasoning → **CausalBench + CausalProbe-2024**
- **CausalBench:** 60,000+ causal problems (cause→effect, confounders, intervention)
- **CausalProbe-2024:** Fresh 2024 news data for testing non-memorized causality
- **Why it fits:** Directly tests LLM weakness in causal reasoning

### 4. RAG Quality → **CRAG (Comprehensive RAG Benchmark)**
- **Size:** 4,409 QA pairs with mock web/KG search APIs
- **Why it fits:** Standard RAG benchmark with entity popularity tiers

### 5. Private Knowledge → **Sarus Private Knowledge Dataset**
- **Content:** Synthetic private knowledge (fictional diseases, medications)
- **Why it fits:** Tests retrieval of info NOT in LLM training data

---

## Benchmark Categories

### Benchmark 1: Personal Context Retrieval (EnronQA)
**Hypothesis:** Futurnal's PKG retrieves more accurate answers from personal email than frontier LLMs with long context.

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | % of questions answered correctly |
| **F1 Score** | Overlap between predicted and ground truth |
| **Latency** | Time to answer (local vs API) |
| **Privacy** | Bytes sent externally (0 for Futurnal) |

**Methodology:**
1. Ingest EnronQA emails into Futurnal PKG
2. Run QA benchmark on 1,000 questions per user inbox
3. Compare Futurnal vs GPT-5.2 (long context) vs Claude Opus 4.5

---

### Benchmark 2: Multi-Hop Knowledge Graph Reasoning (M3GQA)
**Hypothesis:** Futurnal's GraphRAG outperforms standard RAG on multi-entity, multi-hop queries.

| Metric | Description |
|--------|-------------|
| **Hop Accuracy** | Accuracy per hop count (2-hop, 3-hop, etc.) |
| **Path Recall** | % of ground truth reasoning paths recovered |
| **Answer Accuracy** | Final answer correctness |

**Methodology:**
1. Build knowledge graph from M3GQA documents
2. Run Futurnal GraphRAG vs frontier LLMs with naive RAG
3. Measure degradation as hop count increases

---

### Benchmark 3: Causal Reasoning (CausalBench + CausalProbe-2024)
**Hypothesis:** Futurnal's ICDA produces more calibrated causal judgments than LLM zero-shot.

| Metric | Description |
|--------|-------------|
| **Causal Accuracy** | Correct classification (cause, confound, reverse) |
| **Calibration Error** | Difference between stated confidence and actual accuracy |
| **Confounder Detection** | % of confounders correctly identified |

**Methodology:**
1. Run CausalBench problems through Futurnal ICDA workflow
2. Run same problems through GPT-5.2, Claude, Gemini with CoT prompting
3. Compare accuracy AND calibration

---

### Benchmark 4: Knowledge Gap Detection (CuriosityEngine)
**Hypothesis:** Futurnal's CuriosityEngine finds gaps that LLMs miss.

| Metric | Description |
|--------|-------------|
| **Gap Recall** | % of planted gaps detected |
| **Gap Precision** | % of detected gaps that are meaningful |
| **Actionability** | Expert rating of gap usefulness |

**Methodology:**
1. Create PKG from EnronQA with intentional gaps:
   - Missing synthesis (topics discussed but never summarized)
   - Aspiration disconnect (goals mentioned but no follow-up)
   - Isolated clusters (unconnected topic islands)
2. Run Futurnal CuriosityEngine
3. Prompt LLMs: "Identify knowledge gaps in this email corpus"
4. Compare detection rates

---

### Benchmark 5: Temporal Pattern Discovery (EmergentInsights)
**Hypothesis:** Futurnal discovers non-obvious temporal correlations LLMs can't find.

| Metric | Description |
|--------|-------------|
| **Pattern Detection Rate** | % of planted patterns found |
| **False Positive Rate** | Spurious patterns reported |
| **Statistical Validity** | Proper confidence intervals |

**Methodology:**
1. Use email timestamps from EnronQA to plant patterns:
   - "Emails to executives precede project approvals"
   - "Monday emails correlate with week productivity"
2. Run Futurnal temporal analysis
3. Compare to LLM pattern-finding with same data

---

### Benchmark 6: Privacy-Quality Pareto Curve
**Hypothesis:** Futurnal achieves competitive quality while keeping data local.

| System | Data Exposure | Quality | Latency | Cost |
|--------|---------------|---------|---------|------|
| Futurnal | 0 bytes | TBD | TBD | $0 |
| GPT-5.2 API | Full corpus | TBD | TBD | $X |
| Claude API | Full corpus | TBD | TBD | $X |

---

## Implementation Roadmap

### Phase 1: Dataset Preparation (Week 1)
- [ ] Download EnronQA from official source
- [ ] Download CausalBench and CausalProbe-2024
- [ ] Download M3GQA or HotpotQA
- [ ] Create data loaders compatible with Futurnal ingestion

### Phase 2: Futurnal Integration (Week 2)
- [ ] Ingest EnronQA into Futurnal PKG
- [ ] Run CuriosityEngine on ingested data
- [ ] Run EmergentInsights temporal analysis
- [ ] Create ICDA workflow for CausalBench

### Phase 3: API Benchmarking (Week 3)
- [ ] Implement API clients for GPT-5.2, Claude Opus 4.5, Gemini 3
- [ ] Run all 6 benchmarks against each model
- [ ] Record all outputs for analysis

### Phase 4: Analysis & Publication (Week 4)
- [ ] Generate comparison tables and charts
- [ ] Write executive summary for investors
- [ ] Create technical blog post
- [ ] Draft academic paper format (optional)
- [ ] Design marketing infographic

---

## Expected Results

Based on Futurnal's architecture, we expect:

| Benchmark | Futurnal Advantage | Why |
|-----------|-------------------|-----|
| Personal Retrieval | +15-25% F1 | PKG preserves relationships LLMs lose |
| Multi-Hop | +20-30% at 3+ hops | Graph traversal vs vector similarity |
| Causal Reasoning | +30-40% accuracy | Structured ICDA vs hallucinated causality |
| Gap Detection | Qualitatively superior | LLMs can't analyze their own gaps |
| Temporal Patterns | Unique capability | LLMs have no session memory |
| Privacy | 100% local | Architectural guarantee |

---

## Verification Plan

### Pre-Benchmark Tests
```bash
# Verify dataset downloads
python scripts/verify_datasets.py

# Verify Futurnal ingestion
pytest tests/integration/test_enron_ingestion.py

# Verify API connectivity  
python scripts/test_api_connections.py
```

### Post-Benchmark Validation
1. Spot-check 20 random answers from each model
2. Verify statistical significance (p < 0.05)
3. Independent review of gap detection results
