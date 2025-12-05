# Futurnal LLM Model Registry

**Updated**: December 5, 2025
**Purpose**: Comprehensive list of available and recommended LLM models for Futurnal

> **Module 08 Update**: Multimodal models (Whisper V3, DeepSeek-OCR, Orchestrator-8B) are now fully integrated via `futurnal.extraction` package.

---

## Currently Integrated Models

### 1. Phi-3 Mini (3.8B) ✅
**Model**: `microsoft/Phi-3-mini-4k-instruct`  
**Status**: Integrated, Default for fast tests  
**VRAM**: ~4GB  
**Use Case**: Fast baseline testing, CI/CD  
**Pros**: Fast, no auth required, publicly accessible  
**Cons**: Lower accuracy (59% IFEval)  
**Access**: `FUTURNAL_PRODUCTION_LLM=fast` (or omit)

### 2. Llama 3.1 8B ✅
**Model**: `meta-llama/Llama-3.1-8B-Instruct`  
**Status**: Integrated  
**VRAM**: ~8GB (4-bit quantized)  
**Use Case**: Balanced performance for medium-sized tasks  
**Pros**: Good reasoning, efficient  
**Cons**: Gated (requires HuggingFace auth)  
**Access**: `FUTURNAL_PRODUCTION_LLM=llama3.1`

### 3. Llama 3.3 70B ✅
(UPDATE: THIS MODEL WILL NOT BE USED DUE TO BIG SIZE)
**Model**: `meta-llama/Llama-3.3-70B-Instruct`  
**Status**: Integrated  
**VRAM**: ~24GB (8-bit quantized)  
**Use Case**: Maximum reasoning and structured output  
**Pros**: Superior reasoning, function calling, chain-of-thought  
**Cons**: Large, requires powerful GPU  
**Access**: `FUTURNAL_PRODUCTION_LLM=llama`

### 4. Qwen 2.5 Coder 32B ✅ ⚠️
**Model**: `Qwen/Qwen2.5-Coder-32B-Instruct`  
**Status**: Integrated (network issues during testing)  
**VRAM**: ~16GB (4-bit quantized)  
**Use Case**: Code/extraction optimization  
**Pros**: Optimized for structured extraction  
**Cons**: Large download (~22GB), network issues observed  
**Access**: `FUTURNAL_PRODUCTION_LLM=qwen`

### 5. Bielik 4.5B (Polish) ✅
**Model**: `speakleash/Bielik-4.5B-v3.0-Instruct`
**Status**: Integrated
**VRAM**: ~5GB (4-bit quantized)
**Use Case**: Polish language documents
**Pros**: Optimized for Polish, smaller/faster
**Cons**: Limited to Polish language
**Access**: `FUTURNAL_PRODUCTION_LLM=bielik`

### 6. NVIDIA Orchestrator-8B ✅ (Module 08)
**Model**: `nvidia/Orchestrator-8B`
**Status**: Integrated (Module 08 Phase 3)
**VRAM**: ~8GB (4-bit quantized)
**Use Case**: Multi-turn agentic tasks, tool coordination
**Pros**: Designed for coordinating expert models and tools, intelligent routing
**Cons**: Requires Ollama server
**Access**: `FUTURNAL_ORCHESTRATOR_BACKEND=nvidia` or `auto`
**Implementation**: `src/futurnal/extraction/orchestrator_client.py`

### 7. DeepSeek-OCR ✅ (Module 08)
**Model**: `deepseek-ai/DeepSeek-OCR`
**Status**: Integrated (Module 08 Phase 2)
**VRAM**: ~8GB
**Use Case**: Document OCR, image-to-text extraction
**Pros**: SOTA OCR performance (>98% accuracy), layout preservation
**Cons**: Requires image preprocessing
**Access**: `FUTURNAL_VISION_BACKEND=deepseek` or `auto`
**Implementation**: `src/futurnal/extraction/ocr_client.py`
**Fallback**: Tesseract OCR

### 8. Whisper Large V3 ✅ (Module 08)
**Model**: `openai/whisper-large-v3`
**Status**: Integrated (Module 08 Phase 1)
**VRAM**: ~4GB (via Ollama) or ~10GB (HuggingFace)
**Use Case**: Audio transcription, ASR
**Pros**: SOTA automatic speech recognition, 98+ languages, temporal segments
**Cons**: Requires audio files in supported formats
**Access**: `FUTURNAL_AUDIO_BACKEND=ollama` or `auto`
**Implementation**: `src/futurnal/extraction/whisper_client.py`
**Fallback**: HuggingFace Transformers

### 9. Kimi-K2-Thinking ✅
**Model**: `moonshotai/Kimi-K2-Thinking`
**Status**: Integrated
**VRAM**: ~16GB (4-bit quantized)
**Use Case**: Complex reasoning tasks, advanced thinking
**Pros**: Advanced reasoning capabilities, alternative to Llama 3.3 70B
**Cons**: May require custom Ollama model setup
**Access**: `FUTURNAL_PRODUCTION_LLM=kimi`
**Implementation**: `src/futurnal/extraction/local_llm_client.py`

### 10. GPT-OSS-20B-Derestricted ✅
**Model**: `ArliAI/gpt-oss-20b-Derestricted`
**Status**: Integrated
**VRAM**: ~12GB (4-bit quantized)
**Use Case**: Unrestricted content processing
**Pros**: No content moderation guardrails, useful for research
**Cons**: Use responsibly - no safety filters
**Access**: `FUTURNAL_PRODUCTION_LLM=gpt-oss`
**Implementation**: `src/futurnal/extraction/local_llm_client.py`
**Warning**: Use carefully, ensure compliance with use case

---

## Future Model Recommendations

### For Content Safety

#### Bielik-Guard 0.1B
**Model**: `speakleash/Bielik-Guard-0.1B-v1.0`  
**Use Case**: Content moderation, safety filtering  
**Why**: Lightweight safety model  
**Integration Priority**: LOW  
**Proposed Use**: Optional content filtering layer

---

## Model Selection Guide

### By VRAM Availability

| VRAM      | Recommended Model           | Performance |
|-----------|-----------------------------|-------------|
| 4GB       | Phi-3 Mini 3.8B            | Fast/Basic  |
| 5-8GB     | Bielik 4.5B / Llama 3.1 8B | Good        |
| 12GB      | GPT-OSS-20B                | Good (unrestricted) |
| 16GB      | Qwen 2.5 32B / Kimi-K2     | Excellent   |
| 24GB+     | Llama 3.3 70B              | Best        |

### By Use Case

| Task                    | Recommended Model      | Status |
|-------------------------|------------------------|--------|
| Entity extraction (EN)  | Qwen 2.5 32B / Llama 3.1 8B | ✅ Integrated |
| Entity extraction (PL)  | Bielik 4.5B v3        | ✅ Integrated |
| Reasoning tasks         | Llama 3.3 70B / Kimi-K2 | ✅ Integrated |
| Advanced reasoning      | Kimi-K2-Thinking      | ✅ Integrated |
| Unrestricted content    | GPT-OSS-20B           | ✅ Integrated |
| Fast testing            | Phi-3 Mini            | ✅ Integrated |
| OCR preprocessing       | DeepSeek-OCR          | ✅ Integrated (Module 08) |
| Audio transcription     | Whisper V3            | ✅ Integrated (Module 08) |
| Agentic coordination    | Orchestrator-8B       | ✅ Integrated (Module 08) |

### By Language

| Language | Model           |
|----------|-----------------|
| English  | Llama / Qwen    |
| Polish   | Bielik 4.5B     |
| Multi    | Llama 3.3 70B   |

---

## Integration Status

### ✅ Ready to Use (Text LLMs)
- Phi-3 Mini 3.8B
- Llama 3.1 8B
- Llama 3.3 70B
- Bielik 4.5B v3 (Polish)
- Qwen 2.5 32B (with network caveats)
- **Kimi-K2-Thinking** - Advanced reasoning via `FUTURNAL_PRODUCTION_LLM=kimi`
- **GPT-OSS-20B-Derestricted** - Unrestricted use via `FUTURNAL_PRODUCTION_LLM=gpt-oss`

### ✅ Ready to Use (Module 08: Multimodal)
- **NVIDIA Orchestrator-8B** - Agentic coordination via `orchestrator_client.py`
- **DeepSeek-OCR** - Document preprocessing via `ocr_client.py`
- **Whisper Large V3** - Audio transcription via `whisper_client.py`

### 📋 Under Consideration
- Bielik-Guard (content safety)

---

## Usage Examples

### Quick Testing (Fast)
```bash
pytest tests/ -v
# Uses Phi-3 Mini by default
```

### Production Testing (Qwen)
```bash
export FUTURNAL_PRODUCTION_LLM=qwen
pytest tests/ -v
```

### Production Testing (Llama 3.1)
```bash
export FUTURNAL_PRODUCTION_LLM=llama3.1
pytest tests/ -v
```

### Polish Language Documents
```bash
export FUTURNAL_PRODUCTION_LLM=bielik
pytest tests/ -v
```

### Auto-Select Based on VRAM
```bash
export FUTURNAL_PRODUCTION_LLM=auto
pytest tests/ -v
```

### Advanced Reasoning (Kimi-K2)
```bash
export FUTURNAL_PRODUCTION_LLM=kimi
pytest tests/ -v
# Alternative to Llama 3.3 70B for complex reasoning
```

### Unrestricted Content (GPT-OSS-20B)
```bash
export FUTURNAL_PRODUCTION_LLM=gpt-oss
pytest tests/ -v
# WARNING: No content moderation - use responsibly
```

---

## Implementation Notes

### Authentication
Gated models require HuggingFace authentication:
```bash
huggingface-cli login
# or
hf auth login
```

### Quantization
- 4-bit: Recommended for most cases (saves VRAM)
- 8-bit: Better quality for large models (24GB+)
- None: Full precision (CPU/MPS only)

### Model Loading
Models are cached in:
```
~/.cache/huggingface/hub/
```

First download can take time (10-60 minutes for large models).

---

## Module 08: Multimodal Pipeline (IMPLEMENTED)

### Multi-Modal Pipeline (Now Active)
1. **Audio Input** → Whisper V3 → Text (via `AudioAdapter`)
2. **Image Input** → DeepSeek-OCR → Text (via `ImageAdapter`)
3. **Scanned PDF** → DeepSeek-OCR → Text (via `ScannedPDFAdapter`)
4. **Text Processing** → Current entity extraction (via existing adapters)
5. **Coordination** → Orchestrator-8B → Optimal routing (via `MultiModalRouter`)

### Unified API
```python
from futurnal.extraction import extract_from_any_source

# Single file - auto-detects modality
doc = await extract_from_any_source("recording.wav")

# Mixed batch - orchestrator coordinates
docs = await extract_from_any_source([
    "slides.pdf",
    "recording.wav",
    "notes.md"
], strategy="parallel")
```

### Specialized Extraction Paths
- **OCR path**: DeepSeek-OCR + Qwen 2.5 Coder
- **Audio path**: Whisper V3 + Llama 3.1 8B
- **Reasoning path**: Llama 3.3 70B + Orchestrator

### Language-Specific
- Polish documents: Bielik 4.5B
- English documents: Qwen/Llama
- Multilingual: Llama 3.3 70B with language detection

---

## References

- [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [Qwen 2.5 Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
- [Bielik 4.5B](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct)
- [Orchestrator-8B](https://huggingface.co/nvidia/Orchestrator-8B)
- [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Whisper V3](https://huggingface.co/openai/whisper-large-v3)
