# Futurnal LLM Model Registry

**Updated**: December 3, 2025  
**Purpose**: Comprehensive list of available and recommended LLM models for Futurnal

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

---

## Future Model Recommendations

### For Agentic Coordination

#### NVIDIA Orchestrator-8B
**Model**: `nvidia/Orchestrator-8B`  
**Use Case**: Multi-turn agentic tasks, tool coordination  
**Why**: Designed for coordinating expert models and tools  
**Integration Priority**: HIGH  
**Proposed Use**: Could enhance GRPO experiential learning coordination

### For OCR Tasks

#### DeepSeek-OCR
**Model**: `deepseek-ai/DeepSeek-OCR`  
**Use Case**: Document OCR, image-to-text extraction  
**Why**: State-of-the-art OCR performance  
**Integration Priority**: MEDIUM  
**Proposed Use**: Extract text from scanned documents, images before entity extraction

### For Advanced Reasoning

#### Kimi-K2-Thinking
**Model**: `moonshotai/Kimi-K2-Thinking`  
**Use Case**: Complex reasoning tasks  
**Why**: Advanced reasoning capabilities  
**Integration Priority**: MEDIUM  
**Proposed Use**: Alternative to Llama 3.3 70B for reasoning-heavy extraction

### For Unrestricted Use

#### GPT-OSS-20B-Derestricted  
**Model**: `ArliAI/gpt-oss-20b-Derestricted`  
**Use Case**: Unrestricted content processing  
**Why**: No content moderation guardrails  
**Integration Priority**: LOW  
**Caution**: Use carefully, ensure compliance with use case

### For Audio/Speech

#### Whisper Large V3
**Model**: `openai/whisper-large-v3`  
**Use Case**: Audio transcription, ASR  
**Why**: SOTA automatic speech recognition  
**Integration Priority**: MEDIUM  
**Proposed Use**: Transcribe audio notes/meetings before entity extraction

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
| 16GB      | Qwen 2.5 32B Coder         | Excellent   |
| 24GB+     | Llama 3.3 70B              | Best        |

### By Use Case

| Task                    | Recommended Model      |
|-------------------------|------------------------|
| Entity extraction (EN)  | Qwen 2.5 32B / Llama 3.1 8B |
| Entity extraction (PL)  | Bielik 4.5B           |
| Reasoning tasks         | Llama 3.3 70B         |
| Fast testing            | Phi-3 Mini            |
| OCR preprocessing       | DeepSeek-OCR (future) |
| Audio transcription     | Whisper V3 (future)   |
| Agentic coordination    | Orchestrator (future) |

### By Language

| Language | Model           |
|----------|-----------------|
| English  | Llama / Qwen    |
| Polish   | Bielik 4.5B     |
| Multi    | Llama 3.3 70B   |

---

## Integration Status

### ✅ Ready to Use
- Phi-3 Mini 3.8B
- Llama 3.1 8B  
- Llama 3.3 70B
- Bielik 4.5B
- Qwen 2.5 32B (with network caveats)

### 🔄 Planned Integration
- NVIDIA Orchestrator-8B (agentic coordination)
- DeepSeek-OCR (document preprocessing)
- Whisper Large V3 (audio pipeline)

### 📋 Under Consideration
- Kimi-K2-Thinking (reasoning alternative)
- GPT-OSS-20B-Derestricted (use case dependent)
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

## Future Enhancements

### Multi-Modal Pipeline
1. **Audio Input** → Whisper V3 → Text
2. **Image Input** → DeepSeek-OCR → Text  
3. **Text Processing** → Current entity extraction
4. **Coordination** → Orchestrator-8B → Optimal routing

### Specialized Extraction
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
