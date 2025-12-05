"""Local LLM client for privacy-first extraction.

This module provides LOCAL quantized LLM inference for extraction tasks,
following Futurnal's privacy-first principles:
- On-device inference (Llama-3.1 8B, Qwen3-8B)
- No cloud APIs by default
- Ghost model remains frozen
- Experiential learning via token priors (not parameter updates)

Optional cloud escalation only with explicit user consent.
"""

import json
import os
from typing import Any, Dict, Optional, Protocol, List
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM interactions (matches existing Protocol in codebase)."""

    def extract(self, prompt: str, **kwargs) -> Any:
        """Extract information using LLM.

        Args:
            prompt: The extraction prompt
            **kwargs: Additional parameters

        Returns:
            Extraction result
        """
        ...

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using LLM.

        Args:
            prompt: The generation prompt
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        ...


class LocalLLMBackend(str, Enum):
    """Supported local LLM backends for entity extraction."""
    # Production Models
    PHI_3_MINI = "microsoft/Phi-3-mini-4k-instruct"  # Fast baseline (3.8B, open access)
    LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"  # Good balance (8B, gated)
    LLAMA_3_3_70B = "meta-llama/Llama-3.3-70B-Instruct"  # Best reasoning (70B, 24GB VRAM)
    QWEN_2_5_32B_CODER = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Optimized extraction (32B, 16GB VRAM)
    BIELIK_4_5B = "speakleash/Bielik-4.5B-v3.0-Instruct"  # Polish language model (4.5B)

    # Advanced Reasoning Models
    KIMI_K2_THINKING = "moonshotai/Kimi-K2-Thinking"  # Advanced reasoning/thinking (alternative to 70B)

    # Unrestricted Models
    GPT_OSS_20B = "ArliAI/gpt-oss-20b-Derestricted"  # Unrestricted use (20B)

    # Alternative models
    QWEN_3_8B = "Qwen/Qwen2.5-7B-Instruct"  # Alternative production
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"  # Gated, needs auth
    GPT2 = "gpt2"  # Lightweight testing fallback (no auth required)


class QuantizedLocalLLM:
    """Privacy-first local LLM client using quantized models.

    Implements Futurnal's Option B principles:
    - Ghost model remains frozen (no parameter updates)
    - On-device inference for privacy
    - Quantization for efficiency (8-bit/4-bit)
    - Experiential learning via token priors (separate from this class)
    """

    def __init__(
        self,
        model_name: str = LocalLLMBackend.LLAMA_3_2_3B.value,
        device: str = "auto",
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        max_memory: Optional[Dict[int, str]] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize quantized local LLM.

        Args:
            model_name: Model identifier (Llama, Qwen, etc.)
            device: Device placement ("auto", "cuda", "cpu", "mps")
            load_in_8bit: Use 8-bit quantization (good balance)
            load_in_4bit: Use 4-bit quantization (more aggressive)
            max_memory: Per-device memory limits
            cache_dir: Model cache directory (default: ~/.cache/huggingface)
        """
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig
            )
            import torch
        except ImportError as e:
            raise ImportError(
                "Required packages not installed. "
                "Ensure transformers and torch are in requirements.txt"
            ) from e

        self.model_name = model_name
        self.device = device

        # Set cache directory
        if cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = cache_dir

        logger.info(f"Loading local LLM: {model_name}")
        logger.info(f"Quantization: {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'none'}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit and not load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
                bnb_4bit_use_double_quant=True if load_in_4bit else None,
                bnb_4bit_quant_type="nf4" if load_in_4bit else None
            )

        # Load model with quantization
        load_kwargs = {
            "device_map": device,
            "trust_remote_code": True,
            "torch_dtype": torch.float16
        }

        if max_memory:
            load_kwargs["max_memory"] = max_memory

        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        self.model.eval()  # Frozen model (Ghost mode)
        logger.info(f"Model loaded successfully on {device}")

        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")

    def extract(self, prompt: str, **kwargs) -> str:
        """Extract using local model (privacy-preserving).

        Args:
            prompt: Extraction prompt
            **kwargs: Generation parameters
                - max_new_tokens: Maximum tokens to generate (default: 512)
                - temperature: Sampling temperature (default: 0.1 for extraction)
                - top_p: Nucleus sampling (default: 0.95)
                - do_sample: Whether to sample (default: True if temp > 0)

        Returns:
            Generated text (extraction result)
        """
        import torch

        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.1)  # Low temp for extraction
        top_p = kwargs.get("top_p", 0.95)
        do_sample = kwargs.get("do_sample", temperature > 0)

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # Context window
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate (inference only, no gradient)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1  # Reduce repetition
            )

        # Decode only new tokens (not the prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return response

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using local model.

        Alias for extract() with potentially different defaults.
        """
        return self.extract(prompt, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata.

        Returns:
            Dict with model name, device, memory usage, etc.
        """
        import torch

        info = {
            "model_name": self.model_name,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "frozen": not any(p.requires_grad for p in self.model.parameters())
        }

        if torch.cuda.is_available():
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3

        return info


def create_local_llm(
    model: LocalLLMBackend = LocalLLMBackend.LLAMA_3_2_3B,
    **kwargs
) -> LLMClient:
    """Factory function to create local LLM client.

    Args:
        model: Which local model to use
        **kwargs: Additional configuration (device, quantization, etc.)

    Returns:
        LLMClient instance (quantized local model)

    Examples:
        >>> # Default: Llama 3.2 3B (lightweight for testing)
        >>> client = create_local_llm()

        >>> # Production: Llama 3.1 8B (8-bit quantized)
        >>> client = create_local_llm(
        ...     LocalLLMBackend.LLAMA_3_1_8B,
        ...     load_in_8bit=True
        ... )

        >>> # Maximum efficiency: Llama 3.1 8B (4-bit quantized)
        >>> client = create_local_llm(
        ...     LocalLLMBackend.LLAMA_3_1_8B,
        ...     load_in_4bit=True
        ... )
    """
    return QuantizedLocalLLM(model_name=model.value, **kwargs)


def get_test_llm_client(
    fast: bool = True,
    production_model: str = "auto",
    **kwargs
) -> LLMClient:
    """Get local LLM client for testing.

    Args:
        fast: If True, use lightweight model (Phi-3 Mini, open access)
              If False, use production model (default: auto-select based on VRAM)
        production_model: Which production model to use when fast=False:
            - "auto": Auto-select based on available VRAM
            - "qwen": Qwen 2.5 32B Coder (16GB VRAM, optimized extraction)
            - "llama": Llama 3.3 70B (24GB VRAM, max reasoning)
            - Can also use env var: FUTURNAL_PRODUCTION_LLM=qwen|llama
        **kwargs: Additional configuration

    Returns:
        LLMClient configured for testing

    Note:
        This function ALWAYS returns a LOCAL model client.
        No cloud APIs are used (privacy-first principle).

        Fast mode uses Phi-3 Mini which is publicly accessible without
        HuggingFace authentication (good for CI/CD).

        Production models (Llama, Qwen) may require HuggingFace authentication.
        
```
        November 2025 Rankings:
        - Qwen 2.5 32B Coder: Best for 16GB VRAM, excellent extraction
        - Llama 3.3 70B: Best for 24GB+ VRAM, superior reasoning
    """
    import torch
    import os
    
    # Check backend preference
    backend = os.getenv("FUTURNAL_LLM_BACKEND", "auto").lower()
    
    # Auto-detect Ollama if available
    if backend == "auto":
        try:
            from futurnal.extraction.ollama_client import ollama_available
            if ollama_available():
                backend = "ollama"
                logger.info("Auto-detected Ollama backend (10-100x faster!)")
            else:
                backend = "hf"
                logger.info("Using HuggingFace backend (Ollama not running)")
        except ImportError:
            backend = "hf"

    if fast:
        # Use Phi-3 Mini: smaller, fast, and doesn't require auth
        model = LocalLLMBackend.PHI_3_MINI
        logger.info("Using lightweight model for testing (Phi-3 Mini, 3.8B)")
    else:
        # Use production model - check env var first, then parameter
        production_choice = os.getenv("FUTURNAL_PRODUCTION_LLM", production_model).lower()
        
        if production_choice == "qwen":
            model = LocalLLMBackend.QWEN_2_5_32B_CODER
            logger.info("Using production model: Qwen 2.5 32B Coder (Nov 2025)")
            logger.info("Optimized for entity extraction | Requires 16GB VRAM (4-bit)")
        elif production_choice == "llama":
            model = LocalLLMBackend.LLAMA_3_3_70B
            logger.info("Using production model: Llama 3.3 70B (Nov 2025)")
            logger.info("Superior reasoning & structured output | Requires 24GB VRAM (8-bit)")
        elif production_choice == "llama3.1" or production_choice == "llama31":
            model = LocalLLMBackend.LLAMA_3_1_8B
            logger.info("Using production model: Llama 3.1 8B")
            logger.info("Balanced performance | Requires ~8GB VRAM (4-bit)")
        elif production_choice == "bielik":
            model = LocalLLMBackend.BIELIK_4_5B
            logger.info("Using Polish language model: Bielik 4.5B")
            logger.info("Optimized for Polish text | Requires ~5GB VRAM (4-bit)")
        elif production_choice == "kimi" or production_choice == "kimi-k2":
            model = LocalLLMBackend.KIMI_K2_THINKING
            logger.info("Using advanced reasoning model: Kimi-K2-Thinking")
            logger.info("Complex reasoning tasks | Alternative to Llama 3.3 70B")
        elif production_choice in ("gpt-oss", "gpt_oss", "gptoss", "gpt-oss-20b"):
            model = LocalLLMBackend.GPT_OSS_20B
            logger.info("Using unrestricted model: GPT-OSS-20B-Derestricted")
            logger.info("Unrestricted content processing | Requires ~12GB VRAM (4-bit)")
            logger.warning("CAUTION: No content moderation guardrails - use responsibly")
        elif production_choice == "auto":
            # Auto-select based on available VRAM
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 24:
                    model = LocalLLMBackend.LLAMA_3_3_70B
                    logger.info(f"Auto-selected Llama 3.3 70B (detected {vram_gb:.1f}GB VRAM)")
                elif vram_gb >= 16:
                    model = LocalLLMBackend.QWEN_2_5_32B_CODER
                    logger.info(f"Auto-selected Qwen 2.5 32B (detected {vram_gb:.1f}GB VRAM)")
                else:
                    model = LocalLLMBackend.PHI_3_MINI
                    logger.info(f"Auto-selected Phi-3 Mini (detected {vram_gb:.1f}GB VRAM)")
            else:
                # No CUDA, default to smallest model
                model = LocalLLMBackend.PHI_3_MINI
                logger.info("Auto-selected Phi-3 Mini (no CUDA detected, CPU mode)")
        else:
            # Default fallback
            model = LocalLLMBackend.LLAMA_3_1_8B
            logger.warning(f"Unknown production_model '{production_choice}', using Llama 3.1 8B")
        
        logger.info("NOTE: Production models may require HuggingFace authentication")
        logger.info("      Run: huggingface-cli login")

    # Default to 4-bit quantization for testing efficiency ONLY if CUDA is available
    # On non-CUDA systems (macOS, CPU-only), quantization is not supported
    if "load_in_4bit" not in kwargs and "load_in_8bit" not in kwargs:
        if torch.cuda.is_available():
            kwargs["load_in_4bit"] = True
            logger.info("CUDA available: using 4-bit quantization")
        else:
            kwargs["load_in_4bit"] = False
            kwargs["load_in_8bit"] = False
            logger.info("CUDA not available: disabling quantization (CPU/MPS mode)")

    # Return appropriate backend client
    if backend == "ollama":
        try:
            from futurnal.extraction.ollama_client import OllamaLLMClient
            logger.info(f"Creating Ollama client for {model.value}")
            return OllamaLLMClient(model_name=model.value)
        except Exception as e:
            logger.warning(f"Ollama client failed: {e}, falling back to HuggingFace")
            backend = "hf"
    
    # Fallback to HuggingFace
    if backend == "hf":
        logger.info(f"Creating HuggingFace client for {model.value}")
        return create_local_llm(model, **kwargs)
    
    raise ValueError(f"Unknown backend: {backend}")


class ExperientialLLMWrapper:
    """Wrapper that combines Ghost LLM with experiential knowledge (token priors).

    This implements the Ghost→Animal transformation:
    - Ghost: Frozen base model (QuantizedLocalLLM)
    - Experiential knowledge: Natural language patterns prepended to prompts
    - Animal: Ghost + Experience = Improved behavior without parameter updates

    This is the core of Training-Free GRPO approach.
    """

    def __init__(
        self,
        ghost_llm: LLMClient,
        experiential_knowledge: Optional[List[str]] = None
    ):
        """Initialize experiential wrapper.

        Args:
            ghost_llm: Base frozen model (Ghost)
            experiential_knowledge: List of learned patterns as natural language
        """
        self.ghost_llm = ghost_llm
        self.experiential_knowledge = experiential_knowledge or []

    def add_experience(self, pattern: str) -> None:
        """Add experiential knowledge pattern.

        Args:
            pattern: Natural language pattern learned from experience
        """
        self.experiential_knowledge.append(pattern)

    def extract(self, prompt: str, **kwargs) -> Any:
        """Extract with experiential knowledge (Animal behavior).

        Prepends experiential knowledge to prompt as token priors.
        """
        if self.experiential_knowledge:
            # Prepend experience as context (token priors)
            experience_context = "\n\n".join([
                "# Learned Patterns (apply these insights):",
                *[f"- {pattern}" for pattern in self.experiential_knowledge],
                "",
                "# Task:"
            ])
            enhanced_prompt = f"{experience_context}\n{prompt}"
        else:
            enhanced_prompt = prompt

        return self.ghost_llm.extract(enhanced_prompt, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with experiential knowledge."""
        return self.extract(prompt, **kwargs)
