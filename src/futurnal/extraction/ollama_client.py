"""Ollama-based local LLM client for fast inference.

This module provides a fast alternative to HuggingFace transformers
using Ollama's optimized C++ backend. Expected 10-100x speedup.
"""

from __future__ import annotations

import logging
import requests
from typing import Optional, Dict, Any

from futurnal.extraction.local_llm_client import LLMClient, LocalLLMBackend

logger = logging.getLogger(__name__)


# Mapping from HuggingFace model names to Ollama model names
OLLAMA_MODEL_MAP = {
    # Production Models
    "microsoft/Phi-3-mini-4k-instruct": "phi3:mini",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1:8b-instruct-q4_0",
    "meta-llama/Llama-3.3-70B-Instruct": "llama3.3:70b-instruct-q4_0",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "qwen2.5-coder:32b-instruct-q4_0",
    "speakleash/Bielik-4.5B-v3.0-Instruct": "bielik:4.5b-instruct-q4_0",
    # Advanced Reasoning Models (Kimi K2 - 1T MoE, 32B active)
    "moonshotai/Kimi-K2-Thinking": "kimi-k2-thinking",  # Deep reasoning, 200-300 tool calls
    "moonshotai/Kimi-K2-Instruct": "kimi-k2",  # Fast instruct version
    # Local 20B Models
    "ArliAI/gpt-oss-20b": "gpt-oss:20b",  # Local 20B model
}


class OllamaLLMClient(LLMClient):
    """Fast local LLM client using Ollama backend.
    
    Provides 10-100x faster inference compared to HuggingFace transformers
    by using Ollama's C++ optimized backend with llama.cpp.
    
    Benefits:
    - Instant model loading (vs 6+ minutes with HF)
    - Fast inference (C++ vs Python)
    - Automatic quantization handling
    - Simple model management
    - Optimized for Apple Silicon
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 300
    ):
        """Initialize Ollama LLM client.
        
        Args:
            model_name: HuggingFace model name (will be mapped to Ollama)
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        
        # Map to Ollama model name
        self.ollama_model = OLLAMA_MODEL_MAP.get(model_name, model_name)
        
        logger.info(f"Initialized Ollama client: {model_name} -> {self.ollama_model}")
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Ollama server is running")
        except Exception as e:
            logger.warning(f"Ollama server not reachable: {e}")
            logger.warning("Start with: ollama serve")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional Ollama parameters
            
        Returns:
            Generated text
        """
        # Build request
        request_data = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens
        
        # Merge additional kwargs
        if kwargs:
            request_data["options"].update(kwargs)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["response"]
            
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Callable interface for compatibility."""
        return self.generate(prompt, **kwargs)


def ollama_available() -> bool:
    """Check if Ollama is installed and running.
    
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def get_ollama_models() -> list[str]:
    """Get list of available Ollama models.
    
    Returns:
        List of model names installed in Ollama
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    except Exception as e:
        logger.warning(f"Failed to get Ollama models: {e}")
        return []
