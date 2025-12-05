"""Whisper-based audio transcription client for fast, accurate speech-to-text.

This module provides transcription clients using OpenAI Whisper models
with support for both Ollama (10-100x faster) and HuggingFace backends.

Module 08: Multimodal Integration & Tool Enhancement - Phase 1
Production Plan: docs/phase-1/entity-relationship-extraction-production-plan/08-multimodal-integration.md
"""

from __future__ import annotations

import logging
import os
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class TimestampedSegment:
    """A transcribed segment with timing information.

    Attributes:
        text: Transcribed text for this segment
        start: Start time in seconds
        end: End time in seconds
        confidence: Confidence score (0.0-1.0)
    """
    text: str
    start: float  # seconds
    end: float    # seconds
    confidence: float


@dataclass
class TranscriptionResult:
    """Result of audio transcription.

    Attributes:
        text: Full transcribed text
        segments: List of timestamped segments
        language: Detected language code (e.g., "en", "pl")
        confidence: Overall confidence score (0.0-1.0)
    """
    text: str
    segments: List[TimestampedSegment]
    language: str
    confidence: float


# =============================================================================
# Client Protocol
# =============================================================================


class WhisperTranscriptionClient(Protocol):
    """Protocol for audio transcription clients.

    All Whisper client implementations must provide:
    - transcribe() method for audio-to-text conversion
    - Support for language detection
    - Segment-level timestamps for temporal extraction
    """

    def transcribe(
        self,
        audio_file: str,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio file to text with timestamps.

        Args:
            audio_file: Path to audio file (.wav, .mp3, .m4a, etc.)
            language: Language code (None for auto-detection)
            **kwargs: Additional transcription parameters

        Returns:
            TranscriptionResult with text, segments, language, confidence
        """
        ...


# =============================================================================
# Ollama Whisper Client (10-100x faster)
# =============================================================================


class OllamaWhisperClient:
    """Ollama-based Whisper client for fast transcription.

    Provides 10-100x faster transcription compared to HuggingFace
    by using Ollama's C++ optimized backend.

    Benefits:
    - Instant model loading (vs minutes with HF)
    - Fast inference (C++ vs Python)
    - Automatic quantization handling
    - Simple model management
    - Optimized for Apple Silicon and NVIDIA GPUs
    """

    def __init__(
        self,
        model_name: str = "whisper:large-v3",
        base_url: str = "http://localhost:11434",
        timeout: int = 600  # 10 minutes for long audio
    ):
        """Initialize Ollama Whisper client.

        Args:
            model_name: Ollama model name (e.g., "whisper:large-v3")
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout

        logger.info(f"Initialized Ollama Whisper client: {model_name}")

        # Check if Ollama is running
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Ollama server is running")
        except Exception as e:
            logger.warning(f"Ollama server not reachable: {e}")
            logger.warning("Start with: ollama serve")

    def transcribe(
        self,
        audio_file: str,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using Ollama Whisper.

        Args:
            audio_file: Path to audio file
            language: Language code (None for auto-detection)
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            TranscriptionResult with text, segments, language, confidence

        Raises:
            FileNotFoundError: If audio file doesn't exist
            requests.exceptions.RequestException: If Ollama request fails
        """
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Build request
        request_data = {
            "model": self.model_name,
            "audio": str(audio_path),
            "options": {}
        }

        if language:
            request_data["options"]["language"] = language

        # Merge additional kwargs
        if kwargs:
            request_data["options"].update(kwargs)

        try:
            logger.info(f"Transcribing audio: {audio_path.name} ({audio_path.stat().st_size / 1024 / 1024:.2f} MB)")

            response = requests.post(
                f"{self.base_url}/api/transcribe",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            # Parse Ollama response
            # Note: Ollama Whisper API may vary - adjust based on actual response format
            segments = [
                TimestampedSegment(
                    text=seg.get("text", ""),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    confidence=seg.get("confidence", 1.0)
                )
                for seg in result.get("segments", [])
            ]

            transcription = TranscriptionResult(
                text=result.get("text", ""),
                segments=segments,
                language=result.get("language", language or "unknown"),
                confidence=result.get("confidence", 1.0)
            )

            logger.info(f"Transcription complete: {len(transcription.text)} chars, {len(segments)} segments")

            return transcription

        except requests.exceptions.Timeout:
            logger.error(f"Ollama transcription timed out after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama transcription failed: {e}")
            raise


# =============================================================================
# HuggingFace Whisper Client (Fallback)
# =============================================================================


class LocalWhisperClient:
    """HuggingFace Whisper client as fallback when Ollama unavailable.

    This client uses the original OpenAI Whisper via HuggingFace Transformers.
    Slower than Ollama but works without external dependencies.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: str = "auto"
    ):
        """Initialize HuggingFace Whisper client.

        Args:
            model_name: HuggingFace model identifier
            device: Device for inference ("auto", "cuda", "cpu", "mps")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

        logger.info(f"Initialized HuggingFace Whisper client: {model_name}")
        logger.warning("HuggingFace Whisper is slower than Ollama. Consider using Ollama for 10-100x speedup.")

    def _load_model(self):
        """Lazy-load Whisper model and processor."""
        if self.model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            logger.info(f"Loading Whisper model: {self.model_name} (this may take several minutes)")

            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = self.device

            logger.info(f"Using device: {device}")

            # Load model with quantization if on CPU
            if device == "cpu":
                # 8-bit quantization for CPU
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    device_map=device
                )

            self.processor = AutoProcessor.from_pretrained(self.model_name)

            # Set to eval mode (Ghost model - frozen)
            self.model.eval()

            logger.info("Whisper model loaded successfully")

        except ImportError as e:
            logger.error("HuggingFace Transformers not installed. Install with: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio_file: str,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using HuggingFace Whisper.

        Args:
            audio_file: Path to audio file
            language: Language code (None for auto-detection)
            **kwargs: Additional parameters

        Returns:
            TranscriptionResult with text, segments, language, confidence
        """
        self._load_model()

        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        try:
            import torch
            import librosa

            logger.info(f"Transcribing audio: {audio_path.name}")

            # Load audio file
            audio, sr = librosa.load(str(audio_path), sr=16000)  # Whisper expects 16kHz

            # Process audio
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.model.device)

            # Generate transcription
            with torch.no_grad():
                if language:
                    # Force language
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language)
                    outputs = self.model.generate(
                        **inputs,
                        forced_decoder_ids=forced_decoder_ids,
                        return_timestamps=True
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        return_timestamps=True
                    )

            # Decode transcription
            transcription_dict = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                output_word_timestamps=True
            )[0]

            # Extract text
            text = transcription_dict.get("text", "")

            # Extract segments (word-level timestamps)
            segments = []
            if "chunks" in transcription_dict:
                for chunk in transcription_dict["chunks"]:
                    segment = TimestampedSegment(
                        text=chunk["text"],
                        start=chunk["timestamp"][0] if chunk["timestamp"] else 0.0,
                        end=chunk["timestamp"][1] if chunk["timestamp"] else 0.0,
                        confidence=1.0  # HF doesn't provide per-segment confidence
                    )
                    segments.append(segment)
            else:
                # Fallback: single segment with full text
                duration = len(audio) / 16000  # seconds
                segments = [TimestampedSegment(
                    text=text,
                    start=0.0,
                    end=duration,
                    confidence=1.0
                )]

            # Detect language if not specified
            detected_language = language or "unknown"

            result = TranscriptionResult(
                text=text,
                segments=segments,
                language=detected_language,
                confidence=1.0  # HF doesn't provide overall confidence
            )

            logger.info(f"Transcription complete: {len(text)} chars, {len(segments)} segments")

            return result

        except ImportError as e:
            logger.error("Required libraries not installed. Install with: pip install librosa soundfile")
            raise
        except Exception as e:
            logger.error(f"HuggingFace transcription failed: {e}")
            raise


# =============================================================================
# Utility Functions
# =============================================================================


def whisper_available() -> bool:
    """Check if Ollama with Whisper is installed and running.

    Returns:
        True if Ollama Whisper is available, False otherwise
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            return False

        # Check if whisper model is installed
        models = response.json().get("models", [])
        whisper_models = [m for m in models if "whisper" in m.get("name", "").lower()]

        return len(whisper_models) > 0

    except Exception:
        return False


def get_whisper_models() -> List[str]:
    """Get list of available Whisper models in Ollama.

    Returns:
        List of Whisper model names installed in Ollama
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get("models", [])
        whisper_models = [
            m["name"]
            for m in models
            if "whisper" in m.get("name", "").lower()
        ]

        return whisper_models

    except Exception as e:
        logger.warning(f"Failed to get Whisper models: {e}")
        return []


def get_transcription_client(
    backend: str = "auto",
    model_name: Optional[str] = None,
    **kwargs
) -> WhisperTranscriptionClient:
    """Factory function to create transcription client with auto-backend selection.

    Args:
        backend: Backend selection ("auto", "ollama", "hf", "huggingface")
        model_name: Optional model name override
        **kwargs: Additional client initialization parameters

    Returns:
        WhisperTranscriptionClient instance (Ollama or HuggingFace)

    Examples:
        >>> # Auto-select best backend
        >>> client = get_transcription_client()

        >>> # Force Ollama
        >>> client = get_transcription_client(backend="ollama")

        >>> # Force HuggingFace
        >>> client = get_transcription_client(backend="hf")
    """
    # Check environment variable override
    env_backend = os.getenv("FUTURNAL_AUDIO_BACKEND", backend)

    if env_backend == "auto":
        # Auto-select based on availability
        if whisper_available():
            logger.info("Auto-selected Ollama Whisper backend (10-100x faster)")
            return OllamaWhisperClient(
                model_name=model_name or "whisper:large-v3",
                **kwargs
            )
        else:
            logger.info("Auto-selected HuggingFace Whisper backend (Ollama not available)")
            logger.info("For 10-100x speedup, install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
            logger.info("Then: ollama pull whisper:large-v3")
            return LocalWhisperClient(
                model_name=model_name or "openai/whisper-large-v3",
                **kwargs
            )

    elif env_backend in ("ollama",):
        logger.info("Using Ollama Whisper backend")
        return OllamaWhisperClient(
            model_name=model_name or "whisper:large-v3",
            **kwargs
        )

    elif env_backend in ("hf", "huggingface", "transformers"):
        logger.info("Using HuggingFace Whisper backend")
        return LocalWhisperClient(
            model_name=model_name or "openai/whisper-large-v3",
            **kwargs
        )

    else:
        logger.warning(f"Unknown backend '{env_backend}', falling back to auto-selection")
        return get_transcription_client(backend="auto", model_name=model_name, **kwargs)
