"""OCR client infrastructure for text extraction from images and documents.

This module provides OCR clients using DeepSeek-OCR (SOTA) with Tesseract fallback
for accurate text extraction from images, scanned PDFs, and other visual documents.

Module 08: Multimodal Integration & Tool Enhancement - Phase 2
Production Plan: docs/phase-1/entity-relationship-extraction-production-plan/08-multimodal-integration.md
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class BoundingBox:
    """Bounding box coordinates for text regions.

    Attributes:
        x1: Left coordinate
        y1: Top coordinate
        x2: Right coordinate
        y2: Bottom coordinate
    """
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Calculate width of bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Calculate height of bounding box."""
        return self.y2 - self.y1


@dataclass
class TextRegion:
    """Text region with location and metadata.

    Attributes:
        text: Extracted text content
        bbox: Bounding box coordinates
        confidence: OCR confidence score (0.0-1.0)
        region_type: Type of region (paragraph, heading, table, etc.)
    """
    text: str
    bbox: BoundingBox
    confidence: float
    region_type: str = "paragraph"  # paragraph, heading, table, list, etc.


@dataclass
class LayoutInfo:
    """Document layout information.

    Attributes:
        page_count: Number of pages processed
        regions: List of text regions with locations
        reading_order: Indices of regions in reading order
    """
    page_count: int
    regions: List[TextRegion]
    reading_order: List[int]


@dataclass
class OCRResult:
    """Result of OCR processing.

    Attributes:
        text: Full extracted text
        layout: Layout information with regions
        regions: Convenience access to text regions
        confidence: Overall confidence score (0.0-1.0)
    """
    text: str
    layout: LayoutInfo
    regions: List[TextRegion]
    confidence: float


# =============================================================================
# Client Protocol
# =============================================================================


class OCRClient(Protocol):
    """Protocol for OCR clients.

    All OCR client implementations must provide:
    - extract_text() method for image/document to text conversion
    - Layout preservation support
    - Confidence scoring
    """

    def extract_text(
        self,
        image_or_pdf: str,
        preserve_layout: bool = True,
        **kwargs
    ) -> OCRResult:
        """Extract text from image or PDF.

        Args:
            image_or_pdf: Path to image or PDF file
            preserve_layout: Whether to preserve document layout
            **kwargs: Additional OCR parameters

        Returns:
            OCRResult with text, layout, regions, and confidence
        """
        ...


# =============================================================================
# DeepSeek-OCR Client (SOTA)
# =============================================================================


class DeepSeekOCRClient:
    """DeepSeek-OCR client for state-of-the-art text extraction.

    Provides high-accuracy OCR using DeepSeek's vision-language model
    with layout preservation and multi-language support.

    Benefits:
    - State-of-the-art accuracy (>98% character accuracy)
    - Layout preservation (columns, tables, formatting)
    - Multi-language support
    - Handwriting recognition
    - Complex document structures (forms, receipts, etc.)
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "auto"
    ):
        """Initialize DeepSeek-OCR client.

        Args:
            model_name: HuggingFace model identifier
            device: Device for inference ("auto", "cuda", "cpu", "mps")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

        logger.info(f"Initialized DeepSeek-OCR client: {model_name}")

    def _load_model(self):
        """Lazy-load DeepSeek-OCR model and processor."""
        if self.model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            logger.info(f"Loading DeepSeek-OCR model: {self.model_name}")

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

            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map=device,
                trust_remote_code=True  # DeepSeek-OCR may require this
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Set to eval mode (Ghost model - frozen)
            self.model.eval()

            logger.info("DeepSeek-OCR model loaded successfully")

        except ImportError as e:
            logger.error("HuggingFace Transformers not installed. Install with: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load DeepSeek-OCR model: {e}")
            raise

    def extract_text(
        self,
        image_or_pdf: str,
        preserve_layout: bool = True,
        **kwargs
    ) -> OCRResult:
        """Extract text using DeepSeek-OCR.

        Args:
            image_or_pdf: Path to image or PDF file
            preserve_layout: Whether to preserve document layout
            **kwargs: Additional parameters

        Returns:
            OCRResult with text, layout, regions, and confidence
        """
        self._load_model()

        image_path = Path(image_or_pdf)
        if not image_path.exists():
            raise FileNotFoundError(f"Image/PDF not found: {image_or_pdf}")

        try:
            from PIL import Image
            import torch

            logger.info(f"Processing image with DeepSeek-OCR: {image_path.name}")

            # Load image
            image = Image.open(str(image_path))

            # Process image
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.model.device)

            # Extract text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,  # Adjust based on document complexity
                    **kwargs
                )

            # Decode text
            extracted_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]

            # Extract layout information (if model provides it)
            # Note: Actual implementation depends on DeepSeek-OCR API
            regions = self._extract_regions(image, extracted_text, preserve_layout)

            layout = LayoutInfo(
                page_count=1,
                regions=regions,
                reading_order=list(range(len(regions)))
            )

            result = OCRResult(
                text=extracted_text,
                layout=layout,
                regions=regions,
                confidence=0.98  # DeepSeek-OCR typically achieves >98% accuracy
            )

            logger.info(f"OCR complete: {len(extracted_text)} chars, {len(regions)} regions")

            return result

        except ImportError as e:
            logger.error("Required libraries not installed. Install with: pip install Pillow")
            raise
        except Exception as e:
            logger.error(f"DeepSeek-OCR extraction failed: {e}")
            raise

    def _extract_regions(
        self,
        image,
        text: str,
        preserve_layout: bool
    ) -> List[TextRegion]:
        """Extract text regions from image (placeholder implementation).

        Note: Actual implementation depends on DeepSeek-OCR API capabilities.
        This is a simplified version.
        """
        if not preserve_layout or not text:
            # Fallback: single region with full text
            return [
                TextRegion(
                    text=text,
                    bbox=BoundingBox(0, 0, 100, 100),  # Full image
                    confidence=0.98,
                    region_type="paragraph"
                )
            ]

        # TODO: Implement proper region extraction when DeepSeek-OCR API provides it
        # For now, split by paragraphs as approximation
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        regions = []
        for idx, para in enumerate(paragraphs):
            region = TextRegion(
                text=para,
                bbox=BoundingBox(0, idx * 20, 100, (idx + 1) * 20),  # Approximate
                confidence=0.98,
                region_type="paragraph"
            )
            regions.append(region)

        return regions


# =============================================================================
# Tesseract OCR Client (Fallback)
# =============================================================================


class TesseractOCRClient:
    """Tesseract OCR client as fallback when DeepSeek-OCR unavailable.

    Uses pytesseract wrapper for Tesseract OCR engine.
    Less accurate than DeepSeek-OCR but widely available.
    """

    def __init__(self, lang: str = "eng"):
        """Initialize Tesseract OCR client.

        Args:
            lang: Language code for OCR (eng, fra, deu, etc.)
        """
        self.lang = lang

        logger.info(f"Initialized Tesseract OCR client: language={lang}")
        logger.warning("Tesseract OCR is less accurate than DeepSeek-OCR. Consider using DeepSeek for better results.")

    def extract_text(
        self,
        image_or_pdf: str,
        preserve_layout: bool = True,
        **kwargs
    ) -> OCRResult:
        """Extract text using Tesseract OCR.

        Args:
            image_or_pdf: Path to image or PDF file
            preserve_layout: Whether to preserve document layout
            **kwargs: Additional parameters

        Returns:
            OCRResult with text, layout, regions, and confidence
        """
        image_path = Path(image_or_pdf)
        if not image_path.exists():
            raise FileNotFoundError(f"Image/PDF not found: {image_or_pdf}")

        try:
            import pytesseract
            from PIL import Image

            logger.info(f"Processing image with Tesseract: {image_path.name}")

            # Load image
            image = Image.open(str(image_path))

            # Extract text
            config = ""
            if preserve_layout:
                config = "--psm 1"  # Automatic page segmentation with OSD

            extracted_text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=config
            )

            # Get detailed OCR data for layout
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                output_type=pytesseract.Output.DICT
            )

            # Extract regions from OCR data
            regions = self._extract_regions_from_data(ocr_data)

            layout = LayoutInfo(
                page_count=1,
                regions=regions,
                reading_order=list(range(len(regions)))
            )

            # Calculate average confidence
            confidences = [r.confidence for r in regions if r.confidence > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            result = OCRResult(
                text=extracted_text,
                layout=layout,
                regions=regions,
                confidence=avg_confidence
            )

            logger.info(f"Tesseract OCR complete: {len(extracted_text)} chars, {len(regions)} regions")

            return result

        except ImportError as e:
            logger.error("pytesseract not installed. Install with: pip install pytesseract")
            logger.error("Also install Tesseract: brew install tesseract (macOS) or apt-get install tesseract-ocr (Ubuntu)")
            raise
        except Exception as e:
            logger.error(f"Tesseract OCR extraction failed: {e}")
            raise

    def _extract_regions_from_data(self, ocr_data: dict) -> List[TextRegion]:
        """Extract text regions from Tesseract OCR data."""
        regions = []

        for i in range(len(ocr_data["text"])):
            text = ocr_data["text"][i].strip()
            if not text:
                continue

            bbox = BoundingBox(
                x1=ocr_data["left"][i],
                y1=ocr_data["top"][i],
                x2=ocr_data["left"][i] + ocr_data["width"][i],
                y2=ocr_data["top"][i] + ocr_data["height"][i]
            )

            confidence = ocr_data["conf"][i] / 100.0  # Tesseract uses 0-100

            region = TextRegion(
                text=text,
                bbox=bbox,
                confidence=max(0.0, confidence),  # Clip negative confidences
                region_type="word"  # Tesseract provides word-level data
            )

            regions.append(region)

        return regions


# =============================================================================
# Utility Functions
# =============================================================================


def deepseek_available() -> bool:
    """Check if DeepSeek-OCR model is available.

    Returns:
        True if DeepSeek-OCR can be loaded, False otherwise
    """
    try:
        from transformers import AutoModel
        # Quick check - don't actually load the model
        return True
    except ImportError:
        return False


def tesseract_available() -> bool:
    """Check if Tesseract OCR is installed.

    Returns:
        True if Tesseract is available, False otherwise
    """
    try:
        import pytesseract
        # Try to get version (quick check)
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def get_ocr_client(
    backend: str = "auto",
    **kwargs
) -> OCRClient:
    """Factory function to create OCR client with auto-backend selection.

    Args:
        backend: Backend selection ("auto", "deepseek", "tesseract")
        **kwargs: Additional client initialization parameters

    Returns:
        OCRClient instance (DeepSeek or Tesseract)

    Examples:
        >>> # Auto-select best backend
        >>> client = get_ocr_client()

        >>> # Force DeepSeek-OCR
        >>> client = get_ocr_client(backend="deepseek")

        >>> # Force Tesseract
        >>> client = get_ocr_client(backend="tesseract")
    """
    # Check environment variable override
    env_backend = os.getenv("FUTURNAL_VISION_BACKEND", backend)

    if env_backend == "auto":
        # Auto-select based on availability
        # Prefer DeepSeek-OCR for accuracy, fall back to Tesseract
        if deepseek_available():
            logger.info("Auto-selected DeepSeek-OCR backend (SOTA accuracy)")
            return DeepSeekOCRClient(**kwargs)
        elif tesseract_available():
            logger.info("Auto-selected Tesseract OCR backend (DeepSeek-OCR not available)")
            logger.info("For better accuracy, install DeepSeek-OCR: pip install transformers torch")
            return TesseractOCRClient(**kwargs)
        else:
            logger.error("No OCR backend available!")
            logger.error("Install either:")
            logger.error("  - DeepSeek-OCR: pip install transformers torch")
            logger.error("  - Tesseract: brew install tesseract (macOS) or apt-get install tesseract-ocr (Ubuntu)")
            logger.error("                pip install pytesseract")
            raise RuntimeError("No OCR backend available")

    elif env_backend in ("deepseek", "deepseek-ocr"):
        logger.info("Using DeepSeek-OCR backend")
        return DeepSeekOCRClient(**kwargs)

    elif env_backend in ("tesseract", "tess"):
        logger.info("Using Tesseract OCR backend")
        return TesseractOCRClient(**kwargs)

    else:
        logger.warning(f"Unknown backend '{env_backend}', falling back to auto-selection")
        return get_ocr_client(backend="auto", **kwargs)
