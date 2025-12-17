"""Multimodal CLI commands for audio/image/document processing.

Step 08: Frontend Intelligence Integration - Phase 1

Bridges frontend multimodal input to backend processing services:
- Voice transcription via Whisper V3 (Ollama 10-100x faster, or HuggingFace)
- Image OCR via DeepSeek-OCR (>98% accuracy) with Tesseract fallback
- Document processing via existing normalization pipeline

Research Foundation:
- MM-HELIX: Multi-modal hybrid intelligent system
- Youtu-GraphRAG: Multi-modal information integration
"""

import json
import logging
import tempfile
import base64
from pathlib import Path
from typing import Optional

from typer import Typer, Option, Argument

logger = logging.getLogger(__name__)

multimodal_app = Typer(help="Multimodal processing commands (voice, images, documents)")


@multimodal_app.command("transcribe")
def transcribe_audio(
    audio_file: str = Argument(..., help="Path to audio file or '-' for base64 stdin"),
    language: Optional[str] = Option(None, "--language", "-l", help="Language code (auto-detect if not specified)"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Transcribe audio to text using Whisper V3.

    Uses Ollama Whisper (10-100x faster) with HuggingFace fallback.

    Examples:
        futurnal multimodal transcribe recording.wav
        futurnal multimodal transcribe recording.wav --language en --json
        echo "<base64>" | futurnal multimodal transcribe - --json
    """
    import sys
    from ..extraction.whisper_client import get_transcription_client

    try:
        # Handle base64 input from stdin
        if audio_file == "-":
            b64_data = sys.stdin.read().strip()
            audio_bytes = base64.b64decode(b64_data)

            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                audio_path = f.name
        else:
            audio_path = audio_file

        # Get transcription client (auto-selects Ollama or HuggingFace)
        client = get_transcription_client(backend="auto")

        # Transcribe
        result = client.transcribe(audio_path, language=language)

        if output_json:
            output = {
                "success": True,
                "text": result.text,
                "language": result.language,
                "confidence": result.confidence,
                "segments": [
                    {
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end,
                        "confidence": seg.confidence,
                    }
                    for seg in result.segments
                ],
            }
            print(json.dumps(output))
        else:
            print(f"Transcription ({result.language}, confidence: {result.confidence:.2f}):")
            print(result.text)

        # Cleanup temp file
        if audio_file == "-":
            Path(audio_path).unlink(missing_ok=True)

    except FileNotFoundError as e:
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        logger.exception("Transcription failed")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)


@multimodal_app.command("ocr")
def extract_image_text(
    image_file: str = Argument(..., help="Path to image file or '-' for base64 stdin"),
    preserve_layout: bool = Option(True, "--preserve-layout/--no-layout", help="Preserve document layout"),
    backend: str = Option("tesseract", "--backend", "-b", help="OCR backend (tesseract, deepseek, auto)"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Extract text from image using OCR.

    Uses Tesseract by default (reliable, no model download required).
    DeepSeek-OCR available for higher accuracy if configured.

    Examples:
        futurnal multimodal ocr screenshot.png
        futurnal multimodal ocr document.jpg --json
        futurnal multimodal ocr document.jpg --backend deepseek --json
    """
    import sys
    from ..extraction.ocr_client import get_ocr_client, tesseract_available

    try:
        # Handle base64 input from stdin
        if image_file == "-":
            b64_data = sys.stdin.read().strip()
            image_bytes = base64.b64decode(b64_data)

            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(image_bytes)
                image_path = f.name
        else:
            image_path = image_file

        # Get OCR client - default to Tesseract for reliability
        # DeepSeek-OCR requires additional model download and dependencies
        actual_backend = backend
        if backend == "auto" and tesseract_available():
            actual_backend = "tesseract"  # Prefer Tesseract for reliability
        client = get_ocr_client(backend=actual_backend)

        # Extract text
        result = client.extract_text(image_path, preserve_layout=preserve_layout)

        if output_json:
            output = {
                "success": True,
                "text": result.text,
                "confidence": result.confidence,
                "layout": {
                    "regions": [
                        {
                            "text": region.text,
                            "bbox": [region.bbox.x1, region.bbox.y1, region.bbox.x2, region.bbox.y2],
                            "confidence": region.confidence,
                            "type": region.region_type,
                        }
                        for region in result.layout.regions
                    ] if result.layout else [],
                },
            }
            print(json.dumps(output))
        else:
            print(f"OCR Result (confidence: {result.confidence:.2f}):")
            print(result.text)

        # Cleanup temp file
        if image_file == "-":
            Path(image_path).unlink(missing_ok=True)

    except FileNotFoundError as e:
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        logger.exception("OCR failed")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)


@multimodal_app.command("process")
def process_document(
    file_path: str = Argument(..., help="Path to document file or '-' for base64 stdin"),
    filename: Optional[str] = Option(None, "--filename", "-f", help="Original filename (for type detection)"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Process a document through the normalization pipeline.

    Handles various document types: PDF, DOCX, TXT, MD, HTML, etc.

    Examples:
        futurnal multimodal process document.pdf
        futurnal multimodal process notes.md --json
        echo "<base64>" | futurnal multimodal process - --filename report.pdf --json
    """
    import sys
    import re

    try:
        # Handle base64 input from stdin
        if file_path == "-":
            if not filename:
                raise ValueError("--filename required when reading from stdin")

            b64_data = sys.stdin.read().strip()
            doc_bytes = base64.b64decode(b64_data)

            # Determine suffix from filename
            suffix = Path(filename).suffix or ".txt"

            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(doc_bytes)
                doc_path = f.name
        else:
            doc_path = file_path
            filename = filename or Path(file_path).name

        # Process document
        path = Path(doc_path)

        # Read content based on type
        if path.suffix.lower() in (".md", ".txt", ".html"):
            content = path.read_text(encoding="utf-8")
            # Simple normalization: clean up whitespace and normalize line endings
            normalized = content.replace('\r\n', '\n').replace('\r', '\n')
            # Remove excessive empty lines
            normalized = re.sub(r'\n{3,}', '\n\n', normalized)
            normalized = normalized.strip()

            result = {
                "text": normalized,
                "summary": normalized[:500] + "..." if len(normalized) > 500 else normalized,
                "type": path.suffix.lower(),
                "wordCount": len(normalized.split()),
            }
        else:
            # For binary documents (PDF, DOCX), use unstructured.io
            text = ""
            element_count = 0

            try:
                # Suppress unstructured.io verbose output to prevent JSON corruption
                import sys
                import io
                import warnings
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        from unstructured.partition.auto import partition
                        elements = partition(filename=str(path))
                        text = "\n\n".join([str(el) for el in elements])
                        element_count = len(elements)
                    finally:
                        sys.stdout = old_stdout
            except ImportError:
                logger.warning("unstructured.io not available, trying fallback methods")
            except Exception as partition_error:
                logger.warning(f"unstructured.io partition failed: {partition_error}")

            # Fallback: try reading as text if partition failed
            if not text:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    text = content.strip()
                except Exception as read_error:
                    logger.warning(f"Text read failed: {read_error}")
                    # For truly binary files, note that we couldn't extract text
                    text = f"[Binary file: {filename} - text extraction not available]"

            result = {
                "text": text,
                "summary": text[:500] + "..." if len(text) > 500 else text,
                "type": path.suffix.lower(),
                "wordCount": len(text.split()) if text else 0,
            }
            if element_count > 0:
                result["elementCount"] = element_count

        if output_json:
            output = {
                "success": True,
                "filename": filename,
                **result,
            }
            print(json.dumps(output))
        else:
            print(f"Document: {filename}")
            print(f"Type: {result['type']}, Words: {result['word_count']}")
            print("-" * 40)
            print(result["summary"])

        # Cleanup temp file
        if file_path == "-":
            Path(doc_path).unlink(missing_ok=True)

    except FileNotFoundError as e:
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        logger.exception("Document processing failed")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)


@multimodal_app.command("describe")
def describe_image(
    image_file: str = Argument(..., help="Path to image file"),
    model: str = Option("llava:7b", "--model", "-m", help="Vision model to use (e.g., llava:7b, llava:13b)"),
    prompt: Optional[str] = Option(None, "--prompt", "-p", help="Custom prompt for description"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Describe an image using a vision-capable LLM.

    Uses Ollama with vision models like LLaVA for understanding images
    that don't contain extractable text.

    Examples:
        futurnal multimodal describe photo.jpg
        futurnal multimodal describe diagram.png --model llava:13b
        futurnal multimodal describe chart.png --prompt "What data does this chart show?" --json
    """
    import subprocess

    try:
        path = Path(image_file)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")

        # Read image and encode as base64
        image_bytes = path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Default prompt for general image description
        description_prompt = prompt or (
            "Describe this image in detail. Include: "
            "1) What type of image it is (photo, diagram, screenshot, chart, etc.) "
            "2) Main subjects or content "
            "3) Any text visible in the image "
            "4) Key details that would help someone understand the image without seeing it"
        )

        # Call Ollama with vision model
        # Ollama vision API expects: ollama run model "prompt" --images file
        # Or via API: POST /api/generate with images array
        try:
            import requests

            # Use Ollama API directly for vision
            api_url = "http://localhost:11434/api/generate"
            payload = {
                "model": model,
                "prompt": description_prompt,
                "images": [image_b64],
                "stream": False,
            }

            response = requests.post(api_url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            description = result.get("response", "").strip()

            if output_json:
                output = {
                    "success": True,
                    "description": description,
                    "model": model,
                    "imageType": path.suffix.lower().lstrip("."),
                }
                print(json.dumps(output))
            else:
                print(f"Image Description (model: {model}):")
                print("-" * 40)
                print(description)

        except requests.exceptions.ConnectionError:
            error_msg = "Ollama not running. Start with: ollama serve"
            if output_json:
                print(json.dumps({"success": False, "error": error_msg}))
            else:
                print(f"Error: {error_msg}")
            raise SystemExit(1)
        except requests.exceptions.HTTPError as e:
            # Model might not be installed
            if "not found" in str(e).lower() or e.response.status_code == 404:
                error_msg = f"Vision model '{model}' not installed. Install with: ollama pull {model}"
            else:
                error_msg = f"Ollama API error: {e}"
            if output_json:
                print(json.dumps({"success": False, "error": error_msg}))
            else:
                print(f"Error: {error_msg}")
            raise SystemExit(1)

    except FileNotFoundError as e:
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        logger.exception("Image description failed")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")
        raise SystemExit(1)


@multimodal_app.command("status")
def multimodal_status(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Check availability of multimodal backends.

    Shows status of:
    - Whisper V3 (Ollama / HuggingFace)
    - DeepSeek-OCR / Tesseract
    - Vision models (LLaVA, etc.)
    """
    from ..extraction.whisper_client import whisper_available, get_whisper_models
    from ..extraction.ocr_client import ocr_available

    whisper_ok = whisper_available()
    whisper_models = get_whisper_models() if whisper_ok else []
    ocr_ok = ocr_available()

    # Check for vision models via Ollama
    vision_models = []
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.ok:
            models = response.json().get("models", [])
            # Filter for known vision models
            vision_prefixes = ("llava", "bakllava", "moondream", "cogvlm")
            vision_models = [
                m.get("name", "")
                for m in models
                if any(m.get("name", "").lower().startswith(p) for p in vision_prefixes)
            ]
    except Exception:
        pass  # Ollama not available

    if output_json:
        output = {
            "success": True,
            "whisper": {
                "available": whisper_ok,
                "backend": "ollama" if whisper_ok else "huggingface",
                "models": whisper_models,
            },
            "ocr": {
                "available": ocr_ok,
                "backend": "deepseek" if ocr_ok else "tesseract",
            },
            "vision": {
                "available": len(vision_models) > 0,
                "models": vision_models,
            },
        }
        print(json.dumps(output))
    else:
        print("Multimodal Backend Status:")
        print("-" * 40)
        print(f"Whisper V3: {'Ollama (fast)' if whisper_ok else 'HuggingFace (fallback)'}")
        if whisper_models:
            print(f"  Models: {', '.join(whisper_models)}")
        print(f"OCR: {'DeepSeek-OCR (SOTA)' if ocr_ok else 'Tesseract (fallback)'}")
        print(f"Vision: {len(vision_models)} model(s) available")
        if vision_models:
            print(f"  Models: {', '.join(vision_models)}")
