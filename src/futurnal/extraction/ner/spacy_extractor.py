"""SpaCy-based Named Entity Recognition extractor.

Extracts Person, Organization, and Event entities from text content.
Uses spaCy's pre-trained English model for fast on-device NER.

Phase 1: Uses en_core_web_sm (12MB, fast)
Phase 2: Can upgrade to en_core_web_lg for better accuracy
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy load spaCy to avoid import overhead when not needed
_nlp = None


@dataclass
class NEREntity:
    """Entity extracted by NER."""

    text: str
    label: str  # PERSON, ORG, EVENT, GPE, etc.
    start_char: int
    end_char: int
    confidence: float = 0.85  # spaCy doesn't provide confidence by default


class SpacyEntityExtractor:
    """Extract entities from text using spaCy NER.

    Provides Person, Organization, and Event extraction from
    unstructured document content.

    Example:
        extractor = SpacyEntityExtractor()
        entities = extractor.extract("John Smith works at Anthropic.")
        # [NEREntity(text='John Smith', label='PERSON', ...),
        #  NEREntity(text='Anthropic', label='ORG', ...)]
    """

    # Map spaCy labels to our entity types
    LABEL_MAP = {
        "PERSON": "Person",
        "ORG": "Organization",
        "EVENT": "Event",
        "GPE": "Organization",  # Geopolitical entities often act like orgs
        "NORP": "Organization",  # Nationalities, religious groups
        "FAC": "Concept",  # Facilities
        "PRODUCT": "Concept",
        "WORK_OF_ART": "Concept",
    }

    # Labels we care about extracting
    RELEVANT_LABELS = {"PERSON", "ORG", "EVENT", "GPE", "NORP"}

    def __init__(self, model: str = "en_core_web_sm", max_length: int = 10000):
        """Initialize SpaCy extractor.

        Args:
            model: spaCy model name (en_core_web_sm, en_core_web_md, en_core_web_lg)
            max_length: Maximum text length to process (for performance)
        """
        self.model_name = model
        self.max_length = max_length
        self._nlp = None
        self._available = None

    def _load_model(self) -> bool:
        """Lazily load spaCy model."""
        if self._nlp is not None:
            return self._available

        try:
            import spacy

            self._nlp = spacy.load(self.model_name)
            self._available = True
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except ImportError:
            logger.warning("spaCy not installed. NER extraction disabled.")
            self._available = False
        except OSError:
            logger.warning(
                f"spaCy model '{self.model_name}' not found. "
                f"Run: python -m spacy download {self.model_name}"
            )
            self._available = False

        return self._available

    @property
    def is_available(self) -> bool:
        """Check if spaCy is available and model is loaded."""
        if self._available is None:
            self._load_model()
        return self._available

    def extract(self, text: str) -> List[NEREntity]:
        """Extract named entities from text.

        Args:
            text: Input text to analyze

        Returns:
            List of extracted entities
        """
        if not self._load_model():
            return []

        if not text or not text.strip():
            return []

        # Truncate for performance
        if len(text) > self.max_length:
            text = text[: self.max_length]
            logger.debug(f"Truncated text to {self.max_length} chars for NER")

        try:
            doc = self._nlp(text)

            entities = []
            seen_texts = set()  # Deduplicate

            for ent in doc.ents:
                if ent.label_ not in self.RELEVANT_LABELS:
                    continue

                # Normalize text for deduplication
                normalized = ent.text.strip().lower()
                if normalized in seen_texts:
                    continue
                seen_texts.add(normalized)

                entities.append(
                    NEREntity(
                        text=ent.text.strip(),
                        label=ent.label_,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        confidence=0.85,  # Default confidence
                    )
                )

            return entities

        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []

    def extract_as_dict(self, text: str) -> List[dict]:
        """Extract entities and return as dictionaries.

        Args:
            text: Input text to analyze

        Returns:
            List of entity dictionaries compatible with EntityExtractor
        """
        entities = self.extract(text)
        return [
            {
                "name": ent.text,
                "entity_type": self.LABEL_MAP.get(ent.label, "Concept"),
                "confidence": ent.confidence,
                "extraction_method": "ner_spacy",
            }
            for ent in entities
        ]


def extract_entities_from_text(text: str) -> List[dict]:
    """Convenience function for entity extraction.

    Args:
        text: Input text to analyze

    Returns:
        List of entity dictionaries
    """
    extractor = SpacyEntityExtractor()
    return extractor.extract_as_dict(text)
