"""Named Entity Recognition (NER) extraction module.

Provides entity extraction from unstructured text content using spaCy.

Phase 1: Basic NER with spaCy en_core_web_sm
Phase 2: Custom model training for personal domain entities
"""

from .spacy_extractor import SpacyEntityExtractor, NEREntity

__all__ = ["SpacyEntityExtractor", "NEREntity"]
