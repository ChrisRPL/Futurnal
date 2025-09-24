"""Production-grade pipeline integrations for normalization and storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class PKGWriter(Protocol):
    def write_document(self, payload: dict) -> None:
        ...

    def remove_document(self, sha256: str) -> None:
        ...


class VectorWriter(Protocol):
    def write_embedding(self, payload: dict) -> None:
        ...

    def remove_embedding(self, sha256: str) -> None:
        ...


@dataclass
class NormalizationSink:
    pkg_writer: PKGWriter
    vector_writer: VectorWriter

    def handle(self, element: dict) -> None:
        with open(element["element_path"], "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        document_payload = {
            "sha256": element["sha256"],
            "path": element["path"],
            "source": element["source"],
            "metadata": payload.get("metadata", {}),
            "text": payload.get("text"),
        }
        self.pkg_writer.write_document(document_payload)
        embedding_payload = {
            "sha256": element["sha256"],
            "path": element["path"],
            "source": element["source"],
            "text": payload.get("text"),
        }
        if "embedding" in payload:
            embedding_payload["embedding"] = payload["embedding"]
        self.vector_writer.write_embedding(embedding_payload)

    def handle_deletion(self, element: dict) -> None:
        self.pkg_writer.remove_document(element["sha256"])
        self.vector_writer.remove_embedding(element["sha256"])

