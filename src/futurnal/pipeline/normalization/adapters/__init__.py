"""Format-specific adapter implementations."""

from .code import CodeAdapter
from .email import EmailAdapter
from .generic import GenericAdapter
from .html import HTMLAdapter
from .markdown import MarkdownAdapter
from .pdf import PDFAdapter
from .text import TextAdapter

__all__ = [
    "MarkdownAdapter",
    "EmailAdapter",
    "HTMLAdapter",
    "CodeAdapter",
    "PDFAdapter",
    "TextAdapter",
    "GenericAdapter",
]
