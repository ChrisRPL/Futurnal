"""Markdown normalizer for Obsidian notes.

This module provides functionality to normalize Obsidian markdown to a consistent
intermediate form suitable for Unstructured.io processing and PKG enrichment
while preserving round-trip fidelity and supporting all Obsidian-specific syntax.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import BaseModel, Field
from mdit_py_plugins import tasklists, footnote
from mdit_py_plugins.front_matter import front_matter_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.deflist import deflist_plugin

from .security import PathTraversalValidator, ResourceLimiter, SecurityError, validate_yaml_safety
from .performance import (
    MemoryMonitor,
    ChunkedProcessor,
    get_content_cache,
    get_performance_profiler
)
from .assets import ObsidianAsset, AssetDetector, AssetResolver

logger = logging.getLogger(__name__)


class CalloutType(str, Enum):
    """Supported Obsidian callout types."""
    NOTE = "note"
    TIP = "tip"
    WARNING = "warning"
    DANGER = "danger"
    BUG = "bug"
    QUOTE = "quote"
    SUCCESS = "success"
    FAILURE = "failure"
    INFO = "info"
    ABSTRACT = "abstract"
    TODO = "todo"
    EXAMPLE = "example"


class FoldState(str, Enum):
    """Callout folding state."""
    DEFAULT = "default"
    EXPANDED = "+"
    COLLAPSED = "-"


@dataclass
class ObsidianCallout:
    """Represents an Obsidian callout block."""
    type: CalloutType
    title: Optional[str] = None
    fold_state: FoldState = FoldState.DEFAULT
    content: str = ""
    start_line: Optional[int] = None
    end_line: Optional[int] = None


@dataclass
class ObsidianLink:
    """Represents an Obsidian wikilink or embed."""
    target: str
    display_text: Optional[str] = None
    is_embed: bool = False
    section: Optional[str] = None  # For [[Note#Section]] links
    block_id: Optional[str] = None  # For [[Note^block-id]] links
    resolved_path: Optional[Path] = None
    is_broken: bool = False
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None


@dataclass 
class ObsidianTag:
    """Represents an Obsidian tag."""
    name: str
    is_nested: bool = False
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None


@dataclass
class MarkdownTable:
    """Represents a markdown table."""
    headers: List[str]
    rows: List[List[str]]
    alignments: List[Optional[str]] = field(default_factory=list)  # 'left', 'center', 'right', None
    start_line: Optional[int] = None
    end_line: Optional[int] = None


@dataclass
class TaskListItem:
    """Represents a task list item."""
    text: str
    checked: bool = False
    level: int = 0  # Indentation level
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None


@dataclass
class MarkdownFootnote:
    """Represents a markdown footnote."""
    label: str
    content: str
    reference_positions: List[int] = field(default_factory=list)  # Positions where it's referenced
    definition_line: Optional[int] = None


@dataclass
class MarkdownBlock:
    """Represents a structural block in the markdown document."""
    type: str  # heading, paragraph, code, list, etc.
    content: str
    level: Optional[int] = None  # For headings
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Structured metadata extracted from the document."""
    frontmatter: Dict[str, Any] = Field(default_factory=dict)
    frontmatter_raw: Optional[str] = Field(default=None, description="Raw frontmatter for round-trip")
    headings: List[MarkdownBlock] = Field(default_factory=list)
    links: List[ObsidianLink] = Field(default_factory=list)
    assets: List[ObsidianAsset] = Field(default_factory=list)
    tags: List[ObsidianTag] = Field(default_factory=list)
    callouts: List[ObsidianCallout] = Field(default_factory=list)
    blocks: List[MarkdownBlock] = Field(default_factory=list)
    tables: List[MarkdownTable] = Field(default_factory=list)
    task_lists: List[TaskListItem] = Field(default_factory=list)
    footnotes: List[MarkdownFootnote] = Field(default_factory=list)
    word_count: int = Field(default=0)
    reading_time_minutes: int = Field(default=0)


class ProvenanceInfo(BaseModel):
    """Provenance and deduplication information."""
    source_path: Path
    vault_id: Optional[str] = None
    content_checksum: str = Field(..., description="SHA-256 of normalized content")
    metadata_checksum: str = Field(..., description="SHA-256 of structured metadata")
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    normalizer_version: str = Field(default="1.0.0")


class NormalizedDocument(BaseModel):
    """Complete normalized document with content, metadata, and provenance."""
    content: str = Field(..., description="Normalized text for vector embedding")
    metadata: DocumentMetadata = Field(..., description="Structured metadata for graph triples") 
    provenance: ProvenanceInfo = Field(..., description="Checksum and lineage information")


class FrontmatterParser:
    """Robust YAML frontmatter parser with error handling and security validation."""
    
    FRONTMATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n', 
        re.MULTILINE | re.DOTALL
    )
    
    # Known Obsidian frontmatter keys that get special handling
    KNOWN_KEYS = {
        'title', 'aliases', 'tags', 'cssclass', 'publish', 'created', 'modified',
        'author', 'description', 'category', 'type', 'status', 'priority'
    }
    
    def __init__(self, resource_limiter: Optional[ResourceLimiter] = None):
        self.resource_limiter = resource_limiter or ResourceLimiter()
    
    @get_performance_profiler().time_operation("frontmatter_parsing")
    def parse(self, content: str) -> tuple[Dict[str, Any], Optional[str], str]:
        """Parse frontmatter from content.
        
        Returns:
            (parsed_frontmatter, raw_frontmatter, content_without_frontmatter)
        """
        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            return {}, None, content
            
        raw_frontmatter = match.group(1)
        content_without = content[match.end():]
        
        # Security validation
        try:
            self.resource_limiter.validate_frontmatter_size(raw_frontmatter)
            validate_yaml_safety(raw_frontmatter)
        except SecurityError as e:
            logger.warning(f"Frontmatter security validation failed: {e}")
            return {"_security_error": str(e)}, raw_frontmatter, content_without
        
        try:
            parsed = yaml.safe_load(raw_frontmatter) or {}
            if not isinstance(parsed, dict):
                # Invalid frontmatter structure, preserve as-is
                return {"_invalid_frontmatter": str(parsed)}, raw_frontmatter, content_without
                
            # Normalize known keys
            normalized = self._normalize_frontmatter(parsed)
            return normalized, raw_frontmatter, content_without
            
        except yaml.YAMLError as e:
            # Malformed YAML, preserve raw for round-trip
            return {"_yaml_error": str(e), "_raw": raw_frontmatter}, raw_frontmatter, content_without
    
    def _normalize_frontmatter(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize frontmatter keys to standard format."""
        normalized = {}
        extra = {}
        
        for key, value in parsed.items():
            if key.lower() in self.KNOWN_KEYS:
                # Normalize known keys
                if key.lower() == 'tags':
                    normalized['tags'] = self._normalize_tags(value)
                elif key.lower() == 'aliases':
                    normalized['aliases'] = self._normalize_aliases(value)
                else:
                    normalized[key.lower()] = value
            else:
                # Preserve unknown keys in extra
                extra[key] = value
        
        if extra:
            normalized['extra'] = extra
            
        return normalized
    
    def _normalize_tags(self, tags: Any) -> List[str]:
        """Normalize tags to consistent format."""
        if isinstance(tags, str):
            return [tags.strip('#').strip()]
        elif isinstance(tags, list):
            return [str(tag).strip('#').strip() for tag in tags if tag]
        else:
            return [str(tags).strip('#').strip()] if tags else []
    
    def _normalize_aliases(self, aliases: Any) -> List[str]:
        """Normalize aliases to consistent format."""
        if isinstance(aliases, str):
            return [aliases.strip()]
        elif isinstance(aliases, list):
            return [str(alias).strip() for alias in aliases if alias]
        else:
            return [str(aliases).strip()] if aliases else []


class ObsidianCalloutPlugin:
    """Markdown-it plugin for parsing Obsidian callouts."""
    
    CALLOUT_PATTERN = re.compile(
        r'^>\s*\[!([\w-]+)\](\s*([+-]))?\s*(.*)?$',
        re.MULTILINE
    )
    
    def __init__(self):
        self.callouts: List[ObsidianCallout] = []
    
    def parse(self, content: str) -> List[ObsidianCallout]:
        """Parse callouts from markdown content."""
        self.callouts = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = self.CALLOUT_PATTERN.match(line)
            
            if match:
                callout_type = match.group(1).lower()
                fold_indicator = match.group(3)
                title = match.group(4)
                
                # Validate callout type
                try:
                    ct = CalloutType(callout_type)
                except ValueError:
                    ct = CalloutType.NOTE  # Default fallback
                
                # Parse fold state
                fold_state = FoldState.DEFAULT
                if fold_indicator == '+':
                    fold_state = FoldState.EXPANDED
                elif fold_indicator == '-':
                    fold_state = FoldState.COLLAPSED
                
                # Collect callout content
                content_lines = []
                start_line = i
                i += 1
                
                while i < len(lines) and (lines[i].startswith('> ') or lines[i].strip() == '>'):
                    content_lines.append(lines[i][2:] if lines[i].startswith('> ') else '')
                    i += 1
                
                callout = ObsidianCallout(
                    type=ct,
                    title=title.strip() if title else None,
                    fold_state=fold_state,
                    content='\n'.join(content_lines),
                    start_line=start_line,
                    end_line=i - 1
                )
                self.callouts.append(callout)
            else:
                i += 1
        
        return self.callouts


class ObsidianLinkParser:
    """Parser for Obsidian wikilinks and embeds with security validation."""
    
    WIKILINK_PATTERN = re.compile(
        r'(!?)\[\[([^\]]+?)\]\]',
        re.MULTILINE
    )
    
    def __init__(
        self, 
        vault_root: Optional[Path] = None, 
        path_validator: Optional[PathTraversalValidator] = None,
        resource_limiter: Optional[ResourceLimiter] = None
    ):
        self.vault_root = vault_root
        self.path_validator = path_validator or PathTraversalValidator(vault_root)
        self.resource_limiter = resource_limiter or ResourceLimiter()
        self.links: List[ObsidianLink] = []
    
    @get_performance_profiler().time_operation("link_parsing")
    def parse(self, content: str, current_file_path: Optional[Path] = None) -> List[ObsidianLink]:
        """Parse wikilinks and embeds from content with security validation."""
        self.links = []
        
        for match in self.WIKILINK_PATTERN.finditer(content):
            is_embed = bool(match.group(1))
            link_text = match.group(2)
            
            # Parse link components
            target, display_text = self._parse_link_text(link_text)
            target, section, block_id = self._parse_target(target)
            
            # Security validation
            try:
                self.path_validator.validate_link_path(target, current_file_path)
            except SecurityError as e:
                logger.warning(f"Link security validation failed for '{target}': {e}")
                continue  # Skip dangerous links
            
            # Resolve path if vault root is available
            resolved_path = None
            is_broken = False
            if self.vault_root:
                resolved_path, is_broken = self._resolve_link(target)
            
            link = ObsidianLink(
                target=target,
                display_text=display_text,
                is_embed=is_embed,
                section=section,
                block_id=block_id,
                resolved_path=resolved_path,
                is_broken=is_broken,
                start_pos=match.start(),
                end_pos=match.end()
            )
            self.links.append(link)
        
        # Validate total link count
        self.resource_limiter.validate_element_count(len(self.links), 'links')
        return self.links
    
    def _parse_link_text(self, link_text: str) -> tuple[str, Optional[str]]:
        """Parse link text into target and display text."""
        if '|' in link_text:
            target, display_text = link_text.split('|', 1)
            return target.strip(), display_text.strip()
        return link_text.strip(), None
    
    def _parse_target(self, target: str) -> tuple[str, Optional[str], Optional[str]]:
        """Parse target into note name, section, and block ID."""
        section = None
        block_id = None
        
        # Check for block reference
        if '^' in target:
            target, block_id = target.split('^', 1)
            block_id = block_id.strip()
        
        # Check for section reference  
        if '#' in target:
            target, section = target.split('#', 1)
            section = section.strip()
        
        return target.strip(), section, block_id
    
    def _resolve_link(self, target: str) -> tuple[Optional[Path], bool]:
        """Resolve link target to absolute path."""
        if not self.vault_root or not target:
            return None, False
        
        # Try exact match first
        target_path = self.vault_root / f"{target}.md"
        if target_path.exists():
            return target_path, False
        
        # Try case-insensitive search
        for md_file in self.vault_root.rglob("*.md"):
            if md_file.stem.lower() == target.lower():
                return md_file, False
        
        return None, True


class ObsidianTagParser:
    """Parser for Obsidian tags."""
    
    TAG_PATTERN = re.compile(
        r'(?:^|(?<=\s))#([\w-]+(?:/[\w-]+)*)',
        re.MULTILINE
    )
    
    def parse(self, content: str) -> List[ObsidianTag]:
        """Parse inline tags from content."""
        tags = []
        
        for match in self.TAG_PATTERN.finditer(content):
            tag_name = match.group(1)
            is_nested = '/' in tag_name
            
            tag = ObsidianTag(
                name=tag_name,
                is_nested=is_nested,
                start_pos=match.start(),
                end_pos=match.end()
            )
            tags.append(tag)
        
        return tags


class MarkdownTableParser:
    """Parser for markdown tables."""
    
    def __init__(self, resource_limiter: Optional[ResourceLimiter] = None):
        self.resource_limiter = resource_limiter or ResourceLimiter()
    
    def parse(self, tokens: List[Token]) -> List[MarkdownTable]:
        """Parse tables from markdown tokens."""
        tables = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token.type == 'table_open':
                table = self._parse_table_tokens(tokens, i)
                if table:
                    tables.append(table)
            i += 1
        
        # Validate table count
        self.resource_limiter.validate_element_count(len(tables), 'tables')
        return tables
    
    def _parse_table_tokens(self, tokens: List[Token], start_idx: int) -> Optional[MarkdownTable]:
        """Parse a single table from tokens starting at start_idx."""
        headers = []
        rows = []
        alignments = []
        current_row = []
        
        i = start_idx + 1  # Skip table_open
        while i < len(tokens) and tokens[i].type != 'table_close':
            token = tokens[i]
            
            if token.type == 'thead_open':
                # Parse table header
                i += 1
                while i < len(tokens) and tokens[i].type != 'thead_close':
                    if tokens[i].type == 'th_open':
                        # Get the content of this header cell
                        i += 1
                        if i < len(tokens) and tokens[i].type == 'inline':
                            headers.append(tokens[i].content or '')
                        i += 1  # Skip th_close
                    else:
                        i += 1
                        
            elif token.type == 'tbody_open':
                # Parse table body
                i += 1
                while i < len(tokens) and tokens[i].type != 'tbody_close':
                    if tokens[i].type == 'tr_open':
                        current_row = []
                    elif tokens[i].type == 'td_open':
                        # Get the content of this cell
                        i += 1
                        if i < len(tokens) and tokens[i].type == 'inline':
                            current_row.append(tokens[i].content or '')
                        i += 1  # Skip td_close
                        continue
                    elif tokens[i].type == 'tr_close' and current_row:
                        rows.append(current_row)
                        current_row = []
                    i += 1
            else:
                i += 1
        
        if headers or rows:
            return MarkdownTable(
                headers=headers,
                rows=rows,
                alignments=alignments,
                start_line=tokens[start_idx].map[0] if tokens[start_idx].map else None,
                end_line=tokens[i].map[1] if i < len(tokens) and tokens[i].map else None
            )
        
        return None


class TaskListParser:
    """Parser for task list items."""
    
    TASK_PATTERN = re.compile(
        r'^(\s*)[-*+]\s+\[([ xX])\]\s+(.+)$',
        re.MULTILINE
    )
    
    def parse(self, content: str) -> List[TaskListItem]:
        """Parse task list items from content."""
        tasks = []
        
        for match in self.TASK_PATTERN.finditer(content):
            indent = match.group(1)
            check_state = match.group(2)
            task_text = match.group(3)
            
            level = len(indent) // 2  # Assuming 2-space indentation
            checked = check_state.lower() == 'x'
            
            task = TaskListItem(
                text=task_text.strip(),
                checked=checked,
                level=level,
                start_pos=match.start(),
                end_pos=match.end()
            )
            tasks.append(task)
        
        return tasks


class FootnoteParser:
    """Parser for markdown footnotes."""
    
    FOOTNOTE_REF_PATTERN = re.compile(r'(?<!\n)\[\^([^\]]+)\](?!:)')  # Not after newline, not followed by :
    FOOTNOTE_DEF_PATTERN = re.compile(r'^\[\^([^\]]+)\]:\s*(.+)$', re.MULTILINE)
    
    def parse(self, content: str) -> List[MarkdownFootnote]:
        """Parse footnotes from content."""
        footnotes = {}
        
        # Find footnote definitions
        for match in self.FOOTNOTE_DEF_PATTERN.finditer(content):
            label = match.group(1)
            definition_content = match.group(2)
            
            if label not in footnotes:
                footnotes[label] = MarkdownFootnote(
                    label=label,
                    content=definition_content.strip(),
                    definition_line=content[:match.start()].count('\n') + 1
                )
        
        # Find footnote references
        for match in self.FOOTNOTE_REF_PATTERN.finditer(content):
            label = match.group(1)
            
            if label in footnotes:
                footnotes[label].reference_positions.append(match.start())
            else:
                # Create footnote even if no definition found
                footnotes[label] = MarkdownFootnote(
                    label=label,
                    content="",  # No definition found
                    reference_positions=[match.start()]
                )
        
        return list(footnotes.values())


class MarkdownNormalizer:
    """Main orchestrator for normalizing Obsidian markdown with security and performance features."""
    
    def __init__(
        self, 
        vault_root: Optional[Path] = None, 
        vault_id: Optional[str] = None,
        enable_caching: bool = True,
        resource_limits: Optional[Dict[str, int]] = None
    ):
        self.vault_root = vault_root
        self.vault_id = vault_id
        self.enable_caching = enable_caching
        
        # Initialize security and performance components
        self.resource_limiter = ResourceLimiter(**(resource_limits or {}))
        self.path_validator = PathTraversalValidator(vault_root)
        self.chunked_processor = ChunkedProcessor()
        
        # Initialize parsers with security validation
        self.frontmatter_parser = FrontmatterParser(self.resource_limiter)
        self.callout_plugin = ObsidianCalloutPlugin()
        self.link_parser = ObsidianLinkParser(vault_root, self.path_validator, self.resource_limiter)
        self.asset_detector = AssetDetector()
        self.asset_resolver = AssetResolver(vault_root) if vault_root else None
        self.tag_parser = ObsidianTagParser()
        self.table_parser = MarkdownTableParser(self.resource_limiter)
        self.task_parser = TaskListParser()
        self.footnote_parser = FootnoteParser()
        
        # Initialize markdown-it with security-first configuration and plugins
        self.md = MarkdownIt('commonmark', {
            'html': False,  # Disable HTML for security
            'xhtmlOut': False,
            'breaks': False,
            'langPrefix': 'language-',
            'linkify': True,
            'typographer': False,
        })
        
        # Enable plugins for extended markdown support
        self.md.enable('table')  # Enable GFM tables
        self.md.use(tasklists.tasklists_plugin)  # Enable task lists
        self.md.use(footnote.footnote_plugin)  # Enable footnotes
    
    @get_performance_profiler().time_operation("document_normalization")
    def normalize(self, content: str, source_path: Path) -> NormalizedDocument:
        """Normalize Obsidian markdown document with security validation and caching."""
        
        # Check cache first
        if self.enable_caching:
            cache = get_content_cache()
            cached_result = cache.get(content, source_path)
            if cached_result:
                get_performance_profiler().increment_counter("cache_hits")
                return cached_result
        
        # Validate content size
        try:
            self.resource_limiter.validate_content_size(content)
        except SecurityError as e:
            logger.error(f"Content size validation failed for {source_path}: {e}")
            raise
        
        # Use memory monitoring for large documents
        with MemoryMonitor() as memory_monitor:
            result = self._normalize_internal(content, source_path)
        
        # Cache the result
        if self.enable_caching:
            cache.put(content, source_path, result)
        
        get_performance_profiler().increment_counter("documents_processed")
        return result
    
    def _normalize_internal(self, content: str, source_path: Path) -> NormalizedDocument:
        """Internal normalization implementation."""
        
        # Step 1: Parse frontmatter
        frontmatter, raw_frontmatter, content_body = self.frontmatter_parser.parse(content)
        
        # Step 2: Normalize content format
        normalized_content = self._normalize_content(content_body)
        
        # Step 3: Parse Obsidian-specific elements
        callouts = self.callout_plugin.parse(normalized_content)
        links = self.link_parser.parse(normalized_content, source_path)

        # Parse and resolve assets
        assets = self.asset_detector.detect_assets(normalized_content)
        if self.asset_resolver and assets:
            # Resolve asset paths with security validation
            resolved_assets = []
            for asset in assets:
                try:
                    resolved_asset = self.asset_resolver.resolve_asset(asset, source_path)
                    resolved_assets.append(resolved_asset)
                except Exception as e:
                    logger.warning(f"Failed to resolve asset {asset.target}: {e}")
                    asset.is_broken = True
                    resolved_assets.append(asset)
            assets = resolved_assets

        tags = self.tag_parser.parse(normalized_content)
        task_lists = self.task_parser.parse(normalized_content)
        footnotes = self.footnote_parser.parse(normalized_content)
        
        # Validate element counts
        self.resource_limiter.validate_element_count(len(callouts), 'callouts')
        self.resource_limiter.validate_element_count(len(links), 'links')
        self.resource_limiter.validate_element_count(len(assets), 'assets')
        self.resource_limiter.validate_element_count(len(tags), 'tags')
        self.resource_limiter.validate_element_count(len(task_lists), 'tasks')
        self.resource_limiter.validate_element_count(len(footnotes), 'footnotes')
        
        # Step 4: Parse markdown structure (with chunking for large documents)
        if self.chunked_processor.should_chunk(normalized_content):
            logger.debug(f"Processing large document {source_path.name} in chunks")
            blocks, tables = self._parse_blocks_chunked(normalized_content)
        else:
            blocks, tables = self._parse_blocks(normalized_content)
            
        headings = [block for block in blocks if block.type == 'heading']
        self.resource_limiter.validate_element_count(len(headings), 'headings')
        self.resource_limiter.validate_element_count(len(tables), 'tables')
        
        # Step 5: Calculate text statistics
        word_count = len(normalized_content.split())
        reading_time = max(1, word_count // 200)  # 200 WPM reading speed
        
        # Step 6: Combine all tags (frontmatter + inline)
        all_tags = self._merge_tags(frontmatter.get('tags', []), tags)
        
        # Step 7: Build metadata
        metadata = DocumentMetadata(
            frontmatter=frontmatter,
            frontmatter_raw=raw_frontmatter,
            headings=headings,
            links=links,
            assets=assets,
            tags=all_tags,
            callouts=callouts,
            blocks=blocks,
            tables=tables,
            task_lists=task_lists,
            footnotes=footnotes,
            word_count=word_count,
            reading_time_minutes=reading_time
        )
        
        # Step 8: Generate provenance
        provenance = self._generate_provenance(normalized_content, metadata, source_path)
        
        return NormalizedDocument(
            content=normalized_content,
            metadata=metadata,
            provenance=provenance
        )
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content format."""
        # Standardize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Normalize whitespace (but preserve structure)
        lines = content.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Remove trailing whitespace but preserve leading whitespace for structure
            normalized_lines.append(line.rstrip())
        
        # Remove excessive empty lines (max 2 consecutive)
        result_lines = []
        empty_count = 0
        
        for line in normalized_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _parse_blocks(self, content: str) -> tuple[List[MarkdownBlock], List[MarkdownTable]]:
        """Parse markdown into structural blocks and tables."""
        blocks = []
        tokens = self.md.parse(content)
        
        # Parse tables using the table parser
        tables = self.table_parser.parse(tokens)
        
        current_line = 0
        for token in tokens:
            if token.type in ['heading_open', 'paragraph_open', 'code_block', 'fence']:
                block_type = self._normalize_block_type(token.type)
                
                # Find the content token
                content_token = self._find_content_token(tokens, token)
                content_text = content_token.content if content_token else ""
                
                # Extract metadata
                metadata = {}
                if token.type == 'heading_open':
                    metadata['level'] = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.
                elif token.type == 'fence':
                    metadata['language'] = token.info or ""
                
                block = MarkdownBlock(
                    type=block_type,
                    content=content_text,
                    level=metadata.get('level'),
                    start_line=current_line,
                    end_line=current_line + content_text.count('\n'),
                    metadata=metadata
                )
                blocks.append(block)
                current_line = block.end_line + 1
        
        return blocks, tables
    
    def _parse_blocks_chunked(self, content: str) -> tuple[List[MarkdownBlock], List[MarkdownTable]]:
        """Parse markdown blocks and tables in chunks for large documents."""
        chunks = self.chunked_processor.chunk_content(content)
        all_blocks = []
        all_tables = []
        line_offset = 0
        
        for chunk in chunks:
            chunk_blocks, chunk_tables = self._parse_blocks(chunk)
            
            # Adjust line numbers for chunk offset
            for block in chunk_blocks:
                if block.start_line is not None:
                    block.start_line += line_offset
                if block.end_line is not None:
                    block.end_line += line_offset
            
            for table in chunk_tables:
                if table.start_line is not None:
                    table.start_line += line_offset
                if table.end_line is not None:
                    table.end_line += line_offset
            
            all_blocks.extend(chunk_blocks)
            all_tables.extend(chunk_tables)
            line_offset += chunk.count('\n') + 1
        
        return all_blocks, all_tables
    
    def _normalize_block_type(self, token_type: str) -> str:
        """Normalize token types to standard block types."""
        mapping = {
            'heading_open': 'heading',
            'paragraph_open': 'paragraph', 
            'code_block': 'code',
            'fence': 'code',
            'list_item_open': 'list_item',
            'blockquote_open': 'blockquote',
        }
        return mapping.get(token_type, token_type)
    
    def _find_content_token(self, tokens: List[Token], open_token: Token) -> Optional[Token]:
        """Find the content token corresponding to an open token."""
        for i, token in enumerate(tokens):
            if token == open_token and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.type.endswith('_inline') or next_token.type == 'code_block':
                    return next_token
        return None
    
    def _merge_tags(self, frontmatter_tags: List[str], inline_tags: List[ObsidianTag]) -> List[ObsidianTag]:
        """Merge tags from frontmatter and inline tags."""
        all_tags = []
        tag_names = set()
        
        # Add frontmatter tags
        for tag_name in frontmatter_tags:
            if tag_name not in tag_names:
                all_tags.append(ObsidianTag(name=tag_name, is_nested='/' in tag_name))
                tag_names.add(tag_name)
        
        # Add inline tags (avoiding duplicates)
        for tag in inline_tags:
            if tag.name not in tag_names:
                all_tags.append(tag)
                tag_names.add(tag.name)
        
        return all_tags
    
    def _generate_provenance(
        self, 
        content: str, 
        metadata: DocumentMetadata, 
        source_path: Path
    ) -> ProvenanceInfo:
        """Generate provenance and checksums."""
        content_checksum = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Create deterministic metadata representation for checksumming
        metadata_repr = {
            'frontmatter': metadata.frontmatter,
            'heading_count': len(metadata.headings),
            'link_count': len(metadata.links),
            'tag_count': len(metadata.tags),
            'callout_count': len(metadata.callouts),
            'table_count': len(metadata.tables),
            'task_count': len(metadata.task_lists),
            'footnote_count': len(metadata.footnotes),
            'word_count': metadata.word_count,
        }
        metadata_json = str(sorted(metadata_repr.items()))
        metadata_checksum = hashlib.sha256(metadata_json.encode('utf-8')).hexdigest()
        
        return ProvenanceInfo(
            source_path=source_path,
            vault_id=self.vault_id,
            content_checksum=content_checksum,
            metadata_checksum=metadata_checksum
        )


def normalize_obsidian_document(
    content: str,
    source_path: Path,
    *,
    vault_root: Optional[Path] = None,
    vault_id: Optional[str] = None
) -> NormalizedDocument:
    """Convenience function to normalize an Obsidian document."""
    normalizer = MarkdownNormalizer(vault_root=vault_root, vault_id=vault_id)
    return normalizer.normalize(content, source_path)
