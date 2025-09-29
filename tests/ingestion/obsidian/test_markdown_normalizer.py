"""Tests for Obsidian markdown normalizer."""

import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from futurnal.ingestion.obsidian.normalizer import (
    MarkdownNormalizer,
    FrontmatterParser,
    ObsidianCalloutPlugin,
    ObsidianLinkParser,
    ObsidianTagParser,
    MarkdownTableParser,
    TaskListParser,
    FootnoteParser,
    normalize_obsidian_document,
    CalloutType,
    FoldState,
)


class TestFrontmatterParser:
    """Test frontmatter parsing functionality."""
    
    def test_valid_yaml_frontmatter(self):
        parser = FrontmatterParser()
        content = """---
title: Test Note
tags: [test, sample]
created: 2023-01-01
---

This is the content."""
        
        frontmatter, raw, content_body = parser.parse(content)
        
        assert frontmatter['title'] == "Test Note"
        assert frontmatter['tags'] == ['test', 'sample']
        assert frontmatter['created'] == datetime(2023, 1, 1).date()
        assert raw is not None
        assert content_body == "This is the content."
    
    def test_malformed_yaml_frontmatter(self):
        parser = FrontmatterParser()
        content = """---
title: Test Note
tags: [invalid yaml structure
created: 2023-01-01
---

This is the content."""
        
        frontmatter, raw, content_body = parser.parse(content)
        
        assert "_yaml_error" in frontmatter
        assert "_raw" in frontmatter
        assert raw is not None
        assert content_body == "This is the content."
    
    def test_no_frontmatter(self):
        parser = FrontmatterParser()
        content = "This is just regular content without frontmatter."
        
        frontmatter, raw, content_body = parser.parse(content)
        
        assert frontmatter == {}
        assert raw is None
        assert content_body == content
    
    def test_frontmatter_tag_normalization(self):
        parser = FrontmatterParser()
        content = """---
tags: 
  - "#tag1"
  - "tag2"
  - tag3
---

Content here."""
        
        frontmatter, _, _ = parser.parse(content)
        
        assert frontmatter['tags'] == ['tag1', 'tag2', 'tag3']
    
    def test_unknown_keys_preserved(self):
        parser = FrontmatterParser()
        content = """---
title: Test
custom_field: custom_value
another_field: 123
---

Content."""
        
        frontmatter, _, _ = parser.parse(content)
        
        assert frontmatter['title'] == "Test"
        assert frontmatter['extra']['custom_field'] == "custom_value"
        assert frontmatter['extra']['another_field'] == 123


class TestObsidianCalloutPlugin:
    """Test Obsidian callout parsing."""
    
    def test_simple_callout(self):
        plugin = ObsidianCalloutPlugin()
        content = """> [!note] This is a note
> This is the content of the note.
> It spans multiple lines."""
        
        callouts = plugin.parse(content)
        
        assert len(callouts) == 1
        callout = callouts[0]
        assert callout.type == CalloutType.NOTE
        assert callout.title == "This is a note"
        assert callout.fold_state == FoldState.DEFAULT
        assert "This is the content" in callout.content
    
    def test_foldable_callouts(self):
        plugin = ObsidianCalloutPlugin()
        content = """> [!tip]+ Expanded tip
> This tip is expanded by default.

> [!warning]- Collapsed warning  
> This warning is collapsed by default."""
        
        callouts = plugin.parse(content)
        
        assert len(callouts) == 2
        
        # Expanded tip
        tip = callouts[0]
        assert tip.type == CalloutType.TIP
        assert tip.fold_state == FoldState.EXPANDED
        assert tip.title == "Expanded tip"
        
        # Collapsed warning
        warning = callouts[1]
        assert warning.type == CalloutType.WARNING
        assert warning.fold_state == FoldState.COLLAPSED
        assert warning.title == "Collapsed warning"
    
    def test_callout_without_title(self):
        plugin = ObsidianCalloutPlugin()
        content = """> [!info]
> Just some info without a title."""
        
        callouts = plugin.parse(content)
        
        assert len(callouts) == 1
        callout = callouts[0]
        assert callout.type == CalloutType.INFO
        assert callout.title is None
        assert "Just some info" in callout.content
    
    def test_invalid_callout_type_defaults_to_note(self):
        plugin = ObsidianCalloutPlugin()
        content = """> [!invalid-type] Unknown callout
> This should default to note type."""
        
        callouts = plugin.parse(content)
        
        assert len(callouts) == 1
        assert callouts[0].type == CalloutType.NOTE


class TestObsidianLinkParser:
    """Test Obsidian wikilink parsing."""
    
    def test_simple_wikilink(self):
        parser = ObsidianLinkParser()
        content = "This is a link to [[Another Note]] in the text."
        
        links = parser.parse(content)
        
        assert len(links) == 1
        link = links[0]
        assert link.target == "Another Note"
        assert link.display_text is None
        assert not link.is_embed
        assert link.section is None
        assert link.block_id is None
    
    def test_wikilink_with_display_text(self):
        parser = ObsidianLinkParser()
        content = "Check out [[My Note|this amazing note]] for details."
        
        links = parser.parse(content)
        
        assert len(links) == 1
        link = links[0]
        assert link.target == "My Note"
        assert link.display_text == "this amazing note"
    
    def test_embed_syntax(self):
        parser = ObsidianLinkParser()
        content = "Here's an embedded image: ![[diagram.png]]"
        
        links = parser.parse(content)
        
        assert len(links) == 1
        link = links[0]
        assert link.target == "diagram.png"  # Keep extension for embeds
        assert link.is_embed
    
    def test_section_links(self):
        parser = ObsidianLinkParser()
        content = "See [[My Note#Section Title]] for more info."
        
        links = parser.parse(content)
        
        assert len(links) == 1
        link = links[0]
        assert link.target == "My Note"
        assert link.section == "Section Title"
    
    def test_block_references(self):
        parser = ObsidianLinkParser()
        content = "Reference this block: [[Note^block-id]]"
        
        links = parser.parse(content)
        
        assert len(links) == 1
        link = links[0]
        assert link.target == "Note"
        assert link.block_id == "block-id"
    
    def test_link_resolution_with_vault(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)
            
            # Create test files
            note_file = vault_root / "Test Note.md"
            note_file.write_text("# Test Note\nThis is a test note.")
            
            parser = ObsidianLinkParser(vault_root)
            content = "Link to [[Test Note]] here."
            
            links = parser.parse(content)
            
            assert len(links) == 1
            link = links[0]
            assert link.resolved_path == note_file
            assert not link.is_broken
    
    def test_broken_link_detection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)
            parser = ObsidianLinkParser(vault_root)
            content = "Link to [[Nonexistent Note]] here."
            
            links = parser.parse(content)
            
            assert len(links) == 1
            link = links[0]
            assert link.resolved_path is None
            assert link.is_broken


class TestObsidianTagParser:
    """Test Obsidian tag parsing."""
    
    def test_simple_tags(self):
        parser = ObsidianTagParser()
        content = "This note has #tag1 and #tag2 in it."
        
        tags = parser.parse(content)
        
        assert len(tags) == 2
        assert tags[0].name == "tag1"
        assert tags[1].name == "tag2"
        assert not tags[0].is_nested
        assert not tags[1].is_nested
    
    def test_nested_tags(self):
        parser = ObsidianTagParser()
        content = "Categories: #work/project #personal/health"
        
        tags = parser.parse(content)
        
        assert len(tags) == 2
        assert tags[0].name == "work/project"
        assert tags[1].name == "personal/health"
        assert tags[0].is_nested
        assert tags[1].is_nested
    
    def test_tags_at_word_boundaries(self):
        parser = ObsidianTagParser()
        content = "Email address test@example.com should not be tagged."
        
        tags = parser.parse(content)
        
        assert len(tags) == 0  # @ symbol should not create tags


class TestMarkdownTableParser:
    """Test markdown table parsing."""
    
    def test_simple_table_parsing(self):
        normalizer = MarkdownNormalizer()
        content = """# Test Document

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |

Some text after the table."""
        
        result = normalizer.normalize(content, Path("/test.md"))
        
        assert len(result.metadata.tables) == 1
        table = result.metadata.tables[0]
        assert table.headers == ["Header 1", "Header 2", "Header 3"]
        assert len(table.rows) == 2
        assert table.rows[0] == ["Cell 1", "Cell 2", "Cell 3"]
        assert table.rows[1] == ["Cell 4", "Cell 5", "Cell 6"]
    
    def test_table_with_alignment(self):
        normalizer = MarkdownNormalizer()
        content = """| Left | Center | Right |
|:-----|:------:|------:|
| L1   | C1     | R1    |
| L2   | C2     | R2    |"""
        
        result = normalizer.normalize(content, Path("/test.md"))
        
        assert len(result.metadata.tables) == 1
        table = result.metadata.tables[0]
        assert table.headers == ["Left", "Center", "Right"]
        assert table.rows[0] == ["L1", "C1", "R1"]
        assert table.rows[1] == ["L2", "C2", "R2"]
    
    def test_empty_table_cells(self):
        normalizer = MarkdownNormalizer()
        content = """| Name | Value |
|------|-------|
| Test |       |
|      | Empty |"""
        
        result = normalizer.normalize(content, Path("/test.md"))
        
        assert len(result.metadata.tables) == 1
        table = result.metadata.tables[0]
        assert table.rows[0] == ["Test", ""]
        assert table.rows[1] == ["", "Empty"]


class TestTaskListParser:
    """Test task list parsing."""
    
    def test_simple_task_list(self):
        parser = TaskListParser()
        content = """- [ ] Unchecked task
- [x] Checked task
- [X] Also checked"""
        
        tasks = parser.parse(content)
        
        assert len(tasks) == 3
        assert tasks[0].text == "Unchecked task"
        assert not tasks[0].checked
        assert tasks[1].text == "Checked task"
        assert tasks[1].checked
        assert tasks[2].text == "Also checked"
        assert tasks[2].checked
    
    def test_nested_task_list(self):
        parser = TaskListParser()
        content = """- [x] Main task
  - [ ] Subtask 1
  - [x] Subtask 2
    - [ ] Sub-subtask"""
        
        tasks = parser.parse(content)
        
        assert len(tasks) == 4
        assert tasks[0].level == 0
        assert tasks[0].checked
        assert tasks[1].level == 1
        assert not tasks[1].checked
        assert tasks[2].level == 1
        assert tasks[2].checked
        assert tasks[3].level == 2
        assert not tasks[3].checked
    
    def test_task_list_in_document(self):
        normalizer = MarkdownNormalizer()
        content = """# My Tasks

Today's priorities:
- [x] Review pull requests
- [ ] Update documentation  
- [ ] Fix bug #123

## Shopping List
- [x] Milk
- [ ] Bread
- [ ] Eggs"""
        
        result = normalizer.normalize(content, Path("/tasks.md"))
        
        assert len(result.metadata.task_lists) == 6
        completed_tasks = [task for task in result.metadata.task_lists if task.checked]
        assert len(completed_tasks) == 2
        
        pending_tasks = [task for task in result.metadata.task_lists if not task.checked]
        assert len(pending_tasks) == 4


class TestFootnoteParser:
    """Test footnote parsing."""
    
    def test_simple_footnotes(self):
        parser = FootnoteParser()
        content = """This is a sentence with a footnote[^1].

[^1]: This is the footnote content."""
        
        footnotes = parser.parse(content)
        
        assert len(footnotes) == 1
        footnote = footnotes[0]
        assert footnote.label == "1"
        assert footnote.content == "This is the footnote content."
        assert len(footnote.reference_positions) == 1
    
    def test_multiple_footnotes(self):
        parser = FootnoteParser()
        content = """First footnote[^note1] and second[^note2].

[^note1]: First footnote content.
[^note2]: Second footnote content."""
        
        footnotes = parser.parse(content)
        
        assert len(footnotes) == 2
        labels = [fn.label for fn in footnotes]
        assert "note1" in labels
        assert "note2" in labels
    
    def test_footnote_multiple_references(self):
        parser = FootnoteParser()
        content = """First reference[^1] and another reference[^1].

[^1]: Shared footnote content."""
        
        footnotes = parser.parse(content)
        
        assert len(footnotes) == 1
        footnote = footnotes[0]
        assert footnote.label == "1"
        assert len(footnote.reference_positions) == 2
    
    def test_footnotes_in_document(self):
        normalizer = MarkdownNormalizer()
        content = """# Research Notes

This study[^study1] shows interesting results. Another study[^study2] confirms this.

[^study1]: Smith et al., 2023. "Research Findings."
[^study2]: Jones & Brown, 2022. "Confirmatory Study."  """
        
        result = normalizer.normalize(content, Path("/research.md"))
        
        assert len(result.metadata.footnotes) == 2
        labels = [fn.label for fn in result.metadata.footnotes]
        assert "study1" in labels
        assert "study2" in labels


class TestMarkdownNormalizer:
    """Test the main normalizer functionality."""
    
    def test_simple_document_normalization(self):
        normalizer = MarkdownNormalizer()
        content = """---
title: Test Document
tags: [test]
---

# Heading 1

This is a paragraph with a [[wikilink]] and a #tag.

> [!note] This is a callout
> With some content.

## Heading 2

More content here."""
        
        source_path = Path("/test/document.md")
        result = normalizer.normalize(content, source_path)
        
        # Check basic structure
        assert result.content is not None
        assert result.metadata is not None
        assert result.provenance is not None
        
        # Check metadata
        assert result.metadata.frontmatter['title'] == "Test Document"
        assert 'test' in [tag.name for tag in result.metadata.tags]
        assert len(result.metadata.headings) == 2
        assert len(result.metadata.links) == 1
        assert len(result.metadata.callouts) == 1
        
        # Check provenance
        assert result.provenance.source_path == source_path
        assert result.provenance.content_checksum is not None
        assert result.provenance.metadata_checksum is not None
    
    def test_content_normalization(self):
        normalizer = MarkdownNormalizer()
        content = "Line 1\r\n\r\nLine 2\r\n\r\n\r\n\r\nLine 3   \n\n\n\n\nLine 4"
        
        normalized = normalizer._normalize_content(content)
        
        # Should normalize line endings and limit excessive blank lines
        lines = normalized.split('\n')
        empty_streaks = []
        current_streak = 0
        
        for line in lines:
            if line.strip() == '':
                current_streak += 1
            else:
                if current_streak > 0:
                    empty_streaks.append(current_streak)
                    current_streak = 0
        
        # No more than 2 consecutive empty lines
        assert all(streak <= 2 for streak in empty_streaks)
    
    def test_word_count_and_reading_time(self):
        normalizer = MarkdownNormalizer()
        content = """---
title: Test
---

# Heading

This is a test document with exactly twenty words in this paragraph to test the word counting functionality properly."""
        
        result = normalizer.normalize(content, Path("/test.md"))
        
        # Should count words correctly (20 in paragraph + 1 in heading = 21)
        assert result.metadata.word_count == 21
        assert result.metadata.reading_time_minutes == 1  # Minimum 1 minute
    
    def test_tag_merging(self):
        normalizer = MarkdownNormalizer()
        content = """---
title: Test
tags: [frontmatter-tag, shared-tag]
---

This document has #inline-tag and #shared-tag in it."""
        
        result = normalizer.normalize(content, Path("/test.md"))
        
        tag_names = [tag.name for tag in result.metadata.tags]
        assert "frontmatter-tag" in tag_names
        assert "inline-tag" in tag_names
        assert "shared-tag" in tag_names
        # Should not duplicate shared-tag
        assert tag_names.count("shared-tag") == 1


class TestIntegrationFixtures:
    """Integration tests with complex realistic fixtures."""
    
    def test_complex_obsidian_note(self):
        """Test with a complex note containing all Obsidian features."""
        normalizer = MarkdownNormalizer()
        content = """---
title: "Complex Research Note"
aliases: ["Research", "Complex Note"]
tags: [research, methodology, important]
created: 2023-05-15
author: John Doe
status: in-progress
priority: high
custom_field: custom_value
---

# Complex Research Note

This note demonstrates various [[Obsidian]] features and #research methodologies.

## Introduction

![[research-diagram.png]]

> [!abstract] Research Overview
> This section provides an overview of our #methodology approach.

## Methodology

The approach includes several steps:

1. Initial [[Data Collection|data gathering]]
2. Analysis using [[Statistical Methods#Advanced Techniques]]
3. Validation through [[Peer Review^validation-block]]

> [!warning]+ Important Considerations
> - Ensure data privacy compliance
> - Follow ethical guidelines
> - Maintain #documentation standards

### Task List

- [x] Complete data collection
- [ ] Perform initial analysis
- [ ] Review findings with team
  - [x] Schedule meeting
  - [ ] Prepare presentation

## Results

```python
def analyze_data(dataset):
    return dataset.process()
```

### Key Findings

> [!success] Major Discovery
> We found significant correlations in the #dataset.

The findings show:
- Pattern A correlates with #outcome-variable
- Pattern B shows #temporal-trends

### Data Summary

| Metric | Value | Significance |
|--------|--------|--------------|
| Correlation | 0.85 | High |
| P-value | 0.001 | Very significant |
| Sample Size | 1000 | Adequate |

This research[^1] builds on previous work[^2] in the field.

## References

- [[Smith et al. 2023]] on methodological approaches
- [[Data Standards]] for compliance requirements

[^1]: Our primary research methodology paper.
[^2]: Foundation work by Johnson et al., 2022.

---

*Last updated: 2023-05-15*"""
        
        result = normalizer.normalize(content, Path("/vault/research-note.md"))
        
        # Validate frontmatter processing
        assert result.metadata.frontmatter['title'] == "Complex Research Note"
        assert result.metadata.frontmatter['aliases'] == ["Research", "Complex Note"]
        assert result.metadata.frontmatter['status'] == "in-progress"
        assert result.metadata.frontmatter['extra']['custom_field'] == "custom_value"
        
        # Validate structure parsing
        headings = [h for h in result.metadata.blocks if h.type == 'heading']
        assert len(headings) >= 4  # Should find main headings
        
        # Validate links
        link_targets = [link.target for link in result.metadata.links]
        assert "Obsidian" in link_targets
        assert "Data Collection" in link_targets
        assert "Statistical Methods" in link_targets
        
        # Check for section and block references
        section_links = [link for link in result.metadata.links if link.section]
        block_links = [link for link in result.metadata.links if link.block_id]
        assert len(section_links) >= 1
        assert len(block_links) >= 1
        
        # Validate tags (frontmatter + inline)
        tag_names = [tag.name for tag in result.metadata.tags]
        assert "research" in tag_names
        assert "methodology" in tag_names
        assert "documentation" in tag_names
        assert "dataset" in tag_names
        
        # Validate callouts
        assert len(result.metadata.callouts) >= 3
        callout_types = [callout.type for callout in result.metadata.callouts]
        assert CalloutType.ABSTRACT in callout_types
        assert CalloutType.WARNING in callout_types
        assert CalloutType.SUCCESS in callout_types
        
        # Validate tables
        assert len(result.metadata.tables) == 1
        table = result.metadata.tables[0]
        assert "Metric" in table.headers
        assert "Value" in table.headers
        assert "Significance" in table.headers
        assert len(table.rows) == 3  # Three data rows
        
        # Validate task lists
        assert len(result.metadata.task_lists) == 5
        completed_tasks = [task for task in result.metadata.task_lists if task.checked]
        assert len(completed_tasks) == 2  # Two completed tasks
        
        # Validate footnotes
        assert len(result.metadata.footnotes) == 2
        footnote_labels = [fn.label for fn in result.metadata.footnotes]
        assert "1" in footnote_labels
        assert "2" in footnote_labels
        
        # Validate provenance
        assert result.provenance.content_checksum is not None
        assert result.provenance.metadata_checksum is not None
        assert result.provenance.source_path.name == "research-note.md"
    
    def test_malformed_document_handling(self):
        """Test graceful handling of malformed documents."""
        normalizer = MarkdownNormalizer()
        content = """---
title: Malformed Note
tags: [unclosed yaml list
invalid: yaml: structure
---

# Heading

This has broken [[links with missing] brackets and invalid #tag@symbols.

> [!invalid-type] Unknown callout
> Should handle gracefully.

> [!note Missing closing bracket
> Should still parse other callouts.

> [!tip]+ Valid callout
> This one should work fine."""
        
        result = normalizer.normalize(content, Path("/malformed.md"))
        
        # Should not crash and should preserve what it can
        assert result.content is not None
        assert result.metadata is not None
        
        # Should preserve malformed frontmatter
        assert "_yaml_error" in result.metadata.frontmatter or "_raw" in result.metadata.frontmatter
        
        # Should find the valid callout
        valid_callouts = [c for c in result.metadata.callouts if c.type == CalloutType.TIP]
        assert len(valid_callouts) >= 1
    
    def test_comprehensive_markdown_features(self):
        """Test document with all supported markdown features."""
        normalizer = MarkdownNormalizer()
        content = """---
title: Feature Test Document
tags: [comprehensive, test]
---

# Comprehensive Feature Test

This document tests all supported markdown features.

## Tables

| Feature | Status | Notes |
|---------|--------|-------|
| Tables | ✅ Working | Full support |
| Task Lists | ✅ Working | With nesting |
| Footnotes | ✅ Working | Multiple refs |

## Task Lists

### Project Tasks
- [x] Implement table parsing
- [x] Add task list support
- [ ] Add footnote parsing
  - [x] Basic footnote syntax
  - [ ] Complex footnote content
- [ ] Update documentation

## Footnotes

This is a reference to footnote 1[^fn1]. Here's another footnote[^complex].

Multiple references to the same footnote[^fn1] should work too.

## Callouts

> [!note] Implementation Note
> All features are now fully implemented.

> [!tip]+ Expandable Tip  
> You can combine all these features in one document.

## Links and Tags

Links to [[Other Documents]] and tags like #implementation work as before.

[^fn1]: Simple footnote content.
[^complex]: More complex footnote with **formatting** and links to [[Other Page]]."""
        
        result = normalizer.normalize(content, Path("/comprehensive.md"))
        
        # Verify all features are parsed
        assert len(result.metadata.tables) == 1
        assert len(result.metadata.task_lists) == 6  # All task items
        assert len(result.metadata.footnotes) == 2
        assert len(result.metadata.callouts) == 2
        assert len(result.metadata.links) >= 2  # [[Other Documents]], [[Other Page]]
        
        # Verify table structure
        table = result.metadata.tables[0]
        assert table.headers == ["Feature", "Status", "Notes"]
        assert len(table.rows) == 3
        
        # Verify task hierarchy
        tasks = result.metadata.task_lists
        nested_tasks = [task for task in tasks if task.level > 0]
        assert len(nested_tasks) == 2
        
        # Verify footnote references
        footnotes = result.metadata.footnotes
        fn1 = next(fn for fn in footnotes if fn.label == "fn1")
        assert len(fn1.reference_positions) == 2  # Referenced twice
        
        # Verify metadata counts in provenance
        assert result.provenance.content_checksum is not None
        assert result.provenance.metadata_checksum is not None


def test_normalize_obsidian_document_convenience_function():
    """Test the convenience function."""
    content = """---
title: Test
---

# Test Note

Simple content with a [[link]] and #tag."""
    
    result = normalize_obsidian_document(
        content, 
        Path("/test.md"),
        vault_id="test-vault"
    )
    
    assert result.provenance.vault_id == "test-vault"
    assert result.metadata.frontmatter['title'] == "Test"
    assert len(result.metadata.links) == 1
    assert len(result.metadata.tags) == 1


if __name__ == "__main__":
    pytest.main([__file__])
