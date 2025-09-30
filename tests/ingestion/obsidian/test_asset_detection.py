"""Tests for asset detection functionality."""

import pytest
from pathlib import Path

from futurnal.ingestion.obsidian.assets import (
    AssetDetector,
    ObsidianAsset,
    SUPPORTED_ASSET_EXTENSIONS
)


class TestAssetDetector:
    """Test cases for AssetDetector class."""

    def test_basic_initialization(self):
        """Test basic AssetDetector initialization."""
        detector = AssetDetector()
        assert detector.supported_extensions == SUPPORTED_ASSET_EXTENSIONS

    def test_custom_extensions(self):
        """Test AssetDetector with custom extensions."""
        custom_extensions = {'.jpg', '.png', '.pdf'}
        detector = AssetDetector(custom_extensions)
        assert detector.supported_extensions == custom_extensions

    def test_detect_wikilink_image_embeds(self):
        """Test detection of wikilink image embeds."""
        content = """
        Here's an image: ![[screenshot.png]]
        And another: ![[diagram.jpg]]
        And a PDF: ![[document.pdf]]
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 3

        # Check first asset
        assert assets[0].target == "screenshot.png"
        assert assets[0].is_embed is True
        assert assets[0].display_text is None
        assert assets[0].start_pos is not None
        assert assets[0].end_pos is not None

        # Check second asset
        assert assets[1].target == "diagram.jpg"
        assert assets[1].is_embed is True

        # Check third asset
        assert assets[2].target == "document.pdf"
        assert assets[2].is_embed is True

    def test_detect_markdown_image_links(self):
        """Test detection of markdown image links."""
        content = """
        Here's an image: ![Alt text](images/photo.png)
        No alt text: ![](chart.jpg)
        With title: ![Description](assets/diagram.svg "Title")
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 3

        # Check first asset
        assert assets[0].target == "images/photo.png"
        assert assets[0].display_text == "Alt text"
        assert assets[0].is_embed is False  # Markdown image, not wikilink embed

        # Check second asset (no alt text)
        assert assets[1].target == "chart.jpg"
        assert assets[1].display_text == ""
        assert assets[1].is_embed is False

        # Check third asset (with title - title should be ignored)
        assert assets[2].target == "assets/diagram.svg"
        assert assets[2].display_text == "Description"
        assert assets[2].is_embed is False

    def test_mixed_asset_types(self):
        """Test detection of mixed asset types in one document."""
        content = """
        # Document with assets

        Wikilink embed: ![[flowchart.png]]
        Markdown image: ![Screenshot](./screenshots/app.jpg)
        Another embed: ![[report.pdf]]

        Some text content.

        ![](icon.gif)
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 4

        # Verify asset types
        wikilink_assets = [a for a in assets if a.is_embed]
        markdown_assets = [a for a in assets if not a.is_embed]

        assert len(wikilink_assets) == 2
        assert len(markdown_assets) == 2

    def test_ignore_non_asset_wikilinks(self):
        """Test that non-asset wikilinks are ignored."""
        content = """
        This is a note link: [[Some Note]]
        This is an asset: ![[image.png]]
        Another note: [[Another Note|Display Text]]
        Another asset: ![Alt](photo.jpg)
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 2
        assert assets[0].target == "image.png"
        assert assets[1].target == "photo.jpg"

    def test_unsupported_file_extensions(self):
        """Test that unsupported file extensions are ignored."""
        content = """
        Supported: ![[image.png]]
        Unsupported: ![[document.txt]]
        Also unsupported: ![Video](movie.mp4)
        Supported: ![PDF](file.pdf)
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 2
        assert assets[0].target == "image.png"
        assert assets[1].target == "file.pdf"

    def test_custom_supported_extensions(self):
        """Test detection with custom supported extensions."""
        content = """
        Default supported: ![[image.png]]
        Custom supported: ![[video.mp4]]
        Not supported: ![[document.txt]]
        """

        # Detector with custom extensions
        custom_extensions = {'.png', '.mp4'}
        detector = AssetDetector(custom_extensions)
        assets = detector.detect_assets(content)

        assert len(assets) == 2
        assert assets[0].target == "image.png"
        assert assets[1].target == "video.mp4"

    def test_section_and_block_references_removed(self):
        """Test that section and block references are removed from asset targets."""
        content = """
        Asset with section: ![[image.png#section]]
        Asset with block: ![[diagram.jpg^block-id]]
        Normal asset: ![[photo.gif]]
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 3
        # Section and block references should be removed when checking if it's an asset
        assert assets[0].target == "image.png#section"  # Original target preserved
        assert assets[1].target == "diagram.jpg^block-id"  # Original target preserved
        assert assets[2].target == "photo.gif"

    def test_empty_content(self):
        """Test detection with empty content."""
        detector = AssetDetector()
        assets = detector.detect_assets("")
        assert len(assets) == 0

    def test_no_assets_in_content(self):
        """Test detection with content containing no assets."""
        content = """
        # Regular markdown document

        This is just text with [[regular links]] and no assets.

        - List item
        - Another item

        Some **bold** and *italic* text.
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)
        assert len(assets) == 0

    def test_malformed_syntax_ignored(self):
        """Test that malformed syntax is ignored gracefully."""
        content = """
        Incomplete wikilink: ![[
        Unclosed markdown: ![Alt text](
        Malformed: ![[]]
        Empty markdown: ![](
        Valid: ![[image.png]]
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        # Should only detect the valid asset
        assert len(assets) == 1
        assert assets[0].target == "image.png"

    def test_position_tracking(self):
        """Test that start and end positions are correctly tracked."""
        content = "Here is ![[test.png]] in text."

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 1
        asset = assets[0]

        # Check positions
        assert asset.start_pos == 8  # Position of "![[test.png]]"
        assert asset.end_pos == 21   # End position

        # Verify by extracting the substring
        extracted = content[asset.start_pos:asset.end_pos]
        assert extracted == "![[test.png]]"

    def test_case_sensitive_extensions(self):
        """Test that file extension matching is case-insensitive."""
        content = """
        Lowercase: ![[image.png]]
        Uppercase: ![[IMAGE.PNG]]
        Mixed case: ![[Image.Png]]
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 3
        assert all(detector._is_asset_target(asset.target) for asset in assets)

    def test_whitespace_handling(self):
        """Test handling of whitespace in asset targets."""
        content = """
        With spaces: ![[image with spaces.png]]
        With tabs: ![Alt text](	image	.jpg)
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 2
        assert assets[0].target == "image with spaces.png"
        assert assets[1].target == "	image	.jpg"  # Whitespace preserved

    def test_special_characters_in_filenames(self):
        """Test handling of special characters in asset filenames."""
        content = """
        Underscore: ![[my_image.png]]
        Dash: ![[my-image.png]]
        Parentheses: ![[image(1).jpg]]
        Brackets: ![Alt]([image].png)
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 4
        assert assets[0].target == "my_image.png"
        assert assets[1].target == "my-image.png"
        assert assets[2].target == "image(1).jpg"
        assert assets[3].target == "[image].png"

    def test_multiple_assets_same_line(self):
        """Test detection of multiple assets on the same line."""
        content = "Icons: ![[icon1.png]] and ![[icon2.png]] and ![Alt](icon3.jpg)"

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 3
        assert assets[0].target == "icon1.png"
        assert assets[1].target == "icon2.png"
        assert assets[2].target == "icon3.jpg"

        # Check positions don't overlap
        assert assets[0].end_pos <= assets[1].start_pos
        assert assets[1].end_pos <= assets[2].start_pos

    def test_nested_markdown_structures(self):
        """Test asset detection within nested markdown structures."""
        content = """
        > Blockquote with ![[quote-image.png]]

        - List item with ![](list-icon.jpg)
          - Nested item with ![[nested.png]]

        | Table | Cell |
        |-------|------|
        | ![Icon](table.png) | Content |
        """

        detector = AssetDetector()
        assets = detector.detect_assets(content)

        assert len(assets) == 4
        targets = [asset.target for asset in assets]
        assert "quote-image.png" in targets
        assert "list-icon.jpg" in targets
        assert "nested.png" in targets
        assert "table.png" in targets


class TestAssetTargetValidation:
    """Test cases for asset target validation logic."""

    def test_is_asset_target_valid_extensions(self):
        """Test _is_asset_target with valid extensions."""
        detector = AssetDetector()

        valid_targets = [
            "image.png",
            "photo.jpg",
            "document.pdf",
            "diagram.svg",
            "picture.jpeg",
            "graphics.gif",
            "bitmap.bmp",
            "web_image.webp"
        ]

        for target in valid_targets:
            assert detector._is_asset_target(target), f"Should detect {target} as asset"

    def test_is_asset_target_invalid_extensions(self):
        """Test _is_asset_target with invalid extensions."""
        detector = AssetDetector()

        invalid_targets = [
            "document.txt",
            "file.doc",
            "spreadsheet.xlsx",
            "presentation.pptx",
            "archive.zip",
            "video.mp4",
            "audio.mp3",
            "NoExtension"
        ]

        for target in invalid_targets:
            assert not detector._is_asset_target(target), f"Should not detect {target} as asset"

    def test_is_asset_target_with_sections(self):
        """Test _is_asset_target with section references."""
        detector = AssetDetector()

        # Should ignore section references when checking extension
        assert detector._is_asset_target("image.png#section")
        assert detector._is_asset_target("document.pdf#chapter1")
        assert not detector._is_asset_target("note.md#section")

    def test_is_asset_target_with_blocks(self):
        """Test _is_asset_target with block references."""
        detector = AssetDetector()

        # Should ignore block references when checking extension
        assert detector._is_asset_target("image.png^block123")
        assert detector._is_asset_target("diagram.jpg^my-block")
        assert not detector._is_asset_target("note.md^block")

    def test_is_asset_target_edge_cases(self):
        """Test _is_asset_target with edge cases."""
        detector = AssetDetector()

        # Empty or None targets
        assert not detector._is_asset_target("")
        assert not detector._is_asset_target(None)

        # Invalid path characters should be handled gracefully
        # The implementation should return False for invalid paths rather than raise exceptions
        assert not detector._is_asset_target("invalid/\x00/path.png")

    def test_is_asset_target_case_insensitive(self):
        """Test that extension matching is case insensitive."""
        detector = AssetDetector()

        assert detector._is_asset_target("IMAGE.PNG")
        assert detector._is_asset_target("Photo.JPG")
        assert detector._is_asset_target("Document.PDF")
        assert detector._is_asset_target("Mixed.JpEg")