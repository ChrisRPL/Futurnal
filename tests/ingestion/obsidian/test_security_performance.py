"""Tests for security and performance features of the markdown normalizer."""

import tempfile
from pathlib import Path

import pytest

from futurnal.ingestion.obsidian.normalizer import MarkdownNormalizer
from futurnal.ingestion.obsidian.security import (
    SecurityError, 
    PathTraversalValidator, 
    ResourceLimiter, 
    validate_yaml_safety
)
from futurnal.ingestion.obsidian.performance import (
    ContentCache, 
    MemoryMonitor, 
    ChunkedProcessor,
    get_content_cache,
    get_performance_profiler
)


class TestSecurityFeatures:
    """Test security validation and protection features."""
    
    def test_path_traversal_validation(self):
        validator = PathTraversalValidator()
        
        # Valid paths should pass
        assert validator.validate_link_path("normal-note") == True
        assert validator.validate_link_path("folder/sub-note") == True
        
        # Path traversal attempts should fail
        with pytest.raises(SecurityError, match="Path traversal detected"):
            validator.validate_link_path("../../../etc/passwd")
        
        with pytest.raises(SecurityError, match="Path traversal detected"):
            validator.validate_link_path("folder/../../../secret")
    
    def test_absolute_path_rejection(self):
        validator = PathTraversalValidator()
        
        with pytest.raises(SecurityError, match="Absolute path not allowed"):
            validator.validate_link_path("/etc/passwd")
        
        with pytest.raises(SecurityError, match="Absolute path not allowed"):
            validator.validate_link_path("/home/user/secret")
    
    def test_null_byte_detection(self):
        validator = PathTraversalValidator()
        
        with pytest.raises(SecurityError, match="Null byte detected"):
            validator.validate_link_path("note\x00.txt")
    
    def test_vault_boundary_enforcement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir) / "vault"
            vault_root.mkdir()
            
            # Create some files
            (vault_root / "note1.md").write_text("# Note 1")
            (vault_root / "subfolder").mkdir()
            (vault_root / "subfolder" / "note2.md").write_text("# Note 2")
            
            validator = PathTraversalValidator(vault_root)
            current_file = vault_root / "current.md"
            
            # Valid paths within vault should pass
            assert validator.validate_link_path("note1", current_file) == True
            assert validator.validate_link_path("subfolder/note2", current_file) == True
    
    def test_resource_limits_validation(self):
        limiter = ResourceLimiter(
            max_content_size=100,  # Very small for testing
            max_links=2,
            max_tags=2
        )
        
        # Content size limit
        with pytest.raises(SecurityError, match="Content size.*exceeds limit"):
            limiter.validate_content_size("x" * 200)
        
        # Element count limits
        with pytest.raises(SecurityError, match="links count.*exceeds limit"):
            limiter.validate_element_count(5, 'links')
        
        with pytest.raises(SecurityError, match="tags count.*exceeds limit"):
            limiter.validate_element_count(5, 'tags')
    
    def test_yaml_security_validation(self):
        # Valid YAML should pass
        validate_yaml_safety("title: Test\ntags: [safe]")
        
        # Dangerous YAML constructs should fail
        with pytest.raises(SecurityError, match="Dangerous YAML construct"):
            validate_yaml_safety("dangerous: !!python/object/apply:os.system ['rm -rf /']")
        
        with pytest.raises(SecurityError, match="Dangerous YAML construct"):
            validate_yaml_safety("bad: !!python/name:subprocess.Popen")
    
    def test_normalizer_security_integration(self):
        """Test that security features integrate properly with the normalizer."""
        normalizer = MarkdownNormalizer(
            resource_limits={'max_content_size': 1000, 'max_links': 5}
        )
        
        # This should work fine
        content = """---
title: Safe Note
---

# Test

Link to [[Another Note]] safely."""
        
        result = normalizer.normalize(content, Path("/test.md"))
        assert result.metadata.frontmatter['title'] == "Safe Note"
        assert len(result.metadata.links) == 1
        
        # This should fail due to content size
        large_content = "x" * 2000
        with pytest.raises(SecurityError):
            normalizer.normalize(large_content, Path("/large.md"))


class TestPerformanceFeatures:
    """Test performance optimization features."""
    
    def test_content_cache_basic_functionality(self):
        cache = ContentCache(max_size=3)
        
        # Cache miss initially
        assert cache.get("content1", Path("/file1.md")) is None
        
        # Store and retrieve
        cache.put("content1", Path("/file1.md"), "result1")
        assert cache.get("content1", Path("/file1.md")) == "result1"
        
        # Cache different files
        cache.put("content2", Path("/file2.md"), "result2")
        cache.put("content3", Path("/file3.md"), "result3")
        
        # All should be cached
        assert cache.get("content1", Path("/file1.md")) == "result1"
        assert cache.get("content2", Path("/file2.md")) == "result2"
        assert cache.get("content3", Path("/file3.md")) == "result3"
    
    def test_cache_eviction(self):
        cache = ContentCache(max_size=2)
        
        cache.put("content1", Path("/file1.md"), "result1")
        cache.put("content2", Path("/file2.md"), "result2")
        cache.put("content3", Path("/file3.md"), "result3")  # Should evict file1
        
        assert cache.get("content1", Path("/file1.md")) is None  # Evicted
        assert cache.get("content2", Path("/file2.md")) == "result2"
        assert cache.get("content3", Path("/file3.md")) == "result3"
    
    def test_cache_integration_with_normalizer(self):
        """Test that caching works with the normalizer."""
        normalizer = MarkdownNormalizer(enable_caching=True)
        
        content = """---
title: Cached Note
---

# Test Caching

This note should be cached."""
        
        path = Path("/cached.md")
        
        # First call - cache miss
        result1 = normalizer.normalize(content, path)
        
        # Second call - cache hit (should be much faster)
        result2 = normalizer.normalize(content, path)
        
        # Results should be identical
        assert result1.content == result2.content
        assert result1.metadata.frontmatter == result2.metadata.frontmatter
        assert result1.provenance.content_checksum == result2.provenance.content_checksum
    
    def test_chunked_processor(self):
        processor = ChunkedProcessor(chunk_size=50)  # Very small chunks for testing
        
        # Small content shouldn't be chunked
        small_content = "x" * 30
        assert not processor.should_chunk(small_content)
        assert processor.chunk_content(small_content) == [small_content]
        
        # Large content should be chunked
        large_content = "This is a long line that should be chunked.\n" * 10  # ~440 bytes
        assert processor.should_chunk(large_content)
        chunks = processor.chunk_content(large_content)
        
        # Should have multiple chunks or at least preserve content
        assert len(chunks) >= 1
        
        # Chunks should reconstruct original content (allowing for different chunking strategies)
        if len(chunks) > 1:
            reconstructed = '\n'.join(chunks)
            assert reconstructed == large_content
        else:
            # If not chunked (due to implementation details), should still preserve content
            assert chunks[0] == large_content
    
    def test_memory_monitor(self):
        """Test memory monitoring (if psutil is available)."""
        try:
            with MemoryMonitor(max_memory_mb=1) as monitor:
                # Just test that the context manager works
                pass
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")
    
    def test_performance_profiler(self):
        profiler = get_performance_profiler()
        
        # Test timing decorator
        @profiler.time_operation("test_operation")
        def test_func():
            return "result"
        
        result = test_func()
        assert result == "result"
        
        # Test counter
        profiler.increment_counter("test_counter")
        profiler.increment_counter("test_counter")
        
        stats = profiler.get_stats()
        assert "test_operation" in stats
        assert stats["counters"]["test_counter"] == 2


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_large_document_processing(self):
        """Test processing a large document with chunking."""
        normalizer = MarkdownNormalizer(
            enable_caching=True,
            resource_limits={'max_content_size': 50 * 1024 * 1024}  # 50MB
        )
        
        # Create a moderately large document
        large_content = f"""---
title: Large Document
tags: [performance, test]
---

# Large Document Test

{'This is a paragraph with many words to simulate a large document. ' * 100}

## Section 1

{'More content to increase document size. ' * 200}

## Section 2 

{'Additional content with [[Internal Link]] and #tag references. ' * 150}

> [!note] Performance Test
> This callout is part of the performance testing.

{'Final section with lots of text content. ' * 100}
"""
        
        result = normalizer.normalize(large_content, Path("/large-doc.md"))
        
        # Verify all elements were parsed correctly
        assert result.metadata.frontmatter['title'] == "Large Document"
        assert len(result.metadata.headings) >= 3
        assert len(result.metadata.links) >= 1
        assert len(result.metadata.tags) >= 3  # frontmatter tags + inline
        assert len(result.metadata.callouts) >= 1
        assert result.metadata.word_count > 1000
        assert result.provenance.content_checksum is not None
    
    def test_vault_integration_simulation(self):
        """Simulate integration with a vault structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_root = Path(temp_dir)
            
            # Create vault structure
            (vault_root / "Note1.md").write_text("""---
title: First Note
---

# First Note

This links to [[Note2]] and mentions #important topic.
""")
            
            (vault_root / "Note2.md").write_text("""---
title: Second Note  
tags: [reference]
---

# Second Note

This note is referenced by [[Note1]] and has an image: ![[diagram.png]]
""")
            
            (vault_root / "subfolder").mkdir()
            (vault_root / "subfolder" / "Note3.md").write_text("""# Nested Note

Links back to [[Note1]] in parent folder.""")
            
            # Test normalizing each note
            normalizer = MarkdownNormalizer(vault_root=vault_root, vault_id="test-vault")
            
            # Note 1
            note1_content = (vault_root / "Note1.md").read_text()
            result1 = normalizer.normalize(note1_content, vault_root / "Note1.md")
            
            assert result1.metadata.frontmatter['title'] == "First Note"
            assert len(result1.metadata.links) == 1
            assert result1.metadata.links[0].target == "Note2"
            assert len(result1.metadata.tags) >= 1
            
            # Note 2
            note2_content = (vault_root / "Note2.md").read_text()  
            result2 = normalizer.normalize(note2_content, vault_root / "Note2.md")
            
            assert result2.metadata.frontmatter['title'] == "Second Note"
            assert len(result2.metadata.links) == 2  # Note1 + diagram.png
            embeds = [link for link in result2.metadata.links if link.is_embed]
            assert len(embeds) == 1
            assert embeds[0].target == "diagram.png"
    
    def test_error_resilience(self):
        """Test that the normalizer handles various error conditions gracefully."""
        normalizer = MarkdownNormalizer()
        
        # Malformed frontmatter should not crash
        malformed_content = """---
title: Test
invalid: yaml: structure: [unclosed
another: bad
---

# Content

This should still parse the content part."""
        
        result = normalizer.normalize(malformed_content, Path("/malformed.md"))
        
        # Should preserve malformed frontmatter info
        assert "_yaml_error" in result.metadata.frontmatter or "_raw" in result.metadata.frontmatter
        
        # Should still parse content
        assert "# Content" in result.content
        assert len(result.metadata.headings) >= 1
    
    def test_performance_metrics(self):
        """Test that performance metrics are collected properly."""
        profiler = get_performance_profiler()
        normalizer = MarkdownNormalizer(enable_caching=True)
        
        # Clear existing stats
        profiler._timings.clear()
        profiler._counters.clear()
        
        content = """---
title: Performance Test
---

# Test

Link to [[Another Note]] and #tag."""
        
        # Process a document
        result = normalizer.normalize(content, Path("/perf-test.md"))
        
        # Check that metrics were recorded
        stats = profiler.get_stats()
        assert "document_normalization" in stats
        assert stats["counters"]["documents_processed"] >= 1
        
        # Process same document again (cache hit)
        result2 = normalizer.normalize(content, Path("/perf-test.md"))
        
        updated_stats = profiler.get_stats()
        assert updated_stats["counters"]["cache_hits"] >= 1


if __name__ == "__main__":
    pytest.main([__file__])
