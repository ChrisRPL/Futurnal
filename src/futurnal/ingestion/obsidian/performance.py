"""Performance optimization utilities for Obsidian markdown normalizer.

This module provides caching, memory management, and performance monitoring
utilities to optimize the normalization process for large documents and vaults.
"""

import functools
import gc
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)


class ContentCache:
    """LRU cache for normalized content to avoid re-processing unchanged documents."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Tuple[str, Any, float]] = {}  # hash -> (content_hash, result, timestamp)
        self._access_order: List[str] = []
    
    def get(self, content: str, file_path: Path) -> Optional[Any]:
        """Get cached result if content hasn't changed."""
        content_hash = self._hash_content(content, file_path)
        
        if content_hash in self._cache:
            stored_hash, result, timestamp = self._cache[content_hash]
            if stored_hash == content_hash:
                # Move to end (most recently used)
                self._access_order.remove(content_hash)
                self._access_order.append(content_hash)
                logger.debug(f"Cache hit for {file_path.name}")
                return result
        
        logger.debug(f"Cache miss for {file_path.name}")
        return None
    
    def put(self, content: str, file_path: Path, result: Any) -> None:
        """Cache the normalization result."""
        content_hash = self._hash_content(content, file_path)
        
        # Remove if already exists
        if content_hash in self._cache:
            self._access_order.remove(content_hash)
        
        # Add to cache
        self._cache[content_hash] = (content_hash, result, time.time())
        self._access_order.append(content_hash)
        
        # Evict if over limit
        while len(self._cache) > self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        logger.debug(f"Cached result for {file_path.name}")
    
    def _hash_content(self, content: str, file_path: Path) -> str:
        """Generate cache key from content and file metadata."""
        try:
            stat = file_path.stat()
            metadata = f"{stat.st_mtime}:{stat.st_size}:{file_path}"
        except OSError:
            metadata = str(file_path)
        
        combined = f"{content}:{metadata}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._access_order.clear()
        logger.debug("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hit_ratio': getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1)
        }


class MemoryMonitor:
    """Monitor and manage memory usage during normalization."""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self._start_memory = 0
    
    def __enter__(self):
        """Start memory monitoring."""
        import psutil
        process = psutil.Process()
        self._start_memory = process.memory_info().rss / (1024 * 1024)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Check memory usage and cleanup if needed."""
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024 * 1024)
            memory_used = current_memory - self._start_memory
            
            if memory_used > self.max_memory_mb:
                logger.warning(f"High memory usage detected: {memory_used:.1f}MB")
                self._force_cleanup()
            
            logger.debug(f"Memory usage: {memory_used:.1f}MB")
        except ImportError:
            logger.debug("psutil not available for memory monitoring")
    
    def _force_cleanup(self):
        """Force garbage collection and cleanup."""
        gc.collect()
        logger.debug("Forced garbage collection")


class ChunkedProcessor:
    """Process large documents in chunks to manage memory."""
    
    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB chunks
        self.chunk_size = chunk_size
    
    def should_chunk(self, content: str) -> bool:
        """Determine if content should be processed in chunks."""
        return len(content.encode('utf-8')) > self.chunk_size
    
    def chunk_content(self, content: str) -> List[str]:
        """Split content into processable chunks."""
        if not self.should_chunk(content):
            return [content]
        
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line.encode('utf-8'))
            
            if current_size + line_size > self.chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        logger.debug(f"Split content into {len(chunks)} chunks")
        return chunks


class PerformanceProfiler:
    """Profile performance of normalization operations."""
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
        self._counters: Dict[str, int] = {}
    
    def time_operation(self, operation: str):
        """Decorator to time an operation."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start_time
                    self.record_timing(operation, duration)
            return wrapper
        return decorator
    
    def record_timing(self, operation: str, duration: float) -> None:
        """Record timing for an operation."""
        if operation not in self._timings:
            self._timings[operation] = []
        self._timings[operation].append(duration)
        
        if operation not in self._counters:
            self._counters[operation] = 0
        self._counters[operation] += 1
    
    def increment_counter(self, counter: str) -> None:
        """Increment a performance counter."""
        if counter not in self._counters:
            self._counters[counter] = 0
        self._counters[counter] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for operation, timings in self._timings.items():
            if timings:
                stats[operation] = {
                    'count': len(timings),
                    'total_time': sum(timings),
                    'avg_time': sum(timings) / len(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                }
        
        stats['counters'] = self._counters.copy()
        return stats
    
    def log_stats(self) -> None:
        """Log performance statistics."""
        stats = self.get_stats()
        
        for operation, timing_stats in stats.items():
            if operation == 'counters':
                continue
            logger.info(
                f"{operation}: {timing_stats['count']} calls, "
                f"avg {timing_stats['avg_time']:.3f}s, "
                f"total {timing_stats['total_time']:.3f}s"
            )
        
        for counter, value in stats.get('counters', {}).items():
            logger.info(f"{counter}: {value}")


# Global instances
_content_cache = ContentCache()
_performance_profiler = PerformanceProfiler()


def get_content_cache() -> ContentCache:
    """Get the global content cache instance."""
    return _content_cache


def get_performance_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance."""
    return _performance_profiler




