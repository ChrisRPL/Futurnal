"""Optimized file watching for Obsidian vaults with performance enhancements.

This module provides high-performance file watching capabilities specifically
optimized for large Obsidian vaults, with support for intelligent debouncing,
filtering, and optional migration to watchfiles for better performance.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class WatcherBackend(Enum):
    """Available file watching backends."""
    WATCHDOG = "watchdog"
    WATCHFILES = "watchfiles"
    AUTO = "auto"


class FileEventType(Enum):
    """Types of file system events."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileEvent:
    """Represents a file system event."""
    event_type: FileEventType
    path: Path
    timestamp: datetime = field(default_factory=datetime.utcnow)
    src_path: Optional[Path] = None  # For move events
    metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize path
        self.path = Path(self.path).resolve()
        if self.src_path:
            self.src_path = Path(self.src_path).resolve()

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "path": str(self.path),
            "src_path": str(self.src_path) if self.src_path else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class WatcherConfig:
    """Configuration for file watcher."""
    # Performance settings
    debounce_window_seconds: float = 1.0
    batch_window_seconds: float = 2.0
    max_events_per_batch: int = 100
    max_queue_size: int = 10000

    # Filtering settings
    include_patterns: List[str] = field(default_factory=lambda: ["**/*.md", "**/*.markdown"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/.obsidian/**",
        "**/.trash/**",
        "**/.git/**",
        "**/node_modules/**",
        "**/.DS_Store",
        "**/Thumbs.db",
        "**/*.tmp",
        "**/*~",
    ])
    watch_subdirectories: bool = True

    # Backend settings
    preferred_backend: WatcherBackend = WatcherBackend.AUTO
    watchdog_use_polling: bool = False
    watchfiles_ignore_paths: Optional[List[str]] = None

    # Large vault optimizations
    enable_large_vault_mode: bool = False
    large_vault_threshold: int = 5000
    large_vault_sample_rate: float = 1.0  # Fraction of events to process


class FileWatcherBackend(ABC):
    """Abstract base class for file watching backends."""

    @abstractmethod
    async def start_watching(self, path: Path, callback: Callable[[FileEvent], None]) -> None:
        """Start watching a path for changes."""
        pass

    @abstractmethod
    async def stop_watching(self) -> None:
        """Stop watching for changes."""
        pass

    @abstractmethod
    def is_watching(self) -> bool:
        """Check if currently watching."""
        pass


class WatchdogBackend(FileWatcherBackend):
    """Watchdog-based file watching backend."""

    def __init__(self, config: WatcherConfig):
        self.config = config
        self._observer = None
        self._handler = None
        self._callback = None

    async def start_watching(self, path: Path, callback: Callable[[FileEvent], None]) -> None:
        """Start watching using watchdog."""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
            from watchdog.observers.polling import PollingObserver

            self._callback = callback

            class OptimizedEventHandler(FileSystemEventHandler):
                def __init__(self, watcher_config: WatcherConfig, callback_func: Callable):
                    self.config = watcher_config
                    self.callback = callback_func
                    self._last_events: Dict[str, float] = {}

                def on_any_event(self, event):
                    if event.is_directory:
                        return

                    # Apply debouncing
                    event_key = f"{event.event_type}:{event.src_path or event.dest_path}"
                    current_time = time.time()
                    last_time = self._last_events.get(event_key, 0)

                    if current_time - last_time < self.config.debounce_window_seconds:
                        return

                    self._last_events[event_key] = current_time

                    # Convert to our event format
                    file_event = self._convert_watchdog_event(event)
                    if file_event and self._should_include_event(file_event):
                        self.callback(file_event)

                def _convert_watchdog_event(self, event) -> Optional[FileEvent]:
                    """Convert watchdog event to our FileEvent format."""
                    try:
                        if hasattr(event, 'dest_path'):
                            # Move event
                            return FileEvent(
                                event_type=FileEventType.MOVED,
                                path=Path(event.dest_path),
                                src_path=Path(event.src_path),
                                metadata={"watchdog_type": event.event_type}
                            )
                        else:
                            # Other events
                            event_type_map = {
                                'created': FileEventType.CREATED,
                                'modified': FileEventType.MODIFIED,
                                'deleted': FileEventType.DELETED,
                            }

                            file_event_type = event_type_map.get(event.event_type)
                            if not file_event_type:
                                return None

                            return FileEvent(
                                event_type=file_event_type,
                                path=Path(event.src_path),
                                metadata={"watchdog_type": event.event_type}
                            )
                    except Exception as e:
                        logger.debug(f"Failed to convert watchdog event: {e}")
                        return None

                def _should_include_event(self, file_event: FileEvent) -> bool:
                    """Check if event should be included based on filters."""
                    return _matches_patterns(file_event.path, self.config.include_patterns, self.config.exclude_patterns)

            self._handler = OptimizedEventHandler(self.config, callback)

            # Choose observer type
            if self.config.watchdog_use_polling:
                self._observer = PollingObserver()
            else:
                self._observer = Observer()

            self._observer.schedule(
                self._handler,
                str(path),
                recursive=self.config.watch_subdirectories
            )
            self._observer.start()

            logger.info(f"Started watchdog file watching on {path}")

        except ImportError:
            logger.error("Watchdog is not available. Please install with: pip install watchdog")
            raise
        except Exception as e:
            logger.error(f"Failed to start watchdog backend: {e}")
            raise

    async def stop_watching(self) -> None:
        """Stop watchdog observer."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
            logger.info("Stopped watchdog file watching")

    def is_watching(self) -> bool:
        """Check if watchdog is currently watching."""
        return self._observer is not None and self._observer.is_alive()


class WatchfilesBackend(FileWatcherBackend):
    """Watchfiles-based file watching backend (Rust implementation)."""

    def __init__(self, config: WatcherConfig):
        self.config = config
        self._watch_task = None
        self._stop_event = None
        self._callback = None

    async def start_watching(self, path: Path, callback: Callable[[FileEvent], None]) -> None:
        """Start watching using watchfiles."""
        try:
            import watchfiles

            self._callback = callback
            self._stop_event = asyncio.Event()

            # Create the watch task
            self._watch_task = asyncio.create_task(self._watch_loop(path, watchfiles))

            logger.info(f"Started watchfiles file watching on {path}")

        except ImportError:
            logger.error("Watchfiles is not available. Please install with: pip install watchfiles")
            raise
        except Exception as e:
            logger.error(f"Failed to start watchfiles backend: {e}")
            raise

    async def _watch_loop(self, path: Path, watchfiles_module) -> None:
        """Main watch loop for watchfiles."""
        try:
            # Configure ignore patterns
            ignore_paths = self.config.watchfiles_ignore_paths or []
            ignore_paths.extend([
                '**/.obsidian/**',
                '**/.trash/**',
                '**/.git/**',
                '**/node_modules/**',
            ])

            async for changes in watchfiles_module.awatch(
                path,
                ignore_paths=ignore_paths,
                watch_filter=self._create_watchfiles_filter(),
                stop_event=self._stop_event,
                debounce=int(self.config.debounce_window_seconds * 1000)  # Convert to ms
            ):
                if self._stop_event.is_set():
                    break

                # Process changes
                for change_type, file_path in changes:
                    file_event = self._convert_watchfiles_event(change_type, file_path)
                    if file_event:
                        self._callback(file_event)

        except Exception as e:
            logger.error(f"Watchfiles loop error: {e}", exc_info=True)

    def _create_watchfiles_filter(self):
        """Create a filter function for watchfiles."""
        def filter_func(change, path: str) -> bool:
            """Filter function for watchfiles."""
            file_path = Path(path)
            return _matches_patterns(file_path, self.config.include_patterns, self.config.exclude_patterns)

        return filter_func

    def _convert_watchfiles_event(self, change_type, file_path: str) -> Optional[FileEvent]:
        """Convert watchfiles event to our FileEvent format."""
        try:
            # Map watchfiles change types to our enum
            from watchfiles import Change

            type_map = {
                Change.added: FileEventType.CREATED,
                Change.modified: FileEventType.MODIFIED,
                Change.deleted: FileEventType.DELETED,
            }

            event_type = type_map.get(change_type)
            if not event_type:
                return None

            return FileEvent(
                event_type=event_type,
                path=Path(file_path),
                metadata={"watchfiles_type": str(change_type)}
            )

        except Exception as e:
            logger.debug(f"Failed to convert watchfiles event: {e}")
            return None

    async def stop_watching(self) -> None:
        """Stop watchfiles watching."""
        if self._stop_event:
            self._stop_event.set()

        if self._watch_task:
            try:
                await asyncio.wait_for(self._watch_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._watch_task.cancel()
                try:
                    await self._watch_task
                except asyncio.CancelledError:
                    pass

            self._watch_task = None

        logger.info("Stopped watchfiles file watching")

    def is_watching(self) -> bool:
        """Check if watchfiles is currently watching."""
        return self._watch_task is not None and not self._watch_task.done()


class OptimizedFileWatcher:
    """High-performance file watcher optimized for large Obsidian vaults."""

    def __init__(
        self,
        config: Optional[WatcherConfig] = None,
        event_callback: Optional[Callable[[List[FileEvent]], None]] = None
    ):
        self.config = config or WatcherConfig()
        self._event_callback = event_callback
        self._backend: Optional[FileWatcherBackend] = None
        self._running = False

        # Event batching and processing
        self._event_queue: deque[FileEvent] = deque(maxlen=self.config.max_queue_size)
        self._last_batch_time = time.time()
        self._processing_task: Optional[asyncio.Task] = None

        # Performance tracking
        self._events_processed = 0
        self._events_dropped = 0
        self._last_stats_log = time.time()

        # Large vault optimization
        self._file_count_estimate = 0
        self._large_vault_mode = False

    async def start_watching(self, path: Path) -> None:
        """Start watching the specified path."""
        if self._running:
            await self.stop_watching()

        # Estimate vault size for optimization
        await self._estimate_vault_size(path)

        # Select and configure backend
        self._backend = await self._create_backend()

        # Start event processing task
        self._processing_task = asyncio.create_task(self._process_event_batches())

        # Start the backend
        await self._backend.start_watching(path, self._handle_file_event)

        self._running = True
        logger.info(f"Started optimized file watching on {path} (large vault mode: {self._large_vault_mode})")

    async def stop_watching(self) -> None:
        """Stop watching."""
        if not self._running:
            return

        self._running = False

        # Stop backend
        if self._backend:
            await self._backend.stop_watching()
            self._backend = None

        # Stop processing task
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

        # Process remaining events
        await self._flush_event_queue()

        logger.info("Stopped optimized file watching")

    def set_event_callback(self, callback: Callable[[List[FileEvent]], None]) -> None:
        """Set the event callback function."""
        self._event_callback = callback

    async def _estimate_vault_size(self, path: Path) -> None:
        """Estimate vault size to enable optimizations."""
        try:
            # Quick estimation by counting markdown files
            md_count = 0
            for md_file in path.rglob("*.md"):
                md_count += 1
                if md_count > self.config.large_vault_threshold:
                    break

            self._file_count_estimate = md_count
            self._large_vault_mode = (
                self.config.enable_large_vault_mode and
                md_count > self.config.large_vault_threshold
            )

            logger.info(f"Estimated vault size: {md_count} markdown files (large vault mode: {self._large_vault_mode})")

        except Exception as e:
            logger.warning(f"Failed to estimate vault size: {e}")
            self._file_count_estimate = 0
            self._large_vault_mode = False

    async def _create_backend(self) -> FileWatcherBackend:
        """Create the appropriate backend based on configuration."""
        backend_type = self.config.preferred_backend

        if backend_type == WatcherBackend.AUTO:
            # Auto-select based on availability and vault size
            try:
                import watchfiles
                if self._large_vault_mode:
                    backend_type = WatcherBackend.WATCHFILES
                else:
                    backend_type = WatcherBackend.WATCHDOG
            except ImportError:
                backend_type = WatcherBackend.WATCHDOG

        if backend_type == WatcherBackend.WATCHFILES:
            return WatchfilesBackend(self.config)
        else:
            return WatchdogBackend(self.config)

    def _handle_file_event(self, event: FileEvent) -> None:
        """Handle a file event from the backend."""
        # Apply large vault optimizations
        if self._large_vault_mode:
            # Sample events to reduce load
            import random
            if random.random() > self.config.large_vault_sample_rate:
                self._events_dropped += 1
                return

        # Add to queue
        try:
            self._event_queue.append(event)
            self._events_processed += 1
        except Exception as e:
            logger.debug(f"Failed to queue event: {e}")
            self._events_dropped += 1

        # Log stats periodically
        current_time = time.time()
        if current_time - self._last_stats_log > 60:  # Every minute
            logger.debug(
                f"File watcher stats: {self._events_processed} processed, "
                f"{self._events_dropped} dropped, queue size: {len(self._event_queue)}"
            )
            self._last_stats_log = current_time

    async def _process_event_batches(self) -> None:
        """Process events in batches."""
        while self._running:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms

                current_time = time.time()
                should_process = (
                    len(self._event_queue) >= self.config.max_events_per_batch or
                    (self._event_queue and current_time - self._last_batch_time >= self.config.batch_window_seconds)
                )

                if should_process:
                    await self._flush_event_queue()

            except Exception as e:
                logger.error(f"Error in event processing loop: {e}", exc_info=True)

    async def _flush_event_queue(self) -> None:
        """Flush the event queue by processing all pending events."""
        if not self._event_queue:
            return

        # Extract events from queue
        events = list(self._event_queue)
        self._event_queue.clear()
        self._last_batch_time = time.time()

        # Group and deduplicate events
        processed_events = self._deduplicate_events(events)

        # Send to callback
        if self._event_callback and processed_events:
            try:
                self._event_callback(processed_events)
            except Exception as e:
                logger.error(f"Event callback failed: {e}", exc_info=True)

        logger.debug(f"Processed batch of {len(processed_events)} events (from {len(events)} raw events)")

    def _deduplicate_events(self, events: List[FileEvent]) -> List[FileEvent]:
        """Deduplicate and merge similar events."""
        # Group events by file path
        path_events: Dict[Path, List[FileEvent]] = defaultdict(list)
        for event in events:
            path_events[event.path].append(event)

        # Process each file's events
        deduplicated = []
        for path, file_events in path_events.items():
            if len(file_events) == 1:
                deduplicated.append(file_events[0])
            else:
                # Keep only the most recent event per file, with some logic for moves
                file_events.sort(key=lambda e: e.timestamp)

                # If there's a move event, prioritize it
                move_events = [e for e in file_events if e.event_type == FileEventType.MOVED]
                if move_events:
                    deduplicated.append(move_events[-1])
                else:
                    deduplicated.append(file_events[-1])

        return deduplicated

    def get_stats(self) -> Dict[str, any]:
        """Get watcher statistics."""
        return {
            "running": self._running,
            "events_processed": self._events_processed,
            "events_dropped": self._events_dropped,
            "queue_size": len(self._event_queue),
            "file_count_estimate": self._file_count_estimate,
            "large_vault_mode": self._large_vault_mode,
            "backend_type": type(self._backend).__name__ if self._backend else None,
        }


def _matches_patterns(path: Path, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    """Check if a path matches include patterns and doesn't match exclude patterns."""
    import fnmatch

    path_str = str(path)

    # Check exclude patterns first
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(path_str, pattern):
            return False

    # Check include patterns
    if not include_patterns:
        return True

    for pattern in include_patterns:
        if fnmatch.fnmatch(path_str, pattern):
            return True

    return False


def create_optimized_watcher(
    config: Optional[WatcherConfig] = None,
    event_callback: Optional[Callable[[List[FileEvent]], None]] = None
) -> OptimizedFileWatcher:
    """Factory function to create an optimized file watcher."""
    return OptimizedFileWatcher(config, event_callback)