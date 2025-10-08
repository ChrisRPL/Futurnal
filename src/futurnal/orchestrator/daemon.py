"""Orchestrator daemon process management.

Provides PID file management, process status checking, and graceful/forced
shutdown capabilities for the ingestion orchestrator daemon.
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class DaemonError(Exception):
    """Base exception for daemon management errors."""


class AlreadyRunningError(DaemonError):
    """Raised when attempting to start an already-running daemon."""


class NotRunningError(DaemonError):
    """Raised when attempting to stop a daemon that isn't running."""


@dataclass
class DaemonStatus:
    """Status information for orchestrator daemon."""

    running: bool
    pid: Optional[int] = None
    stale_pid_file: bool = False

    def __str__(self) -> str:
        if self.running:
            return f"Running (PID: {self.pid})"
        elif self.stale_pid_file:
            return "Not running (stale PID file detected)"
        else:
            return "Not running"


class PIDFileManager:
    """Manages PID file for orchestrator daemon process.

    The PID file is used to track whether the orchestrator is running
    and to coordinate shutdown across CLI commands and the daemon itself.
    """

    def __init__(self, pid_file_path: Path) -> None:
        """Initialize PID file manager.

        Args:
            pid_file_path: Path to PID file (e.g., ~/.futurnal/orchestrator.pid)
        """
        self._path = pid_file_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, pid: Optional[int] = None) -> None:
        """Write current process ID to PID file.

        Args:
            pid: Process ID to write (defaults to current process)

        Raises:
            AlreadyRunningError: If daemon is already running
        """
        # Check if already running
        status = self.status()
        if status.running:
            raise AlreadyRunningError(
                f"Orchestrator already running with PID {status.pid}"
            )

        # Clean up stale PID file if present
        if status.stale_pid_file:
            self.remove()

        # Write PID
        pid_to_write = pid or os.getpid()
        self._path.write_text(str(pid_to_write), encoding="utf-8")

    def read(self) -> Optional[int]:
        """Read PID from PID file.

        Returns:
            Process ID if file exists and contains valid PID, None otherwise
        """
        if not self._path.exists():
            return None

        try:
            pid_str = self._path.read_text(encoding="utf-8").strip()
            return int(pid_str)
        except (ValueError, IOError):
            return None

    def remove(self) -> None:
        """Remove PID file."""
        if self._path.exists():
            self._path.unlink()

    def status(self) -> DaemonStatus:
        """Check if daemon is running based on PID file.

        Returns:
            DaemonStatus with running state and PID information
        """
        pid = self.read()

        if pid is None:
            return DaemonStatus(running=False)

        # Check if process is actually running
        if self._is_process_running(pid):
            return DaemonStatus(running=True, pid=pid)
        else:
            # PID file exists but process is not running (stale)
            return DaemonStatus(running=False, pid=pid, stale_pid_file=True)

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check if a process with given PID is running.

        Args:
            pid: Process ID to check

        Returns:
            True if process is running, False otherwise
        """
        try:
            # Send signal 0 (null signal) to check if process exists
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class OrchestratorDaemon:
    """Manages orchestrator daemon lifecycle.

    Provides methods for starting, stopping (graceful and forced),
    and checking status of the orchestrator daemon process.
    """

    def __init__(self, workspace_dir: Path) -> None:
        """Initialize orchestrator daemon manager.

        Args:
            workspace_dir: Workspace directory (e.g., ~/.futurnal)
        """
        self._workspace_dir = workspace_dir
        self._pid_manager = PIDFileManager(workspace_dir / "orchestrator.pid")

    def status(self) -> DaemonStatus:
        """Get current daemon status.

        Returns:
            DaemonStatus with running state and PID information
        """
        return self._pid_manager.status()

    def register_start(self) -> None:
        """Register daemon start by creating PID file.

        Should be called when orchestrator daemon starts.

        Raises:
            AlreadyRunningError: If daemon is already running
        """
        self._pid_manager.write()

    def register_stop(self) -> None:
        """Register daemon stop by removing PID file.

        Should be called when orchestrator daemon stops gracefully.
        """
        self._pid_manager.remove()

    def stop(self, *, force: bool = False, timeout: float = 30.0) -> None:
        """Stop running orchestrator daemon.

        Args:
            force: If True, send SIGKILL immediately. If False, send SIGTERM
                   and wait for graceful shutdown
            timeout: Maximum time to wait for graceful shutdown (seconds)

        Raises:
            NotRunningError: If daemon is not running
        """
        status = self.status()

        if not status.running:
            if status.stale_pid_file:
                # Clean up stale PID file
                self._pid_manager.remove()
                raise NotRunningError(
                    "Orchestrator is not running (cleaned up stale PID file)"
                )
            else:
                raise NotRunningError("Orchestrator is not running")

        pid = status.pid
        assert pid is not None, "Running daemon must have PID"

        if force:
            # Force kill immediately
            self._kill_process(pid, signal.SIGKILL)
            # Wait brief moment for process to die
            time.sleep(0.5)
            # Remove PID file
            self._pid_manager.remove()
        else:
            # Graceful shutdown
            self._kill_process(pid, signal.SIGTERM)

            # Wait for process to exit
            start_time = time.time()
            while time.time() - start_time < timeout:
                if not self._pid_manager._is_process_running(pid):
                    # Process exited cleanly
                    self._pid_manager.remove()
                    return
                time.sleep(0.5)

            # Timeout - force kill
            self._kill_process(pid, signal.SIGKILL)
            time.sleep(0.5)
            self._pid_manager.remove()
            raise DaemonError(
                f"Graceful shutdown timed out after {timeout}s, "
                f"sent SIGKILL to PID {pid}"
            )

    @staticmethod
    def _kill_process(pid: int, sig: signal.Signals) -> None:
        """Send signal to process.

        Args:
            pid: Process ID
            sig: Signal to send

        Raises:
            DaemonError: If unable to send signal
        """
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            # Process already gone
            pass
        except OSError as e:
            raise DaemonError(f"Failed to send signal {sig} to PID {pid}: {e}")
