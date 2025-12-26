"""Infrastructure CLI commands for managing Futurnal services.

Handles automatic startup and management of:
- Neo4j database (via Docker)
- Orchestrator daemon
- Other required services

This module is called by the desktop app on startup to ensure
all services are running.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from typer import Typer, Option

logger = logging.getLogger(__name__)

infrastructure_app = Typer(help="Infrastructure management commands")


def _suppress_logging_for_json():
    """Suppress all logging output for clean JSON responses."""
    # Suppress all loggers to prevent any output before JSON
    logging.disable(logging.CRITICAL)
    # Also suppress warnings
    import warnings
    warnings.filterwarnings("ignore")

# Default Neo4j Docker configuration
NEO4J_CONTAINER_NAME = "futurnal-neo4j"
NEO4J_IMAGE = "neo4j:5.15-community"
NEO4J_HTTP_PORT = 7474
NEO4J_BOLT_PORT = 7687
NEO4J_DEFAULT_PASSWORD = "futurnal_local"


def _is_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _is_neo4j_container_running() -> bool:
    """Check if the Neo4j container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={NEO4J_CONTAINER_NAME}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return NEO4J_CONTAINER_NAME in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _is_neo4j_container_exists() -> bool:
    """Check if the Neo4j container exists (running or stopped)."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={NEO4J_CONTAINER_NAME}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return NEO4J_CONTAINER_NAME in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _start_neo4j_container(data_path: Path) -> bool:
    """Start the Neo4j Docker container."""
    # Ensure data directory exists
    data_path.mkdir(parents=True, exist_ok=True)

    if _is_neo4j_container_exists():
        # Start existing container
        logger.info(f"Starting existing Neo4j container: {NEO4J_CONTAINER_NAME}")
        result = subprocess.run(
            ["docker", "start", NEO4J_CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(f"Failed to start Neo4j: {result.stderr}")
            return False
    else:
        # Create and start new container
        logger.info(f"Creating Neo4j container: {NEO4J_CONTAINER_NAME}")
        cmd = [
            "docker", "run",
            "-d",  # Detached
            "--name", NEO4J_CONTAINER_NAME,
            "-p", f"{NEO4J_HTTP_PORT}:7474",
            "-p", f"{NEO4J_BOLT_PORT}:7687",
            "-v", f"{data_path}/data:/data",
            "-v", f"{data_path}/logs:/logs",
            "-e", f"NEO4J_AUTH=neo4j/{NEO4J_DEFAULT_PASSWORD}",
            "-e", "NEO4J_PLUGINS=[\"apoc\"]",
            "-e", "NEO4J_dbms_security_procedures_unrestricted=apoc.*",
            "--restart", "unless-stopped",
            NEO4J_IMAGE,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"Failed to create Neo4j container: {result.stderr}")
            return False

    return True


def _wait_for_neo4j_ready(timeout: int = 60) -> bool:
    """Wait for Neo4j to be ready to accept connections."""
    import urllib.request
    import urllib.error

    start_time = time.time()
    url = f"http://localhost:{NEO4J_HTTP_PORT}"

    while time.time() - start_time < timeout:
        try:
            response = urllib.request.urlopen(url, timeout=5)
            if response.status == 200:
                logger.info("Neo4j is ready")
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(1)

    logger.warning(f"Neo4j not ready after {timeout}s")
    return False


def _get_workspace_path() -> Path:
    """Get the workspace path."""
    home = Path.home()
    return home / ".futurnal" / "workspace"


def _get_orchestrator_status() -> dict:
    """Get orchestrator daemon status."""
    try:
        from futurnal.orchestrator.daemon import OrchestratorDaemon

        workspace = _get_workspace_path()
        pid_file = workspace / "orchestrator.pid"

        if not pid_file.exists():
            return {"running": False, "pid": None}

        pid = int(pid_file.read_text().strip())

        # Check if process is running
        try:
            os.kill(pid, 0)
            return {"running": True, "pid": pid}
        except OSError:
            return {"running": False, "pid": None, "stale_pid": True}
    except Exception as e:
        logger.warning(f"Error getting orchestrator status: {e}")
        return {"running": False, "pid": None, "error": str(e)}


def _start_orchestrator() -> bool:
    """Start the orchestrator daemon."""
    try:
        workspace = _get_workspace_path()
        workspace.mkdir(parents=True, exist_ok=True)

        # Use subprocess to start orchestrator in background
        python_path = shutil.which("python") or shutil.which("python3")
        if not python_path:
            logger.error("Python not found")
            return False

        cmd = [python_path, "-m", "futurnal.cli", "orchestrator", "start", "--workspace", str(workspace)]

        # Start detached
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait a moment for it to start
        time.sleep(1)

        status = _get_orchestrator_status()
        return status.get("running", False)
    except Exception as e:
        logger.error(f"Failed to start orchestrator: {e}")
        return False


@infrastructure_app.command("status")
def get_status(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get status of all infrastructure services.

    Examples:
        futurnal infrastructure status
        futurnal infrastructure status --json
    """
    docker_available = _is_docker_available()
    neo4j_running = _is_neo4j_container_running() if docker_available else False
    orchestrator_status = _get_orchestrator_status()

    status = {
        "success": True,
        "services": {
            "docker": {
                "available": docker_available,
                "status": "available" if docker_available else "not_installed",
            },
            "neo4j": {
                "running": neo4j_running,
                "status": "running" if neo4j_running else "stopped",
                "port": NEO4J_BOLT_PORT if neo4j_running else None,
            },
            "orchestrator": {
                "running": orchestrator_status.get("running", False),
                "status": "running" if orchestrator_status.get("running") else "stopped",
                "pid": orchestrator_status.get("pid"),
            },
        },
        "allHealthy": docker_available and neo4j_running and orchestrator_status.get("running", False),
    }

    if output_json:
        print(json.dumps(status))
    else:
        print("\nInfrastructure Status")
        print("-" * 40)
        print(f"Docker:       {'Available' if docker_available else 'Not installed'}")
        print(f"Neo4j:        {'Running' if neo4j_running else 'Stopped'}")
        print(f"Orchestrator: {'Running (PID: ' + str(orchestrator_status.get('pid')) + ')' if orchestrator_status.get('running') else 'Stopped'}")
        print("-" * 40)
        print(f"All Healthy:  {'Yes' if status['allHealthy'] else 'No'}")


@infrastructure_app.command("start")
def start_all(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Start all infrastructure services.

    This command:
    1. Starts Neo4j Docker container (creates if needed)
    2. Waits for Neo4j to be ready
    3. Starts the orchestrator daemon

    Examples:
        futurnal infrastructure start
        futurnal infrastructure start --json
    """
    results = {
        "success": True,
        "services": {},
        "errors": [],
    }

    workspace = _get_workspace_path()
    neo4j_data_path = workspace / "neo4j"

    # 1. Check Docker
    if not _is_docker_available():
        results["services"]["docker"] = {"started": False, "error": "Docker not available"}
        results["errors"].append("Docker is not installed or not running. Please install Docker Desktop.")
        results["success"] = False
    else:
        results["services"]["docker"] = {"available": True}

        # 2. Start Neo4j
        if _is_neo4j_container_running():
            results["services"]["neo4j"] = {"started": False, "alreadyRunning": True}
            logger.info("Neo4j already running")
        else:
            logger.info("Starting Neo4j...")
            if _start_neo4j_container(neo4j_data_path):
                # Wait for Neo4j to be ready
                if _wait_for_neo4j_ready(timeout=60):
                    results["services"]["neo4j"] = {"started": True, "port": NEO4J_BOLT_PORT}
                else:
                    results["services"]["neo4j"] = {"started": True, "ready": False, "error": "Timeout waiting for Neo4j"}
                    results["errors"].append("Neo4j started but not ready within timeout")
            else:
                results["services"]["neo4j"] = {"started": False, "error": "Failed to start container"}
                results["errors"].append("Failed to start Neo4j container")
                results["success"] = False

    # 3. Start Orchestrator
    orchestrator_status = _get_orchestrator_status()
    if orchestrator_status.get("running"):
        results["services"]["orchestrator"] = {"started": False, "alreadyRunning": True, "pid": orchestrator_status.get("pid")}
        logger.info(f"Orchestrator already running (PID: {orchestrator_status.get('pid')})")
    else:
        logger.info("Starting orchestrator...")
        if _start_orchestrator():
            new_status = _get_orchestrator_status()
            results["services"]["orchestrator"] = {"started": True, "pid": new_status.get("pid")}
        else:
            results["services"]["orchestrator"] = {"started": False, "error": "Failed to start"}
            results["errors"].append("Failed to start orchestrator daemon")
            # Don't fail overall - orchestrator is less critical

    if output_json:
        print(json.dumps(results))
    else:
        print("\nStarting Infrastructure Services")
        print("-" * 40)

        for service, info in results["services"].items():
            if info.get("alreadyRunning"):
                status = "Already running"
            elif info.get("started"):
                status = "Started"
            elif info.get("available"):
                status = "Available"
            else:
                status = f"Failed: {info.get('error', 'Unknown error')}"
            print(f"{service.capitalize():15} {status}")

        if results["errors"]:
            print("\nErrors:")
            for error in results["errors"]:
                print(f"  - {error}")

        print("-" * 40)
        print(f"Overall: {'Success' if results['success'] else 'Failed'}")


@infrastructure_app.command("stop")
def stop_all(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Stop all infrastructure services.

    Examples:
        futurnal infrastructure stop
        futurnal infrastructure stop --json
    """
    results = {
        "success": True,
        "services": {},
    }

    # Stop orchestrator
    orchestrator_status = _get_orchestrator_status()
    if orchestrator_status.get("running"):
        pid = orchestrator_status.get("pid")
        try:
            os.kill(pid, 15)  # SIGTERM
            results["services"]["orchestrator"] = {"stopped": True, "pid": pid}
        except Exception as e:
            results["services"]["orchestrator"] = {"stopped": False, "error": str(e)}
    else:
        results["services"]["orchestrator"] = {"stopped": False, "wasNotRunning": True}

    # Stop Neo4j
    if _is_neo4j_container_running():
        try:
            subprocess.run(
                ["docker", "stop", NEO4J_CONTAINER_NAME],
                capture_output=True,
                timeout=30,
            )
            results["services"]["neo4j"] = {"stopped": True}
        except Exception as e:
            results["services"]["neo4j"] = {"stopped": False, "error": str(e)}
    else:
        results["services"]["neo4j"] = {"stopped": False, "wasNotRunning": True}

    if output_json:
        print(json.dumps(results))
    else:
        print("\nStopping Infrastructure Services")
        print("-" * 40)
        for service, info in results["services"].items():
            if info.get("wasNotRunning"):
                status = "Was not running"
            elif info.get("stopped"):
                status = "Stopped"
            else:
                status = f"Failed: {info.get('error', 'Unknown')}"
            print(f"{service.capitalize():15} {status}")
