"""OAuth Device Flow implementation for GitHub authentication.

This module implements the OAuth Device Flow as the recommended authentication
method for CLI applications, providing a secure and user-friendly way to
authorize GitHub access without requiring a browser-based redirect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

import requests
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DeviceFlowStatus(str, Enum):
    """Status codes for device flow polling."""

    AUTHORIZATION_PENDING = "authorization_pending"
    SLOW_DOWN = "slow_down"
    EXPIRED_TOKEN = "expired_token"
    ACCESS_DENIED = "access_denied"
    SUCCESS = "success"
    ERROR = "error"


class DeviceCodeResponse(BaseModel):
    """Response from device code request."""

    device_code: str = Field(..., description="Device verification code")
    user_code: str = Field(..., description="User verification code to display")
    verification_uri: str = Field(..., description="URL for user to visit")
    expires_in: int = Field(..., description="Expiration time in seconds")
    interval: int = Field(default=5, description="Polling interval in seconds")


class DeviceFlowResult(BaseModel):
    """Result from device flow completion."""

    access_token: str = Field(..., description="GitHub access token")
    token_type: str = Field(default="bearer", description="Token type")
    scope: str = Field(default="", description="Granted scopes")


# ---------------------------------------------------------------------------
# GitHub OAuth Device Flow
# ---------------------------------------------------------------------------


@dataclass
class GitHubOAuthDeviceFlow:
    """OAuth Device Flow handler for GitHub authentication."""

    client_id: str
    github_host: str = "github.com"
    scopes: Optional[list[str]] = None
    timeout: int = 900  # 15 minutes
    max_poll_attempts: int = 180

    def __post_init__(self) -> None:
        """Initialize OAuth endpoints."""
        if self.github_host == "github.com":
            self.device_code_url = "https://github.com/login/device/code"
            self.access_token_url = "https://github.com/login/oauth/access_token"
            self.verification_url_base = "https://github.com/login/device"
        else:
            # GitHub Enterprise Server
            self.device_code_url = f"https://{self.github_host}/login/device/code"
            self.access_token_url = (
                f"https://{self.github_host}/login/oauth/access_token"
            )
            self.verification_url_base = f"https://{self.github_host}/login/device"

        if self.scopes is None:
            # Default to repo scope for private repository access
            self.scopes = ["repo"]

    def initiate_device_flow(self) -> DeviceCodeResponse:
        """Initiate the device authorization flow.

        Returns:
            DeviceCodeResponse with user code and verification URL

        Raises:
            RuntimeError: If device code request fails
        """
        scope_str = " ".join(self.scopes) if self.scopes else ""
        data = {"client_id": self.client_id, "scope": scope_str}

        response = requests.post(
            self.device_code_url,
            data=data,
            headers={"Accept": "application/json"},
            timeout=15,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Device code request failed: {response.status_code} - {response.text}"
            )

        result = response.json()

        return DeviceCodeResponse(
            device_code=result["device_code"],
            user_code=result["user_code"],
            verification_uri=result.get("verification_uri", self.verification_url_base),
            expires_in=result.get("expires_in", 900),
            interval=result.get("interval", 5),
        )

    def poll_for_token(
        self,
        device_code: str,
        interval: int = 5,
        *,
        on_pending: Optional[callable] = None,
    ) -> DeviceFlowResult:
        """Poll for access token after user authorization.

        Args:
            device_code: Device code from initiate_device_flow
            interval: Initial polling interval in seconds
            on_pending: Optional callback for pending status

        Returns:
            DeviceFlowResult with access token

        Raises:
            RuntimeError: If polling fails, expires, or is denied
        """
        current_interval = interval
        expiration = datetime.utcnow() + timedelta(seconds=self.timeout)
        attempts = 0

        while datetime.utcnow() < expiration and attempts < self.max_poll_attempts:
            attempts += 1
            time.sleep(current_interval)

            try:
                result = self._check_authorization(device_code)

                if result["status"] == DeviceFlowStatus.SUCCESS:
                    return DeviceFlowResult(
                        access_token=result["access_token"],
                        token_type=result.get("token_type", "bearer"),
                        scope=result.get("scope", ""),
                    )

                elif result["status"] == DeviceFlowStatus.AUTHORIZATION_PENDING:
                    if on_pending:
                        on_pending(attempts, current_interval)
                    continue

                elif result["status"] == DeviceFlowStatus.SLOW_DOWN:
                    # Increase interval by 5 seconds as per GitHub spec
                    current_interval += 5
                    continue

                elif result["status"] == DeviceFlowStatus.EXPIRED_TOKEN:
                    raise RuntimeError("Device code expired. Please restart authorization.")

                elif result["status"] == DeviceFlowStatus.ACCESS_DENIED:
                    raise RuntimeError("User denied authorization request.")

                else:
                    raise RuntimeError(
                        f"Unexpected status: {result.get('status', 'unknown')}"
                    )

            except requests.RequestException as e:
                # Network error, continue polling
                if on_pending:
                    on_pending(attempts, current_interval)
                continue

        raise RuntimeError("Authorization timed out. Please restart the process.")

    def _check_authorization(self, device_code: str) -> Dict[str, Any]:
        """Check authorization status."""
        data = {
            "client_id": self.client_id,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }

        response = requests.post(
            self.access_token_url,
            data=data,
            headers={"Accept": "application/json"},
            timeout=15,
        )

        if response.status_code != 200:
            return {"status": DeviceFlowStatus.ERROR, "error": response.text}

        result = response.json()

        # Check for errors
        if "error" in result:
            error = result["error"]
            if error == "authorization_pending":
                return {"status": DeviceFlowStatus.AUTHORIZATION_PENDING}
            elif error == "slow_down":
                return {"status": DeviceFlowStatus.SLOW_DOWN}
            elif error == "expired_token":
                return {"status": DeviceFlowStatus.EXPIRED_TOKEN}
            elif error == "access_denied":
                return {"status": DeviceFlowStatus.ACCESS_DENIED}
            else:
                return {"status": DeviceFlowStatus.ERROR, "error": error}

        # Success
        return {
            "status": DeviceFlowStatus.SUCCESS,
            "access_token": result["access_token"],
            "token_type": result.get("token_type", "bearer"),
            "scope": result.get("scope", ""),
        }

    def run_flow(self, *, on_user_code: Optional[callable] = None, on_pending: Optional[callable] = None) -> DeviceFlowResult:
        """Run complete device flow from start to finish.

        Args:
            on_user_code: Callback(user_code, verification_uri) when code is ready
            on_pending: Callback(attempt, interval) during polling

        Returns:
            DeviceFlowResult with access token

        Raises:
            RuntimeError: If flow fails
        """
        # Step 1: Get device code
        device_response = self.initiate_device_flow()

        # Step 2: Display user code
        if on_user_code:
            on_user_code(device_response.user_code, device_response.verification_uri)

        # Step 3: Poll for token
        return self.poll_for_token(
            device_response.device_code,
            interval=device_response.interval,
            on_pending=on_pending,
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def start_github_oauth_flow(
    client_id: str,
    scopes: Optional[list[str]] = None,
    github_host: str = "github.com",
    *,
    on_user_code: Optional[callable] = None,
    on_pending: Optional[callable] = None,
) -> DeviceFlowResult:
    """Start GitHub OAuth Device Flow (convenience function).

    Args:
        client_id: GitHub OAuth App client ID
        scopes: List of OAuth scopes to request
        github_host: GitHub hostname (default: github.com)
        on_user_code: Callback(user_code, verification_uri) when code is ready
        on_pending: Callback(attempt, interval) during polling

    Returns:
        DeviceFlowResult with access token

    Example:
        ```python
        def show_user_code(code, url):
            print(f"Go to {url} and enter code: {code}")

        def show_pending(attempt, interval):
            print(f"Waiting for authorization... (attempt {attempt})")

        result = start_github_oauth_flow(
            client_id="Iv1.abc123",
            scopes=["repo"],
            on_user_code=show_user_code,
            on_pending=show_pending,
        )
        print(f"Access token: {result.access_token}")
        ```
    """
    flow = GitHubOAuthDeviceFlow(
        client_id=client_id, scopes=scopes, github_host=github_host
    )
    return flow.run_flow(on_user_code=on_user_code, on_pending=on_pending)


__all__ = [
    "DeviceCodeResponse",
    "DeviceFlowResult",
    "DeviceFlowStatus",
    "GitHubOAuthDeviceFlow",
    "start_github_oauth_flow",
]
