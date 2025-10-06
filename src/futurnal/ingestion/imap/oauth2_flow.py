"""OAuth2 authentication flow for IMAP connectors.

This module provides OAuth2 authentication flows for common email providers
(Gmail, Office 365) with browser-based consent and token management.
"""

from __future__ import annotations

import json
import logging
import secrets
import socket
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import requests

logger = logging.getLogger(__name__)


@dataclass
class OAuth2Config:
    """OAuth2 provider configuration."""

    client_id: str
    client_secret: str
    auth_url: str
    token_url: str
    scopes: list[str]
    redirect_uri: str = "http://localhost:8080/oauth2callback"


# Gmail OAuth2 configuration (requires user to provide their own client ID/secret)
GMAIL_CONFIG_TEMPLATE = OAuth2Config(
    client_id="",  # User must provide
    client_secret="",  # User must provide
    auth_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    scopes=[
        "https://mail.google.com/",  # Full Gmail access
    ],
)

# Office 365 OAuth2 configuration
OFFICE365_CONFIG_TEMPLATE = OAuth2Config(
    client_id="",  # User must provide
    client_secret="",  # User must provide
    auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
    token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
    scopes=[
        "https://outlook.office365.com/IMAP.AccessAsUser.All",
        "offline_access",
    ],
)


class OAuth2CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth2 callback."""

    def do_GET(self):
        """Handle OAuth2 callback GET request."""
        # Parse query parameters
        query = urlparse(self.path).query
        params = parse_qs(query)

        # Store authorization code
        if 'code' in params:
            self.server.auth_code = params['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <head><title>Authentication Successful</title></head>
                <body>
                    <h1>Authentication Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            """)
        elif 'error' in params:
            self.server.auth_error = params['error'][0]
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f"""
                <html>
                <head><title>Authentication Failed</title></head>
                <body>
                    <h1>Authentication Failed</h1>
                    <p>Error: {params['error'][0]}</p>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            """.encode())
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress HTTP server logging."""
        pass


class OAuth2Flow:
    """OAuth2 authentication flow manager."""

    def __init__(self, config: OAuth2Config):
        """Initialize OAuth2 flow.

        Args:
            config: OAuth2 provider configuration
        """
        self.config = config

    def get_authorization_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """Generate authorization URL for user consent.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Tuple of (authorization_url, state)
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        params = {
            'client_id': self.config.client_id,
            'redirect_uri': self.config.redirect_uri,
            'response_type': 'code',
            'scope': ' '.join(self.config.scopes),
            'state': state,
            'access_type': 'offline',  # Request refresh token
            'prompt': 'consent',  # Force consent screen to get refresh token
        }

        auth_url = f"{self.config.auth_url}?{urlencode(params)}"
        return auth_url, state

    def exchange_code_for_tokens(self, auth_code: str) -> dict:
        """Exchange authorization code for access and refresh tokens.

        Args:
            auth_code: Authorization code from callback

        Returns:
            Dictionary containing access_token, refresh_token, expires_in, etc.

        Raises:
            RuntimeError: If token exchange fails
        """
        data = {
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'code': auth_code,
            'redirect_uri': self.config.redirect_uri,
            'grant_type': 'authorization_code',
        }

        try:
            response = requests.post(self.config.token_url, data=data, timeout=30)
            response.raise_for_status()
            tokens = response.json()

            # Add expiry timestamp
            if 'expires_in' in tokens:
                tokens['expires_at'] = (
                    datetime.utcnow() + timedelta(seconds=tokens['expires_in'])
                ).isoformat()

            return tokens
        except requests.RequestException as e:
            logger.error(f"Token exchange failed: {e}")
            raise RuntimeError(f"Failed to exchange authorization code: {e}")

    def refresh_access_token(self, refresh_token: str) -> dict:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            Dictionary containing new access_token and expires_in

        Raises:
            RuntimeError: If token refresh fails
        """
        data = {
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token',
        }

        try:
            response = requests.post(self.config.token_url, data=data, timeout=30)
            response.raise_for_status()
            tokens = response.json()

            # Add expiry timestamp
            if 'expires_in' in tokens:
                tokens['expires_at'] = (
                    datetime.utcnow() + timedelta(seconds=tokens['expires_in'])
                ).isoformat()

            return tokens
        except requests.RequestException as e:
            logger.error(f"Token refresh failed: {e}")
            raise RuntimeError(f"Failed to refresh access token: {e}")

    def run_local_server_flow(self, port: int = 8080) -> dict:
        """Run local server to handle OAuth2 callback.

        Opens browser for user consent and waits for callback.

        Args:
            port: Local server port (default: 8080)

        Returns:
            Dictionary containing access_token, refresh_token, etc.

        Raises:
            RuntimeError: If authentication fails
        """
        # Generate authorization URL
        auth_url, state = self.get_authorization_url()

        # Start local server
        server = HTTPServer(('localhost', port), OAuth2CallbackHandler)
        server.auth_code = None
        server.auth_error = None
        server.timeout = 300  # 5 minute timeout

        print(f"\nOpening browser for authentication...")
        print(f"If browser doesn't open, visit: {auth_url}\n")

        # Open browser
        webbrowser.open(auth_url)

        # Wait for callback
        while server.auth_code is None and server.auth_error is None:
            server.handle_request()

        # Check for errors
        if server.auth_error:
            raise RuntimeError(f"OAuth2 authentication failed: {server.auth_error}")

        if not server.auth_code:
            raise RuntimeError("No authorization code received")

        # Exchange code for tokens
        tokens = self.exchange_code_for_tokens(server.auth_code)

        return tokens


def get_provider_config(provider: str, client_id: str, client_secret: str) -> OAuth2Config:
    """Get OAuth2 config for provider.

    Args:
        provider: Provider name (gmail, office365)
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret

    Returns:
        OAuth2Config for the provider

    Raises:
        ValueError: If provider not supported
    """
    provider = provider.lower()

    if provider == "gmail":
        config = GMAIL_CONFIG_TEMPLATE
    elif provider in ("office365", "outlook"):
        config = OFFICE365_CONFIG_TEMPLATE
    else:
        raise ValueError(f"Unsupported OAuth2 provider: {provider}")

    # Set credentials
    config.client_id = client_id
    config.client_secret = client_secret

    return config


__all__ = [
    "OAuth2Config",
    "OAuth2Flow",
    "get_provider_config",
    "GMAIL_CONFIG_TEMPLATE",
    "OFFICE365_CONFIG_TEMPLATE",
]
