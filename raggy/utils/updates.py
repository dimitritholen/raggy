"""Update checking utilities for version management."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Version information
__version__ = "2.0.0"

# Constants
SESSION_CACHE_HOURS = 24  # Hours before update check
UPDATE_TIMEOUT_SECONDS = 2  # API timeout for update checks


def check_for_updates(
    quiet: bool = False, config: Optional[Dict[str, Any]] = None
) -> None:
    """Check GitHub for latest version once per session (non-intrusive).

    Args:
        quiet: If True, suppress output
        config: Optional configuration dictionary with update settings
    """
    if quiet:
        return

    # Load configuration for update settings
    if config is None:
        config = {}

    updates_config = config.get("updates", {})
    if not updates_config.get("check_enabled", True):
        return

    # Use configured repo or default placeholder
    github_repo = updates_config.get("github_repo", "dimitritholen/raggy")

    # Session tracking to avoid frequent checks
    session_file = Path.home() / ".raggy_session"

    # Check if already checked in last 24 hours
    if session_file.exists():
        try:
            cache_age = time.time() - session_file.stat().st_mtime
            if cache_age < SESSION_CACHE_HOURS * 3600:  # 24 hours
                return
        except (OSError, AttributeError):
            pass  # If we can't check file time, proceed with check

    try:
        # Import urllib only when needed to avoid startup cost
        import urllib.request
        import urllib.error

        # Quick timeout to not delay startup
        api_url = f"https://api.github.com/repos/{github_repo}/releases/latest"

        with urllib.request.urlopen(api_url, timeout=UPDATE_TIMEOUT_SECONDS) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                latest_version = data.get("tag_name", "").lstrip("v")

                if latest_version and latest_version != __version__:
                    # Use HTML URL from response or construct fallback
                    github_url = data.get("html_url")
                    if not github_url:
                        base_url = f"https://github.com/{github_repo}"
                        github_url = f"{base_url}/releases/latest"

                    print(f"📦 Raggy update available: v{latest_version} → {github_url}")

        # Update session file to mark check as done
        try:
            session_file.touch()
        except (OSError, PermissionError):
            pass  # If we can't create session file, just skip tracking

    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        ConnectionError,
        TimeoutError,
        Exception
    ):
        # Silently fail - don't interrupt user workflow with network issues
        # This includes any import errors, network timeouts, or API issues
        pass