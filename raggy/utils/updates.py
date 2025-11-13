"""Update checking utilities for version management."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .logging import log_warning

# Version information
__version__ = "2.0.0"

# Constants
SESSION_CACHE_HOURS = 24  # Hours before update check
UPDATE_TIMEOUT_SECONDS = 2  # API timeout for update checks


class UpdateChecker:
    """Handles version update checks with session caching."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize update checker.

        Args:
            config: Optional configuration dictionary with update settings
        """
        self.config = config or {}
        self.updates_config = self.config.get("updates", {})
        self.session_file = Path.home() / ".raggy_session"
        self.github_repo = self.updates_config.get("github_repo", "dimitritholen/raggy")

    def check(self, quiet: bool = False) -> None:
        """Check GitHub for latest version once per session.

        Args:
            quiet: If True, suppress output
        """
        if not self._should_check(quiet):
            return

        latest_version = self._fetch_latest_version()
        if latest_version and self._is_newer(latest_version):
            self._display_update_notice(latest_version)

        self._update_session_cache()

    def _should_check(self, quiet: bool) -> bool:
        """Determine if update check should run.

        Args:
            quiet: If True, check should not run

        Returns:
            bool: True if check should proceed
        """
        if quiet:
            return False

        if not self.updates_config.get("check_enabled", True):
            return False

        return not self._is_recently_checked()

    def _is_recently_checked(self) -> bool:
        """Check if update was checked in last 24 hours.

        Returns:
            bool: True if recently checked
        """
        if not self.session_file.exists():
            return False

        try:
            cache_age = time.time() - self.session_file.stat().st_mtime
            return cache_age < SESSION_CACHE_HOURS * 3600
        except (OSError, AttributeError) as e:
            log_warning(
                f"Could not read session file {self.session_file.name}, treating as expired",
                e,
                quiet=True
            )
            return False

    def _fetch_latest_version(self) -> Optional[str]:
        """Fetch latest version from GitHub API.

        Returns:
            Optional[str]: Latest version string or None if fetch fails
        """
        try:
            import urllib.error
            import urllib.request

            api_url = f"https://api.github.com/repos/{self.github_repo}/releases/latest"

            with urllib.request.urlopen(api_url, timeout=UPDATE_TIMEOUT_SECONDS) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    latest_version = data.get("tag_name", "").lstrip("v")
                    if latest_version:
                        self._cached_release_url = data.get("html_url")
                        return latest_version

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            ConnectionError,
            TimeoutError,
            Exception
        ):
            # Silently fail - don't interrupt user workflow
            pass

        return None

    def _is_newer(self, latest_version: str) -> bool:
        """Check if latest version is newer than current.

        Args:
            latest_version: Version string to compare

        Returns:
            bool: True if latest version is different from current
        """
        return latest_version != __version__

    def _display_update_notice(self, latest_version: str) -> None:
        """Display update notification to user.

        Args:
            latest_version: Version string to display
        """
        github_url = getattr(self, '_cached_release_url', None)
        if not github_url:
            base_url = f"https://github.com/{self.github_repo}"
            github_url = f"{base_url}/releases/latest"

        print(f"📦 Raggy update available: v{latest_version} → {github_url}")

    def _update_session_cache(self) -> None:
        """Update session file to mark check as done."""
        try:
            self.session_file.touch()
        except (OSError, PermissionError) as e:
            log_warning(
                f"Could not create session file {self.session_file.name}, update check will run again on next startup",
                e,
                quiet=True
            )


def check_for_updates(
    quiet: bool = False, config: Optional[Dict[str, Any]] = None
) -> None:
    """Check GitHub for latest version once per session (non-intrusive).

    Args:
        quiet: If True, suppress output
        config: Optional configuration dictionary with update settings
    """
    checker = UpdateChecker(config)
    checker.check(quiet)
