"""Cache management for dependencies and other temporary data."""

import json
from pathlib import Path
from typing import Any, Dict

from ..utils.logging import log_warning


def get_cache_file() -> Path:
    """Get path for dependency cache file.

    Returns:
        Path: Path to the cache file
    """
    return Path.cwd() / ".raggy_deps_cache.json"


def load_deps_cache() -> Dict[str, Any]:
    """Load dependency cache from file.

    Returns:
        Dict[str, Any]: Cached dependency information or empty dict if not found
    """
    cache_file = get_cache_file()
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            # Cache loading is optional - use empty cache if unavailable
            log_warning(
                f"Could not load dependency cache from {cache_file.name}, using empty cache",
                e,
                quiet=True  # Debug-level issue, don't show to users
            )
    return {}


def save_deps_cache(cache: Dict[str, Any]) -> None:
    """Save dependency cache to file.

    Args:
        cache: Cache dictionary to save
    """
    cache_file = get_cache_file()
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f)
    except (OSError, PermissionError) as e:
        # Cache saving is optional - continue without cache if write fails
        log_warning(
            f"Could not save dependency cache to {cache_file.name}, cache will not persist",
            e,
            quiet=True  # Debug-level issue, don't show to users
        )
