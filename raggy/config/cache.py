"""Cache management for dependencies and other temporary data."""

import json
from pathlib import Path
from typing import Any, Dict


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
            with open(cache_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            pass
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
    except (PermissionError, IOError):
        pass  # Silently fail if we can't write cache