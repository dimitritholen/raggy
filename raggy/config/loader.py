"""Configuration loading and management."""

from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logging import log_warning
from .constants import DEFAULT_CONFIG


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load optional configuration file.

    Args:
        config_path: Optional path to configuration file (defaults to raggy_config.yaml)

    Returns:
        Dict[str, Any]: Merged configuration dictionary
    """
    default_config = DEFAULT_CONFIG.copy()

    # Try to load config file
    config_file = Path(config_path or "raggy_config.yaml")
    if config_file.exists():
        try:
            import yaml

            with open(config_file, "r") as f:
                user_config = yaml.safe_load(f)

            # Merge with defaults
            _merge_configs(default_config, user_config)
        except ImportError:
            log_warning("PyYAML not installed, using default config", quiet=False)
        except (FileNotFoundError, PermissionError, OSError) as e:
            log_warning(f"Could not access config file {config_file}", e, quiet=False)
        except (AttributeError, TypeError, ValueError) as e:
            # Handle YAML parsing errors - yaml.YAMLError inherits from Exception
            # but we catch common parsing issues (invalid structure, types, values)
            log_warning(f"Invalid YAML format in {config_file}", e, quiet=False)

    return default_config


def _merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> None:
    """Recursively merge user config into default config.

    Args:
        default: Default configuration dictionary (modified in place)
        user: User configuration dictionary to merge
    """
    for key, value in user.items():
        if (
            key in default
            and isinstance(default[key], dict)
            and isinstance(value, dict)
        ):
            _merge_configs(default[key], value)
        else:
            default[key] = value