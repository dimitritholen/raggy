"""Configuration management for Raggy.

This module handles loading and validating .raggy.json configuration files,
with support for environment variable substitution and multiple discovery methods.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional


class RaggyConfig:
    """Raggy configuration manager with support for .raggy.json files."""

    # Default configuration
    DEFAULT_CONFIG = {
        "vectorStore": {
            "provider": "chromadb",
            "chromadb": {
                "path": "./vectordb"
            }
        },
        "embedding": {
            "provider": "sentence-transformers",
            "sentenceTransformers": {
                "model": "all-MiniLM-L6-v2"
            }
        },
        "memory": {
            "categoriesMode": "append",
            "categories": {
                "add": [],
                "remove": [],
                "replace": []
            }
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Optional explicit path to config file.
                If not provided, will attempt discovery in order:
                1. RAGGY_CONFIG_PATH environment variable
                2. .raggy.json in current working directory

        """
        self.config_path = self._discover_config(config_path)
        self.config = self._load_config()

    def _discover_config(self, explicit_path: Optional[str] = None) -> Optional[Path]:
        """Discover configuration file.

        Priority order:
        1. Explicit path argument
        2. RAGGY_CONFIG_PATH environment variable
        3. .raggy.json in current working directory

        Args:
            explicit_path: Optional explicit path to config file

        Returns:
            Path to config file if found, None otherwise

        """
        # 1. Check explicit path argument
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Config file not found: {explicit_path}")

        # 2. Check environment variable
        env_path = os.getenv("RAGGY_CONFIG_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path
            raise FileNotFoundError(
                f"Config file not found at RAGGY_CONFIG_PATH: {env_path}"
            )

        # 3. Check current working directory
        cwd_config = Path.cwd() / ".raggy.json"
        if cwd_config.exists():
            return cwd_config

        # No config found - use defaults
        return None

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration.

        Returns:
            Dict: Merged configuration (defaults + file config)

        """
        # Start with defaults
        config = self._deep_copy(self.DEFAULT_CONFIG)

        # If no config file, return defaults
        if not self.config_path:
            return config

        # Load config file
        try:
            with open(self.config_path, encoding="utf-8") as f:
                file_config = json.load(f)

            # Merge with defaults (file config takes precedence)
            config = self._deep_merge(config, file_config)

            # Substitute environment variables
            return self._substitute_env_vars(config)


        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {self.config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")

    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy a nested dictionary.

        Args:
            obj: Object to copy

        Returns:
            Deep copy of object

        """
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)

        Returns:
            Merged dictionary

        """
        result = self._deep_copy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = self._deep_copy(value)

        return result

    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute ${ENV_VAR} placeholders with environment variables.

        Args:
            obj: Object to process (can be dict, list, str, or other)

        Returns:
            Object with substituted values

        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Match ${VAR_NAME} pattern
            pattern = r'\$\{([^}]+)\}'

            def replace_env_var(match):
                var_name = match.group(1)
                value = os.getenv(var_name)
                if value is None:
                    raise ValueError(
                        f"Environment variable not found: {var_name}. "
                        f"Please set {var_name} or update your .raggy.json config."
                    )
                return value

            return re.sub(pattern, replace_env_var, obj)
        else:
            return obj

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path.

        Args:
            key_path: Dot-separated path (e.g., "vectorStore.provider")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = RaggyConfig()
            >>> config.get("vectorStore.provider")
            'chromadb'
            >>> config.get("vectorStore.pinecone.apiKey")
            None

        """
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration.

        Returns:
            Dict with provider and provider-specific config

        """
        return self.config.get("vectorStore", {})

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration.

        Returns:
            Dict with provider and provider-specific config

        """
        return self.config.get("embedding", {})

    def get_memory_categories(self) -> Dict[str, Any]:
        """Get memory categories configuration.

        Returns:
            Dict with categoriesMode and categories

        """
        return self.config.get("memory", {})

    def get_resolved_categories(self, default_categories: set) -> set:
        """Get resolved memory categories based on configuration mode.

        Args:
            default_categories: Default category set

        Returns:
            Set of resolved categories

        Example:
            >>> config = RaggyConfig()
            >>> defaults = {"decision", "solution", "pattern", "learning", "error", "note"}
            >>> config.get_resolved_categories(defaults)
            {'decision', 'solution', 'pattern', 'learning', 'error', 'note'}

        """
        memory_config = self.get_memory_categories()
        mode = memory_config.get("categoriesMode", "append")
        categories_config = memory_config.get("categories", {})

        if mode == "replace":
            # Use only the replacement categories
            replace_list = categories_config.get("replace", [])
            if not replace_list:
                raise ValueError(
                    "categoriesMode is 'replace' but no replacement categories provided"
                )
            return set(replace_list)

        elif mode == "custom":
            # Use only custom added categories (no defaults)
            add_list = categories_config.get("add", [])
            if not add_list:
                raise ValueError(
                    "categoriesMode is 'custom' but no categories to add provided"
                )
            return set(add_list)

        else:  # mode == "append" (default)
            # Start with defaults
            result = set(default_categories)

            # Add custom categories
            add_list = categories_config.get("add", [])
            result.update(add_list)

            # Remove specified categories
            remove_list = categories_config.get("remove", [])
            result.difference_update(remove_list)

            return result

    def __repr__(self) -> str:
        """String representation of config."""
        config_source = str(self.config_path) if self.config_path else "defaults"
        return f"RaggyConfig(source={config_source})"
