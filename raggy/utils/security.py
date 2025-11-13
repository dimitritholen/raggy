"""Security utility functions for path validation and error sanitization."""

import re
from pathlib import Path
from typing import Optional

# Pre-compiled regex patterns for security scanning
WINDOWS_PATH_PATTERN = re.compile(r'[A-Za-z]:[\\\/][^\\\/\s]*[\\\/]')
UNIX_PATH_PATTERN = re.compile(r'\/[^\/\s]*\/')
FILE_URL_PATTERN = re.compile(r'\bfile:\/\/[^\s]*')


def validate_path(file_path: Path, base_path: Optional[Path] = None) -> bool:
    """Validate file path to prevent directory traversal attacks.

    Args:
        file_path: The path to validate
        base_path: The base directory to check against (defaults to current working directory)

    Returns:
        bool: True if the path is safe (within base directory), False otherwise
    """
    try:
        # Resolve the path to get absolute path
        resolved_path = file_path.resolve()

        if base_path is None:
            base_path = Path.cwd()
        else:
            base_path = base_path.resolve()

        # Check if the resolved path is within the base directory
        try:
            resolved_path.relative_to(base_path)
            return True
        except ValueError:
            # Path is outside the base directory
            return False
    except (OSError, ValueError):
        return False


def sanitize_error_message(error_msg: str) -> str:
    """Sanitize error messages to prevent information leakage.

    Args:
        error_msg: The error message to sanitize

    Returns:
        str: Sanitized error message with sensitive paths removed
    """
    # Remove potentially sensitive path information using pre-compiled patterns
    sanitized = WINDOWS_PATH_PATTERN.sub('', error_msg)  # Windows paths
    sanitized = UNIX_PATH_PATTERN.sub('/', sanitized)  # Unix paths
    sanitized = FILE_URL_PATTERN.sub('[FILE_PATH]', sanitized)
    return sanitized