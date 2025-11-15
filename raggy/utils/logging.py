"""Logging utility functions for consistent error and warning handling."""

from pathlib import Path
from typing import Optional

from .security import sanitize_error_message


def log_error(message: str, error: Optional[Exception] = None, *, quiet: bool = False) -> None:
    """Centralized error logging with consistent formatting.

    Args:
        message: The error message to log
        error: Optional exception to include in the message
        quiet: If True, suppress output

    """
    if quiet:
        return

    if error:
        sanitized_error = sanitize_error_message(str(error))
        print(f"ERROR: {message}: {sanitized_error}")
    else:
        print(f"ERROR: {message}")


def log_warning(message: str, error: Optional[Exception] = None, *, quiet: bool = False) -> None:
    """Centralized warning logging with consistent formatting.

    Args:
        message: The warning message to log
        error: Optional exception to include in the message
        quiet: If True, suppress output

    """
    if quiet:
        return

    if error:
        sanitized_error = sanitize_error_message(str(error))
        print(f"Warning: {message}: {sanitized_error}")
    else:
        print(f"Warning: {message}")


def handle_file_error(file_path: Path, operation: str, error: Exception, *, quiet: bool = False) -> None:
    """Standardized file operation error handling.

    Args:
        file_path: The path to the file that caused the error
        operation: The operation being performed (e.g., 'read', 'write')
        error: The exception that occurred
        quiet: If True, suppress output

    """
    if isinstance(error, (FileNotFoundError, PermissionError)):
        log_error(f"Cannot {operation} {file_path.name} - {type(error).__name__}", quiet=quiet)
    elif isinstance(error, UnicodeDecodeError):
        log_error(f"Cannot {operation} {file_path.name} - encoding issue", quiet=quiet)
    else:
        log_error(f"Cannot {operation} {file_path.name}", error, quiet=quiet)
