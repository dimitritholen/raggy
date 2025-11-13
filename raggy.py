#!/usr/bin/env python3
"""Universal ChromaDB RAG Setup Script v2.0.0 - DEPRECATED WRAPPER.

⚠️  DEPRECATION NOTICE:
================================================================================
This monolithic raggy.py file is DEPRECATED and will be removed in v3.0.0.

Please use one of these alternatives:
  - Recommended: python raggy_cli.py [command]
  - As module: python -m raggy [command]
  - Installed: raggy [command]

This file now acts as a thin wrapper for backward compatibility only.
All functionality has been refactored into the modular raggy/ package.
================================================================================

Original features preserved through the raggy package:
• Hybrid Search: Combines semantic + BM25 keyword ranking for exact matches
• Smart Chunking: Markdown-aware chunking preserving document structure
• Normalized Scoring: 0-1 scores with human-readable labels
• Query Processing: Automatic expansion of domain terms
• Model Presets: --model-preset fast/balanced/multilingual/accurate
• Config Support: Optional raggy_config.yaml for customization
• Multilingual: Enhanced Dutch/English mixed content support
• Backward Compatible: All v1.x commands work unchanged
"""

import sys
import warnings

# Show deprecation warning when this file is executed
warnings.warn(
    "\n" + "="*80 + "\n"
    "⚠️  raggy.py is DEPRECATED and will be removed in v3.0.0.\n"
    "Please use 'python raggy_cli.py' or 'python -m raggy' instead.\n"
    "This wrapper exists only for backward compatibility.\n" +
    "="*80,
    DeprecationWarning,
    stacklevel=2
)

# ============================================================================
# IMPORTS FROM REFACTORED RAGGY PACKAGE
# All functionality now lives in the modular raggy/ package
# ============================================================================

# Core functionality
from raggy import (
    UniversalRAG,
    SearchEngine,
    DatabaseManager,
    DocumentProcessor,
    BM25Scorer,
    QueryProcessor,
    CommandFactory,
    __version__,
)

# Configuration and setup
from raggy import (
    load_config,
    setup_environment,
    setup_dependencies,
    install_if_missing,
    check_for_updates,
)

# Scoring and normalization functions
from raggy import (
    normalize_cosine_distance,
    normalize_hybrid_score,
    interpret_score,
)

# Command implementations
from raggy.cli.commands import (
    InitCommand,
    BuildCommand,
    SearchCommand,
    InteractiveCommand,
    StatusCommand,
    OptimizeCommand,
    TestCommand,
    DiagnoseCommand,
    ValidateCommand,
)

# Utility functions
from raggy.utils.logging import (
    log_error,
    log_warning,
    handle_file_error,
)

from raggy.utils.security import (
    validate_path,
    sanitize_error_message,
)

from raggy.utils.symbols import (
    get_symbols,
    SYMBOLS,
)

# Cache utilities
from raggy.config.cache import (
    get_cache_file,
    load_deps_cache,
    save_deps_cache,
)

# Constants - re-export for backward compatibility
from raggy.config.constants import (
    CHUNK_READ_SIZE,
    MAX_CACHE_SIZE,
    CACHE_TTL,
    MAX_FILE_SIZE_MB,
    SESSION_CACHE_HOURS,
    UPDATE_TIMEOUT_SECONDS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_RESULTS,
    DEFAULT_CONTEXT_CHARS,
    DEFAULT_HYBRID_WEIGHT,
    SUPPORTED_EXTENSIONS,
    GLOB_PATTERNS,
    FAST_MODEL,
    DEFAULT_MODEL,
    MULTILINGUAL_MODEL,
    ACCURATE_MODEL,
)

# ============================================================================
# BACKWARD COMPATIBILITY EXPORTS
# Re-export everything for scripts that import from raggy
# ============================================================================

__all__ = [
    # Core classes
    "UniversalRAG",
    "SearchEngine",
    "DatabaseManager",
    "DocumentProcessor",
    "BM25Scorer",
    "QueryProcessor",
    "CommandFactory",

    # Commands
    "InitCommand",
    "BuildCommand",
    "SearchCommand",
    "InteractiveCommand",
    "StatusCommand",
    "OptimizeCommand",
    "TestCommand",
    "DiagnoseCommand",
    "ValidateCommand",

    # Functions
    "load_config",
    "setup_environment",
    "setup_dependencies",
    "install_if_missing",
    "check_for_updates",
    "normalize_cosine_distance",
    "normalize_hybrid_score",
    "interpret_score",
    "log_error",
    "log_warning",
    "handle_file_error",
    "validate_path",
    "sanitize_error_message",
    "get_symbols",
    "get_cache_file",
    "load_deps_cache",
    "save_deps_cache",

    # Constants
    "CHUNK_READ_SIZE",
    "MAX_CACHE_SIZE",
    "CACHE_TTL",
    "MAX_FILE_SIZE_MB",
    "SESSION_CACHE_HOURS",
    "UPDATE_TIMEOUT_SECONDS",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_RESULTS",
    "DEFAULT_CONTEXT_CHARS",
    "DEFAULT_HYBRID_WEIGHT",
    "SUPPORTED_EXTENSIONS",
    "GLOB_PATTERNS",
    "FAST_MODEL",
    "DEFAULT_MODEL",
    "MULTILINGUAL_MODEL",
    "ACCURATE_MODEL",
    "SYMBOLS",

    # Version
    "__version__",
]

# ============================================================================
# MAIN ENTRY POINT - Delegates to raggy_cli
# ============================================================================

def parse_args():
    """Legacy parse_args function - delegates to raggy_cli."""
    # Import here to avoid circular dependency
    from raggy_cli import parse_args as cli_parse_args
    return cli_parse_args()


def main():
    """Legacy entry point - delegates to raggy_cli.py implementation.

    This function exists only for backward compatibility.
    New users should use raggy_cli.py directly.
    """
    # Show another warning when main is called
    print("\n" + "="*80, file=sys.stderr)
    print("⚠️  NOTE: You are using the deprecated raggy.py wrapper.", file=sys.stderr)
    print("   Please switch to: python raggy_cli.py [command]", file=sys.stderr)
    print("   This wrapper will be removed in version 3.0.0", file=sys.stderr)
    print("="*80 + "\n", file=sys.stderr)

    # Delegate to the new CLI implementation
    from raggy_cli import main as cli_main
    cli_main()


# Legacy helper function for backward compatibility
def _determine_model(args):
    """Legacy model determination - delegates to raggy_cli."""
    from raggy_cli import _determine_model as cli_determine_model
    return cli_determine_model(args)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()