#!/usr/bin/env python3
"""Universal ChromaDB RAG Setup Script v2.0.0 - Entry Point.

This is a thin wrapper that imports the refactored raggy package.
The actual implementation is in the raggy/ package.
"""

import argparse
import sys
from typing import Any

from raggy import (
    CommandFactory,
    UniversalRAG,
    __version__,
    check_for_updates,
    load_config,
    setup_dependencies,
)
from raggy.config.constants import DEFAULT_MODEL, FAST_MODEL
from raggy.utils.logging import log_error


def parse_args() -> Any:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Universal ChromaDB RAG Setup Script v2.0.0 - Enhanced with hybrid search and smart chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Setup:
    %(prog)s init                               # Initialize project environment (first-time setup)

  Basic Usage:
    %(prog)s build                              # Build/update index with smart chunking
    %(prog)s search "your search term"         # Semantic search with normalized scores
    %(prog)s status                             # Database statistics and configuration

  Enhanced Search:
    %(prog)s search "exact phrase" --hybrid    # Hybrid semantic + keyword search
    %(prog)s search "api" --expand             # Query expansion (api → application programming interface)
    %(prog)s search "documentation" --hybrid --expand # Combined hybrid + expansion

  Model Selection:
    %(prog)s build --model-preset multilingual  # Use multilingual model for non-English content
    %(prog)s search "query" --model-preset fast # Quick search with smaller model

  Output & Analysis:
    %(prog)s search "query" --json             # Enhanced JSON with score breakdown
    %(prog)s optimize                           # Benchmark semantic vs hybrid search
    %(prog)s interactive --quiet                # Interactive mode, minimal output

  Memory Management:
    %(prog)s remember "Fixed bug in search"    # Store development context
    %(prog)s recall "bug fix"                   # Search memories
    %(prog)s forget <memory_id>                 # Delete specific memory
    %(prog)s forget --archive --older-than 90d  # Archive old memories
    %(prog)s forget --all                       # Delete all memories (requires strict confirmation)

  Advanced:
    %(prog)s rebuild --config custom.yaml       # Use custom configuration
    %(prog)s search "term" --results 10        # More results with quality scores
        """,
    )

    parser.add_argument(
        "command",
        choices=["init", "build", "rebuild", "search", "interactive", "status", "optimize", "test", "diagnose", "validate", "remember", "recall", "forget"],
        help="Command to execute",
    )
    parser.add_argument("query", nargs="*", help="Search query (for search/recall commands), memory text (for remember command), or memory ID (for forget command)")

    # Options
    parser.add_argument(
        "--docs-dir", default="./docs", help="Documents directory (default: ./docs)"
    )
    parser.add_argument(
        "--db-dir",
        default="./vectordb",
        help="Vector database directory (default: ./vectordb)",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="Embedding model name"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Text chunk size (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Text chunk overlap (default: 200)",
    )
    parser.add_argument(
        "--results", type=int, default=5, help="Number of search results (default: 5)"
    )

    # Flags
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster, smaller model (paraphrase-MiniLM-L3-v2)",
    )
    parser.add_argument(
        "--hybrid", action="store_true", help="Use hybrid semantic+keyword search"
    )
    parser.add_argument(
        "--expand", action="store_true", help="Expand query with synonyms"
    )
    parser.add_argument(
        "--model-preset",
        choices=["fast", "balanced", "multilingual", "accurate"],
        help="Use model preset (overrides --model)",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency checks (faster startup)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument(
        "--json", action="store_true", help="Output search results as JSON"
    )
    parser.add_argument(
        "--config", help="Path to config file (default: raggy_config.yaml)"
    )
    parser.add_argument("--version", action="version", version=f"raggy {__version__}")

    # Init command specific arguments
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive setup questionnaire (for init command)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive setup questionnaire (for init command)"
    )

    # Remember command specific arguments
    parser.add_argument(
        "--file",
        help="Read memory text from file (for remember command)"
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read memory text from stdin (for remember command)"
    )
    parser.add_argument(
        "--type",
        choices=["decision", "solution", "pattern", "learning", "error", "note"],
        default=None,
        help="Memory type (for remember command, default: note; for recall command: filter by type)"
    )
    parser.add_argument(
        "--tags",
        help="Comma-separated tags (for remember command, e.g., 'api,refactor')"
    )
    parser.add_argument(
        "--priority",
        choices=["high", "medium", "low"],
        default="medium",
        help="Priority level (for remember command, default: medium)"
    )
    parser.add_argument(
        "--files",
        help="Comma-separated file paths involved (for remember command)"
    )

    # Recall command specific arguments
    parser.add_argument(
        "--since",
        help="Filter memories after this ISO date (for recall command, e.g., '2025-01-01')"
    )
    parser.add_argument(
        "--last",
        help="Filter memories from relative time ago (for recall command, e.g., '7d', '2w', '30d', '3m')"
    )
    parser.add_argument(
        "--include-docs",
        action="store_true",
        help="Also search documentation (for recall command, unified search)"
    )

    # Forget command specific arguments
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete all memories (for forget command, requires strict confirmation)"
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Archive old memories instead of deleting (for forget command)"
    )
    parser.add_argument(
        "--older-than",
        help="Archive memories older than this time (for forget command with --archive, e.g., '90d', '6m', '1y')"
    )

    # Search command enhancement
    parser.add_argument(
        "--include-memory",
        action="store_true",
        help="Also search memory (for search command, unified search)"
    )

    return parser.parse_args()


def _determine_model(args: Any) -> str:
    """Determine which model to use based on arguments."""
    if args.model_preset:
        config = load_config(args.config)
        preset_models = {
            "fast": config["models"]["fast"],
            "multilingual": config["models"]["multilingual"],
            "accurate": config["models"]["accurate"],
        }
        return preset_models.get(args.model_preset, config["models"]["default"])
    else:
        return FAST_MODEL if args.fast else args.model


def main() -> None:
    """Main entry point using Command pattern."""
    args = parse_args()

    # Check for updates early (non-intrusive, once per session)
    try:
        config = load_config(args.config) if hasattr(args, 'config') else {}
        check_for_updates(quiet=args.quiet, config=config)
    except (OSError, RuntimeError, ValueError, ConnectionError) as e:
        # Update check failure - don't interrupt workflow, just log at debug level
        if not args.quiet:
            print(f"Debug: Update check failed: {e}")

    # Handle forget command memory_id extraction
    if args.command == "forget":
        # Extract memory_id from query argument if provided
        if args.query and len(args.query) > 0:
            args.memory_id = args.query[0]
        else:
            args.memory_id = None

    # Create and execute command
    try:
        command = CommandFactory.create_command(args.command)

        # Handle init, remember, and forget commands specially (no RAG instance needed)
        if args.command in ("init", "remember", "forget"):
            command.execute(args)
            return

        # Setup dependencies for other commands
        if not args.skip_deps:
            setup_dependencies(quiet=args.quiet)
        else:
            # Still need to import even if skipping dependency checks
            try:
                import chromadb
                import pypdf
                from sentence_transformers import SentenceTransformer

                try:
                    import magic
                except ImportError:
                    pass
            except ImportError as e:
                log_error(f"Missing dependency: {e}", quiet=args.quiet)
                log_error("Run without --skip-deps or install dependencies manually", quiet=args.quiet)
                return

        # Determine model to use
        model_name = _determine_model(args)

        # Initialize RAG system
        rag = UniversalRAG(
            docs_dir=args.docs_dir,
            db_dir=args.db_dir,
            model_name=model_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            quiet=args.quiet,
            config_path=args.config,
        )

        # Execute the command
        command.execute(args, rag)

    except ValueError as e:
        # Invalid command arguments or parameters
        log_error(str(e), quiet=args.quiet)
        sys.exit(1)
    except (ImportError, ModuleNotFoundError) as e:
        # Missing dependencies
        log_error(f"Missing dependency executing command '{args.command}'", e, quiet=args.quiet)
        sys.exit(1)
    except (OSError, RuntimeError) as e:
        # File system or runtime errors
        log_error(f"Error executing command '{args.command}'", e, quiet=args.quiet)
        sys.exit(1)


if __name__ == "__main__":
    main()