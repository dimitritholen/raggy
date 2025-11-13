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

  Advanced:
    %(prog)s rebuild --config custom.yaml       # Use custom configuration
    %(prog)s search "term" --results 10        # More results with quality scores
        """,
    )

    parser.add_argument(
        "command",
        choices=["init", "build", "rebuild", "search", "interactive", "status", "optimize", "test", "diagnose", "validate"],
        help="Command to execute",
    )
    parser.add_argument("query", nargs="*", help="Search query (for search command)")

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
    except Exception:
        pass  # Silently fail - don't interrupt user workflow

    # Create and execute command
    try:
        command = CommandFactory.create_command(args.command)

        # Handle init command specially (no RAG instance needed)
        if args.command == "init":
            command.execute(args)
            return

        # Setup dependencies for other commands
        if not args.skip_deps:
            setup_dependencies(quiet=args.quiet)
        else:
            # Still need to import even if skipping dependency checks
            try:
                import chromadb
                import PyPDF2
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
        log_error(str(e), quiet=args.quiet)
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error executing command '{args.command}'", e, quiet=args.quiet)
        sys.exit(1)


if __name__ == "__main__":
    main()