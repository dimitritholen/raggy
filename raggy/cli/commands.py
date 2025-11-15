"""Command implementations for the CLI."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.memory import MemoryManager
from ..setup.environment import setup_environment
from ..setup.interactive import run_interactive_setup
from ..utils.logging import log_error
from ..utils.symbols import SYMBOLS
from .base import Command


class InitCommand(Command):
    """Initialize project environment."""

    def execute(self, args: Any, rag: Any = None) -> None:
        """Execute init command.

        Args:
            args: Command-line arguments with optional --interactive/--non-interactive flags
            rag: Unused (init command doesn't require RAG instance)

        """
        # Determine if interactive mode should be used
        use_interactive = self._should_use_interactive(args)

        if use_interactive:
            # Run interactive setup questionnaire
            success = run_interactive_setup(quiet=args.quiet)
            if not success:
                sys.exit(1)

            # Also run environment setup (venv, dependencies, docs directory)
            print("\n🔧 Setting up environment...")
            success = setup_environment(quiet=args.quiet)
            if not success:
                sys.exit(1)
        else:
            # Non-interactive mode: just run environment setup
            success = setup_environment(quiet=args.quiet)
            if not success:
                sys.exit(1)

    def _should_use_interactive(self, args: Any) -> bool:
        """Determine if interactive mode should be used.

        Args:
            args: Command-line arguments

        Returns:
            bool: True if interactive mode should be used

        """
        # Explicit --non-interactive flag disables interactive mode
        if hasattr(args, 'non_interactive') and args.non_interactive:
            return False

        # Explicit --interactive flag enables interactive mode
        if hasattr(args, 'interactive') and args.interactive:
            return True

        # Default: use interactive mode if stdin is a TTY (user terminal)
        return sys.stdin.isatty()


class BuildCommand(Command):
    """Build or rebuild the vector database."""

    def execute(self, args: Any, rag: Any) -> None:
        """Execute build command."""
        force_rebuild = hasattr(args, 'force_rebuild') and args.force_rebuild
        if hasattr(args, 'command') and args.command == 'rebuild':
            force_rebuild = True
        rag.build(force_rebuild=force_rebuild)


class SearchCommand(Command):
    """Search the vector database."""

    def execute(self, args: Any, rag: Any) -> None:
        """Execute search command."""
        if not args.query:
            log_error("Please provide a search query", quiet=args.quiet)
            return

        query = " ".join(args.query)

        # Check if unified search (docs + memory) is requested
        include_memory = hasattr(args, 'include_memory') and args.include_memory

        if include_memory:
            self._unified_search(query, args, rag)
            return

        # Original behavior: search docs only
        results = rag.search(
            query, n_results=args.results, hybrid=args.hybrid, expand_query=args.expand
        )

        if not results:
            print("No results found.")
            return

        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "text": r["text"],
                            "source": r["metadata"]["source"],
                            "chunk": r["metadata"]["chunk_index"] + 1,
                            "final_score": r.get("final_score", r.get("similarity", 0)),
                            "semantic_score": r.get("semantic_score", 0),
                            "keyword_score": r.get("keyword_score", 0),
                            "interpretation": r.get("score_interpretation", "Unknown"),
                        }
                        for r in results
                    ],
                    indent=2,
                )
            )
        else:
            print(f"\n{SYMBOLS['search']} Search results for: '{query}'")
            if args.hybrid:
                print("(Using hybrid semantic + keyword search)")
            if args.expand:
                print("(Using query expansion)")
            print("=" * 50)

            for i, result in enumerate(results, 1):
                score = result.get("final_score", result.get("similarity", 0))
                interpretation = result.get("score_interpretation", "")

                score_str = f" ({interpretation}: {score:.3f})" if score else ""
                print(
                    f"\n{i}. {result['metadata']['source']} (chunk {result['metadata']['chunk_index'] + 1}){score_str}"
                )

                # Show highlighted text if available, otherwise truncated text
                display_text = result.get(
                    "highlighted_text", result["text"][:300] + "..."
                )
                print(f"   {display_text}")

    def _unified_search(self, query: str, args: Any, rag: Any) -> None:
        """Perform unified search across docs and memory collections.

        Args:
            query: Search query string
            args: Command-line arguments
            rag: RAG instance for document search

        """
        # Search documents
        doc_results = rag.search(
            query, n_results=args.results, hybrid=args.hybrid, expand_query=args.expand
        )

        # Search memory
        try:
            memory_manager = MemoryManager(
                db_dir=args.db_dir if hasattr(args, 'db_dir') else "./vectordb",
                config_path=args.config if hasattr(args, 'config') else None,
                quiet=args.quiet
            )
            memory_results = memory_manager.search(query=query, limit=args.results)
        except Exception as e:
            log_error(f"Memory search failed: {e}", quiet=args.quiet)
            memory_results = []

        # Combine and sort results
        combined = []

        for doc in doc_results:
            combined.append({
                "type": "doc",
                "score": doc.get("final_score", doc.get("similarity", 0)),
                "data": doc
            })

        for mem in memory_results:
            combined.append({
                "type": "memory",
                "score": 1.0 - mem.get("distance", 0.5),  # Convert distance to similarity
                "data": mem
            })

        # Sort by score (descending)
        combined.sort(key=lambda x: x["score"], reverse=True)

        if not combined:
            print("No results found.")
            return

        # Display unified results
        print(f"\n{SYMBOLS['search']} Unified search results for: '{query}'")
        print("(Searching both documentation and memory)")
        print("=" * 70)

        for i, item in enumerate(combined, 1):
            if item["type"] == "doc":
                result = item["data"]
                source = result["metadata"]["source"]
                chunk = result["metadata"]["chunk_index"] + 1
                score = item["score"]

                print(f"\n{i}. [DOCS] {source} (chunk {chunk}) - score: {score:.3f}")

                text = result.get("highlighted_text", result["text"])
                display_text = text if len(text) <= 200 else text[:200] + "..."
                print(f"   {display_text}")

            else:  # memory
                result = item["data"]
                metadata = result["metadata"]

                # Parse timestamp
                timestamp_str = metadata.get("timestamp", "Unknown")
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    display_time = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, AttributeError):
                    display_time = timestamp_str[:16]

                memory_type = metadata.get("memory_type", "note")
                score = item["score"]

                print(f"\n{i}. [MEMORY] {display_time} - {memory_type} - score: {score:.3f}")

                text = result["text"]
                display_text = text if len(text) <= 200 else text[:200] + "..."
                print(f"   {display_text}")


class InteractiveCommand(Command):
    """Interactive search mode."""

    def execute(self, args: Any, rag: Any) -> None:
        """Execute interactive command."""
        rag.interactive_search()


class StatusCommand(Command):
    """Show database status and statistics."""

    def execute(self, args: Any, rag: Any) -> None:
        """Execute status command."""
        stats = rag.get_stats()
        if "error" in stats:
            print(f"Error getting stats: {stats['error']}")
        else:
            print("Database Statistics:")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Database path: {stats['db_path']}")
            print(f"  Model: {rag.model_name}")
            print(f"  Config: {'Custom' if args.config else 'Default'}")
            print("  Documents:")
            for source, count in sorted(stats["sources"].items()):
                print(f"    {source}: {count} chunks")


class OptimizeCommand(Command):
    """Benchmark and optimize search settings."""

    def execute(self, args: Any, rag: Any) -> None:
        """Execute optimize command."""
        print(
            f"\n{SYMBOLS['search']} Running benchmark queries to optimize settings..."
        )

        stats = rag.get_stats()
        if not self._validate_database(stats):
            return

        test_queries = self._generate_test_queries(stats)
        print(
            f"Testing with queries derived from your content: {', '.join(test_queries)}"
        )

        results_semantic, results_hybrid = self._run_benchmarks(rag, test_queries)
        self._display_results(results_semantic, results_hybrid)
        self._display_usage_suggestions()

    def _validate_database(self, stats: Dict[str, Any]) -> bool:
        """Validate database has content.

        Args:
            stats: Database statistics

        Returns:
            bool: True if database is valid

        """
        if "error" in stats or stats["total_chunks"] == 0:
            print("Error: No indexed content found. Run 'build' first.")
            return False
        return True

    def _generate_test_queries(self, stats: Dict[str, Any]) -> List[str]:
        """Generate test queries from content.

        Args:
            stats: Database statistics

        Returns:
            List[str]: Test queries

        """
        test_queries = []
        doc_names = list(stats["sources"].keys())

        # Add filename-based queries (without extensions)
        for doc in doc_names[:3]:  # Use first 3 documents
            base_name = Path(doc).stem.replace("-", " ").replace("_", " ")
            test_queries.append(base_name)

        # Add universal technical queries
        universal_queries = [
            "configuration",
            "setup",
            "guide",
            "documentation",
            "features",
        ]
        test_queries.extend(universal_queries[:2])

        return test_queries

    def _run_benchmarks(self, rag: Any, test_queries: List[str]) -> tuple:
        """Run benchmark tests on queries.

        Args:
            rag: RAG instance
            test_queries: List of test queries

        Returns:
            tuple: (semantic_scores, hybrid_scores)

        """
        results_semantic = []
        results_hybrid = []

        for query in test_queries:
            print(f"Testing: {query}")

            # Test semantic search
            sem_score = self._test_search_mode(rag, query, hybrid=False)
            results_semantic.append(sem_score)

            # Test hybrid search
            hyb_score = self._test_search_mode(rag, query, hybrid=True)
            results_hybrid.append(hyb_score)

        return results_semantic, results_hybrid

    def _test_search_mode(self, rag: Any, query: str, hybrid: bool) -> float:
        """Test single search mode and return average score.

        Args:
            rag: RAG instance
            query: Search query
            hybrid: Whether to use hybrid search

        Returns:
            float: Average score

        """
        results = rag.search(query, n_results=3, hybrid=hybrid)
        if not results:
            return 0.0
        return sum(r.get("final_score", 0) for r in results) / len(results)

    def _display_results(self, results_semantic: List[float], results_hybrid: List[float]) -> None:
        """Display optimization results.

        Args:
            results_semantic: Semantic search scores
            results_hybrid: Hybrid search scores

        """
        avg_semantic = sum(results_semantic) / len(results_semantic)
        avg_hybrid = sum(results_hybrid) / len(results_hybrid)

        print(f"\n{SYMBOLS['found']} Optimization Results:")
        print(f"  Average Semantic Score: {avg_semantic:.3f}")
        print(f"  Average Hybrid Score: {avg_hybrid:.3f}")

        self._display_recommendation(avg_semantic, avg_hybrid)

    def _display_recommendation(self, avg_semantic: float, avg_hybrid: float) -> None:
        """Display search mode recommendation.

        Args:
            avg_semantic: Average semantic score
            avg_hybrid: Average hybrid score

        """
        if avg_hybrid > avg_semantic * 1.1:
            print(
                f"\n{SYMBOLS['success']} Recommendation: Use --hybrid flag for better results"
            )
        elif avg_semantic > avg_hybrid * 1.1:
            print(
                f"\n{SYMBOLS['success']} Recommendation: Semantic search performs best for this content"
            )
        else:
            print(f"\n{SYMBOLS['success']} Both search modes perform similarly")

    def _display_usage_suggestions(self) -> None:
        """Display suggested usage examples."""
        print("\nSuggested usage:")
        print(
            f'  python {sys.argv[0]} search "your query" --hybrid    # For exact matches'
        )
        print(
            f'  python {sys.argv[0]} search "your query" --expand    # For broader results'
        )


class TestCommand(Command):
    """Run built-in self-tests."""

    def execute(self, args: Any, rag: Any) -> None:
        """Execute test command."""
        success = rag.run_self_tests()
        if not success:
            sys.exit(1)


class DiagnoseCommand(Command):
    """Diagnose system setup and dependencies."""

    def execute(self, args: Any, rag: Any) -> None:
        """Execute diagnose command."""
        rag.diagnose_system()


class ValidateCommand(Command):
    """Validate configuration and setup."""

    def execute(self, args: Any, rag: Any) -> None:
        """Execute validate command."""
        success = rag.validate_configuration()
        if not success:
            sys.exit(1)


class RememberCommand(Command):
    """Store AI development context in memory system."""

    def execute(self, args: Any, rag: Any = None) -> None:
        """Execute remember command.

        Args:
            args: Command-line arguments with memory parameters
            rag: Unused (memory system uses separate collection)

        """
        # Get memory text from various sources
        text = self._get_memory_text(args)
        if not text:
            log_error("No memory text provided. Use positional argument, --file, or --stdin", quiet=args.quiet)
            sys.exit(1)

        # Parse tags from comma-separated string
        tags = self._parse_tags(args.tags) if hasattr(args, 'tags') and args.tags else None

        # Parse files from comma-separated string
        files = self._parse_files(args.files) if hasattr(args, 'files') and args.files else None

        # Initialize memory manager
        try:
            memory_manager = MemoryManager(
                db_dir=args.db_dir if hasattr(args, 'db_dir') else "./vectordb",
                config_path=args.config if hasattr(args, 'config') else None,
                quiet=args.quiet
            )

            # Store memory
            memory_id = memory_manager.add(
                text=text,
                memory_type=args.type if hasattr(args, 'type') and args.type else "note",
                tags=tags,
                priority=args.priority if hasattr(args, 'priority') else "medium",
                files_involved=files
            )

            if not args.quiet:
                print(f"\n{SYMBOLS['success']} Memory stored successfully!")
                print(f"Memory ID: {memory_id}")

        except ValueError as e:
            log_error(f"Invalid input: {e}", quiet=args.quiet)
            sys.exit(1)
        except RuntimeError as e:
            log_error(f"Failed to store memory: {e}", quiet=args.quiet)
            sys.exit(1)
        except Exception as e:
            log_error(f"Unexpected error: {e}", quiet=args.quiet)
            sys.exit(1)

    def _get_memory_text(self, args: Any) -> str:
        """Get memory text from positional arg, file, or stdin.

        Args:
            args: Command-line arguments

        Returns:
            str: Memory text content (empty string if none provided)

        """
        # Priority 1: Positional argument (query is used for both search and remember)
        if hasattr(args, 'query') and args.query:
            return " ".join(args.query)

        # Priority 2: --stdin flag
        if hasattr(args, 'stdin') and args.stdin:
            try:
                return sys.stdin.read().strip()
            except OSError as e:
                log_error(f"Failed to read from stdin: {e}", quiet=args.quiet)
                return ""

        # Priority 3: --file argument
        if hasattr(args, 'file') and args.file:
            try:
                file_path = Path(args.file)
                if not file_path.exists():
                    log_error(f"File not found: {args.file}", quiet=args.quiet)
                    return ""
                if not file_path.is_file():
                    log_error(f"Not a file: {args.file}", quiet=args.quiet)
                    return ""
                return file_path.read_text(encoding='utf-8').strip()
            except (OSError, UnicodeDecodeError) as e:
                log_error(f"Failed to read file {args.file}: {e}", quiet=args.quiet)
                return ""

        return ""

    def _parse_tags(self, tags_str: str) -> List[str]:
        """Parse comma-separated tags string.

        Args:
            tags_str: Comma-separated tags (e.g., "tag1,tag2,tag3")

        Returns:
            List[str]: List of cleaned tag strings

        """
        if not tags_str:
            return []

        return [tag.strip() for tag in tags_str.split(',') if tag.strip()]

    def _parse_files(self, files_str: str) -> List[str]:
        """Parse comma-separated file paths string.

        Args:
            files_str: Comma-separated file paths

        Returns:
            List[str]: List of file paths

        """
        if not files_str:
            return []

        return [f.strip() for f in files_str.split(',') if f.strip()]


class RecallCommand(Command):
    """Search AI development memory."""

    def execute(self, args: Any, rag: Any = None) -> None:
        """Execute recall command.

        Args:
            args: Command-line arguments with search parameters
            rag: RAG instance (used for --include-docs)

        """
        # Get query from positional args
        if not hasattr(args, 'query') or not args.query:
            log_error("Please provide a search query", quiet=args.quiet)
            sys.exit(1)

        query = " ".join(args.query)

        # Initialize memory manager
        try:
            memory_manager = MemoryManager(
                db_dir=args.db_dir if hasattr(args, 'db_dir') else "./vectordb",
                config_path=args.config if hasattr(args, 'config') else None,
                quiet=args.quiet
            )

            # Process temporal filters
            since = self._process_temporal_filters(args)

            # Parse memory types filter
            memory_types = None
            if hasattr(args, 'type') and args.type:
                memory_types = [args.type]

            # Parse tags filter
            tags = self._parse_tags(args.tags) if hasattr(args, 'tags') and args.tags else None

            # Get result limit
            limit = args.results if hasattr(args, 'results') and args.results else 10

            # Search memory
            memory_results = memory_manager.search(
                query=query,
                memory_types=memory_types,
                tags=tags,
                since=since,
                limit=limit
            )

            # Handle --include-docs flag
            if hasattr(args, 'include_docs') and args.include_docs and rag:
                doc_results = rag.search(query, n_results=limit)
                self._display_unified_results(memory_results, doc_results, args.quiet)
            else:
                self._display_memory_results(memory_results, query, args.quiet)

        except ValueError as e:
            log_error(f"Invalid input: {e}", quiet=args.quiet)
            sys.exit(1)
        except RuntimeError as e:
            log_error(f"Failed to search memory: {e}", quiet=args.quiet)
            sys.exit(1)
        except Exception as e:
            log_error(f"Unexpected error: {e}", quiet=args.quiet)
            sys.exit(1)

    def _process_temporal_filters(self, args: Any) -> Optional[str]:
        """Process --since and --last flags to get ISO timestamp.

        Args:
            args: Command-line arguments

        Returns:
            Optional[str]: ISO timestamp or None

        """
        # Priority 1: --since (explicit ISO date)
        if hasattr(args, 'since') and args.since:
            return args.since

        # Priority 2: --last (relative time)
        if hasattr(args, 'last') and args.last:
            return self._parse_relative_time(args.last)

        return None

    def _parse_relative_time(self, relative: str) -> str:
        """Convert relative time to ISO timestamp.

        Args:
            relative: Time string like "7d", "2w", "30d", "3m"

        Returns:
            ISO 8601 timestamp string

        Raises:
            ValueError: If format is invalid

        """
        import re
        from datetime import datetime, timedelta, timezone

        # Parse format: number + unit (d=days, w=weeks, m=months)
        match = re.match(r'^(\d+)([dwm])$', relative.lower())
        if not match:
            raise ValueError(
                f"Invalid relative time format: '{relative}'. "
                "Expected format: <number><unit> where unit is d (days), w (weeks), or m (months). "
                "Examples: 7d, 2w, 30d, 3m"
            )

        amount = int(match.group(1))
        unit = match.group(2)

        now = datetime.now(timezone.utc)

        if unit == 'd':
            past_time = now - timedelta(days=amount)
        elif unit == 'w':
            past_time = now - timedelta(weeks=amount)
        elif unit == 'm':
            # Approximate months as 30 days
            past_time = now - timedelta(days=amount * 30)
        else:
            raise ValueError(f"Invalid time unit: '{unit}'")

        return past_time.isoformat()

    def _parse_tags(self, tags_str: str) -> List[str]:
        """Parse comma-separated tags string.

        Args:
            tags_str: Comma-separated tags (e.g., "tag1,tag2,tag3")

        Returns:
            List[str]: List of cleaned tag strings

        """
        if not tags_str:
            return []

        return [tag.strip() for tag in tags_str.split(',') if tag.strip()]

    def _display_memory_results(self, results: List[Dict[str, Any]], query: str, quiet: bool) -> None:
        """Display memory search results.

        Args:
            results: List of memory search results
            query: Original search query
            quiet: Quiet mode flag

        """
        if not results:
            print("No memory results found.")
            return

        print(f"\n{SYMBOLS['search']} Memory results for: '{query}'")
        print("=" * 70)

        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            text = result["text"]

            # Parse timestamp
            timestamp_str = metadata.get("timestamp", "Unknown")
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                display_time = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, AttributeError):
                display_time = timestamp_str[:16]  # Fallback

            # Format memory type and tags
            memory_type = metadata.get("memory_type", "note")
            tags = metadata.get("tags", [])
            tag_str = " ".join(f"#{tag}" for tag in tags) if tags else ""

            # Display header
            print(f"\n{i}. [MEMORY] {display_time} | {memory_type}", end="")
            if tag_str:
                print(f" | {tag_str}", end="")
            print()

            # Display text (truncate if too long)
            display_text = text if len(text) <= 200 else text[:200] + "..."
            print(f"   {display_text}")

            # Display metadata
            files = metadata.get("files_involved", [])
            if files:
                files_str = ", ".join(files[:3])  # Show first 3 files
                if len(files) > 3:
                    files_str += f" (+{len(files) - 3} more)"
                print(f"   Files: {files_str}")

            priority = metadata.get("priority")
            if priority and priority != "medium":
                print(f"   Priority: {priority}")

    def _display_unified_results(
        self,
        memory_results: List[Dict[str, Any]],
        doc_results: List[Dict[str, Any]],
        quiet: bool
    ) -> None:
        """Display unified search results from both memory and docs.

        Args:
            memory_results: Memory search results
            doc_results: Document search results
            quiet: Quiet mode flag

        """
        # Combine results with type indicator
        combined = []

        for mem in memory_results:
            combined.append({
                "type": "memory",
                "score": 1.0 - mem.get("distance", 0.5),  # Convert distance to similarity
                "data": mem
            })

        for doc in doc_results:
            combined.append({
                "type": "doc",
                "score": doc.get("final_score", doc.get("similarity", 0)),
                "data": doc
            })

        # Sort by score (descending)
        combined.sort(key=lambda x: x["score"], reverse=True)

        if not combined:
            print("No results found.")
            return

        print(f"\n{SYMBOLS['search']} Unified search results (Memory + Docs)")
        print("=" * 70)

        for i, item in enumerate(combined, 1):
            if item["type"] == "memory":
                result = item["data"]
                metadata = result["metadata"]

                # Parse timestamp
                timestamp_str = metadata.get("timestamp", "Unknown")
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    display_time = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, AttributeError):
                    display_time = timestamp_str[:16]

                memory_type = metadata.get("memory_type", "note")
                print(f"\n{i}. [MEMORY] {display_time} - {memory_type} (score: {item['score']:.3f})")

                text = result["text"]
                display_text = text if len(text) <= 150 else text[:150] + "..."
                print(f"   {display_text}")

            else:  # doc
                result = item["data"]
                source = result["metadata"]["source"]
                chunk = result["metadata"]["chunk_index"] + 1
                print(f"\n{i}. [DOCS] {source} (chunk {chunk}, score: {item['score']:.3f})")

                text = result.get("highlighted_text", result["text"])
                display_text = text if len(text) <= 150 else text[:150] + "..."
                print(f"   {display_text}")


class ForgetCommand(Command):
    """Delete or archive memories from the memory system."""

    def execute(self, args: Any, rag: Any = None) -> None:
        """Execute forget command.

        Args:
            args: Command-line arguments with forget parameters
            rag: Unused (memory system uses separate collection)

        """
        # Initialize memory manager
        try:
            memory_manager = MemoryManager(
                db_dir=args.db_dir if hasattr(args, 'db_dir') else "./vectordb",
                config_path=args.config if hasattr(args, 'config') else None,
                quiet=args.quiet
            )

            # Determine operation mode
            if hasattr(args, 'all') and args.all:
                # Mode 1: Delete all memories
                self._delete_all(memory_manager, args.quiet)

            elif hasattr(args, 'archive') and args.archive:
                # Mode 2: Archive old memories
                if not hasattr(args, 'older_than') or not args.older_than:
                    log_error("--archive requires --older-than parameter", quiet=args.quiet)
                    sys.exit(1)
                self._archive_old(memory_manager, args.older_than, args.quiet)

            elif hasattr(args, 'memory_id') and args.memory_id:
                # Mode 3: Delete by ID
                self._delete_by_id(memory_manager, args.memory_id, args.quiet)

            else:
                log_error(
                    "forget command requires one of: <memory_id>, --archive --older-than <date>, or --all",
                    quiet=args.quiet
                )
                sys.exit(1)

        except ValueError as e:
            log_error(f"Invalid input: {e}", quiet=args.quiet)
            sys.exit(1)
        except RuntimeError as e:
            log_error(f"Operation failed: {e}", quiet=args.quiet)
            sys.exit(1)
        except Exception as e:
            log_error(f"Unexpected error: {e}", quiet=args.quiet)
            sys.exit(1)

    def _delete_by_id(self, memory_manager: MemoryManager, memory_id: str, quiet: bool) -> None:
        """Delete a single memory by ID with confirmation.

        Args:
            memory_manager: MemoryManager instance
            memory_id: Memory ID to delete
            quiet: Quiet mode flag

        """
        # Get memory to show user before deletion
        memory = memory_manager.get_by_id(memory_id)

        if not memory:
            log_error(f"Memory not found: {memory_id}", quiet=quiet)
            sys.exit(1)

        # Display memory details
        if not quiet:
            metadata = memory["metadata"]
            memory_type = metadata.get("memory_type", "note")
            timestamp_str = metadata.get("timestamp", "")

            # Format timestamp
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                display_time = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, AttributeError):
                display_time = timestamp_str[:16] if timestamp_str else "unknown"

            print(f"\n{SYMBOLS['found']} Memory to delete:")
            print(f"Type: [{memory_type}] {display_time}")

            text = memory["text"]
            display_text = text if len(text) <= 200 else text[:200] + "..."
            print(f"Content: {display_text}\n")

        # Confirm deletion
        if not self._confirm_action(f"Delete memory '{memory_id}'?", quiet):
            if not quiet:
                print("Operation cancelled")
            return

        # Delete memory
        memory_manager.delete(memory_id)

        if not quiet:
            print(f"{SYMBOLS['success']} Memory deleted: {memory_id}")

    def _archive_old(self, memory_manager: MemoryManager, older_than: str, quiet: bool) -> None:
        """Archive memories older than specified date.

        Args:
            memory_manager: MemoryManager instance
            older_than: Relative time string (e.g., "90d", "6m")
            quiet: Quiet mode flag

        """
        # Convert relative time to ISO timestamp
        iso_date = self._parse_relative_time(older_than)

        # Preview how many will be archived
        try:
            from datetime import datetime
            cutoff_date = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))

            # Count memories that will be archived
            collection = memory_manager.database_manager.get_collection()
            result = collection.get(include=["metadatas"])

            old_count = 0
            for metadata in result["metadatas"]:
                memory_date_str = metadata.get("timestamp")
                if memory_date_str:
                    memory_date = datetime.fromisoformat(memory_date_str.replace('Z', '+00:00'))
                    if memory_date < cutoff_date:
                        old_count += 1

            if old_count == 0:
                if not quiet:
                    print(f"No memories found older than {cutoff_date.strftime('%Y-%m-%d')}")
                return

            if not quiet:
                print(f"\n{SYMBOLS['found']} Found {old_count} memories older than {cutoff_date.strftime('%Y-%m-%d')}")

            # Confirm archival
            if not self._confirm_action(f"Archive {old_count} memories?", quiet):
                if not quiet:
                    print("Operation cancelled")
                return

            # Perform archive
            archived_count = memory_manager.archive(iso_date)

            if not quiet:
                print(f"{SYMBOLS['success']} Archived {archived_count} memories to project_memory_archive")

        except Exception as e:
            log_error(f"Failed to archive memories: {e}", quiet=quiet)
            sys.exit(1)

    def _delete_all(self, memory_manager: MemoryManager, quiet: bool) -> None:
        """Delete all memories with strict confirmation.

        Args:
            memory_manager: MemoryManager instance
            quiet: Quiet mode flag

        """
        # Count total memories
        total_count = memory_manager.count()

        if total_count == 0:
            if not quiet:
                print("No memories to delete")
            return

        if not quiet:
            print(f"\n⚠️  WARNING: This will delete ALL {total_count} memories from project_memory")
            print("This action cannot be undone!")

        # Require explicit "yes" confirmation for delete all
        if not self._confirm_action_strict("Type 'yes' to confirm deletion of all memories", quiet):
            if not quiet:
                print("Operation cancelled")
            return

        # Delete all memories
        deleted_count = memory_manager.delete_all()

        if not quiet:
            print(f"{SYMBOLS['success']} Deleted {deleted_count} memories")

    def _confirm_action(self, prompt: str, quiet: bool) -> bool:
        """Prompt user for y/n confirmation.

        Args:
            prompt: Confirmation prompt
            quiet: Quiet mode flag (auto-confirms in quiet mode)

        Returns:
            bool: True if confirmed, False otherwise

        """
        if quiet:
            return True

        try:
            response = input(f"{prompt} (y/n): ").strip().lower()
            return response in ('y', 'yes')
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled")
            return False

    def _confirm_action_strict(self, prompt: str, quiet: bool) -> bool:
        """Prompt user for strict 'yes' confirmation.

        Args:
            prompt: Confirmation prompt
            quiet: Quiet mode flag (auto-confirms in quiet mode)

        Returns:
            bool: True if user typed exactly 'yes', False otherwise

        """
        if quiet:
            return True

        try:
            response = input(f"{prompt}: ").strip()
            return response == 'yes'
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled")
            return False

    def _parse_relative_time(self, relative: str) -> str:
        """Convert relative time to ISO timestamp.

        Args:
            relative: Time string like "7d", "2w", "30d", "3m", "1y"

        Returns:
            ISO 8601 timestamp string

        Raises:
            ValueError: If format is invalid

        """
        import re
        from datetime import datetime, timedelta, timezone

        # Parse format: number + unit (d=days, w=weeks, m=months, y=years)
        match = re.match(r'^(\d+)([dwmy])$', relative.lower())
        if not match:
            raise ValueError(
                f"Invalid relative time format: '{relative}'. "
                "Expected format: <number><unit> where unit is d (days), w (weeks), m (months), or y (years). "
                "Examples: 7d, 2w, 30d, 3m, 1y"
            )

        amount = int(match.group(1))
        unit = match.group(2)

        now = datetime.now(timezone.utc)

        if unit == 'd':
            past_time = now - timedelta(days=amount)
        elif unit == 'w':
            past_time = now - timedelta(weeks=amount)
        elif unit == 'm':
            # Approximate: 30 days per month
            past_time = now - timedelta(days=amount * 30)
        elif unit == 'y':
            # Approximate: 365 days per year
            past_time = now - timedelta(days=amount * 365)
        else:
            raise ValueError(f"Unknown time unit: {unit}")

        return past_time.isoformat()
