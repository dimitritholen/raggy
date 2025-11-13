"""Command implementations for the CLI."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..setup.environment import setup_environment
from ..utils.logging import log_error
from ..utils.symbols import SYMBOLS
from .base import Command


class InitCommand(Command):
    """Initialize project environment."""

    def execute(self, args: Any, rag: Any = None) -> None:
        """Execute init command."""
        success = setup_environment(quiet=args.quiet)
        if not success:
            sys.exit(1)


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
            print(f"Database Statistics:")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Database path: {stats['db_path']}")
            print(f"  Model: {rag.model_name}")
            print(f"  Config: {'Custom' if args.config else 'Default'}")
            print(f"  Documents:")
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
        print(f"\nSuggested usage:")
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