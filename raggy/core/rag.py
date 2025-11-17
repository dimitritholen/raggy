"""Main orchestrator for the RAG system."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    DEFAULT_RESULTS,
    MAX_CHUNK_SIZE,
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    MIN_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    MIN_TOP_K,
)
from ..config.loader import load_config
from ..query.processor import QueryProcessor
from ..scoring.bm25 import BM25Scorer
from ..scoring.normalization import interpret_score, normalize_cosine_distance
from ..utils.logging import log_error
from ..utils.security import validate_path
from ..utils.symbols import SYMBOLS
from .database import DatabaseManager
from .database_interface import VectorDatabase
from .document import DocumentProcessor
from .search import SearchEngine


class UniversalRAG:
    """Main orchestrator for the RAG system."""

    def __init__(
        self,
        docs_dir: str = "./docs",
        db_dir: str = "./vectordb",
        model_name: str = DEFAULT_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        quiet: bool = False,
        config_path: Optional[str] = None,
        database: Optional[VectorDatabase] = None,
    ) -> None:
        """Initialize the RAG system.

        Args:
            docs_dir: Directory containing documents
            db_dir: Directory for database storage
            model_name: Name of the embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            quiet: If True, suppress output
            config_path: Optional path to configuration file
            database: Optional VectorDatabase implementation (defaults to ChromaDB)

        Raises:
            TypeError: If parameters have incorrect types
            ValueError: If parameters are out of valid range or logically inconsistent

        """
        # Validate all inputs
        self._validate_init_params(docs_dir, db_dir, model_name, chunk_size, chunk_overlap)

        self.docs_dir = Path(docs_dir)
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.quiet = quiet

        # Load configuration
        self.config = load_config(config_path)

        # Initialize components
        self.document_processor = DocumentProcessor(
            self.docs_dir, self.config, quiet=self.quiet
        )
        self.database_manager = DatabaseManager(
            self.db_dir, quiet=self.quiet, database=database
        )
        self.query_processor = QueryProcessor(
            self.config["search"].get("expansions", {})
        )
        self.search_engine = SearchEngine(
            self.database_manager,
            self.query_processor,
            self.config,
            quiet=self.quiet
        )

        # Lazy-loaded attributes
        self._embedding_model = None

    def _validate_init_params(
        self,
        docs_dir: str,
        db_dir: str,
        model_name: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """Validate initialization parameters.

        Args:
            docs_dir: Directory containing documents
            db_dir: Directory for vector database
            model_name: Name of embedding model
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks

        Raises:
            TypeError: If parameters have incorrect types
            ValueError: If parameters are out of valid range or logically inconsistent

        """
        # Validate chunk_size
        if not isinstance(chunk_size, int):
            raise TypeError(f"chunk_size must be int, got {type(chunk_size).__name__}")
        if chunk_size < MIN_CHUNK_SIZE:
            raise ValueError(f"chunk_size must be >= {MIN_CHUNK_SIZE}, got {chunk_size}")
        if chunk_size > MAX_CHUNK_SIZE:
            raise ValueError(f"chunk_size must be <= {MAX_CHUNK_SIZE}, got {chunk_size}")

        # Validate chunk_overlap
        if not isinstance(chunk_overlap, int):
            raise TypeError(f"chunk_overlap must be int, got {type(chunk_overlap).__name__}")
        if chunk_overlap < MIN_CHUNK_OVERLAP:
            raise ValueError(f"chunk_overlap must be >= {MIN_CHUNK_OVERLAP}, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )

        # Validate paths
        if not docs_dir or not isinstance(docs_dir, str):
            raise ValueError("docs_dir must be a non-empty string")
        if not db_dir or not isinstance(db_dir, str):
            raise ValueError("db_dir must be a non-empty string")

        # Validate model_name
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

    @property
    def collection_name(self) -> str:
        """Get collection name from database manager (backward compatibility).

        Returns:
            str: Name of the collection

        """
        return self.database_manager.collection_name

    @property
    def _client(self):
        """Get database client (backward compatibility).

        Returns:
            Database client from manager

        """
        # For backward compatibility with tests
        return getattr(self.database_manager, '_database', None)

    @property
    def embedding_model(self):
        """Lazy-load embedding model.

        Returns:
            SentenceTransformer: Loaded embedding model

        """
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            if not self.quiet:
                print(f"Loading embedding model ({self.model_name})...")
            self._embedding_model = SentenceTransformer(self.model_name)
        return self._embedding_model

    def build(self, force_rebuild: bool = False) -> None:
        """Build or update the vector database.

        Args:
            force_rebuild: If True, delete existing collection first

        """
        start_time = time.time()

        files = self.document_processor.find_documents()
        if not self._validate_documents_found(files):
            return

        all_documents = self._process_documents(files)
        if not self._validate_documents_extracted(all_documents):
            return

        self._build_index(all_documents, force_rebuild)
        self._display_build_summary(len(all_documents), len(files), start_time)

    def _validate_documents_found(self, files: List[Path]) -> bool:
        """Validate documents were found.

        Args:
            files: List of document file paths

        Returns:
            bool: True if documents found

        """
        if not files:
            log_error("No documents found in docs/ directory", quiet=self.quiet)
            if not self.quiet:
                print("Solution: Add supported files to the docs/ directory")
                print("Supported formats: .md, .pdf, .docx, .txt")
                print("Example: docs/readme.md, docs/guide.pdf, docs/manual.docx, docs/notes.txt")
            return False

        if not self.quiet:
            print(f"Found {len(files)} documents")
        return True

    def _process_documents(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Process all documents into chunks.

        Args:
            files: List of document file paths

        Returns:
            List of document chunks with metadata

        """
        all_documents = []
        for i, file_path in enumerate(files, 1):
            if not self.quiet:
                print(f"[{i}/{len(files)}] Processing {file_path.name}...")
            docs = self.document_processor.process_document(file_path)
            all_documents.extend(docs)
        return all_documents

    def _validate_documents_extracted(self, all_documents: List[Dict[str, Any]]) -> bool:
        """Validate content was extracted from documents.

        Args:
            all_documents: List of extracted document chunks

        Returns:
            bool: True if content extracted

        """
        if not all_documents:
            log_error("No content could be extracted from documents", quiet=self.quiet)
            if not self.quiet:
                self._display_extraction_error_hints()
            return False

        if not self.quiet:
            print(f"Generated {len(all_documents)} text chunks")
        return True

    def _display_extraction_error_hints(self) -> None:
        """Display hints for extraction errors."""
        print("This could mean:")
        print("- PDF files are corrupted or password-protected")
        print("- Word documents (.docx) are corrupted")
        print("- Text files are empty or have encoding issues")
        print("- Markdown files are empty")
        print("- Files are not readable")
        print("Check your files and try again.")

    def _build_index(self, all_documents: List[Dict[str, Any]], force_rebuild: bool) -> None:
        """Build vector database index.

        Args:
            all_documents: List of document chunks
            force_rebuild: Whether to force rebuild

        """
        if not self.quiet:
            print("Generating embeddings...")

        texts = [doc["text"] for doc in all_documents]
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=not self.quiet
        )

        self.database_manager.build_index(
            all_documents, embeddings, force_rebuild=force_rebuild
        )

    def _display_build_summary(self, num_chunks: int, num_files: int, start_time: float) -> None:
        """Display build completion summary.

        Args:
            num_chunks: Number of chunks indexed
            num_files: Number of files processed
            start_time: Build start time

        """
        elapsed = time.time() - start_time
        print(
            f"{SYMBOLS['success']} Successfully indexed {num_chunks} chunks from {num_files} files"
        )
        print(f"Database saved to: {self.db_dir}")
        if not self.quiet:
            print(f"Build completed in {elapsed:.1f} seconds")

    def search(
        self,
        query: str,
        n_results: int = DEFAULT_RESULTS,
        hybrid: bool = False,
        expand_query: bool = False,
        show_scores: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Search the vector database with enhanced capabilities.

        Args:
            query: Search query
            n_results: Number of results to return
            hybrid: If True, use hybrid search
            expand_query: If True, expand query with synonyms
            show_scores: If True, show scores in results

        Returns:
            List[Dict[str, Any]]: Search results

        Raises:
            TypeError: If parameters have incorrect types
            ValueError: If parameters are out of valid range

        """
        # Validate query
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("query cannot be empty or whitespace-only")
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"query too long ({len(query)} chars, max {MAX_QUERY_LENGTH})")

        # Validate n_results
        if not isinstance(n_results, int):
            raise TypeError(f"n_results must be int, got {type(n_results).__name__}")
        if n_results < MIN_TOP_K:
            raise ValueError(f"n_results must be >= {MIN_TOP_K}, got {n_results}")
        if n_results > MAX_TOP_K:
            raise ValueError(f"n_results must be <= {MAX_TOP_K}, got {n_results}")

        return self.search_engine.search(
            query,
            self.embedding_model,
            n_results,
            hybrid,
            expand_query,
            show_scores
        )

    def interactive_search(self) -> None:
        """Interactive search mode."""
        print(f"\n{SYMBOLS['search']} Interactive Search Mode")
        print("Type your queries (or 'quit' to exit)")
        print("-" * 50)

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break

                if not query:
                    continue

                start_time = time.time()
                results = self.search(query)
                elapsed = time.time() - start_time

                if not results:
                    print("No results found.")
                    continue

                print(
                    f"\n{SYMBOLS['found']} Found {len(results)} results (in {elapsed:.3f}s):"
                )
                for i, result in enumerate(results, 1):
                    print(f"\n--- Result {i} ---")
                    print(f"Source: {result['metadata']['source']}")
                    print(
                        f"Chunk: {result['metadata']['chunk_index'] + 1}/{result['metadata']['total_chunks']}"
                    )
                    if result["similarity"]:
                        print(f"Similarity: {result['similarity']:.3f}")
                    print(f"Text preview: {result['text'][:200]}...")

            except KeyboardInterrupt:
                break

        print(f"\n{SYMBOLS['bye']} Goodbye!")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dict[str, Any]: Database statistics

        """
        return self.database_manager.get_stats()

    def run_self_tests(self) -> bool:
        """Run built-in self-tests for raggy functionality.

        Returns:
            bool: True if all tests pass

        """
        print(f"\n{SYMBOLS['search']} Running raggy self-tests...")

        tests_passed = 0
        tests_total = 0

        # Run individual tests
        if self._test_bm25_scorer():
            tests_passed += 1
        tests_total += 1

        if self._test_query_processor():
            tests_passed += 1
        tests_total += 1

        if self._test_path_validation():
            tests_passed += 1
        tests_total += 1

        if self._test_scoring_normalizer():
            tests_passed += 1
        tests_total += 1

        # Summary
        return self._print_test_summary(tests_passed, tests_total)

    def _test_bm25_scorer(self) -> bool:
        """Test BM25 scorer functionality.

        Returns:
            bool: True if test passes

        """
        try:
            print("Testing BM25 scorer...")
            scorer = BM25Scorer()
            test_docs = ["hello world", "world of warcraft", "hello there"]
            scorer.fit(test_docs)
            score = scorer.score("hello world", 0)
            if score > 0:
                print("✓ BM25 scorer working correctly")
                return True
            else:
                print("✗ BM25 scorer test failed")
                return False
        except (ImportError, AttributeError, ValueError, RuntimeError) as e:
            print(f"✗ BM25 scorer error: {e}")
            return False

    def _test_query_processor(self) -> bool:
        """Test query processor functionality.

        Returns:
            bool: True if test passes

        """
        try:
            print("Testing query processor...")
            processor = QueryProcessor()
            result = processor.process("test query")
            if result["original"] == "test query" and "terms" in result:
                print("✓ Query processor working correctly")
                return True
            else:
                print("✗ Query processor test failed")
                return False
        except (ImportError, AttributeError, ValueError, RuntimeError) as e:
            print(f"✗ Query processor error: {e}")
            return False

    def _test_path_validation(self) -> bool:
        """Test path validation functionality.

        Returns:
            bool: True if test passes

        """
        try:
            print("Testing path validation...")
            test_path = Path("test.txt")
            is_valid = validate_path(test_path)
            if isinstance(is_valid, bool):
                print("✓ Path validation working correctly")
                return True
            else:
                print("✗ Path validation test failed")
                return False
        except (ImportError, AttributeError, ValueError, RuntimeError) as e:
            print(f"✗ Path validation error: {e}")
            return False

    def _test_scoring_normalizer(self) -> bool:
        """Test scoring normalizer functionality.

        Returns:
            bool: True if test passes

        """
        try:
            print("Testing scoring normalizer...")
            score = normalize_cosine_distance(0.5)
            interpretation = interpret_score(0.7)
            if 0 <= score <= 1 and interpretation == "Good":
                print("✓ Scoring normalizer working correctly")
                return True
            else:
                print("✗ Scoring normalizer test failed")
                return False
        except (ImportError, AttributeError, ValueError, RuntimeError) as e:
            print(f"✗ Scoring normalizer error: {e}")
            return False

    def _print_test_summary(self, tests_passed: int, tests_total: int) -> bool:
        """Print test summary and return overall result.

        Args:
            tests_passed: Number of tests that passed
            tests_total: Total number of tests run

        Returns:
            bool: True if all tests passed

        """
        print(f"\nTest Results: {tests_passed}/{tests_total} tests passed")
        if tests_passed == tests_total:
            print(f"{SYMBOLS['success']} All tests passed!")
            return True
        else:
            print(f"⚠️  {tests_total - tests_passed} tests failed")
            return False

    def diagnose_system(self) -> None:
        """Diagnose system setup and dependencies."""
        import sys

        print(f"\n{SYMBOLS['search']} Diagnosing raggy system setup...")

        self._check_python_version(sys.version_info)
        self._check_directories()
        deps_status = self._check_dependencies()
        self._check_embedding_model(deps_status)
        self._check_database_status()

        print(f"\n{SYMBOLS['success']} Diagnosis complete!")

    def _check_python_version(self, version_info) -> None:
        """Check Python version compatibility.

        Args:
            version_info: sys.version_info object

        """
        python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        print(f"Python version: {python_version}")
        if version_info >= (3, 8):
            print("✓ Python version compatible")
        else:
            print("⚠️  Python 3.8+ recommended")

    def _check_directories(self) -> None:
        """Check required directories exist."""
        print(f"Docs directory: {self.docs_dir}")
        if self.docs_dir.exists():
            doc_count = len(list(self.docs_dir.glob("**/*")))
            print(f"✓ Docs directory exists ({doc_count} files)")
        else:
            print("⚠️  Docs directory not found")

        print(f"Database directory: {self.db_dir}")
        if self.db_dir.exists():
            print("✓ Database directory exists")
        else:
            print("ℹ️  Database directory will be created on first build")

    def _check_dependencies(self) -> list:
        """Check required dependencies are installed.

        Returns:
            list: Status of each dependency (True if installed)

        """
        print("\nDependency check:")
        deps_status = []

        deps_status.append(self._check_dependency("chromadb", "✗ ChromaDB not installed"))
        deps_status.append(self._check_dependency("sentence_transformers", "✗ sentence-transformers not installed", import_from="sentence_transformers.SentenceTransformer"))
        deps_status.append(self._check_dependency("pypdf", "⚠️  pypdf not installed (PDF support disabled)"))
        deps_status.append(self._check_dependency("docx", "⚠️  python-docx not installed (DOCX support disabled)", import_from="docx.Document"))

        return deps_status

    def _check_dependency(self, package: str, error_msg: str, import_from: str = None) -> bool:
        """Check if a single dependency is installed.

        Args:
            package: Package name to check
            error_msg: Error message to display if not found
            import_from: Specific import path (e.g., "sentence_transformers.SentenceTransformer")

        Returns:
            bool: True if dependency is installed

        """
        try:
            if import_from:
                module_path, class_name = import_from.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)
            else:
                __import__(package)

            # Use friendly package names for display
            display_name = {
                "chromadb": "ChromaDB",
                "sentence_transformers": "sentence-transformers",
                "pypdf": "pypdf",
                "docx": "python-docx"
            }.get(package, package)
            print(f"✓ {display_name} installed")
            return True
        except ImportError:
            print(error_msg)
            return False

    def _check_embedding_model(self, deps_status: list) -> None:
        """Check embedding model can be loaded.

        Args:
            deps_status: List of dependency statuses

        """
        if not all(deps_status[:2]):  # ChromaDB and sentence-transformers required
            return

        try:
            from sentence_transformers import SentenceTransformer

            print(f"\nTesting embedding model: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            test_embedding = model.encode(["test"])
            print(f"✓ Embedding model loaded successfully (dimensions: {len(test_embedding[0])})")
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            print(f"⚠️  Embedding model error: {e}")

    def _check_database_status(self) -> None:
        """Check database accessibility and stats."""
        try:
            stats = self.get_stats()
            print("\nDatabase status:")
            if "error" not in stats:
                print("✓ Database accessible")
                print(f"  Total chunks: {stats['total_chunks']}")
                print(f"  Documents indexed: {len(stats['sources'])}")
            else:
                print("ℹ️  No database found - run 'raggy build' to create")
        except (OSError, RuntimeError, ValueError) as e:
            print(f"⚠️  Database check error: {e}")

    def validate_configuration(self) -> bool:
        """Validate configuration and setup.

        Returns:
            bool: True if configuration is valid

        """
        print(f"\n{SYMBOLS['search']} Validating raggy configuration...")

        issues = []
        issues.extend(self._validate_search_config())
        issues.extend(self._validate_chunking_config())
        issues.extend(self._validate_models_config())
        issues.extend(self._validate_expansions())

        return self._report_validation_results(issues)

    def _validate_search_config(self) -> list:
        """Validate search configuration parameters.

        Returns:
            list: List of validation error messages

        """
        issues = []
        search_config = self.config.get("search", {})

        # Validate hybrid_weight
        hybrid_weight = search_config.get("hybrid_weight", 0.7)
        if not isinstance(hybrid_weight, (int, float)) or not (0 <= hybrid_weight <= 1):
            issues.append("Invalid hybrid_weight in search config (should be 0.0-1.0)")

        # Validate chunk_size
        chunk_size = search_config.get("chunk_size", 1000)
        if not isinstance(chunk_size, int) or chunk_size < 100:
            issues.append("Invalid chunk_size in search config (should be >= 100)")

        # Validate max_results
        max_results = search_config.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1:
            issues.append("Invalid max_results in search config (should be >= 1)")

        return issues

    def _validate_chunking_config(self) -> list:
        """Validate chunking configuration parameters.

        Returns:
            list: List of validation error messages

        """
        issues = []
        chunking_config = self.config.get("chunking", {})

        min_size = chunking_config.get("min_chunk_size", 300)
        max_size = chunking_config.get("max_chunk_size", 1500)

        if not isinstance(min_size, int) or min_size < 50:
            issues.append("Invalid min_chunk_size (should be >= 50)")

        if not isinstance(max_size, int) or max_size < min_size:
            issues.append("max_chunk_size should be >= min_chunk_size")

        return issues

    def _validate_models_config(self) -> list:
        """Validate model presets configuration.

        Returns:
            list: List of validation error messages

        """
        issues = []
        models_config = self.config.get("models", {})
        required_models = ["default", "fast", "multilingual", "accurate"]

        for model_type in required_models:
            if model_type not in models_config:
                issues.append(f"Missing {model_type} model in configuration")

        return issues

    def _validate_expansions(self) -> list:
        """Validate query expansion configuration.

        Returns:
            list: List of validation error messages

        """
        issues = []
        search_config = self.config.get("search", {})
        expansions = search_config.get("expansions", {})

        if not expansions:
            return issues

        for term, expansion_list in expansions.items():
            if not isinstance(expansion_list, list) or len(expansion_list) < 2:
                issues.append(
                    f"Invalid expansion for '{term}' "
                    "(should be list with original + synonyms)"
                )

        return issues

    def _report_validation_results(self, issues: list) -> bool:
        """Report validation results to user.

        Args:
            issues: List of validation error messages

        Returns:
            bool: True if no issues found

        """
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"⚠️  {issue}")
            print(f"\n{len(issues)} issues need attention")
            return False

        print("✓ Configuration is valid")
        print(f"{SYMBOLS['success']} All validation checks passed!")
        return True

    # =========================================================================
    # MEMORY SYSTEM INTEGRATION (New in 2.0)
    # =========================================================================

    def remember(
        self,
        text: str,
        memory_type: str = "note",
        tags: Optional[List[str]] = None,
        priority: str = "medium",
        **kwargs
    ) -> str:
        """Remember AI development context.

        Convenience wrapper around Memory.add() that reuses the RAG system's
        database connection. Stores development context (decisions, solutions,
        patterns) that can be retrieved later.

        This allows using a single UniversalRAG instance for both document
        search AND memory management without multiple database connections.

        Args:
            text: Memory content to store (max 100KB)
            memory_type: Type of memory. One of:
                - "decision": Architecture or design decisions
                - "solution": Problem solutions and workarounds
                - "pattern": Code patterns and best practices
                - "learning": Lessons learned and insights
                - "error": Error resolutions and debugging notes
                - "note": General development notes
                Default: "note"
            tags: Optional list of tags for categorization
            priority: Priority level (high|medium|low). Default: "medium"
            **kwargs: Additional metadata (files_involved, session_id, etc.)

        Returns:
            str: Unique memory ID (format: mem_YYYYMMDD_HHMMSS_hash)

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If memory storage fails

        Example:
            >>> rag = UniversalRAG(docs_dir="./docs", db_dir="./vectordb")
            >>>
            >>> # Store a decision while working with documents
            >>> mem_id = rag.remember(
            ...     "Decided to use hybrid search (BM25 + semantic) for better "
            ...     "accuracy on technical queries",
            ...     memory_type="decision",
            ...     tags=["search", "architecture"],
            ...     priority="high"
            ... )
            >>>
            >>> # Store a bug fix
            >>> rag.remember(
            ...     "Fixed ChromaDB metadata error by removing empty lists",
            ...     memory_type="solution",
            ...     tags=["chromadb", "bug-fix"]
            ... )
            >>>
            >>> # Later: search documents AND recall memory
            >>> doc_results = rag.search("database architecture")
            >>> memory_results = rag.recall("database architecture decisions")

        Note:
            Memory is stored in a separate collection ("project_memory") from
            documents, enabling different lifecycle management.

        """
        # Lazy-initialize memory manager (reuse database connection)
        if not hasattr(self, '_memory_manager'):
            from .memory import MemoryManager
            self._memory_manager = MemoryManager(
                db_dir=str(self.db_dir),
                model_name=self.model_name,
                collection_name="project_memory",
                quiet=self.quiet,
                database=None  # Uses default ChromaDB
            )

        return self._memory_manager.add(
            text=text,
            memory_type=memory_type,
            tags=tags,
            priority=priority,
            **kwargs
        )

    def recall(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        since: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Recall AI development context.

        Convenience wrapper around Memory.search() that reuses the RAG system's
        database connection. Searches stored development context by semantic
        similarity.

        Args:
            query: Search query text
            memory_types: Optional list of memory types to filter by
            tags: Optional list of tags to filter by (OR logic)
            since: Optional ISO 8601 timestamp to filter after this date
            limit: Maximum results to return. Default: 10
            **kwargs: Additional search parameters

        Returns:
            List[Dict[str, Any]]: List of memory entries with text and metadata

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If search fails

        Example:
            >>> rag = UniversalRAG(docs_dir="./docs", db_dir="./vectordb")
            >>>
            >>> # Search for architecture decisions
            >>> results = rag.recall(
            ...     "database architecture decisions",
            ...     memory_types=["decision", "pattern"],
            ...     tags=["architecture"],
            ...     limit=5
            ... )
            >>>
            >>> for result in results:
            ...     print(f"{result['metadata']['memory_type']}: {result['text'][:80]}...")
            >>>
            >>> # Combine document search with memory recall
            >>> doc_results = rag.search("API design patterns")
            >>> memory_context = rag.recall("API design decisions", limit=3)
            >>>
            >>> # Use both for comprehensive context
            >>> print("Documents found:", len(doc_results))
            >>> print("Related decisions:", len(memory_context))

        """
        # Lazy-initialize memory manager (reuse database connection)
        if not hasattr(self, '_memory_manager'):
            from .memory import MemoryManager
            self._memory_manager = MemoryManager(
                db_dir=str(self.db_dir),
                model_name=self.model_name,
                collection_name="project_memory",
                quiet=self.quiet,
                database=None  # Uses default ChromaDB
            )

        return self._memory_manager.search(
            query=query,
            memory_types=memory_types,
            tags=tags,
            since=since,
            limit=limit
        )
