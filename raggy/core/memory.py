"""AI Development Memory System - Core memory management.

This module provides both internal (MemoryManager) and public (Memory) APIs
for managing AI development context across coding sessions.

Example:
    >>> from raggy import Memory
    >>> memory = Memory(db_dir="./vectordb")
    >>> mem_id = memory.add(
    ...     text="Decided to use dependency injection for database layer",
    ...     memory_type="decision",
    ...     tags=["architecture", "database"]
    ... )
    >>> results = memory.search("database architecture decisions")
"""

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.loader import load_config
from ..utils.logging import log_error
from ..utils.security import validate_path
from .database import DatabaseManager
from .database_interface import VectorDatabase

# Memory type constants
MEMORY_TYPES = {
    "decision",
    "solution",
    "pattern",
    "learning",
    "error",
    "note"
}

# Priority levels
PRIORITY_LEVELS = {"high", "medium", "low"}

# Maximum memory text size (100KB)
MAX_MEMORY_SIZE = 100 * 1024


class MemoryManager:
    """Manages AI development memory storage and retrieval.

    This class provides an interface for storing, searching, and managing
    development context (decisions, solutions, patterns, learnings) that can
    be used by AI coding assistants to maintain context across sessions.

    The memory system uses a separate collection from documentation to enable
    different lifecycle management and search strategies.
    """

    def __init__(
        self,
        db_dir: str = "./vectordb",
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "project_memory",
        quiet: bool = False,
        config_path: Optional[str] = None,
        database: Optional[VectorDatabase] = None,
    ) -> None:
        """Initialize the Memory Manager.

        Args:
            db_dir: Directory for database storage
            model_name: Name of the embedding model
            collection_name: Name of the memory collection
            quiet: If True, suppress output
            config_path: Optional path to configuration file
            database: Optional VectorDatabase implementation (defaults to ChromaDB)

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_init_params(db_dir, model_name, collection_name)

        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self.collection_name = collection_name
        self.quiet = quiet

        # Load configuration
        self.config = load_config(config_path)

        # Initialize database manager with memory collection
        self.database_manager = DatabaseManager(
            self.db_dir,
            collection_name=self.collection_name,
            quiet=self.quiet,
            database=database
        )

        # Lazy-loaded embedding model
        self._embedding_model = None

    def _validate_init_params(
        self,
        db_dir: str,
        model_name: str,
        collection_name: str
    ) -> None:
        """Validate initialization parameters.

        Args:
            db_dir: Directory for vector database
            model_name: Name of embedding model
            collection_name: Name of memory collection

        Raises:
            ValueError: If parameters are invalid
        """
        if not db_dir or not isinstance(db_dir, str):
            raise ValueError("db_dir must be a non-empty string")

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")

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

    def add(
        self,
        text: str,
        memory_type: str = "note",
        tags: Optional[List[str]] = None,
        priority: str = "medium",
        files_involved: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        ai_model: Optional[str] = None,
        confidence: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """Add a new memory entry to the system.

        Args:
            text: The memory content (max 100KB)
            memory_type: Type of memory (decision|solution|pattern|learning|error|note)
            tags: Optional list of tags for categorization
            priority: Priority level (high|medium|low)
            files_involved: Optional list of file paths related to this memory
            session_id: Optional session identifier
            ai_model: Optional AI model name that created this memory
            confidence: Optional confidence score (0.0-1.0)
            **kwargs: Additional metadata fields

        Returns:
            str: Unique memory ID

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If memory storage fails
        """
        # Validate inputs
        self._validate_add_params(text, memory_type, priority, confidence)

        # Validate file paths if provided
        if files_involved:
            files_involved = self._validate_file_paths(files_involved)

        # Get git context (branch and commit)
        git_context = self._get_git_context()

        # Generate unique memory ID
        memory_id = self._generate_memory_id(text)

        # Create metadata
        metadata = self._create_metadata(
            memory_id=memory_id,
            memory_type=memory_type,
            tags=tags or [],
            priority=priority,
            files_involved=files_involved or [],
            git_branch=git_context.get("branch"),
            git_commit=git_context.get("commit"),
            session_id=session_id,
            ai_model=ai_model,
            confidence=confidence,
            **kwargs
        )

        # Generate embedding
        try:
            embedding = self.embedding_model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
        except Exception as e:
            log_error("Failed to generate embedding for memory", e, quiet=self.quiet)
            raise RuntimeError(f"Embedding generation failed: {e}") from e

        # Store in database
        try:
            collection = self.database_manager.get_collection()
            collection.add(
                ids=[memory_id],
                documents=[text],
                embeddings=[embedding.tolist()],
                metadatas=[metadata]
            )

            if not self.quiet:
                print(f"Memory stored: {memory_id}")
                print(f"  Type: {memory_type}")
                print(f"  Priority: {priority}")
                if tags:
                    print(f"  Tags: {', '.join(tags)}")

            return memory_id

        except Exception as e:
            log_error(f"Failed to store memory {memory_id}", e, quiet=self.quiet)
            raise RuntimeError(f"Memory storage failed: {e}") from e

    def _validate_add_params(
        self,
        text: str,
        memory_type: str,
        priority: str,
        confidence: Optional[float]
    ) -> None:
        """Validate add() parameters.

        Args:
            text: Memory text content
            memory_type: Type of memory
            priority: Priority level
            confidence: Optional confidence score

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate text
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if len(text.encode('utf-8')) > MAX_MEMORY_SIZE:
            raise ValueError(
                f"text size exceeds maximum of {MAX_MEMORY_SIZE} bytes "
                f"(got {len(text.encode('utf-8'))} bytes)"
            )

        # Validate memory_type
        if memory_type not in MEMORY_TYPES:
            raise ValueError(
                f"memory_type must be one of {sorted(MEMORY_TYPES)}, got '{memory_type}'"
            )

        # Validate priority
        if priority not in PRIORITY_LEVELS:
            raise ValueError(
                f"priority must be one of {sorted(PRIORITY_LEVELS)}, got '{priority}'"
            )

        # Validate confidence if provided
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                raise ValueError(f"confidence must be a number, got {type(confidence).__name__}")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")

    def _validate_file_paths(self, files: List[str]) -> List[str]:
        """Validate and normalize file paths.

        Args:
            files: List of file paths

        Returns:
            List[str]: Validated file paths

        Raises:
            ValueError: If any path is invalid or contains path traversal
        """
        validated = []
        for file_path in files:
            if not isinstance(file_path, str) or not file_path.strip():
                raise ValueError(f"Invalid file path: {file_path}")

            # Check for path traversal attempts
            try:
                validate_path(file_path)
            except ValueError as e:
                raise ValueError(f"Invalid file path '{file_path}': {e}") from e

            validated.append(file_path)

        return validated

    def _get_git_context(self) -> Dict[str, Optional[str]]:
        """Get current git context (branch and commit).

        Returns:
            Dict with 'branch' and 'commit' keys (values may be None)
        """
        context = {"branch": None, "commit": None}

        try:
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if result.returncode == 0:
                context["branch"] = result.stdout.strip()

            # Get latest commit hash
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if result.returncode == 0:
                context["commit"] = result.stdout.strip()

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # Git not available or not in a git repo - graceful degradation
            pass

        return context

    def _generate_memory_id(self, text: str) -> str:
        """Generate unique memory ID with timestamp and content hash.

        Args:
            text: Memory text content

        Returns:
            str: Unique memory ID (format: mem_YYYYMMDD_HHMMSS_hash)
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
        return f"mem_{timestamp}_{content_hash}"

    def _create_metadata(
        self,
        memory_id: str,
        memory_type: str,
        tags: List[str],
        priority: str,
        files_involved: List[str],
        git_branch: Optional[str],
        git_commit: Optional[str],
        session_id: Optional[str],
        ai_model: Optional[str],
        confidence: Optional[float],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Create metadata dictionary for memory entry.

        Args:
            memory_id: Unique memory identifier
            memory_type: Type of memory
            tags: List of tags
            priority: Priority level
            files_involved: List of file paths
            git_branch: Git branch name (if available)
            git_commit: Git commit hash (if available)
            session_id: Session identifier (if provided)
            ai_model: AI model name (if provided)
            confidence: Confidence score (if provided)
            **kwargs: Additional metadata fields

        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        metadata = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": priority,
        }

        # ChromaDB doesn't allow empty lists in metadata, only add if non-empty
        if tags:
            metadata["tags"] = tags
        if files_involved:
            metadata["files_involved"] = files_involved

        # Add optional fields if provided
        if git_branch:
            metadata["git_branch"] = git_branch
        if git_commit:
            metadata["git_commit"] = git_commit
        if session_id:
            metadata["session_id"] = session_id
        if ai_model:
            metadata["ai_model"] = ai_model
        if confidence is not None:
            metadata["confidence"] = confidence

        # Add any additional kwargs
        metadata.update(kwargs)

        return metadata

    def search(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        since: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memory entries by semantic similarity and filters.

        Args:
            query: Search query text
            memory_types: Optional list of memory types to filter by
            tags: Optional list of tags to filter by (OR logic)
            since: Optional ISO timestamp to filter memories after this date
            limit: Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of matching memory entries with text and metadata

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If search fails
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        # Validate memory_types if provided
        if memory_types:
            invalid = set(memory_types) - MEMORY_TYPES
            if invalid:
                raise ValueError(f"Invalid memory types: {invalid}")

        # Generate query embedding
        try:
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False
            )
        except Exception as e:
            log_error("Failed to generate query embedding", e, quiet=self.quiet)
            raise RuntimeError(f"Query embedding generation failed: {e}") from e

        # Build metadata filter
        where = self._build_where_filter(memory_types, tags, since)

        # Search collection
        try:
            collection = self.database_manager.get_collection()
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            # Transform results
            memories = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    memories.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                    })

            return memories

        except Exception as e:
            log_error("Memory search failed", e, quiet=self.quiet)
            raise RuntimeError(f"Search operation failed: {e}") from e

    def _build_where_filter(
        self,
        memory_types: Optional[List[str]],
        tags: Optional[List[str]],
        since: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Build metadata filter for search query.

        Args:
            memory_types: Optional list of memory types
            tags: Optional list of tags
            since: Optional ISO timestamp

        Returns:
            Optional[Dict[str, Any]]: Metadata filter or None
        """
        where = {}

        if memory_types:
            where["memory_type"] = {"$in": memory_types}

        if tags:
            # Note: ChromaDB may not support tag filtering in all versions
            # This would need to be handled in post-processing if not supported
            where["tags"] = {"$in": tags}

        if since:
            where["timestamp"] = {"$gte": since}

        return where if where else None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry by ID.

        Args:
            memory_id: Unique memory identifier

        Returns:
            bool: True if deleted successfully

        Raises:
            ValueError: If memory_id is invalid
            RuntimeError: If deletion fails
        """
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")

        try:
            collection = self.database_manager.get_collection()
            collection.delete(ids=[memory_id])

            if not self.quiet:
                print(f"Memory deleted: {memory_id}")

            return True

        except Exception as e:
            log_error(f"Failed to delete memory {memory_id}", e, quiet=self.quiet)
            raise RuntimeError(f"Memory deletion failed: {e}") from e

    def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory entry by ID.

        Args:
            memory_id: Unique memory identifier

        Returns:
            Optional[Dict]: Memory entry with text, metadata, and ID, or None if not found

        Raises:
            ValueError: If memory_id is invalid
            RuntimeError: If retrieval fails
        """
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")

        try:
            collection = self.database_manager.get_collection()
            result = collection.get(ids=[memory_id], include=["documents", "metadatas"])

            if not result["ids"] or len(result["ids"]) == 0:
                return None

            return {
                "id": result["ids"][0],
                "text": result["documents"][0],
                "metadata": result["metadatas"][0]
            }

        except Exception as e:
            log_error(f"Failed to get memory {memory_id}", e, quiet=self.quiet)
            raise RuntimeError(f"Memory retrieval failed: {e}") from e

    def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        """Count memory entries.

        Args:
            where: Optional metadata filter

        Returns:
            int: Number of memory entries matching filter

        Raises:
            RuntimeError: If count operation fails
        """
        try:
            collection = self.database_manager.get_collection()
            result = collection.get(where=where, include=[])
            return len(result["ids"])

        except Exception as e:
            log_error("Failed to count memories", e, quiet=self.quiet)
            raise RuntimeError(f"Memory count failed: {e}") from e

    def delete_all(self) -> int:
        """Delete all memory entries from collection.

        Returns:
            int: Number of memories deleted

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            collection = self.database_manager.get_collection()

            # Get all IDs
            result = collection.get(include=[])
            memory_ids = result["ids"]
            count = len(memory_ids)

            if count > 0:
                collection.delete(ids=memory_ids)

            if not self.quiet:
                print(f"Deleted {count} memories")

            return count

        except Exception as e:
            log_error("Failed to delete all memories", e, quiet=self.quiet)
            raise RuntimeError(f"Memory deletion failed: {e}") from e

    def archive(self, older_than: str) -> int:
        """Archive memories older than specified date.

        Moves memories from main collection to archive collection.

        Args:
            older_than: ISO 8601 date string (e.g., "2024-10-15T00:00:00Z")

        Returns:
            int: Number of memories archived

        Raises:
            ValueError: If older_than format is invalid
            RuntimeError: If archive operation fails
        """
        if not isinstance(older_than, str) or not older_than.strip():
            raise ValueError("older_than must be a non-empty string")

        try:
            # Parse and validate ISO date
            from datetime import datetime
            cutoff_date = datetime.fromisoformat(older_than.replace('Z', '+00:00'))

            # Search for old memories
            collection = self.database_manager.get_collection()
            result = collection.get(include=["documents", "metadatas"])

            # Filter by timestamp
            old_memories = []
            old_ids = []

            for i, metadata in enumerate(result["metadatas"]):
                memory_date_str = metadata.get("timestamp")
                if memory_date_str:
                    memory_date = datetime.fromisoformat(memory_date_str.replace('Z', '+00:00'))
                    if memory_date < cutoff_date:
                        old_memories.append({
                            "id": result["ids"][i],
                            "text": result["documents"][i],
                            "metadata": metadata
                        })
                        old_ids.append(result["ids"][i])

            if len(old_memories) == 0:
                if not self.quiet:
                    print("No memories found older than", older_than)
                return 0

            # Create archive collection
            archive_manager = DatabaseManager(
                self.db_dir,
                collection_name=f"{self.collection_name}_archive",
                quiet=self.quiet
            )
            archive_collection = archive_manager.get_collection()

            # Add to archive
            for memory in old_memories:
                embedding = self.embedding_model.encode([memory["text"]])[0]
                archive_collection.add(
                    documents=[memory["text"]],
                    embeddings=[embedding],
                    metadatas=[memory["metadata"]],
                    ids=[memory["id"]]
                )

            # Delete from main collection
            collection.delete(ids=old_ids)

            if not self.quiet:
                print(f"Archived {len(old_memories)} memories to {self.collection_name}_archive")

            return len(old_memories)

        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}") from e
        except Exception as e:
            log_error("Failed to archive memories", e, quiet=self.quiet)
            raise RuntimeError(f"Memory archival failed: {e}") from e

    def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """Get formatted context string for AI prompts.

        Searches memories and formats them for inclusion in AI prompts.

        Args:
            query: Query to find relevant context
            max_tokens: Maximum tokens to include (approximate)

        Returns:
            str: Formatted context string

        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if max_tokens < 100:
            raise ValueError(f"max_tokens must be >= 100, got {max_tokens}")

        # Search for relevant memories
        memories = self.search(query, limit=10)

        if not memories:
            return "No relevant development context found."

        # Format memories for prompt
        context_parts = ["## Relevant Development Context\n"]

        current_tokens = 0
        for memory in memories:
            # Approximate token count (1 token ≈ 4 characters)
            memory_text = memory["text"]
            memory_tokens = len(memory_text) // 4

            if current_tokens + memory_tokens > max_tokens:
                break

            metadata = memory["metadata"]
            context_parts.append(
                f"\n### {metadata.get('memory_type', 'note').title()} "
                f"(Priority: {metadata.get('priority', 'medium')})\n"
            )

            if metadata.get('tags'):
                context_parts.append(f"Tags: {', '.join(metadata['tags'])}\n")

            if metadata.get('files_involved'):
                context_parts.append(f"Files: {', '.join(metadata['files_involved'])}\n")

            context_parts.append(f"\n{memory_text}\n")

            current_tokens += memory_tokens

        return "".join(context_parts)


# =============================================================================
# PUBLIC API - High-level Memory Interface
# =============================================================================


class Memory:
    """High-level API for AI development memory system.

    This class provides a simplified, production-ready interface for storing,
    searching, and managing development context (decisions, solutions, patterns,
    learnings) that AI coding assistants can use to maintain context across
    sessions.

    The Memory class wraps MemoryManager with a cleaner API focused on common
    use cases while maintaining full functionality.

    Attributes:
        db_dir (Path): Directory for vector database storage
        collection_name (str): Name of the memory collection
        quiet (bool): Whether to suppress output messages

    Example:
        >>> from raggy import Memory
        >>> memory = Memory(db_dir="./vectordb")
        >>>
        >>> # Store a decision
        >>> mem_id = memory.add(
        ...     text="Decided to use dependency injection for database layer",
        ...     memory_type="decision",
        ...     tags=["architecture", "database"],
        ...     priority="high"
        ... )
        >>>
        >>> # Search for relevant context
        >>> results = memory.search(
        ...     query="database architecture",
        ...     memory_types=["decision", "pattern"],
        ...     limit=5
        ... )
        >>>
        >>> # Get context for AI prompt
        >>> context = memory.get_context_for_prompt(
        ...     query="current architecture decisions",
        ...     max_tokens=2000
        ... )
    """

    def __init__(
        self,
        db_dir: str = "./vectordb",
        collection_name: str = "project_memory",
        model_name: str = "all-MiniLM-L6-v2",
        quiet: bool = False,
        config_path: Optional[str] = None,
        database: Optional[VectorDatabase] = None,
    ) -> None:
        """Initialize the Memory system.

        Args:
            db_dir: Directory for vector database storage. Default: "./vectordb"
            collection_name: Name of the memory collection. Default: "project_memory"
            model_name: Name of the sentence transformer embedding model.
                Default: "all-MiniLM-L6-v2" (384-dim, fast, good quality)
            quiet: If True, suppress output messages. Default: False
            config_path: Optional path to configuration file
            database: Optional VectorDatabase implementation. If None, uses ChromaDB.

        Raises:
            ValueError: If parameters are invalid

        Example:
            >>> # Basic initialization
            >>> memory = Memory()
            >>>
            >>> # Custom database directory
            >>> memory = Memory(db_dir="./my_vectordb")
            >>>
            >>> # Quiet mode (no output)
            >>> memory = Memory(quiet=True)
        """
        self._manager = MemoryManager(
            db_dir=db_dir,
            model_name=model_name,
            collection_name=collection_name,
            quiet=quiet,
            config_path=config_path,
            database=database,
        )

        # Expose commonly used attributes
        self.db_dir = self._manager.db_dir
        self.collection_name = self._manager.collection_name
        self.quiet = self._manager.quiet

    def add(
        self,
        text: str,
        memory_type: str = "note",
        tags: Optional[List[str]] = None,
        files_involved: Optional[List[str]] = None,
        priority: str = "medium",
        **kwargs: Any
    ) -> str:
        """Add a memory entry to the system.

        Stores development context (decisions, solutions, patterns, learnings)
        that can be retrieved later through semantic search. Automatically
        captures git context (branch, commit) and timestamps.

        Args:
            text: The memory content to store. Max 100KB. Should be clear,
                descriptive text that can be retrieved via semantic search.
            memory_type: Type of memory. Must be one of:
                - "decision": Architecture or design decisions
                - "solution": Problem solutions and workarounds
                - "pattern": Code patterns and best practices
                - "learning": Lessons learned and insights
                - "error": Error resolutions and debugging notes
                - "note": General development notes
                Default: "note"
            tags: Optional list of tags for categorization (e.g., ["api", "database"]).
                Use for filtering searches. Default: None
            files_involved: Optional list of file paths related to this memory
                (e.g., ["src/api/routes.py"]). Default: None
            priority: Priority level: "high", "medium", or "low". Default: "medium"
            **kwargs: Additional metadata fields (e.g., session_id, ai_model, confidence)

        Returns:
            str: Unique memory ID (format: mem_YYYYMMDD_HHMMSS_hash)
                Can be used later to retrieve or delete this specific memory.

        Raises:
            ValueError: If text is empty, too large (>100KB), or parameters are invalid
            RuntimeError: If memory storage fails (e.g., database error)

        Example:
            >>> memory = Memory()
            >>>
            >>> # Store an architecture decision
            >>> mem_id = memory.add(
            ...     text="Decided to use dependency injection pattern for database "
            ...          "layer to enable testing with mock databases and support "
            ...          "multiple database backends (ChromaDB, Pinecone, etc.)",
            ...     memory_type="decision",
            ...     tags=["architecture", "database", "testing"],
            ...     files_involved=["core/database.py", "core/database_interface.py"],
            ...     priority="high"
            ... )
            >>> print(f"Stored decision: {mem_id}")
            mem_20250113_142700_abc123
            >>>
            >>> # Store a bug fix solution
            >>> mem_id = memory.add(
            ...     text="Fixed ChromaDB 'empty list' error by not including empty "
            ...          "lists in metadata. ChromaDB doesn't allow empty list values.",
            ...     memory_type="solution",
            ...     tags=["chromadb", "bug-fix"],
            ...     priority="medium"
            ... )
            >>>
            >>> # Store a code pattern
            >>> mem_id = memory.add(
            ...     text="Using Strategy pattern for document parsers: PDFParser, "
            ...          "DOCXParser, MarkdownParser with common DocumentParser interface",
            ...     memory_type="pattern",
            ...     tags=["design-pattern", "document-processing"]
            ... )
        """
        return self._manager.add(
            text=text,
            memory_type=memory_type,
            tags=tags,
            files_involved=files_involved,
            priority=priority,
            **kwargs
        )

    def search(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        since: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memory entries by semantic similarity and filters.

        Uses semantic search (sentence embeddings) to find relevant memories.
        Results are ranked by similarity to the query and can be filtered by
        type, tags, and date.

        Args:
            query: Search query text. Use natural language to describe what
                you're looking for (e.g., "database architecture decisions").
            memory_types: Optional list of memory types to filter by. Must be
                subset of: ["decision", "solution", "pattern", "learning",
                "error", "note"]. Default: None (search all types)
            tags: Optional list of tags to filter by (OR logic - matches any tag).
                Default: None (no tag filtering)
            since: Optional ISO 8601 timestamp to filter memories created after
                this date (e.g., "2025-01-01T00:00:00Z"). Default: None (all dates)
            limit: Maximum number of results to return. Default: 10

        Returns:
            List[Dict[str, Any]]: List of memory entries, sorted by relevance.
                Each entry contains:
                - id (str): Memory ID
                - text (str): Memory content
                - metadata (dict): Memory metadata (type, tags, timestamp, etc.)
                - distance (float): Similarity distance (lower = more similar)

        Raises:
            ValueError: If query is empty, limit < 1, or memory_types are invalid
            RuntimeError: If search operation fails

        Example:
            >>> memory = Memory()
            >>>
            >>> # Simple search
            >>> results = memory.search("database architecture")
            >>> for result in results:
            ...     print(f"{result['metadata']['memory_type']}: {result['text'][:80]}...")
            decision: Decided to use dependency injection pattern for database layer to enable...
            pattern: Using Strategy pattern for document parsers: PDFParser, DOCXParser...
            >>>
            >>> # Search with filters
            >>> results = memory.search(
            ...     query="API design patterns",
            ...     memory_types=["decision", "pattern"],
            ...     tags=["api", "architecture"],
            ...     since="2025-01-01T00:00:00Z",
            ...     limit=5
            ... )
            >>>
            >>> # Access result details
            >>> if results:
            ...     top_result = results[0]
            ...     print(f"Memory ID: {top_result['id']}")
            ...     print(f"Type: {top_result['metadata']['memory_type']}")
            ...     print(f"Priority: {top_result['metadata']['priority']}")
            ...     print(f"Tags: {', '.join(top_result['metadata'].get('tags', []))}")
            ...     print(f"Text: {top_result['text']}")
        """
        return self._manager.search(
            query=query,
            memory_types=memory_types,
            tags=tags,
            since=since,
            limit=limit
        )

    def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 2000,
        memory_types: Optional[List[str]] = None,
        since: Optional[str] = None
    ) -> str:
        """Get formatted context string for AI prompts.

        Searches memories and formats them as a context block suitable for
        injection into AI assistant prompts. This enables AI to maintain
        awareness of project decisions, patterns, and learnings.

        Args:
            query: Query to find relevant context (e.g., "current architecture")
            max_tokens: Maximum tokens to include (approximate, based on 1 token ≈ 4 chars).
                Default: 2000 (~8KB of text)
            memory_types: Optional list of memory types to include. Default: None (all types)
            since: Optional ISO timestamp to filter memories. Default: None (all dates)

        Returns:
            str: Formatted context string in markdown format, ready for prompt injection.
                Contains relevant memories grouped by type with metadata.
                Returns message if no relevant memories found.

        Raises:
            ValueError: If query is empty or max_tokens < 100

        Example:
            >>> memory = Memory()
            >>>
            >>> # Get context for AI prompt about database layer
            >>> context = memory.get_context_for_prompt(
            ...     query="database architecture and patterns",
            ...     max_tokens=2000,
            ...     memory_types=["decision", "pattern"]
            ... )
            >>>
            >>> # Use in AI prompt
            >>> prompt = f'''
            ... {context}
            ...
            ... Based on the above development context, help me implement a new
            ... database adapter for PostgreSQL following our existing patterns.
            ... '''
            >>>
            >>> # Example output format:
            >>> print(context)
            ## Relevant Development Context
            <BLANKLINE>
            ### Decision (Priority: high)
            Tags: architecture, database
            Files: core/database.py, core/database_interface.py
            <BLANKLINE>
            Decided to use dependency injection pattern for database layer...
            <BLANKLINE>
            ### Pattern (Priority: medium)
            Tags: design-pattern, document-processing
            <BLANKLINE>
            Using Strategy pattern for document parsers...
        """
        # Search with filters if provided
        memories = self.search(
            query=query,
            memory_types=memory_types,
            since=since,
            limit=10
        )

        if not memories:
            return "No relevant development context found."

        # Format memories for prompt
        context_parts = ["## Relevant Development Context\n"]

        current_tokens = 0
        for memory in memories:
            # Approximate token count (1 token ≈ 4 characters)
            memory_text = memory["text"]
            memory_tokens = len(memory_text) // 4

            if current_tokens + memory_tokens > max_tokens:
                break

            metadata = memory["metadata"]
            context_parts.append(
                f"\n### {metadata.get('memory_type', 'note').title()} "
                f"(Priority: {metadata.get('priority', 'medium')})\n"
            )

            if metadata.get('tags'):
                context_parts.append(f"Tags: {', '.join(metadata['tags'])}\n")

            if metadata.get('files_involved'):
                context_parts.append(f"Files: {', '.join(metadata['files_involved'])}\n")

            context_parts.append(f"\n{memory_text}\n")

            current_tokens += memory_tokens

        return "".join(context_parts)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry by ID.

        Args:
            memory_id: Unique memory identifier (returned by add())

        Returns:
            bool: True if deleted successfully

        Raises:
            ValueError: If memory_id is invalid
            RuntimeError: If deletion fails

        Example:
            >>> memory = Memory()
            >>> mem_id = memory.add("Temporary note", memory_type="note")
            >>> memory.delete(mem_id)
            True
        """
        return self._manager.delete(memory_id)

    def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory entry by ID.

        Args:
            memory_id: Unique memory identifier

        Returns:
            Optional[Dict]: Memory entry with text and metadata, or None if not found

        Raises:
            ValueError: If memory_id is invalid
            RuntimeError: If retrieval fails

        Example:
            >>> memory = Memory()
            >>> mem_id = memory.add("Test memory", memory_type="note")
            >>> entry = memory.get_by_id(mem_id)
            >>> print(entry['text'])
            Test memory
        """
        return self._manager.get_by_id(memory_id)

    def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        """Count memory entries.

        Args:
            where: Optional metadata filter

        Returns:
            int: Number of memory entries matching filter

        Raises:
            RuntimeError: If count operation fails

        Example:
            >>> memory = Memory()
            >>> total = memory.count()
            >>> decisions = memory.count(where={"memory_type": "decision"})
        """
        return self._manager.count(where)

    def delete_all(self) -> int:
        """Delete all memory entries.

        WARNING: This operation cannot be undone. Use with caution.

        Returns:
            int: Number of memories deleted

        Raises:
            RuntimeError: If deletion fails

        Example:
            >>> memory = Memory()
            >>> count = memory.delete_all()
            >>> print(f"Deleted {count} memories")
        """
        return self._manager.delete_all()

    def archive(self, older_than: str) -> int:
        """Archive memories older than specified date.

        Moves old memories to an archive collection to keep the main
        collection focused on recent, relevant context.

        Args:
            older_than: ISO 8601 date string (e.g., "2024-10-15T00:00:00Z")

        Returns:
            int: Number of memories archived

        Raises:
            ValueError: If date format is invalid
            RuntimeError: If archive operation fails

        Example:
            >>> memory = Memory()
            >>> # Archive memories older than 6 months
            >>> count = memory.archive("2024-07-01T00:00:00Z")
            >>> print(f"Archived {count} old memories")
        """
        return self._manager.archive(older_than)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================


def remember(
    text: str,
    db_dir: str = "./vectordb",
    memory_type: str = "note",
    tags: Optional[List[str]] = None,
    priority: str = "medium",
    **kwargs: Any
) -> str:
    """Quick memory add - convenience function.

    Creates a temporary Memory instance and stores a memory entry.
    Use for one-off memory additions. For multiple operations, create
    a Memory instance to reuse the database connection.

    Args:
        text: Memory content to store
        db_dir: Directory for vector database. Default: "./vectordb"
        memory_type: Type of memory (decision|solution|pattern|learning|error|note).
            Default: "note"
        tags: Optional list of tags. Default: None
        priority: Priority level (high|medium|low). Default: "medium"
        **kwargs: Additional metadata (files_involved, session_id, etc.)

    Returns:
        str: Unique memory ID (format: mem_YYYYMMDD_HHMMSS_hash)

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If memory storage fails

    Example:
        >>> from raggy import remember
        >>>
        >>> # Quick memory storage
        >>> mem_id = remember(
        ...     "Decided to use FastAPI for API layer",
        ...     memory_type="decision",
        ...     tags=["architecture", "api"],
        ...     priority="high"
        ... )
        >>>
        >>> # With file tracking
        >>> mem_id = remember(
        ...     "Fixed CORS issue by adding middleware",
        ...     memory_type="solution",
        ...     tags=["api", "bug-fix"],
        ...     files_involved=["api/main.py"]
        ... )
    """
    memory = Memory(db_dir=db_dir, quiet=True)
    return memory.add(
        text=text,
        memory_type=memory_type,
        tags=tags,
        priority=priority,
        **kwargs
    )


def recall(
    query: str,
    db_dir: str = "./vectordb",
    memory_types: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    limit: int = 10,
    **kwargs: Any
) -> List[Dict[str, Any]]:
    """Quick memory search - convenience function.

    Creates a temporary Memory instance and searches for memories.
    Use for one-off searches. For multiple operations, create
    a Memory instance to reuse the database connection.

    Args:
        query: Search query text
        db_dir: Directory for vector database. Default: "./vectordb"
        memory_types: Optional list of memory types to filter by. Default: None (all)
        tags: Optional list of tags to filter by. Default: None (all)
        limit: Maximum results to return. Default: 10
        **kwargs: Additional search parameters (since, etc.)

    Returns:
        List[Dict[str, Any]]: List of matching memory entries with text and metadata

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If search fails

    Example:
        >>> from raggy import recall
        >>>
        >>> # Quick search
        >>> results = recall("database architecture decisions")
        >>> for result in results:
        ...     print(f"{result['metadata']['memory_type']}: {result['text'][:80]}...")
        >>>
        >>> # With filters
        >>> results = recall(
        ...     "API patterns",
        ...     memory_types=["decision", "pattern"],
        ...     tags=["api"],
        ...     limit=5
        ... )
    """
    memory = Memory(db_dir=db_dir, quiet=True)
    return memory.search(
        query=query,
        memory_types=memory_types,
        tags=tags,
        limit=limit,
        **kwargs
    )
