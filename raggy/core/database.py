"""Database management for vector storage using abstract interface."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import log_error
from .chromadb_adapter import ChromaDBAdapter
from .database_interface import VectorDatabase


class DatabaseManager:
    """Handles vector database operations through abstract interface."""

    def __init__(
        self,
        db_dir: Path,
        collection_name: str = "project_docs",
        quiet: bool = False,
        database: Optional[VectorDatabase] = None
    ) -> None:
        """Initialize database manager.

        Args:
            db_dir: Directory for database storage
            collection_name: Name of the collection
            quiet: If True, suppress output
            database: Optional VectorDatabase implementation (defaults to ChromaDB)
        """
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.quiet = quiet

        # Use provided database or default to ChromaDBAdapter
        self._database = database or ChromaDBAdapter(path=str(self.db_dir))

    @property
    def client(self):
        """Get database instance for backward compatibility.

        Returns:
            VectorDatabase instance
        """
        return self._database

    def build_index(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Any,
        force_rebuild: bool = False
    ) -> None:
        """Build or update the vector database.

        Args:
            documents: List of document chunks with text and metadata
            embeddings: Document embeddings array
            force_rebuild: If True, delete existing collection first
        """
        try:
            if force_rebuild:
                try:
                    self._database.delete_collection(self.collection_name)
                    if not self.quiet:
                        print("Deleted existing collection")
                except (ValueError, RuntimeError) as e:
                    # Collection may not exist - this is expected on first run
                    log_error(f"Could not delete collection (may not exist)", e, quiet=True)

            collection = self._database.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Project documentation embeddings"},
            )

            # Add to database through abstract interface
            texts = [doc["text"] for doc in documents]
            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[doc["metadata"] for doc in documents],
                ids=[doc["id"] for doc in documents],
            )

        except (ValueError, RuntimeError, OSError) as e:
            # Database errors: invalid parameters, connection issues
            log_error("Failed to build index", e, quiet=self.quiet)
            raise

    def get_collection(self):
        """Get the collection for search operations.

        Returns:
            Collection instance from abstract interface
        """
        return self._database.get_collection(self.collection_name)

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dict[str, Any]: Statistics including chunk count and sources
        """
        try:
            collection = self.get_collection()
            count = collection.count()

            # Get source distribution
            all_data = collection.get()
            sources = {}
            for meta in all_data["metadatas"]:
                src = meta["source"]
                sources[src] = sources.get(src, 0) + 1

            return {
                "total_chunks": count,
                "sources": sources,
                "db_path": str(self.db_dir),
            }
        except (ValueError, RuntimeError, OSError) as e:
            # Database not initialized or connection error
            log_error("Database stats unavailable", e, quiet=True)
            return {
                "error": "Database not found. Run 'python raggy.py build' first to index your documents."
            }