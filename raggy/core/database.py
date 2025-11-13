"""Database management for vector storage using ChromaDB."""

from pathlib import Path
from typing import Any, Dict, List

from ..utils.logging import log_error


class DatabaseManager:
    """Handles ChromaDB operations and collection management."""

    def __init__(
        self,
        db_dir: Path,
        collection_name: str = "project_docs",
        quiet: bool = False
    ) -> None:
        """Initialize database manager.

        Args:
            db_dir: Directory for database storage
            collection_name: Name of the collection
            quiet: If True, suppress output
        """
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.quiet = quiet
        self._client = None

    @property
    def client(self):
        """Lazy-load ChromaDB client.

        Returns:
            ChromaDB client instance
        """
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.db_dir))
        return self._client

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
                    self.client.delete_collection(self.collection_name)
                    if not self.quiet:
                        print("Deleted existing collection")
                except Exception:
                    pass  # Collection may not exist

            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Project documentation embeddings"},
            )

            # Add to ChromaDB
            texts = [doc["text"] for doc in documents]
            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[doc["metadata"] for doc in documents],
                ids=[doc["id"] for doc in documents],
            )

        except Exception as e:
            log_error("Failed to build index", e, quiet=self.quiet)
            raise

    def get_collection(self):
        """Get the collection for search operations.

        Returns:
            ChromaDB collection instance
        """
        return self.client.get_collection(self.collection_name)

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
        except Exception:
            return {
                "error": "Database not found. Run 'python raggy.py build' first to index your documents."
            }