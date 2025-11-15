"""Abstract interface for vector database operations.

This module defines the abstract base classes that all vector database
implementations must follow, enabling dependency inversion and allowing
multiple database backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorDatabase(ABC):
    """Abstract interface for vector database operations.

    All vector database implementations (ChromaDB, Pinecone, Weaviate, etc.)
    must implement this interface to be compatible with the RAG system.
    """

    @abstractmethod
    def create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "Collection":
        """Create a new collection.

        Args:
            name: Name of the collection to create
            metadata: Optional metadata dictionary for the collection

        Returns:
            Collection: Abstract collection instance

        Raises:
            ValueError: If collection already exists
            RuntimeError: If database operation fails

        """

    @abstractmethod
    def get_collection(self, name: str) -> "Collection":
        """Get an existing collection.

        Args:
            name: Name of the collection to retrieve

        Returns:
            Collection: Abstract collection instance

        Raises:
            ValueError: If collection does not exist
            RuntimeError: If database operation fails

        """

    @abstractmethod
    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "Collection":
        """Get an existing collection or create if it doesn't exist.

        Args:
            name: Name of the collection
            metadata: Optional metadata dictionary for the collection

        Returns:
            Collection: Abstract collection instance

        Raises:
            RuntimeError: If database operation fails

        """

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Args:
            name: Name of the collection to delete

        Raises:
            ValueError: If collection does not exist
            RuntimeError: If database operation fails

        """

    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collection names.

        Returns:
            List[str]: List of collection names

        Raises:
            RuntimeError: If database operation fails

        """


class Collection(ABC):
    """Abstract interface for collection operations.

    Represents a collection/index within a vector database where
    documents and their embeddings are stored.
    """

    @abstractmethod
    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents with embeddings to the collection.

        Args:
            ids: Unique identifiers for each document
            documents: Text content of documents
            embeddings: Vector embeddings for each document
            metadatas: Optional metadata for each document

        Raises:
            ValueError: If input lists have different lengths
            RuntimeError: If database operation fails

        """

    @abstractmethod
    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query the collection for similar documents.

        Args:
            query_texts: Query text(s) to search for
            query_embeddings: Optional pre-computed query embeddings
            n_results: Number of results to return per query
            where: Optional metadata filter
            include: Optional list of fields to include in results
                   (e.g., ["documents", "metadatas", "distances"])

        Returns:
            Dict[str, Any]: Query results with structure:
                {
                    "ids": [[...]],  # List of lists of IDs
                    "documents": [[...]],  # List of lists of documents
                    "metadatas": [[...]],  # List of lists of metadata
                    "distances": [[...]],  # List of lists of distances
                }

        Raises:
            ValueError: If query parameters are invalid
            RuntimeError: If database operation fails

        """

    @abstractmethod
    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get documents from the collection.

        Args:
            ids: Optional list of IDs to retrieve
            where: Optional metadata filter
            limit: Optional maximum number of results
            offset: Optional number of results to skip
            include: Optional list of fields to include

        Returns:
            Dict[str, Any]: Documents with structure similar to query()

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If database operation fails

        """

    @abstractmethod
    def count(self) -> int:
        """Get the total number of documents in the collection.

        Returns:
            int: Number of documents

        Raises:
            RuntimeError: If database operation fails

        """

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete documents from the collection.

        Args:
            ids: Optional list of IDs to delete
            where: Optional metadata filter for deletion

        Raises:
            ValueError: If neither ids nor where is provided
            RuntimeError: If database operation fails

        """

    @abstractmethod
    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update existing documents in the collection.

        Args:
            ids: IDs of documents to update
            documents: Optional new document texts
            embeddings: Optional new embeddings
            metadatas: Optional new metadata

        Raises:
            ValueError: If IDs don't exist or parameters are invalid
            RuntimeError: If database operation fails

        """
