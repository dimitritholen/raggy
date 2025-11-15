"""ChromaDB implementation of the VectorDatabase interface.

This module provides a concrete implementation of the abstract
VectorDatabase and Collection interfaces using ChromaDB as the backend.
"""

from typing import Any, Dict, List, Optional

import chromadb

from .database_interface import Collection, VectorDatabase


class ChromaDBAdapter(VectorDatabase):
    """ChromaDB implementation of VectorDatabase interface.

    This adapter wraps ChromaDB's PersistentClient and provides
    a standardized interface for vector database operations.
    """

    def __init__(self, path: str):
        """Initialize ChromaDB adapter with persistent storage.

        Args:
            path: Directory path for ChromaDB persistent storage

        """
        self._client = chromadb.PersistentClient(path=path)

    def create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """Create a new ChromaDB collection.

        Args:
            name: Name of the collection to create
            metadata: Optional metadata dictionary for the collection

        Returns:
            ChromaCollection: Wrapped ChromaDB collection

        Raises:
            ValueError: If collection already exists
            RuntimeError: If ChromaDB operation fails

        """
        try:
            chroma_collection = self._client.create_collection(
                name=name,
                metadata=metadata or {}
            )
            return ChromaCollection(chroma_collection)
        except ValueError as e:
            # Re-raise ValueError for existing collection
            raise ValueError(f"Collection '{name}' already exists") from e
        except Exception as e:
            # Wrap other exceptions as RuntimeError
            raise RuntimeError(f"Failed to create collection: {e}") from e

    def get_collection(self, name: str) -> Collection:
        """Get an existing ChromaDB collection.

        Args:
            name: Name of the collection to retrieve

        Returns:
            ChromaCollection: Wrapped ChromaDB collection

        Raises:
            ValueError: If collection does not exist
            RuntimeError: If ChromaDB operation fails

        """
        try:
            chroma_collection = self._client.get_collection(name=name)
            return ChromaCollection(chroma_collection)
        except ValueError as e:
            # Re-raise ValueError for missing collection
            raise ValueError(f"Collection '{name}' does not exist") from e
        except Exception as e:
            # Wrap other exceptions as RuntimeError
            raise RuntimeError(f"Failed to get collection: {e}") from e

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """Get an existing collection or create if it doesn't exist.

        Args:
            name: Name of the collection
            metadata: Optional metadata dictionary for the collection

        Returns:
            ChromaCollection: Wrapped ChromaDB collection

        Raises:
            RuntimeError: If ChromaDB operation fails

        """
        try:
            chroma_collection = self._client.get_or_create_collection(
                name=name,
                metadata=metadata or {}
            )
            return ChromaCollection(chroma_collection)
        except Exception as e:
            raise RuntimeError(f"Failed to get or create collection: {e}") from e

    def delete_collection(self, name: str) -> None:
        """Delete a ChromaDB collection.

        Args:
            name: Name of the collection to delete

        Raises:
            ValueError: If collection does not exist
            RuntimeError: If ChromaDB operation fails

        """
        try:
            self._client.delete_collection(name=name)
        except ValueError as e:
            # Re-raise ValueError for missing collection
            raise ValueError(f"Collection '{name}' does not exist") from e
        except Exception as e:
            # Wrap other exceptions as RuntimeError
            raise RuntimeError(f"Failed to delete collection: {e}") from e

    def list_collections(self) -> List[str]:
        """List all ChromaDB collection names.

        Returns:
            List[str]: List of collection names

        Raises:
            RuntimeError: If ChromaDB operation fails

        """
        try:
            collections = self._client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            raise RuntimeError(f"Failed to list collections: {e}") from e


class ChromaCollection(Collection):
    """ChromaDB implementation of Collection interface.

    This class wraps a ChromaDB collection and provides
    a standardized interface for collection operations.
    """

    def __init__(self, collection):
        """Initialize ChromaDB collection wrapper.

        Args:
            collection: ChromaDB collection instance

        """
        self._collection = collection

    @staticmethod
    def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert list values in metadata to comma-separated strings.

        ChromaDB 1.3.4+ only accepts str, int, float, bool in metadata.
        This method converts lists to comma-separated strings.

        Args:
            metadata: Original metadata with possible list values

        Returns:
            Metadata with lists converted to comma-separated strings

        """
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                # Convert list to comma-separated string
                serialized[key] = ",".join(str(item) for item in value)
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def _deserialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert comma-separated strings back to lists for known list fields.

        Args:
            metadata: Metadata from ChromaDB with comma-separated strings

        Returns:
            Metadata with comma-separated strings converted back to lists

        """
        # Known fields that should be lists
        list_fields = {"tags", "files_involved"}

        deserialized = {}
        for key, value in metadata.items():
            if key in list_fields and isinstance(value, str):
                # Convert comma-separated string back to list
                # Filter out empty strings from split
                deserialized[key] = [item.strip() for item in value.split(",") if item.strip()]
            else:
                deserialized[key] = value
        return deserialized

    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents with embeddings to the ChromaDB collection.

        Args:
            ids: Unique identifiers for each document
            documents: Text content of documents
            embeddings: Vector embeddings for each document
            metadatas: Optional metadata for each document

        Raises:
            ValueError: If input lists have different lengths
            RuntimeError: If ChromaDB operation fails

        """
        try:
            # Validate input lengths
            if not (len(ids) == len(documents) == len(embeddings)):
                raise ValueError(
                    "ids, documents, and embeddings must have the same length"
                )
            if metadatas and len(metadatas) != len(ids):
                raise ValueError("metadatas must have the same length as ids")

            # Serialize metadata (convert lists to comma-separated strings)
            serialized_metadatas = (
                [self._serialize_metadata(meta) for meta in metadatas]
                if metadatas
                else [{} for _ in ids]
            )

            # ChromaDB's add method
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=serialized_metadatas
            )
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            # Wrap other exceptions as RuntimeError
            raise RuntimeError(f"Failed to add documents: {e}") from e

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query the ChromaDB collection for similar documents.

        Args:
            query_texts: Query text(s) to search for
            query_embeddings: Optional pre-computed query embeddings
            n_results: Number of results to return per query
            where: Optional metadata filter
            include: Optional list of fields to include in results

        Returns:
            Dict[str, Any]: Query results from ChromaDB

        Raises:
            ValueError: If query parameters are invalid
            RuntimeError: If ChromaDB operation fails

        """
        try:
            # ChromaDB allows either query_texts or query_embeddings
            if query_embeddings is not None:
                results = self._collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where,
                    include=include
                )
            else:
                results = self._collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where,
                    include=include
                )

            # Deserialize metadata in results (convert comma-separated strings back to lists)
            if "metadatas" in results:
                results["metadatas"] = [
                    [self._deserialize_metadata(meta) for meta in query_metas]
                    for query_metas in results["metadatas"]
                ]

            return results
        except ValueError as e:
            # Re-raise ValueError for invalid parameters
            raise ValueError(f"Invalid query parameters: {e}") from e
        except Exception as e:
            # Wrap other exceptions as RuntimeError
            raise RuntimeError(f"Query failed: {e}") from e

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get documents from the ChromaDB collection.

        Args:
            ids: Optional list of IDs to retrieve
            where: Optional metadata filter
            limit: Optional maximum number of results
            offset: Optional number of results to skip
            include: Optional list of fields to include

        Returns:
            Dict[str, Any]: Documents from ChromaDB

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If ChromaDB operation fails

        """
        try:
            results = self._collection.get(
                ids=ids,
                where=where,
                limit=limit,
                offset=offset,
                include=include
            )

            # Deserialize metadata in results (convert comma-separated strings back to lists)
            if "metadatas" in results and results["metadatas"]:
                results["metadatas"] = [
                    self._deserialize_metadata(meta) for meta in results["metadatas"]
                ]

            return results
        except ValueError as e:
            raise ValueError(f"Invalid get parameters: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Get operation failed: {e}") from e

    def count(self) -> int:
        """Get the total number of documents in the ChromaDB collection.

        Returns:
            int: Number of documents

        Raises:
            RuntimeError: If ChromaDB operation fails

        """
        try:
            return self._collection.count()
        except Exception as e:
            raise RuntimeError(f"Count operation failed: {e}") from e

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete documents from the ChromaDB collection.

        Args:
            ids: Optional list of IDs to delete
            where: Optional metadata filter for deletion

        Raises:
            ValueError: If neither ids nor where is provided
            RuntimeError: If ChromaDB operation fails

        """
        try:
            if ids is None and where is None:
                raise ValueError("Either ids or where must be provided")

            self._collection.delete(ids=ids, where=where)
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            raise RuntimeError(f"Delete operation failed: {e}") from e

    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update existing documents in the ChromaDB collection.

        Args:
            ids: IDs of documents to update
            documents: Optional new document texts
            embeddings: Optional new embeddings
            metadatas: Optional new metadata

        Raises:
            ValueError: If IDs don't exist or parameters are invalid
            RuntimeError: If ChromaDB operation fails

        """
        try:
            # Serialize metadata if provided (convert lists to comma-separated strings)
            serialized_metadatas = (
                [self._serialize_metadata(meta) for meta in metadatas]
                if metadatas
                else None
            )

            self._collection.update(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=serialized_metadatas
            )
        except ValueError as e:
            raise ValueError(f"Update failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Update operation failed: {e}") from e
