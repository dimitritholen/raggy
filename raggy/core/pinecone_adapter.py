"""Pinecone implementation of the VectorDatabase interface.

This module provides a Pinecone-based vector database adapter that implements
the VectorDatabase and Collection interfaces for cloud-based vector storage.
"""

from typing import Any, Dict, List, Optional

from .database_interface import Collection, VectorDatabase


class PineconeAdapter(VectorDatabase):
    """Pinecone implementation of VectorDatabase interface.

    This adapter wraps Pinecone's cloud vector database and provides
    a standardized interface compatible with ChromaDB adapter.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str = "raggy-index",
        dimension: int = 384,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        """Initialize Pinecone adapter.

        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            dimension: Dimension of embeddings (default 384 for all-MiniLM-L6-v2)
            cloud: Cloud provider ('aws', 'gcp', or 'azure')
            region: Cloud region (e.g., 'us-east-1', 'us-west-2', 'eu-west-1')

        Raises:
            ImportError: If pinecone package not installed
            RuntimeError: If Pinecone initialization fails

        """
        try:
            from pinecone import ServerlessSpec
            from pinecone.grpc import PineconeGRPC as Pinecone
        except ImportError as e:
            raise ImportError(
                "Pinecone package not installed. "
                "Install with: pip install \"pinecone[grpc]\""
            ) from e

        self.api_key = api_key
        self.cloud = cloud
        self.region = region
        self.index_name = index_name
        self.dimension = dimension

        try:
            # Initialize Pinecone client
            self._client = Pinecone(api_key=api_key)

            # Create index if it doesn't exist
            existing_indexes = [idx.name for idx in self._client.list_indexes()]
            if index_name not in existing_indexes:
                self._client.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )

            # Get index reference
            self._index = self._client.Index(index_name)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {e}") from e

    def create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """Create a new collection (namespace in Pinecone).

        Args:
            name: Name of the collection (becomes namespace)
            metadata: Optional metadata (stored but not used by Pinecone)

        Returns:
            PineconeCollection: Wrapped Pinecone namespace

        """
        # Pinecone uses namespaces instead of collections
        # No explicit creation needed - namespace created on first upsert
        return PineconeCollection(self._index, name, self.dimension)

    def get_collection(self, name: str) -> Collection:
        """Get an existing collection (namespace).

        Args:
            name: Name of the collection/namespace

        Returns:
            PineconeCollection: Wrapped Pinecone namespace

        """
        return PineconeCollection(self._index, name, self.dimension)

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """Get or create a collection (namespace).

        Args:
            name: Name of the collection/namespace
            metadata: Optional metadata

        Returns:
            PineconeCollection: Wrapped Pinecone namespace

        """
        return self.get_collection(name)

    def delete_collection(self, name: str) -> None:
        """Delete a collection (namespace).

        Note: Pinecone requires deleting all vectors in namespace manually.

        Args:
            name: Name of the collection/namespace to delete

        Raises:
            RuntimeError: If deletion fails

        """
        try:
            # Delete all vectors in namespace
            self._index.delete(delete_all=True, namespace=name)
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection '{name}': {e}") from e

    def list_collections(self) -> List[str]:
        """List all collections (namespaces).

        Note: Pinecone doesn't provide a direct way to list namespaces.
        This returns a note about the limitation.

        Returns:
            List[str]: List of namespaces (may be incomplete)

        """
        # Pinecone doesn't provide namespace listing
        # Return stats which includes namespace info
        try:
            stats = self._index.describe_index_stats()
            # Pinecone gRPC v7.3.0: DescribeIndexStatsResponse has .namespaces attribute (dict)
            namespaces = stats.namespaces if hasattr(stats, 'namespaces') else {}
            return list(namespaces.keys())
        except (KeyError, AttributeError, TypeError):
            # Stats structure changed or index not ready - return empty list
            # This is non-critical, listing is best-effort
            return []
        except Exception as e:
            # Network errors, API errors, or auth failures
            # Re-raise as these indicate real problems user should know about
            raise RuntimeError(f"Failed to list Pinecone collections: {e}") from e


class PineconeCollection(Collection):
    """Pinecone implementation of Collection interface.

    This class wraps a Pinecone namespace and provides collection operations
    compatible with the ChromaDB adapter interface.
    """

    def __init__(self, index, namespace: str, dimension: int):
        """Initialize Pinecone collection wrapper.

        Args:
            index: Pinecone index instance
            namespace: Namespace name
            dimension: Expected embedding dimension

        """
        self._index = index
        self._namespace = namespace
        self._dimension = dimension

    @staticmethod
    def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert list values in metadata to comma-separated strings.

        Pinecone only accepts str, int, float, bool in metadata.
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
            metadata: Metadata from Pinecone with comma-separated strings

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
        """Add documents with embeddings to Pinecone.

        Args:
            ids: Unique identifiers for each document
            documents: Text content of documents (stored in metadata)
            embeddings: Vector embeddings for each document
            metadatas: Optional metadata for each document

        Raises:
            ValueError: If input lists have different lengths
            RuntimeError: If Pinecone operation fails

        """
        try:
            # Validate input lengths
            if not (len(ids) == len(documents) == len(embeddings)):
                raise ValueError(
                    "ids, documents, and embeddings must have the same length"
                )
            if metadatas and len(metadatas) != len(ids):
                raise ValueError("metadatas must have the same length as ids")

            # Prepare vectors for upsert
            vectors = []
            for i, (id_, doc, embedding) in enumerate(zip(ids, documents, embeddings)):
                metadata = metadatas[i].copy() if metadatas else {}
                # Store document text in metadata
                metadata["document"] = doc
                # Serialize metadata (convert lists to comma-separated strings)
                metadata = self._serialize_metadata(metadata)

                vectors.append(
                    {
                        "id": id_,
                        "values": embedding,
                        "metadata": metadata,
                    }
                )

            # Upsert to Pinecone (batch size 100)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self._index.upsert(vectors=batch, namespace=self._namespace)

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to Pinecone: {e}") from e

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query Pinecone for similar documents.

        Args:
            query_texts: Query text(s) (not used if query_embeddings provided)
            query_embeddings: Pre-computed query embeddings
            n_results: Number of results to return per query
            where: Optional metadata filter
            include: Optional list of fields to include

        Returns:
            Dict: Query results in ChromaDB-compatible format

        Raises:
            ValueError: If query parameters are invalid
            RuntimeError: If Pinecone operation fails

        """
        try:
            if query_embeddings is None:
                raise ValueError(
                    "query_embeddings required for Pinecone adapter. "
                    "Embedding generation should happen before calling query()."
                )

            # Pinecone only supports single query at a time
            if len(query_embeddings) != 1:
                raise ValueError("Pinecone adapter only supports single query at a time")

            query_vector = query_embeddings[0]

            # Build filter from 'where' clause
            filter_dict = self._build_filter(where) if where else None

            # Query Pinecone
            results = self._index.query(
                vector=query_vector,
                top_k=n_results,
                namespace=self._namespace,
                filter=filter_dict,
                include_metadata=True,
            )

            # Transform to ChromaDB format
            ids = [[]]
            documents = [[]]
            metadatas = [[]]
            distances = [[]]

            # Pinecone gRPC v7.3.0: QueryResponse has .matches attribute (list)
            matches = results.matches if hasattr(results, 'matches') else []
            for match in matches:
                # Match object has .id, .metadata, .score attributes
                match_id = match.id if hasattr(match, 'id') else match.get("id", "")
                match_metadata = match.metadata if hasattr(match, 'metadata') else match.get("metadata", {})
                match_score = match.score if hasattr(match, 'score') else match.get("score", 0.0)

                ids[0].append(match_id)
                # Extract document from metadata
                # Metadata is a dict, can use .get() and .pop()
                metadata = dict(match_metadata) if match_metadata else {}
                doc = metadata.pop("document", "")
                # Deserialize metadata (convert comma-separated strings back to lists)
                metadata = self._deserialize_metadata(metadata)
                documents[0].append(doc)
                metadatas[0].append(metadata)
                # Pinecone returns similarity score, convert to distance
                # distance = 1 - similarity for cosine
                distances[0].append(1.0 - match_score)

            return {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "distances": distances,
            }

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Pinecone query failed: {e}") from e

    def _build_filter(self, where: Dict[str, Any]) -> Dict[str, Any]:
        """Build Pinecone filter from ChromaDB-style where clause.

        Args:
            where: ChromaDB-style filter (e.g., {"memory_type": {"$in": ["decision"]}})

        Returns:
            Pinecone-compatible filter

        """
        # Convert ChromaDB operators to Pinecone operators
        filter_dict = {}

        for key, value in where.items():
            if isinstance(value, dict):
                # Handle operators like $in, $eq, $gte
                if "$in" in value:
                    filter_dict[key] = {"$in": value["$in"]}
                elif "$eq" in value:
                    filter_dict[key] = value["$eq"]
                elif "$gte" in value:
                    filter_dict[key] = {"$gte": value["$gte"]}
                elif "$lte" in value:
                    filter_dict[key] = {"$lte": value["$lte"]}
            else:
                # Direct equality
                filter_dict[key] = value

        return filter_dict

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get documents from Pinecone.

        Args:
            ids: Optional list of IDs to retrieve
            where: Optional metadata filter
            limit: Optional maximum number of results
            offset: Optional number of results to skip
            include: Optional list of fields to include

        Returns:
            Dict: Documents in ChromaDB-compatible format

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If Pinecone operation fails

        """
        try:
            if ids:
                # Fetch by IDs
                response = self._index.fetch(ids=ids, namespace=self._namespace)

                result_ids = []
                result_docs = []
                result_metadatas = []

                # Pinecone gRPC v7.3.0: FetchResponse has .vectors attribute (dict)
                vectors_dict = response.vectors if hasattr(response, 'vectors') else {}
                for id_, vector_data in vectors_dict.items():
                    result_ids.append(id_)
                    metadata = vector_data.get("metadata", {})
                    doc = metadata.pop("document", "")
                    # Deserialize metadata (convert comma-separated strings back to lists)
                    metadata = self._deserialize_metadata(metadata)
                    result_docs.append(doc)
                    result_metadatas.append(metadata)

                return {
                    "ids": result_ids,
                    "documents": result_docs,
                    "metadatas": result_metadatas,
                }
            else:
                # Pinecone doesn't support scanning/listing all vectors
                # This is a limitation - return empty results
                return {"ids": [], "documents": [], "metadatas": []}

        except Exception as e:
            raise RuntimeError(f"Pinecone get operation failed: {e}") from e

    def count(self) -> int:
        """Get the total number of documents in the namespace.

        Returns:
            int: Number of documents

        Raises:
            RuntimeError: If Pinecone operation fails

        """
        try:
            stats = self._index.describe_index_stats()
            # Pinecone gRPC v7.3.0: DescribeIndexStatsResponse has .namespaces attribute (dict)
            namespaces = stats.namespaces if hasattr(stats, 'namespaces') else {}
            namespace_stats = namespaces.get(self._namespace, {})
            # namespace_stats is a dict-like object with vector_count
            return namespace_stats.get("vector_count", 0) if namespace_stats else 0
        except Exception as e:
            raise RuntimeError(f"Pinecone count operation failed: {e}") from e

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete documents from Pinecone.

        Args:
            ids: Optional list of IDs to delete
            where: Optional metadata filter for deletion

        Raises:
            ValueError: If neither ids nor where is provided
            RuntimeError: If Pinecone operation fails

        """
        try:
            if ids is None and where is None:
                raise ValueError("Either ids or where must be provided")

            if ids:
                # Delete by IDs
                self._index.delete(ids=ids, namespace=self._namespace)
            elif where:
                # Delete by filter
                filter_dict = self._build_filter(where)
                self._index.delete(filter=filter_dict, namespace=self._namespace)

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Pinecone delete operation failed: {e}") from e

    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update existing documents in Pinecone.

        Note: Pinecone updates are done via upsert.

        Args:
            ids: IDs of documents to update
            documents: Optional new document texts
            embeddings: Optional new embeddings
            metadatas: Optional new metadata

        Raises:
            ValueError: If IDs are invalid
            RuntimeError: If Pinecone operation fails

        """
        try:
            # Fetch existing vectors
            response = self._index.fetch(ids=ids, namespace=self._namespace)

            # Pinecone gRPC v7.3.0: FetchResponse has .vectors attribute (dict)
            vectors_dict = response.vectors if hasattr(response, 'vectors') else {}

            vectors = []
            for i, id_ in enumerate(ids):
                existing = vectors_dict.get(id_)
                if not existing:
                    raise ValueError(f"ID not found: {id_}")

                # Build updated vector
                vector = {
                    "id": id_,
                    "values": embeddings[i] if embeddings else existing["values"],
                }

                # Update metadata
                metadata = existing.get("metadata", {})
                if documents and i < len(documents):
                    metadata["document"] = documents[i]
                if metadatas and i < len(metadatas):
                    metadata.update(metadatas[i])

                # Serialize metadata (convert lists to comma-separated strings)
                metadata = self._serialize_metadata(metadata)
                vector["metadata"] = metadata
                vectors.append(vector)

            # Upsert updated vectors
            self._index.upsert(vectors=vectors, namespace=self._namespace)

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Pinecone update operation failed: {e}") from e
