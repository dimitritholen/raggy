"""Supabase (PostgreSQL + pgvector) implementation of the VectorDatabase interface.

This module provides a Supabase-based vector database adapter using pgvector
extension for PostgreSQL, implementing the VectorDatabase and Collection interfaces.
"""

from contextlib import suppress
from typing import Any, Dict, List, Optional

from .database_interface import Collection, VectorDatabase


class SupabaseAdapter(VectorDatabase):
    """Supabase implementation of VectorDatabase interface.

    This adapter uses Supabase (PostgreSQL with pgvector extension) for cloud-based
    vector storage with a standardized interface compatible with ChromaDB adapter.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        dimension: int = 384,
    ):
        """Initialize Supabase adapter.

        Args:
            url: Supabase project URL
            api_key: Supabase anon/service role key
            dimension: Dimension of embeddings (default 384 for all-MiniLM-L6-v2)

        Raises:
            ImportError: If supabase package not installed
            RuntimeError: If Supabase initialization fails

        """
        try:
            from supabase import create_client
        except ImportError as e:
            raise ImportError(
                "Supabase package not installed. "
                "Install with: pip install supabase"
            ) from e

        self.url = url
        self.api_key = api_key
        self.dimension = dimension

        try:
            # Initialize Supabase client
            self._client = create_client(url, api_key)

            # Ensure pgvector extension is enabled
            # Note: This requires database permissions
            self._ensure_pgvector_enabled()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Supabase: {e}") from e

    def _ensure_pgvector_enabled(self) -> None:
        """Ensure pgvector extension is enabled.

        Note: This requires superuser privileges. If using Supabase cloud,
        pgvector should already be enabled.
        """
        # Extension might already exist or user doesn't have permissions
        # Supabase cloud has pgvector pre-enabled, so we can continue
        with suppress(Exception):
            self._client.rpc(
                "exec_sql",
                {"query": "CREATE EXTENSION IF NOT EXISTS vector;"}
            ).execute()

    def create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """Create a new collection (table in Supabase).

        Args:
            name: Name of the collection (becomes table name)
            metadata: Optional metadata (not currently used)

        Returns:
            SupabaseCollection: Wrapped Supabase table

        Raises:
            ValueError: If table already exists
            RuntimeError: If table creation fails

        """
        try:
            # Create table with pgvector column
            # Table schema:
            # - id: text (primary key)
            # - document: text
            # - embedding: vector(dimension)
            # - metadata: jsonb

            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {name} (
                    id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    embedding vector({self.dimension}) NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb
                );
            """

            self._client.rpc("exec_sql", {"query": create_table_sql}).execute()

            # Create index for vector similarity search
            create_index_sql = f"""
                CREATE INDEX IF NOT EXISTS {name}_embedding_idx
                ON {name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """

            self._client.rpc("exec_sql", {"query": create_index_sql}).execute()

            return SupabaseCollection(self._client, name, self.dimension)

        except Exception as e:
            if "already exists" in str(e).lower():
                raise ValueError(f"Collection '{name}' already exists") from e
            raise RuntimeError(f"Failed to create collection: {e}") from e

    def get_collection(self, name: str) -> Collection:
        """Get an existing collection (table).

        Args:
            name: Name of the collection/table

        Returns:
            SupabaseCollection: Wrapped Supabase table

        Raises:
            ValueError: If table does not exist

        """
        # Check if table exists
        try:
            # Simple query to check table existence
            self._client.table(name).select("id").limit(1).execute()
            return SupabaseCollection(self._client, name, self.dimension)
        except Exception as e:
            if "does not exist" in str(e).lower():
                raise ValueError(f"Collection '{name}' does not exist") from e
            raise RuntimeError(f"Failed to get collection: {e}") from e

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """Get or create a collection (table).

        Args:
            name: Name of the collection/table
            metadata: Optional metadata

        Returns:
            SupabaseCollection: Wrapped Supabase table

        """
        try:
            return self.get_collection(name)
        except ValueError:
            return self.create_collection(name, metadata)

    def delete_collection(self, name: str) -> None:
        """Delete a collection (table).

        Args:
            name: Name of the collection/table to delete

        Raises:
            ValueError: If table does not exist
            RuntimeError: If deletion fails

        """
        try:
            drop_table_sql = f"DROP TABLE IF EXISTS {name};"
            self._client.rpc("exec_sql", {"query": drop_table_sql}).execute()
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection '{name}': {e}") from e

    def list_collections(self) -> List[str]:
        """List all collections (tables).

        Returns:
            List[str]: List of table names

        """
        try:
            # Query information_schema to get table names
            list_tables_sql = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                AND table_name NOT LIKE 'pg_%'
                AND table_name NOT IN ('spatial_ref_sys');
            """

            result = self._client.rpc("exec_sql", {"query": list_tables_sql}).execute()

            if hasattr(result, "data") and result.data:
                return [row.get("table_name") for row in result.data]
            return []
        except (KeyError, AttributeError, TypeError):
            # Result structure unexpected or RPC not available - return empty list
            # This is non-critical, listing is best-effort
            return []
        except Exception as e:
            # Database errors, permission issues, or auth failures
            # Re-raise as these indicate real problems user should know about
            raise RuntimeError(f"Failed to list Supabase collections: {e}") from e


class SupabaseCollection(Collection):
    """Supabase implementation of Collection interface.

    This class wraps a Supabase table with pgvector support and provides
    collection operations compatible with the ChromaDB adapter interface.
    """

    def __init__(self, client, table_name: str, dimension: int):
        """Initialize Supabase collection wrapper.

        Args:
            client: Supabase client instance
            table_name: Name of the table
            dimension: Expected embedding dimension

        """
        self._client = client
        self._table_name = table_name
        self._dimension = dimension

    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents with embeddings to Supabase.

        Args:
            ids: Unique identifiers for each document
            documents: Text content of documents
            embeddings: Vector embeddings for each document
            metadatas: Optional metadata for each document

        Raises:
            ValueError: If input lists have different lengths
            RuntimeError: If Supabase operation fails

        """
        try:
            # Validate input lengths
            if not (len(ids) == len(documents) == len(embeddings)):
                raise ValueError(
                    "ids, documents, and embeddings must have the same length"
                )
            if metadatas and len(metadatas) != len(ids):
                raise ValueError("metadatas must have the same length as ids")

            # Prepare rows for insertion
            rows = []
            for i, (id_, doc, embedding) in enumerate(zip(ids, documents, embeddings)):
                metadata = metadatas[i] if metadatas else {}

                rows.append({
                    "id": id_,
                    "document": doc,
                    "embedding": embedding,  # pgvector handles list conversion
                    "metadata": metadata,
                })

            # Insert rows (upsert to handle duplicates)
            self._client.table(self._table_name).upsert(rows).execute()

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to Supabase: {e}") from e

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query Supabase for similar documents using pgvector.

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
            RuntimeError: If Supabase operation fails

        """
        try:
            if query_embeddings is None:
                raise ValueError(
                    "query_embeddings required for Supabase adapter. "
                    "Embedding generation should happen before calling query()."
                )

            # Supabase only supports single query at a time
            if len(query_embeddings) != 1:
                raise ValueError(
                    "Supabase adapter only supports single query at a time"
                )

            query_vector = query_embeddings[0]

            # Use RPC function for similarity search
            # We need to create this function in Supabase
            result = self._client.rpc(
                "match_documents",
                {
                    "query_embedding": query_vector,
                    "match_threshold": 0.0,
                    "match_count": n_results,
                    "table_name": self._table_name,
                }
            ).execute()

            # Transform to ChromaDB format
            ids = [[]]
            documents = [[]]
            metadatas = [[]]
            distances = [[]]

            for row in result.data:
                ids[0].append(row["id"])
                documents[0].append(row["document"])
                metadatas[0].append(row.get("metadata", {}))
                # pgvector returns similarity, convert to distance
                distances[0].append(1.0 - row.get("similarity", 0.0))

            return {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "distances": distances,
            }

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Supabase query failed: {e}") from e

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get documents from Supabase.

        Args:
            ids: Optional list of IDs to retrieve
            where: Optional metadata filter
            limit: Optional maximum number of results
            offset: Optional number of results to skip
            include: Optional list of fields to include

        Returns:
            Dict: Documents in ChromaDB-compatible format

        Raises:
            RuntimeError: If Supabase operation fails

        """
        try:
            query = self._client.table(self._table_name).select("*")
            query = self._build_query_filters(query, ids, where)
            query = self._apply_pagination(query, limit, offset)

            result = query.execute()
            return self._format_get_results(result.data)

        except Exception as e:
            raise RuntimeError(f"Supabase get operation failed: {e}") from e

    def _build_query_filters(
        self,
        query,
        ids: Optional[List[str]],
        where: Optional[Dict[str, Any]],
    ):
        """Build query filters for IDs and metadata.

        Args:
            query: Supabase query object
            ids: Optional list of IDs to filter
            where: Optional metadata filter

        Returns:
            Modified query object with filters applied

        """
        if ids:
            query = query.in_("id", ids)

        if where:
            # Apply metadata filters
            for key, value in where.items():
                if isinstance(value, dict) and "$in" in value:
                    query = query.in_(f"metadata->{key}", value["$in"])
                else:
                    query = query.eq(f"metadata->{key}", value)

        return query

    def _apply_pagination(
        self,
        query,
        limit: Optional[int],
        offset: Optional[int],
    ):
        """Apply pagination to query.

        Args:
            query: Supabase query object
            limit: Optional maximum number of results
            offset: Optional number of results to skip

        Returns:
            Modified query object with pagination applied

        """
        if limit:
            query = query.limit(limit)

        if offset:
            query = query.offset(offset)

        return query

    def _format_get_results(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format query results into ChromaDB-compatible format.

        Args:
            data: Raw query results from Supabase

        Returns:
            Dict: Formatted results with ids, documents, and metadatas

        """
        ids_list = []
        docs_list = []
        metadatas_list = []

        for row in data:
            ids_list.append(row["id"])
            docs_list.append(row["document"])
            metadatas_list.append(row.get("metadata", {}))

        return {
            "ids": ids_list,
            "documents": docs_list,
            "metadatas": metadatas_list,
        }

    def count(self) -> int:
        """Get the total number of documents in the table.

        Returns:
            int: Number of documents

        Raises:
            RuntimeError: If Supabase operation fails

        """
        try:
            result = self._client.table(self._table_name).select(
                "id", count="exact"
            ).execute()
            return result.count if hasattr(result, "count") else 0
        except Exception as e:
            raise RuntimeError(f"Supabase count operation failed: {e}") from e

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete documents from Supabase.

        Args:
            ids: Optional list of IDs to delete
            where: Optional metadata filter for deletion

        Raises:
            ValueError: If neither ids nor where is provided
            RuntimeError: If Supabase operation fails

        """
        try:
            if ids is None and where is None:
                raise ValueError("Either ids or where must be provided")

            query = self._client.table(self._table_name).delete()

            if ids:
                query = query.in_("id", ids)
            elif where:
                for key, value in where.items():
                    query = query.eq(f"metadata->{key}", value)

            query.execute()

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Supabase delete operation failed: {e}") from e

    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update existing documents in Supabase.

        Args:
            ids: IDs of documents to update
            documents: Optional new document texts
            embeddings: Optional new embeddings
            metadatas: Optional new metadata

        Raises:
            ValueError: If IDs are invalid
            RuntimeError: If Supabase operation fails

        """
        try:
            self._validate_update_inputs(ids)

            # Build and execute updates for each document
            for i, id_ in enumerate(ids):
                update_data = self._prepare_update_payload(
                    i, documents, embeddings, metadatas
                )
                if update_data:
                    self._execute_update(id_, update_data)

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Supabase update operation failed: {e}") from e

    def _validate_update_inputs(self, ids: List[str]) -> None:
        """Validate update inputs before processing.

        Args:
            ids: IDs of documents to update

        Raises:
            ValueError: If no documents found for given IDs

        """
        existing = self.get(ids=ids)
        if not existing["ids"]:
            raise ValueError(f"No documents found for IDs: {ids}")

    def _prepare_update_payload(
        self,
        index: int,
        documents: Optional[List[str]],
        embeddings: Optional[List[List[float]]],
        metadatas: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Prepare update payload for a single document.

        Args:
            index: Index of the document being updated
            documents: Optional new document texts
            embeddings: Optional new embeddings
            metadatas: Optional new metadata

        Returns:
            Dict[str, Any]: Update payload with fields to update

        """
        update_data = {}

        if documents and index < len(documents):
            update_data["document"] = documents[index]

        if embeddings and index < len(embeddings):
            update_data["embedding"] = embeddings[index]

        if metadatas and index < len(metadatas):
            update_data["metadata"] = metadatas[index]

        return update_data

    def _execute_update(self, id_: str, update_data: Dict[str, Any]) -> None:
        """Execute update operation for a single document.

        Args:
            id_: Document ID to update
            update_data: Fields to update

        """
        self._client.table(self._table_name).update(
            update_data
        ).eq("id", id_).execute()
