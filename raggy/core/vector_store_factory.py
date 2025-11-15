"""Factory for creating vector store adapters based on configuration."""

from typing import Any, Dict

from .chromadb_adapter import ChromaDBAdapter
from .database_interface import VectorDatabase


def create_vector_store(config: Dict[str, Any]) -> VectorDatabase:
    """Create a vector store adapter based on configuration.

    Args:
        config: Vector store configuration dictionary with structure:
            {
                "provider": "chromadb" | "pinecone" | "supabase",
                "chromadb": {"path": "..."},
                "pinecone": {"apiKey": "...", "environment": "...", "indexName": "..."},
                "supabase": {"url": "...", "apiKey": "...", "tableName": "..."}
            }

    Returns:
        VectorDatabase: Configured vector store adapter instance

    Raises:
        ValueError: If provider is unknown or configuration is invalid
        RuntimeError: If adapter initialization fails

    Example:
        >>> config = {
        ...     "provider": "chromadb",
        ...     "chromadb": {
        ...         "path": "./vectordb"
        ...     }
        ... }
        >>> vector_store = create_vector_store(config)

    """
    provider_type = config.get("provider", "chromadb")

    if provider_type == "chromadb":
        chromadb_config = config.get("chromadb", {})
        path = chromadb_config.get("path", "./vectordb")

        return ChromaDBAdapter(path=path)

    elif provider_type == "pinecone":
        try:
            from .pinecone_adapter import PineconeAdapter
        except ImportError as e:
            raise ImportError(
                "Pinecone adapter requires pinecone-client. "
                "Install with: pip install pinecone-client"
            ) from e

        pinecone_config = config.get("pinecone", {})

        if not pinecone_config:
            raise ValueError(
                "Pinecone configuration missing. Please provide 'pinecone' config with "
                "'apiKey', 'environment', and 'indexName'."
            )

        api_key = pinecone_config.get("apiKey")
        if not api_key:
            raise ValueError(
                "Pinecone API key missing. Please set 'vectorStore.pinecone.apiKey' in .raggy.json "
                "or use environment variable: ${PINECONE_API_KEY}"
            )

        environment = pinecone_config.get("environment")
        if not environment:
            raise ValueError(
                "Pinecone environment missing. Please set 'vectorStore.pinecone.environment' "
                "(e.g., 'us-east-1-aws')"
            )

        index_name = pinecone_config.get("indexName", "raggy-index")
        dimension = pinecone_config.get("dimension", 384)

        return PineconeAdapter(
            api_key=api_key,
            environment=environment,
            index_name=index_name,
            dimension=dimension,
        )

    elif provider_type == "supabase":
        try:
            from .supabase_adapter import SupabaseAdapter
        except ImportError as e:
            raise ImportError(
                "Supabase adapter requires supabase package. "
                "Install with: pip install supabase"
            ) from e

        supabase_config = config.get("supabase", {})

        if not supabase_config:
            raise ValueError(
                "Supabase configuration missing. Please provide 'supabase' config with "
                "'url' and 'apiKey'."
            )

        url = supabase_config.get("url")
        if not url:
            raise ValueError(
                "Supabase URL missing. Please set 'vectorStore.supabase.url' in .raggy.json "
                "or use environment variable: ${SUPABASE_URL}"
            )

        api_key = supabase_config.get("apiKey")
        if not api_key:
            raise ValueError(
                "Supabase API key missing. Please set 'vectorStore.supabase.apiKey' in .raggy.json "
                "or use environment variable: ${SUPABASE_ANON_KEY}"
            )

        dimension = supabase_config.get("dimension", 384)

        return SupabaseAdapter(
            url=url,
            api_key=api_key,
            dimension=dimension,
        )

    else:
        raise ValueError(
            f"Unknown vector store provider: {provider_type}. "
            f"Supported providers: chromadb, pinecone, supabase"
        )
