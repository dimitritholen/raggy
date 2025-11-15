"""Factory for creating embedding providers based on configuration."""

from typing import Any, Dict

from .openai_provider import OpenAIProvider
from .provider import EmbeddingProvider
from .sentence_transformers_provider import SentenceTransformersProvider


def create_embedding_provider(config: Dict[str, Any]) -> EmbeddingProvider:
    """Create an embedding provider based on configuration.

    Args:
        config: Embedding configuration dictionary with structure:
            {
                "provider": "sentence-transformers" | "openai",
                "sentenceTransformers": {"model": "..."},
                "openai": {"apiKey": "...", "model": "..."}
            }

    Returns:
        EmbeddingProvider: Configured embedding provider instance

    Raises:
        ValueError: If provider is unknown or configuration is invalid
        RuntimeError: If provider initialization fails

    Example:
        >>> config = {
        ...     "provider": "openai",
        ...     "openai": {
        ...         "apiKey": "sk-...",
        ...         "model": "text-embedding-3-small"
        ...     }
        ... }
        >>> provider = create_embedding_provider(config)

    """
    provider_type = config.get("provider", "sentence-transformers")

    if provider_type == "sentence-transformers":
        st_config = config.get("sentenceTransformers", {})
        model_name = st_config.get("model", "all-MiniLM-L6-v2")
        device = st_config.get("device", "cpu")

        return SentenceTransformersProvider(
            model_name=model_name,
            device=device
        )

    elif provider_type == "openai":
        openai_config = config.get("openai", {})

        if not openai_config:
            raise ValueError(
                "OpenAI configuration missing. Please provide 'openai' config with 'apiKey' and 'model'."
            )

        api_key = openai_config.get("apiKey")
        if not api_key:
            raise ValueError(
                "OpenAI API key missing. Please set 'embedding.openai.apiKey' in .raggy.json "
                "or use environment variable: ${OPENAI_API_KEY}"
            )

        model = openai_config.get("model", "text-embedding-3-small")

        return OpenAIProvider(
            api_key=api_key,
            model=model
        )

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_type}. "
            f"Supported providers: sentence-transformers, openai"
        )
