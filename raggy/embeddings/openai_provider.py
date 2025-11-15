"""OpenAI embedding provider.

This module provides a cloud-based embedding provider using OpenAI's
text-embedding models via API.
"""

from typing import List, Union

import numpy as np

from .provider import EmbeddingProvider


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding models.

    This provider uses OpenAI's API to generate embeddings using models like
    text-embedding-3-small, text-embedding-3-large, or text-embedding-ada-002.
    """

    # Model dimensions (cached to avoid API calls)
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            model: Model name (text-embedding-3-small, text-embedding-3-large, etc.)

        Raises:
            ImportError: If openai package not installed
            ValueError: If model is not supported
            RuntimeError: If OpenAI initialization fails

        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            ) from e

        if model not in self.MODEL_DIMENSIONS:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported models: {list(self.MODEL_DIMENSIONS.keys())}"
            )

        self.api_key = api_key
        self.model = model
        self._dimension = self.MODEL_DIMENSIONS[model]

        try:
            self._client = OpenAI(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 100,  # OpenAI allows up to 2048 texts per request
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode text(s) into embeddings using OpenAI API.

        Args:
            texts: Single text string or list of texts to encode
            batch_size: Batch size for API requests (max 2048 for OpenAI)
            show_progress: Whether to show progress (not implemented for OpenAI)

        Returns:
            np.ndarray: Embeddings array of shape (num_texts, embedding_dim)

        Raises:
            ValueError: If texts is empty or invalid
            RuntimeError: If API call fails

        """
        if not texts:
            raise ValueError("texts cannot be empty")

        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]

        try:
            all_embeddings = []

            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Call OpenAI API
                response = self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                )

                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            # Convert to numpy array
            return np.array(all_embeddings, dtype=np.float32)

        except Exception as e:
            # Check for common errors
            error_msg = str(e).lower()
            if "api key" in error_msg or "auth" in error_msg:
                raise RuntimeError(
                    f"OpenAI authentication failed. Please check your API key: {e}"
                ) from e
            if "rate limit" in error_msg:
                raise RuntimeError(
                    f"OpenAI rate limit exceeded. Please try again later: {e}"
                ) from e
            if "quota" in error_msg:
                raise RuntimeError(
                    f"OpenAI quota exceeded. Please check your usage: {e}"
                ) from e
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    def get_dimension(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            int: Embedding dimension

        """
        return self._dimension

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            str: Model name

        """
        return self.model
