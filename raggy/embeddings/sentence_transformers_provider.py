"""Sentence Transformers embedding provider.

This module provides a local embedding provider using the sentence-transformers
library for offline, privacy-preserving embeddings.
"""

from typing import List, Union

import numpy as np

from .provider import EmbeddingProvider


class SentenceTransformersProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers.

    This provider uses the sentence-transformers library to generate embeddings
    locally without requiring API calls or internet connectivity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """Initialize sentence-transformers provider.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on ("cpu" or "cuda")

        Raises:
            ImportError: If sentence-transformers not installed
            RuntimeError: If model loading fails

        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e

        self.model_name = model_name
        self.device = device

        try:
            self._model = SentenceTransformer(model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}") from e

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode text(s) into embeddings.

        Args:
            texts: Single text string or list of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            np.ndarray: Embeddings array of shape (num_texts, embedding_dim)

        Raises:
            ValueError: If texts is empty or invalid
            RuntimeError: If encoding fails

        """
        if not texts:
            raise ValueError("texts cannot be empty")

        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]

        try:
            return self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to encode texts: {e}") from e

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
        return self.model_name
