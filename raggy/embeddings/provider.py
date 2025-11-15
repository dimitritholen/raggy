"""Abstract interface for embedding providers.

This module defines the standard interface that all embedding providers
must implement, allowing for pluggable local and cloud embedding models.
"""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding providers (local models, OpenAI, etc.) must implement
    this interface to ensure compatibility with Raggy's RAG system.
    """

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode text(s) into embeddings.

        Args:
            texts: Single text string or list of texts to encode
            batch_size: Batch size for processing (used by some providers)
            show_progress: Whether to show progress bar

        Returns:
            np.ndarray: Embeddings array of shape (num_texts, embedding_dim)
                For single text input, returns shape (1, embedding_dim)

        Raises:
            ValueError: If texts is empty or invalid
            RuntimeError: If encoding fails

        """

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider.

        Returns:
            int: Embedding dimension (e.g., 384, 1536, 3072)

        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name/identifier of the embedding model.

        Returns:
            str: Model name (e.g., "all-MiniLM-L6-v2", "text-embedding-3-small")

        """

    def __repr__(self) -> str:
        """String representation of provider."""
        return f"{self.__class__.__name__}(model={self.get_model_name()}, dim={self.get_dimension()})"
