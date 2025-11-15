"""Embedding providers for Raggy.

This module provides a pluggable embedding provider system supporting
both local models (sentence-transformers) and cloud APIs (OpenAI).
"""

from .factory import create_embedding_provider
from .openai_provider import OpenAIProvider
from .provider import EmbeddingProvider
from .sentence_transformers_provider import SentenceTransformersProvider

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformersProvider",
    "OpenAIProvider",
    "create_embedding_provider",
]
