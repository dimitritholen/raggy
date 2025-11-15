"""Raggy - Universal RAG system for document search and retrieval.

This package provides:
- UniversalRAG: Main RAG system for document search and retrieval
- Memory: AI development memory system for context persistence
- remember/recall: Convenience functions for quick memory operations

Example:
    >>> from raggy import UniversalRAG, Memory
    >>>
    >>> # Document search
    >>> rag = UniversalRAG(docs_dir="./docs")
    >>> results = rag.search("machine learning algorithms")
    >>>
    >>> # Development memory
    >>> memory = Memory(db_dir="./vectordb")
    >>> mem_id = memory.add(
    ...     "Decided to use ChromaDB for vector storage",
    ...     memory_type="decision",
    ...     tags=["architecture", "database"]
    ... )
    >>> results = memory.search("database decisions")

"""

from raggy.cli.factory import CommandFactory
from raggy.config.loader import load_config
from raggy.core.database import DatabaseManager
from raggy.core.document import DocumentProcessor
from raggy.core.memory import Memory, recall, remember
from raggy.core.rag import UniversalRAG
from raggy.core.search import SearchEngine
from raggy.query.processor import QueryProcessor
from raggy.scoring.bm25 import BM25Scorer
from raggy.scoring.normalization import (
    interpret_score,
    normalize_cosine_distance,
    normalize_hybrid_score,
)
from raggy.setup.dependencies import install_if_missing, setup_dependencies
from raggy.setup.environment import setup_environment
from raggy.utils.updates import check_for_updates

__version__ = "2.0.0"

__all__ = [
    # Core RAG system
    "UniversalRAG",
    "SearchEngine",
    "DatabaseManager",
    "DocumentProcessor",
    # Memory system (new in 2.0)
    "Memory",
    "remember",
    "recall",
    # Scoring and normalization
    "normalize_cosine_distance",
    "normalize_hybrid_score",
    "interpret_score",
    "BM25Scorer",
    # Query processing
    "QueryProcessor",
    # CLI and configuration
    "CommandFactory",
    "load_config",
    # Setup utilities
    "setup_environment",
    "setup_dependencies",
    "install_if_missing",
    "check_for_updates",
]
