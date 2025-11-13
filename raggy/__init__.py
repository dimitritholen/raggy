"""Raggy - Universal RAG system for document search and retrieval."""

from raggy.cli.factory import CommandFactory
from raggy.config.loader import load_config
from raggy.core.database import DatabaseManager
from raggy.core.document import DocumentProcessor
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
    "UniversalRAG",
    "SearchEngine",
    "DatabaseManager",
    "DocumentProcessor",
    "normalize_cosine_distance",
    "normalize_hybrid_score",
    "interpret_score",
    "BM25Scorer",
    "QueryProcessor",
    "CommandFactory",
    "load_config",
    "setup_environment",
    "setup_dependencies",
    "install_if_missing",
    "check_for_updates",
]