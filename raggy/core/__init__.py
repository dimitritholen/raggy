"""Core business logic for the RAG system."""

from .chromadb_adapter import ChromaCollection, ChromaDBAdapter
from .database import DatabaseManager
from .database_interface import Collection, VectorDatabase
from .document import DocumentProcessor
from .rag import UniversalRAG
from .search import SearchEngine

__all__ = [
    # Main components
    "UniversalRAG",
    "DatabaseManager",
    "DocumentProcessor",
    "SearchEngine",
    # Database interfaces
    "VectorDatabase",
    "Collection",
    # Database implementations
    "ChromaDBAdapter",
    "ChromaCollection",
]