"""Configuration constants for the RAG system."""

from typing import Dict, Any

# Version information
__version__ = "2.0.0"

# File reading constants
CHUNK_READ_SIZE = 8192  # 8KB chunks for file reading
MAX_CACHE_SIZE = 1000  # Maximum number of cached embeddings
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB
SESSION_CACHE_HOURS = 24  # Hours before update check
UPDATE_TIMEOUT_SECONDS = 2  # API timeout for update checks

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RESULTS = 5
DEFAULT_CONTEXT_CHARS = 200
DEFAULT_HYBRID_WEIGHT = 0.7

# File type constants
SUPPORTED_EXTENSIONS = [".md", ".pdf", ".docx", ".txt"]
GLOB_PATTERNS = ["**/*.md", "**/*.pdf", "**/*.docx", "**/*.txt"]

# Model presets
FAST_MODEL = "paraphrase-MiniLM-L3-v2"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ACCURATE_MODEL = "all-mpnet-base-v2"

# Default configuration structure
DEFAULT_CONFIG: Dict[str, Any] = {
    "model": DEFAULT_MODEL,
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "default_results": DEFAULT_RESULTS,
    "context_chars": DEFAULT_CONTEXT_CHARS,
    "excluded_dirs": [
        # Version control and dependencies
        ".git", "node_modules", ".venv", "venv", "__pycache__",
        # Build and distribution
        "dist", "build", "*.egg-info",
        # IDEs and editors
        ".idea", ".vscode",
        # Misc
        "chroma_db", "vectordb", ".chromadb", ".raggydb",
        ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ],
    "supported_extensions": SUPPORTED_EXTENSIONS,
    "search": {
        "hybrid_weight": DEFAULT_HYBRID_WEIGHT,
        "expand_queries": False,
        "boost_exact": True,
    },
    "updates": {
        "check_enabled": True,
        "github_repo": "dimitritholen/raggy",
    }
}