"""Shared test fixtures for raggy testing."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import sys
import os

# Add parent directory to path so we can import raggy
sys.path.insert(0, str(Path(__file__).parent.parent))

import raggy


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_docs_dir(temp_dir: Path) -> Path:
    """Create a temporary docs directory with sample files."""
    docs_dir = temp_dir / "docs"
    docs_dir.mkdir()
    return docs_dir


@pytest.fixture
def sample_md_content() -> str:
    """Sample markdown content for testing."""
    return """# Test Document

This is a test markdown document for testing raggy functionality.

## Features

- Feature 1: Text extraction
- Feature 2: Chunking
- Feature 3: Search

## API Documentation

The API provides the following methods:

### Search Method

```python
def search(query: str) -> List[Dict[str, Any]]:
    pass
```

### Configuration

The system supports various configuration options:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `model_name`: Embedding model to use

## Conclusion

This document contains enough content to test various chunking and search scenarios.
"""


@pytest.fixture
def sample_txt_content() -> str:
    """Sample text content for testing."""
    return """This is a plain text document for testing.

It contains multiple paragraphs with various technical terms like API, machine learning, 
user interface, and configuration settings.

The document discusses various aspects of software development including:
- Code quality
- Testing strategies
- Documentation practices
- Performance optimization

This content will be useful for testing query expansion and keyword matching.
"""


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "search": {
            "hybrid_weight": 0.7,
            "chunk_size": 500,  # Smaller for testing
            "chunk_overlap": 100,
            "rerank": True,
            "show_scores": True,
            "context_chars": 200,
            "max_results": 5,
            "expansions": {
                "api": ["api", "application programming interface"],
                "ml": ["ml", "machine learning"],
                "test": ["test", "testing", "unit test"]
            }
        },
        "models": {
            "default": "all-MiniLM-L6-v2",
            "fast": "paraphrase-MiniLM-L3-v2"
        },
        "chunking": {
            "smart": False,  # Disable for predictable testing
            "preserve_headers": True,
            "min_chunk_size": 100,
            "max_chunk_size": 800
        }
    }


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model that returns predictable embeddings."""
    class MockEmbeddingModel:
        def __init__(self, model_name: str):
            self.model_name = model_name
        
        def encode(self, texts, show_progress_bar=False):
            """Return mock embeddings based on text length."""
            import numpy as np
            embeddings = []
            for text in texts:
                # Create deterministic embeddings based on text content
                # Use hash of text to ensure consistency
                seed = hash(text) % (2**31)  # Ensure positive seed
                np.random.seed(seed)
                # Create 384-dimensional embeddings (typical for MiniLM)
                embedding = np.random.normal(0, 1, 384)
                # Normalize to unit vector
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            return np.array(embeddings)
    
    return MockEmbeddingModel


@pytest.fixture
def sample_documents(sample_docs_dir: Path, sample_md_content: str, sample_txt_content: str) -> Path:
    """Create sample documents for testing."""
    # Create markdown file
    md_file = sample_docs_dir / "test_doc.md"
    md_file.write_text(sample_md_content, encoding="utf-8")
    
    # Create text file
    txt_file = sample_docs_dir / "test_notes.txt"
    txt_file.write_text(sample_txt_content, encoding="utf-8")
    
    # Create a README file
    readme_content = """# Project README

This is a sample README file for testing document processing.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the application with:

```bash
python app.py
```
"""
    readme_file = sample_docs_dir / "README.md"
    readme_file.write_text(readme_content, encoding="utf-8")
    
    return sample_docs_dir


@pytest.fixture
def bm25_sample_documents() -> list:
    """Sample documents for BM25 testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog outran a quick fox",
        "The dog was lazy but the fox was quick",
        "Machine learning algorithms can process natural language",
        "Natural language processing uses machine learning techniques",
        "API documentation should be clear and comprehensive",
        "The application programming interface provides REST endpoints"
    ]


@pytest.fixture
def query_processor_test_cases() -> Dict[str, Dict[str, Any]]:
    """Test cases for query processor."""
    return {
        "simple_keyword": {
            "query": "machine learning",
            "expected_type": "keyword",
            "expected_terms": ["machine", "learning"]
        },
        "quoted_phrase": {
            "query": '"exact phrase"',
            "expected_type": "exact",
            "expected_boost": True
        },
        "question": {
            "query": "How does machine learning work?",
            "expected_type": "question",
            "expected_terms": ["how", "does", "machine", "learning", "work"]
        },
        "boolean_query": {
            "query": "machine learning AND algorithms",
            "expected_type": "boolean",
            "expected_must_have": ["machine"]
        },
        "negative_query": {
            "query": "machine learning -deep",
            "expected_type": "boolean",
            "expected_must_not": ["deep"]
        },
        "expandable_term": {
            "query": "api documentation",
            "expected_expansion": True,
            "expected_contains": "application programming interface"
        }
    }


# Environment setup for testing
os.environ.setdefault("RAGGY_TEST_MODE", "true")