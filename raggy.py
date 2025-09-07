#!/usr/bin/env python3
"""
Universal ChromaDB RAG Setup Script v2.0.0
Enhanced with hybrid search, smart chunking, and normalized scoring.

Drop this into any project and run:
  python raggy.py build                       # Build/index all docs with smart chunking
  python raggy.py rebuild --fast              # Clean rebuild with faster model
  python raggy.py search "your query"         # Semantic search with normalized scores
  python raggy.py search "exact term" --hybrid # Hybrid semantic+keyword search
  python raggy.py search "api" --expand        # Query expansion with synonyms
  python raggy.py interactive --quiet         # Interactive search mode
  python raggy.py status                      # Database stats with model info
  python raggy.py optimize                    # Benchmark and tune search modes
  python raggy.py search "query" --json       # Enhanced JSON output with scores

Key Features:
• Hybrid Search: Combines semantic + BM25 keyword ranking for exact matches
• Smart Chunking: Markdown-aware chunking preserving document structure
• Normalized Scoring: 0-1 scores with human-readable labels (Excellent/Good/Fair/Poor)
• Query Processing: Automatic expansion of domain terms (api → application programming interface)
• Model Presets: --model-preset fast/balanced/multilingual/accurate
• Config Support: Optional raggy_config.yaml for customization
• Multilingual: Enhanced Dutch/English mixed content support
• Backward Compatible: All v1.x commands work unchanged
"""

import os
import sys
import glob
import hashlib
import argparse
import json
import time
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import subprocess
import importlib.util
from collections import defaultdict, Counter

# Configure UTF-8 encoding for cross-platform compatibility
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Try to configure console for UTF-8 on Windows
    try:
        import codecs

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except:
        pass


# Cross-platform emoji/symbol support
def get_symbols():
    """Get appropriate symbols based on platform/terminal support"""
    try:
        # Test if terminal supports unicode
        test = "🔍"
        print(test, end="")
        print("\b \b", end="")  # backspace and clear
        return {"search": "🔍", "found": "📋", "success": "✅", "bye": "👋"}
    except UnicodeEncodeError:
        return {
            "search": "[Search]",
            "found": "[Found]",
            "success": "[Success]",
            "bye": "[Bye]",
        }


SYMBOLS = get_symbols()


class BM25Scorer:
    """Lightweight BM25 implementation for keyword scoring"""

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_count = 0
        self.term_frequencies = []
        self.idf_scores = {}

    def fit(self, documents: List[str]):
        """Build BM25 index from documents"""
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.term_frequencies = []
        doc_term_counts = defaultdict(int)

        # Calculate term frequencies and document lengths
        for doc in documents:
            terms = self._tokenize(doc)
            self.doc_lengths.append(len(terms))

            term_freq = Counter(terms)
            self.term_frequencies.append(term_freq)

            # Count documents containing each term
            for term in set(terms):
                doc_term_counts[term] += 1

        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        )

        # Calculate IDF scores
        for term, doc_freq in doc_term_counts.items():
            self.idf_scores[term] = math.log(
                (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5)
            )

    def score(self, query: str, doc_index: int) -> float:
        """Calculate BM25 score for query against document"""
        if doc_index >= len(self.term_frequencies):
            return 0.0

        query_terms = self._tokenize(query)
        score = 0.0
        doc_length = self.doc_lengths[doc_index]
        term_freq = self.term_frequencies[doc_index]

        for term in query_terms:
            if term in term_freq:
                tf = term_freq[term]
                idf = self.idf_scores.get(term, 0)

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )
                score += idf * (numerator / denominator)

        return max(0, score)  # Ensure non-negative scores

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and extract alphanumeric sequences
        return re.findall(r"\b\w+\b", text.lower())


class QueryProcessor:
    """Enhanced query processing with expansion and operators"""

    def __init__(self, custom_expansions: Optional[Dict[str, List[str]]] = None):
        # Default expansions - can be overridden via config
        self.expansions = custom_expansions or {
            # Common technical terms
            "api": ["api", "application programming interface"],
            "ml": ["ml", "machine learning"],
            "ai": ["ai", "artificial intelligence"],
            "ui": ["ui", "user interface"],
            "ux": ["ux", "user experience"],
            # Can be extended via configuration file
        }

    def process(self, query: str) -> Dict[str, Any]:
        """Process query and return enhanced version with metadata"""
        original = query.strip()

        # Detect query type
        query_type = self._detect_type(original)

        # Handle exact phrase queries (quoted)
        if query_type == "exact":
            phrase = re.findall(r'"([^"]+)\"', original)[0]
            return {
                "processed": phrase,
                "original": original,
                "type": "exact",
                "boost_exact": True,
                "terms": [phrase],
            }

        # Expand terms
        expanded = self._expand_query(original)

        # Extract boolean operators
        must_have, must_not = self._extract_operators(expanded)

        return {
            "processed": expanded,
            "original": original,
            "type": query_type,
            "boost_exact": False,
            "must_have": must_have,
            "must_not": must_not,
            "terms": re.findall(r"\b\w+\b", expanded.lower()),
        }

    def _detect_type(self, query: str) -> str:
        """Detect query type"""
        if '"' in query:
            return "exact"
        elif any(
            word in query.lower()
            for word in ["how", "what", "why", "when", "where", "who"]
        ):
            return "question"
        elif " AND " in query.upper() or " OR " in query.upper() or " -" in query:
            return "boolean"
        else:
            return "keyword"

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms"""
        expanded = query.lower()
        for term, expansions in self.expansions.items():
            if term in expanded:
                # Add expansions as OR terms
                expansion_str = " OR ".join(expansions[1:])  # Skip the original term
                if expansion_str:
                    expanded = expanded.replace(term, f"({term} OR {expansion_str})")
        return expanded

    def _extract_operators(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract boolean operators"""
        must_have = []
        must_not = []

        # Extract negative terms (preceded by -)
        negative_terms = re.findall(r"-\w+", query)
        for term in negative_terms:
            must_not.append(term[1:])  # Remove the -

        # Extract AND terms
        and_terms = re.findall(r"\w+(?=\s+AND)", query, re.IGNORECASE)
        must_have.extend(and_terms)

        return must_have, must_not


class ScoringNormalizer:
    """Normalize and interpret similarity scores"""

    @staticmethod
    def normalize_cosine_distance(distance: float) -> float:
        """Convert cosine distance to similarity (0-1, higher is better)"""
        # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
        # Convert to similarity: 1 - (distance / 2)
        similarity = max(0, min(1, 1 - (distance / 2)))
        return similarity

    @staticmethod
    def normalize_hybrid_score(
        semantic_score: float, keyword_score: float, semantic_weight: float = 0.7
    ) -> float:
        """Combine semantic and keyword scores"""
        # Normalize keyword score to 0-1 range (assuming max BM25 ~10)
        normalized_keyword = min(1.0, keyword_score / 10.0)

        return (
            semantic_weight * semantic_score
            + (1 - semantic_weight) * normalized_keyword
        )

    @staticmethod
    def interpret_score(score: float) -> str:
        """Provide human-readable score interpretation"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load optional configuration file"""
    default_config = {
        "search": {
            "hybrid_weight": 0.7,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "rerank": True,
            "show_scores": True,
            "context_chars": 200,
            "max_results": 5,
            "expansions": {
                # Add domain-specific expansions here
                "api": ["api", "application programming interface"],
                "ml": ["ml", "machine learning"],
                "ai": ["ai", "artificial intelligence"],
                "ui": ["ui", "user interface"],
                "ux": ["ux", "user experience"],
            },
        },
        "models": {
            "default": "all-MiniLM-L6-v2",
            "fast": "paraphrase-MiniLM-L3-v2",
            "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "accurate": "all-mpnet-base-v2",
        },
        "chunking": {
            "smart": False,  # Disable by default to prevent issues
            "preserve_headers": True,
            "min_chunk_size": 300,
            "max_chunk_size": 1500,
        },
    }

    # Try to load config file
    config_file = Path(config_path or "raggy_config.yaml")
    if config_file.exists():
        try:
            import yaml

            with open(config_file, "r") as f:
                user_config = yaml.safe_load(f)

            # Merge with defaults
            def merge_configs(default, user):
                for key, value in user.items():
                    if (
                        key in default
                        and isinstance(default[key], dict)
                        and isinstance(value, dict)
                    ):
                        merge_configs(default[key], value)
                    else:
                        default[key] = value

            merge_configs(default_config, user_config)
        except ImportError:
            print("Note: PyYAML not installed, using default config")
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")

    return default_config


def get_cache_file():
    """Get path for dependency cache file"""
    return Path.cwd() / ".raggy_deps_cache.json"


def load_deps_cache():
    """Load dependency cache"""
    cache_file = get_cache_file()
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def save_deps_cache(cache):
    """Save dependency cache"""
    cache_file = get_cache_file()
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
    except:
        pass


def check_uv_available():
    """Check if uv is available"""
    try:
        subprocess.check_call(
            ["uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: uv is not available or not in PATH")
        print(
            "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
        )
        print("Or run: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False


def check_environment_setup():
    """Check if project environment is properly set up"""
    venv_path = Path(".venv")
    pyproject_path = Path("pyproject.toml")
    
    if not venv_path.exists():
        return False, "virtual_environment"
    
    if not pyproject_path.exists():
        return False, "pyproject"
    
    # Check if virtual environment is activated or can be used
    try:
        # Check if we can run python in the venv
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        
        if not python_exe.exists():
            return False, "invalid_venv"
    except:
        return False, "invalid_venv"
    
    return True, "ok"


def setup_environment(quiet: bool = False):
    """Set up the project environment from scratch"""
    if not quiet:
        print("🚀 Setting up raggy environment...")
    
    # Check if uv is available
    if not check_uv_available():
        return False
    
    # Create virtual environment
    venv_path = Path(".venv")
    if not venv_path.exists():
        if not quiet:
            print("Creating virtual environment...")
        try:
            subprocess.check_call(["uv", "venv"], stdout=subprocess.DEVNULL if quiet else None)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to create virtual environment: {e}")
            return False
    
    # Create minimal pyproject.toml if it doesn't exist
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        if not quiet:
            print("Creating pyproject.toml...")
        
        pyproject_content = """[project]
name = "raggy-project"
version = "0.1.0"
description = "RAG project using Universal ChromaDB RAG System"
requires-python = ">=3.8"
dependencies = [
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "PyPDF2>=3.0.0",
]

[project.optional-dependencies]
magic-win = ["python-magic-bin>=0.4.14"]
magic-unix = ["python-magic"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
        
        try:
            with open(pyproject_path, "w", encoding="utf-8") as f:
                f.write(pyproject_content)
        except Exception as e:
            print(f"ERROR: Failed to create pyproject.toml: {e}")
            return False
    
    # Install dependencies
    if not quiet:
        print("Installing dependencies...")
    
    try:
        # Install base dependencies
        subprocess.check_call([
            "uv", "pip", "install", 
            "chromadb>=0.4.0", 
            "sentence-transformers>=2.2.0", 
            "PyPDF2>=3.0.0"
        ], stdout=subprocess.DEVNULL if quiet else None)
        
        # Install platform-specific magic library
        if sys.platform == "win32":
            try:
                subprocess.check_call([
                    "uv", "pip", "install", "python-magic-bin>=0.4.14"
                ], stdout=subprocess.DEVNULL if quiet else None)
            except subprocess.CalledProcessError:
                if not quiet:
                    print("Warning: Could not install python-magic-bin. File type detection may be limited.")
        else:
            try:
                subprocess.check_call([
                    "uv", "pip", "install", "python-magic"
                ], stdout=subprocess.DEVNULL if quiet else None)
            except subprocess.CalledProcessError:
                if not quiet:
                    print("Warning: Could not install python-magic. File type detection may be limited.")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        print("You can try installing manually with: uv pip install chromadb sentence-transformers PyPDF2")
        return False
    
    # Create docs directory if it doesn't exist
    docs_path = Path("docs")
    if not docs_path.exists():
        docs_path.mkdir()
        if not quiet:
            print("Created docs/ directory - add your .md or .pdf files here")
    
    if not quiet:
        print("✅ Environment setup complete!")
        print("\nNext steps:")
        print("1. Add your documentation files to the docs/ directory")
        print("2. Run: python raggy.py build")
        print("3. Run: python raggy.py search \"your query\"")
    
    return True


def install_if_missing(packages: List[str], skip_cache: bool = False):
    """Auto-install required packages if missing using uv"""
    if not check_uv_available():
        sys.exit(1)

    # Check if environment is set up properly
    env_ok, env_issue = check_environment_setup()
    if not env_ok:
        if env_issue == "virtual_environment":
            print("ERROR: No virtual environment found.")
            print("Run 'python raggy.py init' to set up the project environment.")
        elif env_issue == "pyproject":
            print("ERROR: No pyproject.toml found.")
            print("Run 'python raggy.py init' to set up the project environment.")
        elif env_issue == "invalid_venv":
            print("ERROR: Invalid virtual environment found.")
            print("Delete .venv directory and run 'python raggy.py init' to recreate it.")
        sys.exit(1)

    # Load cache unless skipped
    cache = {} if skip_cache else load_deps_cache()
    cache_updated = False

    for package_spec in packages:
        # Extract just the package name (before any version specifiers)
        package_name = package_spec.split(">=")[0].split("==")[0].split("[")[0]

        # Check cache first
        if not skip_cache and package_name in cache.get("installed", {}):
            continue

        # Handle special cases for import names
        if package_name == "python-magic-bin":
            import_name = "magic"
        elif package_name == "PyPDF2":
            import_name = "PyPDF2"
        else:
            import_name = package_name.replace("-", "_")

        try:
            # Try to import the module
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                raise ImportError(f"No module named '{import_name}'")

            # Cache successful import
            if "installed" not in cache:
                cache["installed"] = {}
            cache["installed"][package_name] = time.time()
            cache_updated = True

        except ImportError:
            print(f"Installing {package_name}...")
            try:
                # Use uv pip install with the virtual environment
                subprocess.check_call(["uv", "pip", "install", package_spec])
                # Cache successful installation
                if "installed" not in cache:
                    cache["installed"] = {}
                cache["installed"][package_name] = time.time()
                cache_updated = True
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package_name}: {e}")
                if package_name == "python-magic-bin":
                    print("Trying alternative magic package...")
                    try:
                        subprocess.check_call(["uv", "pip", "install", "python-magic"])
                        cache["installed"][package_name] = time.time()
                        cache_updated = True
                    except:
                        print("Warning: Could not install python-magic. File type detection may be limited.")

    # Save updated cache
    if cache_updated:
        save_deps_cache(cache)


def setup_dependencies(skip_cache: bool = False, quiet: bool = False):
    """Setup dependencies with optional caching"""
    
    # Check if we're in a virtual environment and switch to it if needed
    env_ok, env_issue = check_environment_setup()
    if env_ok:
        # Ensure we're using the virtual environment's Python
        venv_path = Path(".venv")
        if sys.platform == "win32":
            venv_python = venv_path / "Scripts" / "python.exe"
        else:
            venv_python = venv_path / "bin" / "python"
        
        # If we're not already running in the venv, restart with venv python
        if str(venv_python.resolve()) != str(Path(sys.executable).resolve()):
            if not quiet:
                print("Switching to virtual environment...")
            # Re-run the current command with the venv python
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)
    
    required_packages = [
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "PyPDF2>=3.0.0",
    ]

    # Add platform-specific packages
    if sys.platform == "win32":
        required_packages.append("python-magic-bin>=0.4.14")
    else:
        required_packages.append("python-magic")

    if not quiet:
        print("Checking dependencies...")
    install_if_missing(required_packages, skip_cache)

    # Import after installation
    global chromadb, SentenceTransformer, PyPDF2, HAS_MAGIC, magic

    import chromadb
    from sentence_transformers import SentenceTransformer
    import PyPDF2

    # Optional import for file type detection
    try:
        import magic

        HAS_MAGIC = True
    except ImportError:
        HAS_MAGIC = False
        if not quiet:
            print(
                "Note: python-magic not available. Using file extensions for type detection."
            )


class UniversalRAG:
    def __init__(
        self,
        docs_dir: str = "./docs",
        db_dir: str = "./vectordb",
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        quiet: bool = False,
        config_path: Optional[str] = None,
    ):
        self.docs_dir = Path(docs_dir)
        self.db_dir = Path(db_dir)
        self.collection_name = "project_docs"
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.quiet = quiet

        # Load configuration
        self.config = load_config(config_path)

        # Initialize processors
        self.query_processor = QueryProcessor(
            self.config["search"].get("expansions", {})
        )
        self.scoring = ScoringNormalizer()

        # Lazy-loaded attributes
        self._client = None
        self._embedding_model = None
        self._bm25_scorer = None
        self._documents_cache = None

    @property
    def client(self):
        """Lazy-load ChromaDB client"""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.db_dir))
        return self._client

    @property
    def embedding_model(self):
        """Lazy-load embedding model"""
        if self._embedding_model is None:
            if not self.quiet:
                print(f"Loading embedding model ({self.model_name})...")
            self._embedding_model = SentenceTransformer(self.model_name)
        return self._embedding_model

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file for change detection"""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Warning: Could not extract text from {file_path}: {e}")
            return ""

    def _extract_text_from_md(self, file_path: Path) -> str:
        """Extract text from Markdown file"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return ""

    def _chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        smart: bool = True,
    ) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with optional smart chunking"""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        if smart and self.config["chunking"]["smart"]:
            return self._chunk_text_smart(text, chunk_size, overlap)
        else:
            return self._chunk_text_simple(text, chunk_size, overlap)

    def _chunk_text_simple(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[Dict[str, Any]]:
        """Simple chunking for backward compatibility"""
        if len(text) <= chunk_size:
            return [{"text": text, "metadata": {"chunk_type": "simple"}}]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in ".!?\n":
                        end = i + 1
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {"text": chunk_text, "metadata": {"chunk_type": "simple"}}
                )

            start = end - overlap

        return chunks

    def _chunk_text_smart(
        self, text: str, base_chunk_size: int, overlap: int
    ) -> List[Dict[str, Any]]:
        """Smart chunking with markdown awareness"""
        chunks = []

        # Split by major sections first (headers)
        sections = re.split(r"(^#{1,6}\s+.*$)", text, flags=re.MULTILINE)

        current_header = None
        current_content = ""

        for section in sections:
            if re.match(r"^#{1,6}\s+", section):
                # Process previous section if exists
                if current_content.strip():
                    section_chunks = self._process_section(
                        current_content, current_header, base_chunk_size, overlap
                    )
                    chunks.extend(section_chunks)

                # Start new section
                current_header = section.strip()
                current_content = ""
            else:
                current_content += section

        # Process final section
        if current_content.strip():
            section_chunks = self._process_section(
                current_content, current_header, base_chunk_size, overlap
            )
            chunks.extend(section_chunks)

        # If no headers found, fall back to simple chunking
        if not chunks:
            return self._chunk_text_simple(text, base_chunk_size, overlap)

        return chunks

    def _process_section(
        self, content: str, header: Optional[str], chunk_size: int, overlap: int
    ) -> List[Dict[str, Any]]:
        """Process a section with its header"""
        chunks = []
        content = content.strip()

        if not content:
            return chunks

        # Determine chunk size based on content type
        lines = content.split("\n")
        if any(line.strip().startswith(("-", "*", "1.")) for line in lines[:5]):
            # List content - use smaller chunks
            target_size = min(chunk_size, self.config["chunking"]["min_chunk_size"] * 2)
        else:
            # Regular content - use dynamic sizing
            target_size = min(
                max(len(content) // 3, self.config["chunking"]["min_chunk_size"]),
                self.config["chunking"]["max_chunk_size"],
            )

        # Include header in first chunk if preserving headers
        if header and self.config["chunking"]["preserve_headers"]:
            content = f"{header}\n\n{content}"

        # Split content into chunks
        if len(content) <= target_size:
            chunks.append(
                {
                    "text": content,
                    "metadata": {
                        "chunk_type": "smart",
                        "section_header": header,
                        "header_depth": len(re.findall(r"^#", header or "")),
                    },
                }
            )
        else:
            start = 0
            chunk_index = 0

            while start < len(content):
                end = start + target_size

                # Try to break at paragraph or sentence boundary
                if end < len(content):
                    # Look for paragraph breaks first
                    for i in range(end, max(start + target_size - 300, start), -1):
                        if i > start and content[i - 2 : i] == "\n\n":
                            end = i
                            break
                    else:
                        # Fall back to sentence breaks
                        for i in range(end, max(start + target_size - 200, start), -1):
                            if content[i] in ".!?\n":
                                end = i + 1
                                break

                chunk_text = content[start:end].strip()
                if chunk_text:
                    chunks.append(
                        {
                            "text": chunk_text,
                            "metadata": {
                                "chunk_type": "smart",
                                "section_header": header,
                                "header_depth": len(re.findall(r"^#", header or "")),
                                "section_chunk_index": chunk_index,
                            },
                        }
                    )
                    chunk_index += 1

                start = end - overlap

        return chunks

    def _find_documents(self) -> List[Path]:
        """Find all supported documents in docs directory"""
        if not self.docs_dir.exists():
            print(f"Creating docs directory: {self.docs_dir}")
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            print(f"Please add your documentation files to {self.docs_dir}")
            return []

        patterns = ["**/*.md", "**/*.pdf"]
        files = []

        for pattern in patterns:
            files.extend(self.docs_dir.glob(pattern))

        return sorted(files)

    def _process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single document into chunks"""
        if not self.quiet:
            print(f"Processing: {file_path.relative_to(self.docs_dir)}")

        try:
            # Extract text based on file type
            if file_path.suffix.lower() == ".pdf":
                text = self._extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() == ".md":
                text = self._extract_text_from_md(file_path)
            else:
                if not self.quiet:
                    print(f"Skipping unsupported file type: {file_path}")
                return []

            if not text.strip():
                if not self.quiet:
                    print(f"Warning: No text extracted from {file_path}")
                return []

            # Generate chunks
            chunk_data = self._chunk_text(text)

            # Create document entries
            documents = []
            file_hash = self._get_file_hash(file_path)

            for i, chunk_info in enumerate(chunk_data):
                doc_id = f"{file_path.stem}_{file_hash[:8]}_{i}"

                # Merge chunk metadata with file metadata
                metadata = {
                    "source": str(file_path.relative_to(self.docs_dir)),
                    "chunk_index": i,
                    "total_chunks": len(chunk_data),
                    "file_hash": file_hash,
                    "file_type": file_path.suffix.lower(),
                }
                metadata.update(chunk_info.get("metadata", {}))

                documents.append(
                    {"id": doc_id, "text": chunk_info["text"], "metadata": metadata}
                )

            return documents

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def build(self, force_rebuild: bool = False):
        """Build or update the vector database"""
        start_time = time.time()

        # Get or create collection
        try:
            if force_rebuild:
                try:
                    self.client.delete_collection(self.collection_name)
                    if not self.quiet:
                        print("Deleted existing collection")
                except:
                    pass

            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Project documentation embeddings"},
            )
        except Exception as e:
            print(f"Error with collection: {e}")
            return

        # Find documents
        files = self._find_documents()
        if not files:
            print("ERROR: No documents found in docs/ directory.")
            print("Solution: Add .md or .pdf files to the docs/ directory")
            print("Example: docs/readme.md, docs/guide.pdf")
            return

        if not self.quiet:
            print(f"Found {len(files)} documents")

        # Process each document
        all_documents = []
        for i, file_path in enumerate(files, 1):
            if not self.quiet:
                print(f"[{i}/{len(files)}] Processing {file_path.name}...")
            docs = self._process_document(file_path)
            all_documents.extend(docs)

        if not all_documents:
            print("ERROR: No content could be extracted from documents")
            print("This could mean:")
            print("- PDF files are corrupted or password-protected")
            print("- Markdown files are empty")
            print("- Files are not readable")
            print("Check your files and try again.")
            return

        if not self.quiet:
            print(f"Generated {len(all_documents)} text chunks")
            print("Generating embeddings...")

        # Generate embeddings and add to collection
        texts = [doc["text"] for doc in all_documents]
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=not self.quiet
        )

        # Add to ChromaDB
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[doc["metadata"] for doc in all_documents],
            ids=[doc["id"] for doc in all_documents],
        )

        elapsed = time.time() - start_time
        print(
            f"{SYMBOLS['success']} Successfully indexed {len(all_documents)} chunks from {len(files)} files"
        )
        print(f"Database saved to: {self.db_dir}")
        if not self.quiet:
            print(f"Build completed in {elapsed:.1f} seconds")

    def search(
        self,
        query: str,
        n_results: int = 5,
        hybrid: bool = False,
        expand_query: bool = False,
        show_scores: bool = None,
    ) -> List[Dict[str, Any]]:
        """Search the vector database with enhanced capabilities"""
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception as e:
            print("ERROR: Database collection not found.")
            print("This usually means you haven't indexed any documents yet.")
            print(f"Solution: Run 'python {sys.argv[0]} build' first")
            if not Path("docs").exists():
                print("Note: You'll also need to create a 'docs/' directory and add .md or .pdf files")
            return []

        try:
            # Process query
            if expand_query:
                query_info = self.query_processor.process(query)
                processed_query = query_info["processed"]
            else:
                query_info = {
                    "original": query,
                    "type": "keyword",
                    "boost_exact": False,
                }
                processed_query = query

            # Get semantic results
            results = collection.query(
                query_texts=[processed_query],
                n_results=(
                    n_results * 2 if hybrid else n_results
                ),  # Get more for hybrid filtering
            )

            formatted_results = []

            # Initialize BM25 scorer for hybrid search
            if hybrid and self._bm25_scorer is None:
                self._init_bm25_scorer(collection)

            for i in range(len(results["documents"][0])):
                distance = (
                    results["distances"][0][i] if "distances" in results else None
                )

                # Normalize semantic similarity score
                semantic_score = (
                    self.scoring.normalize_cosine_distance(distance)
                    if distance is not None
                    else 0
                )

                # Calculate keyword score if using hybrid search
                if hybrid and self._bm25_scorer:
                    keyword_score = self._bm25_scorer.score(query, i)
                    # Combine scores
                    final_score = self.scoring.normalize_hybrid_score(
                        semantic_score,
                        keyword_score,
                        self.config["search"]["hybrid_weight"],
                    )
                else:
                    keyword_score = 0
                    final_score = semantic_score

                # Apply exact match boost
                if (
                    query_info.get("boost_exact")
                    and query.lower() in results["documents"][0][i].lower()
                ):
                    final_score = min(1.0, final_score * 1.5)

                formatted_results.append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "semantic_score": semantic_score,
                        "keyword_score": keyword_score,
                        "final_score": final_score,
                        "score_interpretation": self.scoring.interpret_score(
                            final_score
                        ),
                        "distance": distance,  # Keep for backward compatibility
                        "similarity": final_score,  # Keep for backward compatibility
                    }
                )

            # Sort by final score and limit results
            formatted_results.sort(key=lambda x: x["final_score"], reverse=True)
            formatted_results = formatted_results[:n_results]

            # Rerank results if enabled
            if self.config["search"]["rerank"]:
                formatted_results = self._rerank_results(query, formatted_results)

            # Add highlighting if requested
            show_scores = (
                show_scores
                if show_scores is not None
                else self.config["search"]["show_scores"]
            )
            if show_scores:
                for result in formatted_results:
                    result["highlighted_text"] = self._highlight_matches(
                        query, result["text"]
                    )

            return formatted_results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def _init_bm25_scorer(self, collection):
        """Initialize BM25 scorer with collection documents"""
        if self._documents_cache is None:
            # Get all documents from collection
            all_data = collection.get()
            self._documents_cache = all_data["documents"]

        self._bm25_scorer = BM25Scorer()
        self._bm25_scorer.fit(self._documents_cache)

    def _rerank_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results to improve diversity and relevance"""
        if len(results) <= 2:
            return results

        reranked = []
        used_sources = set()

        # First pass: take best result from each source
        for result in results:
            source = result["metadata"]["source"]
            if source not in used_sources:
                reranked.append(result)
                used_sources.add(source)
                if len(reranked) >= len(results) // 2:
                    break

        # Second pass: add remaining results
        for result in results:
            if result not in reranked:
                reranked.append(result)

        return reranked[: len(results)]

    def _highlight_matches(
        self, query: str, text: str, context_chars: int = None
    ) -> str:
        """Highlight matching terms in text"""
        context_chars = context_chars or self.config["search"]["context_chars"]

        # Simple highlighting - find first match and show context
        query_terms = re.findall(r"\b\w+\b", query.lower())
        text_lower = text.lower()

        # Find first match position
        match_pos = -1
        for term in query_terms:
            pos = text_lower.find(term)
            if pos != -1:
                match_pos = pos
                break

        if match_pos == -1:
            # No direct match, return beginning
            return text[:context_chars] + "..." if len(text) > context_chars else text

        # Calculate context window around match
        start = max(0, match_pos - context_chars // 2)
        end = min(len(text), match_pos + context_chars // 2)

        # Extend to word boundaries
        while start > 0 and text[start] != " ":
            start -= 1
        while end < len(text) and text[end] != " ":
            end += 1

        excerpt = text[start:end].strip()
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(text):
            excerpt = excerpt + "..."

        return excerpt

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()

            # Get source distribution
            all_data = collection.get()
            sources = {}
            for meta in all_data["metadatas"]:
                src = meta["source"]
                sources[src] = sources.get(src, 0) + 1

            return {
                "total_chunks": count,
                "sources": sources,
                "db_path": str(self.db_dir),
            }
        except Exception as e:
            return {"error": "Database not found. Run 'python raggy.py build' first to index your documents."}

    def interactive_search(self):
        """Interactive search mode"""
        print(f"\n{SYMBOLS['search']} Interactive Search Mode")
        print("Type your queries (or 'quit' to exit)")
        print("-" * 50)

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break

                if not query:
                    continue

                start_time = time.time()
                results = self.search(query)
                elapsed = time.time() - start_time

                if not results:
                    print("No results found.")
                    continue

                print(
                    f"\n{SYMBOLS['found']} Found {len(results)} results (in {elapsed:.3f}s):"
                )
                for i, result in enumerate(results, 1):
                    print(f"\n--- Result {i} ---")
                    print(f"Source: {result['metadata']['source']}")
                    print(
                        f"Chunk: {result['metadata']['chunk_index'] + 1}/{result['metadata']['total_chunks']}"
                    )
                    if result["similarity"]:
                        print(f"Similarity: {result['similarity']:.3f}")
                    print(f"Text preview: {result['text'][:200]}...")

            except KeyboardInterrupt:
                break

        print(f"\n{SYMBOLS['bye']} Goodbye!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Universal ChromaDB RAG Setup Script v2.0.0 - Enhanced with hybrid search and smart chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Setup:
    %(prog)s init                               # Initialize project environment (first-time setup)
    
  Basic Usage:
    %(prog)s build                              # Build/update index with smart chunking
    %(prog)s search "your search term"         # Semantic search with normalized scores
    %(prog)s status                             # Database statistics and configuration
    
  Enhanced Search:
    %(prog)s search "exact phrase" --hybrid    # Hybrid semantic + keyword search
    %(prog)s search "api" --expand             # Query expansion (api → application programming interface)  
    %(prog)s search "documentation" --hybrid --expand # Combined hybrid + expansion
    
  Model Selection:
    %(prog)s build --model-preset multilingual  # Use multilingual model for non-English content
    %(prog)s search "query" --model-preset fast # Quick search with smaller model
    
  Output & Analysis:
    %(prog)s search "query" --json             # Enhanced JSON with score breakdown
    %(prog)s optimize                           # Benchmark semantic vs hybrid search
    %(prog)s interactive --quiet                # Interactive mode, minimal output
    
  Advanced:
    %(prog)s rebuild --config custom.yaml       # Use custom configuration
    %(prog)s search "term" --results 10        # More results with quality scores
        """,
    )

    parser.add_argument(
        "command",
        choices=["init", "build", "rebuild", "search", "interactive", "status", "optimize"],
        help="Command to execute",
    )
    parser.add_argument("query", nargs="*", help="Search query (for search command)")

    # Options
    parser.add_argument(
        "--docs-dir", default="./docs", help="Documents directory (default: ./docs)"
    )
    parser.add_argument(
        "--db-dir",
        default="./vectordb",
        help="Vector database directory (default: ./vectordb)",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="Embedding model name"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Text chunk size (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Text chunk overlap (default: 200)",
    )
    parser.add_argument(
        "--results", type=int, default=5, help="Number of search results (default: 5)"
    )

    # Flags
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster, smaller model (paraphrase-MiniLM-L3-v2)",
    )
    parser.add_argument(
        "--hybrid", action="store_true", help="Use hybrid semantic+keyword search"
    )
    parser.add_argument(
        "--expand", action="store_true", help="Expand query with synonyms"
    )
    parser.add_argument(
        "--model-preset",
        choices=["fast", "balanced", "multilingual", "accurate"],
        help="Use model preset (overrides --model)",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency checks (faster startup)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument(
        "--json", action="store_true", help="Output search results as JSON"
    )
    parser.add_argument(
        "--config", help="Path to config file (default: raggy_config.yaml)"
    )
    parser.add_argument("--version", action="version", version="raggy 2.0.0")

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle init command before dependency setup
    if args.command == "init":
        success = setup_environment(quiet=args.quiet)
        if not success:
            sys.exit(1)
        return

    # Setup dependencies (unless skipped)
    if not args.skip_deps:
        setup_dependencies(quiet=args.quiet)
    else:
        # Still need to import even if skipping dependency checks
        try:
            global chromadb, SentenceTransformer, PyPDF2, HAS_MAGIC, magic
            import chromadb
            from sentence_transformers import SentenceTransformer
            import PyPDF2

            try:
                import magic

                HAS_MAGIC = True
            except ImportError:
                HAS_MAGIC = False
        except ImportError as e:
            print(f"Missing dependency: {e}")
            print("Run without --skip-deps or install dependencies manually")
            return

    # Determine model to use
    if args.model_preset:
        # Load config to get model presets
        config = load_config(args.config)
        if args.model_preset == "fast":
            model_name = config["models"]["fast"]
        elif args.model_preset == "multilingual":
            model_name = config["models"]["multilingual"]
        elif args.model_preset == "accurate":
            model_name = config["models"]["accurate"]
        else:  # balanced
            model_name = config["models"]["default"]
    else:
        # Use fast model if requested, otherwise use specified model
        model_name = "paraphrase-MiniLM-L3-v2" if args.fast else args.model

    # Initialize RAG with configuration
    rag = UniversalRAG(
        docs_dir=args.docs_dir,
        db_dir=args.db_dir,
        model_name=model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        quiet=args.quiet,
        config_path=args.config,
    )

    # Execute command
    if args.command == "build":
        rag.build()
    elif args.command == "rebuild":
        rag.build(force_rebuild=True)
    elif args.command == "search":
        if not args.query:
            print("Please provide a search query")
            return

        query = " ".join(args.query)
        results = rag.search(
            query, n_results=args.results, hybrid=args.hybrid, expand_query=args.expand
        )

        if not results:
            print("No results found.")
            return

        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "text": r["text"],
                            "source": r["metadata"]["source"],
                            "chunk": r["metadata"]["chunk_index"] + 1,
                            "final_score": r.get("final_score", r.get("similarity", 0)),
                            "semantic_score": r.get("semantic_score", 0),
                            "keyword_score": r.get("keyword_score", 0),
                            "interpretation": r.get("score_interpretation", "Unknown"),
                        }
                        for r in results
                    ],
                    indent=2,
                )
            )
        else:
            print(f"\n{SYMBOLS['search']} Search results for: '{query}'")
            if args.hybrid:
                print("(Using hybrid semantic + keyword search)")
            if args.expand:
                print("(Using query expansion)")
            print("=" * 50)

            for i, result in enumerate(results, 1):
                score = result.get("final_score", result.get("similarity", 0))
                interpretation = result.get("score_interpretation", "")

                score_str = f" ({interpretation}: {score:.3f})" if score else ""
                print(
                    f"\n{i}. {result['metadata']['source']} (chunk {result['metadata']['chunk_index'] + 1}){score_str}"
                )

                # Show highlighted text if available, otherwise truncated text
                display_text = result.get(
                    "highlighted_text", result["text"][:300] + "..."
                )
                print(f"   {display_text}")

    elif args.command == "interactive":
        rag.interactive_search()
    elif args.command == "status":
        stats = rag.get_stats()
        if "error" in stats:
            print(f"Error getting stats: {stats['error']}")
        else:
            print(f"Database Statistics:")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Database path: {stats['db_path']}")
            print(f"  Model: {rag.model_name}")
            print(f"  Config: {'Custom' if args.config else 'Default'}")
            print(f"  Documents:")
            for source, count in sorted(stats["sources"].items()):
                print(f"    {source}: {count} chunks")
    elif args.command == "optimize":
        print(
            f"\n{SYMBOLS['search']} Running benchmark queries to optimize settings..."
        )

        # Get sample content from the database to generate test queries
        stats = rag.get_stats()
        if "error" in stats or stats["total_chunks"] == 0:
            print("Error: No indexed content found. Run 'build' first.")
            return

        # Generate universal test queries based on document names and common terms
        test_queries = []
        doc_names = list(stats["sources"].keys())

        # Add filename-based queries (without extensions)
        for doc in doc_names[:3]:  # Use first 3 documents
            base_name = Path(doc).stem.replace("-", " ").replace("_", " ")
            test_queries.append(base_name)

        # Add some universal technical queries
        universal_queries = [
            "configuration",
            "setup",
            "guide",
            "documentation",
            "features",
        ]
        test_queries.extend(universal_queries[:2])  # Add 2 universal terms

        print(
            f"Testing with queries derived from your content: {', '.join(test_queries)}"
        )

        results_semantic = []
        results_hybrid = []

        for query in test_queries:
            print(f"Testing: {query}")

            # Test semantic search
            sem_results = rag.search(query, n_results=3, hybrid=False)
            avg_sem_score = (
                sum(r.get("final_score", 0) for r in sem_results) / len(sem_results)
                if sem_results
                else 0
            )
            results_semantic.append(avg_sem_score)

            # Test hybrid search
            hyb_results = rag.search(query, n_results=3, hybrid=True)
            avg_hyb_score = (
                sum(r.get("final_score", 0) for r in hyb_results) / len(hyb_results)
                if hyb_results
                else 0
            )
            results_hybrid.append(avg_hyb_score)

        avg_semantic = sum(results_semantic) / len(results_semantic)
        avg_hybrid = sum(results_hybrid) / len(results_hybrid)

        print(f"\n{SYMBOLS['found']} Optimization Results:")
        print(f"  Average Semantic Score: {avg_semantic:.3f}")
        print(f"  Average Hybrid Score: {avg_hybrid:.3f}")

        if avg_hybrid > avg_semantic * 1.1:
            print(
                f"\n{SYMBOLS['success']} Recommendation: Use --hybrid flag for better results"
            )
        elif avg_semantic > avg_hybrid * 1.1:
            print(
                f"\n{SYMBOLS['success']} Recommendation: Semantic search performs best for this content"
            )
        else:
            print(f"\n{SYMBOLS['success']} Both search modes perform similarly")

        print(f"\nSuggested usage:")
        print(
            f'  python {sys.argv[0]} search \\"your query\\" --hybrid    # For exact matches'
        )
        print(
            f'  python {sys.argv[0]} search \\"your query\\" --expand    # For broader results'
        )


if __name__ == "__main__":
    main()
