"""Environment setup and validation functions."""

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from ..utils.logging import log_error, log_warning


def check_uv_available() -> bool:
    """Check if uv is available.

    Returns:
        bool: True if uv is available, False otherwise
    """
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


def check_environment_setup() -> Tuple[bool, str]:
    """Check if project environment is properly set up.

    Returns:
        Tuple[bool, str]: (is_setup, issue_type) where issue_type is:
            - "ok" if everything is fine
            - "virtual_environment" if venv is missing
            - "pyproject" if pyproject.toml is missing
            - "invalid_venv" if venv exists but is invalid
    """
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
    except (OSError, AttributeError):
        return False, "invalid_venv"

    return True, "ok"


def _create_virtual_environment(quiet: bool = False) -> bool:
    """Create virtual environment if it doesn't exist.

    Args:
        quiet: If True, suppress output

    Returns:
        bool: True if successful, False otherwise
    """
    venv_path = Path(".venv")
    if not venv_path.exists():
        if not quiet:
            print("Creating virtual environment...")
        try:
            subprocess.check_call(
                ["uv", "venv"],
                stdout=subprocess.DEVNULL if quiet else None
            )
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to create virtual environment: {e}")
            return False
    return True


def _create_project_config(quiet: bool = False) -> bool:
    """Create minimal pyproject.toml if it doesn't exist.

    Args:
        quiet: If True, suppress output

    Returns:
        bool: True if successful, False otherwise
    """
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
    "python-docx>=1.0.0",
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
        except (PermissionError, OSError) as e:
            log_error("Failed to create pyproject.toml", e, quiet=False)
            return False
    return True


def _install_dependencies(quiet: bool = False) -> bool:
    """Install core and platform-specific dependencies.

    Args:
        quiet: If True, suppress output

    Returns:
        bool: True if successful, False otherwise
    """
    if not quiet:
        print("Installing dependencies...")

    try:
        # Install base dependencies
        base_deps = [
            "chromadb>=0.4.0",
            "sentence-transformers>=2.2.0",
            "PyPDF2>=3.0.0",
            "python-docx>=1.0.0"
        ]
        subprocess.check_call(
            ["uv", "pip", "install"] + base_deps,
            stdout=subprocess.DEVNULL if quiet else None
        )

        # Install platform-specific magic library
        _install_magic_library(quiet)

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        print("Manual install: uv pip install chromadb sentence-transformers PyPDF2")
        return False

    return True


def _install_magic_library(quiet: bool = False) -> None:
    """Install platform-specific magic library for file type detection.

    Args:
        quiet: If True, suppress output
    """
    magic_package = (
        "python-magic-bin>=0.4.14" if sys.platform == "win32"
        else "python-magic"
    )

    try:
        subprocess.check_call(
            ["uv", "pip", "install", magic_package],
            stdout=subprocess.DEVNULL if quiet else None
        )
    except subprocess.CalledProcessError:
        if not quiet:
            package_name = magic_package.split(">")[0]  # Remove version spec
            warning = f"Warning: Could not install {package_name}. "
            warning += "File type detection may be limited."
            print(warning)


def _create_docs_directory(quiet: bool = False) -> Optional[Path]:
    """Create docs directory if it doesn't exist.

    Args:
        quiet: If True, suppress output

    Returns:
        Optional[Path]: Path to docs directory or None on failure
    """
    docs_path = Path("docs")
    if not docs_path.exists():
        try:
            docs_path.mkdir()
            if not quiet:
                print("Created docs/ directory - add your documentation files here")
        except OSError as e:
            print(f"ERROR: Failed to create docs directory: {e}")
            return None
    return docs_path


def _create_development_state_file(docs_path: Path, quiet: bool = False) -> bool:
    """Create initial DEVELOPMENT_STATE.md for AI workflow tracking.

    Args:
        docs_path: Path to docs directory
        quiet: If True, suppress output

    Returns:
        bool: True if successful, False otherwise
    """
    dev_state_path = docs_path / "DEVELOPMENT_STATE.md"
    if not dev_state_path.exists():
        if not quiet:
            print("Creating initial DEVELOPMENT_STATE.md...")

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        dev_state_content = f"""# Development State

**Last Updated:** {timestamp}
**RAG System:** Raggy v2.0.0 - Universal ChromaDB RAG Setup

## Project Status: INITIALIZED

### COMPLETED:
- ✅ Raggy environment initialized with `python raggy.py init`
- ✅ Virtual environment (.venv) created and activated
- ✅ Dependencies installed: chromadb, sentence-transformers, PyPDF2, python-docx
- ✅ Project configuration (pyproject.toml) generated
- ✅ Example configuration (raggy_config_example.yaml) created
- ✅ Documentation directory (docs/) established
- ✅ Development state tracking initialized

### CURRENT SETUP:
- **Supported formats:** .md (Markdown), .pdf (PDF), .docx (Word), .txt (Plain text)
- **Search modes:** Semantic, Hybrid (semantic + keyword), Query expansion
- **Model presets:** fast/balanced/multilingual/accurate
- **Local processing:** 100% offline, zero API costs

### NEXT STEPS:
1. **Add documentation files** to the docs/ directory
2. **Optional:** Copy raggy_config_example.yaml to raggy_config.yaml and customize expansions
3. **Build the RAG database:** Run `python raggy.py build`
4. **Test search functionality:** Run `python raggy.py search "your query"`
5. **Configure AI agents** with the knowledge-driven workflow from README.md

### DECISIONS:
- Chose raggy for zero-cost local RAG implementation
- Configured for multi-format document support (.md, .pdf, .docx, .txt)
- Set up for AI agent integration with continuous development state tracking

### ARCHITECTURE:
- **Vector Database:** ChromaDB (local storage in ./vectordb/)
- **Embeddings:** sentence-transformers (local, no API costs)
- **Search Engine:** Hybrid semantic + BM25 keyword ranking
- **Document Processing:** Smart chunking with markdown awareness

### BLOCKERS:
- None - system ready for document ingestion and usage

---

*This file tracks development progress for AI agent continuity. Update after each significant task or decision.*
"""

        try:
            with open(dev_state_path, "w", encoding="utf-8") as f:
                f.write(dev_state_content)
        except (PermissionError, OSError) as e:
            log_warning("Could not create DEVELOPMENT_STATE.md", e, quiet=False)
            return False

    return True


def _create_example_config(quiet: bool = False) -> bool:
    """Create example configuration file.

    Args:
        quiet: If True, suppress output

    Returns:
        bool: True if successful, False otherwise
    """
    config_example_path = Path("raggy_config_example.yaml")
    if not config_example_path.exists():
        if not quiet:
            print("Creating raggy_config_example.yaml...")

        config_content = """# raggy_config_example.yaml - Example Configuration File
# Copy this to raggy_config.yaml and customize for your domain

search:
  hybrid_weight: 0.7  # Balance between semantic (0.7) and keyword (0.3) search
  chunk_size: 1000
  chunk_overlap: 200
  rerank: true
  show_scores: true
  context_chars: 200
  max_results: 5

  # Domain-specific query expansions
  # Add your own terms here for automatic expansion
  expansions:
    # Technical terms (examples)
    api: ["api", "application programming interface", "rest api", "web service"]
    ml: ["ml", "machine learning", "artificial intelligence"]
    ai: ["ai", "artificial intelligence", "machine learning"]
    ui: ["ui", "user interface", "frontend", "user experience"]
    ux: ["ux", "user experience", "usability", "user interface"]

    # Business terms (examples)
    roi: ["roi", "return on investment", "profitability"]
    kpi: ["kpi", "key performance indicator", "metrics"]
    crm: ["crm", "customer relationship management", "customer management"]

    # Development terms (examples)
    ci: ["ci", "continuous integration", "build automation"]
    cd: ["cd", "continuous deployment", "continuous delivery"]
    devops: ["devops", "development operations", "infrastructure"]

    # Add your domain-specific terms here:
    # mycompany: ["mycompany", "company name", "organization"]
    # myproduct: ["myproduct", "product name", "solution"]

models:
  default: "all-MiniLM-L6-v2"           # Balanced speed/accuracy
  fast: "paraphrase-MiniLM-L3-v2"       # Fastest, smaller model
  multilingual: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multi-language
  accurate: "all-mpnet-base-v2"         # Best accuracy, slower

chunking:
  smart: false          # Enable markdown-aware smart chunking (experimental)
  preserve_headers: true # Include section headers in chunks
  min_chunk_size: 300   # Minimum chunk size in characters
  max_chunk_size: 1500  # Maximum chunk size in characters

# Usage:
# 1. Copy this file to raggy_config.yaml
# 2. Customize the expansions section with your domain terms
# 3. Adjust model and chunking settings as needed
# 4. Run: python raggy.py search "your-term" --expand
"""

        try:
            with open(config_example_path, "w", encoding="utf-8") as f:
                f.write(config_content)
        except (PermissionError, OSError) as e:
            log_warning("Could not create raggy_config_example.yaml", e, quiet=False)
            return False

    return True


def _print_setup_summary() -> None:
    """Print summary of environment setup completion."""
    print("✅ Environment setup complete!")
    print("\nCreated files:")
    print("- .venv/ (virtual environment)")
    print("- pyproject.toml (project configuration)")
    print("- raggy_config_example.yaml (example configuration)")
    print("- docs/DEVELOPMENT_STATE.md (AI agent continuity tracking)")
    print("\nNext steps:")
    print("1. Add your documentation files to the docs/ directory")
    print("2. Optional: Copy raggy_config_example.yaml to raggy_config.yaml and customize")
    print("3. Run: python raggy.py build")
    print("4. Run: python raggy.py search \"your query\"")


def setup_environment(quiet: bool = False) -> bool:
    """Set up the project environment from scratch.

    Args:
        quiet: If True, suppress output

    Returns:
        bool: True if successful, False otherwise
    """
    if not quiet:
        print("🚀 Setting up raggy environment...")

    # Check if uv is available
    if not check_uv_available():
        return False

    # Create virtual environment
    if not _create_virtual_environment(quiet):
        return False

    # Create minimal pyproject.toml if it doesn't exist
    if not _create_project_config(quiet):
        return False

    # Install dependencies
    if not _install_dependencies(quiet):
        return False

    # Create docs directory if it doesn't exist
    docs_path = _create_docs_directory(quiet)
    if docs_path is None:
        return False

    # Create initial DEVELOPMENT_STATE.md for AI workflow
    if not _create_development_state_file(docs_path, quiet):
        # Warning already printed, continue anyway
        pass

    # Create example config file if it doesn't exist
    if not _create_example_config(quiet):
        # Warning already printed, continue anyway
        pass

    if not quiet:
        _print_setup_summary()

    return True