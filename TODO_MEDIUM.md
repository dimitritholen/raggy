# 🟡 MEDIUM Priority Issues

**Source:** Code Audit 2025-11-12
**Estimated Total Effort:** 12-18 hours

---

## #1: Broad Exception Handling - 19 Bare "except Exception" Blocks ✅ **COMPLETED**

**Severity:** MEDIUM
**File:** `raggy.py` (multiple locations)
**Estimated Effort:** 2-3 hours
**Completed:** 2025-11-13 (commit 3cad682)

### Problem

The codebase contains 19 instances of bare `except Exception` blocks that catch all exceptions indiscriminately. This anti-pattern:

- **Hides Bugs**: Catches programming errors (KeyError, AttributeError, etc.) that should crash
- **Difficult Debugging**: Generic handlers obscure root cause
- **Swallows Important Errors**: May mask critical failures (OOM, SystemExit, KeyboardInterrupt)
- **Violates Fail-Fast**: Continues execution in undefined state

**Locations of Broad Exception Handlers:**
- `raggy.py:519` - YAML config loading
- `raggy.py:523` - Config file access
- `raggy.py:1167` - Document processing
- `raggy.py:1193` - Text extraction template
- Multiple locations in file I/O operations

### Exception Handling Best Practices Violated

```python
# ANTI-PATTERN (current code):
except Exception as e:
    log_warning(f"Unexpected error loading config file", e, quiet=False)

# BETTER:
except (yaml.YAMLError, yaml.scanner.ScannerError) as e:
    log_warning(f"Invalid YAML syntax in config file", e, quiet=False)
except (FileNotFoundError, PermissionError) as e:
    log_warning(f"Could not access config file", e, quiet=False)
# Let programming errors (KeyError, AttributeError, etc.) crash
```

### Impact Analysis

**Critical Paths with Broad Handlers:**
1. Configuration loading (startup)
2. Document processing (data ingestion)
3. Text extraction (core functionality)
4. Database operations (data persistence)

**Risk:** Silent failures could corrupt database or index invalid documents.

### Acceptance Criteria

- [ ] All `except Exception` replaced with specific exception types
- [ ] Programming errors (KeyError, AttributeError, TypeError) allowed to crash
- [ ] Each exception handler has clear recovery strategy
- [ ] Error messages indicate specific failure mode
- [ ] Tests verify correct exception types are caught

### Recommended Approach

**Step 1: Audit Each Exception Handler**
```bash
rg "except Exception" raggy.py -A 3 > exception_audit.txt
```

**Step 2: Replace with Specific Handlers**

```python
# Example 1: Configuration Loading (line 519)
# BEFORE:
except Exception as yaml_error:
    # Handle YAML parsing errors (yaml module imported locally)
    log_warning(f"YAML parsing error in {config_file}", yaml_error, quiet=False)

# AFTER:
except (yaml.YAMLError, yaml.scanner.ScannerError, yaml.parser.ParserError) as e:
    log_error(f"Invalid YAML syntax in {config_file}: {e}", quiet=False)
except (UnicodeDecodeError, LookupError) as e:
    log_error(f"Encoding error in {config_file}: {e}", quiet=False)
# Let KeyError, AttributeError, etc. crash (indicates bug in code)

# Example 2: Document Processing (line 1167)
# BEFORE:
except Exception as e:
    handle_file_error(file_path, "process", e, quiet=self.quiet)
    return []

# AFTER:
except (PyPDF2.errors.PdfReadError, PyPDF2.errors.PdfStreamError) as e:
    log_error(f"Corrupted PDF {file_path.name}: {e}", quiet=self.quiet)
    return []
except (docx.opc.exceptions.PackageNotFoundError, KeyError) as e:
    log_error(f"Invalid DOCX format {file_path.name}: {e}", quiet=self.quiet)
    return []
except (UnicodeDecodeError, LookupError) as e:
    log_error(f"Text encoding error in {file_path.name}: {e}", quiet=self.quiet)
    return []
except (MemoryError, OSError) as e:
    log_error(f"Resource error processing {file_path.name}: {e}", quiet=self.quiet)
    return []
# Remove catch-all - let programming errors crash
```

**Step 3: Add Tests for Exception Handling**
```python
# tests/test_error_handling.py
def test_invalid_yaml_config_raises_specific_error():
    """Ensure YAML errors are caught specifically."""
    with pytest.raises(yaml.YAMLError):
        load_config("invalid: yaml: {{{}}")

def test_programming_errors_not_caught():
    """Ensure programming errors crash (fail-fast)."""
    # This should raise KeyError, not be silently caught
    with pytest.raises(KeyError):
        # Simulate code bug accessing missing key
        load_config_with_bug()
```

### Files to Modify

- `raggy.py` (19 locations - see audit file)
- Add `tests/test_error_handling.py` for validation
- Update error messages to be more specific

---

## #2: Silent Exception Handling with 'pass' Statements ✅ **COMPLETED**

**Severity:** MEDIUM
**File:** `raggy.py` (4 locations)
**Estimated Effort:** 1 hour
**Completed:** 2025-11-13 (commit c557ea2)

### Problem

Four exception handlers use bare `pass` statements, silently swallowing errors without logging. This makes debugging nearly impossible.

**Locations:**
1. `raggy.py:91-92` - UTF-8 encoding configuration
2. `raggy.py:224-225` - Session file timestamp check
3. `raggy.py:252-253` - Session file creation
4. `raggy.py:564` - Cache file save

### Current vs. Recommended

```python
# LOCATION 1 (line 91-92)
# CURRENT:
except (AttributeError, OSError):
    pass  # Ignore encoding configuration errors

# RECOMMENDED:
except (AttributeError, OSError) as e:
    # Encoding setup is optional, log but continue
    if not quiet:
        print(f"Warning: Could not configure UTF-8 encoding: {e}", file=sys.stderr)

# LOCATION 2 (line 224-225)
# CURRENT:
except (OSError, AttributeError):
    pass  # If we can't check file time, proceed with check

# RECOMMENDED:
except (OSError, AttributeError) as e:
    # If session file is unreadable, treat as expired (safe default)
    log_warning(f"Could not read session cache, checking for updates: {e}", quiet=quiet)

# LOCATION 3 (line 252-253)
# CURRENT:
except (OSError, PermissionError):
    pass  # If we can't create session file, just skip tracking

# RECOMMENDED:
except (OSError, PermissionError) as e:
    # Session tracking is optional, log once per session
    log_warning(f"Could not create session file: {e}", quiet=quiet)

# LOCATION 4 (line 564)
# CURRENT:
except (PermissionError, OSError):
    pass  # Ignore cache save errors

# RECOMMENDED:
except (PermissionError, OSError) as e:
    # Cache is performance optimization, not critical
    log_warning(f"Could not save dependency cache: {e}", quiet=True)
```

### Acceptance Criteria

- [ ] All 4 `pass` statements replaced with logging
- [ ] Log messages explain why error is non-critical
- [ ] Quiet mode still respected
- [ ] Documentation updated to note optional features

### Files to Modify

- `raggy.py:91-92` (add warning for encoding setup)
- `raggy.py:224-225` (log session file read error)
- `raggy.py:252-253` (log session file write error)
- `raggy.py:564` (log cache save error)

---

## #3: Multiple High-Complexity Functions Need Refactoring ✅ **COMPLETED**

**Severity:** MEDIUM
**File:** `raggy.py` (multiple locations)
**Estimated Effort:** 6-8 hours
**Completed:** 2025-11-13

### Problem

In addition to the critical complexity-20 function, there are several medium-complexity functions (10-19) that should be refactored.

**Functions with Complexity 10-19:**
- `check_for_updates` - Complexity 13 (line 197)
- `SearchEngine._highlight_matches` - Complexity 12 (line 1687)
- `UniversalRAG.run_self_tests` - Complexity 12 (line 2249)
- `UniversalRAG.diagnose_system` - Complexity 12 (line 2323)
- `OptimizeCommand.execute` - Complexity 12 (line 2682)
- `UniversalRAG.build` - Complexity 11 (line 1780)
- `UniversalRAG._process_document` - Complexity 11 (line 2179)
- `setup_environment` - Complexity 10 (line 803)
- `DocumentProcessor.process_document` - Complexity 10 (line 1103)
- `SearchCommand.execute` - Complexity 10 (line 2598)

### Acceptance Criteria

- [ ] All functions reduced to complexity < 10
- [ ] Extract helper functions for conditional logic
- [ ] Each function has single responsibility
- [ ] 100% test coverage for extracted functions

### Recommended Approach

**Example: Refactor check_for_updates (Complexity 13)**

```python
# BEFORE (lines 197-265, complexity 13):
def check_for_updates(quiet: bool = False, config: Optional[Dict[str, Any]] = None) -> None:
    """Check GitHub for latest version once per session (non-intrusive)."""
    if quiet:
        return

    # Load configuration for update settings
    if config is None:
        config = {}

    updates_config = config.get("updates", {})
    if not updates_config.get("check_enabled", True):
        return

    # Use configured repo or default placeholder
    github_repo = updates_config.get("github_repo", "dimitritholen/raggy")

    # Session tracking to avoid frequent checks
    session_file = Path.home() / ".raggy_session"

    # Check if already checked in last 24 hours
    if session_file.exists():
        try:
            cache_age = time.time() - session_file.stat().st_mtime
            if cache_age < SESSION_CACHE_HOURS * 3600:  # 24 hours
                return
        except (OSError, AttributeError):
            pass  # If we can't check file time, proceed with check

    try:
        # Import urllib only when needed to avoid startup cost
        import urllib.request
        import urllib.error

        # Quick timeout to not delay startup
        api_url = f"https://api.github.com/repos/{github_repo}/releases/latest"

        with urllib.request.urlopen(api_url, timeout=UPDATE_TIMEOUT_SECONDS) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                latest_version = data.get("tag_name", "").lstrip("v")

                if latest_version and latest_version != __version__:
                    # Use HTML URL from response or construct fallback
                    github_url = data.get("html_url")
                    if not github_url:
                        base_url = f"https://github.com/{github_repo}"
                        github_url = f"{base_url}/releases/latest"

                    print(f"📦 Raggy update available: v{latest_version} → {github_url}")

        # Update session file to mark check as done
        try:
            session_file.touch()
        except (OSError, PermissionError):
            pass  # If we can't create session file, just skip tracking

    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        ConnectionError,
        TimeoutError,
        Exception
    ):
        # Silently fail - don't interrupt user workflow with network issues
        # This includes any import errors, network timeouts, or API issues
        pass

# AFTER (complexity < 10):
class UpdateChecker:
    """Handles version update checks with session caching."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.updates_config = self.config.get("updates", {})
        self.session_file = Path.home() / ".raggy_session"

    def check_for_updates(self, quiet: bool = False) -> None:
        """Check GitHub for latest version once per session."""
        if not self._should_check(quiet):
            return

        latest_version = self._fetch_latest_version()
        if latest_version and self._is_newer(latest_version):
            self._display_update_notice(latest_version)

        self._update_session_cache()

    def _should_check(self, quiet: bool) -> bool:
        """Determine if update check should run."""
        if quiet:
            return False

        if not self.updates_config.get("check_enabled", True):
            return False

        return not self._is_recently_checked()

    def _is_recently_checked(self) -> bool:
        """Check if update was checked in last 24 hours."""
        if not self.session_file.exists():
            return False

        try:
            cache_age = time.time() - self.session_file.stat().st_mtime
            return cache_age < SESSION_CACHE_HOURS * 3600
        except (OSError, AttributeError):
            # If we can't read session file, assume expired
            return False

    def _fetch_latest_version(self) -> Optional[str]:
        """Fetch latest version from GitHub API."""
        github_repo = self.updates_config.get("github_repo", "dimitritholen/raggy")
        api_url = f"https://api.github.com/repos/{github_repo}/releases/latest"

        try:
            import urllib.request
            import urllib.error

            with urllib.request.urlopen(api_url, timeout=UPDATE_TIMEOUT_SECONDS) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    return data.get("tag_name", "").lstrip("v")
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ConnectionError, TimeoutError):
            # Network errors are expected and non-critical
            pass

        return None

    def _is_newer(self, version: str) -> bool:
        """Check if fetched version is newer than current."""
        return version != __version__

    def _display_update_notice(self, latest_version: str) -> None:
        """Display update notification to user."""
        github_repo = self.updates_config.get("github_repo", "dimitritholen/raggy")
        github_url = f"https://github.com/{github_repo}/releases/latest"
        print(f"📦 Raggy update available: v{latest_version} → {github_url}")

    def _update_session_cache(self) -> None:
        """Update session file to mark check as done."""
        try:
            self.session_file.touch()
        except (OSError, PermissionError):
            # Session tracking is optional
            pass


# Public API
def check_for_updates(quiet: bool = False, config: Optional[Dict[str, Any]] = None) -> None:
    """Check GitHub for latest version once per session (non-intrusive)."""
    checker = UpdateChecker(config)
    checker.check_for_updates(quiet)
```

Apply similar pattern to other complex functions.

### Files to Modify

- `raggy.py:197-265` (refactor check_for_updates)
- `raggy.py:1687-1725` (refactor _highlight_matches)
- `raggy.py:2249-2321` (refactor run_self_tests)
- `raggy.py:2323-2409` (refactor diagnose_system)
- `raggy.py:2682-2765` (refactor OptimizeCommand)
- Additional 5 functions with complexity 10-11
- Add comprehensive tests for each refactored function

---

## #4: Tight Coupling Between UniversalRAG and ChromaDB

**Severity:** MEDIUM
**File:** `raggy.py:1728-2468`
**Estimated Effort:** 3-4 hours

### Problem

`UniversalRAG` is tightly coupled to ChromaDB implementation details. This violates **Dependency Inversion Principle** (SOLID) and prevents:

- Switching to alternative vector databases
- Testing with mock database
- Supporting multiple backends
- Database-agnostic deployment

**Evidence of Tight Coupling:**
```python
class UniversalRAG:
    def __init__(self, ...):
        # Direct ChromaDB dependency
        self._client: Optional[chromadb.PersistentClient] = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.db_dir))
        return self._client
```

### Acceptance Criteria

- [ ] Define abstract VectorDatabase interface
- [ ] ChromaDB becomes one implementation
- [ ] UniversalRAG depends on interface, not concrete class
- [ ] Easy to add new database backends
- [ ] All tests pass with mock database

### Recommended Approach

```python
# Create abstraction in core/database_interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorDatabase(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def create_collection(self, name: str, metadata: Dict[str, Any]) -> 'Collection':
        """Create a new collection."""
        pass

    @abstractmethod
    def get_collection(self, name: str) -> 'Collection':
        """Get existing collection."""
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        pass


class Collection(ABC):
    """Abstract interface for collection operations."""

    @abstractmethod
    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Add documents to collection."""
        pass

    @abstractmethod
    def query(self, query_texts: List[str], n_results: int, where: Optional[Dict] = None) -> Dict[str, Any]:
        """Query the collection."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get document count."""
        pass

    @abstractmethod
    def get(self, where: Optional[Dict] = None) -> Dict[str, Any]:
        """Get documents from collection."""
        pass


# Implement ChromaDB adapter
class ChromaDBAdapter(VectorDatabase):
    """ChromaDB implementation of VectorDatabase interface."""

    def __init__(self, path: str):
        import chromadb
        self._client = chromadb.PersistentClient(path=path)

    def create_collection(self, name: str, metadata: Dict[str, Any]) -> Collection:
        chroma_collection = self._client.create_collection(name=name, metadata=metadata)
        return ChromaCollection(chroma_collection)

    def get_collection(self, name: str) -> Collection:
        chroma_collection = self._client.get_collection(name=name)
        return ChromaCollection(chroma_collection)

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name=name)


class ChromaCollection(Collection):
    """ChromaDB implementation of Collection interface."""

    def __init__(self, collection):
        self._collection = collection

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_texts: List[str], n_results: int, where: Optional[Dict] = None) -> Dict[str, Any]:
        return self._collection.query(query_texts=query_texts, n_results=n_results, where=where)

    def count(self) -> int:
        return self._collection.count()

    def get(self, where: Optional[Dict] = None) -> Dict[str, Any]:
        return self._collection.get(where=where)


# Update UniversalRAG to use interface
class UniversalRAG:
    def __init__(
        self,
        docs_dir: str = "./docs",
        db_dir: str = "./vectordb",
        database: Optional[VectorDatabase] = None,  # Dependency injection
        **kwargs
    ):
        self.docs_dir = Path(docs_dir)
        self.db_dir = Path(db_dir)

        # Use provided database or default to ChromaDB
        self._database = database or ChromaDBAdapter(path=str(self.db_dir))
```

### Files to Modify

- Create `core/database_interface.py` (new file, ~150 lines)
- Create `core/chromadb_adapter.py` (new file, ~100 lines)
- `raggy.py:1728-1778` (update UniversalRAG.__init__ and client property)
- `raggy.py:1431-1518` (update DatabaseManager similarly)
- Add `tests/test_database_interface.py` (mock implementations)

---

## #5: Missing Input Validation in Public APIs

**Severity:** MEDIUM
**File:** `raggy.py` (multiple public methods)
**Estimated Effort:** 2-3 hours

### Problem

Public API methods don't validate inputs, leading to confusing errors when users pass invalid arguments.

**Locations Missing Validation:**
1. `UniversalRAG.__init__` - No validation of paths, chunk sizes
2. `UniversalRAG.search` - No validation of query length, result count
3. `BM25Scorer.fit` - No validation of empty document list
4. `QueryProcessor.__init__` - No validation of expansion dictionary format
5. `load_config` - No validation of config structure after loading

### Example Issues

```python
# User can pass invalid values without clear error:
rag = UniversalRAG(chunk_size=-100)  # Should fail fast with clear message
rag = UniversalRAG(chunk_overlap=2000, chunk_size=1000)  # Overlap > size
results = rag.search("query", top_k=-5)  # Negative result count
```

### Acceptance Criteria

- [ ] All public methods validate inputs
- [ ] Clear error messages for invalid inputs
- [ ] Type hints enforced at runtime for critical paths
- [ ] Documentation updated with valid ranges
- [ ] Tests for validation logic

### Recommended Approach

```python
class UniversalRAG:
    def __init__(
        self,
        docs_dir: str = "./docs",
        db_dir: str = "./vectordb",
        model_name: str = DEFAULT_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        quiet: bool = False,
    ):
        # Validate inputs
        self._validate_init_params(
            docs_dir, db_dir, model_name, chunk_size, chunk_overlap
        )

        # ... rest of init ...

    def _validate_init_params(
        self,
        docs_dir: str,
        db_dir: str,
        model_name: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """Validate initialization parameters."""
        # Validate chunk size
        if not isinstance(chunk_size, int):
            raise TypeError(f"chunk_size must be int, got {type(chunk_size)}")
        if chunk_size < 100:
            raise ValueError(f"chunk_size must be >= 100, got {chunk_size}")
        if chunk_size > 10000:
            raise ValueError(f"chunk_size must be <= 10000, got {chunk_size}")

        # Validate chunk overlap
        if not isinstance(chunk_overlap, int):
            raise TypeError(f"chunk_overlap must be int, got {type(chunk_overlap)}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )

        # Validate paths
        if not docs_dir:
            raise ValueError("docs_dir cannot be empty")
        if not db_dir:
            raise ValueError("db_dir cannot be empty")

        # Validate model name
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty string")

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_RESULTS,
        mode: str = "hybrid",
        expand: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search documents with input validation."""
        # Validate query
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query)}")
        if not query.strip():
            raise ValueError("query cannot be empty")
        if len(query) > 10000:
            raise ValueError(f"query too long ({len(query)} chars, max 10000)")

        # Validate top_k
        if not isinstance(top_k, int):
            raise TypeError(f"top_k must be int, got {type(top_k)}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k > 100:
            raise ValueError(f"top_k must be <= 100, got {top_k}")

        # Validate mode
        valid_modes = {"semantic", "hybrid", "keyword"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

        # Continue with search...
```

### Files to Modify

- `raggy.py:1731-1769` (add validation to UniversalRAG.__init__)
- `raggy.py:1840-1857` (add validation to UniversalRAG.search)
- `raggy.py:280-308` (add validation to BM25Scorer.fit)
- `raggy.py:343-355` (add validation to QueryProcessor.__init__)
- `raggy.py:470-534` (add validation to load_config)
- Add `tests/test_input_validation.py` with extensive validation tests

---

## Progress Tracking

- [x] Issue #1: Broad Exception Handling - 19 Bare "except Exception" Blocks ✅ **COMPLETED** (2025-11-13, commit 3cad682)
- [x] Issue #2: Silent Exception Handling with 'pass' Statements ✅ **COMPLETED** (2025-11-13, commit c557ea2)
- [x] Issue #3: Multiple High-Complexity Functions Need Refactoring ✅ **COMPLETED** (2025-11-13)
- [ ] Issue #4: Tight Coupling Between UniversalRAG and ChromaDB
- [ ] Issue #5: Missing Input Validation in Public APIs

**Total:** 3/5 completed (60%)

**Remaining Effort:** 5-7 hours of development work.
