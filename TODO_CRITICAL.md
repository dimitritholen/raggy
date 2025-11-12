# 🔴 CRITICAL Priority Issues

**Source:** Code Audit 2025-11-12
**Estimated Total Effort:** 16-24 hours

---

## #1: Broken Test Suite - Missing ScoringNormalizer Class

**Severity:** CRITICAL
**File:** `tests/test_raggy.py:8`
**Estimated Effort:** 30 minutes

### Problem

Tests import a non-existent `ScoringNormalizer` class, causing complete test suite failure. This prevents running any tests and achieving the 85% coverage target.

**Current State:**
- Tests fail to import: `ImportError: cannot import name 'ScoringNormalizer' from 'raggy'`
- Current coverage: **12%** (target: **85%**)
- 92 tests cannot run

**Impact:**
- No test validation possible
- Production deployment risk
- Cannot verify bug fixes
- Code quality unknown

### Root Cause

The `ScoringNormalizer` class was likely removed or renamed during refactoring, but tests were not updated accordingly. The scoring functions now exist as module-level functions (`normalize_cosine_distance`, `normalize_hybrid_score`, `interpret_score`).

### Acceptance Criteria

- [ ] All test imports resolve correctly
- [ ] Test suite runs without import errors
- [ ] Minimum 85% code coverage achieved
- [ ] All 92 tests execute successfully

### Recommended Approach

```python
# In tests/test_raggy.py, change:
from raggy import UniversalRAG, ScoringNormalizer

# To:
from raggy import (
    UniversalRAG,
    normalize_cosine_distance,
    normalize_hybrid_score,
    interpret_score
)
```

Then update test methods to use the module-level functions instead of class methods.

### Files to Modify

- `tests/test_raggy.py` (fix imports and update test methods)
- Review all test files for similar import issues

---

## #2: God Module - 2901 Lines in Single File

**Severity:** CRITICAL
**File:** `raggy.py:1-2901`
**Estimated Effort:** 2-3 days

### Problem

The entire application exists in a single 2901-line file with 16 classes and 103 functions/methods. This is a classic **God Module** anti-pattern that severely impacts:

- **Maintainability**: Impossible to navigate and understand
- **Testing**: Cannot test components in isolation
- **Collaboration**: Merge conflicts inevitable
- **Performance**: Import time, memory footprint
- **Reusability**: Cannot import specific components

**Current Structure:**
- 16 classes crammed in one file
- 103 functions/methods
- Multiple responsibilities mixed together
- No logical separation of concerns

### Responsibilities Currently Mixed

1. **CLI/Command Layer**: `Command`, `InitCommand`, `BuildCommand`, etc. (10 classes)
2. **Core Business Logic**: `UniversalRAG`, `SearchEngine`, `DatabaseManager`, `DocumentProcessor` (4 classes)
3. **Utilities**: `BM25Scorer`, `QueryProcessor` (2 classes)
4. **Infrastructure**: Setup, dependencies, error handling (30+ functions)
5. **Configuration**: Config loading, validation (10+ functions)

### Acceptance Criteria

- [ ] Code split into logical modules (max 500 lines each)
- [ ] Clear separation of concerns
- [ ] Proper package structure
- [ ] Import time < 1 second
- [ ] All existing functionality preserved
- [ ] Tests pass after refactoring

### Recommended Approach

**Phase 1: Create Package Structure**
```
raggy/
├── __init__.py              # Public API exports
├── core/
│   ├── __init__.py
│   ├── rag.py              # UniversalRAG class
│   ├── search.py           # SearchEngine class
│   ├── database.py         # DatabaseManager class
│   └── document.py         # DocumentProcessor class
├── scoring/
│   ├── __init__.py
│   ├── bm25.py             # BM25Scorer class
│   └── normalization.py    # Scoring functions
├── query/
│   ├── __init__.py
│   └── processor.py        # QueryProcessor class
├── cli/
│   ├── __init__.py
│   ├── base.py             # Command base class
│   ├── commands.py         # All command classes
│   └── factory.py          # CommandFactory
├── config/
│   ├── __init__.py
│   ├── loader.py           # load_config, _merge_configs
│   └── constants.py        # All constants
├── setup/
│   ├── __init__.py
│   ├── environment.py      # setup_environment, check functions
│   └── dependencies.py     # install_if_missing, setup_dependencies
└── utils/
    ├── __init__.py
    ├── security.py         # validate_path, sanitize_error_message
    ├── logging.py          # log_error, log_warning, handle_file_error
    └── updates.py          # check_for_updates
```

**Phase 2: Move Code Incrementally**
1. Move utility functions first (low risk)
2. Move classes with no dependencies
3. Move dependent classes
4. Update imports throughout

**Phase 3: Validation**
1. Run full test suite after each move
2. Verify import times
3. Check for circular dependencies
4. Run bandit security scan

### Files to Modify

- `raggy.py` → Split into 15+ module files as shown above
- Update all imports in tests
- Create proper `__init__.py` files with public API

---

## #3: Massive Code Duplication - DocumentProcessor vs UniversalRAG

**Severity:** CRITICAL
**File:** `raggy.py:1066-1430` and `raggy.py:1728-2468`
**Estimated Effort:** 4-6 hours

### Problem

`DocumentProcessor` and `UniversalRAG` contain **identical implementations** of critical methods. This violates DRY principle and creates serious maintenance burden.

### Duplication Includes

**Extraction Methods (100% duplicated):**
- `_extract_pdf_content` (lines 1214-1223 and 1947-1956)
- `_extract_md_content` (lines 1225-1228 and 1958-1961)
- `_extract_docx_content` (lines 1230-1252 and 1963-1985)
- `_extract_txt_content` (lines 1254-1263 and 1987-1996)
- `_extract_text_template` (lines 1180-1196 and 1913-1929)

**Chunking Methods (100% duplicated):**
- `_chunk_text` (lines 1265-1279 and 1998-2012)
- `_chunk_text_simple` (lines 1281-1310 and 2014-2043)
- `_chunk_text_smart` (lines 1312-1350 and 2049-2083)
- `_process_section` (lines 1352-1428 and 2085-2161)

**Additional Duplication:**
- `_get_file_hash` (lines 1171-1178 and 1904-1911)
- `_find_documents` (appears in both with minor variations)

**Total Duplicated LOC:** ~400 lines (14% of codebase)

### Impact

- **Bug Multiplication**: Fix in one place, miss the other
- **Inconsistent Behavior**: Methods can drift apart
- **Maintenance Nightmare**: Double the work for any change
- **Test Coverage**: Need to test same logic twice
- **Memory Waste**: Two copies of same code loaded

### Root Cause

`UniversalRAG` reimplements all `DocumentProcessor` functionality instead of delegating to it. This suggests:
1. `DocumentProcessor` was added later as refactoring attempt
2. Old methods in `UniversalRAG` were never removed
3. No code review caught the duplication

### Acceptance Criteria

- [ ] Remove all duplicated methods from `UniversalRAG`
- [ ] `UniversalRAG` delegates to `DocumentProcessor` for all file operations
- [ ] Single source of truth for each operation
- [ ] All tests pass with same behavior
- [ ] Code reduced by ~400 lines

### Recommended Approach

**Step 1: Ensure DocumentProcessor is fully tested**
```bash
# Verify DocumentProcessor has proper test coverage
pytest tests/test_document_processing.py -v --cov=raggy
```

**Step 2: Refactor UniversalRAG to use DocumentProcessor**
```python
class UniversalRAG:
    def __init__(self, ...):
        # ...existing init code...

        # Add DocumentProcessor as collaborator
        self.doc_processor = DocumentProcessor(
            docs_dir=self.docs_dir,
            config=self.config,
            quiet=self.quiet
        )

    def build(self, force_rebuild: bool = False) -> None:
        # Replace direct document processing with delegation
        files = self.doc_processor.find_documents()

        for file_path in files:
            documents = self.doc_processor.process_document(file_path)
            # ... existing indexing code ...
```

**Step 3: Remove all duplicated methods from UniversalRAG**
```python
# DELETE these methods from UniversalRAG class:
# - _extract_pdf_content
# - _extract_md_content
# - _extract_docx_content
# - _extract_txt_content
# - _extract_text_template
# - _extract_text_from_pdf/md/docx/txt (wrapper methods)
# - _chunk_text
# - _chunk_text_simple
# - _chunk_text_smart
# - _process_section
# - _get_file_hash
# - _find_documents (if exists)
```

**Step 4: Update internal calls**
```python
# Replace any internal UniversalRAG methods that called the now-deleted methods
# to use self.doc_processor instead
```

**Step 5: Run full test suite**
```bash
pytest tests/ -v --cov=raggy --cov-fail-under=85
```

### Files to Modify

- `raggy.py:1728-2468` (remove ~400 lines of duplication from UniversalRAG)
- `raggy.py:1731` (add DocumentProcessor initialization in UniversalRAG.__init__)
- `raggy.py:1780-1838` (update build method to use doc_processor)
- Tests: Ensure no tests directly call the removed methods

---

## #4: Dangerous os.execv() Usage

**Severity:** CRITICAL
**File:** `raggy.py:1027`
**Estimated Effort:** 2 hours

### Problem

The code uses `os.execv()` to restart the Python process with the virtual environment's interpreter:

```python
# Line 1027
os.execv(str(venv_python), [str(venv_python)] + sys.argv)
```

**Security Risks:**
- **Command Injection**: If `sys.argv` is compromised, arbitrary code execution possible
- **Path Traversal**: `venv_python` path is constructed from `.venv` (could be symlink attack)
- **Process Replacement**: Current process is replaced, losing context
- **Signal Handling**: May bypass cleanup handlers

**Reliability Risks:**
- **File Descriptors**: Open files not properly closed
- **State Loss**: In-memory state is lost
- **Nested Calls**: Could cause infinite recursion if venv detection fails
- **Platform-Specific**: Behavior differs on Windows vs Unix

### Root Cause

Code attempts automatic virtual environment switching, which is:
1. Unexpected behavior (violates principle of least surprise)
2. Unnecessary (user should activate venv themselves)
3. Dangerous (process replacement with external input)

### Acceptance Criteria

- [ ] Remove `os.execv()` call
- [ ] Replace with clear error message instructing user to activate venv
- [ ] Add documentation on proper venv activation
- [ ] Security scan passes (bandit)
- [ ] Tested on Windows and Linux

### Recommended Approach

**Option 1: Remove Auto-Switching (Recommended)**
```python
def setup_dependencies(skip_cache: bool = False, quiet: bool = False):
    """Setup dependencies with optional caching"""

    # Check if we're in a virtual environment
    env_ok, env_issue = check_environment_setup()

    if not env_ok:
        print("ERROR: Virtual environment is not active.")
        print("Please activate your virtual environment first:")
        if sys.platform == "win32":
            print("  .venv\\Scripts\\activate")
        else:
            print("  source .venv/bin/activate")
        sys.exit(1)

    # Verify we're using the venv's Python
    venv_path = Path(".venv")
    if sys.platform == "win32":
        venv_python = venv_path / "Scripts" / "python.exe"
    else:
        venv_python = venv_path / "bin" / "python"

    if str(venv_python.resolve()) != str(Path(sys.executable).resolve()):
        print("ERROR: Not using virtual environment's Python interpreter.")
        print(f"Expected: {venv_python}")
        print(f"Current:  {sys.executable}")
        print("\nPlease activate your virtual environment first.")
        sys.exit(1)

    # Continue with dependency setup...
    required_packages = [...]
```

**Option 2: Use subprocess (Safer, but still not recommended)**
```python
def setup_dependencies(skip_cache: bool = False, quiet: bool = False):
    # ... env checks ...

    # If not in venv, suggest proper activation instead of auto-switching
    if str(venv_python.resolve()) != str(Path(sys.executable).resolve()):
        print("Virtual environment detected but not active.")
        print("Restart with:")
        print(f"  {venv_python} {' '.join(sys.argv)}")
        sys.exit(1)
```

### Files to Modify

- `raggy.py:1009-1028` (remove `os.execv`, add error message)
- `README.md` (add clear venv activation instructions)
- `docs/setup.md` (if exists, document proper workflow)

---

## #5: install_if_missing Function Has Cyclomatic Complexity 20

**Severity:** CRITICAL
**File:** `raggy.py:932-1007`
**Estimated Effort:** 3-4 hours

### Problem

The `install_if_missing` function has a cyclomatic complexity of **20** (threshold: 10), making it:
- Nearly impossible to test all paths
- Prone to bugs in edge cases
- Difficult to understand and maintain
- High risk for regression errors

**Complexity Breakdown:**
- 8 branching conditions (if/elif)
- 2 loops (for packages, for fallback)
- 4 exception handlers
- 3 nested levels
- Special case handling for different package names

### Additional High-Complexity Functions

Also critically complex:
- `UniversalRAG.validate_configuration` - Complexity 19 (raggy.py:2411)
- `DocumentProcessor._process_section` - Complexity 18 (raggy.py:1352)
- `SearchEngine.search` - Complexity 18 (raggy.py:1537)
- `UniversalRAG._process_section` - Complexity 18 (raggy.py:2085) [duplicate of DocumentProcessor]

### Acceptance Criteria

- [ ] `install_if_missing` reduced to complexity < 10
- [ ] Each function has single, clear responsibility
- [ ] All edge cases have unit tests
- [ ] All 5 high-complexity functions refactored

### Recommended Approach

**Refactor install_if_missing:**

```python
# raggy.py or setup/dependencies.py

class PackageInstaller:
    """Handles package installation with caching."""

    def __init__(self, skip_cache: bool = False):
        self.skip_cache = skip_cache
        self.cache = {} if skip_cache else load_deps_cache()
        self.cache_updated = False

    def install_packages(self, packages: List[str]) -> None:
        """Install multiple packages with caching."""
        self._validate_environment()

        for package_spec in packages:
            if not self._install_package(package_spec):
                self._handle_install_failure(package_spec)

        if self.cache_updated:
            save_deps_cache(self.cache)

    def _validate_environment(self) -> None:
        """Validate environment is ready for installation."""
        if not check_uv_available():
            sys.exit(1)

        env_ok, env_issue = check_environment_setup()
        if not env_ok:
            self._report_env_issue(env_issue)
            sys.exit(1)

    def _install_package(self, package_spec: str) -> bool:
        """Install a single package. Returns True if successful."""
        package_name = self._extract_package_name(package_spec)

        if self._is_cached(package_name):
            return True

        if self._is_already_installed(package_name):
            self._update_cache(package_name)
            return True

        return self._perform_install(package_spec, package_name)

    def _extract_package_name(self, package_spec: str) -> str:
        """Extract package name from spec like 'package>=1.0'."""
        return package_spec.split(">=")[0].split("==")[0].split("[")[0]

    def _get_import_name(self, package_name: str) -> str:
        """Get the import name for a package (may differ from package name)."""
        import_name_map = {
            "python-magic-bin": "magic",
            "PyPDF2": "PyPDF2",
        }
        return import_name_map.get(
            package_name,
            package_name.replace("-", "_")
        )

    def _is_cached(self, package_name: str) -> bool:
        """Check if package is in install cache."""
        return (
            not self.skip_cache
            and package_name in self.cache.get("installed", {})
        )

    def _is_already_installed(self, package_name: str) -> bool:
        """Check if package is importable."""
        import_name = self._get_import_name(package_name)
        try:
            spec = importlib.util.find_spec(import_name)
            return spec is not None
        except ImportError:
            return False

    def _perform_install(self, package_spec: str, package_name: str) -> bool:
        """Perform the actual installation."""
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call(["uv", "pip", "install", package_spec])
            self._update_cache(package_name)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_name}: {e}")
            return False

    def _update_cache(self, package_name: str) -> None:
        """Update installation cache."""
        if "installed" not in self.cache:
            self.cache["installed"] = {}
        self.cache["installed"][package_name] = time.time()
        self.cache_updated = True

    def _handle_install_failure(self, package_spec: str) -> None:
        """Handle failed installations with fallbacks."""
        package_name = self._extract_package_name(package_spec)

        if package_name == "python-magic-bin":
            self._try_magic_fallback()

    def _try_magic_fallback(self) -> None:
        """Try alternative magic package on failure."""
        print("Trying alternative magic package...")
        try:
            subprocess.check_call(["uv", "pip", "install", "python-magic"])
            self._update_cache("python-magic-bin")
        except subprocess.CalledProcessError:
            print("Warning: Could not install python-magic. File type detection may be limited.")

    def _report_env_issue(self, env_issue: str) -> None:
        """Report specific environment issues."""
        messages = {
            "virtual_environment": (
                "ERROR: No virtual environment found.\n"
                "Run 'python raggy.py init' to set up the project environment."
            ),
            "pyproject": (
                "ERROR: No pyproject.toml found.\n"
                "Run 'python raggy.py init' to set up the project environment."
            ),
            "invalid_venv": (
                "ERROR: Invalid virtual environment found.\n"
                "Delete .venv directory and run 'python raggy.py init' to recreate it."
            ),
        }
        print(messages.get(env_issue, f"ERROR: Environment issue: {env_issue}"))


# Simplified public API
def install_if_missing(packages: List[str], skip_cache: bool = False):
    """Auto-install required packages if missing using uv."""
    installer = PackageInstaller(skip_cache=skip_cache)
    installer.install_packages(packages)
```

### Files to Modify

- `raggy.py:932-1007` (refactor into PackageInstaller class)
- Add tests for each method in PackageInstaller
- Update documentation

---

## Progress Tracking

- [x] Issue #1: Broken Test Suite - Missing ScoringNormalizer Class ✅ **COMPLETED**
- [ ] Issue #2: God Module - 2901 Lines in Single File
- [ ] Issue #3: Massive Code Duplication - DocumentProcessor vs UniversalRAG
- [ ] Issue #4: Dangerous os.execv() Usage
- [ ] Issue #5: install_if_missing Function Has Cyclomatic Complexity 20

**Total:** 1/5 completed

---

## Recommended Resolution Order

1. **Issue #1** (30 min) - Fix tests first to enable validation
2. **Issue #5** (3-4 hours) - Reduce complexity to make code testable
3. **Issue #3** (4-6 hours) - Remove duplication to simplify codebase
4. **Issue #4** (2 hours) - Remove security vulnerability
5. **Issue #2** (2-3 days) - Split into modules (requires most time)

**Total Critical Issues Effort:** 16-24 hours of focused development work.
