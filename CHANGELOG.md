# Changelog

All notable changes to the raggy project will be documented in this file.

## [Unreleased]

### Changed (BREAKING)
- **BREAKING**: Minimum Python version increased from 3.8 to 3.10
  - Required for pytest 9.x and other modern dependency upgrades
  - Python 3.8 reached end-of-life in October 2024
  - Python 3.9 reaches end-of-life in October 2025

- **BREAKING**: Migrated from deprecated PyPDF2 to pypdf 6.2.0
  - PyPDF2 is officially deprecated and no longer maintained
  - pypdf is the official successor maintained by the same team
  - API is backwards compatible for PDF reading operations
  - All PDF extraction functionality preserved

- **BREAKING**: Upgraded ChromaDB from 0.4.x to 1.3.3 (irreversible migration)
  - **CRITICAL**: Migration is IRREVERSIBLE - backup databases before upgrading
  - `.persist()` method removed (ChromaDB now auto-saves instantly)
  - Embeddings now return 2D NumPy arrays (not Python lists)
  - `get_or_create()` ignores metadata if collection exists
  - See rollback instructions below if migration fails

### Changed
- Upgraded sentence-transformers from 2.2.x to 5.1.2
  - Backwards compatible upgrade (no code changes required)
  - New optional APIs: `encode_query()` and `encode_document()` for information retrieval
  - All existing `.encode()` usage continues to work

- Upgraded python-docx from 1.0.x to 1.2.0
  - Minor version upgrade with bug fixes
  - Requires Python >=3.9 (satisfied by Python 3.10 requirement)
  - No breaking API changes

### Optional Dependencies (User Opt-In)
- Updated optional dependency: pinecone-client from 2.0.0 to 6.0.0
  - Optional feature for cloud vector storage ([pinecone], [cloud-stores], [cloud], [all] extras)
  - Major version upgrade (2.x → 6.x) with enhanced API
  - NOT used in Raggy core - only affects users who install [pinecone] extra

- Updated optional dependency: supabase from 2.0.0 to 2.24.0
  - Optional feature for cloud vector storage ([supabase], [cloud-stores], [cloud], [all] extras)
  - Minor version update within stable 2.x series (backwards compatible)
  - NOT used in Raggy core - only affects users who install [supabase] extra

- Updated optional dependency: openai from 1.0.0 to 2.8.0
  - Optional feature for cloud embeddings ([openai], [cloud-embeddings], [cloud], [all] extras)
  - Major version upgrade (1.x → 2.x) with backwards-compatible API
  - Used in raggy/embeddings/openai_provider.py (v1.x API maintained in v2.x)

- Added version constraint to python-magic (>=0.4.14)
  - Optional feature for file type detection ([magic-unix], [all] extras)
  - Consistency improvement (matches python-magic-bin version)
  - Platform-specific: python-magic (Unix), python-magic-bin (Windows)

**Note**: All optional dependency upgrades do NOT affect core Raggy functionality. Users must explicitly install extras (e.g., `pip install raggy[cloud]`) to use these features. No breaking changes impact existing code.

### Infrastructure
- Updated pyproject.toml Python version constraints
- Updated ruff target version to py310
- Updated mypy Python version to 3.10
- Removed Python 3.8 and 3.9 from supported version classifiers
- Updated core dependency constraints in pyproject.toml:
  - `pypdf>=6.2.0` (was `PyPDF2>=3.0.0`)
  - `chromadb>=1.3.3` (was `chromadb>=0.4.0`)
  - `sentence-transformers>=5.1.2` (was `sentence-transformers>=2.2.0`)
  - `python-docx>=1.2.0` (was `python-docx>=1.0.0`)

### Migration Notes

#### ChromaDB Migration (0.4.x → 1.3.3)
**IRREVERSIBLE MIGRATION - Follow these steps carefully:**

1. **Backup databases** (MANDATORY before upgrading):
   ```bash
   timestamp=$(date +%Y%m%d-%H%M%S)
   for db in $(find . -type d -name ".chromadb" 2>/dev/null); do
     cp -r "$db" "${db}.backup-${timestamp}"
     echo "Backed up: $db → ${db}.backup-${timestamp}"
   done
   ```

2. **Upgrade ChromaDB**:
   ```bash
   pip install 'chromadb>=1.3.3'
   ```

3. **Verify functionality**:
   ```bash
   pytest tests/test_memory.py -v
   ```

**If migration fails - Rollback procedure**:
1. Stop all Raggy processes
2. Delete migrated `.chromadb` directory
3. Restore from backup:
   ```bash
   rm -rf .chromadb
   cp -r .chromadb.backup-YYYYMMDD-HHMMSS .chromadb
   ```
4. Downgrade ChromaDB:
   ```bash
   pip install 'chromadb>=0.4.0,<1.0.0'
   ```

#### PyPDF2 Migration (deprecated → pypdf)
No special migration needed - drop-in replacement. If issues occur:
1. Revert code changes: `git checkout raggy/core/document.py`
2. Reinstall PyPDF2: `pip install 'PyPDF2>=3.0.0'`
3. Update pyproject.toml back to `PyPDF2>=3.0.0`

## 2025-11-13

### Fixed
- **Exception Handling Security**: Replaced all 18 bare `except Exception` handlers with specific exception types
  - Eliminated OWASP A09:2021 violations (Security Logging Failures)
  - Removed all silent failure patterns (bare `pass` statements)
  - Implemented fail-fast design - programming errors now crash as intended
  - Added specific handlers: `FileNotFoundError`, `PermissionError`, `yaml.YAMLError`, `UnicodeDecodeError`, etc.
  - Files modified: `raggy/config/loader.py`, `raggy/core/database.py`, `raggy/core/document.py`, `raggy/core/rag.py`, `raggy/core/search.py`, `raggy/setup/dependencies.py`, `raggy_cli.py`
  - Security verified: 0 HIGH severity issues in bandit scan
  - Issue #1 from TODO_MEDIUM.md resolved (2-3 hours effort)

- **Silent Exception Logging**: Replaced 4 bare `pass` statements with proper logging
  - Added context-aware logging for cache operations and session file handling
  - All logging respects quiet mode (`quiet=True` for debug-level issues)
  - Files modified: `raggy/config/cache.py`, `raggy/utils/updates.py`
  - No silent failures remain in codebase (verified with `rg` search)
  - Issue #2 from TODO_MEDIUM.md resolved (1 hour effort)

### Changed
- **DEPRECATED raggy.py**: Converted monolithic 2,919-line file to thin 243-line wrapper
  - Reduced from 106 KB to 6.6 KB (94% reduction)
  - All functionality now imported from modular `raggy/` package
  - Added prominent deprecation warnings (will remove in v3.0.0)
  - Maintained 100% backward compatibility - all existing scripts continue working
  - Shows migration instructions pointing to `raggy_cli.py`
  - Eliminated massive code duplication between raggy.py and raggy/ package

### Technical Details
- **Before**: 2,919 lines with CC=18-20 functions, 106 KB file size
- **After**: 243 lines with CC=1 functions (simple delegates), 6.6 KB file size
- **Imports preserved**: All classes, functions, and constants re-exported for compatibility
- **Entry points preserved**: main(), parse_args(), _determine_model() all delegate to raggy_cli
- **User impact**: Zero breaking changes, clear migration path shown

## 2025-11-12

### Added
- **Specialized Sub-Agents**: Created 7 production-grade Python agents in `.claude/agents/`:
  - `python-testing-engineer.md` - Fix broken tests, achieve 85% coverage
  - `python-refactoring-architect.md` - Decompose God Module, eliminate duplication
  - `python-complexity-reducer.md` - Reduce cyclomatic complexity from 20 to ≤10
  - `python-security-auditor.md` - Fix os.execv vulnerability, OWASP compliance
  - `python-rag-backend-engineer.md` - ChromaDB abstraction, hybrid search
  - `python-document-processor.md` - PDF/DOCX/Markdown extraction with Strategy pattern
  - `python-code-quality-engineer.md` - Ruff linting, mypy strict, docstrings

- **Project Instructions**: Created `.claude/CLAUDE.md` with mandatory agent delegation protocol:
  - LEVEL 0 enforcement: MUST delegate to specialists (direct implementation forbidden)
  - Task-to-Agent mapping with detailed decision tree
  - Verification checklist before any code changes
  - Multi-domain task coordination guidelines
  - Quality gates and commit guidelines

### Fixed
- **Broken Test Suite**: Fixed ImportError in `tests/test_raggy.py` preventing all 92 tests from running
  - Replaced non-existent `ScoringNormalizer` class import with module-level functions
  - Updated 20 function calls to use `normalize_cosine_distance`, `normalize_hybrid_score`, `interpret_score`
  - All 5 scoring normalization tests now passing (100%)
  - Test suite operational: 116 tests collected (up from 0)
  - Coverage improved: 15% (up from 12%, target: 85%)
  - Issue #1 from TODO_CRITICAL.md resolved

### Context
- Agents generated based on comprehensive code audit findings
- Total remediation effort: 34-52 hours (4-6 weeks at 10 hours/week)
- Each agent includes:
  - Maximum enforcement (BLOCKING quality gates)
  - LEVEL 0/1/2 constraint hierarchy
  - Anti-hallucination safeguards
  - Few-shot examples (BEFORE/AFTER)
  - 5 blocking quality gates each
  - Context7 verification for external APIs
