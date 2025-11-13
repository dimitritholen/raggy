# Changelog

All notable changes to the raggy project will be documented in this file.

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
