# Changelog

All notable changes to the raggy project will be documented in this file.

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
