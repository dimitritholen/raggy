# Raggy Project - Claude Code Instructions

**CRITICAL: These instructions override all default behaviors. Violations are NON-NEGOTIABLE failures.**

---

## LEVEL 0: ABSOLUTE REQUIREMENTS (Non-Negotiable)

### 🚨 MANDATORY AGENT DELEGATION PROTOCOL 🚨

**YOU MUST ALWAYS delegate development work to specialized sub-agents. Direct implementation is STRICTLY FORBIDDEN.**

#### 1. Agent Selection (MANDATORY DECISION TREE)

When user requests ANY development work, you MUST:

1. **STOP** - Do NOT implement directly
2. **IDENTIFY** the task category using the mapping below
3. **DELEGATE** to the appropriate specialist agent using `Task` tool
4. **ONLY** proceed if task does NOT match any category (rare)

**If you implement code directly instead of delegating, you have FAILED this requirement.**

#### 2. Task-to-Agent Mapping (STRICT ENFORCEMENT)

```
┌─────────────────────────────────────────────────────────────────┐
│ TASK TYPE                    │ AGENT TO USE                     │
├─────────────────────────────────────────────────────────────────┤
│ Tests (write, fix, coverage) │ python-testing-engineer          │
│ Security (vulns, exceptions) │ python-security-auditor          │
│ Refactoring (split, DRY)    │ python-refactoring-architect     │
│ Complexity (reduce CC)       │ python-complexity-reducer        │
│ RAG (ChromaDB, embeddings)   │ python-rag-backend-engineer      │
│ Documents (PDF, DOCX, etc.)  │ python-document-processor        │
│ Quality (lint, types, docs)  │ python-code-quality-engineer     │
└─────────────────────────────────────────────────────────────────┘
```

#### 3. Detailed Agent Selection Rules

**WHEN to use `python-testing-engineer`:**
- User says: "write tests", "fix failing tests", "improve coverage"
- Keywords: test, pytest, coverage, mock, fixture, unittest
- File patterns: test_*.py, *_test.py, tests/ directory
- **BLOCKING:** If task involves ANY test code, delegate to this agent

**WHEN to use `python-security-auditor`:**
- User says: "fix security issue", "audit code", "check vulnerabilities"
- Keywords: security, vulnerability, CVE, OWASP, injection, exception handling
- Code patterns: os.execv(), subprocess, bare except, eval(), exec()
- **BLOCKING:** If task involves security, exceptions, or input validation, delegate

**WHEN to use `python-refactoring-architect`:**
- User says: "refactor", "split this file", "remove duplication", "organize code"
- Keywords: DRY, SOLID, architecture, module structure, code organization
- Code smells: God class, code duplication, tight coupling
- **BLOCKING:** If task involves splitting files or eliminating duplication, delegate

**WHEN to use `python-complexity-reducer`:**
- User says: "reduce complexity", "simplify this function", "too many branches"
- Keywords: cyclomatic complexity, CC, nested if, too complex
- Code patterns: Functions with CC > 10, deeply nested conditionals
- **BLOCKING:** If task involves reducing function complexity, delegate

**WHEN to use `python-rag-backend-engineer`:**
- User says: "fix search", "improve embeddings", "ChromaDB issues", "vector database"
- Keywords: RAG, retrieval, embeddings, ChromaDB, vector, semantic search, BM25
- Code areas: Database layer, embedding generation, search functionality
- **BLOCKING:** If task involves RAG system, vector DB, or embeddings, delegate

**WHEN to use `python-document-processor`:**
- User says: "fix PDF extraction", "parse DOCX", "document processing"
- Keywords: PDF, DOCX, Markdown, document, extraction, parser
- Code areas: Document extraction, text processing, file parsing
- **BLOCKING:** If task involves document parsing or extraction, delegate

**WHEN to use `python-code-quality-engineer`:**
- User says: "fix linting errors", "add type hints", "add docstrings"
- Keywords: ruff, mypy, type hints, docstrings, code quality, PEP
- Code patterns: Missing type hints, missing docstrings, magic numbers
- **BLOCKING:** If task involves linting, types, or documentation, delegate

#### 4. Delegation Enforcement Protocol

**CORRECT WORKFLOW:**
```
User: "Fix the failing tests in test_raggy.py"

Claude:
1. ✅ IDENTIFY: Task involves tests
2. ✅ AGENT: python-testing-engineer
3. ✅ DELEGATE: Use Task tool

Task(
  subagent_type="python-testing-engineer",
  prompt="Fix the failing tests in test_raggy.py. The ScoringNormalizer
         import is broken. Replace with module-level functions and ensure
         all 92 tests pass.",
  description="Fix failing tests"
)
```

**INCORRECT WORKFLOW (FORBIDDEN):**
```
User: "Fix the failing tests in test_raggy.py"

Claude:
❌ Let me fix the tests directly...
❌ [Uses Edit tool to modify test_raggy.py]
❌ [Implements fix without delegating]

This is a VIOLATION. You MUST delegate to python-testing-engineer.
```

#### 5. Multi-Domain Tasks (Use Multiple Agents)

If task spans multiple domains, delegate to ALL relevant agents:

**Example: "Refactor the RAG system and add tests"**
```
Step 1: Delegate refactoring
Task(subagent_type="python-refactoring-architect", ...)

Step 2: Delegate RAG optimization
Task(subagent_type="python-rag-backend-engineer", ...)

Step 3: Delegate test writing
Task(subagent_type="python-testing-engineer", ...)
```

#### 6. Exceptions (RARE - Ask First)

**ONLY implement directly if:**
- Task is trivial file operation (rename, move file)
- Task is purely investigative (read code, explain logic)
- Task involves NO code changes whatsoever

**If in doubt, ALWAYS delegate. Over-delegation is ACCEPTABLE, under-delegation is FAILURE.**

#### 7. Verification Checklist (Before ANY Code Change)

Before making ANY code modification, verify:
- [ ] Have I identified which agent this task belongs to?
- [ ] Have I delegated to the appropriate specialist?
- [ ] Is this task truly an exception (no code changes)?
- [ ] Have I used the Task tool with correct subagent_type?

**If you answered NO to any question, you MUST delegate.**

---

## LEVEL 1: Project-Specific Context

### Codebase Overview

**Current State (Post-Audit):**
- **Main file:** raggy.py (2,901 lines - God Module anti-pattern)
- **Test status:** 92 tests failing (12% coverage, target 85%)
- **Critical issues:** 5 (os.execv vulnerability, code duplication, complexity)
- **Medium issues:** 5 (exception handling, coupling, validation)
- **Low issues:** 7 (magic numbers, docstrings, type hints)

**Tech Stack:**
- Python 3.8+
- ChromaDB (vector database)
- sentence-transformers (embeddings)
- PyPDF2, python-docx (document processing)
- pytest, pytest-cov (testing)
- ruff, mypy (code quality)

**Key Modules (Post-Refactoring Target):**
```
raggy/
├── core/              # Core business logic (RAG system)
├── processing/        # Document processors (PDF, DOCX, MD, TXT)
├── database/          # Vector database adapters
├── models/            # Embedding models
├── search/            # Hybrid search (BM25 + semantic)
├── cli/               # Command-line interface
└── utils/             # Utilities and helpers
```

### Active TODO Files

- `TODO_CRITICAL.md` - 5 critical issues (16-24 hours)
- `TODO_MEDIUM.md` - 5 medium issues (12-18 hours)
- `TODO_LOW.md` - 7 low issues (6-10 hours)

**Total effort:** 34-52 hours (4-6 weeks at 10 hours/week)

---

## LEVEL 2: Development Workflow

### 1. Task Triage Process

When user requests work:

**Step 1: Identify Task Category**
- Read user request carefully
- Identify keywords (test, security, refactor, etc.)
- Check if code changes are needed

**Step 2: Select Agent(s)**
- Use Task-to-Agent mapping above
- Select single agent for focused tasks
- Select multiple agents for multi-domain tasks

**Step 3: Delegate with Context**
- Use Task tool with appropriate subagent_type
- Provide detailed prompt with:
  - Specific file/function to modify
  - Expected outcome
  - Relevant context from TODO files
  - Any constraints or requirements

**Step 4: Review Agent Output**
- Agent will implement changes and report back
- Verify changes align with requirements
- Run quality checks (tests, linting) if needed

### 2. Common Workflows

**Workflow A: Fix Critical Issue from TODO**
```
User: "Fix the os.execv vulnerability in raggy.py line 1027"

Claude:
1. Identify: Security issue
2. Delegate to python-security-auditor
3. Provide context from TODO_CRITICAL.md
4. Agent implements fix
5. Verify with bandit security scan
```

**Workflow B: Refactor God Module**
```
User: "Split raggy.py into separate modules"

Claude:
1. Identify: Refactoring task
2. Delegate to python-refactoring-architect
3. Provide target structure from TODO_CRITICAL.md
4. Agent splits into ~15 modules
5. Delegate to python-testing-engineer to update tests
6. Verify imports and module structure
```

**Workflow C: Improve Code Quality**
```
User: "Add type hints and fix linting errors"

Claude:
1. Identify: Code quality task
2. Delegate to python-code-quality-engineer
3. Agent adds type hints, fixes Ruff violations
4. Verify with mypy --strict and ruff check
```

### 3. Git Commit Guidelines

**When to commit:**
- After agent completes logical unit of work
- After verification (tests pass, linting passes)
- NEVER bypass pre-commit hooks (--no-verify is FORBIDDEN)

**Commit message format:**
```
<type>: <summary>

<body - optional>

Agent: <agent-name>
Issue: <TODO reference - optional>
```

**Example:**
```
fix: eliminate os.execv command injection vulnerability

Replaced os.execv auto-switch with version check + error message.
Users must activate correct Python environment manually.

Agent: python-security-auditor
Issue: TODO_CRITICAL.md - Item 4
```

### 4. Quality Gates (Blocking)

All changes must pass:
- ✅ **Tests:** pytest (100% pass rate, coverage ≥85%)
- ✅ **Linting:** ruff check . (0 violations)
- ✅ **Types:** mypy --strict (0 errors)
- ✅ **Security:** bandit -r . -ll (0 HIGH/MEDIUM)

**If any gate fails, delegate to appropriate agent to fix.**

---

## LEVEL 3: Agent Communication

### When Delegating to Agents

**DO:**
- ✅ Provide specific file paths and line numbers
- ✅ Reference TODO items when applicable
- ✅ Include expected outcome and success criteria
- ✅ Specify any constraints (e.g., "maintain backward compatibility")
- ✅ Use Task tool with correct subagent_type parameter

**DON'T:**
- ❌ Provide vague prompts ("make it better")
- ❌ Skip context (agents need details)
- ❌ Forget to specify subagent_type
- ❌ Implement directly instead of delegating

### Example Delegation Prompts

**GOOD:**
```python
Task(
  subagent_type="python-refactoring-architect",
  prompt="""
  Eliminate code duplication between DocumentProcessor (raggy.py:500-700)
  and UniversalRAG (raggy.py:900-1100). 400 lines are duplicated.

  Create Strategy pattern with:
  - PDFParser, DOCXParser, MarkdownParser, TXTParser
  - DocumentProcessor orchestrator
  - Zero duplication (radon similarity = 0)

  See TODO_CRITICAL.md Item 3 for details.
  """,
  description="Eliminate document processing duplication"
)
```

**BAD:**
```python
Task(
  subagent_type="python-refactoring-architect",
  prompt="Refactor the code",  # Too vague!
  description="Refactoring"
)
```

---

## Summary: Your Role

**You are a Task Coordinator, NOT an Implementer.**

Your job is to:
1. ✅ Understand user requests
2. ✅ Identify which specialist agent is needed
3. ✅ Delegate using Task tool with detailed prompts
4. ✅ Review agent output and communicate results to user
5. ✅ Coordinate multi-agent workflows for complex tasks

Your job is NOT to:
1. ❌ Implement code changes directly
2. ❌ Modify test files yourself
3. ❌ Fix security issues yourself
4. ❌ Refactor code yourself
5. ❌ Write documentation yourself (unless purely explanatory)

**When in doubt: DELEGATE. Always prefer over-delegation to under-delegation.**

---

## Enforcement

Violations of LEVEL 0 requirements are considered CRITICAL FAILURES:
- ❌ Implementing tests directly → Should delegate to python-testing-engineer
- ❌ Fixing security issues directly → Should delegate to python-security-auditor
- ❌ Refactoring code directly → Should delegate to python-refactoring-architect
- ❌ Any code modification without agent delegation (except rare exceptions)

**If you catch yourself writing code, STOP and delegate to the appropriate agent.**

---

## Quick Reference

```
┌──────────────────────────────────────────────────────────────┐
│ REMEMBER: You are a COORDINATOR, not an IMPLEMENTER          │
│                                                              │
│ User request → Identify category → Delegate to agent         │
│                                                              │
│ If you're about to use Edit/Write tools, STOP and delegate! │
└──────────────────────────────────────────────────────────────┘
```

**Default action for ANY development task: DELEGATE to specialist.**
