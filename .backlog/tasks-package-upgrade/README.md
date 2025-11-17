# Package Upgrade Project - Task Manager v3.1

**Generated:** 2025-11-16
**Project:** Raggy v2.0.0 Dependency Upgrade
**Total Estimated Time:** 23.5-47 hours
**Risk Level:** MEDIUM-HIGH

---

## Overview

This directory contains an **incremental task breakdown** for upgrading all Raggy dependencies from legacy versions (Python 3.8 era) to the latest versions as of November 2025.

### Why Incremental?

Instead of generating 100+ tasks upfront, Task Manager v3.1 uses an **incremental workflow**:

1. ✅ **Initial phases generated** (Phase 0-1): Foundation and Python version upgrade
2. ⏳ **Remaining phases on-demand** (Phase 2-5): Generated after completing earlier phases
3. 🎯 **Focus**: Work on 1-2 phases at a time without overwhelming task lists
4. 🔄 **Adaptive**: Adjust plans based on results from earlier phases

---

## Quick Start

### Step 1: Execute Phase 0 (Pre-Upgrade Foundation)

**File:** `00-pre-upgrade-foundation.yaml`

**Purpose:** Create backups, baselines, and safety infrastructure

**Tasks:** 7 tasks, ~1-2 hours

**Start with:**
```bash
# Task PRE-001: Create feature branch
git checkout -b feature/package-upgrades
git push -u origin feature/package-upgrades
```

**Then continue through:** PRE-002 → PRE-007

**⚠️ CRITICAL CHECK:** Task PRE-006 verifies Python >=3.10
- If you have Python < 3.10, **STOP** and upgrade Python first
- pytest 9.x requires Python >=3.10 (blocking requirement)

---

### Step 2: Execute Phase 1 (Python Version Upgrade)

**File:** `01-python-version-upgrade.yaml`

**Purpose:** Update Python version constraints from 3.8 to 3.10

**Tasks:** 11 tasks, ~1-2 hours

**Key changes:**
- `pyproject.toml`: `requires-python = ">=3.10"`
- `pyproject.toml`: ruff `target-version = "py310"`
- `pyproject.toml`: mypy `python_version = "3.10"`
- Remove Python 3.8/3.9 classifiers
- Update CHANGELOG with breaking change notice

**Final task (PY-011):** Commit all changes

---

### Step 3: Request Phase 2 Tasks

**After completing Phase 1**, tell Claude:

```
Continue to Phase 2 (Core Dependencies)
```

Claude will then generate:
- `02-core-dependencies-upgrade.yaml` (15-25 tasks, ~8-16 hours)
- Updated `state.yaml` tracking file
- Updated `execution-plan.yaml`

---

## File Structure

```
.backlog/tasks-package-upgrade/
├── README.md                        # This file
├── state.yaml                       # Current progress tracking
├── tech-stack.yaml                  # Tech stack detection
├── execution-plan.yaml              # Overall execution plan
│
├── 00-pre-upgrade-foundation.yaml   # ✅ Phase 0 (generated)
├── 01-python-version-upgrade.yaml   # ✅ Phase 1 (generated)
│
├── 02-core-dependencies-upgrade.yaml     # ⏳ Phase 2 (on-demand)
├── 03-optional-dependencies-upgrade.yaml # ⏳ Phase 3 (on-demand)
├── 04-dev-dependencies-upgrade.yaml      # ⏳ Phase 4 (on-demand)
├── 05-build-system-upgrade.yaml          # ⏳ Phase 5 (on-demand)
│
├── ROLLBACK.md                      # Rollback procedures (from PRE-007)
├── baseline-packages.txt            # Package versions before upgrade
├── baseline-test-results.txt        # Test results before upgrade
├── baseline-ruff.txt                # Ruff results before upgrade
├── baseline-mypy.txt                # Mypy results before upgrade
├── baseline-bandit.txt              # Bandit results before upgrade
└── chromadb-backup/                 # ChromaDB database backup
```

---

## Phase Overview

| Phase | Name | Status | Est. Hours | Priority | Tasks |
|-------|------|--------|------------|----------|-------|
| 0 | Pre-Upgrade Foundation | NOT_STARTED | 1-2 | CRITICAL | 7 |
| 1 | Python Version Upgrade | NOT_STARTED | 1-2 | CRITICAL | 11 |
| 2 | Core Dependencies | NOT_GENERATED | 8-16 | HIGH | 15-25 |
| 3 | Optional Dependencies | NOT_GENERATED | 2-4 | MEDIUM | 8-12 |
| 4 | Dev Dependencies | NOT_GENERATED | 4-8 | HIGH | 10-15 |
| 5 | Build System | NOT_GENERATED | 0.5-1 | LOW | 3-5 |

**Total:** 23.5-47 hours across 6 phases

---

## Critical Compatibility Issues

### 1. Python 3.10 Requirement (BLOCKING)
- **Current:** `requires-python = ">=3.8"`
- **Required:** `requires-python = ">=3.10"`
- **Why:** pytest 9.x requires Python >=3.10
- **Impact:** Blocks all dev dependency upgrades
- **Resolution:** Phase 1 updates configuration; user must have Python 3.10+ installed

### 2. PyPDF2 Deprecation (HIGH RISK)
- **Current:** `PyPDF2>=3.0.0`
- **Status:** Deprecated and archived
- **Migration:** `PyPDF2` → `pypdf>=5.2.0`
- **Impact:** All PDF extraction code must be updated
- **Resolution:** Phase 2 will handle migration

### 3. ChromaDB Irreversible Migration (HIGH RISK)
- **Current:** `chromadb>=0.4.0`
- **Target:** `chromadb>=1.3.3`
- **Risk:** Database migration is IRREVERSIBLE
- **Mitigation:** Phase 0 creates backup with PRE-005
- **Resolution:** Phase 2 will run `chroma-migrate` tool

---

## Safety Measures

### Git Branch Isolation
- All work happens in `feature/package-upgrades` branch
- Main branch remains untouched
- Easy rollback: `git branch -D feature/package-upgrades`

### Baseline Snapshots
- **Package versions:** `baseline-packages.txt`
- **Test results:** `baseline-test-results.txt`
- **Linting:** `baseline-ruff.txt`
- **Type checking:** `baseline-mypy.txt`
- **Security:** `baseline-bandit.txt`

### Database Backup
- ChromaDB backup in `chromadb-backup/`
- Restores database if migration fails
- Created by PRE-005

### Rollback Documentation
- Full rollback procedures in `ROLLBACK.md`
- Created by PRE-007
- Recovery time: 5-10 minutes

---

## Quality Gates

Each phase must pass these checks before proceeding:

✅ **Tests:** `pytest --verbose` (compare with baseline)
✅ **Linting:** `ruff check .` (compare with baseline)
✅ **Types:** `mypy raggy.py` (compare with baseline)
✅ **Commit:** Changes committed to feature branch
✅ **CHANGELOG:** Updated with changes

---

## Workflow Pattern

```
┌─────────────────────────────────────────────────────────────┐
│ INCREMENTAL WORKFLOW PATTERN                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Execute current phase tasks (00, 01, etc.)              │
│ 2. Verify quality gates pass                               │
│ 3. Commit changes to feature branch                        │
│ 4. Request next phase: "Continue to Phase N"               │
│ 5. Claude generates next phase tasks                       │
│ 6. Repeat until all phases complete                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Commands Reference

### Phase 0 Commands
```bash
# PRE-001: Create branch
git checkout -b feature/package-upgrades
git push -u origin feature/package-upgrades

# PRE-002: Baseline packages
pip freeze > .backlog/tasks-package-upgrade/baseline-packages.txt

# PRE-003: Baseline tests
pytest --verbose > .backlog/tasks-package-upgrade/baseline-test-results.txt 2>&1 || true

# PRE-004: Baseline quality checks
ruff check . > .backlog/tasks-package-upgrade/baseline-ruff.txt 2>&1 || true
mypy raggy.py > .backlog/tasks-package-upgrade/baseline-mypy.txt 2>&1 || true
bandit -r . -ll > .backlog/tasks-package-upgrade/baseline-bandit.txt 2>&1 || true

# PRE-005: Backup ChromaDB
mkdir -p .backlog/chromadb-backup
cp -r .raggy/chroma .backlog/chromadb-backup/ 2>/dev/null || echo 'No ChromaDB data'

# PRE-006: Verify Python version
python --version
python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || echo 'NEEDS UPGRADE'
```

### Phase 1 Key Changes
See `01-python-version-upgrade.yaml` for detailed edits to `pyproject.toml`

---

## Getting Next Phases

After completing a phase, request the next one:

```
Continue to Phase 2 (Core Dependencies)
Continue to Phase 3 (Optional Dependencies)
Continue to Phase 4 (Dev Dependencies)
Continue to Phase 5 (Build System)
```

Claude will generate the task file and update tracking files.

---

## Troubleshooting

### If Python < 3.10
**Problem:** PRE-006 fails, cannot proceed

**Solution:**
1. Install Python 3.10+ using your system package manager
2. Update virtual environment: `python3.10 -m venv .venv`
3. Activate: `source .venv/bin/activate`
4. Retry PRE-006

### If pre-commit hooks fail
**Problem:** PY-011 commit fails due to hooks

**Solution:**
1. Read hook error messages
2. Fix reported issues (usually linting/formatting)
3. Stage fixes: `git add -u`
4. Retry commit (hooks run automatically)
5. **NEVER** use `git commit --no-verify`

### If ChromaDB migration fails
**Problem:** Phase 2 migration corrupts database

**Solution:**
1. Follow rollback procedure in `ROLLBACK.md`
2. Restore from `chromadb-backup/`
3. Report issue for investigation
4. Consider alternative migration approach

---

## Support Files

- **Source PRD:** `./docs/artifacts/PACKAGE_UPGRADE.md`
- **Tracking:** `state.yaml`
- **Tech Stack:** `tech-stack.yaml`
- **Execution Plan:** `execution-plan.yaml`
- **Rollback Procedures:** `ROLLBACK.md`

---

## Project Status

**Created:** 2025-11-16
**Current Phase:** 0 (Pre-Upgrade Foundation)
**Phases Generated:** 2 of 6
**Next Action:** Execute Phase 0 tasks (PRE-001 through PRE-007)

**Progress:** 🟦🟦⬜⬜⬜⬜ (2/6 phases generated)

---

## Notes

- This is an **incremental workflow** - phases are generated on-demand
- Each phase builds on the previous phase
- Rollback procedures are available at every step
- Quality gates ensure no regressions
- Feature branch keeps main branch safe
- ChromaDB backup protects against irreversible migration

**When in doubt:** Check `ROLLBACK.md` for recovery procedures
