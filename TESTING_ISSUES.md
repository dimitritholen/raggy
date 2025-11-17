# CLI Testing Issues Found

## Testing Date: 2025-11-17

## Critical Issues

### 1. PyPDF2 Import Error (CRITICAL)
**File:** `raggy/setup/environment.py:56`
**Issue:** Code attempts to import `PyPDF2` but the package is `pypdf` (as specified in pyproject.toml line 28)
**Impact:** When raggy is installed system-wide, ALL commands fail with "Local .venv exists but is not activated" error
**Root Cause:**
- pyproject.toml specifies: `pypdf>=6.2.0`
- environment.py checks: `import PyPDF2`
- PyPDF2 was deprecated and replaced with pypdf
- The import fails even though pypdf is installed
- This causes the environment check to return "virtual_environment" error
- All CLI commands then refuse to run

**Error Flow:**
1. User runs `raggy build` (or any command)
2. `check_dependencies()` is called (dependencies.py:275)
3. `check_environment_setup()` is called (environment.py:33)
4. Line 47: `in_venv = False` (not in a virtual environment)
5. Lines 64-68: Try to import dependencies
6. Line 56: `import PyPDF2` → FAILS (ModuleNotFoundError)
7. Falls through to line 76
8. Line 76-77: Checks if .venv exists in current directory → TRUE
9. Line 90: Returns `False, "virtual_environment"`
10. dependencies.py:288: Prints error and exits

**Fix Required:**
Change line 56 in raggy/setup/environment.py from:
```python
import PyPDF2
```
to:
```python
import pypdf  # or: from pypdf import PdfReader
```

**Also check:**
- raggy/setup/dependencies.py:312 (uses PyPDF2>=3.0.0)
- Any other files that reference PyPDF2

### 2. Inconsistent Instructions in init Command (MEDIUM)
**File:** `raggy/setup/environment.py` (init command output)
**Issue:** When raggy is installed system-wide via pip, the `raggy init` command outputs instructions that say:
```
3. Run: python raggy.py build
4. Run: python raggy.py search "your query"
```

**Impact:** Users are confused about how to use the installed command
**Expected:** Instructions should say:
```
3. Run: raggy build
4. Run: raggy search "your query"
```

**Fix Required:** Update the output messages in the init command to use `raggy` instead of `python raggy.py` when detecting system-wide installation.

### 3. Virtual Environment Creation in System-Wide Installation (MEDIUM)
**File:** `raggy/setup/environment.py` (init command)
**Issue:** `raggy init` creates a .venv directory even when raggy is already installed system-wide
**Impact:**
- Unnecessary virtual environment created
- Clutters user's project directory
- Creates confusion about whether to use venv or system installation

**Expected Behavior:**
- When raggy is installed system-wide (pip install raggy), init should NOT create a .venv
- Only create .venv when raggy.py is being used directly for development

**Fix Required:**
- Detect if raggy is installed as a package (check if running as `raggy` command vs `python raggy.py`)
- Skip venv creation if installed system-wide
- Update instructions accordingly

## Testing Results Summary

**Commands Tested:**
- ✅ `raggy --help` - Works
- ✅ `raggy --version` - Works (needs testing)
- ❌ `raggy init` - Works but creates unnecessary venv
- ❌ `raggy build` - Fails with venv error
- ❌ `raggy search` - Fails with venv error
- ❌ `raggy status` - Fails with venv error
- ❌ All other commands - Expected to fail with same venv error

**Workaround:**
Run commands from directories without .venv, but this defeats the purpose of system-wide installation.

## Reproduction Steps

1. Install raggy: `pip install -e .`
2. Create test directory: `mkdir -p /tmp/test/docs && cd /tmp/test`
3. Run: `raggy init --non-interactive`
4. Run: `raggy build` → FAILS with venv error
5. Remove .venv: `rm -rf .venv`
6. Run: `raggy build` → Still FAILS (because /tmp/test has no .venv but CWD might have one)

## Recommended Fix Priority

1. **CRITICAL:** Fix PyPDF2 import (environment.py:56 and dependencies.py:312)
2. **MEDIUM:** Update init command instructions
3. **MEDIUM:** Skip venv creation for system-wide installations
4. **LOW:** Add detection for system-wide vs development mode
