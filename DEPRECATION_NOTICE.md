# DEPRECATION NOTICE: raggy.py → raggy_cli.py

## Summary

The monolithic `raggy.py` file has been **deprecated** and converted to a thin wrapper that imports from the modular `raggy/` package. This eliminates massive code duplication and improves maintainability.

## Migration Timeline

- **v2.0.0** (Current): `raggy.py` deprecated but functional with warnings
- **v3.0.0** (Future): `raggy.py` will be removed entirely

## What Changed

### Before (Monolithic)
- **File**: `raggy.py`
- **Size**: 105.8 KB
- **Lines**: 2,919
- **Structure**: Single massive file with all code
- **Complexity**: Multiple functions with CC=18-20
- **Problem**: Code duplicated between raggy.py and raggy/ package

### After (Modular)
- **File**: `raggy.py` (thin wrapper) + `raggy/` package
- **Size**: 6.6 KB (wrapper only)
- **Lines**: 243 (wrapper only)
- **Structure**: Clean modular architecture in raggy/ package
- **Complexity**: All functions CC=1 (simple delegates)
- **Solution**: No duplication - everything imports from raggy/

## Migration Guide

### For Command Line Users

**Old way (deprecated):**
```bash
python raggy.py build
python raggy.py search "query"
```

**New way (recommended):**
```bash
python raggy_cli.py build
python raggy_cli.py search "query"
```

### For Python Scripts

**Old way (still works):**
```python
from raggy import UniversalRAG, SearchEngine

rag = UniversalRAG(...)
```

**New way (identical, but cleaner):**
```python
from raggy import UniversalRAG, SearchEngine

rag = UniversalRAG(...)
```

The Python API hasn't changed - imports work exactly the same way. The difference is that now everything comes from the modular `raggy/` package instead of the monolithic file.

## Backward Compatibility

✅ **100% Backward Compatible**

- All existing scripts continue to work
- All imports are preserved
- All functions are available
- All commands function identically
- Only difference: deprecation warnings shown

## File Structure

```
raggy/
├── raggy.py              # DEPRECATED: Thin wrapper (243 lines)
├── raggy_cli.py          # NEW: Recommended entry point (203 lines)
├── raggy/                # Modular package with all functionality
│   ├── __init__.py       # Public API exports
│   ├── core/             # Core RAG functionality
│   ├── cli/              # Command implementations
│   ├── config/           # Configuration management
│   ├── database/         # Database abstractions
│   ├── processing/       # Document processing
│   ├── query/            # Query processing
│   ├── scoring/          # Scoring algorithms
│   ├── search/           # Search functionality
│   ├── setup/            # Setup and dependencies
│   └── utils/            # Utility functions
└── tests/                # Test suite
```

## Benefits

1. **Eliminated Duplication**: Removed 2,676 lines of duplicate code
2. **Improved Maintainability**: Modular structure with single responsibility
3. **Better Testing**: Each module can be tested independently
4. **Cleaner Architecture**: Separation of concerns, SOLID principles
5. **Future-Proof**: Easy to extend without modifying core

## Actions Required

### Immediate (Optional but Recommended)
- Update scripts to use `python raggy_cli.py` instead of `python raggy.py`
- Update documentation to reference new entry point
- Update CI/CD pipelines if applicable

### Before v3.0.0 (Required)
- Complete migration to `raggy_cli.py`
- Remove any dependencies on `raggy.py`
- Update all references in documentation

## Technical Details

The deprecated `raggy.py` now:
1. Shows deprecation warnings when executed
2. Imports all functionality from `raggy/` package
3. Re-exports everything for backward compatibility
4. Delegates main() to raggy_cli.py
5. Maintains identical command-line interface

## Questions?

If you encounter any issues during migration:
1. Check that all imports still work: `python -c "from raggy import UniversalRAG"`
2. Verify commands work: `python raggy_cli.py --help`
3. Run tests: `pytest tests/`
4. Report issues on GitHub