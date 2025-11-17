# Package Upgrade Rollback Procedure

## If Upgrade Fails

### 1. Rollback Git Branch
```bash
git checkout main
git branch -D feature/package-upgrades
```

### 2. Restore Package Versions
```bash
pip install -r .backlog/tasks-package-upgrade/baseline-packages.txt
```

### 3. Restore ChromaDB Database
```bash
rm -rf .raggy/chroma
cp -r .backlog/chromadb-backup/chroma .raggy/
```

### 4. Verify Rollback Success
```bash
pytest --verbose --tb=short
ruff check .
mypy raggy.py
```

## Baseline Files
- `baseline-packages.txt` - Exact package versions before upgrade
- `baseline-test-results.txt` - Test results before upgrade
- `baseline-ruff.txt` - Ruff linting results before upgrade
- `baseline-mypy.txt` - Mypy type checking results before upgrade
- `baseline-bandit.txt` - Bandit security scan before upgrade
- `chromadb-backup/` - ChromaDB database backup

## Recovery Time
Estimated: 5-10 minutes for complete rollback
