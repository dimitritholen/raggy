# Universal RAG System v2.0.0

Enhanced ChromaDB RAG setup with hybrid search, smart chunking, and normalized scoring.

## Quick Start

```bash
# Basic usage
python raggy.py build                    # Index your docs
python raggy.py search "your query"      # Search with normalized scores

# Enhanced features  
python raggy.py search "term" --hybrid   # Hybrid semantic + keyword search
python raggy.py search "api" --expand    # Query expansion
python raggy.py optimize                 # Benchmark search modes
```

## Configuration

1. Copy the example config:
   ```bash
   cp raggy_config_example.yaml raggy_config.yaml
   ```

2. Customize the `expansions` section with your domain terms:
   ```yaml
   search:
     expansions:
       myterm: ["myterm", "synonym1", "synonym2"]
       acronym: ["acronym", "full phrase"]
   ```

3. Use expanded search:
   ```bash
   python raggy.py search "myterm" --expand
   ```

## Key Features

- **Hybrid Search**: Combines semantic + BM25 keyword ranking
- **Smart Chunking**: Markdown-aware document processing
- **Normalized Scoring**: 0-1 scores with quality labels (Excellent/Good/Fair/Poor)
- **Query Expansion**: Automatic synonym expansion for domain terms
- **Model Presets**: fast/balanced/multilingual/accurate options
- **Universal**: Works with any project's documentation

## Requirements

- Python 3.8+
- `uv` package manager (auto-installed)
- Documents in `./docs/` directory (markdown or PDF)

The system automatically installs dependencies using `uv` on first run.