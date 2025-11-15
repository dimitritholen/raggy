# Configuration Guide

Raggy can be configured through CLI arguments, configuration files, or Python API parameters.

## Configuration Files

Raggy supports two configuration formats:

### .raggy.json (Recommended for v2.0+)

Modern JSON-based configuration with support for cloud vector databases and embedding providers:

```json
{
  "vectorStore": {
    "provider": "chromadb",
    "chromadb": {
      "path": "./vectordb"
    }
  },
  "embedding": {
    "provider": "sentenceTransformers",
    "sentenceTransformers": {
      "model": "all-MiniLM-L6-v2"
    }
  }
}
```

**Supported vector stores:** `chromadb`, `pinecone`, `supabase`
**Supported embedding providers:** `sentenceTransformers`, `openai`

See [Vector Databases Guide](./vector-databases.md) for detailed configuration examples.

### raggy_config.yaml (Legacy)

Create `raggy_config.yaml` in your project root:

```yaml
# Document and database paths
docs_dir: "./docs"
db_dir: "./vectordb"

# Embedding model
model: "all-MiniLM-L6-v2"

# Text chunking
chunk_size: 1000
chunk_overlap: 200

# Search settings
top_k: 5
hybrid: true
expand_query: false

# Memory system
memory_db_dir: "./memory_db"
```

Load configuration:

```bash
python raggy_cli.py build --config raggy_config.yaml
```

## Configuration Options

### Paths

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `docs_dir` | string | `"./docs"` | Directory containing documents |
| `db_dir` | string | `"./vectordb"` | Vector database directory |
| `memory_db_dir` | string | `"./memory_db"` | Memory database directory |

### Model Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | string | `"all-MiniLM-L6-v2"` | Embedding model name |
| `model_preset` | string | `null` | Preset: fast, balanced, multilingual, accurate |

### Chunking Parameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | integer | `1000` | Characters per chunk |
| `chunk_overlap` | integer | `200` | Overlap between chunks |

**Recommended values:**
- **Short documents** (tweets, comments): `chunk_size=500`, `chunk_overlap=50`
- **Standard documents** (articles, docs): `chunk_size=1000`, `chunk_overlap=200`
- **Long documents** (books, research): `chunk_size=1500`, `chunk_overlap=300`

### Search Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `top_k` | integer | `5` | Number of results to return |
| `hybrid` | boolean | `false` | Enable hybrid search |
| `expand_query` | boolean | `false` | Enable query expansion |

## Model Presets

### Fast
```yaml
model_preset: fast
```
- Model: `paraphrase-MiniLM-L3-v2`
- Size: 17MB
- Speed: Very fast
- Accuracy: Good
- Use case: Quick searches, prototyping

### Balanced (Default)
```yaml
model_preset: balanced
```
- Model: `all-MiniLM-L6-v2`
- Size: 80MB
- Speed: Fast
- Accuracy: Very good
- Use case: General purpose

### Multilingual
```yaml
model_preset: multilingual
```
- Model: `paraphrase-multilingual-MiniLM-L12-v2`
- Size: 420MB
- Speed: Moderate
- Accuracy: Very good
- Languages: 50+
- Use case: Non-English content

### Accurate
```yaml
model_preset: accurate
```
- Model: `all-mpnet-base-v2`
- Size: 420MB
- Speed: Slower
- Accuracy: Excellent
- Use case: Production systems requiring highest quality

## Environment-Specific Configuration

### Development

```yaml
# dev_config.yaml
docs_dir: "./test_docs"
db_dir: "./test_vectordb"
model_preset: fast
chunk_size: 500
top_k: 3
```

### Production

```yaml
# prod_config.yaml
docs_dir: "/app/documents"
db_dir: "/app/vectordb"
model_preset: accurate
chunk_size: 1000
chunk_overlap: 200
top_k: 10
hybrid: true
```

## Python API Configuration

### Basic Configuration

```python
from raggy import UniversalRAG

rag = UniversalRAG(
    docs_dir="./docs",
    db_dir="./vectordb",
    model="all-MiniLM-L6-v2",
    chunk_size=1000,
    chunk_overlap=200
)
```

### Advanced Configuration

```python
from raggy import UniversalRAG
from raggy.config.loader import load_config

# Load from file
config = load_config("raggy_config.yaml")

# Override specific settings
config["top_k"] = 10
config["hybrid"] = True

# Initialize with config
rag = UniversalRAG(**config)
```

## Memory System Configuration

### CLI Configuration

```bash
# Custom memory database location
python raggy_cli.py remember "content" --db-dir ./custom_memory
```

### Python API Configuration

```python
from raggy import Memory

memory = Memory(
    db_dir="./memory_db",
    model="all-MiniLM-L6-v2",
    chunk_size=1000
)
```

## Performance Tuning

### For Speed

```yaml
model_preset: fast
chunk_size: 800
top_k: 5
```

### For Accuracy

```yaml
model_preset: accurate
chunk_size: 1200
chunk_overlap: 250
top_k: 15
hybrid: true
expand_query: true
```

### For Multilingual

```yaml
model_preset: multilingual
chunk_size: 1000
chunk_overlap: 200
```

## Example Configurations

### Technical Documentation

```yaml
docs_dir: "./api-docs"
db_dir: "./vectordb"
model_preset: balanced
chunk_size: 1500
chunk_overlap: 300
hybrid: true
top_k: 10
```

### Research Papers

```yaml
docs_dir: "./papers"
db_dir: "./vectordb"
model_preset: accurate
chunk_size: 2000
chunk_overlap: 400
expand_query: true
top_k: 15
```

### Quick Notes Search

```yaml
docs_dir: "./notes"
db_dir: "./vectordb"
model_preset: fast
chunk_size: 500
chunk_overlap: 50
top_k: 5
```

## Configuration Priority

When multiple configuration sources are present:

1. **CLI arguments** (highest priority)
2. **Configuration file** (`--config` flag)
3. **Default values** (lowest priority)

Example:

```bash
# chunk_size will be 1500 (CLI overrides config file)
python raggy_cli.py build --config config.yaml --chunk-size 1500
```

## Cloud Vector Database Configuration

### Pinecone Configuration (.raggy.json)

```json
{
  "vectorStore": {
    "provider": "pinecone",
    "pinecone": {
      "apiKey": "${PINECONE_API_KEY}",
      "environment": "us-east-1-aws",
      "indexName": "raggy-index",
      "dimension": 1536
    }
  },
  "embedding": {
    "provider": "openai",
    "openai": {
      "apiKey": "${OPENAI_API_KEY}",
      "model": "text-embedding-3-small"
    }
  }
}
```

**Environment variables:**
```bash
export PINECONE_API_KEY="pcsk_..."
export OPENAI_API_KEY="sk-proj-..."
```

### Supabase Configuration (.raggy.json)

```json
{
  "vectorStore": {
    "provider": "supabase",
    "supabase": {
      "url": "${SUPABASE_URL}",
      "apiKey": "${SUPABASE_ANON_KEY}",
      "dimension": 384
    }
  },
  "embedding": {
    "provider": "sentenceTransformers",
    "sentenceTransformers": {
      "model": "all-MiniLM-L6-v2"
    }
  }
}
```

**Environment variables:**
```bash
export SUPABASE_URL="https://xxxxx.supabase.co"
export SUPABASE_ANON_KEY="eyJhbGc..."
```

### Interactive Setup

The easiest way to configure cloud databases:

```bash
python raggy_cli.py init --interactive
```

This will guide you through:
1. Selecting a vector database (ChromaDB, Pinecone, Supabase)
2. Selecting an embedding provider (SentenceTransformers, OpenAI)
3. Entering API keys and credentials
4. Creating `.raggy.json` configuration file

## Next Steps

- [Vector Databases Guide](./vector-databases.md) - Detailed cloud database setup
- [Performance Tuning](./performance.md)
- [Model Selection Guide](./model-selection.md)
- [API Reference](./api-reference.md)
