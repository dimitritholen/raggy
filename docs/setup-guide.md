# Setup Guide

Quick setup guide for getting started with Raggy, including cloud vector database configuration.

## Quick Start (Local ChromaDB)

**1. Install Raggy:**
```bash
pip install raggy
```

**2. Initialize Project:**
```bash
python raggy_cli.py init
```

**3. Build Vector Database:**
```bash
python raggy_cli.py build
```

**4. Search Documents:**
```bash
python raggy_cli.py search "your query"
```

**5. Store Memories:**
```bash
python raggy_cli.py remember "Fixed critical bug in authentication"
```

**6. Recall Memories:**
```bash
python raggy_cli.py recall "bug fix"
```

Done! You now have a fully functional RAG system with development memory.

## Interactive Cloud Setup

For production deployments with Pinecone or Supabase:

**1. Install with cloud support:**
```bash
# For Pinecone
pip install "raggy[pinecone]"

# For Supabase
pip install "raggy[supabase]"

# For OpenAI embeddings
pip install openai
```

**2. Run interactive setup:**
```bash
python raggy_cli.py init --interactive
```

**3. Follow the prompts:**

```
Welcome to Raggy Interactive Setup!

? Select vector database provider:
  > ChromaDB (Local - recommended for development)
    Pinecone (Cloud - serverless, auto-scaling)
    Supabase (Cloud - PostgreSQL + pgvector)

? Select embedding provider:
    SentenceTransformers (Local - free, no API key)
  > OpenAI (Cloud - high quality, requires API key)

? Enter OpenAI API key: sk-proj-...
? Enter Pinecone API key: pcsk_...
? Enter Pinecone region (e.g., us-east-1-aws): us-east-1-aws
? Enter Pinecone index name [raggy-index]:
? Enter embedding dimension [1536]:

✓ Configuration saved to .raggy.json
✓ Setup complete!
```

**4. Test the configuration:**
```bash
python raggy_cli.py remember "Testing cloud setup" --type note
python raggy_cli.py recall "cloud setup"
```

## Manual Configuration

### Option 1: Local Development (ChromaDB + SentenceTransformers)

Create `.raggy.json`:
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

**Pros:**
- ✅ Zero cost (fully local)
- ✅ No API keys required
- ✅ Offline support
- ✅ Fast iteration

**Cons:**
- ❌ Single machine only
- ❌ No cloud sync
- ❌ Manual scaling

### Option 2: Cloud Production (Pinecone + OpenAI)

**Step 1: Get API Keys**

1. **Pinecone**: Sign up at [pinecone.io](https://www.pinecone.io)
   - Create API key in dashboard
   - Note your environment (e.g., us-east-1-aws)

2. **OpenAI**: Sign up at [platform.openai.com](https://platform.openai.com)
   - Create API key in API Keys section
   - Add billing information

**Step 2: Set Environment Variables**
```bash
export PINECONE_API_KEY="pcsk_..."
export OPENAI_API_KEY="sk-proj-..."
```

**Step 3: Create Pinecone Index**

Via Pinecone Console:
1. Go to Indexes → Create Index
2. Name: `raggy-index`
3. Dimensions: `1536`
4. Metric: `cosine`
5. Cloud: `aws`
6. Region: `us-east-1`

Via Python:
```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")
pc.create_index(
    name="raggy-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

**Step 4: Create `.raggy.json`**
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

**Step 5: Test**
```bash
python raggy_cli.py remember "Cloud setup complete" --priority high
python raggy_cli.py recall "setup"
```

**Pros:**
- ✅ Auto-scaling
- ✅ High quality embeddings
- ✅ Multi-user support
- ✅ Global low latency

**Cons:**
- ❌ Requires API keys
- ❌ Monthly costs (free tier available)
- ❌ Internet dependency

### Option 3: PostgreSQL Users (Supabase + SentenceTransformers)

**Step 1: Create Supabase Project**

1. Sign up at [supabase.com](https://supabase.com)
2. Create new project
3. Wait for project initialization (~2 minutes)

**Step 2: Get Credentials**

In Supabase Dashboard:
- Project URL: Settings → API → Project URL
- Anon Key: Settings → API → anon/public key

**Step 3: Enable pgvector**

In SQL Editor, run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Step 4: Create Match Function**

In SQL Editor, run:
```sql
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(384),
  match_threshold float DEFAULT 0.0,
  match_count int DEFAULT 5,
  table_name text DEFAULT 'project_memory'
)
RETURNS TABLE (
  id text,
  document text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  EXECUTE format('
    SELECT id, document, metadata,
           1 - (embedding <=> $1) AS similarity
    FROM %I
    WHERE 1 - (embedding <=> $1) > $2
    ORDER BY embedding <=> $1
    LIMIT $3
  ', table_name)
  USING query_embedding, match_threshold, match_count;
END;
$$;
```

**Step 5: Set Environment Variables**
```bash
export SUPABASE_URL="https://xxxxx.supabase.co"
export SUPABASE_ANON_KEY="eyJhbGc..."
```

**Step 6: Create `.raggy.json`**
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

**Step 7: Test**
```bash
python raggy_cli.py remember "Supabase configured successfully"
python raggy_cli.py recall "supabase"
```

**Pros:**
- ✅ PostgreSQL-based (familiar SQL)
- ✅ Row-level security
- ✅ Integrated with Supabase ecosystem
- ✅ Free tier (500 MB)
- ✅ No OpenAI costs (local embeddings)

**Cons:**
- ❌ More setup steps
- ❌ Requires PostgreSQL knowledge
- ❌ Manual scaling

## Verifying Your Setup

### Test Vector Database
```bash
# Store a test memory
python raggy_cli.py remember "Setup verification test" --type note

# Retrieve it
python raggy_cli.py recall "verification"

# Expected output:
# 🔍 Memory results for: 'verification'
# 1. [MEMORY] 2025-11-15 12:00 | note
#    Setup verification test
```

### Test Embedding Provider
```python
from raggy.core.embedding_factory import create_embedding_provider
from raggy.config.raggy_config import RaggyConfig

config = RaggyConfig()
embedding_provider = create_embedding_provider(config.config)

# Generate test embedding
text = "Hello world"
embedding = embedding_provider.embed(text)

print(f"Embedding provider: {type(embedding_provider).__name__}")
print(f"Embedding dimension: {len(embedding)}")
print(f"Sample values: {embedding[:5]}")

# Expected output (Pinecone + OpenAI):
# Embedding provider: OpenAIProvider
# Embedding dimension: 1536
# Sample values: [0.123, -0.456, 0.789, ...]

# Expected output (ChromaDB + SentenceTransformers):
# Embedding provider: SentenceTransformersProvider
# Embedding dimension: 384
# Sample values: [0.234, -0.567, 0.890, ...]
```

### Test Full Pipeline
```bash
# 1. Build document index
echo "Test document content" > test.txt
python raggy_cli.py build

# 2. Search documents
python raggy_cli.py search "test document"

# 3. Store memory
python raggy_cli.py remember "Tested full pipeline successfully"

# 4. Unified search
python raggy_cli.py search "pipeline" --include-memory
```

## Troubleshooting

### "Module not found" errors
```bash
# Pinecone
pip install "pinecone[grpc]"

# Supabase
pip install supabase

# OpenAI
pip install openai
```

### "API key not found"
```bash
# Verify environment variables are set
echo $PINECONE_API_KEY
echo $OPENAI_API_KEY
echo $SUPABASE_URL

# If empty, export them:
export PINECONE_API_KEY="your-key"
```

### "Index not found" (Pinecone)
```bash
# Verify index exists
python -c "from pinecone import Pinecone; pc = Pinecone(api_key='your-key'); print(pc.list_indexes())"

# Create if missing (see Step 3 in Pinecone setup)
```

### "Dimension mismatch"
```
Error: Vector dimension mismatch: expected 1536, got 384
```

**Fix:** Match embedding model dimension with vector database configuration:
- OpenAI `text-embedding-3-small` → dimension `1536`
- SentenceTransformers `all-MiniLM-L6-v2` → dimension `384`

Update `.raggy.json`:
```json
{
  "vectorStore": {
    "pinecone": {
      "dimension": 1536  // Match OpenAI
    }
  },
  "embedding": {
    "openai": {
      "model": "text-embedding-3-small"  // 1536 dims
    }
  }
}
```

### "Table does not exist" (Supabase)
Raggy creates tables automatically on first use. Verify:
1. pgvector extension is enabled: `SELECT * FROM pg_extension WHERE extname = 'vector';`
2. Your API key has table creation permissions
3. Run `match_documents` SQL function (Step 4 in Supabase setup)

## Next Steps

- [Configuration Guide](./configuration.md) - Detailed configuration options
- [Vector Databases Guide](./vector-databases.md) - In-depth cloud database setup
- [Memory System](./memory-system.md) - Development memory features
- [Quick Start Tutorial](./quickstart.md) - Complete tutorial

## Getting Help

- **Documentation**: [docs/](.)
- **Issues**: [GitHub Issues](https://github.com/yourusername/raggy/issues)
- **Examples**: See `examples/` directory
- **FAQ**: [docs/faq.md](./faq.md)
