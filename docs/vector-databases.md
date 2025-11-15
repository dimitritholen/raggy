# Vector Database Support

Raggy supports multiple vector database backends for both document storage (RAG) and development memory. Choose the best option for your deployment needs.

## Supported Vector Databases

### ChromaDB (Default - Local)
**Best for**: Development, local projects, offline use

- ✅ Zero configuration required
- ✅ Fully local, no API keys needed
- ✅ Fast setup and iteration
- ✅ Automatic persistence to disk
- ❌ Single-machine only (no cloud sync)

**Installation:**
```bash
pip install raggy  # ChromaDB included by default
```

**Configuration (.raggy.json):**
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

### Pinecone (Cloud - Serverless)
**Best for**: Production, multi-user, cloud deployments, auto-scaling

- ✅ Serverless architecture (auto-scaling)
- ✅ Low latency globally distributed
- ✅ Free tier: 100K vectors
- ✅ Managed backups and high availability
- ❌ Requires API key and internet connection

**Installation:**
```bash
pip install "raggy[pinecone]"
# or
pip install raggy pinecone[grpc]
```

**Configuration (.raggy.json):**
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

**Setup Steps:**

1. **Create Pinecone Account**: Sign up at [pinecone.io](https://www.pinecone.io)

2. **Get API Key**: Dashboard → API Keys → Create Key

3. **Create Index** (via Pinecone Console or API):
   ```python
   from pinecone import Pinecone, ServerlessSpec

   pc = Pinecone(api_key="your-api-key")
   pc.create_index(
       name="raggy-index",
       dimension=1536,  # Match your embedding model
       metric="cosine",
       spec=ServerlessSpec(cloud="aws", region="us-east-1")
   )
   ```

4. **Set Environment Variables**:
   ```bash
   export PINECONE_API_KEY="pcsk_..."
   export OPENAI_API_KEY="sk-proj-..."
   ```

5. **Initialize Raggy**:
   ```bash
   python raggy_cli.py init --interactive
   ```

**Dimension Requirements:**
- OpenAI `text-embedding-3-small`: 1536 dimensions
- OpenAI `text-embedding-3-large`: 3072 dimensions
- SentenceTransformers `all-MiniLM-L6-v2`: 384 dimensions

### Supabase (Cloud - PostgreSQL + pgvector)
**Best for**: Full-stack apps, existing PostgreSQL users, SQL access

- ✅ PostgreSQL-based (familiar SQL interface)
- ✅ Integrated with Supabase ecosystem
- ✅ Free tier: 500 MB database
- ✅ Row-level security and multi-tenancy
- ❌ Requires Supabase project setup

**Installation:**
```bash
pip install "raggy[supabase]"
# or
pip install raggy supabase
```

**Configuration (.raggy.json):**
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

**Setup Steps:**

1. **Create Supabase Project**: Sign up at [supabase.com](https://supabase.com)

2. **Get Credentials**:
   - Project URL: Settings → API → Project URL
   - Anon Key: Settings → API → anon/public key

3. **Enable pgvector Extension** (via SQL Editor):
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. **Create RPC Function** (for similarity search):
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

5. **Set Environment Variables**:
   ```bash
   export SUPABASE_URL="https://xxxxx.supabase.co"
   export SUPABASE_ANON_KEY="eyJhbGc..."
   ```

6. **Initialize Raggy**:
   ```bash
   python raggy_cli.py init --interactive
   ```

## Comparison Matrix

| Feature | ChromaDB | Pinecone | Supabase |
|---------|----------|----------|----------|
| **Deployment** | Local only | Cloud (serverless) | Cloud (PostgreSQL) |
| **Setup Complexity** | ⭐ Easy | ⭐⭐ Moderate | ⭐⭐⭐ Advanced |
| **Free Tier** | Unlimited (local) | 100K vectors | 500 MB database |
| **Scaling** | Manual (single machine) | Auto-scaling | Manual (upgrade plan) |
| **Multi-user** | ❌ No | ✅ Yes | ✅ Yes |
| **SQL Access** | ❌ No | ❌ No | ✅ Yes |
| **Latency** | <1ms (local) | 10-50ms (global) | 20-100ms (global) |
| **Best Use Case** | Development, prototyping | Production apps, SaaS | Full-stack apps, PostgreSQL users |

## Embedding Provider Pairing

### Recommended Combinations

**Local Development:**
```json
{
  "vectorStore": {"provider": "chromadb"},
  "embedding": {"provider": "sentenceTransformers"}
}
```
- Fast, no API costs
- Great for prototyping

**Production (Cloud):**
```json
{
  "vectorStore": {"provider": "pinecone"},
  "embedding": {"provider": "openai"}
}
```
- High quality embeddings
- Scalable infrastructure
- Pay-per-use pricing

**PostgreSQL Users:**
```json
{
  "vectorStore": {"provider": "supabase"},
  "embedding": {"provider": "sentenceTransformers"}
}
```
- Leverage existing Supabase setup
- No OpenAI costs (local embeddings)
- SQL access for complex queries

## Migration Between Databases

### Export from ChromaDB
```python
from raggy import MemoryManager

# Export memories
memory = MemoryManager(db_dir="./vectordb", config_path=".raggy.json")
results = memory.search("", limit=10000)  # Get all

# Save to JSON
import json
with open("memories_export.json", "w") as f:
    json.dump(results, f)
```

### Import to Pinecone/Supabase
```python
# Update .raggy.json to new provider
# Then reimport:

import json
from raggy import MemoryManager

with open("memories_export.json", "r") as f:
    memories = json.load(f)

memory = MemoryManager(config_path=".raggy.json")
for mem in memories:
    memory.add(
        text=mem["text"],
        memory_type=mem["metadata"].get("memory_type", "note"),
        tags=mem["metadata"].get("tags", []),
        priority=mem["metadata"].get("priority", "medium")
    )
```

## Configuration via Environment Variables

All API keys can use environment variable substitution:

```json
{
  "vectorStore": {
    "provider": "pinecone",
    "pinecone": {
      "apiKey": "${PINECONE_API_KEY}",
      "indexName": "${PINECONE_INDEX_NAME:-raggy-index}"
    }
  }
}
```

**Supported syntax:**
- `${VAR}` - Required variable (error if missing)
- `${VAR:-default}` - Optional with default value

## Troubleshooting

### Pinecone Issues

**"Index not found"**
```bash
# Verify index exists
python -c "from pinecone import Pinecone; pc = Pinecone(api_key='your-key'); print(pc.list_indexes())"
```

**"Dimension mismatch"**
- Ensure `dimension` in config matches your embedding model
- OpenAI text-embedding-3-small = 1536
- SentenceTransformers all-MiniLM-L6-v2 = 384

**"gRPC module not found"**
```bash
pip install "pinecone[grpc]"
```

### Supabase Issues

**"exec_sql RPC not found"**
- Execute the `match_documents` SQL function in Supabase SQL Editor
- Verify pgvector extension is enabled: `SELECT * FROM pg_extension WHERE extname = 'vector';`

**"Table does not exist"**
- Raggy creates tables automatically on first use
- Verify your API key has table creation permissions

### ChromaDB Issues

**"Database locked"**
- Close other processes using the same `db_dir`
- Delete `./vectordb/chroma.sqlite3-wal` if stuck

**"Collection not found"**
```bash
python raggy_cli.py build  # Rebuild index
```

## Performance Tips

### Pinecone
- Use closest region to your users (us-east-1, eu-west-1, etc.)
- Batch upserts (up to 100 vectors per call)
- Use namespace isolation for multi-tenancy

### Supabase
- Create indexes on metadata fields for filtered queries
- Use connection pooling for high-traffic apps
- Consider `pgbouncer` for connection management

### ChromaDB
- Use SSD storage for better performance
- Limit collection size (<1M vectors for optimal speed)
- Regular vacuum/optimize operations

## Security Best Practices

1. **Never commit API keys**
   ```bash
   # Add to .gitignore
   echo ".raggy.json" >> .gitignore
   ```

2. **Use environment variables**
   ```bash
   export PINECONE_API_KEY="..."
   export OPENAI_API_KEY="..."
   ```

3. **Rotate keys regularly**
   - Pinecone: Dashboard → API Keys → Rotate
   - Supabase: Settings → API → Generate New Key

4. **Use read-only keys where possible**
   - Supabase supports service role vs anon keys
   - Pinecone supports read-only API keys

## Cost Estimation

### Pinecone
- Free: 100K vectors (1536 dims)
- Starter: $0.096/GB/month (~1M vectors = $15/month)
- Enterprise: Volume discounts

### Supabase
- Free: 500 MB database
- Pro: $25/month (8 GB)
- Scale: Usage-based pricing

### OpenAI Embeddings
- text-embedding-3-small: $0.02 per 1M tokens
- ~1,500 tokens = 1 document (average)
- 10,000 documents ≈ $0.30

### ChromaDB (Local)
- $0 (runs on your machine)
- Storage: ~200 MB per 100K vectors (384 dims)

## See Also

- [Configuration Guide](./configuration.md) - Full config reference
- [Memory System](./memory-system.md) - Development memory features
- [API Reference](./api-reference.md) - Python API documentation
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions
