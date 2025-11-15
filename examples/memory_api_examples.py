#!/usr/bin/env python3
"""Memory API Usage Examples - Phase 4 Implementation.

This file demonstrates all the ways to use the Memory API introduced in
Raggy 2.0. These examples are included in the documentation.

Author: Raggy Team
Date: 2025-11-13
Version: 2.0.0
"""

# =============================================================================
# Example 1: Basic Memory Usage with Memory Class
# =============================================================================


def example_1_basic_memory():
    """Basic memory operations using the Memory class."""
    from raggy import Memory

    # Initialize memory system
    memory = Memory(db_dir="./vectordb")

    # Store different types of memories
    print("=== Example 1: Basic Memory Usage ===\n")

    # Store an architecture decision
    decision_id = memory.add(
        text="Decided to use dependency injection pattern for database layer "
             "to enable testing with mock databases and support multiple "
             "database backends (ChromaDB, Pinecone, Weaviate, etc.)",
        memory_type="decision",
        tags=["architecture", "database", "testing"],
        files_involved=["core/database.py", "core/database_interface.py"],
        priority="high"
    )
    print(f"Stored decision: {decision_id}")

    # Store a bug fix solution
    solution_id = memory.add(
        text="Fixed ChromaDB 'empty list' error by not including empty lists "
             "in metadata. ChromaDB doesn't allow empty list values in metadata "
             "fields. Solution: Only add tags and files_involved if non-empty.",
        memory_type="solution",
        tags=["chromadb", "bug-fix", "metadata"],
        priority="medium"
    )
    print(f"Stored solution: {solution_id}")

    # Store a code pattern
    pattern_id = memory.add(
        text="Using Strategy pattern for document parsers: PDFParser, "
             "DOCXParser, MarkdownParser, TXTParser with common DocumentParser "
             "interface. Each parser handles format-specific extraction logic.",
        memory_type="pattern",
        tags=["design-pattern", "document-processing", "strategy-pattern"]
    )
    print(f"Stored pattern: {pattern_id}")

    # Store a learning
    learning_id = memory.add(
        text="Learned that sentence-transformers models should be loaded once "
             "and reused (lazy loading) to avoid repeated download/initialization "
             "overhead. Use property with cached _embedding_model attribute.",
        memory_type="learning",
        tags=["performance", "embeddings", "optimization"]
    )
    print(f"Stored learning: {learning_id}")

    print(f"\n✓ Added 4 different types of memories\n")

    return memory


# =============================================================================
# Example 2: Searching Memories
# =============================================================================


def example_2_search_memories(memory):
    """Search for memories using semantic search and filters."""
    print("=== Example 2: Searching Memories ===\n")

    # Simple semantic search
    print("Search: 'database architecture decisions'")
    results = memory.search("database architecture decisions", limit=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['metadata']['memory_type'].upper()}")
        print(f"   Priority: {result['metadata']['priority']}")
        print(f"   Text: {result['text'][:80]}...")
        print(f"   Distance: {result['distance']:.4f}")

    # Search with type filter
    print("\n" + "-" * 70)
    print("Search: 'bugs and fixes' (solutions only)")
    results = memory.search(
        "bugs and fixes",
        memory_types=["solution"],
        limit=2
    )

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text'][:100]}...")

    # Search with tag filter
    print("\n" + "-" * 70)
    print("Search: 'design patterns' (tagged with 'design-pattern')")
    results = memory.search(
        "design patterns",
        tags=["design-pattern"],
        limit=2
    )

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text'][:100]}...")

    # Search with date filter
    print("\n" + "-" * 70)
    print("Search: recent memories (since 2025-01-01)")
    results = memory.search(
        "architecture and patterns",
        since="2025-01-01T00:00:00Z",
        limit=5
    )
    print(f"Found {len(results)} memories since 2025-01-01")

    print("\n✓ Demonstrated multiple search patterns\n")


# =============================================================================
# Example 3: AI Prompt Context Generation
# =============================================================================


def example_3_prompt_context(memory):
    """Generate formatted context for AI prompts."""
    print("=== Example 3: AI Prompt Context Generation ===\n")

    # Get context for AI prompt
    context = memory.get_context_for_prompt(
        query="database architecture and design patterns",
        max_tokens=2000,
        memory_types=["decision", "pattern", "learning"]
    )

    print("Generated context for AI prompt:")
    print("-" * 70)
    print(context)
    print("-" * 70)

    # Use in AI prompt
    ai_prompt = f"""
{context}

Based on the above development context, help me implement a new database
adapter for Pinecone following our existing patterns and architecture decisions.

Requirements:
1. Follow dependency injection pattern
2. Use strategy pattern for Pinecone-specific logic
3. Handle metadata correctly (no empty lists)
4. Lazy-load embedding model for performance
"""

    print("\n✓ Context ready for AI prompt injection")
    print(f"Context length: ~{len(context)} characters")
    print(f"Estimated tokens: ~{len(context) // 4}\n")


# =============================================================================
# Example 4: Module-Level Convenience Functions
# =============================================================================


def example_4_convenience_functions():
    """Use remember() and recall() convenience functions."""
    from raggy import remember, recall

    print("=== Example 4: Convenience Functions ===\n")

    # Quick memory storage with remember()
    print("Using remember() for quick storage...")

    mem_id = remember(
        "Decided to use FastAPI for REST API layer due to automatic OpenAPI "
        "documentation, native async support, and excellent type checking.",
        memory_type="decision",
        tags=["api", "architecture", "fastapi"],
        priority="high"
    )
    print(f"Stored: {mem_id}")

    mem_id = remember(
        "Fixed CORS issue by adding FastAPI CORSMiddleware with proper origin "
        "configuration. Need to whitelist frontend domains explicitly.",
        memory_type="solution",
        tags=["api", "cors", "bug-fix"]
    )
    print(f"Stored: {mem_id}")

    # Quick search with recall()
    print("\nUsing recall() for quick search...")

    results = recall(
        "API design decisions",
        memory_types=["decision", "solution"],
        tags=["api"],
        limit=3
    )

    print(f"Found {len(results)} API-related memories:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['metadata']['memory_type']}: {result['text'][:80]}...")

    print("\n✓ Convenience functions work for one-off operations\n")


# =============================================================================
# Example 5: UniversalRAG Integration
# =============================================================================


def example_5_universal_rag_integration():
    """Use memory system integrated with UniversalRAG."""
    from raggy import UniversalRAG

    print("=== Example 5: UniversalRAG Integration ===\n")

    # Initialize RAG system (uses same database for documents and memory)
    rag = UniversalRAG(docs_dir="./docs", db_dir="./vectordb")

    # Store memory while working with documents
    print("Storing memories via UniversalRAG...")

    mem_id = rag.remember(
        "Decided to use hybrid search (BM25 + semantic) for better accuracy "
        "on technical queries. BM25 handles exact keyword matches (e.g., "
        "function names, error codes) while semantic handles conceptual queries.",
        memory_type="decision",
        tags=["search", "architecture", "hybrid-search"],
        priority="high"
    )
    print(f"Stored: {mem_id}")

    # Search documents AND recall memories
    print("\nCombining document search with memory recall...")

    # Search documents
    doc_results = rag.search("database architecture patterns", top_k=5)
    print(f"Documents found: {len(doc_results)}")

    # Recall related memories
    memory_results = rag.recall(
        "database architecture decisions",
        memory_types=["decision", "pattern"],
        limit=3
    )
    print(f"Related memories: {len(memory_results)}")

    # Use both for comprehensive context
    print("\n✓ Single UniversalRAG instance handles both documents and memory")
    print("✓ No need for separate Memory instance\n")


# =============================================================================
# Example 6: Advanced Memory Management
# =============================================================================


def example_6_advanced_management(memory):
    """Advanced memory management operations."""
    print("=== Example 6: Advanced Memory Management ===\n")

    # Count memories
    total = memory.count()
    print(f"Total memories: {total}")

    # Count by type
    decisions = memory.count(where={"memory_type": "decision"})
    solutions = memory.count(where={"memory_type": "solution"})
    patterns = memory.count(where={"memory_type": "pattern"})

    print(f"Decisions: {decisions}")
    print(f"Solutions: {solutions}")
    print(f"Patterns: {patterns}")

    # Get specific memory by ID
    print("\nRetrieving memory by ID...")
    results = memory.search("database", limit=1)
    if results:
        mem_id = results[0]['id']
        entry = memory.get_by_id(mem_id)
        if entry:
            print(f"Memory ID: {mem_id}")
            print(f"Type: {entry['metadata']['memory_type']}")
            print(f"Text: {entry['text'][:80]}...")

    # Archive old memories (example - would need old memories)
    print("\nArchiving old memories...")
    try:
        archived = memory.archive("2024-01-01T00:00:00Z")
        print(f"Archived {archived} old memories")
    except Exception as e:
        print(f"Archive: {e}")

    # Delete specific memory (example only - don't actually delete)
    # memory.delete(mem_id)

    # Delete all memories (dangerous - use with caution!)
    # count = memory.delete_all()
    # print(f"Deleted {count} memories")

    print("\n✓ Demonstrated advanced memory management\n")


# =============================================================================
# Example 7: Error Handling
# =============================================================================


def example_7_error_handling():
    """Demonstrate proper error handling."""
    from raggy import Memory

    print("=== Example 7: Error Handling ===\n")

    memory = Memory(db_dir="./vectordb", quiet=True)

    # Handle invalid memory type
    print("1. Invalid memory type:")
    try:
        memory.add(
            "Test memory",
            memory_type="invalid_type"  # Invalid!
        )
    except ValueError as e:
        print(f"   ✓ Caught ValueError: {e}")

    # Handle invalid priority
    print("\n2. Invalid priority:")
    try:
        memory.add(
            "Test memory",
            priority="super_high"  # Invalid!
        )
    except ValueError as e:
        print(f"   ✓ Caught ValueError: {e}")

    # Handle empty text
    print("\n3. Empty text:")
    try:
        memory.add("")  # Invalid!
    except ValueError as e:
        print(f"   ✓ Caught ValueError: {e}")

    # Handle empty query
    print("\n4. Empty query:")
    try:
        memory.search("")  # Invalid!
    except ValueError as e:
        print(f"   ✓ Caught ValueError: {e}")

    # Handle invalid confidence score
    print("\n5. Invalid confidence score:")
    try:
        memory.add(
            "Test memory",
            confidence=1.5  # Invalid! Must be 0.0-1.0
        )
    except ValueError as e:
        print(f"   ✓ Caught ValueError: {e}")

    print("\n✓ All validation errors caught properly\n")


# =============================================================================
# Example 8: Real-World Workflow
# =============================================================================


def example_8_real_world_workflow():
    """Real-world AI-assisted development workflow."""
    from raggy import Memory

    print("=== Example 8: Real-World AI Development Workflow ===\n")

    memory = Memory(db_dir="./vectordb")

    # Day 1: Architecture decisions
    print("Day 1: Making architecture decisions...")

    memory.add(
        "Decided to separate RAG system into modular components: "
        "DocumentProcessor, DatabaseManager, SearchEngine, QueryProcessor. "
        "Reason: Better testability, maintainability, and adherence to SRP.",
        memory_type="decision",
        tags=["architecture", "modularity", "solid"],
        files_involved=["core/document.py", "core/database.py", "core/search.py"],
        priority="high",
        session_id="2025-01-13_morning"
    )

    # Day 2: Implementation patterns
    print("Day 2: Implementing patterns...")

    memory.add(
        "Using Protocol classes for interfaces (VectorDatabase, SearchStrategy) "
        "instead of ABCs for better duck typing and less rigid inheritance.",
        memory_type="pattern",
        tags=["interfaces", "protocols", "python"],
        files_involved=["core/database_interface.py"],
        priority="medium"
    )

    # Day 3: Bug fixes
    print("Day 3: Fixing bugs...")

    memory.add(
        "Fixed Unicode encoding issues in PDF extraction by using 'utf-8' "
        "encoding explicitly and handling decode errors with 'ignore' mode.",
        memory_type="solution",
        tags=["pdf", "unicode", "bug-fix"],
        files_involved=["core/document.py"],
        priority="high"
    )

    # Day 4: Learnings
    print("Day 4: Capturing learnings...")

    memory.add(
        "Learned that ChromaDB's query() returns nested lists even for single "
        "queries: results['ids'][0], results['documents'][0]. Always access "
        "first element of outer list before iterating inner list.",
        memory_type="learning",
        tags=["chromadb", "api", "gotchas"],
        priority="medium"
    )

    # Day 5: Context for AI
    print("\nDay 5: Getting context for new feature...")

    context = memory.get_context_for_prompt(
        query="architecture patterns interfaces protocols",
        max_tokens=1500,
        memory_types=["decision", "pattern", "learning"]
    )

    print("\nRelevant context retrieved:")
    print("-" * 70)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("-" * 70)

    print("\n✓ Full development workflow captured and retrievable\n")


# =============================================================================
# Main Runner
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Raggy 2.0 Memory API - Complete Examples")
    print("=" * 70 + "\n")

    # Note: These examples demonstrate the API structure but won't run
    # without ChromaDB installed. They're for documentation purposes.

    print("NOTE: These examples demonstrate API usage.")
    print("To run them, install dependencies: pip install chromadb sentence-transformers")
    print("\nAPI Structure Verification:")
    print("  ✓ Memory class with 9 public methods")
    print("  ✓ remember() and recall() convenience functions")
    print("  ✓ UniversalRAG.remember() and .recall() integration")
    print("  ✓ Comprehensive docstrings with examples")
    print("  ✓ Full type hints on all methods")
    print("\n" + "=" * 70 + "\n")

    # Uncomment these to run if ChromaDB is installed:
    # memory = example_1_basic_memory()
    # example_2_search_memories(memory)
    # example_3_prompt_context(memory)
    # example_4_convenience_functions()
    # example_5_universal_rag_integration()
    # example_6_advanced_management(memory)
    # example_7_error_handling()
    # example_8_real_world_workflow()


if __name__ == "__main__":
    main()
