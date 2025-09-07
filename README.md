# Universal RAG System v2.0.0

Enhanced ChromaDB RAG setup with hybrid search, smart chunking, and normalized scoring.

## Quick Start

```bash
# First-time setup (required)
python raggy.py init                     # Initialize environment and install dependencies

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

> **💡 TIP**: Instead of manually creating expansions, let AI extract domain-specific terms from your documents and generate the config for you! Ask Claude or ChatGPT: "Analyze my documentation and create raggy_config.yaml expansions for these terms: [list key terms from your docs]". This saves time and ensures you capture the right synonyms and related concepts.

## Key Features

- **Hybrid Search**: Combines semantic + BM25 keyword ranking
- **Smart Chunking**: Markdown-aware document processing
- **Normalized Scoring**: 0-1 scores with quality labels (Excellent/Good/Fair/Poor)
- **Query Expansion**: Automatic synonym expansion for domain terms
- **Model Presets**: fast/balanced/multilingual/accurate options
- **Universal**: Works with any project's documentation

## Requirements

- Python 3.8+
- `uv` package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

## Setup

1. **Install uv** (if not already installed):
   ```bash
   # On macOS/Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows:
   # Download and run installer from https://astral.sh/uv/getting-started/installation/
   ```

2. **Initialize the project**:
   ```bash
   python raggy.py init
   ```
   This will:
   - Create a virtual environment (`.venv`)
   - Generate `pyproject.toml`
   - Install all dependencies
   - Create a `docs/` directory

3. **Add your documents** to the `docs/` directory (`.md` or `.pdf` files)

4. **Index your documents**:
   ```bash
   python raggy.py build
   ```

5. **Start searching**:
   ```bash
   python raggy.py search "your query"
   ```

## AI Agent Integration

To create knowledge-driven development with continuous context, add this prompt to your `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, or similar AI agent instruction files:

````markdown
# MANDATORY: Knowledge-Driven Development Workflow

You are a senior development partner. For EVERY task, you MUST follow this exact workflow:

## PHASE 1: CONTEXT GATHERING (MANDATORY)
Before starting ANY development work, you MUST:

1. **Read Development State**:
   - ALWAYS read `./docs/DEVELOPMENT_STATE.md` first to understand:
     - What was accomplished in the previous task
     - Current project status and active features
     - Next planned steps and priorities
     - Any blockers or decisions pending

2. **Query Project Knowledge**:
   - Run: `python raggy.py search "[current task/feature keywords]"`
   - Run: `python raggy.py search "architecture patterns"`
   - Run: `python raggy.py search "coding standards"`
   - Run: `python raggy.py search "[relevant tech stack/framework]"`
   - Search for ANY technical context related to the current task

3. **Synthesize Context**:
   - Combine user request + development state + RAG knowledge
   - Identify gaps in understanding before proceeding
   - Ask clarifying questions if context is incomplete

## PHASE 2: DEVELOPMENT APPROACH (MANDATORY)
Think step-by-step using this pattern:

1. **Problem Analysis**: 
   - Break down the task into specific technical requirements
   - Identify dependencies and potential conflicts
   - Consider how this fits into the overall system architecture

2. **Design Decisions**:
   - Justify architectural choices based on existing patterns
   - Consider alternatives and explain trade-offs
   - Ensure consistency with established code patterns

3. **Implementation Plan**:
   - Create concrete steps with clear success criteria
   - Identify testing approach and validation methods
   - Plan for error handling and edge cases

## PHASE 3: EXECUTION WITH VERIFICATION
During development:
1. **Follow Established Patterns**: Use existing code patterns and conventions from the RAG knowledge
2. **Progressive Validation**: Test each step before moving to the next
3. **Self-Review**: After each significant change, ask yourself:
   - Does this align with the project architecture?
   - Am I following the established coding standards?
   - Have I handled error cases appropriately?
   - Is this solution maintainable and extensible?

## PHASE 4: DOCUMENTATION (MANDATORY)
After EVERY task completion, you MUST:

1. **Update Development State**:
   Update `./docs/DEVELOPMENT_STATE.md` with:
   - **COMPLETED**: Detailed description of what was implemented
   - **DECISIONS**: All architectural and technical decisions made
   - **CHANGES**: Files modified, new dependencies, configuration changes
   - **TESTING**: What was tested and validation results
   - **NEXT STEPS**: Immediate follow-up tasks and long-term considerations
   - **BLOCKERS**: Any issues discovered or decisions needed

2. **Log to RAG Database**:
   Create `./docs/dev_log_[timestamp].md` with:
   - Technical decisions and rationale
   - Code patterns used and why
   - Integration points and dependencies
   - Performance considerations
   - Security implications
   - Future refactoring opportunities

3. **Rebuild RAG**:
   Run: `python raggy.py build`  # Ensure new knowledge is indexed

## CRITICAL SUCCESS BEHAVIORS:

✅ **ALWAYS** start with `./docs/DEVELOPMENT_STATE.md` - NO EXCEPTIONS  
✅ **ALWAYS** query RAG for relevant context before coding  
✅ **NEVER** make architectural decisions without understanding existing patterns  
✅ **ALWAYS** document decisions immediately, not later  
✅ **ALWAYS** think step-by-step and show your reasoning  
✅ **ALWAYS** validate your work against existing standards  
✅ **ALWAYS** update both `./docs/DEVELOPMENT_STATE.md` and create dev logs  

## FAILURE CONDITIONS:
❌ Starting development without reading `./docs/DEVELOPMENT_STATE.md`  
❌ Making changes without querying relevant RAG context  
❌ Completing tasks without proper documentation updates  
❌ Ignoring established patterns or architectural decisions  
❌ Skipping the knowledge update cycle

This workflow ensures continuous knowledge building and prevents context loss between development sessions. Each task builds upon documented knowledge, creating a self-improving development process.
````

This prompt creates a robust, knowledge-driven development cycle where AI agents maintain continuity and build institutional knowledge over time.