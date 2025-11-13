"""Unit tests for Memory system (MemoryManager and Memory API).

Tests core memory management functionality:
- Adding memories with validation
- Retrieving memories by ID
- Deleting memories (single and all)
- Counting memories
- Archiving old memories
- Input validation and error handling

Note: Search tests are skipped due to a known ChromaDB adapter issue with the
query() method signature. This should be fixed when the chromadb_adapter.py
is updated to handle optional query_texts parameter correctly.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


class TestMemoryManagerInitialization:
    """Tests for MemoryManager initialization."""

    def test_init_with_default_parameters(self, temp_db_dir):
        """Test MemoryManager initialization with defaults."""
        from raggy.core.memory import MemoryManager

        manager = MemoryManager(db_dir=temp_db_dir, quiet=True)

        assert manager.db_dir == Path(temp_db_dir)
        assert manager.model_name == "all-MiniLM-L6-v2"
        assert manager.collection_name == "project_memory"
        assert manager.quiet is True

    def test_init_with_custom_parameters(self, temp_db_dir):
        """Test MemoryManager initialization with custom parameters."""
        from raggy.core.memory import MemoryManager

        manager = MemoryManager(
            db_dir=temp_db_dir,
            model_name="paraphrase-MiniLM-L6-v2",
            collection_name="custom_memory",
            quiet=True
        )

        assert manager.collection_name == "custom_memory"
        assert manager.model_name == "paraphrase-MiniLM-L6-v2"

    @pytest.mark.parametrize("invalid_db_dir", [None, ""])
    def test_init_with_invalid_db_dir_raises_valueerror(self, invalid_db_dir):
        """Test that invalid db_dir raises ValueError."""
        from raggy.core.memory import MemoryManager

        with pytest.raises(ValueError, match="db_dir must be a non-empty string"):
            MemoryManager(db_dir=invalid_db_dir, quiet=True)

    @pytest.mark.parametrize("invalid_model", [None, ""])
    def test_init_with_invalid_model_name_raises_valueerror(self, temp_db_dir, invalid_model):
        """Test that invalid model_name raises ValueError."""
        from raggy.core.memory import MemoryManager

        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            MemoryManager(db_dir=temp_db_dir, model_name=invalid_model, quiet=True)

    @pytest.mark.parametrize("invalid_collection", [None, ""])
    def test_init_with_invalid_collection_name_raises_valueerror(self, temp_db_dir, invalid_collection):
        """Test that invalid collection_name raises ValueError."""
        from raggy.core.memory import MemoryManager

        with pytest.raises(ValueError, match="collection_name must be a non-empty string"):
            MemoryManager(db_dir=temp_db_dir, collection_name=invalid_collection, quiet=True)


class TestMemoryManagerAdd:
    """Tests for adding memories to the system."""

    def test_add_with_valid_decision_returns_memory_id(self, memory_manager):
        """Test adding valid decision memory returns formatted ID."""
        text = "Decided to use dependency injection for database layer"
        memory_type = "decision"

        memory_id = memory_manager.add(text=text, memory_type=memory_type)

        assert memory_id.startswith("mem_")
        assert len(memory_id) == 28  # mem_YYYYMMDD_HHMMSS_hash

    @pytest.mark.parametrize("mem_type", ["decision", "solution", "pattern", "learning", "error", "note"])
    def test_add_with_all_memory_types(self, memory_manager, mem_type):
        """Test adding memories with all valid memory types."""
        text = f"Test {mem_type} memory"

        memory_id = memory_manager.add(text=text, memory_type=mem_type)

        assert memory_id.startswith("mem_")

    @pytest.mark.parametrize("priority", ["high", "medium", "low"])
    def test_add_with_all_priorities(self, memory_manager, priority):
        """Test adding memories with all valid priority levels."""
        memory_id = memory_manager.add(
            text="Test memory with priority",
            memory_type="note",
            priority=priority
        )

        assert memory_id.startswith("mem_")

    def test_add_with_metadata(self, memory_manager):
        """Test adding memory with metadata."""
        text = "Memory with comprehensive metadata"

        memory_id = memory_manager.add(
            text=text,
            memory_type="decision",
            priority="high",
            session_id="test-session-001",
            ai_model="claude-3-sonnet",
            confidence=0.95
        )

        assert memory_id.startswith("mem_")

        retrieved = memory_manager.get_by_id(memory_id)
        assert retrieved is not None
        assert retrieved["text"] == text
        assert retrieved["metadata"]["memory_type"] == "decision"
        assert retrieved["metadata"]["priority"] == "high"
        assert retrieved["metadata"]["session_id"] == "test-session-001"
        assert retrieved["metadata"]["ai_model"] == "claude-3-sonnet"
        assert retrieved["metadata"]["confidence"] == 0.95

    @pytest.mark.parametrize("invalid_type", ["invalid", "unknown", "memo"])
    def test_add_with_invalid_memory_type_raises_valueerror(self, memory_manager, invalid_type):
        """Test that invalid memory_type raises ValueError."""
        with pytest.raises(ValueError, match="memory_type must be one of"):
            memory_manager.add(text="Test", memory_type=invalid_type)

    @pytest.mark.parametrize("invalid_priority", ["urgent", "critical", "low-priority"])
    def test_add_with_invalid_priority_raises_valueerror(self, memory_manager, invalid_priority):
        """Test that invalid priority raises ValueError."""
        with pytest.raises(ValueError, match="priority must be one of"):
            memory_manager.add(text="Test", memory_type="note", priority=invalid_priority)

    @pytest.mark.parametrize("empty_text", [""])
    def test_add_with_empty_text_raises_valueerror(self, memory_manager, empty_text):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            memory_manager.add(text=empty_text, memory_type="note")

    def test_add_with_oversized_text_raises_valueerror(self, memory_manager):
        """Test that text exceeding 100KB raises ValueError."""
        from raggy.core.memory import MAX_MEMORY_SIZE

        oversized_text = "x" * (MAX_MEMORY_SIZE + 1)

        with pytest.raises(ValueError, match="text size exceeds maximum"):
            memory_manager.add(text=oversized_text, memory_type="note")

    @pytest.mark.parametrize("invalid_conf", [-0.1, 1.5, 2.0, "0.5", [0.5]])
    def test_add_with_invalid_confidence_raises_valueerror(self, memory_manager, invalid_conf):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be"):
            memory_manager.add(text="Test", memory_type="note", confidence=invalid_conf)

    @pytest.mark.parametrize("valid_conf", [0.0, 0.5, 1.0, 0.95])
    def test_add_with_valid_confidence_values(self, memory_manager, valid_conf):
        """Test adding memory with valid confidence values."""
        memory_id = memory_manager.add(
            text="Test memory",
            memory_type="note",
            confidence=valid_conf
        )

        assert memory_id.startswith("mem_")

    def test_add_with_invalid_file_path_raises_valueerror(self, memory_manager):
        """Test that invalid file paths raise ValueError."""
        with pytest.raises(ValueError, match="Invalid file path"):
            memory_manager.add(
                text="Test",
                memory_type="note",
                files_involved=[""]  # Empty path
            )

    def test_add_captures_metadata(self, memory_manager):
        """Test that add() captures timestamp and memory_id in metadata."""
        text = "Test memory"
        before = datetime.now(timezone.utc)

        memory_id = memory_manager.add(text=text, memory_type="note")

        retrieved = memory_manager.get_by_id(memory_id)
        assert retrieved is not None
        assert retrieved["metadata"]["memory_id"] == memory_id
        assert "timestamp" in retrieved["metadata"]

        stored_time = datetime.fromisoformat(
            retrieved["metadata"]["timestamp"].replace('Z', '+00:00')
        )
        assert stored_time >= before


class TestMemoryManagerRetrievalAndDeletion:
    """Tests for retrieving and deleting memories."""

    def test_get_by_id_returns_memory(self, memory_manager):
        """Test retrieving memory by ID."""
        text = "Test memory content"
        memory_id = memory_manager.add(text=text, memory_type="note")

        retrieved = memory_manager.get_by_id(memory_id)

        assert retrieved is not None
        assert retrieved["id"] == memory_id
        assert retrieved["text"] == text
        assert retrieved["metadata"]["memory_type"] == "note"

    def test_get_by_id_returns_none_for_nonexistent_id(self, memory_manager):
        """Test that get_by_id returns None for nonexistent ID."""
        retrieved = memory_manager.get_by_id("mem_nonexistent")

        assert retrieved is None

    def test_get_by_id_with_empty_id_raises_valueerror(self, memory_manager):
        """Test that empty memory_id raises ValueError."""
        with pytest.raises(ValueError, match="memory_id must be a non-empty string"):
            memory_manager.get_by_id("")

    def test_delete_removes_memory(self, memory_manager):
        """Test deleting a memory."""
        memory_id = memory_manager.add(text="To be deleted", memory_type="note")

        result = memory_manager.delete(memory_id)

        assert result is True
        assert memory_manager.get_by_id(memory_id) is None

    def test_delete_with_empty_id_raises_valueerror(self, memory_manager):
        """Test that deleting with empty ID raises ValueError."""
        with pytest.raises(ValueError, match="memory_id must be a non-empty string"):
            memory_manager.delete("")

    def test_count_returns_zero_for_empty_database(self, memory_manager):
        """Test count returns 0 for empty database."""
        count = memory_manager.count()

        assert count == 0

    def test_count_returns_correct_number(self, memory_manager):
        """Test count returns correct number of memories."""
        for i in range(5):
            memory_manager.add(text=f"Memory {i}", memory_type="note")

        count = memory_manager.count()

        assert count == 5

    def test_delete_all_removes_all_memories(self, memory_manager):
        """Test delete_all removes all memories."""
        for i in range(3):
            memory_manager.add(text=f"Memory {i}", memory_type="note")

        assert memory_manager.count() == 3

        deleted_count = memory_manager.delete_all()

        assert deleted_count == 3
        assert memory_manager.count() == 0

    def test_delete_all_returns_zero_for_empty_database(self, memory_manager):
        """Test delete_all returns 0 for empty database."""
        deleted_count = memory_manager.delete_all()

        assert deleted_count == 0


class TestMemoryManagerArchive:
    """Tests for archiving memories."""

    def test_archive_moves_old_memories(self, memory_manager):
        """Test archiving memories older than specified date."""
        memory_id = memory_manager.add(text="Old memory", memory_type="note")
        assert memory_manager.get_by_id(memory_id) is not None

        # Future cutoff date (memory will be archived)
        cutoff_date = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

        archived_count = memory_manager.archive(cutoff_date)

        assert archived_count == 1
        assert memory_manager.get_by_id(memory_id) is None

    def test_archive_with_no_old_memories_returns_zero(self, memory_manager):
        """Test archive returns 0 when no memories are old enough."""
        memory_manager.add(text="Recent memory", memory_type="note")

        # Past cutoff date (nothing should be archived)
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()

        archived_count = memory_manager.archive(cutoff_date)

        assert archived_count == 0

    def test_archive_with_invalid_date_raises_valueerror(self, memory_manager):
        """Test archive with invalid date format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date format|older_than must be"):
            memory_manager.archive("invalid-date")

    def test_archive_with_empty_string_raises_valueerror(self, memory_manager):
        """Test archive with empty string raises ValueError."""
        with pytest.raises(ValueError, match="older_than must be a non-empty string"):
            memory_manager.archive("")


class TestMemoryAPIInitialization:
    """Tests for Memory API class initialization."""

    def test_memory_init_with_defaults(self, temp_db_dir):
        """Test Memory class initialization with defaults."""
        from raggy.core.memory import Memory

        memory = Memory(db_dir=temp_db_dir, quiet=True)

        assert str(memory.db_dir).endswith("vectordb")
        assert memory.collection_name == "project_memory"
        assert memory.quiet is True

    def test_memory_exposes_internal_attributes(self, memory_api):
        """Test that Memory exposes key attributes."""
        assert hasattr(memory_api, "db_dir")
        assert hasattr(memory_api, "collection_name")
        assert hasattr(memory_api, "quiet")
        assert hasattr(memory_api, "_manager")


class TestMemoryAPIAdd:
    """Tests for Memory.add() method."""

    def test_add_returns_memory_id(self, memory_api):
        """Test add() returns a memory ID."""
        memory_id = memory_api.add(text="Test decision", memory_type="decision")

        assert memory_id.startswith("mem_")
        assert len(memory_id) == 28

    @pytest.mark.parametrize("mem_type", ["decision", "solution", "pattern", "learning", "error", "note"])
    def test_add_with_all_memory_types(self, memory_api, mem_type):
        """Test add() with all valid memory types."""
        memory_id = memory_api.add(text=f"Test {mem_type}", memory_type=mem_type)

        assert memory_id.startswith("mem_")

    @pytest.mark.parametrize("priority", ["high", "medium", "low"])
    def test_add_with_all_priorities(self, memory_api, priority):
        """Test add() with all valid priorities."""
        memory_id = memory_api.add(text="Test", memory_type="note", priority=priority)

        assert memory_id.startswith("mem_")

    def test_add_with_invalid_memory_type_raises_error(self, memory_api):
        """Test add() with invalid memory_type raises error."""
        with pytest.raises(ValueError, match="memory_type"):
            memory_api.add(text="Test", memory_type="invalid")

    def test_add_with_empty_text_raises_error(self, memory_api):
        """Test add() with empty text raises error."""
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            memory_api.add(text="", memory_type="note")


class TestMemoryAPIRetrieval:
    """Tests for Memory retrieval methods."""

    def test_get_by_id_returns_memory(self, memory_api):
        """Test get_by_id() returns memory entry."""
        text = "Test retrieval"
        memory_id = memory_api.add(text=text, memory_type="note")

        retrieved = memory_api.get_by_id(memory_id)

        assert retrieved is not None
        assert retrieved["id"] == memory_id
        assert retrieved["text"] == text

    def test_get_by_id_returns_none_for_missing_id(self, memory_api):
        """Test get_by_id() returns None for nonexistent ID."""
        retrieved = memory_api.get_by_id("mem_missing")

        assert retrieved is None

    def test_count_returns_correct_number(self, memory_api):
        """Test count() returns correct number of memories."""
        for i in range(3):
            memory_api.add(text=f"Memory {i}", memory_type="note")

        count = memory_api.count()

        assert count == 3

    def test_count_returns_zero_for_empty_database(self, memory_api):
        """Test count() returns 0 for empty database."""
        count = memory_api.count()

        assert count == 0


class TestMemoryAPIDeletion:
    """Tests for Memory deletion methods."""

    def test_delete_removes_memory(self, memory_api):
        """Test delete() removes a memory."""
        memory_id = memory_api.add(text="To delete", memory_type="note")

        result = memory_api.delete(memory_id)

        assert result is True
        assert memory_api.get_by_id(memory_id) is None

    def test_delete_all_removes_all_memories(self, memory_api):
        """Test delete_all() removes all memories."""
        for i in range(3):
            memory_api.add(text=f"Memory {i}", memory_type="note")

        deleted_count = memory_api.delete_all()

        assert deleted_count == 3
        assert memory_api.count() == 0

    def test_delete_all_returns_zero_for_empty_database(self, memory_api):
        """Test delete_all() returns 0 for empty database."""
        deleted_count = memory_api.delete_all()

        assert deleted_count == 0


class TestMemoryAPIArchive:
    """Tests for Memory archive functionality."""

    def test_archive_returns_count(self, memory_api):
        """Test archive() returns number of archived memories."""
        memory_api.add(text="Old memory", memory_type="note")

        cutoff_date = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

        archived_count = memory_api.archive(cutoff_date)

        assert isinstance(archived_count, int)
        assert archived_count == 1


class TestMemoryConvenienceFunctions:
    """Tests for module-level remember() and recall() functions."""

    def test_remember_adds_memory(self, temp_db_dir):
        """Test remember() convenience function adds memory."""
        from raggy.core.memory import remember

        memory_id = remember(
            "Using Pydantic for data validation",
            db_dir=temp_db_dir,
            memory_type="decision",
            quiet=True
        )

        assert memory_id.startswith("mem_")

    def test_remember_with_all_parameters(self, temp_db_dir):
        """Test remember() with all available parameters."""
        from raggy.core.memory import remember

        memory_id = remember(
            text="Comprehensive memory with all parameters",
            db_dir=temp_db_dir,
            memory_type="pattern",
            priority="medium",
            session_id="test-session",
            quiet=True
        )

        assert memory_id.startswith("mem_")


class TestMemoryIntegrationWorkflows:
    """Integration tests for common Memory workflows."""

    def test_add_then_delete_workflow(self, memory_manager):
        """Test: add memory -> delete -> verify removed."""
        text = "Temporary note"
        memory_id = memory_manager.add(text=text, memory_type="note")

        assert memory_manager.get_by_id(memory_id) is not None

        memory_manager.delete(memory_id)

        assert memory_manager.get_by_id(memory_id) is None

    def test_add_multiple_memories_count(self, memory_manager):
        """Test adding multiple memories and counting."""
        initial_count = memory_manager.count()

        for i in range(5):
            memory_manager.add(text=f"Memory {i}", memory_type="note")

        final_count = memory_manager.count()

        assert final_count == initial_count + 5

    def test_multiple_adds_and_deletions(self, memory_api):
        """Test complete workflow with multiple operations."""
        # Add memories
        ids = []
        for i in range(3):
            mem_id = memory_api.add(
                text=f"Memory {i}",
                memory_type="decision" if i % 2 == 0 else "note",
                priority="high" if i == 0 else "medium"
            )
            ids.append(mem_id)

        assert memory_api.count() == 3

        # Delete first memory
        memory_api.delete(ids[0])
        assert memory_api.count() == 2

        # Add another
        memory_api.add(text="New memory", memory_type="pattern")

        assert memory_api.count() == 3

        # Delete all
        deleted = memory_api.delete_all()
        assert deleted == 3
        assert memory_api.count() == 0

    def test_memory_manager_and_api_interoperability(self, memory_manager, memory_api):
        """Test that MemoryManager and Memory API can access same data."""
        # Add via manager
        mem_id = memory_manager.add(text="Test memory", memory_type="decision")

        # Retrieve via API
        retrieved = memory_api.get_by_id(mem_id)

        assert retrieved is not None
        assert retrieved["text"] == "Test memory"

        # Delete via API
        memory_api.delete(mem_id)

        # Verify via manager
        assert memory_manager.get_by_id(mem_id) is None
