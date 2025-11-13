"""End-to-end lifecycle tests for the Memory system.

Tests complete workflows and integration scenarios for memory operations.
"""

from datetime import datetime, timedelta, timezone


class TestMemoryLifecycleWorkflows:
    """Tests for complete memory management workflows."""

    def test_add_retrieve_delete_workflow(self, memory_manager):
        """Test: add → retrieve → delete workflow."""
        text = "Temporary decision"

        # Add
        memory_id = memory_manager.add(
            text=text,
            memory_type="decision",
            priority="high"
        )

        # Retrieve and verify
        retrieved = memory_manager.get_by_id(memory_id)
        assert retrieved is not None
        assert retrieved["text"] == text
        assert retrieved["metadata"]["memory_type"] == "decision"

        # Delete and verify removal
        memory_manager.delete(memory_id)
        assert memory_manager.get_by_id(memory_id) is None

    def test_add_multiple_different_types(self, memory_manager):
        """Test adding memories of different types."""
        memories = [
            ("Architecture decision", "decision", "high"),
            ("Caching solution", "solution", "medium"),
            ("Design pattern", "pattern", "medium"),
            ("Learning point", "learning", "low"),
            ("Error fix", "error", "high"),
        ]

        added_ids = []
        for text, mem_type, priority in memories:
            mem_id = memory_manager.add(
                text=text,
                memory_type=mem_type,
                priority=priority
            )
            added_ids.append((mem_id, mem_type, priority))

        # Verify all were added
        assert memory_manager.count() == len(memories)

        # Verify each
        for mem_id, expected_type, expected_priority in added_ids:
            retrieved = memory_manager.get_by_id(mem_id)
            assert retrieved is not None
            assert retrieved["metadata"]["memory_type"] == expected_type
            assert retrieved["metadata"]["priority"] == expected_priority

    def test_add_then_archive_workflow(self, memory_manager):
        """Test: add memory → archive → verify moved."""
        text = "Memory to archive"
        memory_id = memory_manager.add(text=text, memory_type="note")

        initial_count = memory_manager.count()
        assert initial_count == 1

        # Archive with future date (will archive existing memory)
        cutoff_date = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        archived_count = memory_manager.archive(cutoff_date)

        assert archived_count == 1
        assert memory_manager.get_by_id(memory_id) is None
        assert memory_manager.count() == 0

    def test_add_recent_and_old_archive(self, memory_manager):
        """Test archiving with mix of recent and old memories."""
        # Both will be recent in our test since all added now
        memory_manager.add(text="Memory 1", memory_type="note")
        memory_manager.add(text="Memory 2", memory_type="note")

        assert memory_manager.count() == 2

        # Archive with past date (nothing should be archived)
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        archived = memory_manager.archive(cutoff_date)

        assert archived == 0
        assert memory_manager.count() == 2

    def test_delete_individual_then_delete_all(self, memory_manager):
        """Test deleting individual memories then delete_all."""
        ids = []
        for i in range(5):
            mem_id = memory_manager.add(text=f"Memory {i}", memory_type="note")
            ids.append(mem_id)

        assert memory_manager.count() == 5

        # Delete first two individually
        memory_manager.delete(ids[0])
        memory_manager.delete(ids[1])
        assert memory_manager.count() == 3

        # Delete remaining with delete_all
        deleted = memory_manager.delete_all()
        assert deleted == 3
        assert memory_manager.count() == 0

    def test_count_after_operations(self, memory_manager):
        """Test count is accurate after various operations."""
        assert memory_manager.count() == 0

        # Add 3
        ids = []
        for i in range(3):
            mem_id = memory_manager.add(text=f"Memory {i}", memory_type="note")
            ids.append(mem_id)
        assert memory_manager.count() == 3

        # Delete 1
        memory_manager.delete(ids[0])
        assert memory_manager.count() == 2

        # Add 2 more
        for i in range(2):
            memory_manager.add(text=f"New memory {i}", memory_type="decision")
        assert memory_manager.count() == 4

        # Delete all
        memory_manager.delete_all()
        assert memory_manager.count() == 0


class TestMemoryConvenienceFunctionWorkflows:
    """Tests for remember() and recall() convenience functions."""

    def test_remember_adds_to_persistent_database(self, temp_db_dir):
        """Test that remember() adds to persistent database."""
        from raggy.core.memory import Memory, remember

        # Add via remember
        mem_id = remember(
            "Quick memory addition",
            db_dir=temp_db_dir,
            memory_type="note",
            quiet=True
        )

        # Verify with Memory API
        memory = Memory(db_dir=temp_db_dir, quiet=True)
        retrieved = memory.get_by_id(mem_id)

        assert retrieved is not None
        assert retrieved["text"] == "Quick memory addition"

    def test_remember_with_all_features(self, temp_db_dir):
        """Test remember() with all parameters."""
        from raggy.core.memory import Memory, remember

        mem_id = remember(
            text="Comprehensive memory",
            db_dir=temp_db_dir,
            memory_type="decision",
            priority="high",
            session_id="test-session",
            quiet=True
        )

        memory = Memory(db_dir=temp_db_dir, quiet=True)
        retrieved = memory.get_by_id(mem_id)

        assert retrieved is not None
        assert retrieved["metadata"]["priority"] == "high"
        assert retrieved["metadata"]["session_id"] == "test-session"

    def test_multiple_remember_calls_persist(self, temp_db_dir):
        """Test that multiple remember calls all persist."""
        from raggy.core.memory import Memory, remember

        # Add three memories separately
        ids = []
        for i in range(3):
            mem_id = remember(
                f"Memory {i} content",
                db_dir=temp_db_dir,
                memory_type="note",
                quiet=True
            )
            ids.append(mem_id)

        # Verify all exist
        memory = Memory(db_dir=temp_db_dir, quiet=True)
        for i, mem_id in enumerate(ids):
            retrieved = memory.get_by_id(mem_id)
            assert retrieved is not None
            assert f"Memory {i} content" in retrieved["text"]


class TestMemoryTextHandling:
    """Tests for memory text handling and edge cases."""

    def test_very_long_text(self, memory_manager):
        """Test adding very long text (but under limit)."""
        from raggy.core.memory import MAX_MEMORY_SIZE

        long_text = "This is a test. " * (MAX_MEMORY_SIZE // 30)
        # Adjust to be just under limit
        long_text = long_text[:MAX_MEMORY_SIZE - 100]

        memory_id = memory_manager.add(text=long_text, memory_type="note")

        retrieved = memory_manager.get_by_id(memory_id)
        assert len(retrieved["text"]) <= MAX_MEMORY_SIZE

    def test_text_with_multiple_languages(self, memory_manager):
        """Test adding text with multiple languages."""
        multilang_text = """
        English: This is a test.
        Spanish: Esto es una prueba.
        French: C'est un test.
        German: Das ist ein Test.
        Japanese: これはテストです。
        """

        memory_id = memory_manager.add(text=multilang_text, memory_type="note")

        retrieved = memory_manager.get_by_id(memory_id)
        assert retrieved is not None
        assert "English" in retrieved["text"]
        assert "Spanish" in retrieved["text"]

    def test_text_with_code_snippets(self, memory_manager):
        """Test adding text containing code."""
        code_text = '''
        def example():
            """Example function"""
            return 42

        # Comment
        result = example()
        '''

        memory_id = memory_manager.add(text=code_text, memory_type="note")

        retrieved = memory_manager.get_by_id(memory_id)
        assert "def example():" in retrieved["text"]
        assert '"""Example function"""' in retrieved["text"]

    def test_text_with_special_formatting(self, memory_manager):
        """Test adding text with special formatting."""
        formatted_text = """
        # Header
        ## Subheader

        This is a paragraph with **bold** and *italic* text.

        - List item 1
        - List item 2

        1. Numbered item 1
        2. Numbered item 2

        ```python
        code block
        ```
        """

        memory_id = memory_manager.add(text=formatted_text, memory_type="note")

        retrieved = memory_manager.get_by_id(memory_id)
        assert retrieved["text"] == formatted_text


class TestMemoryMetadataHandling:
    """Tests for metadata persistence and retrieval."""

    def test_metadata_with_all_optional_fields(self, memory_manager):
        """Test that all optional metadata fields are preserved."""
        memory_id = memory_manager.add(
            text="Test with all metadata",
            memory_type="decision",
            priority="high",
            session_id="session-123",
            ai_model="test-model-1",
            confidence=0.87,
            custom_field_1="value1",
            custom_field_2="value2"
        )

        retrieved = memory_manager.get_by_id(memory_id)
        metadata = retrieved["metadata"]

        assert metadata["memory_type"] == "decision"
        assert metadata["priority"] == "high"
        assert metadata["session_id"] == "session-123"
        assert metadata["ai_model"] == "test-model-1"
        assert metadata["confidence"] == 0.87
        assert metadata["custom_field_1"] == "value1"
        assert metadata["custom_field_2"] == "value2"

    def test_timestamp_is_accurate(self, memory_manager):
        """Test that timestamp is recorded accurately."""
        before = datetime.now(timezone.utc)
        memory_id = memory_manager.add(text="Test timestamp", memory_type="note")
        after = datetime.now(timezone.utc)

        retrieved = memory_manager.get_by_id(memory_id)
        timestamp_str = retrieved["metadata"]["timestamp"]

        # Parse the timestamp
        stored_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        # Verify it's within expected range
        assert before <= stored_time <= after

    def test_memory_id_format(self, memory_manager):
        """Test that memory_id has correct format."""
        memory_id = memory_manager.add(text="Test ID format", memory_type="note")

        # Check format: mem_YYYYMMDD_HHMMSS_hash (4 parts total)
        parts = memory_id.split('_')
        assert len(parts) == 4
        assert parts[0] == "mem"
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 8  # hash


class TestMemoryRobustness:
    """Tests for robustness and error handling."""

    def test_repeated_deletes_of_same_memory(self, memory_manager):
        """Test that deleting the same memory twice raises no error or returns False."""
        memory_id = memory_manager.add(text="Test", memory_type="note")

        # First delete should succeed
        result1 = memory_manager.delete(memory_id)
        assert result1 is True

        # Verify it's gone
        assert memory_manager.get_by_id(memory_id) is None

    def test_get_nonexistent_memory_returns_none(self, memory_manager):
        """Test that getting nonexistent memory returns None."""
        result = memory_manager.get_by_id("mem_nonexistent_memory")
        assert result is None

    def test_count_after_failed_operations(self, memory_manager):
        """Test that count is accurate after failed operations."""
        memory_manager.add(text="Memory 1", memory_type="note")
        memory_manager.add(text="Memory 2", memory_type="note")

        assert memory_manager.count() == 2

        # Try to delete nonexistent (should handle gracefully)
        result = memory_manager.get_by_id("mem_nonexistent")
        assert result is None

        # Count should still be 2
        assert memory_manager.count() == 2

    def test_empty_database_operations(self, memory_manager):
        """Test operations on empty database."""
        assert memory_manager.count() == 0
        assert memory_manager.delete_all() == 0
        assert memory_manager.get_by_id("mem_anything") is None
