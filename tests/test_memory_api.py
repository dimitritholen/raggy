"""Tests for Memory public API interactions and edge cases."""

from datetime import datetime, timedelta, timezone

import pytest


class TestMemoryAPIEdgeCases:
    """Tests for Memory API edge cases."""

    def test_add_with_maximum_text_size(self, memory_api):
        """Test adding memory with maximum allowed text size."""
        from raggy.core.memory import MAX_MEMORY_SIZE

        # Create text that's just under the limit
        max_text = "x" * (MAX_MEMORY_SIZE - 10)

        memory_id = memory_api.add(text=max_text, memory_type="note")

        assert memory_id.startswith("mem_")
        retrieved = memory_api.get_by_id(memory_id)
        assert len(retrieved["text"]) == MAX_MEMORY_SIZE - 10

    def test_add_with_unicode_text(self, memory_api):
        """Test adding memory with unicode characters."""
        unicode_text = "Testing with émojis 🎉 and spëcial çharacters"

        memory_id = memory_api.add(text=unicode_text, memory_type="note")

        retrieved = memory_api.get_by_id(memory_id)
        assert unicode_text in retrieved["text"]

    def test_add_with_newlines_and_special_chars(self, memory_api):
        """Test adding memory with newlines and special characters."""
        special_text = """Line 1
Line 2
Tab:	here
Quote: "quoted"
Apostrophe: it's"""

        memory_id = memory_api.add(text=special_text, memory_type="note")

        retrieved = memory_api.get_by_id(memory_id)
        assert retrieved["text"] == special_text

    def test_add_with_confidence_boundaries(self, memory_api):
        """Test adding with confidence at exact boundaries."""
        # Test exact 0.0
        id1 = memory_api.add(text="Min confidence", memory_type="note", confidence=0.0)
        result1 = memory_api.get_by_id(id1)
        assert result1["metadata"]["confidence"] == 0.0

        # Test exact 1.0
        id2 = memory_api.add(text="Max confidence", memory_type="note", confidence=1.0)
        result2 = memory_api.get_by_id(id2)
        assert result2["metadata"]["confidence"] == 1.0

    def test_consecutive_deletes_and_adds(self, memory_api):
        """Test alternating delete and add operations."""
        ids = []
        for i in range(3):
            mem_id = memory_api.add(text=f"Memory {i}", memory_type="note")
            ids.append(mem_id)

        # Delete first
        memory_api.delete(ids[0])
        assert memory_api.get_by_id(ids[0]) is None
        assert memory_api.get_by_id(ids[1]) is not None

        # Add new
        new_id = memory_api.add(text="New memory 1", memory_type="decision")
        assert memory_api.get_by_id(new_id) is not None

        # Delete another
        memory_api.delete(ids[1])

        # Count should be 2 (new + ids[2])
        assert memory_api.count() == 2

    def test_add_many_memories_performance(self, memory_api):
        """Test adding many memories."""
        for i in range(20):
            memory_api.add(
                text=f"Memory content {i}",
                memory_type="note",
                priority="high" if i % 3 == 0 else "medium"
            )

        count = memory_api.count()
        assert count == 20

    def test_metadata_persistence(self, memory_api):
        """Test that metadata is correctly persisted and retrieved."""
        memory_id = memory_api.add(
            text="Memory with metadata",
            memory_type="decision",
            priority="high",
            session_id="session-123",
            ai_model="test-model",
            confidence=0.85,
            custom_key="custom_value"
        )

        retrieved = memory_api.get_by_id(memory_id)
        metadata = retrieved["metadata"]

        assert metadata["memory_type"] == "decision"
        assert metadata["priority"] == "high"
        assert metadata["session_id"] == "session-123"
        assert metadata["ai_model"] == "test-model"
        assert metadata["confidence"] == 0.85
        assert metadata["custom_key"] == "custom_value"

    def test_archive_with_valid_iso_dates(self, memory_api):
        """Test archive with various ISO date formats."""
        memory_api.add(text="Memory for archiving", memory_type="note")

        # ISO format with Z
        cutoff_date = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        if not cutoff_date.endswith('Z'):
            cutoff_date = cutoff_date.split('+')[0] + 'Z'

        archived = memory_api.archive(cutoff_date)
        assert archived == 1


class TestMemoryAPIValidation:
    """Tests for Memory API input validation."""

    @pytest.mark.parametrize("invalid_query", [""])
    def test_get_context_with_empty_query_raises_error(self, memory_api, invalid_query):
        """Test get_context_for_prompt with empty query raises error."""
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            memory_api.get_context_for_prompt(invalid_query)

    @pytest.mark.skip(reason="get_context_for_prompt calls search - ChromaDB adapter issue")
    def test_get_context_with_low_max_tokens_raises_error(self, memory_api):
        """Test get_context_for_prompt with max_tokens < 100 raises error."""
        with pytest.raises(ValueError, match="max_tokens must be >= 100"):
            memory_api.get_context_for_prompt("test query", max_tokens=50)

    def test_delete_with_invalid_id_types(self, memory_api):
        """Test delete with different invalid ID types."""
        with pytest.raises(ValueError, match="memory_id must be a non-empty string"):
            memory_api.delete("")

    def test_get_by_id_with_invalid_id_types(self, memory_api):
        """Test get_by_id with different invalid ID types."""
        with pytest.raises(ValueError, match="memory_id must be a non-empty string"):
            memory_api.get_by_id("")


class TestMemoryPriorityAndType:
    """Tests for priority and memory type handling."""

    def test_all_memory_types_stored_and_retrieved(self, memory_api):
        """Test all memory types can be stored and retrieved."""
        memory_types = ["decision", "solution", "pattern", "learning", "error", "note"]
        added_ids = {}

        for mem_type in memory_types:
            mem_id = memory_api.add(
                text=f"Test {mem_type} memory",
                memory_type=mem_type
            )
            added_ids[mem_type] = mem_id

        # Verify all can be retrieved
        for mem_type, mem_id in added_ids.items():
            retrieved = memory_api.get_by_id(mem_id)
            assert retrieved is not None
            assert retrieved["metadata"]["memory_type"] == mem_type

    def test_all_priorities_stored_and_retrieved(self, memory_api):
        """Test all priority levels can be stored and retrieved."""
        priorities = ["high", "medium", "low"]
        added_ids = {}

        for priority in priorities:
            mem_id = memory_api.add(
                text=f"Test {priority} priority memory",
                memory_type="note",
                priority=priority
            )
            added_ids[priority] = mem_id

        # Verify all can be retrieved
        for priority, mem_id in added_ids.items():
            retrieved = memory_api.get_by_id(mem_id)
            assert retrieved is not None
            assert retrieved["metadata"]["priority"] == priority

    @pytest.mark.parametrize("invalid_type", ["unknown", "memo", "event", ""])
    def test_invalid_memory_types_rejected(self, memory_api, invalid_type):
        """Test that invalid memory types are rejected."""
        with pytest.raises(ValueError, match="memory_type"):
            memory_api.add(text="Test", memory_type=invalid_type)

    @pytest.mark.parametrize("invalid_priority", ["urgent", "critical", ""])
    def test_invalid_priorities_rejected(self, memory_api, invalid_priority):
        """Test that invalid priorities are rejected."""
        with pytest.raises(ValueError, match="priority"):
            memory_api.add(text="Test", memory_type="note", priority=invalid_priority)
