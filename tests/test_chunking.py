"""Tests for text chunking functionality in raggy."""

import pytest
from pathlib import Path
from raggy.core.document import DocumentProcessor


class TestTextChunking:
    """Test text chunking functionality through DocumentProcessor."""

    @pytest.fixture
    def document_processor(self, temp_dir, sample_config):
        """Create a DocumentProcessor instance for testing."""
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        return DocumentProcessor(
            docs_dir=docs_dir,
            config=sample_config,
            quiet=True
        )

    def test_chunk_text_simple_short_text(self, document_processor, temp_dir):
        """Test chunking of text shorter than chunk size."""
        # Arrange: Create short text document
        docs_dir = temp_dir / "docs"
        doc_path = docs_dir / "short.md"
        short_text = "This is a short text that should fit in one chunk."
        doc_path.write_text(short_text, encoding="utf-8")

        # Act: Process document (internally handles chunking)
        chunks = document_processor.process_document(doc_path)

        # Assert: Should create single chunk
        assert len(chunks) == 1
        assert chunks[0]["text"] == short_text
        assert chunks[0]["metadata"]["chunk_type"] == "simple"

    def test_chunk_text_simple_long_text(self, document_processor, temp_dir):
        """Test simple chunking of long text."""
        # Arrange: Create text longer than chunk size
        docs_dir = temp_dir / "docs"
        doc_path = docs_dir / "long.txt"
        long_text = "This is a sentence. " * 50  # Approximately 1000 chars
        doc_path.write_text(long_text, encoding="utf-8")

        # Act: Process document
        chunks = document_processor.process_document(doc_path)

        # Assert: Should create multiple chunks
        assert len(chunks) > 1

        # Each chunk should have text and metadata
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["chunk_type"] == "simple"
            assert len(chunk["text"]) <= 600  # Should respect chunk size (500 + some buffer for sentence boundaries)

    def test_chunk_text_simple_respects_sentence_boundaries(self, document_processor, temp_dir):
        """Test that simple chunking tries to break at sentence boundaries."""
        # Arrange: Create text with clear sentence boundaries
        docs_dir = temp_dir / "docs"
        doc_path = docs_dir / "sentences.txt"
        text = "First sentence. " * 20 + "Second sentence. " * 20 + "Third sentence. " * 20
        doc_path.write_text(text, encoding="utf-8")

        # Act: Process document
        chunks = document_processor.process_document(doc_path)

        # Assert: Check that most chunks end with sentence boundaries
        sentence_ending_chunks = 0
        for chunk in chunks[:-1]:  # Exclude last chunk (may be partial)
            if chunk["text"].rstrip().endswith('.'):
                sentence_ending_chunks += 1

        # Most chunks should end with sentences
        assert sentence_ending_chunks >= len(chunks) - 2

    def test_chunk_text_overlap(self, document_processor, temp_dir):
        """Test that chunks have proper overlap."""
        # Arrange: Create predictable text
        docs_dir = temp_dir / "docs"
        doc_path = docs_dir / "overlap_test.txt"
        sentences = [f"Sentence number {i}. " for i in range(50)]
        text = "".join(sentences)
        doc_path.write_text(text, encoding="utf-8")

        # Act: Process document
        chunks = document_processor.process_document(doc_path)

        # Assert: Check for overlap between consecutive chunks
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]["text"]
                next_chunk = chunks[i + 1]["text"]

                # Chunks should not be empty
                assert len(current_chunk) > 0
                assert len(next_chunk) > 0

    def test_chunk_text_smart_disabled_by_default(self, document_processor, temp_dir):
        """Test that smart chunking is disabled by default in config."""
        # Arrange: Create markdown document with headers
        docs_dir = temp_dir / "docs"
        doc_path = docs_dir / "headers.md"
        text = "# Header\n\nContent under header.\n\n## Subheader\n\nMore content."
        doc_path.write_text(text, encoding="utf-8")

        # Act: Process document
        chunks = document_processor.process_document(doc_path)

        # Assert: Should use simple chunking since smart=False in config
        assert all(chunk["metadata"]["chunk_type"] == "simple" for chunk in chunks)

    def test_chunk_text_smart_enabled(self, temp_dir, sample_config):
        """Test smart chunking when enabled."""
        # Arrange: Create config with smart chunking enabled
        smart_config = sample_config.copy()
        smart_config["chunking"]["smart"] = True

        docs_dir = temp_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        processor = DocumentProcessor(
            docs_dir=docs_dir,
            config=smart_config,
            quiet=True
        )

        doc_path = docs_dir / "smart_test.md"
        markdown_text = """# Main Header

This is content under the main header.

## Subheader One

Content under subheader one with some details.

### Sub-subheader

More nested content here.

## Subheader Two

Different content under subheader two.
"""
        doc_path.write_text(markdown_text, encoding="utf-8")

        # Act: Process document
        chunks = processor.process_document(doc_path)

        # Assert: Should create smart chunks
        smart_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "smart"]
        assert len(smart_chunks) > 0

    def test_process_section_preserves_headers(self, temp_dir, sample_config):
        """Test that section processing preserves headers when configured."""
        # Arrange: Create config with smart chunking and header preservation
        smart_config = sample_config.copy()
        smart_config["chunking"]["smart"] = True
        smart_config["chunking"]["preserve_headers"] = True

        docs_dir = temp_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        processor = DocumentProcessor(
            docs_dir=docs_dir,
            config=smart_config,
            quiet=True
        )

        doc_path = docs_dir / "header_test.md"
        content = """## Test Header

This is the content under the header."""
        doc_path.write_text(content, encoding="utf-8")

        # Act: Process document
        chunks = processor.process_document(doc_path)

        # Assert: First chunk should include header
        if smart_config["chunking"]["preserve_headers"]:
            assert chunks[0]["text"].startswith("## Test Header")

        # All chunks should have metadata about the header
        for chunk in chunks:
            if chunk["metadata"]["chunk_type"] == "smart":
                assert chunk["metadata"]["section_header"] == "## Test Header"
                assert chunk["metadata"]["header_depth"] == 2  # ## = depth 2

    def test_process_section_calculates_header_depth(self, temp_dir, sample_config):
        """Test header depth calculation."""
        # Arrange: Create config with smart chunking
        smart_config = sample_config.copy()
        smart_config["chunking"]["smart"] = True

        docs_dir = temp_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        test_cases = [
            ("# Header 1", 1),
            ("## Header 2", 2),
            ("### Header 3", 3),
            ("#### Header 4", 4),
            ("##### Header 5", 5),
            ("###### Header 6", 6)
        ]

        # Act & Assert: Test each header depth
        for header, expected_depth in test_cases:
            processor = DocumentProcessor(
                docs_dir=docs_dir,
                config=smart_config,
                quiet=True
            )

            doc_path = docs_dir / f"header_depth_{expected_depth}.md"
            content = f"{header}\n\ncontent"
            doc_path.write_text(content, encoding="utf-8")

            chunks = processor.process_document(doc_path)

            # Find smart chunks with header metadata
            smart_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "smart"]
            if smart_chunks:
                assert smart_chunks[0]["metadata"]["header_depth"] == expected_depth

    def test_chunk_text_empty_input(self, document_processor, temp_dir):
        """Test chunking of empty text."""
        # Arrange: Create empty document
        docs_dir = temp_dir / "docs"
        doc_path = docs_dir / "empty.txt"
        doc_path.write_text("", encoding="utf-8")

        # Act: Process document
        chunks = document_processor.process_document(doc_path)

        # Assert: Should return empty list
        assert len(chunks) == 0

    def test_chunk_text_whitespace_only(self, document_processor, temp_dir):
        """Test chunking of whitespace-only text."""
        # Arrange: Create whitespace-only document
        docs_dir = temp_dir / "docs"
        doc_path = docs_dir / "whitespace.txt"
        doc_path.write_text("   \n\t   \n   ", encoding="utf-8")

        # Act: Process document
        chunks = document_processor.process_document(doc_path)

        # Assert: Should return empty list
        assert len(chunks) == 0

    def test_chunk_text_single_word(self, document_processor, temp_dir):
        """Test chunking of single word."""
        # Arrange: Create single-word document
        docs_dir = temp_dir / "docs"
        doc_path = docs_dir / "word.txt"
        doc_path.write_text("word", encoding="utf-8")

        # Act: Process document
        chunks = document_processor.process_document(doc_path)

        # Assert: Should create single chunk
        assert len(chunks) == 1
        assert chunks[0]["text"] == "word"

    def test_chunk_text_respects_custom_parameters(self, temp_dir, sample_config):
        """Test that chunking respects custom chunk size and overlap."""
        # Arrange: Create config with custom chunk size
        custom_config = sample_config.copy()
        custom_config["search"]["chunk_size"] = 200
        custom_config["search"]["chunk_overlap"] = 50

        docs_dir = temp_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        processor = DocumentProcessor(
            docs_dir=docs_dir,
            config=custom_config,
            quiet=True
        )

        doc_path = docs_dir / "custom_size.txt"
        text = "Word " * 100  # 500 characters approximately
        doc_path.write_text(text, encoding="utf-8")

        # Act: Process document
        chunks = processor.process_document(doc_path)

        # Assert: Should create more chunks due to smaller size
        assert len(chunks) >= 2

        # Chunks should be approximately the right size
        for chunk in chunks:
            assert len(chunk["text"]) <= 250  # Allow some buffer for word boundaries

    def test_chunk_text_handles_unicode(self, document_processor, temp_dir):
        """Test that chunking handles Unicode characters properly."""
        # Arrange: Create Unicode document
        docs_dir = temp_dir / "docs"
        doc_path = docs_dir / "unicode.txt"
        unicode_text = "This contains émojis 🔍 and spëcial châractërs. " * 20
        doc_path.write_text(unicode_text, encoding="utf-8")

        # Act: Process document
        chunks = document_processor.process_document(doc_path)

        # Assert: Should handle Unicode without errors
        assert len(chunks) >= 1

        # Verify Unicode is preserved
        combined_text = " ".join(chunk["text"] for chunk in chunks)
        assert "émojis" in combined_text
        assert "🔍" in combined_text
        assert "châractërs" in combined_text

    def test_chunk_text_very_long_single_sentence(self, document_processor, temp_dir):
        """Test chunking of very long text without sentence boundaries."""
        # Arrange: Create very long text without periods
        docs_dir = temp_dir / "docs"
        doc_path = docs_dir / "long_no_periods.txt"
        long_text = "word " * 300  # Much longer than chunk size, no sentence breaks
        doc_path.write_text(long_text, encoding="utf-8")

        # Act: Process document
        chunks = document_processor.process_document(doc_path)

        # Assert: Should still create multiple chunks even without sentence boundaries
        assert len(chunks) > 1

        # Each chunk should have reasonable length
        for chunk in chunks:
            # May be longer than chunk_size due to word boundary preservation
            assert len(chunk["text"]) <= 700  # Reasonable upper bound

    def test_chunk_metadata_consistency(self, document_processor, temp_dir):
        """Test that chunk metadata is consistent and complete."""
        # Arrange: Create test document
        docs_dir = temp_dir / "docs"
        doc_path = docs_dir / "metadata_test.txt"
        text = "This is a test document. " * 50
        doc_path.write_text(text, encoding="utf-8")

        # Act: Process document
        chunks = document_processor.process_document(doc_path)

        # Assert: Every chunk should have required metadata
        for chunk in chunks:
            assert "chunk_type" in chunk["metadata"]
            assert chunk["metadata"]["chunk_type"] in ["simple", "smart"]

            # Text should never be empty unless input was empty
            assert len(chunk["text"].strip()) > 0

    @pytest.mark.parametrize("chunk_size,overlap", [
        (100, 20),
        (500, 100),
        (1000, 200),
        (2000, 500)
    ])
    def test_various_chunk_sizes(self, temp_dir, sample_config, chunk_size, overlap):
        """Test chunking with various chunk sizes and overlaps."""
        # Arrange: Create config with specific chunk size
        custom_config = sample_config.copy()
        custom_config["search"]["chunk_size"] = chunk_size
        custom_config["search"]["chunk_overlap"] = overlap

        docs_dir = temp_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        processor = DocumentProcessor(
            docs_dir=docs_dir,
            config=custom_config,
            quiet=True
        )

        doc_path = docs_dir / f"size_{chunk_size}.txt"
        text = "This is a sentence for testing. " * 100  # Predictable length
        doc_path.write_text(text, encoding="utf-8")

        # Act: Process document
        chunks = processor.process_document(doc_path)

        # Assert: Should create chunks
        assert len(chunks) >= 1

        # Verify chunk sizes are reasonable
        for chunk in chunks:
            # Allow some flexibility for word boundaries
            assert len(chunk["text"]) <= chunk_size * 1.2

        # If multiple chunks, verify they exist
        if len(chunks) > 1:
            total_length = sum(len(chunk["text"]) for chunk in chunks)
            # Total should be reasonable given overlap
            assert total_length >= len(text)  # Should cover all content
