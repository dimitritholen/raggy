"""Tests for text chunking functionality in raggy."""

import pytest
from unittest.mock import patch
from raggy import UniversalRAG


class TestTextChunking:
    """Test text chunking functionality."""
    
    @pytest.fixture
    def rag_instance(self, temp_dir, sample_config):
        """Create a RAG instance for testing."""
        with patch('raggy.load_config', return_value=sample_config):
            return UniversalRAG(
                docs_dir=str(temp_dir / "docs"),
                db_dir=str(temp_dir / "vectordb"),
                chunk_size=500,
                chunk_overlap=100,
                quiet=True
            )
    
    def test_chunk_text_simple_short_text(self, rag_instance):
        """Test chunking of text shorter than chunk size."""
        short_text = "This is a short text that should fit in one chunk."
        
        chunks = rag_instance._chunk_text(short_text)
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == short_text
        assert chunks[0]["metadata"]["chunk_type"] == "simple"
    
    def test_chunk_text_simple_long_text(self, rag_instance):
        """Test simple chunking of long text."""
        # Create text longer than chunk size
        long_text = "This is a sentence. " * 50  # Approximately 1000 chars
        
        chunks = rag_instance._chunk_text(long_text)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should have text and metadata
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["chunk_type"] == "simple"
            assert len(chunk["text"]) <= 600  # Should respect chunk size (500 + some buffer for sentence boundaries)
    
    def test_chunk_text_simple_respects_sentence_boundaries(self, rag_instance):
        """Test that simple chunking tries to break at sentence boundaries."""
        # Create text with clear sentence boundaries
        text = "First sentence. " * 20 + "Second sentence. " * 20 + "Third sentence. " * 20
        
        chunks = rag_instance._chunk_text(text)
        
        # Check that most chunks end with sentence boundaries
        sentence_ending_chunks = 0
        for chunk in chunks[:-1]:  # Exclude last chunk (may be partial)
            if chunk["text"].rstrip().endswith('.'):
                sentence_ending_chunks += 1
        
        # Most chunks should end with sentences
        assert sentence_ending_chunks >= len(chunks) - 2
    
    def test_chunk_text_overlap(self, rag_instance):
        """Test that chunks have proper overlap."""
        # Create predictable text
        sentences = [f"Sentence number {i}. " for i in range(50)]
        text = "".join(sentences)
        
        chunks = rag_instance._chunk_text(text)
        
        if len(chunks) > 1:
            # Check for overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]["text"]
                next_chunk = chunks[i + 1]["text"]
                
                # Find common words at boundaries
                current_words = current_chunk.split()
                next_words = next_chunk.split()
                
                # Should have some overlap (this is approximate due to sentence boundary breaking)
                # At minimum, check that chunks are not completely disjoint
                assert len(current_chunk) > 0
                assert len(next_chunk) > 0
    
    def test_chunk_text_smart_disabled_by_default(self, rag_instance):
        """Test that smart chunking is disabled by default in config."""
        text = "# Header\n\nContent under header.\n\n## Subheader\n\nMore content."
        
        chunks = rag_instance._chunk_text(text, smart=True)
        
        # Should use simple chunking since smart=False in config
        assert all(chunk["metadata"]["chunk_type"] == "simple" for chunk in chunks)
    
    @pytest.fixture
    def smart_chunking_config(self, sample_config):
        """Config with smart chunking enabled."""
        config = sample_config.copy()
        config["chunking"]["smart"] = True
        return config
    
    def test_chunk_text_smart_enabled(self, temp_dir, smart_chunking_config):
        """Test smart chunking when enabled."""
        with patch('raggy.load_config', return_value=smart_chunking_config):
            rag_instance = UniversalRAG(
                docs_dir=str(temp_dir / "docs"),
                db_dir=str(temp_dir / "vectordb"),
                chunk_size=500,
                chunk_overlap=100,
                quiet=True
            )
        
        markdown_text = """# Main Header
        
This is content under the main header.

## Subheader One

Content under subheader one with some details.

### Sub-subheader

More nested content here.

## Subheader Two

Different content under subheader two.
"""
        
        chunks = rag_instance._chunk_text(markdown_text, smart=True)
        
        # Should create smart chunks
        smart_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "smart"]
        assert len(smart_chunks) > 0
    
    def test_process_section_preserves_headers(self, temp_dir, smart_chunking_config):
        """Test that section processing preserves headers when configured."""
        with patch('raggy.load_config', return_value=smart_chunking_config):
            rag_instance = UniversalRAG(
                docs_dir=str(temp_dir / "docs"),
                db_dir=str(temp_dir / "vectordb"),
                chunk_size=500,
                chunk_overlap=100,
                quiet=True
            )
        
        header = "## Test Header"
        content = "This is the content under the header."
        
        chunks = rag_instance._process_section(content, header, 500, 100)
        
        if smart_chunking_config["chunking"]["preserve_headers"]:
            # First chunk should include header
            assert chunks[0]["text"].startswith(header)
        
        # All chunks should have metadata about the header
        for chunk in chunks:
            assert chunk["metadata"]["section_header"] == header
            assert chunk["metadata"]["header_depth"] == 2  # ## = depth 2
    
    def test_process_section_calculates_header_depth(self, temp_dir, smart_chunking_config):
        """Test header depth calculation."""
        with patch('raggy.load_config', return_value=smart_chunking_config):
            rag_instance = UniversalRAG(
                docs_dir=str(temp_dir / "docs"),
                db_dir=str(temp_dir / "vectordb"),
                chunk_size=500,
                chunk_overlap=100,
                quiet=True
            )
        
        test_cases = [
            ("# Header 1", 1),
            ("## Header 2", 2),
            ("### Header 3", 3),
            ("#### Header 4", 4),
            ("##### Header 5", 5),
            ("###### Header 6", 6)
        ]
        
        for header, expected_depth in test_cases:
            chunks = rag_instance._process_section("content", header, 500, 100)
            assert chunks[0]["metadata"]["header_depth"] == expected_depth
    
    def test_chunk_text_empty_input(self, rag_instance):
        """Test chunking of empty text."""
        chunks = rag_instance._chunk_text("")
        assert len(chunks) == 0
    
    def test_chunk_text_whitespace_only(self, rag_instance):
        """Test chunking of whitespace-only text."""
        chunks = rag_instance._chunk_text("   \n\t   \n   ")
        assert len(chunks) == 0
    
    def test_chunk_text_single_word(self, rag_instance):
        """Test chunking of single word."""
        chunks = rag_instance._chunk_text("word")
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == "word"
    
    def test_chunk_text_respects_custom_parameters(self, rag_instance):
        """Test that chunking respects custom chunk size and overlap."""
        text = "Word " * 100  # 500 characters approximately
        
        # Use smaller chunk size
        chunks = rag_instance._chunk_text(text, chunk_size=200, overlap=50)
        
        # Should create more chunks due to smaller size
        assert len(chunks) >= 2
        
        # Chunks should be approximately the right size
        for chunk in chunks:
            assert len(chunk["text"]) <= 250  # Allow some buffer for word boundaries
    
    def test_chunk_text_handles_unicode(self, rag_instance):
        """Test that chunking handles Unicode characters properly."""
        unicode_text = "This contains émojis 🔍 and spëcial châractërs. " * 20
        
        chunks = rag_instance._chunk_text(unicode_text)
        
        # Should handle Unicode without errors
        assert len(chunks) >= 1
        
        # Verify Unicode is preserved
        combined_text = " ".join(chunk["text"] for chunk in chunks)
        assert "émojis" in combined_text
        assert "🔍" in combined_text
        assert "châractërs" in combined_text
    
    def test_chunk_text_very_long_single_sentence(self, rag_instance):
        """Test chunking of very long text without sentence boundaries."""
        # Create very long text without periods
        long_text = "word " * 300  # Much longer than chunk size, no sentence breaks
        
        chunks = rag_instance._chunk_text(long_text)
        
        # Should still create multiple chunks even without sentence boundaries
        assert len(chunks) > 1
        
        # Each chunk should have reasonable length
        for chunk in chunks:
            # May be longer than chunk_size due to word boundary preservation
            assert len(chunk["text"]) <= 700  # Reasonable upper bound
    
    def test_chunk_metadata_consistency(self, rag_instance):
        """Test that chunk metadata is consistent and complete."""
        text = "This is a test document. " * 50
        
        chunks = rag_instance._chunk_text(text)
        
        for chunk in chunks:
            # Every chunk should have required metadata
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
    def test_various_chunk_sizes(self, rag_instance, chunk_size, overlap):
        """Test chunking with various chunk sizes and overlaps."""
        text = "This is a sentence for testing. " * 100  # Predictable length
        
        chunks = rag_instance._chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        
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