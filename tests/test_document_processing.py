"""Tests for document processing functionality in raggy."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from raggy import UniversalRAG


class TestDocumentProcessing:
    """Test document processing functionality."""
    
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
    
    def test_find_documents_empty_directory(self, rag_instance, temp_dir):
        """Test finding documents in empty directory."""
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()
        
        files = rag_instance._find_documents()
        assert len(files) == 0
    
    def test_find_documents_creates_directory_if_missing(self, rag_instance, temp_dir):
        """Test that find_documents creates docs directory if it doesn't exist."""
        # Ensure docs directory doesn't exist
        docs_dir = Path(rag_instance.docs_dir)
        if docs_dir.exists():
            docs_dir.rmdir()
        
        files = rag_instance._find_documents()
        
        # Should create directory and return empty list
        assert docs_dir.exists()
        assert len(files) == 0
    
    def test_find_documents_supported_formats(self, rag_instance, sample_documents):
        """Test finding all supported document formats."""
        # sample_documents fixture creates .md and .txt files
        # Let's create additional formats
        
        # Create a PDF placeholder (we'll mock the content extraction)
        pdf_file = sample_documents / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake pdf content for testing")
        
        files = rag_instance._find_documents()
        
        # Should find all files
        file_extensions = {f.suffix.lower() for f in files}
        expected_extensions = {".md", ".txt"}  # PDF will be found but not processed in this test
        
        # Check that we find the expected types
        assert ".md" in file_extensions
        assert ".txt" in file_extensions
        # Note: PDF needs proper handling, so we test it separately
    
    def test_find_documents_nested_directories(self, rag_instance, sample_documents):
        """Test finding documents in nested directories."""
        # Create nested directory structure
        nested_dir = sample_documents / "subdir"
        nested_dir.mkdir()
        
        nested_file = nested_dir / "nested_doc.md"
        nested_file.write_text("# Nested Document\nContent in subdirectory.")
        
        files = rag_instance._find_documents()
        
        # Should find files in subdirectories
        file_paths = [str(f.relative_to(sample_documents)) for f in files]
        assert any("subdir" in path for path in file_paths)
    
    def test_extract_text_from_md(self, rag_instance, sample_documents, sample_md_content):
        """Test extracting text from markdown files."""
        md_file = sample_documents / "test.md"
        
        extracted_text = rag_instance._extract_text_from_md(md_file)
        
        assert extracted_text == sample_md_content
        assert "# Test Document" in extracted_text
        assert "## Features" in extracted_text
    
    def test_extract_text_from_txt(self, rag_instance, sample_documents, sample_txt_content):
        """Test extracting text from plain text files."""
        txt_file = sample_documents / "test.txt"
        txt_file.write_text(sample_txt_content, encoding="utf-8")
        
        extracted_text = rag_instance._extract_text_from_txt(txt_file)
        
        assert extracted_text == sample_txt_content
        assert "plain text document" in extracted_text
    
    def test_extract_text_from_txt_encoding_fallback(self, rag_instance, sample_documents):
        """Test text extraction with encoding fallback."""
        txt_file = sample_documents / "latin1_test.txt"
        
        # Write file with latin-1 encoding
        latin1_content = "Café résumé naïve"
        txt_file.write_text(latin1_content, encoding="latin-1")
        
        # Should handle encoding gracefully
        extracted_text = rag_instance._extract_text_from_txt(txt_file)
        
        # Content should be extracted (may have encoding differences)
        assert len(extracted_text) > 0
    
    def test_extract_text_from_nonexistent_file(self, rag_instance, temp_dir):
        """Test extracting text from non-existent file."""
        nonexistent_file = temp_dir / "missing.md"
        
        # Should handle gracefully and return empty string
        result = rag_instance._extract_text_from_md(nonexistent_file)
        assert result == ""
    
    def test_get_file_hash(self, rag_instance, sample_documents):
        """Test file hash generation."""
        test_file = sample_documents / "hash_test.txt"
        content = "Test content for hashing"
        test_file.write_text(content, encoding="utf-8")
        
        hash1 = rag_instance._get_file_hash(test_file)
        hash2 = rag_instance._get_file_hash(test_file)
        
        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length (will change to SHA256)
        
        # Different content should produce different hash
        test_file.write_text("Different content", encoding="utf-8")
        hash3 = rag_instance._get_file_hash(test_file)
        assert hash1 != hash3
    
    def test_process_document_markdown(self, rag_instance, sample_documents, sample_md_content):
        """Test processing a markdown document."""
        md_file = sample_documents / "test_doc.md"
        
        documents = rag_instance._process_document(md_file)
        
        # Should create at least one document chunk
        assert len(documents) > 0
        
        # Check document structure
        for doc in documents:
            assert "id" in doc
            assert "text" in doc
            assert "metadata" in doc
            
            # Check metadata
            metadata = doc["metadata"]
            assert metadata["source"] == "test_doc.md"
            assert metadata["file_type"] == ".md"
            assert "chunk_index" in metadata
            assert "total_chunks" in metadata
            assert "file_hash" in metadata
            
            # Text should not be empty
            assert len(doc["text"].strip()) > 0
    
    def test_process_document_text_file(self, rag_instance, sample_documents, sample_txt_content):
        """Test processing a plain text document.""" 
        txt_file = sample_documents / "test_notes.txt"
        
        documents = rag_instance._process_document(txt_file)
        
        assert len(documents) > 0
        
        # Verify content is extracted
        combined_text = " ".join(doc["text"] for doc in documents)
        assert "plain text document" in combined_text
        
        # Check file type metadata
        assert all(doc["metadata"]["file_type"] == ".txt" for doc in documents)
    
    def test_process_document_creates_proper_ids(self, rag_instance, sample_documents):
        """Test that document processing creates proper unique IDs."""
        md_file = sample_documents / "id_test.md"
        md_file.write_text("# Test\nContent for ID testing.\n\nMore content to ensure multiple chunks.")
        
        documents = rag_instance._process_document(md_file)
        
        if len(documents) > 1:
            # IDs should be unique
            ids = [doc["id"] for doc in documents]
            assert len(ids) == len(set(ids))  # All unique
            
            # IDs should follow pattern: filename_hash_chunkindex
            for i, doc_id in enumerate(ids):
                assert doc_id.endswith(f"_{i}")
                assert "id_test" in doc_id  # Contains filename
                assert len(doc_id.split("_")) >= 3  # filename_hash_index pattern
    
    def test_process_document_chunk_indices(self, rag_instance, sample_documents):
        """Test that chunk indices are assigned correctly."""
        # Create document that will definitely create multiple chunks
        long_content = "This is a sentence for chunk testing. " * 50
        long_file = sample_documents / "long_doc.md"
        long_file.write_text(long_content)
        
        documents = rag_instance._process_document(long_file)
        
        if len(documents) > 1:
            # Check chunk indices
            for i, doc in enumerate(documents):
                assert doc["metadata"]["chunk_index"] == i
                assert doc["metadata"]["total_chunks"] == len(documents)
    
    def test_process_document_unsupported_format(self, rag_instance, sample_documents):
        """Test processing unsupported file format."""
        unsupported_file = sample_documents / "test.xyz"
        unsupported_file.write_text("Content in unsupported format")
        
        documents = rag_instance._process_document(unsupported_file)
        
        # Should return empty list for unsupported format
        assert len(documents) == 0
    
    def test_process_document_empty_file(self, rag_instance, sample_documents):
        """Test processing empty file."""
        empty_file = sample_documents / "empty.md"
        empty_file.write_text("")
        
        documents = rag_instance._process_document(empty_file)
        
        # Should return empty list for empty file
        assert len(documents) == 0
    
    def test_process_document_whitespace_only_file(self, rag_instance, sample_documents):
        """Test processing file with only whitespace."""
        whitespace_file = sample_documents / "whitespace.md"
        whitespace_file.write_text("   \n\t   \n   ")
        
        documents = rag_instance._process_document(whitespace_file)
        
        # Should return empty list for whitespace-only file
        assert len(documents) == 0
    
    @patch('raggy.PyPDF2')
    def test_extract_text_from_pdf_success(self, mock_pypdf2, rag_instance, sample_documents):
        """Test successful PDF text extraction."""
        # Mock PyPDF2 components
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Extracted PDF content"
        
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        pdf_file = sample_documents / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")
        
        extracted_text = rag_instance._extract_text_from_pdf(pdf_file)
        
        assert extracted_text == "Extracted PDF content"
        mock_pypdf2.PdfReader.assert_called_once()
    
    @patch('raggy.PyPDF2')
    def test_extract_text_from_pdf_error(self, mock_pypdf2, rag_instance, sample_documents):
        """Test PDF text extraction with error."""
        # Mock PyPDF2 to raise exception
        mock_pypdf2.PdfReader.side_effect = Exception("PDF parsing error")
        
        pdf_file = sample_documents / "corrupt.pdf"
        pdf_file.write_bytes(b"corrupted pdf content")
        
        extracted_text = rag_instance._extract_text_from_pdf(pdf_file)
        
        # Should return empty string on error
        assert extracted_text == ""
    
    @patch('raggy.Document')
    def test_extract_text_from_docx_success(self, mock_document_class, rag_instance, sample_documents):
        """Test successful DOCX text extraction."""
        # Mock python-docx components
        mock_paragraph = MagicMock()
        mock_paragraph.text = "DOCX paragraph content"
        
        mock_document = MagicMock()
        mock_document.paragraphs = [mock_paragraph]
        mock_document.tables = []  # No tables for this test
        
        mock_document_class.return_value = mock_document
        
        docx_file = sample_documents / "test.docx"
        docx_file.write_bytes(b"fake docx content")
        
        extracted_text = rag_instance._extract_text_from_docx(docx_file)
        
        assert "DOCX paragraph content" in extracted_text
        mock_document_class.assert_called_once_with(docx_file)
    
    @patch('raggy.Document')
    def test_extract_text_from_docx_with_tables(self, mock_document_class, rag_instance, sample_documents):
        """Test DOCX text extraction including tables."""
        # Mock table structure
        mock_cell = MagicMock()
        mock_cell.text = "Cell content"
        
        mock_row = MagicMock()
        mock_row.cells = [mock_cell, mock_cell]
        
        mock_table = MagicMock()
        mock_table.rows = [mock_row]
        
        mock_document = MagicMock()
        mock_document.paragraphs = []
        mock_document.tables = [mock_table]
        
        mock_document_class.return_value = mock_document
        
        docx_file = sample_documents / "table_test.docx"
        docx_file.write_bytes(b"fake docx with tables")
        
        extracted_text = rag_instance._extract_text_from_docx(docx_file)
        
        assert "Cell content | Cell content" in extracted_text
    
    def test_process_document_preserves_relative_path(self, rag_instance, sample_documents):
        """Test that document processing preserves relative paths correctly."""
        # Create nested structure
        nested_dir = sample_documents / "category" / "subcategory"
        nested_dir.mkdir(parents=True)
        
        nested_file = nested_dir / "nested_doc.md"
        nested_file.write_text("# Nested Document\nContent in nested directory.")
        
        documents = rag_instance._process_document(nested_file)
        
        assert len(documents) > 0
        
        # Source should be relative path from docs directory
        expected_path = "category/subcategory/nested_doc.md"
        assert documents[0]["metadata"]["source"] == expected_path
    
    def test_process_document_file_hash_consistency(self, rag_instance, sample_documents):
        """Test that file hash is consistent for same content."""
        test_file = sample_documents / "hash_consistency.md"
        content = "# Consistent Content\nThis content should hash consistently."
        test_file.write_text(content)
        
        documents1 = rag_instance._process_document(test_file)
        documents2 = rag_instance._process_document(test_file)
        
        # File hashes should be identical
        hash1 = documents1[0]["metadata"]["file_hash"]
        hash2 = documents2[0]["metadata"]["file_hash"]
        assert hash1 == hash2
        
        # Modify file and verify hash changes
        test_file.write_text(content + "\nAdditional content.")
        documents3 = rag_instance._process_document(test_file)
        hash3 = documents3[0]["metadata"]["file_hash"]
        assert hash1 != hash3