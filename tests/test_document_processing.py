"""Tests for document processing functionality in raggy."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from raggy.core.document import DocumentProcessor


class TestDocumentProcessing:
    """Test document processing functionality."""

    @pytest.fixture
    def document_processor(self, sample_docs_dir, sample_config):
        """Create a DocumentProcessor instance for testing."""
        return DocumentProcessor(
            docs_dir=sample_docs_dir,
            config=sample_config,
            quiet=True
        )
    
    def test_find_documents_empty_directory(self, document_processor):
        """Test finding documents in empty directory."""
        # Arrange: docs_dir already created by fixture

        # Act: Find documents
        files = document_processor.find_documents()

        # Assert: Should return empty list
        assert len(files) == 0
        assert isinstance(files, list)
    
    def test_find_documents_creates_directory_if_missing(self, document_processor):
        """Test that find_documents creates docs directory if it doesn't exist."""
        # Arrange: Remove docs directory
        import shutil
        docs_dir = document_processor.docs_dir
        if docs_dir.exists():
            shutil.rmtree(docs_dir)

        # Act: Find documents (should create directory)
        files = document_processor.find_documents()

        # Assert: Should create directory and return empty list
        assert docs_dir.exists()
        assert len(files) == 0
        assert isinstance(files, list)
    
    def test_find_documents_supported_formats(self, document_processor, sample_documents):
        """Test finding all supported document formats."""
        # Arrange: sample_documents fixture creates .md and .txt files
        # Update document_processor to use sample_documents directory
        document_processor.docs_dir = sample_documents

        # Create a PDF placeholder
        pdf_file = sample_documents / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake pdf content for testing")

        # Act: Find documents
        files = document_processor.find_documents()

        # Assert: Should find all files
        file_extensions = {f.suffix.lower() for f in files}
        expected_extensions = {".md", ".txt"}  # PDF will be found but not processed in this test

        # Check that we find the expected types
        assert ".md" in file_extensions
        assert ".txt" in file_extensions
        # Note: PDF needs proper handling, so we test it separately
    
    def test_find_documents_nested_directories(self, document_processor, sample_documents):
        """Test finding documents in nested directories."""
        # Arrange: Update document_processor to use sample_documents directory
        document_processor.docs_dir = sample_documents

        # Create nested directory structure
        nested_dir = sample_documents / "subdir"
        nested_dir.mkdir()

        nested_file = nested_dir / "nested_doc.md"
        nested_file.write_text("# Nested Document\nContent in subdirectory.")

        # Act: Find documents
        files = document_processor.find_documents()

        # Assert: Should find files in subdirectories
        file_paths = [str(f.relative_to(sample_documents)) for f in files]
        assert any("subdir" in path for path in file_paths)
    
    def test_extract_text_from_md(self, document_processor, sample_documents, sample_md_content):
        """Test extracting text from markdown files."""
        # Arrange: Create markdown file
        md_file = sample_documents / "test.md"
        md_file.write_text(sample_md_content)

        # Act: Extract text
        extracted_text = document_processor._extract_text_from_md(md_file)

        # Assert: Should extract full content (strip to handle trailing newlines)
        assert extracted_text.strip() == sample_md_content.strip()
        assert "# Test Document" in extracted_text
        assert "## Features" in extracted_text
    
    def test_extract_text_from_txt(self, document_processor, sample_documents, sample_txt_content):
        """Test extracting text from plain text files."""
        # Arrange: Create text file
        txt_file = sample_documents / "test.txt"
        txt_file.write_text(sample_txt_content, encoding="utf-8")

        # Act: Extract text
        extracted_text = document_processor._extract_text_from_txt(txt_file)

        # Assert: Should extract full content (strip to handle trailing newlines)
        assert extracted_text.strip() == sample_txt_content.strip()
        assert "plain text document" in extracted_text
    
    def test_extract_text_from_txt_encoding_fallback(self, document_processor, sample_documents):
        """Test text extraction with encoding fallback."""
        # Arrange: Write file with latin-1 encoding
        txt_file = sample_documents / "latin1_test.txt"
        latin1_content = "Café résumé naïve"
        txt_file.write_text(latin1_content, encoding="latin-1")

        # Act: Extract text (should handle encoding gracefully)
        extracted_text = document_processor._extract_text_from_txt(txt_file)

        # Assert: Content should be extracted (may have encoding differences)
        assert len(extracted_text) > 0
    
    def test_extract_text_from_nonexistent_file(self, document_processor, temp_dir):
        """Test extracting text from non-existent file."""
        # Arrange: Create path to non-existent file
        nonexistent_file = temp_dir / "missing.md"

        # Act: Extract text
        result = document_processor._extract_text_from_md(nonexistent_file)

        # Assert: Should handle gracefully and return empty string
        assert result == ""
    
    def test_get_file_hash(self, document_processor, sample_documents):
        """Test file hash generation."""
        # Arrange: Create test file
        test_file = sample_documents / "hash_test.txt"
        content = "Test content for hashing"
        test_file.write_text(content, encoding="utf-8")

        # Act: Generate hashes
        hash1 = document_processor._get_file_hash(test_file)
        hash2 = document_processor._get_file_hash(test_file)

        # Assert: Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hash length (hex)

        # Arrange: Modify file
        test_file.write_text("Different content", encoding="utf-8")

        # Act: Generate new hash
        hash3 = document_processor._get_file_hash(test_file)

        # Assert: Different content should produce different hash
        assert hash1 != hash3
    
    def test_process_document_markdown(self, document_processor, sample_documents, sample_md_content):
        """Test processing a markdown document."""
        # Arrange: Update processor to use sample_documents directory
        document_processor.docs_dir = sample_documents
        md_file = sample_documents / "test_doc.md"

        # Act: Process document
        documents = document_processor.process_document(md_file)

        # Assert: Should create at least one document chunk
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
    
    def test_process_document_text_file(self, document_processor, sample_documents, sample_txt_content):
        """Test processing a plain text document."""
        # Arrange: Update processor to use sample_documents directory
        document_processor.docs_dir = sample_documents
        txt_file = sample_documents / "test_notes.txt"

        # Act: Process document
        documents = document_processor.process_document(txt_file)

        # Assert: Should create at least one document chunk
        assert len(documents) > 0

        # Verify content is extracted
        combined_text = " ".join(doc["text"] for doc in documents)
        assert "plain text document" in combined_text

        # Check file type metadata
        assert all(doc["metadata"]["file_type"] == ".txt" for doc in documents)
    
    def test_process_document_creates_proper_ids(self, document_processor, sample_documents):
        """Test that document processing creates proper unique IDs."""
        # Arrange: Update processor to use sample_documents directory
        document_processor.docs_dir = sample_documents
        md_file = sample_documents / "id_test.md"
        md_file.write_text("# Test\nContent for ID testing.\n\nMore content to ensure multiple chunks.")

        # Act: Process document
        documents = document_processor.process_document(md_file)

        # Assert: Should create proper IDs
        if len(documents) > 1:
            # IDs should be unique
            ids = [doc["id"] for doc in documents]
            assert len(ids) == len(set(ids))  # All unique

            # IDs should follow pattern: filename_hash_chunkindex
            for i, doc_id in enumerate(ids):
                assert doc_id.endswith(f"_{i}")
                assert "id_test" in doc_id  # Contains filename
                assert len(doc_id.split("_")) >= 3  # filename_hash_index pattern
    
    def test_process_document_chunk_indices(self, document_processor, sample_documents):
        """Test that chunk indices are assigned correctly."""
        # Arrange: Update processor to use sample_documents directory
        document_processor.docs_dir = sample_documents

        # Create document that will definitely create multiple chunks
        long_content = "This is a sentence for chunk testing. " * 50
        long_file = sample_documents / "long_doc.md"
        long_file.write_text(long_content)

        # Act: Process document
        documents = document_processor.process_document(long_file)

        # Assert: Check chunk indices
        if len(documents) > 1:
            for i, doc in enumerate(documents):
                assert doc["metadata"]["chunk_index"] == i
                assert doc["metadata"]["total_chunks"] == len(documents)
    
    def test_process_document_unsupported_format(self, document_processor, sample_documents):
        """Test processing unsupported file format."""
        # Arrange: Update processor to use sample_documents directory
        document_processor.docs_dir = sample_documents
        unsupported_file = sample_documents / "test.xyz"
        unsupported_file.write_text("Content in unsupported format")

        # Act: Process document
        documents = document_processor.process_document(unsupported_file)

        # Assert: Should return empty list for unsupported format
        assert len(documents) == 0
    
    def test_process_document_empty_file(self, document_processor, sample_documents):
        """Test processing empty file."""
        # Arrange: Update processor to use sample_documents directory
        document_processor.docs_dir = sample_documents
        empty_file = sample_documents / "empty.md"
        empty_file.write_text("")

        # Act: Process document
        documents = document_processor.process_document(empty_file)

        # Assert: Should return empty list for empty file
        assert len(documents) == 0
    
    def test_process_document_whitespace_only_file(self, document_processor, sample_documents):
        """Test processing file with only whitespace."""
        # Arrange: Update processor to use sample_documents directory
        document_processor.docs_dir = sample_documents
        whitespace_file = sample_documents / "whitespace.md"
        whitespace_file.write_text("   \n\t   \n   ")

        # Act: Process document
        documents = document_processor.process_document(whitespace_file)

        # Assert: Should return empty list for whitespace-only file
        assert len(documents) == 0
    
    def test_extract_text_from_pdf_success(self, document_processor, sample_documents):
        """Test successful PDF text extraction."""
        # Arrange: Mock pypdf components
        with patch('pypdf.PdfReader') as mock_pdf_reader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Extracted PDF content"

            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]

            mock_pdf_reader.return_value = mock_reader

            pdf_file = sample_documents / "test.pdf"
            pdf_file.write_bytes(b"%PDF-1.4 fake content")

            # Act: Extract text from PDF
            extracted_text = document_processor._extract_text_from_pdf(pdf_file)

            # Assert: Should extract PDF content
            assert extracted_text == "Extracted PDF content"
            mock_pdf_reader.assert_called_once()
    
    def test_extract_text_from_pdf_error(self, document_processor, sample_documents):
        """Test PDF text extraction with error."""
        # Arrange: Mock pypdf to raise ValueError (caught by _extract_text_template)
        with patch('pypdf.PdfReader', side_effect=ValueError("PDF parsing error")):
            pdf_file = sample_documents / "corrupt.pdf"
            pdf_file.write_bytes(b"corrupted pdf content")

            # Act: Extract text (should handle exception)
            extracted_text = document_processor._extract_text_from_pdf(pdf_file)

            # Assert: Should return empty string on error
            assert extracted_text == ""
    
    def test_extract_text_from_docx_success(self, document_processor, sample_documents):
        """Test successful DOCX text extraction."""
        # Arrange: Mock python-docx components
        with patch('docx.Document') as mock_document_class:
            mock_paragraph = MagicMock()
            mock_paragraph.text = "DOCX paragraph content"

            mock_document = MagicMock()
            mock_document.paragraphs = [mock_paragraph]
            mock_document.tables = []  # No tables for this test

            mock_document_class.return_value = mock_document

            docx_file = sample_documents / "test.docx"
            docx_file.write_bytes(b"fake docx content")

            # Act: Extract text from DOCX
            extracted_text = document_processor._extract_text_from_docx(docx_file)

            # Assert: Should extract DOCX content
            assert "DOCX paragraph content" in extracted_text
            mock_document_class.assert_called_once_with(docx_file)
    
    def test_extract_text_from_docx_with_tables(self, document_processor, sample_documents):
        """Test DOCX text extraction including tables."""
        # Arrange: Mock table structure
        with patch('docx.Document') as mock_document_class:
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

            # Act: Extract text from DOCX with tables
            extracted_text = document_processor._extract_text_from_docx(docx_file)

            # Assert: Should extract table content
            assert "Cell content | Cell content" in extracted_text
    
    def test_process_document_preserves_relative_path(self, document_processor, sample_documents):
        """Test that document processing preserves relative paths correctly."""
        # Arrange: Update processor to use sample_documents directory
        document_processor.docs_dir = sample_documents

        # Create nested structure
        nested_dir = sample_documents / "category" / "subcategory"
        nested_dir.mkdir(parents=True)

        nested_file = nested_dir / "nested_doc.md"
        nested_file.write_text("# Nested Document\nContent in nested directory.")

        # Act: Process document
        documents = document_processor.process_document(nested_file)

        # Assert: Should preserve relative path
        assert len(documents) > 0

        # Source should be relative path from docs directory
        expected_path = "category/subcategory/nested_doc.md"
        assert documents[0]["metadata"]["source"] == expected_path
    
    def test_process_document_file_hash_consistency(self, document_processor, sample_documents):
        """Test that file hash is consistent for same content."""
        # Arrange: Update processor to use sample_documents directory
        document_processor.docs_dir = sample_documents
        test_file = sample_documents / "hash_consistency.md"
        content = "# Consistent Content\nThis content should hash consistently."
        test_file.write_text(content)

        # Act: Process document twice
        documents1 = document_processor.process_document(test_file)
        documents2 = document_processor.process_document(test_file)

        # Assert: File hashes should be identical
        hash1 = documents1[0]["metadata"]["file_hash"]
        hash2 = documents2[0]["metadata"]["file_hash"]
        assert hash1 == hash2

        # Arrange: Modify file
        test_file.write_text(content + "\nAdditional content.")

        # Act: Process modified file
        documents3 = document_processor.process_document(test_file)

        # Assert: Hash should change
        hash3 = documents3[0]["metadata"]["file_hash"]
        assert hash1 != hash3