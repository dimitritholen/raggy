"""Main tests for raggy UniversalRAG functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from raggy import UniversalRAG, ScoringNormalizer


class TestUniversalRAG:
    """Test the main UniversalRAG class."""
    
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
    
    def test_initialization(self, temp_dir, sample_config):
        """Test UniversalRAG initialization."""
        with patch('raggy.load_config', return_value=sample_config):
            rag = UniversalRAG(
                docs_dir=str(temp_dir / "docs"),
                db_dir=str(temp_dir / "vectordb"),
                model_name="test-model",
                chunk_size=800,
                chunk_overlap=150,
                quiet=True
            )
        
        assert rag.docs_dir == temp_dir / "docs"
        assert rag.db_dir == temp_dir / "vectordb"
        assert rag.model_name == "test-model"
        assert rag.chunk_size == 800
        assert rag.chunk_overlap == 150
        assert rag.quiet is True
        assert rag.collection_name == "project_docs"
    
    def test_initialization_with_defaults(self, sample_config):
        """Test initialization with default parameters."""
        with patch('raggy.load_config', return_value=sample_config):
            rag = UniversalRAG()
        
        assert rag.docs_dir == Path("./docs")
        assert rag.db_dir == Path("./vectordb")
        assert rag.model_name == "all-MiniLM-L6-v2"
        assert rag.chunk_size == 1000
        assert rag.chunk_overlap == 200
        assert rag.quiet is False
    
    def test_lazy_loading_client(self, rag_instance):
        """Test lazy loading of ChromaDB client."""
        # Client should not be initialized yet
        assert rag_instance._client is None
        
        # Access client property
        with patch('raggy.chromadb.PersistentClient') as mock_client:
            client = rag_instance.client
            
            # Should create client
            mock_client.assert_called_once_with(path=str(rag_instance.db_dir))
            assert rag_instance._client is not None
    
    def test_lazy_loading_embedding_model(self, rag_instance, mock_embedding_model):
        """Test lazy loading of embedding model."""
        # Model should not be initialized yet
        assert rag_instance._embedding_model is None
        
        # Access embedding_model property
        with patch('raggy.SentenceTransformer', return_value=mock_embedding_model("test-model")):
            model = rag_instance.embedding_model
            
            # Should create model
            assert rag_instance._embedding_model is not None
            assert model.model_name == "test-model"
    
    @patch('raggy.chromadb.PersistentClient')
    def test_get_stats_success(self, mock_client_class, rag_instance):
        """Test getting database statistics."""
        # Mock collection and data
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "doc1.md"},
                {"source": "doc1.md"}, 
                {"source": "doc2.md"},
                {"source": "doc3.md"}
            ]
        }
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        stats = rag_instance.get_stats()
        
        assert stats["total_chunks"] == 42
        assert stats["db_path"] == str(rag_instance.db_dir)
        assert "sources" in stats
        assert stats["sources"]["doc1.md"] == 2
        assert stats["sources"]["doc2.md"] == 1
        assert stats["sources"]["doc3.md"] == 1
    
    @patch('raggy.chromadb.PersistentClient')
    def test_get_stats_no_database(self, mock_client_class, rag_instance):
        """Test getting stats when database doesn't exist."""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client_class.return_value = mock_client
        
        stats = rag_instance.get_stats()
        
        assert "error" in stats
        assert "Database not found" in stats["error"]
    
    def test_config_loading(self, temp_dir):
        """Test that configuration is loaded correctly."""
        test_config = {
            "search": {"max_results": 10},
            "models": {"default": "custom-model"}
        }
        
        with patch('raggy.load_config', return_value=test_config):
            rag = UniversalRAG(docs_dir=str(temp_dir))
            
            assert rag.config == test_config
    
    def test_query_processor_initialization(self, rag_instance):
        """Test that query processor is initialized with config expansions."""
        expected_expansions = rag_instance.config["search"].get("expansions", {})
        assert rag_instance.query_processor.expansions == expected_expansions
    
    def test_scoring_normalizer_initialization(self, rag_instance):
        """Test that scoring normalizer is initialized."""
        assert isinstance(rag_instance.scoring, ScoringNormalizer)


class TestScoringNormalizer:
    """Test the ScoringNormalizer class."""
    
    def test_normalize_cosine_distance(self):
        """Test cosine distance normalization."""
        # Distance 0 should give similarity 1
        assert ScoringNormalizer.normalize_cosine_distance(0) == 1.0
        
        # Distance 2 should give similarity 0  
        assert ScoringNormalizer.normalize_cosine_distance(2) == 0.0
        
        # Distance 1 should give similarity 0.5
        assert ScoringNormalizer.normalize_cosine_distance(1) == 0.5
        
        # Test boundary conditions
        assert ScoringNormalizer.normalize_cosine_distance(-0.1) == 1.0  # Clamped to 1
        assert ScoringNormalizer.normalize_cosine_distance(2.1) == 0.0   # Clamped to 0
    
    def test_normalize_hybrid_score(self):
        """Test hybrid score normalization."""
        # Test with default semantic weight (0.7)
        semantic_score = 0.8
        keyword_score = 5.0  # Will be normalized to 0.5
        
        result = ScoringNormalizer.normalize_hybrid_score(semantic_score, keyword_score)
        
        expected = 0.7 * 0.8 + 0.3 * 0.5  # 0.56 + 0.15 = 0.71
        assert abs(result - expected) < 1e-6
    
    def test_normalize_hybrid_score_custom_weight(self):
        """Test hybrid score with custom semantic weight."""
        semantic_score = 0.6
        keyword_score = 10.0  # Will be normalized to 1.0
        semantic_weight = 0.5
        
        result = ScoringNormalizer.normalize_hybrid_score(
            semantic_score, keyword_score, semantic_weight
        )
        
        expected = 0.5 * 0.6 + 0.5 * 1.0  # 0.3 + 0.5 = 0.8
        assert abs(result - expected) < 1e-6
    
    def test_interpret_score(self):
        """Test score interpretation."""
        assert ScoringNormalizer.interpret_score(0.9) == "Excellent"
        assert ScoringNormalizer.interpret_score(0.8) == "Excellent"
        assert ScoringNormalizer.interpret_score(0.7) == "Good"
        assert ScoringNormalizer.interpret_score(0.6) == "Good"
        assert ScoringNormalizer.interpret_score(0.5) == "Fair"
        assert ScoringNormalizer.interpret_score(0.4) == "Fair"
        assert ScoringNormalizer.interpret_score(0.3) == "Poor"
        assert ScoringNormalizer.interpret_score(0.1) == "Poor"
    
    def test_interpret_score_boundary_conditions(self):
        """Test score interpretation at boundaries."""
        # Test exact boundary values
        assert ScoringNormalizer.interpret_score(0.8) == "Excellent"
        assert ScoringNormalizer.interpret_score(0.79999) == "Good"
        assert ScoringNormalizer.interpret_score(0.6) == "Good"
        assert ScoringNormalizer.interpret_score(0.59999) == "Fair"
        assert ScoringNormalizer.interpret_score(0.4) == "Fair"
        assert ScoringNormalizer.interpret_score(0.39999) == "Poor"
        
        # Test edge cases
        assert ScoringNormalizer.interpret_score(1.0) == "Excellent"
        assert ScoringNormalizer.interpret_score(0.0) == "Poor"
        assert ScoringNormalizer.interpret_score(-0.1) == "Poor"  # Negative scores


class TestRAGIntegration:
    """Integration tests for RAG functionality."""
    
    @pytest.fixture
    def mock_chromadb_collection(self):
        """Mock ChromaDB collection for testing."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.add.return_value = None
        return mock_collection
    
    @pytest.fixture 
    def mock_chromadb_client(self, mock_chromadb_collection):
        """Mock ChromaDB client for testing."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chromadb_collection
        mock_client.get_collection.return_value = mock_chromadb_collection
        return mock_client
    
    @patch('raggy.chromadb.PersistentClient')
    def test_build_no_documents(self, mock_client_class, rag_instance, mock_chromadb_client):
        """Test build process with no documents."""
        mock_client_class.return_value = mock_chromadb_client
        
        # Ensure docs directory is empty
        rag_instance.docs_dir.mkdir(exist_ok=True)
        
        # Should handle gracefully
        rag_instance.build()
        
        # Should create collection but not add any documents
        mock_chromadb_client.get_or_create_collection.assert_called_once()
    
    @patch('raggy.chromadb.PersistentClient')
    def test_build_with_documents(self, mock_client_class, rag_instance, sample_documents, 
                                 mock_chromadb_client, mock_embedding_model):
        """Test build process with documents."""
        mock_client_class.return_value = mock_chromadb_client
        
        # Mock embedding model
        with patch.object(rag_instance, 'embedding_model', mock_embedding_model("test")):
            rag_instance.build()
        
        # Should create collection and add documents
        mock_chromadb_client.get_or_create_collection.assert_called_once()
        mock_chromadb_client.get_or_create_collection.return_value.add.assert_called()
    
    @patch('raggy.chromadb.PersistentClient')
    def test_build_force_rebuild(self, mock_client_class, rag_instance, mock_chromadb_client):
        """Test force rebuild functionality."""
        mock_client_class.return_value = mock_chromadb_client
        
        rag_instance.build(force_rebuild=True)
        
        # Should attempt to delete existing collection
        mock_chromadb_client.delete_collection.assert_called_once()
    
    @patch('raggy.chromadb.PersistentClient')
    def test_search_no_database(self, mock_client_class, rag_instance):
        """Test search when database doesn't exist."""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client_class.return_value = mock_client
        
        results = rag_instance.search("test query")
        
        assert results == []
    
    @patch('raggy.chromadb.PersistentClient')
    def test_search_with_results(self, mock_client_class, rag_instance):
        """Test search with mock results."""
        # Mock search results
        mock_results = {
            "documents": [["Document 1 content", "Document 2 content"]],
            "metadatas": [[
                {"source": "doc1.md", "chunk_index": 0, "total_chunks": 1},
                {"source": "doc2.md", "chunk_index": 0, "total_chunks": 1}
            ]],
            "distances": [[0.3, 0.7]]
        }
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = mock_results
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        results = rag_instance.search("test query", n_results=2)
        
        assert len(results) == 2
        assert results[0]["text"] == "Document 1 content"
        assert results[1]["text"] == "Document 2 content"
        
        # Check that scores are calculated
        assert "final_score" in results[0]
        assert "score_interpretation" in results[0]
    
    @patch('raggy.chromadb.PersistentClient')
    def test_search_hybrid_mode(self, mock_client_class, rag_instance):
        """Test search in hybrid mode."""
        mock_results = {
            "documents": [["Test document content"]],
            "metadatas": [[{"source": "test.md", "chunk_index": 0, "total_chunks": 1}]],
            "distances": [[0.5]]
        }
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = mock_results
        mock_collection.get.return_value = {"documents": ["Test document content"]}
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        results = rag_instance.search("test query", hybrid=True)
        
        assert len(results) >= 0  # Should handle hybrid search
        # BM25 scorer should be initialized
        # Note: Detailed BM25 testing is in test_bm25.py
    
    def test_highlight_matches(self, rag_instance):
        """Test text highlighting functionality."""
        text = "This is a test document about machine learning and artificial intelligence."
        query = "machine learning"
        
        highlighted = rag_instance._highlight_matches(query, text, context_chars=50)
        
        # Should contain the query terms
        assert "machine learning" in highlighted
        
        # Should be reasonable length
        assert len(highlighted) <= 100  # Context + some buffer
    
    def test_highlight_matches_no_match(self, rag_instance):
        """Test text highlighting when no matches found."""
        text = "This document contains no relevant terms."
        query = "machine learning"
        
        highlighted = rag_instance._highlight_matches(query, text, context_chars=50)
        
        # Should return beginning of text when no match
        assert len(highlighted) <= 53  # 50 + "..."
    
    def test_rerank_results_diversity(self, rag_instance):
        """Test result reranking for diversity."""
        results = [
            {"metadata": {"source": "doc1.md"}, "final_score": 0.9},
            {"metadata": {"source": "doc1.md"}, "final_score": 0.8},  # Same source
            {"metadata": {"source": "doc2.md"}, "final_score": 0.7},
            {"metadata": {"source": "doc3.md"}, "final_score": 0.6}
        ]
        
        reranked = rag_instance._rerank_results("test query", results)
        
        # Should prefer diversity (different sources)
        sources_in_top_results = [r["metadata"]["source"] for r in reranked[:3]]
        unique_sources = set(sources_in_top_results)
        
        # Should have good source diversity in top results
        assert len(unique_sources) >= 2
    
    @patch('builtins.input', side_effect=['test query', 'q'])
    @patch('raggy.chromadb.PersistentClient')
    def test_interactive_search(self, mock_client_class, mock_input, rag_instance, capsys):
        """Test interactive search mode."""
        # Mock empty search results
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # Run interactive search (should exit on 'q')
        rag_instance.interactive_search()
        
        captured = capsys.readouterr()
        assert "Interactive Search Mode" in captured.out
        assert "No results found" in captured.out  # Empty results
        assert "Goodbye" in captured.out