"""Tests for BM25Scorer functionality."""

import pytest
import math
from raggy import BM25Scorer


class TestBM25Scorer:
    """Test the BM25Scorer class."""
    
    def test_initialization(self):
        """Test BM25Scorer initialization with default parameters."""
        scorer = BM25Scorer()
        assert scorer.k1 == 1.2
        assert scorer.b == 0.75
        assert scorer.doc_count == 0
        assert scorer.avg_doc_length == 0
        assert len(scorer.doc_lengths) == 0
        assert len(scorer.term_frequencies) == 0
        assert len(scorer.idf_scores) == 0
    
    def test_initialization_custom_params(self):
        """Test BM25Scorer initialization with custom parameters."""
        scorer = BM25Scorer(k1=1.5, b=0.8)
        assert scorer.k1 == 1.5
        assert scorer.b == 0.8
    
    def test_tokenize(self):
        """Test the tokenization method."""
        scorer = BM25Scorer()
        
        # Test basic tokenization
        tokens = scorer._tokenize("The quick brown fox")
        assert tokens == ["the", "quick", "brown", "fox"]
        
        # Test with punctuation
        tokens = scorer._tokenize("Hello, world! How are you?")
        assert tokens == ["hello", "world", "how", "are", "you"]
        
        # Test with numbers and special characters
        tokens = scorer._tokenize("API-v1.2 test_function() $variable")
        assert tokens == ["api", "v1", "2", "test_function", "variable"]
        
        # Test empty string
        tokens = scorer._tokenize("")
        assert tokens == []
    
    def test_fit_simple_documents(self, bm25_sample_documents):
        """Test fitting BM25 with simple documents."""
        scorer = BM25Scorer()
        scorer.fit(bm25_sample_documents)
        
        # Check basic stats
        assert scorer.doc_count == len(bm25_sample_documents)
        assert len(scorer.doc_lengths) == len(bm25_sample_documents)
        assert len(scorer.term_frequencies) == len(bm25_sample_documents)
        assert scorer.avg_doc_length > 0
        
        # Check that IDF scores were calculated
        assert len(scorer.idf_scores) > 0
        
        # Verify some expected terms are present
        assert "the" in scorer.idf_scores
        assert "quick" in scorer.idf_scores
        assert "fox" in scorer.idf_scores
    
    def test_fit_calculates_correct_doc_lengths(self):
        """Test that document lengths are calculated correctly."""
        documents = [
            "one two three",  # 3 words
            "four five",      # 2 words  
            "six seven eight nine ten"  # 5 words
        ]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        assert scorer.doc_lengths == [3, 2, 5]
        assert scorer.avg_doc_length == (3 + 2 + 5) / 3
    
    def test_fit_calculates_idf_scores(self):
        """Test IDF score calculation."""
        documents = [
            "the quick brown fox",    # 'the' appears in doc 0
            "a quick brown dog",      # 'the' doesn't appear
            "the lazy dog sleeps"     # 'the' appears in doc 2
        ]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        # 'the' appears in 2 out of 3 documents
        # IDF = log((3 - 2 + 0.5) / (2 + 0.5)) = log(1.5/2.5) = log(0.6)
        expected_idf_the = math.log((3 - 2 + 0.5) / (2 + 0.5))
        assert abs(scorer.idf_scores["the"] - expected_idf_the) < 1e-6
        
        # 'quick' appears in 2 out of 3 documents  
        expected_idf_quick = math.log((3 - 2 + 0.5) / (2 + 0.5))
        assert abs(scorer.idf_scores["quick"] - expected_idf_quick) < 1e-6
        
        # 'fox' appears in 1 out of 3 documents
        expected_idf_fox = math.log((3 - 1 + 0.5) / (1 + 0.5))
        assert abs(scorer.idf_scores["fox"] - expected_idf_fox) < 1e-6
    
    def test_score_simple_query(self):
        """Test scoring a simple query against documents."""
        documents = [
            "the quick brown fox jumps",
            "the lazy dog sleeps", 
            "a fox runs quickly"
        ]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        # Score query "fox" against each document
        scores = [scorer.score("fox", i) for i in range(len(documents))]
        
        # Document 0 and 2 contain "fox", document 1 doesn't
        assert scores[0] > 0  # Contains "fox"
        assert scores[1] == 0  # Doesn't contain "fox"  
        assert scores[2] > 0  # Contains "fox"
    
    def test_score_multi_term_query(self):
        """Test scoring a multi-term query."""
        documents = [
            "machine learning algorithms",
            "natural language processing", 
            "machine learning techniques for natural language"
        ]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        # Query contains both terms in documents 0 and 2
        scores = [scorer.score("machine learning", i) for i in range(len(documents))]
        
        assert scores[0] > 0  # Contains both terms
        assert scores[1] == 0  # Contains neither term
        assert scores[2] > 0  # Contains both terms
        
        # Document 2 has both terms plus additional context, might score differently
        assert all(score >= 0 for score in scores)  # All scores non-negative
    
    def test_score_nonexistent_query_term(self):
        """Test scoring with query terms not in any document."""
        documents = ["cat dog bird", "fish whale shark"]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        # Query with term not in any document
        scores = [scorer.score("elephant", i) for i in range(len(documents))]
        
        # All scores should be 0
        assert all(score == 0 for score in scores)
    
    def test_score_invalid_document_index(self):
        """Test scoring with invalid document index."""
        documents = ["test document"]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        # Invalid document index should return 0
        assert scorer.score("test", 5) == 0  # Index out of range
        assert scorer.score("test", -1) == 0  # Negative index
    
    def test_score_relevance_ranking(self):
        """Test that BM25 scores rank documents by relevance correctly."""
        documents = [
            "machine learning is a subset of artificial intelligence",  # Relevant
            "cats and dogs are pets",  # Not relevant
            "machine learning algorithms use statistical methods",  # Very relevant  
            "learning to cook is fun"  # Somewhat relevant (contains 'learning')
        ]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        query = "machine learning algorithms"
        scores = [scorer.score(query, i) for i in range(len(documents))]
        
        # Document 2 should score highest (has all query terms)
        # Document 0 should score second (has 'machine learning')  
        # Document 1 should score lowest (no relevant terms)
        # Document 3 should score low (only has 'learning')
        
        assert scores[2] > scores[0]  # Most relevant doc scores highest
        assert scores[0] > scores[3]  # More relevant than partial match
        assert scores[3] > scores[1] or scores[3] == 0  # Partial match better than none
        assert scores[1] == 0  # Irrelevant document scores 0
    
    def test_score_frequency_affects_ranking(self):
        """Test that term frequency affects BM25 scores."""
        documents = [
            "machine learning",  # Term appears once each
            "machine machine learning learning machine",  # Terms appear multiple times
            "deep neural networks"  # Different terms
        ]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        query = "machine learning"
        scores = [scorer.score(query, i) for i in range(len(documents))]
        
        # Document 1 has higher term frequency, should score higher than document 0
        assert scores[1] > scores[0]
        assert scores[2] == 0  # No matching terms
    
    def test_empty_documents_list(self):
        """Test BM25 with empty documents list."""
        scorer = BM25Scorer()
        scorer.fit([])
        
        assert scorer.doc_count == 0
        assert scorer.avg_doc_length == 0
        assert len(scorer.idf_scores) == 0
    
    def test_single_document(self):
        """Test BM25 with a single document."""
        documents = ["single test document"]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        assert scorer.doc_count == 1
        assert len(scorer.doc_lengths) == 1
        
        # Score should work with single document
        score = scorer.score("test", 0)
        assert score > 0
    
    @pytest.mark.parametrize("k1,b", [
        (1.0, 0.5),
        (1.5, 1.0),
        (2.0, 0.0),
        (0.5, 0.75)
    ])
    def test_different_parameters(self, k1, b):
        """Test BM25 with different k1 and b parameters."""
        documents = ["test document for parameter testing"]
        
        scorer = BM25Scorer(k1=k1, b=b)
        scorer.fit(documents)
        
        score = scorer.score("test", 0)
        assert score >= 0  # Score should always be non-negative
        assert isinstance(score, (int, float))