"""Tests for QueryProcessor functionality."""

import pytest
from raggy import QueryProcessor


class TestQueryProcessor:
    """Test the QueryProcessor class."""
    
    def test_initialization_default(self):
        """Test QueryProcessor initialization with default expansions."""
        processor = QueryProcessor()
        
        # Check default expansions are loaded
        assert "api" in processor.expansions
        assert "ml" in processor.expansions
        assert "ai" in processor.expansions
        assert "ui" in processor.expansions
        assert "ux" in processor.expansions
        
        # Verify expansion contents
        assert processor.expansions["api"] == ["api", "application programming interface"]
        assert processor.expansions["ml"] == ["ml", "machine learning"]
    
    def test_initialization_custom_expansions(self):
        """Test QueryProcessor initialization with custom expansions."""
        custom_expansions = {
            "db": ["db", "database"],
            "js": ["js", "javascript"]
        }
        
        processor = QueryProcessor(custom_expansions)
        
        # Should use custom expansions only
        assert processor.expansions == custom_expansions
        assert "api" not in processor.expansions  # Default not loaded
        assert "db" in processor.expansions
        assert "js" in processor.expansions
    
    def test_detect_type_keyword(self):
        """Test detection of keyword query type."""
        processor = QueryProcessor()
        
        assert processor._detect_type("machine learning") == "keyword"
        assert processor._detect_type("python programming") == "keyword"
        assert processor._detect_type("single") == "keyword"
    
    def test_detect_type_exact(self):
        """Test detection of exact phrase query type."""
        processor = QueryProcessor()
        
        assert processor._detect_type('"exact phrase"') == "exact"
        assert processor._detect_type('"machine learning"') == "exact"
        assert processor._detect_type('"single word"') == "exact"
    
    def test_detect_type_question(self):
        """Test detection of question query type."""
        processor = QueryProcessor()
        
        assert processor._detect_type("How does this work?") == "question"
        assert processor._detect_type("What is machine learning?") == "question"
        assert processor._detect_type("Why use Python?") == "question"
        assert processor._detect_type("When should I use this?") == "question"
        assert processor._detect_type("Where can I find docs?") == "question"
        assert processor._detect_type("Who created this?") == "question"
    
    def test_detect_type_boolean(self):
        """Test detection of boolean query type."""
        processor = QueryProcessor()
        
        assert processor._detect_type("machine learning AND algorithms") == "boolean"
        assert processor._detect_type("python OR javascript") == "boolean"
        assert processor._detect_type("api -deprecated") == "boolean"
        assert processor._detect_type("search -old") == "boolean"
    
    def test_expand_query_simple(self):
        """Test simple query expansion."""
        processor = QueryProcessor()
        
        # Test API expansion
        expanded = processor._expand_query("api documentation")
        assert "application programming interface" in expanded
        assert "api" in expanded
        
        # Test ML expansion  
        expanded = processor._expand_query("ml algorithms")
        assert "machine learning" in expanded
        assert "ml" in expanded
    
    def test_expand_query_multiple_terms(self):
        """Test query expansion with multiple expandable terms."""
        processor = QueryProcessor()
        
        expanded = processor._expand_query("api and ml")
        
        # Should expand both terms
        assert "application programming interface" in expanded
        assert "machine learning" in expanded
        assert "OR" in expanded  # Should use OR syntax
    
    def test_expand_query_no_expansion_needed(self):
        """Test query expansion when no terms need expanding."""
        processor = QueryProcessor()
        
        original = "python programming tutorial"
        expanded = processor._expand_query(original)
        
        # Should return same query (lowercased)
        assert expanded == original.lower()
    
    def test_extract_operators_negative_terms(self):
        """Test extraction of negative terms."""
        processor = QueryProcessor()
        
        must_have, must_not = processor._extract_operators("machine learning -deprecated -old")
        
        assert must_not == ["deprecated", "old"]
        assert must_have == []  # No AND terms in this example
    
    def test_extract_operators_and_terms(self):
        """Test extraction of AND terms."""
        processor = QueryProcessor()
        
        must_have, must_not = processor._extract_operators("machine AND learning AND algorithms")
        
        assert "machine" in must_have
        assert "learning" in must_have  
        assert must_not == []
    
    def test_extract_operators_mixed(self):
        """Test extraction of mixed boolean operators."""
        processor = QueryProcessor()
        
        must_have, must_not = processor._extract_operators("machine AND learning -deprecated")
        
        assert "machine" in must_have
        assert "deprecated" in must_not
    
    def test_process_keyword_query(self, query_processor_test_cases):
        """Test processing of keyword queries."""
        processor = QueryProcessor()
        
        result = processor.process("machine learning")
        
        assert result["original"] == "machine learning"
        assert result["type"] == "keyword"
        assert result["boost_exact"] is False
        assert "machine" in result["terms"]
        assert "learning" in result["terms"]
    
    def test_process_exact_phrase_query(self):
        """Test processing of exact phrase queries."""
        processor = QueryProcessor()
        
        result = processor.process('"machine learning"')
        
        assert result["original"] == '"machine learning"'
        assert result["type"] == "exact"
        assert result["boost_exact"] is True
        assert result["processed"] == "machine learning"
        assert result["terms"] == ["machine learning"]
    
    def test_process_question_query(self):
        """Test processing of question queries."""
        processor = QueryProcessor()
        
        result = processor.process("How does machine learning work?")
        
        assert result["type"] == "question"
        assert result["boost_exact"] is False
        assert "how" in result["terms"]
        assert "machine" in result["terms"]
        assert "learning" in result["terms"]
    
    def test_process_boolean_query_with_negation(self):
        """Test processing of boolean queries with negation."""
        processor = QueryProcessor()
        
        result = processor.process("machine learning -deep")
        
        assert result["type"] == "boolean"
        assert "deep" in result["must_not"]
        assert result["boost_exact"] is False
    
    def test_process_boolean_query_with_and(self):
        """Test processing of boolean queries with AND."""
        processor = QueryProcessor()
        
        result = processor.process("machine AND learning")
        
        assert result["type"] == "boolean"
        assert "machine" in result["must_have"]
        assert result["boost_exact"] is False
    
    def test_process_query_with_expansion(self):
        """Test processing with query expansion."""
        processor = QueryProcessor()
        
        result = processor.process("api documentation")
        
        # Should expand 'api' term
        assert "application programming interface" in result["processed"]
        assert result["original"] == "api documentation"
    
    def test_process_preserves_original_query(self):
        """Test that original query is preserved during processing."""
        processor = QueryProcessor()
        
        original = "API Development Guide"
        result = processor.process(original)
        
        assert result["original"] == original
        assert result["processed"].lower() != original.lower()  # Should be different due to expansion
    
    def test_process_empty_query(self):
        """Test processing of empty query."""
        processor = QueryProcessor()
        
        result = processor.process("")
        
        assert result["original"] == ""
        assert result["processed"] == ""
        assert result["type"] == "keyword"  # Default type
        assert result["terms"] == []
    
    def test_process_whitespace_only_query(self):
        """Test processing of whitespace-only query."""
        processor = QueryProcessor()
        
        result = processor.process("   \t\n   ")
        
        assert result["original"] == "   \t\n   "
        assert result["processed"] == ""
        assert result["terms"] == []
    
    def test_case_insensitive_expansion(self):
        """Test that query expansion is case insensitive."""
        processor = QueryProcessor()
        
        # Test uppercase
        result_upper = processor.process("API documentation")
        assert "application programming interface" in result_upper["processed"]
        
        # Test mixed case
        result_mixed = processor.process("Api Documentation")  
        assert "application programming interface" in result_mixed["processed"]
        
        # Test lowercase (already tested in other tests)
        result_lower = processor.process("api documentation")
        assert "application programming interface" in result_lower["processed"]
    
    def test_custom_expansions_work(self):
        """Test that custom expansions work correctly."""
        custom_expansions = {
            "db": ["db", "database", "data store"],
            "ui": ["ui", "user interface", "frontend"]
        }
        
        processor = QueryProcessor(custom_expansions)
        
        result = processor.process("db design")
        
        assert "database" in result["processed"]
        assert "data store" in result["processed"]
    
    def test_expansion_preserves_other_terms(self):
        """Test that expansion preserves non-expandable terms."""
        processor = QueryProcessor()
        
        result = processor.process("api server configuration")
        
        # 'api' should be expanded
        assert "application programming interface" in result["processed"]
        # Other terms should be preserved
        assert "server" in result["processed"]
        assert "configuration" in result["processed"]
    
    def test_multiple_exact_phrases_not_supported(self):
        """Test behavior with multiple quoted phrases (edge case)."""
        processor = QueryProcessor()
        
        # This is an edge case - typically only one quoted phrase expected
        result = processor.process('"first phrase" "second phrase"')
        
        # Should detect as exact type and process first phrase
        assert result["type"] == "exact"
        # Behavior may vary, but should handle gracefully
    
    def test_malformed_quotes_handling(self):
        """Test handling of malformed quote queries."""
        processor = QueryProcessor()
        
        # Unmatched quote
        result = processor.process('machine learning"')
        # Should not be detected as exact phrase
        assert result["type"] != "exact"
        
        # Empty quotes
        result = processor.process('""')
        # Should handle gracefully
        assert result["type"] == "exact"
    
    @pytest.mark.parametrize("query,expected_type", [
        ("simple query", "keyword"),
        ('"exact phrase"', "exact"), 
        ("How does this work?", "question"),
        ("term1 AND term2", "boolean"),
        ("term -exclude", "boolean"),
        ("What is API?", "question"),  # Question with expandable term
        ("", "keyword")  # Empty defaults to keyword
    ])
    def test_query_type_detection_parametrized(self, query, expected_type):
        """Parametrized test for query type detection."""
        processor = QueryProcessor()
        result = processor.process(query)
        assert result["type"] == expected_type