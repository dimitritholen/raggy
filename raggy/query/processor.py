"""Query processing and expansion functionality."""

from typing import Any, Dict, List, Optional, Tuple

from ..utils.patterns import (
    AND_TERM_PATTERN,
    NEGATIVE_TERM_PATTERN,
    QUOTED_PHRASE_PATTERN,
    WORD_PATTERN,
)


class QueryProcessor:
    """Enhanced query processing with expansion and operators.

    Handles:
    - Query expansion with synonyms
    - Exact phrase matching (quoted strings)
    - Boolean operators (AND, OR, NOT)
    - Query type detection
    """

    def __init__(
        self, custom_expansions: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """Initialize query processor with optional custom expansions.

        Args:
            custom_expansions: Optional dictionary of term expansions

        """
        # Default expansions - can be overridden via config
        self.expansions = custom_expansions or {
            # Common technical terms
            "api": ["api", "application programming interface"],
            "ml": ["ml", "machine learning"],
            "ai": ["ai", "artificial intelligence"],
            "ui": ["ui", "user interface"],
            "ux": ["ux", "user experience"],
            # Can be extended via configuration file
        }

    def process(self, query: str) -> Dict[str, Any]:
        """Process query and return enhanced version with metadata.

        Args:
            query: Raw query string

        Returns:
            Dict containing:
                - processed: Enhanced query string
                - original: Original query
                - type: Query type (exact, question, boolean, keyword)
                - boost_exact: Whether to boost exact matches
                - must_have: List of required terms
                - must_not: List of excluded terms
                - terms: List of query terms

        """
        # Preserve original query exactly as provided
        original = query
        # Use cleaned version for processing
        cleaned = query.strip()

        # Detect query type
        query_type = self._detect_type(cleaned)

        # Handle exact phrase queries (quoted)
        if query_type == "exact":
            # Defensively check if pattern found valid quoted phrase
            matches = QUOTED_PHRASE_PATTERN.findall(cleaned)
            if matches:
                phrase = matches[0]
                return {
                    "processed": phrase,
                    "original": original,
                    "type": "exact",
                    "boost_exact": True,
                    "terms": [phrase],
                }
            # Handle empty quotes case
            elif '""' in cleaned:
                return {
                    "processed": "",
                    "original": original,
                    "type": "exact",
                    "boost_exact": True,
                    "terms": [],
                }
            # If no valid match found, fall back to keyword search
            query_type = "keyword"

        # Expand terms
        expanded = self._expand_query(cleaned)

        # Extract boolean operators
        must_have, must_not = self._extract_operators(expanded)

        return {
            "processed": expanded,
            "original": original,
            "type": query_type,
            "boost_exact": False,
            "must_have": must_have,
            "must_not": must_not,
            "terms": WORD_PATTERN.findall(expanded.lower()),
        }

    def _detect_type(self, query: str) -> str:
        """Detect query type from content.

        Args:
            query: Query string

        Returns:
            str: Query type (exact, question, boolean, or keyword)

        """
        # Check for valid quoted phrases (including empty quotes "")
        # Pattern matches non-empty quotes, but we also check for paired empty quotes
        if QUOTED_PHRASE_PATTERN.findall(query) or '""' in query:
            return "exact"

        question_words = ["how", "what", "why", "when", "where", "who"]
        if any(word in query.lower() for word in question_words):
            return "question"

        boolean_operators = [" AND ", " OR ", " -"]
        query_upper = query.upper()
        if any(op in query_upper or op.strip() in query for op in boolean_operators):
            return "boolean"

        return "keyword"

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms.

        Args:
            query: Query string

        Returns:
            str: Expanded query with OR clauses for synonyms

        """
        expanded = query.lower()
        for term, expansions in self.expansions.items():
            if term in expanded:
                # Add expansions as OR terms
                expansion_str = " OR ".join(expansions[1:])  # Skip the original term
                if expansion_str:
                    expanded = expanded.replace(term, f"({term} OR {expansion_str})")
        return expanded

    def _extract_operators(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract boolean operators from query.

        Args:
            query: Query string

        Returns:
            Tuple[List[str], List[str]]: (must_have_terms, must_not_terms)

        """
        must_have = []
        must_not = []

        # Extract negative terms (preceded by -)
        negative_terms = NEGATIVE_TERM_PATTERN.findall(query)
        for term in negative_terms:
            must_not.append(term[1:])  # Remove the -

        # Extract AND terms
        and_terms = AND_TERM_PATTERN.findall(query)
        must_have.extend(and_terms)

        return must_have, must_not
