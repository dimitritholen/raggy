"""Search engine for semantic and hybrid search operations."""

import re
from typing import Any, Dict, List, Optional

from ..config.constants import DEFAULT_RESULTS
from ..query.processor import QueryProcessor
from ..scoring.bm25 import BM25Scorer
from ..scoring.normalization import (
    interpret_score,
    normalize_cosine_distance,
    normalize_hybrid_score,
)
from ..utils.logging import log_error
from .database import DatabaseManager


class SearchEngine:
    """Handles semantic search, hybrid search, and scoring operations."""

    def __init__(
        self,
        database_manager: DatabaseManager,
        query_processor: QueryProcessor,
        config: Dict[str, Any],
        quiet: bool = False
    ) -> None:
        """Initialize search engine.

        Args:
            database_manager: Database manager instance
            query_processor: Query processor instance
            config: Configuration dictionary
            quiet: If True, suppress output

        """
        self.database_manager = database_manager
        self.query_processor = query_processor
        self.config = config
        self.quiet = quiet
        self._bm25_scorer = None
        self._documents_cache = None

    def search(
        self,
        query: str,
        embedding_model: Any,
        n_results: int = DEFAULT_RESULTS,
        hybrid: bool = False,
        expand_query: bool = False,
        show_scores: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Search the vector database with enhanced capabilities.

        Args:
            query: Search query
            embedding_model: Model for generating embeddings (unused in current impl)
            n_results: Number of results to return
            hybrid: If True, use hybrid search (semantic + keyword)
            expand_query: If True, expand query with synonyms
            show_scores: If True, add highlighted text to results

        Returns:
            List[Dict[str, Any]]: Search results with scores and metadata

        """
        collection = self._get_collection()
        if collection is None:
            return []

        try:
            query_info, processed_query = self._process_query(query, expand_query)
            raw_results = self._execute_semantic_search(
                collection, processed_query, n_results, hybrid
            )
            formatted_results = self._format_results(
                raw_results, query, query_info, hybrid
            )
            return self._post_process_results(
                formatted_results, query, n_results, show_scores
            )

        except (ValueError, RuntimeError) as e:
            # Search operation errors (invalid query, ChromaDB errors)
            log_error("Search error", e, quiet=self.quiet)
            return []
        except (OSError, ConnectionError) as e:
            # Database connection/access errors
            log_error("Database access error during search", e, quiet=self.quiet)
            return []

    def _get_collection(self):
        """Get database collection with error handling.

        Returns:
            Collection instance or None if not available

        """
        try:
            return self.database_manager.get_collection()
        except (ValueError, RuntimeError, OSError) as e:
            # Database not initialized or collection doesn't exist
            log_error(
                "Database collection not found - run 'python raggy.py build' first",
                e,
                quiet=self.quiet,
            )
            return None

    def _process_query(self, query: str, expand_query: bool) -> tuple:
        """Process and optionally expand query.

        Args:
            query: Original search query
            expand_query: Whether to expand with synonyms

        Returns:
            tuple: (query_info dict, processed_query string)

        """
        if expand_query:
            query_info = self.query_processor.process(query)
            processed_query = query_info["processed"]
        else:
            query_info = {
                "original": query,
                "type": "keyword",
                "boost_exact": False,
            }
            processed_query = query

        return query_info, processed_query

    def _execute_semantic_search(
        self, collection, processed_query: str, n_results: int, hybrid: bool
    ) -> Dict[str, Any]:
        """Execute semantic search against collection.

        Args:
            collection: ChromaDB collection
            processed_query: Query text to search
            n_results: Number of results to retrieve
            hybrid: If True, get extra results for hybrid filtering

        Returns:
            Dict[str, Any]: Raw search results from ChromaDB

        """
        # Initialize BM25 scorer for hybrid search
        if hybrid and self._bm25_scorer is None:
            self._init_bm25_scorer(collection)

        # Get more results for hybrid mode to allow for diversity
        return collection.query(
            query_texts=[processed_query],
            n_results=(n_results * 2 if hybrid else n_results),
        )

    def _format_results(
        self,
        raw_results: Dict[str, Any],
        query: str,
        query_info: Dict[str, Any],
        hybrid: bool,
    ) -> List[Dict[str, Any]]:
        """Format raw results with scores and metadata.

        Args:
            raw_results: Raw results from ChromaDB
            query: Original query string
            query_info: Query processing metadata
            hybrid: Whether hybrid search is enabled

        Returns:
            List[Dict[str, Any]]: Formatted results with scores

        """
        formatted_results = []

        for i in range(len(raw_results["documents"][0])):
            result_dict = self._create_result_dict(
                raw_results, i, query, query_info, hybrid
            )
            formatted_results.append(result_dict)

        return formatted_results

    def _create_result_dict(
        self,
        raw_results: Dict[str, Any],
        index: int,
        query: str,
        query_info: Dict[str, Any],
        hybrid: bool,
    ) -> Dict[str, Any]:
        """Create formatted result dictionary for single result.

        Args:
            raw_results: Raw results from ChromaDB
            index: Index of current result
            query: Original query string
            query_info: Query processing metadata
            hybrid: Whether hybrid search is enabled

        Returns:
            Dict[str, Any]: Formatted result with scores

        """
        distance = raw_results["distances"][0][index] if "distances" in raw_results else None
        semantic_score = normalize_cosine_distance(distance) if distance is not None else 0

        # Calculate scores
        keyword_score, final_score = self._calculate_scores(
            semantic_score, query, index, hybrid
        )

        # Apply exact match boost if needed
        if query_info.get("boost_exact") and query.lower() in raw_results["documents"][0][index].lower():
            final_score = min(1.0, final_score * 1.5)

        return {
            "text": raw_results["documents"][0][index],
            "metadata": raw_results["metadatas"][0][index],
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "final_score": final_score,
            "score_interpretation": interpret_score(final_score),
            "distance": distance,  # Keep for backward compatibility
            "similarity": final_score,  # Keep for backward compatibility
        }

    def _calculate_scores(
        self, semantic_score: float, query: str, index: int, hybrid: bool
    ) -> tuple:
        """Calculate keyword and final scores.

        Args:
            semantic_score: Semantic similarity score
            query: Query string for keyword matching
            index: Result index for BM25 scoring
            hybrid: Whether to use hybrid scoring

        Returns:
            tuple: (keyword_score, final_score)

        """
        if hybrid and self._bm25_scorer:
            keyword_score = self._bm25_scorer.score(query, index)
            final_score = normalize_hybrid_score(
                semantic_score,
                keyword_score,
                self.config["search"]["hybrid_weight"],
            )
        else:
            keyword_score = 0
            final_score = semantic_score

        return keyword_score, final_score

    def _post_process_results(
        self,
        formatted_results: List[Dict[str, Any]],
        query: str,
        n_results: int,
        show_scores: Optional[bool],
    ) -> List[Dict[str, Any]]:
        """Apply post-processing: sorting, reranking, highlighting.

        Args:
            formatted_results: Formatted search results
            query: Original query
            n_results: Number of final results to return
            show_scores: Whether to add highlighted text

        Returns:
            List[Dict[str, Any]]: Post-processed results

        """
        # Sort by final score and limit
        formatted_results.sort(key=lambda x: x["final_score"], reverse=True)
        formatted_results = formatted_results[:n_results]

        # Rerank if enabled
        if self.config["search"]["rerank"]:
            formatted_results = self._rerank_results(query, formatted_results)

        # Add highlighting if requested
        should_highlight = (
            show_scores if show_scores is not None else self.config["search"]["show_scores"]
        )
        if should_highlight:
            for result in formatted_results:
                result["highlighted_text"] = self._highlight_matches(query, result["text"])

        return formatted_results

    def _init_bm25_scorer(self, collection):
        """Initialize BM25 scorer with collection documents.

        Args:
            collection: ChromaDB collection instance

        """
        if self._documents_cache is None:
            # Get all documents from collection
            all_data = collection.get()
            self._documents_cache = all_data["documents"]

        self._bm25_scorer = BM25Scorer()
        self._bm25_scorer.fit(self._documents_cache)

    def _rerank_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results to improve diversity and relevance.

        Args:
            query: Original query (unused in current impl)
            results: List of search results

        Returns:
            List[Dict[str, Any]]: Reranked results

        """
        if len(results) <= 2:
            return results

        reranked = []
        used_sources = set()

        # First pass: take best result from each source
        for result in results:
            source = result["metadata"]["source"]
            if source not in used_sources:
                reranked.append(result)
                used_sources.add(source)
                if len(reranked) >= len(results) // 2:
                    break

        # Second pass: add remaining results
        for result in results:
            if result not in reranked:
                reranked.append(result)

        return reranked[: len(results)]

    def _highlight_matches(
        self, query: str, text: str, context_chars: Optional[int] = None
    ) -> str:
        """Highlight matching terms in text.

        Args:
            query: Search query
            text: Text to highlight
            context_chars: Number of context characters to show

        Returns:
            str: Highlighted text excerpt

        """
        context_chars = context_chars or self.config["search"]["context_chars"]

        query_terms = re.findall(r"\b\w+\b", query.lower())
        match_pos = self._find_first_match(query_terms, text.lower())

        if match_pos == -1:
            return self._get_default_excerpt(text, context_chars)

        return self._extract_context_window(text, match_pos, context_chars)

    def _find_first_match(self, query_terms: list, text_lower: str) -> int:
        """Find position of first matching query term.

        Args:
            query_terms: List of query terms to search for
            text_lower: Lowercased text to search in

        Returns:
            int: Position of first match or -1 if no match found

        """
        for term in query_terms:
            pos = text_lower.find(term)
            if pos != -1:
                return pos
        return -1

    def _get_default_excerpt(self, text: str, context_chars: int) -> str:
        """Get default excerpt when no match found.

        Args:
            text: Full text
            context_chars: Maximum characters to return

        Returns:
            str: Excerpt from beginning of text

        """
        if len(text) > context_chars:
            return text[:context_chars] + "..."
        return text

    def _extract_context_window(self, text: str, match_pos: int, context_chars: int) -> str:
        """Extract context window around match position.

        Args:
            text: Full text
            match_pos: Position of match
            context_chars: Size of context window

        Returns:
            str: Excerpt with context around match

        """
        start = max(0, match_pos - context_chars // 2)
        end = min(len(text), match_pos + context_chars // 2)

        # Extend to word boundaries
        start = self._extend_to_word_boundary(text, start, direction='left')
        end = self._extend_to_word_boundary(text, end, direction='right')

        excerpt = text[start:end].strip()
        return self._add_ellipsis(excerpt, start, end, len(text))

    def _extend_to_word_boundary(self, text: str, pos: int, direction: str) -> int:
        """Extend position to nearest word boundary.

        Args:
            text: Full text
            pos: Current position
            direction: 'left' or 'right'

        Returns:
            int: Adjusted position at word boundary

        """
        if direction == 'left':
            while pos > 0 and text[pos] != " ":
                pos -= 1
        else:  # direction == 'right'
            while pos < len(text) and text[pos] != " ":
                pos += 1
        return pos

    def _add_ellipsis(self, excerpt: str, start: int, end: int, text_len: int) -> str:
        """Add ellipsis to excerpt if truncated.

        Args:
            excerpt: Extracted text excerpt
            start: Start position in original text
            end: End position in original text
            text_len: Length of original text

        Returns:
            str: Excerpt with appropriate ellipsis

        """
        if start > 0:
            excerpt = "..." + excerpt
        if end < text_len:
            excerpt = excerpt + "..."
        return excerpt
