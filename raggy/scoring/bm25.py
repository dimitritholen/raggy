"""BM25 scoring implementation for keyword-based search."""

import math
from collections import Counter, defaultdict
from typing import Dict, List

from ..utils.patterns import WORD_PATTERN


class BM25Scorer:
    """Lightweight BM25 implementation for keyword scoring.

    BM25 is a probabilistic ranking function used for estimating the relevance
    of documents to a given search query.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        """Initialize BM25 scorer with tuning parameters.

        Args:
            k1: Controls term frequency saturation (default 1.2)
            b: Controls length normalization (default 0.75)

        """
        self.k1 = k1
        self.b = b
        self.doc_lengths: List[int] = []
        self.avg_doc_length = 0.0
        self.doc_count = 0
        self.term_frequencies: List[Dict[str, int]] = []
        self.idf_scores: Dict[str, float] = {}

    def fit(self, documents: List[str]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of document texts to index

        """
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.term_frequencies = []
        doc_term_counts: Dict[str, int] = defaultdict(int)

        # Calculate term frequencies and document lengths
        for doc in documents:
            terms = self._tokenize(doc)
            self.doc_lengths.append(len(terms))

            term_freq = Counter(terms)
            self.term_frequencies.append(term_freq)

            # Count documents containing each term
            for term in set(terms):
                doc_term_counts[term] += 1

        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths else 0.0
        )

        # Calculate IDF scores
        for term, doc_freq in doc_term_counts.items():
            # Use standard BM25 IDF: log((N + 1) / df)
            # This avoids negative scores and is more stable for small datasets
            self.idf_scores[term] = math.log((self.doc_count + 1) / doc_freq)

    def score(self, query: str, doc_index: int) -> float:
        """Calculate BM25 score for query against document.

        Args:
            query: Search query text
            doc_index: Index of document to score

        Returns:
            float: BM25 relevance score (non-negative)

        """
        if doc_index < 0 or doc_index >= len(self.term_frequencies):
            return 0.0

        query_terms = self._tokenize(query)
        score = 0.0
        doc_length = self.doc_lengths[doc_index]
        term_freq = self.term_frequencies[doc_index]

        for term in query_terms:
            if term in term_freq:
                tf = term_freq[term]
                idf = self.idf_scores.get(term, 0.0)

                numerator = tf * (self.k1 + 1)
                length_normalization = (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )
                denominator = tf + self.k1 * length_normalization
                score += idf * (numerator / denominator)

        return max(0.0, score)  # Ensure non-negative scores

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for text processing.

        Args:
            text: Text to tokenize

        Returns:
            List[str]: List of lowercase tokens

        """
        # Convert to lowercase and extract alphanumeric sequences using pre-compiled pattern
        return WORD_PATTERN.findall(text.lower())
