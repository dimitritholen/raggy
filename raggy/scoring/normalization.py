"""Score normalization functions for search results."""

from typing import Optional


def normalize_cosine_distance(distance: float) -> float:
    """Normalize cosine distance (0-2 range) to similarity score (0-1 range).

    Args:
        distance: Cosine distance value (0-2 range, where 0 is identical)

    Returns:
        float: Normalized score (0-1 range, where 1 is perfect match)

    """
    # Convert cosine distance (0-2) to similarity (0-1)
    # Distance of 0 = similarity of 1 (identical)
    # Distance of 2 = similarity of 0 (opposite)
    return max(0.0, min(1.0, 1.0 - (distance / 2.0)))


def normalize_hybrid_score(
    semantic_score: float,
    keyword_score: float,
    weight: float = 0.7,
    semantic_boost: Optional[float] = None
) -> float:
    """Combine and normalize semantic and keyword scores.

    Args:
        semantic_score: Normalized semantic similarity score (0-1)
        keyword_score: BM25 keyword score (unbounded)
        weight: Weight for semantic score (0-1), remainder goes to keyword
        semantic_boost: Optional boost factor for high semantic scores

    Returns:
        float: Combined normalized score (0-1)

    """
    # Normalize BM25 score to 0-1 range (sigmoid-like transformation)
    # BM25 scores typically range from 0-20, we'll use a soft cap at 10
    normalized_keyword = min(1.0, keyword_score / 10.0)

    # Apply semantic boost if specified and semantic score is high
    if semantic_boost and semantic_score > 0.8:
        semantic_score = min(1.0, semantic_score * semantic_boost)

    # Weighted combination
    combined = (weight * semantic_score) + ((1 - weight) * normalized_keyword)

    return min(1.0, combined)  # Ensure max score is 1.0


def interpret_score(score: float) -> str:
    """Convert normalized score to human-readable interpretation.

    Args:
        score: Normalized score (0-1 range)

    Returns:
        str: Human-readable score interpretation

    """
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.5:
        return "Fair"
    elif score >= 0.3:
        return "Weak"
    else:
        return "Poor"
