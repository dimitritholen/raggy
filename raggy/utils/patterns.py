"""Pre-compiled regex patterns for performance."""

import re

# Text processing patterns
WORD_PATTERN = re.compile(r"\b\w+\b")
NEGATIVE_TERM_PATTERN = re.compile(r"-\w+")
AND_TERM_PATTERN = re.compile(r"\w+(?=\s+AND)", re.IGNORECASE)
QUOTED_PHRASE_PATTERN = re.compile(r'"([^"]+)"')

# Document structure patterns
HEADER_PATTERN = re.compile(r"(^#{1,6}\s+.*$)", re.MULTILINE)
SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?\n]")