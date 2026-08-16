"""Deterministic token normalization for indexing and querying."""

import re
from dataclasses import dataclass, field

from nltk.stem import PorterStemmer  # type: ignore[import-untyped]

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)

DEFAULT_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "more",
        "most",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "should",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


@dataclass(frozen=True, slots=True)
class AnalyzedToken:
    """A normalized term and its position before stop-word removal."""

    term: str
    position: int


@dataclass(slots=True)
class TextAnalyzer:
    """Apply case folding, stop-word removal, and Porter stemming."""

    stop_words: frozenset[str] = DEFAULT_STOP_WORDS
    min_token_length: int = 2
    _stemmer: PorterStemmer = field(default_factory=PorterStemmer, init=False)

    def analyze_with_positions(self, text: str) -> list[AnalyzedToken]:
        """Return searchable terms while retaining original token positions."""

        analyzed: list[AnalyzedToken] = []
        for position, match in enumerate(_TOKEN_PATTERN.finditer(text)):
            normalized = match.group(0).casefold()
            if len(normalized) < self.min_token_length or normalized in self.stop_words:
                continue
            term = self._stemmer.stem(normalized)
            if term:
                analyzed.append(AnalyzedToken(term=term, position=position))
        return analyzed

    def analyze(self, text: str) -> list[str]:
        """Return normalized terms without positional metadata."""

        return [token.term for token in self.analyze_with_positions(text)]
