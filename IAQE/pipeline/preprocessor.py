from spellchecker import SpellChecker
from dataclasses import dataclass
from enum import Enum


class Complexity(Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


@dataclass
class PreprocessedQuery:
    original: str
    normalized_q: str
    complexity: str


class QueryPreprocessor:
    # Domain-specific terms to protect from spell correction
    DOMAIN_TERMS = {
        "apac", "emea", "na", "yoy", "q1", "q2", "q3", "q4",
        "revenue", "subcategory", "aov", "qty", "pct", "csv",
        "mumbai", "delhi", "bangalore", "berlin", "london", "paris",
        "duckdb", "sql", "pandas", "avg", "jan", "feb", "mar",
    }

    def __init__(self):
        self.spell = SpellChecker()
        self.spell.word_frequency.load_words(self.DOMAIN_TERMS)

    def process(self, query):
        normalized_q = self._normalize(query)
        corrected_q = self._spell_check(normalized_q)
        complexity = self._estimate_complexity(corrected_q)

        return PreprocessedQuery(
            original=query,
            normalized_q=corrected_q,
            complexity=complexity.value  # SIMPLE | MEDIUM | COMPLEX
        )

    def _normalize(self, query):
        return " ".join(str(query).strip().lower().split())

    def _spell_check(self, query):
        corrected_words = []
        for word in query.split():
            corrected = self.spell.correction(word)
            corrected_words.append(corrected if corrected is not None else word)
        return " ".join(corrected_words)

    def _estimate_complexity(self, q):
        complex_signals = [
            "compare", "compared to", "vs", "versus", "difference between",
            "top n", "top 10", "highest", "lowest", "rank", "ranking",
            "percentage", "share", "contribution", "trend", "over time",
            "month to month", "quarterly", "yearly", "by region and", "across",
            "within", "for each", "per customer", "per product",
            "group by", "pivot", "breakdown", "distribution"
        ]

        medium_signals = [
            "by", "where", "filter", "group", "sort", "last month",
            "this year", "between", "average", "sum", "count", "total",
            "monthly", "weekly", "daily", "segment", "category", "region",
            "product", "customer", "sales", "revenue"
        ]

        simple_signals = [
            "total", "sum", "count", "average", "list", "show", "give me",
            "how many", "what is", "what are", "latest", "today", "current"
        ]

        text = q.lower()

        if any(signal in text for signal in complex_signals):
            return Complexity.COMPLEX
        if any(signal in text for signal in medium_signals):
            return Complexity.MEDIUM
        return Complexity.SIMPLE