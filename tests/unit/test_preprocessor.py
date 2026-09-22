"""
Unit tests for QueryPreprocessor — no LLM, no API keys required.
Tests normalization, spell checking with domain vocabulary, and complexity estimation.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from IAQE.pipeline.preprocessor import QueryPreprocessor, PreprocessedQuery


@pytest.fixture
def preprocessor():
    return QueryPreprocessor()


# ─── Normalization ─────────────────────────────────────────────────────

class TestNormalization:
    def test_lowercase(self, preprocessor):
        result = preprocessor.process("Total Sales In INDIA")
        assert result.normalized_q == result.normalized_q.lower()

    def test_strips_extra_whitespace(self, preprocessor):
        result = preprocessor.process("  total   sales   in  India  ")
        assert "  " not in result.normalized_q
        assert result.normalized_q == result.normalized_q.strip()

    def test_preserves_original(self, preprocessor):
        original = "Total Sales In INDIA For March"
        result = preprocessor.process(original)
        assert result.original == original

    def test_empty_query(self, preprocessor):
        result = preprocessor.process("")
        assert isinstance(result, PreprocessedQuery)
        assert result.original == ""


# ─── Domain Vocabulary Protection ──────────────────────────────────────

class TestSpellCheck:
    def test_domain_term_apac_not_corrupted(self, preprocessor):
        result = preprocessor.process("sales in APAC region")
        assert "apac" in result.normalized_q

    def test_domain_term_emea_not_corrupted(self, preprocessor):
        result = preprocessor.process("revenue in EMEA")
        assert "emea" in result.normalized_q

    def test_domain_term_yoy_not_corrupted(self, preprocessor):
        result = preprocessor.process("YoY growth in revenue")
        assert "yoy" in result.normalized_q

    def test_domain_term_subcategory_not_corrupted(self, preprocessor):
        result = preprocessor.process("sales by subcategory")
        assert "subcategory" in result.normalized_q

    def test_domain_term_aov_not_corrupted(self, preprocessor):
        result = preprocessor.process("average AOV by region")
        assert "aov" in result.normalized_q

    def test_domain_term_sql_not_corrupted(self, preprocessor):
        result = preprocessor.process("run DuckDB SQL query")
        assert "sql" in result.normalized_q


# ─── Complexity Estimation ─────────────────────────────────────────────

class TestComplexity:
    def test_simple_total_query(self, preprocessor):
        # "total" appears in both SIMPLE and MEDIUM lists; MEDIUM is checked first
        # so "total profit" → MEDIUM. A pure SIMPLE query has no MEDIUM keywords.
        result = preprocessor.process("list all orders")
        assert result.complexity == "SIMPLE"

    def test_medium_group_by(self, preprocessor):
        result = preprocessor.process("sales by region")
        assert result.complexity == "MEDIUM"

    def test_medium_filter(self, preprocessor):
        result = preprocessor.process("total revenue where country is India")
        assert result.complexity == "MEDIUM"

    def test_complex_comparison(self, preprocessor):
        result = preprocessor.process("compare sales across all regions")
        assert result.complexity == "COMPLEX"

    def test_complex_contribution(self, preprocessor):
        result = preprocessor.process("Sales contribution % by category")
        assert result.complexity == "COMPLEX"

    def test_complex_ranking(self, preprocessor):
        result = preprocessor.process("top 5 cities by profit ranking")
        assert result.complexity == "COMPLEX"

    def test_complex_per_customer(self, preprocessor):
        result = preprocessor.process("revenue per customer in each region")
        assert result.complexity == "COMPLEX"


# ─── Output Shape ──────────────────────────────────────────────────────

class TestOutputShape:
    def test_returns_preprocessed_query(self, preprocessor):
        result = preprocessor.process("test query")
        assert isinstance(result, PreprocessedQuery)
        assert hasattr(result, 'original')
        assert hasattr(result, 'normalized_q')
        assert hasattr(result, 'complexity')

    def test_complexity_is_valid_enum(self, preprocessor):
        result = preprocessor.process("average order value by region")
        assert result.complexity in ("SIMPLE", "MEDIUM", "COMPLEX")

    def test_all_nl_queries_produce_valid_output(self, preprocessor):
        """Run all 8 queries from the assignment through the preprocessor."""
        queries = [
            "Total sales in India for March",
            "Top 2 cities by profit",
            "Average order value by region",
            "Which region missed its target in Feb?",
            "Sales contribution % by category",
            "Top product in each region",
            "YoY growth in revenue",
            "Revenue of top 3 customers per region",
        ]
        for q in queries:
            result = preprocessor.process(q)
            assert result.original == q
            assert len(result.normalized_q) > 0
            assert result.complexity in ("SIMPLE", "MEDIUM", "COMPLEX")
