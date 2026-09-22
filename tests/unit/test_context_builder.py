"""
Unit tests for DataContextBuilder — loads actual dataset files,
validates DuckDB tables, schema digest, and glossary output.
No LLM calls needed.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from IAQE.pipeline.context_builder import DataContextBuilder

CSV_PATHS = ["dataset/sales_data.csv", "dataset/targets.csv"]
DATA_DICT = "dataset/data_dictionary.json"


@pytest.fixture(scope="module")
def builder():
    """Module-scoped: load once, share across tests."""
    return DataContextBuilder(csv_paths=CSV_PATHS, data_dict_path=DATA_DICT)


# ─── CSV Loading ────────────────────────────────────────────────────

class TestCSVLoading:
    def test_sales_data_has_15_columns(self, builder):
        df = builder.dataframes["sales_data"]
        assert len(df.columns) == 15

    def test_targets_has_3_columns(self, builder):
        df = builder.dataframes["targets"]
        assert len(df.columns) == 3

    def test_sales_data_has_10_rows(self, builder):
        df = builder.dataframes["sales_data"]
        assert len(df) == 10

    def test_targets_has_9_rows(self, builder):
        df = builder.dataframes["targets"]
        assert len(df) == 9

    def test_sales_data_key_columns_present(self, builder):
        df = builder.dataframes["sales_data"]
        expected = {"order_id", "region", "country", "product_category", "profit"}
        assert expected.issubset(set(df.columns))

    def test_targets_key_columns_present(self, builder):
        df = builder.dataframes["targets"]
        expected = {"region", "month", "target_revenue"}
        assert expected == set(df.columns)


# ─── DuckDB Tables ──────────────────────────────────────────────────

class TestDuckDB:
    def test_two_tables_registered(self, builder):
        tables = builder.conn.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        assert table_names == {"sales_data", "targets"}

    def test_sql_query_runs(self, builder):
        result = builder.conn.execute(
            "SELECT COUNT(*) FROM sales_data"
        ).fetchone()
        assert result[0] == 10

    def test_join_works(self, builder):
        result = builder.conn.execute("""
            SELECT s.region, t.target_revenue
            FROM sales_data s
            JOIN targets t ON s.region = t.region
            LIMIT 5
        """).fetchall()
        assert len(result) > 0


# ─── Schema Digest ──────────────────────────────────────────────────

class TestSchemaDigest:
    def test_digest_is_nonempty_string(self, builder):
        digest = builder.build_schema_digest()
        assert isinstance(digest, str)
        assert len(digest) > 100

    def test_digest_contains_table_names(self, builder):
        digest = builder.build_schema_digest()
        assert "sales_data" in digest
        assert "targets" in digest

    def test_digest_contains_column_types(self, builder):
        digest = builder.build_schema_digest()
        # Should contain at least one type indicator
        assert any(t in digest for t in ["BIGINT", "VARCHAR", "INTEGER", "DOUBLE"])

    def test_digest_contains_sample_values(self, builder):
        digest = builder.build_schema_digest()
        # Region column should show sample values
        assert any(r in digest for r in ["APAC", "EMEA", "NA"])


# ─── Glossary ───────────────────────────────────────────────────────

class TestGlossary:
    def test_glossary_is_nonempty(self, builder):
        glossary = builder.build_glossary()
        assert len(glossary) > 0

    def test_glossary_contains_revenue_formula(self, builder):
        glossary = builder.build_glossary()
        assert "revenue" in glossary.lower()
        assert "quantity" in glossary.lower()

    def test_glossary_contains_synonyms(self, builder):
        glossary = builder.build_glossary()
        # "sales" → revenue is a documented synonym
        assert "sales" in glossary.lower()

    def test_glossary_contains_dimensions(self, builder):
        glossary = builder.build_glossary()
        assert "dimensions" in glossary.lower() or "region" in glossary.lower()


# ─── Column Set ─────────────────────────────────────────────────────

class TestGetAllColumns:
    def test_returns_set(self, builder):
        cols = builder.get_all_columns()
        assert isinstance(cols, set)

    def test_contains_key_columns(self, builder):
        cols = builder.get_all_columns()
        expected = {"order_id", "region", "profit", "target_revenue"}
        assert expected.issubset(cols)

    def test_no_duplicates_conceptually(self, builder):
        cols = builder.get_all_columns()
        # "region" exists in both tables but should appear once in the set
        assert "region" in cols
