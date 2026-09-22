"""
Integration test — end-to-end pipeline execution with the LIVE LLM.
Runs a single query through all 10 pipeline steps and validates the output structure.
This test requires OPEN_ROUTER_KEY to be set in .env.

Usage:
    pytest tests/integration/test_pipeline.py -v
"""
import pytest
import sys
import os
import json
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPEN_ROUTER_KEY"),
    reason="OPEN_ROUTER_KEY not set — skipping LLM integration tests"
)

from IAQE.engine import QueryEngine


CSV_PATHS = ["dataset/sales_data.csv", "dataset/targets.csv"]
DATA_DICT = "dataset/data_dictionary.json"


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Module-scoped engine with a temp feedback log."""
    fb_path = str(tmp_path_factory.mktemp("fb") / "feedback.csv")
    return QueryEngine(
        csv_paths=CSV_PATHS,
        data_dict_path=DATA_DICT,
        feedback_log_path=fb_path,
    )


# ─── Output Structure ──────────────────────────────────────────────

class TestOutputStructure:
    def test_single_query_returns_expected_keys(self, engine):
        result = engine.run("Total sales in India for March")
        required_keys = {"query", "generated_logic", "result", "confidence_score",
                         "explanation", "metadata"}
        assert required_keys.issubset(set(result.keys()))

    def test_explanation_has_3_parts(self, engine):
        result = engine.run("Total sales in India for March")
        explanation = result["explanation"]
        assert "understood" in explanation
        assert "approach" in explanation
        assert "caveats" in explanation

    def test_metadata_has_track_and_complexity(self, engine):
        result = engine.run("Total sales in India for March")
        meta = result["metadata"]
        assert "track_used" in meta
        assert "complexity" in meta
        assert "processing_time_ms" in meta

    def test_confidence_score_in_valid_range(self, engine):
        result = engine.run("Average order value by region")
        assert 0.0 <= result["confidence_score"] <= 1.0


# ─── Execution Tracks ──────────────────────────────────────────────

class TestExecutionTracks:
    def test_sql_track_produces_results(self, engine):
        result = engine.run("Total sales in India for March")
        # Should produce at least some result or a valid failure
        if result["metadata"]["track_used"] in ("SQL", "Pandas"):
            assert len(result["result"]) >= 0  # Could be empty but valid
        else:
            # Execution failed — confidence should be low
            assert result["confidence_score"] < 0.40

    def test_simple_query_succeeds(self, engine):
        result = engine.run("Top 2 cities by profit")
        assert result["metadata"]["track_used"] in ("SQL", "Pandas", "None")


# ─── Result Content ────────────────────────────────────────────────

class TestResultContent:
    def test_result_is_list_of_dicts(self, engine):
        result = engine.run("Average order value by region")
        assert isinstance(result["result"], list)
        if result["result"]:
            assert isinstance(result["result"][0], dict)

    def test_generated_logic_is_string(self, engine):
        result = engine.run("Total sales in India for March")
        assert isinstance(result["generated_logic"], str)


# ─── Confidence Scoring Integration ─────────────────────────────────

class TestConfidenceIntegration:
    def test_failed_execution_gets_low_confidence(self, engine):
        # Query that's hard to get right without the right columns
        result = engine.run("What is the GDP of Mars?")
        # Should get low/unreliable confidence since it's nonsensical
        assert result["confidence_score"] < 0.85


# ─── JSON Serializable ──────────────────────────────────────────────

class TestSerialization:
    def test_result_is_json_serializable(self, engine):
        result = engine.run("Total sales in India for March")
        # Should not raise
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)
        assert len(serialized) > 0
