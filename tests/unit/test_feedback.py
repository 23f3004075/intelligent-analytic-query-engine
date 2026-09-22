
import pytest
import sys
import os
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from IAQE.pipeline.feedback import (
    FeedbackRetriever, FeedbackLogger, FeedbackEntry, RetrievedFeedback,
)


def _make_entry(query, logic="SELECT 1", feedback="pending", score=0.9, note=""):
    return FeedbackEntry(
        timestamp="2026-01-01T00:00:00",
        query=query,
        generated_logic=logic,
        result_preview="[]",
        confidence_score=score,
        track_used="SQL",
        intent_json='{}',
        feedback=feedback,
        feedback_note=note,
    )


@pytest.fixture
def tmp_log(tmp_path):
    return str(tmp_path / "feedback.csv")



class TestColdStart:
    def test_retriever_inactive_without_file(self, tmp_log):
        retriever = FeedbackRetriever(tmp_log)
        assert retriever.is_active is False

    def test_retriever_returns_empty_on_cold_start(self, tmp_log):
        retriever = FeedbackRetriever(tmp_log)
        results = retriever.retrieve("total sales")
        assert results == []



class TestLogger:
    def test_first_write_creates_file(self, tmp_log):
        logger = FeedbackLogger(tmp_log)
        entry = _make_entry("total sales in India")
        logger.log(entry)
        assert Path(tmp_log).exists()

    def test_writes_csv_with_header(self, tmp_log):
        logger = FeedbackLogger(tmp_log)
        logger.log(_make_entry("query 1"))
        df = pd.read_csv(tmp_log)
        assert "query" in df.columns
        assert "feedback" in df.columns
        assert len(df) == 1

    def test_appends_without_duplicating_header(self, tmp_log):
        logger = FeedbackLogger(tmp_log)
        logger.log(_make_entry("query 1"))
        logger.log(_make_entry("query 2"))
        df = pd.read_csv(tmp_log)
        assert len(df) == 2



class TestRetrieverWithData:
    def _seed_log(self, tmp_log):
        logger = FeedbackLogger(tmp_log)
        logger.log(_make_entry("total sales in India for March", feedback="correct"))
        logger.log(_make_entry("top 5 cities by profit", feedback="correct"))
        logger.log(_make_entry("revenue by region", feedback="incorrect", note="wrong column"))

    def test_retriever_activates_with_evaluated_data(self, tmp_log):
        self._seed_log(tmp_log)
        retriever = FeedbackRetriever(tmp_log)
        assert retriever.is_active is True

    def test_retrieves_similar_query(self, tmp_log):
        self._seed_log(tmp_log)
        retriever = FeedbackRetriever(tmp_log)
        results = retriever.retrieve("total sales in India")
        # Should find the similar "total sales in India for March"
        assert len(results) > 0
        assert any("India" in r.query for r in results)

    def test_retrieved_entries_have_was_correct(self, tmp_log):
        self._seed_log(tmp_log)
        retriever = FeedbackRetriever(tmp_log)
        results = retriever.retrieve("total sales")
        for r in results:
            assert isinstance(r.was_correct, bool)
            assert isinstance(r.score, float)

    def test_pending_entries_excluded(self, tmp_log):
        logger = FeedbackLogger(tmp_log)
        logger.log(_make_entry("some query", feedback="pending"))
        retriever = FeedbackRetriever(tmp_log)
        assert retriever.is_active is False

    def test_low_similarity_excluded(self, tmp_log):
        self._seed_log(tmp_log)
        retriever = FeedbackRetriever(tmp_log)
        results = retriever.retrieve("completely unrelated astronomy question")
        # Should get no matches above 0.6 threshold
        assert all(r.score > 0.6 for r in results)
