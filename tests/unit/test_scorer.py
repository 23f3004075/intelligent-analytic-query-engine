"""
Unit tests for ConfidenceScorer — pure math, no LLM needed.
Tests weight computation, hard penalties, ambiguity discount, and feedback boost.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from IAQE.pipeline.scorer import ConfidenceScorer, ConfidenceSignal


@pytest.fixture
def scorer():
    return ConfidenceScorer()


def _make_signals(**overrides):
    """Helper to create a ConfidenceSignal with sensible defaults."""
    defaults = dict(
        schema_match=1.0,
        code_validity=1.0,
        execution_success=1.0,
        result_validation=1.0,
        llm_self_score=0.9,
    )
    defaults.update(overrides)
    return ConfidenceSignal(**defaults)


# ─── Perfect Scores ───────────────────────────────────────────────────

class TestPerfectScore:
    def test_all_signals_perfect(self, scorer):
        signals = _make_signals(llm_self_score=1.0)
        score = scorer.score(signals)
        # Capped at 0.97 by feedback_boost logic (min(raw + 0, 0.97))
        assert score == 0.97

    def test_perfect_with_feedback_boost_caps_at_097(self, scorer):
        signals = _make_signals(llm_self_score=1.0)
        score = scorer.score(signals, feedback_boost=0.05)
        assert score <= 0.97

    def test_typical_high_confidence(self, scorer):
        signals = _make_signals(llm_self_score=0.9)
        score = scorer.score(signals)
        assert score >= 0.85  # Should be in "High" band


# ─── Weight Distribution ──────────────────────────────────────────────

class TestWeights:
    def test_weights_sum_to_one(self, scorer):
        assert sum(scorer.WEIGHTS) == pytest.approx(1.0)

    def test_schema_match_has_025_weight(self, scorer):
        """Verify schema_match weight is 0.25 per SYSTEM_DESIGN.md."""
        assert scorer.WEIGHTS[0] == 0.25

    def test_execution_success_has_025_weight(self, scorer):
        assert scorer.WEIGHTS[2] == 0.25

    def test_llm_self_score_has_010_weight(self, scorer):
        assert scorer.WEIGHTS[4] == 0.10


# ─── Hard Penalties ───────────────────────────────────────────────────

class TestHardPenalties:
    def test_execution_failure_caps_at_015(self, scorer):
        """If execution failed, score MUST be <= 0.15."""
        signals = _make_signals(execution_success=0.0)
        score = scorer.score(signals)
        assert score <= 0.15

    def test_low_schema_match_caps_at_030(self, scorer):
        """If schema match < 0.5, score capped at 0.30."""
        signals = _make_signals(schema_match=0.3)
        score = scorer.score(signals)
        assert score <= 0.30

    def test_both_penalties_apply_strictest(self, scorer):
        """Both penalties: execution fail + low schema → should be <= 0.15."""
        signals = _make_signals(execution_success=0.0, schema_match=0.2)
        score = scorer.score(signals)
        assert score <= 0.15


# ─── Ambiguity Discount ──────────────────────────────────────────────

class TestAmbiguityDiscount:
    def test_no_ambiguities_no_discount(self, scorer):
        signals = _make_signals()
        score_no_amb = scorer.score(signals, ambiguities_count=0)
        score_few_amb = scorer.score(signals, ambiguities_count=2)
        assert score_no_amb == score_few_amb  # <=2 does NOT trigger discount

    def test_high_ambiguities_apply_085_multiplier(self, scorer):
        signals = _make_signals()
        score_normal = scorer.score(signals, ambiguities_count=0)
        score_ambig = scorer.score(signals, ambiguities_count=3)
        assert score_ambig < score_normal


# ─── Feedback Boost ──────────────────────────────────────────────────

class TestFeedbackBoost:
    def test_positive_boost_increases_score(self, scorer):
        # Use lower signals so the base score is well below the 0.97 cap
        signals = _make_signals(llm_self_score=0.5, result_validation=0.5)
        base = scorer.score(signals, feedback_boost=0.0)
        boosted = scorer.score(signals, feedback_boost=0.05)
        assert boosted > base

    def test_boost_capped_at_097(self, scorer):
        signals = _make_signals(llm_self_score=1.0)
        score = scorer.score(signals, feedback_boost=0.10)
        assert score <= 0.97


# ─── Pandas Fallback Score ───────────────────────────────────────────

class TestPandasFallback:
    def test_pandas_gets_07_execution_score(self, scorer):
        sql_signals = _make_signals(execution_success=1.0)
        pandas_signals = _make_signals(execution_success=0.7)
        sql_score = scorer.score(sql_signals)
        pandas_score = scorer.score(pandas_signals)
        assert sql_score > pandas_score  # SQL track > Pandas track


# ─── Edge Cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_zeros(self, scorer):
        signals = _make_signals(
            schema_match=0.0, code_validity=0.0,
            execution_success=0.0, result_validation=0.0,
            llm_self_score=0.0,
        )
        score = scorer.score(signals)
        assert score >= 0.0
        assert score <= 0.15  # Execution failure penalty

    def test_score_is_rounded_to_3_decimals(self, scorer):
        signals = _make_signals(llm_self_score=0.777)
        score = scorer.score(signals)
        assert score == round(score, 3)
