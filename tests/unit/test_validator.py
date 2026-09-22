"""
Unit tests for ResultValidator — pure DataFrame logic, no LLM needed.
Tests all 5 validation checks from SYSTEM_DESIGN.md Section 3.7.
"""
import pytest
import sys
import os
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from IAQE.pipeline.validator import ResultValidator, ValidationReport, Check


@dataclass
class MockIntent:
    """Lightweight mock matching QueryIntent's shape."""
    primary_operation: str = "AGGREGATE"
    secondary_operations: list = None
    entities: dict = None
    n_value: Optional[int] = None

    def __post_init__(self):
        if self.secondary_operations is None:
            self.secondary_operations = []
        if self.entities is None:
            self.entities = {}


@pytest.fixture
def validator():
    return ResultValidator()


# ─── Non-Empty Check ─────────────────────────────────────────────────

class TestNonEmptyCheck:
    def test_non_empty_result_passes(self, validator):
        df = pd.DataFrame({"x": [1, 2, 3]})
        report = validator.validate(df, MockIntent())
        assert report.pass_rate > 0

    def test_empty_result_fails(self, validator):
        df = pd.DataFrame({"x": []})
        report = validator.validate(df, MockIntent())
        assert any("empty" in issue.lower() for issue in report.issues)

    def test_empty_result_with_count_zero_intent_passes(self, validator):
        df = pd.DataFrame({"x": []})
        intent = MockIntent(primary_operation="COUNT_ZERO")
        report = validator.validate(df, intent)
        empty_issues = [i for i in report.issues if "empty" in i.lower()]
        assert len(empty_issues) == 0


# ─── Top-N Count Check ──────────────────────────────────────────────

class TestTopNCheck:
    def test_top_2_with_2_rows_passes(self, validator):
        df = pd.DataFrame({"city": ["Mumbai", "Delhi"], "profit": [100, 200]})
        intent = MockIntent(primary_operation="RANK", n_value=2)
        report = validator.validate(df, intent)
        top_n_issues = [i for i in report.issues if "Expected" in i and "rows" in i]
        assert len(top_n_issues) == 0

    def test_top_2_with_5_rows_fails(self, validator):
        df = pd.DataFrame({"city": list("abcde"), "profit": range(5)})
        intent = MockIntent(primary_operation="RANK", n_value=2)
        report = validator.validate(df, intent)
        assert any("Expected" in issue for issue in report.issues)

    def test_no_n_value_skips_check(self, validator):
        df = pd.DataFrame({"x": range(100)})
        intent = MockIntent(n_value=None)
        report = validator.validate(df, intent)
        top_n_issues = [i for i in report.issues if "Expected" in i and "rows" in i]
        assert len(top_n_issues) == 0


# ─── Percentage Sanity Check ────────────────────────────────────────

class TestNumericSanity:
    def test_pct_summing_to_100_passes(self, validator):
        df = pd.DataFrame({
            "region": ["APAC", "EMEA", "NA"],
            "pct_share": [40.0, 35.0, 25.0],
        })
        intent = MockIntent(
            primary_operation="AGGREGATE",
            secondary_operations=["GROUP_BY", "PCT_CONTRIBUTION"],
        )
        report = validator.validate(df, intent)
        pct_issues = [i for i in report.issues if "sums to" in i]
        assert len(pct_issues) == 0

    def test_pct_summing_to_200_fails(self, validator):
        df = pd.DataFrame({
            "region": ["APAC", "EMEA"],
            "pct_share": [100.0, 100.0],
        })
        intent = MockIntent(
            primary_operation="AGGREGATE",
            secondary_operations=["PCT_CONTRIBUTION"],
        )
        report = validator.validate(df, intent)
        assert any("sums to" in issue for issue in report.issues)

    def test_no_pct_operation_skips_check(self, validator):
        df = pd.DataFrame({"pct_share": [200.0, 300.0]})
        intent = MockIntent(secondary_operations=[])
        report = validator.validate(df, intent)
        pct_issues = [i for i in report.issues if "sums to" in i]
        assert len(pct_issues) == 0


# ─── Expected Columns Check ─────────────────────────────────────────

class TestExpectedColumns:
    def test_result_with_columns_passes(self, validator):
        df = pd.DataFrame({"region": ["APAC"], "revenue": [1000]})
        report = validator.validate(df, MockIntent())
        col_issues = [i for i in report.issues if "no columns" in i.lower()]
        assert len(col_issues) == 0


# ─── Aggregation Consistency Check ───────────────────────────────────

class TestAggregationConsistency:
    def test_global_aggregate_one_row_passes(self, validator):
        df = pd.DataFrame({"total_revenue": [5000]})
        intent = MockIntent(primary_operation="AGGREGATE", secondary_operations=[])
        report = validator.validate(df, intent)
        agg_issues = [i for i in report.issues if "AGGREGATE" in i]
        assert len(agg_issues) == 0

    def test_global_aggregate_multi_rows_without_group_warns(self, validator):
        df = pd.DataFrame({"revenue": [100, 200, 300]})
        intent = MockIntent(primary_operation="AGGREGATE", secondary_operations=[])
        report = validator.validate(df, intent)
        assert any("AGGREGATE" in issue for issue in report.issues)

    def test_grouped_aggregate_multi_rows_passes(self, validator):
        df = pd.DataFrame({"region": ["APAC", "EMEA"], "revenue": [100, 200]})
        intent = MockIntent(
            primary_operation="AGGREGATE",
            secondary_operations=["GROUP_BY"],
        )
        report = validator.validate(df, intent)
        agg_issues = [i for i in report.issues if "AGGREGATE" in i]
        assert len(agg_issues) == 0


# ─── Validation Report Shape ────────────────────────────────────────

class TestReportShape:
    def test_report_has_correct_fields(self, validator):
        df = pd.DataFrame({"x": [1]})
        report = validator.validate(df, MockIntent())
        assert isinstance(report, ValidationReport)
        assert isinstance(report.checks, list)
        assert isinstance(report.pass_rate, float)
        assert isinstance(report.issues, list)

    def test_all_5_checks_run(self, validator):
        df = pd.DataFrame({"x": [1]})
        report = validator.validate(df, MockIntent())
        assert len(report.checks) == 5

    def test_pass_rate_is_between_0_and_1(self, validator):
        df = pd.DataFrame({"x": [1]})
        report = validator.validate(df, MockIntent())
        assert 0.0 <= report.pass_rate <= 1.0
