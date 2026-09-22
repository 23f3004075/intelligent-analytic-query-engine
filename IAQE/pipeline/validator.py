import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class Check:
    passed: bool
    message: str = ""

@dataclass
class ValidationReport:
    checks: List[Check]
    pass_rate: float
    issues: List[str]

class ResultValidator:
    def validate(self, result: pd.DataFrame, intent) -> ValidationReport:
        checks = [
            self._check_not_empty(result, intent),
            self._check_expected_columns(result, intent),
            self._check_numeric_sanity(result, intent),
            self._check_aggregation_consistency(result, intent),
            self._check_top_n_count(result, intent),
        ]
        
        passed = sum(c.passed for c in checks)
        issues = [c.message for c in checks if not c.passed]
        
        return ValidationReport(
            checks=checks,
            pass_rate=passed / len(checks) if checks else 0.0,
            issues=issues
        )
    
    def _check_not_empty(self, df: pd.DataFrame, intent) -> Check:
        if len(df) == 0 and getattr(intent, "primary_operation", "") != "COUNT_ZERO":
            return Check(passed=False, message="Result is empty — possible filter overreach")
        return Check(passed=True)

    def _check_expected_columns(self, df: pd.DataFrame, intent) -> Check:
        if len(df.columns) == 0:
            return Check(passed=False, message="Result has no columns")
        return Check(passed=True)
    
    def _check_top_n_count(self, df: pd.DataFrame, intent) -> Check:
        n_val = getattr(intent, "n_value", None)
        if n_val and len(df) > n_val:
            return Check(passed=False, message=f"Expected ≤{n_val} rows, got {len(df)}")
        return Check(passed=True)
    
    def _check_numeric_sanity(self, df: pd.DataFrame, intent) -> Check:
        sec_ops = getattr(intent, "secondary_operations", [])
        if "PCT_CONTRIBUTION" in sec_ops:
            pct_cols = [c for c in df.columns if "%" in c or "pct" in c.lower()]
            for col in pct_cols:
                total = df[col].sum()
                if not (95 <= total <= 105):
                    return Check(passed=False, message=f"Column {col} sums to {total:.1f}%, expected ~100%")
        return Check(passed=True)

    def _check_aggregation_consistency(self, df: pd.DataFrame, intent) -> Check:
        pri_op = getattr(intent, "primary_operation", "")
        sec_ops = getattr(intent, "secondary_operations", [])
        if pri_op == "AGGREGATE":
            has_grouping = any("group" in str(op).lower() for op in sec_ops)
            if not has_grouping and len(df) > 1:
                return Check(
                    passed=False,
                    message=f"Global AGGREGATE expected 1 row, got {len(df)} rows without GROUP_BY"
                )
        return Check(passed=True)