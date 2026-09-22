
import pytest
import sys
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import duckdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from IAQE.pipeline.executor import SafeExecutor, ExecutionContext, ExecutionResult, SecurityError


@dataclass
class MockCode:
   
    sql: Optional[str] = None
    pandas: Optional[str] = None


@pytest.fixture
def executor():
    return SafeExecutor()


@pytest.fixture
def exec_context():
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE sales_data AS
        SELECT 1001 AS order_id, 'APAC' AS region, 'India' AS country, 100.0 AS profit
        UNION ALL SELECT 1002, 'EMEA', 'Germany', 150.0
        UNION ALL SELECT 1003, 'NA', 'USA', 200.0
    """)
    df = conn.execute("SELECT * FROM sales_data").df()
    return ExecutionContext(
        duckdb_conn=conn,
        dataframes={"sales_data_df": df},
    )



class TestSQLTrack:
    def test_valid_sql_succeeds(self, executor, exec_context):
        code = MockCode(sql="SELECT region, SUM(profit) AS total FROM sales_data GROUP BY region")
        result = executor.execute(code, exec_context)
        assert result.success is True
        assert result.track == "SQL"
        assert isinstance(result.result, pd.DataFrame)
        assert len(result.result) == 3

    def test_sql_select_star(self, executor, exec_context):
        code = MockCode(sql="SELECT * FROM sales_data")
        result = executor.execute(code, exec_context)
        assert result.success is True
        assert len(result.result) == 3



class TestPandasTrack:
    def test_valid_pandas_succeeds(self, executor, exec_context):
        code = MockCode(pandas="result = sales_data_df.groupby('region')['profit'].sum().reset_index()")
        result = executor.execute(code, exec_context)
        assert result.success is True
        assert result.track == "Pandas"
        assert isinstance(result.result, pd.DataFrame)

    def test_pandas_uses_safe_builtins(self, executor, exec_context):
        """len(), sum(), min(), max() etc. should work inside sandbox."""
        code = MockCode(pandas="result = pd.DataFrame({'count': [len(sales_data_df)]})")
        result = executor.execute(code, exec_context)
        assert result.success is True
        assert result.result.iloc[0]["count"] == 3


class TestDualTrack:
    def test_bad_sql_falls_back_to_pandas(self, executor, exec_context):
        code = MockCode(
            sql="SELECT * FROM nonexistent_table",
            pandas="result = sales_data_df.head(2)",
        )
        result = executor.execute(code, exec_context)
        assert result.success is True
        assert result.track == "Pandas"

    def test_both_tracks_fail(self, executor, exec_context):
        code = MockCode(
            sql="SELECT * FROM nonexistent_table",
            pandas="result = nonexistent_df.head()",
        )
        result = executor.execute(code, exec_context)
        assert result.success is False
        assert result.error is not None

    def test_no_code_at_all(self, executor, exec_context):
        code = MockCode(sql=None, pandas=None)
        result = executor.execute(code, exec_context)
        assert result.success is False


class TestSecurity:
    def test_banned_import_os(self, executor, exec_context):
        code = MockCode(pandas="import os; result = pd.DataFrame()")
        with pytest.raises(SecurityError):
            executor.execute(code, exec_context)

    def test_banned_subprocess(self, executor, exec_context):
        code = MockCode(pandas="subprocess.run(['ls']); result = pd.DataFrame()")
        with pytest.raises(SecurityError):
            executor.execute(code, exec_context)

    def test_banned_eval(self, executor, exec_context):
        code = MockCode(pandas="eval('1+1'); result = pd.DataFrame()")
        with pytest.raises(SecurityError):
            executor.execute(code, exec_context)

    def test_banned___import__(self, executor, exec_context):
        code = MockCode(pandas="__import__('os'); result = pd.DataFrame()")
        with pytest.raises(SecurityError):
            executor.execute(code, exec_context)

    def test_banned_open(self, executor, exec_context):
        code = MockCode(sql="open('/etc/passwd')")
        with pytest.raises(SecurityError):
            executor.execute(code, exec_context)


class TestResultShape:
    def test_result_has_correct_fields(self, executor, exec_context):
        code = MockCode(sql="SELECT 1 AS x")
        result = executor.execute(code, exec_context)
        assert isinstance(result, ExecutionResult)
        assert hasattr(result, 'success')
        assert hasattr(result, 'result')
        assert hasattr(result, 'track')
        assert hasattr(result, 'error')
