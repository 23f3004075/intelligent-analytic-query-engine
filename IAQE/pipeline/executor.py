import concurrent.futures
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

class SecurityError(Exception):
    pass

@dataclass
class ExecutionContext:
    duckdb_conn: Any
    dataframes: Dict[str, pd.DataFrame]

@dataclass
class ExecutionResult:
    success: bool
    result: Optional[pd.DataFrame] = None
    track: Optional[str] = None
    error: Optional[str] = None

class SafeExecutor:
    TIMEOUT_SECONDS = 30
    MAX_RESULT_ROWS = 10_000
    BANNED_TOKENS = ["import os", "subprocess", "eval(", "exec(", "__import__", "open(", "sys."]

    # Safe builtins for Pandas code execution (empty {} blocks len, min, sum, etc.)
    SAFE_BUILTINS = {
        'abs': abs, 'len': len, 'min': min, 'max': max, 'sum': sum,
        'round': round, 'range': range, 'enumerate': enumerate,
        'zip': zip, 'sorted': sorted, 'reversed': reversed,
        'str': str, 'int': int, 'float': float, 'bool': bool,
        'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
        'True': True, 'False': False, 'None': None,
        'isinstance': isinstance, 'type': type, 'print': print,
    }

    def execute(self, code, context: ExecutionContext) -> ExecutionResult:
        self._static_check(code)

        if code.sql:
            try:
                result = self._run_sql(code.sql, context.duckdb_conn)
                return ExecutionResult(result=result, track="SQL", success=True)
            except Exception:
                pass

        if code.pandas:
            try:
                result = self._run_pandas(code.pandas, context.dataframes)
                return ExecutionResult(result=result, track="Pandas", success=True)
            except Exception as e:
                return ExecutionResult(success=False, error=str(e))

        return ExecutionResult(success=False, error="Both tracks failed")

    def _run_sql(self, sql: str, conn) -> pd.DataFrame:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(conn.execute, sql)
            result = future.result(timeout=self.TIMEOUT_SECONDS)

        df = result.df()
        if len(df) > self.MAX_RESULT_ROWS:
            df = df.head(self.MAX_RESULT_ROWS)
        return df

    def _run_pandas(self, code: str, dataframes: dict) -> pd.DataFrame:
        local_ns = {**dataframes, "pd": pd, "np": np}
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(exec, code, {"__builtins__": self.SAFE_BUILTINS}, local_ns)
            future.result(timeout=self.TIMEOUT_SECONDS)
        return local_ns.get("result")

    def _static_check(self, code):
        combined = f"{code.sql or ''} {code.pandas or ''}"
        for token in self.BANNED_TOKENS:
            if token in combined:
                raise SecurityError(f"Banned token detected: {token}")