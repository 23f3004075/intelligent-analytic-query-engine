import duckdb
import json
import io
import pandas as pd
from pathlib import Path


class DataContextBuilder:
    """The most critical component — an LLM is only as good as its context.
    
    Loads CSVs into DuckDB, parses the data dictionary, computes per-column
    statistics, and builds the schema digest injected into every LLM prompt.
    """

    def __init__(self, csv_paths: list, data_dict_path: str = None):
        self.conn = duckdb.connect(database=':memory:')
        self.csv_paths = csv_paths
        self.dataframes = {}
        self.data_dictionary = self._load_data_dictionary(data_dict_path)
        self._load_csvs()

    # ── Robust File Loading ──────────────────────────────────────────────

    def _load_data_dictionary(self, path):
        """Robustly load data_dictionary.json, handling various formats.
        
        Handles both standard JSON and the quirky format where each line
        is individually quoted with doubled internal quotes.
        """
        if not path or not Path(path).exists():
            return {}

        with open(path, 'r', encoding='utf-8-sig') as f:
            raw = f.read()

        # Try standard JSON first (recruiter may provide clean JSON)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: handle quirky format where lines are wrapped in quotes
        # with doubled internal quotes (e.g., "  ""key"": ""value"",")
        lines = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 2:
                inner = stripped[1:-1]       # Remove outer quotes
                inner = inner.replace('""', '"')  # Un-double internal quotes
                lines.append(inner)
            else:
                lines.append(stripped)

        cleaned = '\n'.join(lines)
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return {}  # Graceful degradation

    def _load_csv_robust(self, csv_path):
        """Load CSV handling various formats including quoted-line CSVs.
        
        Some CSVs wrap each entire row in quotes — e.g.:
            "order_id,order_date,region,..."
            "1001,2024-01-05,APAC,..."
        This method detects and handles that format.
        """
        path = Path(csv_path)

        # Try standard pandas read first
        try:
            df = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass

        # Fallback: strip outer quotes from each line and re-parse
        with open(path, 'r', encoding='utf-8-sig') as f:
            raw = f.read()

        lines = []
        for line in raw.strip().splitlines():
            stripped = line.strip()
            if stripped.startswith('"') and stripped.endswith('"'):
                stripped = stripped[1:-1]
            lines.append(stripped)

        cleaned = '\n'.join(lines)
        return pd.read_csv(io.StringIO(cleaned), keep_default_na=False)

    def _load_csvs(self):
        """Load all CSV files into both Pandas DataFrames and DuckDB tables."""
        for csv_path in self.csv_paths:
            table_name = Path(csv_path).stem
            df = self._load_csv_robust(csv_path)
            self.dataframes[table_name] = df

            # Register DataFrame in DuckDB as a table
            self.conn.register(f"_tmp_{table_name}", df)
            self.conn.execute(
                f"CREATE TABLE {table_name} AS SELECT * FROM _tmp_{table_name}"
            )
            self.conn.unregister(f"_tmp_{table_name}")

    # ── Accessors ────────────────────────────────────────────────────────

    def get_connection(self):
        """Return the DuckDB connection with all tables loaded."""
        return self.conn

    def get_dataframes(self):
        """Return dict of DataFrames for Pandas fallback execution.
        
        Keys are formatted as <table_name>_df (e.g., sales_data_df).
        """
        return {f"{name}_df": df for name, df in self.dataframes.items()}

    def get_all_columns(self):
        """Return a set of all column names across all tables."""
        columns = set()
        for table_name in self.dataframes:
            try:
                cols = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
                for col_info in cols:
                    columns.add(col_info[0])
            except Exception:
                # Fallback to DataFrame columns
                columns.update(self.dataframes[table_name].columns.tolist())
        return columns

    # ── Schema Digest (injected into every LLM call) ─────────────────────

    def build_schema_digest(self):
        """Build a human-readable schema digest for LLM prompt injection.
        
        Includes table structure, column types, per-column stats, sample
        values, and the business glossary from data_dictionary.json.
        """
        digest_parts = []

        tables = self.conn.execute("SHOW TABLES").fetchall()
        for (table_name,) in tables:
            cols = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            row_count = self.conn.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]

            col_lines = []
            for col_info in cols:
                col_name = col_info[0]
                col_type = col_info[1]
                stats = self._get_column_stats(table_name, col_name, col_type)
                col_lines.append(f"  {col_name:<30} {col_type:<12} {stats}")

            header = f"TABLE: {table_name} ({row_count} rows)"
            digest_parts.append(header + "\n" + "\n".join(col_lines))

        # Append business glossary from data dictionary
        glossary = self.build_glossary()
        if glossary:
            digest_parts.append(glossary)

        return "\n\n".join(digest_parts)

    def _get_column_stats(self, table, col, dtype):
        """Compute per-column statistics for the schema digest."""
        dtype_upper = dtype.upper()

        try:
            if any(t in dtype_upper for t in [
                'INT', 'FLOAT', 'DOUBLE', 'DECIMAL', 'BIGINT', 'NUMERIC', 'HUGEINT'
            ]):
                result = self.conn.execute(
                    f'SELECT MIN("{col}"), MAX("{col}"), '
                    f'ROUND(AVG("{col}"), 2), COUNT(DISTINCT "{col}") '
                    f'FROM {table}'
                ).fetchone()
                return (f"[min: {result[0]}, max: {result[1]}, "
                        f"mean: {result[2]}, {result[3]} unique]")

            elif any(t in dtype_upper for t in ['DATE', 'TIMESTAMP']):
                result = self.conn.execute(
                    f'SELECT MIN("{col}"), MAX("{col}") FROM {table}'
                ).fetchone()
                return f"[{result[0]} -> {result[1]}]"

            else:  # VARCHAR / TEXT
                result = self.conn.execute(
                    f'SELECT COUNT(DISTINCT "{col}") FROM {table}'
                ).fetchone()
                n_unique = result[0]

                samples = self.conn.execute(
                    f'SELECT DISTINCT "{col}" FROM {table} LIMIT 5'
                ).fetchall()
                sample_str = ", ".join(f'"{s[0]}"' for s in samples)

                if n_unique <= 15:
                    return f"[{n_unique} unique] e.g. {sample_str}"
                else:
                    return f"[{n_unique} unique]"

        except Exception:
            return "[stats unavailable]"

    # ── Glossary from Data Dictionary ────────────────────────────────────

    def build_glossary(self):
        """Build business glossary string from data_dictionary.json."""
        if not self.data_dictionary:
            return ""

        lines = ["BUSINESS GLOSSARY (from data_dictionary.json):"]

        # Metrics (computed fields)
        metrics = self.data_dictionary.get("metrics", {})
        if isinstance(metrics, dict):
            for name, formula in metrics.items():
                lines.append(f'  "{name}" -> {formula}')

        # Synonyms (word mappings)
        synonyms = self.data_dictionary.get("synonyms", {})
        if isinstance(synonyms, dict):
            for alias, canonical in synonyms.items():
                lines.append(f'  "{alias}" -> {canonical}')

        # Time mappings
        time_maps = self.data_dictionary.get("time_mappings", {})
        if isinstance(time_maps, dict):
            for phrase, meaning in time_maps.items():
                lines.append(f'  "{phrase}" -> {meaning}')

        # Dimensions (groupable columns)
        dimensions = self.data_dictionary.get("dimensions", [])
        if isinstance(dimensions, list) and dimensions:
            lines.append(f'  Groupable dimensions: {", ".join(dimensions)}')

        return "\n".join(lines) if len(lines) > 1 else ""
