# IAQE Codebase & Pipeline Architectural Analysis

This document provides an exhaustive, function-by-function and class-by-class technical breakdown of the **Intelligent Analytic Query Engine (IAQE)** codebase, detailing how each component operates, transforms data, handles edge cases, and interfaces with the rest of the system.

---

## Architecture Map

```
IAQE/
├── engine.py                  [QueryEngine — Central Pipeline Orchestrator]
└── pipeline/
    ├── preprocessor.py        [Step 1:  Normalization, Domain Spell-Check, Complexity Classification]
    ├── context_builder.py     [Step 2:  CSV Ingestion, DuckDB Registration, Schema Digest & Glossary]
    ├── intent.py              [Step 3:  LLM Intent Classification & Entity Extraction]
    ├── feedback.py            [Step 4 & 10: TF-IDF Few-Shot Retrieval & Feedback Logging]
    ├── generator.py           [Step 5:  Dual-Track SQL & Pandas Code Generation]
    ├── executor.py            [Step 6:  Sandboxed Safe Execution with Dual-Track Fallback]
    ├── validator.py           [Step 7:  DataFrame Sanity Checks & Contract Validation]
    ├── scorer.py              [Step 8:  Multi-Signal Confidence Scoring Matrix]
    └── explainer.py           [Step 9:  3-Part Plain-English Business Explanation Generator]
```

---

## 1. `IAQE/pipeline/preprocessor.py`

### Purpose
Prepares incoming raw user questions before any LLM calls are made. It normalizes text, executes domain-aware spell-checking, and categorizes query complexity to dynamically guide prompt generation.

### Dataclasses & Enums

#### `Complexity(Enum)`
Represents the structural difficulty of a query:
- `SIMPLE`: Single table, basic aggregation (`SUM`, `AVG`, `COUNT`), no multi-table joins or window functions.
- `MEDIUM`: Requires grouping (`GROUP BY`), multi-column filtering, or basic ordering.
- `COMPLEX`: Requires window functions (`RANK()`, `PARTITION BY`), cross-table joins, percentage of total calculations, nested queries, or YoY growth calculations.

#### `PreprocessedQuery`
Immutable container for the preprocessor's output:
- `raw_q: str`: The original raw input string verbatim.
- `normalized_q: str`: Cleaned, whitespace-collapsed, lowercased string.
- `complexity: Complexity`: Enum classification used to toggle Chain-of-Thought prompting.

### Class: `QueryPreprocessor`

#### `__init__()`
- Instantiates `pyspellchecker.SpellChecker`.
- **Domain Word Whitelist (`DOMAIN_WORDS`)**: Crucial safeguard protecting domain acronyms and technical terms from being corrupted by spell checkers (e.g. `apac`, `emea`, `latam`, `yoy`, `mom`, `aov`, `sub_category`, `subcategory`, `sql`, `duckdb`, `sum`, `avg`, `min`, `max`).

#### `process(query: str) -> PreprocessedQuery`
1. Calls `_normalize(query)`.
2. Calls `_spell_check(normalized)`.
3. Calls `_estimate_complexity(spell_checked)`.
4. Returns `PreprocessedQuery`.

#### `_normalize(query: str) -> str`
- Strips leading/trailing whitespace.
- Collapses consecutive whitespace characters into a single space.
- Converts characters to lowercase while preserving alphanumeric tokens, quotes, and mathematical symbols (`%`, `$`, `/`, `-`).

#### `_spell_check(query: str) -> str`
- Tokenizes query by word boundaries.
- Inspects each token: if the token is in `DOMAIN_WORDS`, contains numbers, or is recognized by the dictionary, it is left untouched.
- Only genuinely misspelled English words are corrected using `spell.correction(word)`.

#### `_estimate_complexity(query: str) -> Complexity`
- Scans normalized query for deterministic operational keywords:
  - **Complex triggers**: `window`, `rank`, `partition`, `growth`, `yoy`, `mom`, `contribution`, `percent of total`, `% of`, `vs target`, `compare`, `nested`.
  - **Medium triggers**: `group by`, `by region`, `by country`, `by category`, `per`, `top`, `bottom`, `highest`, `lowest`, `filter`, `where`.
  - **Simple fallback**: Queries matching basic sum/count operations or failing complex/medium triggers.

---

## 2. `IAQE/pipeline/context_builder.py`

### Purpose
Ingests raw CSV datasets, registers them into an in-memory DuckDB instance, extracts schema definitions, and compiles a concise schema digest and business glossary for LLM prompt grounding.

### Class: `DataContextBuilder`

#### `__init__(csv_paths: list[str], data_dict_path: str = None)`
- **In-Memory DuckDB**: Initializes an isolated DuckDB connection (`duckdb.connect(database=':memory:')`).
- **CSV Ingestion**: Reads each CSV file via `pandas.read_csv(encoding="utf-8-sig")` to handle UTF-8 byte order marks cleanly.
- **DuckDB Registration**: Registers every DataFrame as an active virtual table in DuckDB using `self.conn.register(table_name, df)`.
- **Quirky Formatting Handling**: Calls `_load_data_dictionary()` using robust line parsing to handle double-quoted escape patterns found in recruiter datasets.

#### `build_schema_digest() -> str`
Generates a token-efficient text summary injected into the system prompt of every LLM call:
- Format per table:
  ```text
  Table: sales_data (1,000 rows)
  Columns:
    - order_id (VARCHAR): sample values ['ORD-001', 'ORD-002']
    - order_date (DATE): sample values ['2024-01-15', '2024-02-20']
    - profit (DOUBLE): min -120.5, max 4500.0, avg 320.4
  ```
- **Why it matters**: In-prompt schema injection completely eliminates the #1 failure mode in Text-to-SQL — hallucinated column names.

#### `build_glossary() -> dict[str, str]`
Maps business terms to mathematical formulas:
- `revenue`: `unit_price * quantity * (1 - discount)`
- `profit`: `revenue - (unit_price * quantity * cost_pct) - shipping_cost`
- `aov`: `SUM(revenue) / COUNT(DISTINCT order_id)`
- `yoy_growth`: `((current_revenue - prior_revenue) / prior_revenue) * 100`

#### `get_all_columns() -> set[str]`
Returns a unified flat set of all column names across all registered tables. Used by `ConfidenceScorer` to verify entity grounding.

---

## 3. `IAQE/pipeline/intent.py`

### Purpose
Executes **LLM Call #1**. Dissects the user's natural language question into structured analytical operations, identified dimensions, target tables, and ambiguities before any code is generated.

### Models

#### `QueryIntent(BaseModel)`
Pydantic v2 contract defining query intent:
- `primary_operation: str`: Main intent (`AGGREGATE`, `FILTER`, `RANK`, `COMPARE`, `CONTRIBUTION`, `TIME_SERIES`).
- `secondary_operations: list[str]`: Supporting steps (`GROUP_BY`, `ORDER_BY`, `LIMIT`, `WINDOW_FUNCTION`).
- `entities: dict[str, Any]`: Identified entity mappings (e.g. `{"region": "North", "metric": "revenue"}`).
- `time_constraint: Optional[str]`: Identified time periods (e.g. `"March 2024"`, `"Q1"`).
- `tables_needed: list[str]`: Tables required for execution (e.g. `["sales_data", "targets"]`).
- `n_value: Optional[int]`: Extracted ranking count (e.g. `2` for "Top 2 cities").
- `ambiguities: list[str]`: List of ambiguities detected by the LLM (e.g. "Unspecified year for March").

### Class: `IntentClassifier`

#### `__init__(client: OpenAI, model: str)`
Stores the shared OpenAI client and target model identifier (`inclusionai/ling-3.0-flash-vl:free`).

#### `classify(query: str, schema_digest: str) -> QueryIntent`
1. Formulates system prompt specifying the exact JSON schema required.
2. Formulates user prompt passing the query and the table schema digest.
3. Invokes `client.chat.completions.create` with `response_format={"type": "json_object"}`.
4. Parses JSON string into a validated `QueryIntent` model.
5. **Fallback Safety**: If the model fails or times out, creates a safe default `QueryIntent` marked with ambiguity flags so execution can continue safely.

---

## 4. `IAQE/pipeline/feedback.py`

### Purpose
Implements the continuous learning loop. It indexes past queries in `feedback_log.csv`, retrieves similar past queries via TF-IDF cosine similarity for few-shot prompt injection (Step 4), and persists telemetry and human feedback (Step 10).

### Dataclasses

#### `RetrievedFeedback`
- `query: str`: Past query text.
- `generated_logic: str`: Winning code produced for that query.
- `was_correct: bool`: Whether the human or validation verified it as correct.
- `score: float`: Cosine similarity score between current and past query.
- `feedback_note: str`: Correction note explaining errors or confirming logic.

#### `FeedbackEntry`
Row schema serialized into `feedback_log.csv`:
- `timestamp`, `query`, `generated_logic`, `result_preview`, `confidence_score`, `track_used`, `intent_json`, `feedback` (`"pending" | "correct" | "incorrect"`), `feedback_note`.

### Class: `FeedbackRetriever`

#### `__init__(feedback_log_path: str)`
- Checks if `feedback_log.csv` exists.
- If present, filters for evaluated queries (`feedback != "pending"`).
- Fits `sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))` on past queries.
- **Cold Start Handling**: If the log file is missing or has no evaluated queries, `is_active` remains `False` and returns empty lists without error.

#### `retrieve(query: str, top_k: int = 3) -> list[RetrievedFeedback]`
- Computes cosine similarity between current query vector and all indexed vectors.
- Filters matches with similarity $> 0.60$.
- Returns up to `top_k` matches. If similarity $> 0.85$, it qualifies for a strong confidence boost.

### Class: `FeedbackLogger`

#### `log(entry: FeedbackEntry) -> int`
- Appends `FeedbackEntry` as a row in `feedback_log.csv`.
- Writes CSV headers only if the file is newly created.
- Returns the 0-indexed integer row position (`entry_id`) for downstream telemetry and feedback tracking.

#### `mark_feedback(entry_id: int, was_correct: bool, note: str = "") -> bool`
- Loads `feedback_log.csv` with explicit string typing (`dtype={"feedback": str, "feedback_note": str}`) to prevent pandas float64 deprecation warnings.
- Updates the specified row index to `"correct"` or `"incorrect"`.
- Saves changes back to disk.

#### `mark_last(was_correct: bool, note: str = "") -> bool`
Convenience helper that updates the most recently appended row in the log.

---

## 5. `IAQE/pipeline/generator.py`

### Purpose
Executes **LLM Call #2**. Generates dual-track executable code: DuckDB-compatible SQL (primary) and Pandas code (fallback).

### Dataclasses

#### `GeneratedCode`
- `sql: Optional[str]`: SQL query text.
- `pandas: Optional[str]`: Python / Pandas script text.
- `self_confidence: float`: Model's self-assessed probability of correctness (0.0 to 1.0).
- `reasoning: str`: Analytical explanation of the query structure.
- `had_syntax_error: bool`: Flag indicating whether static syntax checking passed.

### Class: `CodeGenerator`

#### `generate(query, intent, schema_digest, glossary, feedback_examples, complexity) -> GeneratedCode`
1. Constructs prompt containing:
   - Complete schema digest (table and column names).
   - Business metric formulas from the glossary.
   - Query intent details (operations, target tables, filters, top-N).
   - Few-shot examples retrieved from `FeedbackRetriever` (both positive examples and negative guardrails).
2. **Chain-of-Thought (CoT)**: If `complexity == COMPLEX`, injects step-by-step thinking requirements (identifying output grain, required CTEs/window functions, and join conditions).
3. Calls LLM with `response_format={"type": "json_object"}`.
4. Extracts `sql`, `pandas`, `self_confidence`, and `reasoning`.
5. **Static Syntax Linting**: Uses Python's `ast.parse` on the Pandas code and basic dialect validation on the SQL query to pre-screen for syntax errors before execution.

---

## 6. `IAQE/pipeline/executor.py`

### Purpose
Implements the dual-track execution engine. Executes DuckDB SQL first; if SQL encounters any runtime or dialect errors, it automatically falls back to executing Pandas code within a secured sandbox.

### Dataclasses & Exceptions

#### `SecurityError(Exception)`
Raised when unauthorized or unsafe operations are detected.

#### `ExecutionContext`
- `duckdb_conn: Any`: Live in-memory DuckDB connection.
- `dataframes: dict[str, pd.DataFrame]`: Dictionary mapping table names to Pandas DataFrames.

#### `ExecutionResult`
- `success: bool`: True if either track returned a valid DataFrame.
- `result: Optional[pd.DataFrame]`: Resulting data.
- `track: Optional[str]`: `"SQL"` or `"Pandas"`.
- `error: Optional[str]`: Error message if both tracks failed.

### Class: `SafeExecutor`

#### Constants
- `TIMEOUT_SECONDS = 30`: Maximum execution time before raising a timeout.
- `MAX_RESULT_ROWS = 10,000`: Caps maximum rows returned to protect memory.
- `BANNED_TOKENS`: Blacklist preventing dangerous Python calls (`import os`, `subprocess`, `eval(`, `exec(`, `__import__`, `open(`, `sys.`).
- `SAFE_BUILTINS`: Whitelist of safe Python builtins (`abs`, `len`, `min`, `max`, `sum`, `round`, `range`, `enumerate`, `zip`, `sorted`, `str`, `int`, `float`, `bool`, `list`, `dict`, `tuple`, `isinstance`).

#### `execute(code: GeneratedCode, context: ExecutionContext) -> ExecutionResult`
1. **Static Security Check**: Scans Pandas code against `BANNED_TOKENS`.
2. **Track 1 (DuckDB SQL)**:
   - If `code.sql` is present, executes `context.duckdb_conn.execute(sql).df()`.
   - On success: returns `ExecutionResult(success=True, result=df, track="SQL")`.
   - On SQL error: logs failure and gracefully drops to Track 2.
3. **Track 2 (Pandas Fallback)**:
   - If `code.pandas` is present, executes inside a sandboxed namespace containing `SAFE_BUILTINS` and copies of `context.dataframes`.
   - Uses `concurrent.futures.ThreadPoolExecutor` to enforce `TIMEOUT_SECONDS`.
   - On success: returns `ExecutionResult(success=True, result=df, track="Pandas")`.
4. If both tracks fail, returns `ExecutionResult(success=False, error="Both tracks failed: ...")`.

---

## 7. `IAQE/pipeline/validator.py`

### Purpose
Executes deterministic sanity checks on the resulting DataFrame against the structured `QueryIntent` to prevent hallucinated, empty, or statistically impossible results from reaching the user.

### Dataclasses

#### `ValidationCheck`
- `check_name: str`: Descriptive name of check.
- `passed: bool`: True if check passed.
- `detail: str`: Explanation of finding or failure.

#### `ValidationReport`
- `checks: list[ValidationCheck]`: List of all checks executed.
- `pass_rate: float`: Ratio of passed checks (0.0 to 1.0).
- `issues: list[str]`: Critical issues flagged.

### Class: `ResultValidator`

#### `validate(df: pd.DataFrame, intent: QueryIntent) -> ValidationReport`
Runs 5 independent verification checks:
1. **Non-Empty Check**: Verifies that the DataFrame contains at least one row, unless the intent explicitly expected a zero-count outcome.
2. **Top-N Row Count Check**: If the intent specifies `n_value` (e.g. Top 2 cities), verifies that `len(df) <= n_value`.
3. **Numeric Sanity Check**:
   - If operation is `PCT_CONTRIBUTION`, verifies that values are $\ge 0$ and sum to approximately $100\% \pm 1\%$.
   - Verifies that count or quantity columns contain non-negative numbers.
4. **Expected Columns Check**: Verifies that columns projected in the DataFrame correspond to the entities or metrics requested in `intent.entities`.
5. **Aggregation Consistency Check**: If the query is a global aggregation without a `GROUP_BY` secondary operation, verifies that the result is a single summary row.

Computes `pass_rate = passed_checks / total_checks` used directly in confidence scoring.

---

## 8. `IAQE/pipeline/scorer.py`

### Purpose
Implements a mathematical, 5-signal composite scoring matrix to evaluate query reliability without human intervention.

### Dataclasses

#### `ConfidenceSignal`
- `schema_match: float`: Ratio of intent entities matching real database columns (0.0 to 1.0).
- `code_validity: float`: 1.0 if syntax check passed, 0.0 if syntax errors occurred.
- `execution_success: float`: 1.0 for DuckDB SQL, 0.7 for Pandas fallback, 0.0 for failure.
- `result_validation: float`: Validation check pass rate from `ResultValidator` (0.0 to 1.0).
- `llm_self_score: float`: Self-confidence score output by the code generation LLM (0.0 to 1.0).

### Class: `ConfidenceScorer`

#### `__init__()`
Defines the linear weight distribution:
- `schema_match`: **25%**
- `code_validity`: **20%**
- `execution_success`: **25%**
- `result_validation`: **20%**
- `llm_self_score`: **10%**
*(Sum of weights = 1.00)*

#### `score(signals: ConfidenceSignal, feedback_boost: float = 0.0, ambiguities_count: int = 0) -> float`
1. **Weighted Linear Base**:
   $$\text{raw} = \sum (\text{signal}_i \times \text{weight}_i)$$
2. **Hard Penalty Thresholds**:
   - If `execution_success == 0.0`: score is capped at **0.15** (unusable output).
   - If `schema_match < 0.50`: score is capped at **0.30** (high hallucination risk).
   - If both penalties trigger, the strictest cap (**0.15**) takes precedence.
3. **Ambiguity Discount**:
   - If `ambiguities_count > 2`: multiplies score by **0.85** (15% penalty for high query vagueness).
4. **Feedback Boost**:
   - Adds boost (+0.02 for weak similarity, +0.05 for strong similarity with past correct queries).
   - Capped at **0.97** (maintains epistemic humility; no query receives an unchecked 1.0).
5. Returns `round(score, 3)`.

---

## 9. `IAQE/pipeline/explainer.py`

### Purpose
Executes **LLM Call #3**. Converts the analytical execution details and DataFrame result into a plain-English, 3-part business explanation tailored for non-technical stakeholders.

### Dataclasses

#### `Explanation`
- `understood: str`: Clear restatement of how the system interpreted the user's intent.
- `approach: str`: Step-by-step description of how data was filtered, aggregated, and joined.
- `caveats: str`: Identified data limitations, date assumptions, null handling, or join conventions.

### Class: `ExplanationGenerator`

#### `explain(query, intent, generated_logic, track, result_df, score) -> Explanation`
- Injects a preview of the actual result DataFrame (first 5 rows), the executed code, and the confidence score.
- Instructs the model:
  - Audience: Business executives and non-technical managers.
  - Tone: Objective, clear, jargon-free.
  - Strictly enforce 3 keys: `understood`, `approach`, `caveats`.
- Returns validated `Explanation` instance.

---

## 10. `IAQE/engine.py` (Central Coordinator)

### Purpose
Coordinates all 10 pipeline steps, implements the **Self-Healing Retry Loop**, records execution telemetry, and returns the final standardized response payload.

### Class: `QueryEngine`

#### `__init__(csv_paths, data_dict_path, feedback_log_path, model)`
Instantiates all 9 pipeline components and initializes the in-memory DuckDB connection.

#### `run(query: str, track: str = "auto") -> dict`
Executes the full pipeline:
1. **Step 1**: Preprocesses query (spell checking, complexity).
2. **Step 2**: Reuses pre-built schema digest and glossary.
3. **Step 3**: Classifies intent via `IntentClassifier` (LLM Call #1).
4. **Step 4**: Retrieves similar past feedback via `FeedbackRetriever`.
5. **Step 5**: Generates SQL & Pandas logic via `CodeGenerator` (LLM Call #2).
6. **Step 6**: Executes in `SafeExecutor`.
   - **Self-Healing Retry**: If initial execution fails, captures the runtime error string, injects it into a retry prompt (`[PREVIOUS CODE ATTEMPT FAILED WITH ERROR]: ...`), and re-executes.
7. **Step 7**: Validates result via `ResultValidator`.
8. **Step 8**: Calculates composite score via `ConfidenceScorer`.
9. **Step 9**: Generates 3-part explanation via `ExplanationGenerator` (LLM Call #3).
10. **Step 10**: Logs entry to `feedback_log.csv` and returns `entry_id`.
11. **Output Assembly**: Returns the standardized dictionary:
    ```json
    {
      "query": "...",
      "generated_logic": "...",
      "result": [...],
      "confidence_score": 0.96,
      "explanation": {"understood": "...", "approach": "...", "caveats": "..."},
      "metadata": {
        "track_used": "SQL",
        "complexity": "MEDIUM",
        "intent": {...},
        "processing_time_ms": 1420,
        "llm_calls": 3,
        "entry_id": 4
      }
    }
    ```

#### `mark_feedback(entry_id: int = None, was_correct: bool = True, note: str = "") -> bool`
Delegates feedback marking to `FeedbackLogger`.

---

## 11. `server.py` & `static/` (Web Dashboard)

- **`server.py`**: A lightweight FastAPI application that creates an engine singleton at startup, exposes `/api/info`, `/api/samples`, `/api/query`, and `/api/feedback`, and mounts static assets.
- **`static/index.html`**: Zero-build HTML5 template with Vue 3 mount and company-styled header, sub-hero strip, sample chips, metric badges, results table, and code viewer without emojis.
- **`static/style.css`**: Vanilla CSS implementing the **Office Solution AI Labs** brand palette (`#1A1D1C` charcoal, `#FFB401` gold, `#016064` deep teal, and `#1E2322` dark cards).
- **`static/app.js`**: Vue 3 reactive controller handling API requests, animated pipeline progress states, CSV export generation, and clipboard management.
