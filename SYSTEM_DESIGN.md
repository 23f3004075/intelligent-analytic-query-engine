# Intelligent Analytics Query Engine — System Design

> **Author:** Sourasish Ghosh  
> **Role Applied For:** Data Engineer / ML Engineer  
> **Time Budget:** 4–8 hours  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Component Deep Dive](#3-component-deep-dive)
4. [Data Flow — Request Lifecycle](#4-data-flow--request-lifecycle)
5. [Tech Stack & Justification](#5-tech-stack--justification)
6. [Prompt Engineering Strategy](#6-prompt-engineering-strategy)
7. [Confidence Scoring System](#7-confidence-scoring-system)
8. [Feedback Loop Design](#8-feedback-loop-design)
9. [Error Recovery Strategy](#9-error-recovery-strategy)
10. [Repository Layout](#10-repository-layout)
11. [Sample Output](#11-sample-output)
12. [Design Trade-offs](#12-design-tradeoffs)
13. [Scalability Path](#13-scalability-path)
14. [If Given More Time](#14-if-given-more-time)

---

## 1. Executive Summary

The Intelligent Analytics Query Engine (IAQE) translates **natural language business questions** into
verified, executable analytical logic over structured CSV data — and returns results with a confidence
score, step-by-step explanation, and self-improving feedback loop.

### Core Design Philosophy

| Principle | How It Shows Up |
|-----------|-----------------|
| **Schema-first grounding** | LLM always sees the full data dictionary before generating any code |
| **Two-track execution** | Every query is attempted in SQL (DuckDB) first; Pandas is the fallback |
| **Verify before trust** | Generated code passes a static lint check and a result-sanity check before being returned |
| **Confidence is earned, not assumed** | A composite scorer weights five independent signals |
| **Feedback compounds** | Past successes become few-shot examples; past failures become guardrails |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     INTELLIGENT ANALYTICS QUERY ENGINE                  │
│                                                                         │
│   ┌──────────┐    ┌────────────────────────────────────────────────┐    │
│   │  INPUTS  │    │               CORE PIPELINE                   │    │
│   │          │    │                                                │    │
│   │ NL Query │───▶│  [1] QueryPreprocessor                        │    │
│   │          │    │        │ Normalize, spell-check               │    │
│   │ CSV Data │    │        ▼                                       │    │
│   │          │───▶│  [2] DataContextBuilder                       │    │
│   │ Data     │    │        │ Schema + sample rows + stats         │    │
│   │ Dict     │───▶│        │                                       │    │
│   │          │    │        ▼                                       │    │
│   │ nl_queries│   │  [3] IntentClassifier (LLM)                   │    │
│   │ .json    │───▶│        │ Intent tags + entity mapping         │    │
│   │          │    │        │                                       │    │
│   │ feedback │    │        ▼                                       │    │
│   │ _log.csv │───▶│  [4] FeedbackRetriever                        │    │
│   │          │    │        │ Semantic search for similar past Q   │    │
│   └──────────┘    │        │                                       │    │
│                   │        ▼                                       │    │
│                   │  [5] CodeGenerator (LLM)                      │    │
│                   │        │ SQL (primary) + Pandas (fallback)    │    │
│                   │        │                                       │    │
│                   │        ▼                                       │    │
│                   │  [6] SafeExecutor                             │    │
│                   │        │ DuckDB → Pandas fallback             │    │
│                   │        │ Timeout + sandbox                    │    │
│                   │        │                                       │    │
│                   │        ▼                                       │    │
│                   │  [7] ResultValidator                          │    │
│                   │        │ Sanity checks on output shape        │    │
│                   │        │                                       │    │
│                   │        ▼                                       │    │
│                   │  [8] ConfidenceScorer                         │    │
│                   │        │ 5-signal composite score             │    │
│                   │        │                                       │    │
│                   │        ▼                                       │    │
│                   │  [9] ExplanationGenerator (LLM)               │    │
│                   │        │ What was understood + how            │    │
│                   │        │                                       │    │
│                   │        ▼                                       │    │
│                   │  [10] FeedbackLogger                          │    │
│                   │        │ Persist result + score               │    │
│                   └────────┼───────────────────────────────────────┘    │
│                            ▼                                            │
│                     ┌─────────────┐                                     │
│                     │   OUTPUT    │                                     │
│                     │  JSON + CSV │                                     │
│                     └─────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Deep Dive

### 3.1 DataContextBuilder

The most critical component — an LLM is only as good as its context.

**Responsibilities:**
- Loads all CSVs into an **in-memory DuckDB** instance (one table per file)
- Parses `data_dictionary.json` → canonical column metadata
- Computes per-column statistics (dtype, null %, min, max, top-5 unique values)
- Builds a **schema digest** string injected into every prompt

**Output — Schema Digest (injected into every LLM call):**
```
TABLE: sales_data
  order_id       TEXT      [unique] e.g. "ORD-001"
  sales_rep      TEXT      [23 unique] e.g. "Alice", "Bob"
  region         TEXT      [4 unique] "North", "South", "East", "West"
  product        TEXT      [12 unique]
  category       TEXT      [3 unique] "Electronics", "Apparel", "Food"
  sale_amount    FLOAT     [min: 50.0, max: 12000.0, mean: 1340.2]
  sale_date      DATE      [2024-01-01 → 2024-12-31]
  units_sold     INT       [min: 1, max: 500]

TABLE: targets
  region         TEXT      [4 unique]
  quarter        TEXT      [Q1-Q4]
  target_amount  FLOAT     [min: 10000, max: 200000]

BUSINESS GLOSSARY (from data_dictionary.json):
  "revenue"       → sales_data.sale_amount
  "reps"          → sales_data.sales_rep
  "performance"   → sale_amount vs targets.target_amount
  "best"/"top"    → ORDER BY sale_amount DESC
  "Q1"            → sale_date BETWEEN '2024-01-01' AND '2024-03-31'
```

---

### 3.2 QueryPreprocessor

**Lightweight, no LLM needed.**

```python
class QueryPreprocessor:
    def process(self, raw_query: str) -> PreprocessedQuery:
        normalized = self._normalize(raw_query)   # lowercase, strip punct
        corrected  = self._spell_check(normalized) # pyspellchecker
        complexity = self._estimate_complexity(corrected)
        return PreprocessedQuery(
            original=raw_query,
            normalized=corrected,
            complexity=complexity  # SIMPLE | MEDIUM | COMPLEX
        )
    
    def _estimate_complexity(self, q: str) -> Complexity:
        signals = {
            Complexity.COMPLEX: ["compared to target", "top N", "each", "per",
                                  "percentage", "contribution", "vs", "rank",
                                  "within", "across", "over time"],
            Complexity.MEDIUM:  ["by", "group", "filter", "where", "last month"],
            Complexity.SIMPLE:  ["total", "average", "count", "sum", "list"],
        }
        # Scores based on keyword hits — returns highest matching tier
```

**Why this matters:** Complexity drives how many LLM tokens we spend and which prompting strategy we use.

---

### 3.3 IntentClassifier (LLM — Fast Call)

A **lightweight, structured LLM call** that classifies intent before the expensive code generation step.

**Input:** Preprocessed query + schema digest  
**Output:** Structured `QueryIntent` object

```python
@dataclass
class QueryIntent:
    primary_operation: str          # AGGREGATE | RANK | FILTER | COMPARE | TIME_SERIES
    secondary_operations: list[str] # ["GROUP_BY", "TOP_N", "PCT_CONTRIBUTION"]
    entities: dict[str, str]        # {"region": "North", "metric": "sale_amount"}
    time_constraint: str | None     # "Q1 2024", "last 3 months", None
    n_value: int | None             # For "top N" queries
    tables_needed: list[str]        # ["sales_data", "targets"]
    ambiguities: list[str]          # Things the system is uncertain about
```

**Prompt template (condensed):**
```
You are a query intent parser. Given a natural language analytics question 
and a database schema, extract a structured JSON intent.

SCHEMA:
{schema_digest}

QUERY: "{query}"

Respond ONLY with a valid JSON matching this schema: {QueryIntent schema}
Never guess column names — use exactly what the schema lists.
```

**Why a separate classifier call?**  
- Cheaper (small output, fast)  
- Intent is reusable if code generation fails and we retry  
- Ambiguities flagged here reduce confidence score downstream  

---

### 3.4 FeedbackRetriever

Before generating new code, check if we've seen something similar before.

```python
class FeedbackRetriever:
    def __init__(self, feedback_log_path: str):
        self.log = pd.read_csv(feedback_log_path)
        # Embed all past queries at startup using TF-IDF + cosine similarity
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2))
        self.vectors = self.vectorizer.fit_transform(self.log["query"])
    
    def retrieve(self, query: str, top_k: int = 3) -> list[FeedbackEntry]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.vectors).flatten()
        top_idx = scores.argsort()[-top_k:][::-1]
        entries = []
        for i in top_idx:
            if scores[i] > 0.6:  # Similarity threshold
                entries.append(FeedbackEntry(
                    query=self.log.iloc[i]["query"],
                    generated_logic=self.log.iloc[i]["generated_logic"],
                    was_correct=self.log.iloc[i]["feedback"] == "correct",
                    score=float(scores[i])
                ))
        return entries
```

**Output used in two ways:**
- Correct past answers → **few-shot examples** in CodeGenerator prompt
- Incorrect past answers → **negative examples** ("avoid this pattern")

---

### 3.5 CodeGenerator (LLM — Main Call)

The core generation step. Uses **Chain-of-Thought** prompting for COMPLEX queries.

**Two output tracks, always attempted in order:**

```
Track A (Primary):   SQL via DuckDB
Track B (Fallback):  Pandas code
```

**Prompt structure by complexity:**

```
SIMPLE:
  → Single-shot: "Generate SQL for: {query}"

MEDIUM:
  → Structured: Think step-by-step, then write SQL

COMPLEX:
  → CoT: 
      Step 1: Identify all tables needed
      Step 2: Identify the grain (what is one row of the result?)
      Step 3: Write any subqueries or CTEs needed
      Step 4: Assemble the final query
      Step 5: Write equivalent Pandas as backup
```

**Full prompt template:**
```
You are an expert SQL and Pandas code generator for business analytics.

--- DATABASE SCHEMA ---
{schema_digest}

--- BUSINESS GLOSSARY ---
{glossary}

--- QUERY INTENT (already parsed) ---
{intent_json}

--- SIMILAR PAST QUERIES (few-shot) ---
{feedback_examples}  ← only included if similarity > 0.6

--- TASK ---
Generate executable code to answer: "{query}"

Rules:
1. Use ONLY column names that exist in the schema. Never invent columns.
2. For SQL: tables are already registered in DuckDB as their file names.
3. For Pandas: DataFrames are pre-loaded as: sales_df, targets_df.
4. Handle NULL values explicitly.
5. Return results as a table (not scalar unless query asks for single value).
6. If joining tables, always specify join keys explicitly.

Output format:
{
  "sql": "<sql query or null if not possible>",
  "pandas": "<pandas code string>",
  "self_confidence": 0.0-1.0,
  "reasoning": "<brief chain of thought>"
}
```

**Why two tracks?**  
SQL is more constrained (easier to validate, less injection surface, declarative).  
Pandas handles edge cases SQL can't cleanly express (rolling windows, complex string ops, iterative logic).

---

### 3.6 SafeExecutor

Runs generated code in a controlled environment.

```python
class SafeExecutor:
    TIMEOUT_SECONDS = 30
    MAX_RESULT_ROWS = 10_000
    BANNED_TOKENS = ["import os", "subprocess", "eval(", "exec(", 
                     "__import__", "open(", "sys."]
    
    def execute(self, code: GeneratedCode, context: ExecutionContext) -> ExecutionResult:
        # 1. Static safety check
        self._static_check(code)
        
        # 2. Try SQL first
        if code.sql:
            try:
                result = self._run_sql(code.sql, context.duckdb_conn)
                return ExecutionResult(result=result, track="SQL", success=True)
            except Exception as e:
                self._log_failure("SQL", e, code.sql)
        
        # 3. Fallback to Pandas
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
            future = pool.submit(exec, code, {"__builtins__": {}}, local_ns)
            future.result(timeout=self.TIMEOUT_SECONDS)
        return local_ns.get("result")  # Code must assign output to `result`
    
    def _static_check(self, code: GeneratedCode):
        combined = f"{code.sql or ''} {code.pandas or ''}"
        for token in self.BANNED_TOKENS:
            if token in combined:
                raise SecurityError(f"Banned token detected: {token}")
```

---

### 3.7 ResultValidator

Catches "technically ran, logically wrong" failures.

```python
class ResultValidator:
    def validate(self, result: pd.DataFrame, intent: QueryIntent) -> ValidationReport:
        checks = [
            self._check_not_empty(result, intent),
            self._check_expected_columns(result, intent),
            self._check_numeric_sanity(result, intent),
            self._check_aggregation_consistency(result, intent),
            self._check_top_n_count(result, intent),
        ]
        passed = sum(c.passed for c in checks)
        return ValidationReport(
            checks=checks,
            pass_rate=passed / len(checks),
            issues=[c.message for c in checks if not c.passed]
        )
    
    def _check_not_empty(self, df, intent) -> Check:
        # Empty is suspicious for most queries unless intent is "count of X" = 0
        if len(df) == 0 and intent.primary_operation != "COUNT_ZERO":
            return Check(passed=False, message="Result is empty — possible filter overreach")
        return Check(passed=True)
    
    def _check_top_n_count(self, df, intent) -> Check:
        # If query says "top 5", result should have ≤ 5 rows (after grouping)
        if intent.n_value and len(df) > intent.n_value:
            return Check(passed=False, 
                         message=f"Expected ≤{intent.n_value} rows, got {len(df)}")
        return Check(passed=True)
    
    def _check_numeric_sanity(self, df, intent) -> Check:
        # Contribution % columns should sum to ~100 if intent is PCT_CONTRIBUTION
        if "PCT_CONTRIBUTION" in intent.secondary_operations:
            pct_cols = [c for c in df.columns if "%" in c or "pct" in c.lower()]
            for col in pct_cols:
                total = df[col].sum()
                if not (95 <= total <= 105):
                    return Check(passed=False, 
                                 message=f"Column {col} sums to {total:.1f}%, expected ~100%")
        return Check(passed=True)
```

---

### 3.8 ConfidenceScorer

**Every query gets a composite score from five independent signals.**

```
confidence = Σ(signal_i × weight_i)   where Σ weights = 1.0
```

| Signal | Weight | What It Measures |
|--------|--------|------------------|
| `schema_match` | 0.25 | How well entities in the intent match actual columns |
| `code_validity` | 0.20 | Did the code parse/lint without errors? |
| `execution_success` | 0.25 | Did both/either execution track succeed? |
| `result_validation` | 0.20 | Pass rate from ResultValidator checks |
| `llm_self_score` | 0.10 | The model's own `self_confidence` in the generation step |

```python
@dataclass
class ConfidenceSignal:
    schema_match:       float  # 0-1: fraction of intent entities found in schema
    code_validity:      float  # 1.0 if no syntax errors, else 0.0
    execution_success:  float  # 1.0=SQL, 0.7=Pandas fallback, 0.0=both failed
    result_validation:  float  # pass_rate from ResultValidator
    llm_self_score:     float  # model's reported confidence

WEIGHTS = [0.25, 0.20, 0.25, 0.20, 0.10]

class ConfidenceScorer:
    def score(self, signals: ConfidenceSignal) -> float:
        values = [
            signals.schema_match,
            signals.code_validity,
            signals.execution_success,
            signals.result_validation,
            signals.llm_self_score,
        ]
        raw = sum(v * w for v, w in zip(values, WEIGHTS))
        
        # Apply penalties for critical failures
        if signals.execution_success == 0.0:
            raw = min(raw, 0.15)   # Cannot be confident if nothing executed
        if signals.schema_match < 0.5:
            raw = min(raw, 0.30)   # Schema mismatch is a hard penalty
        
        return round(raw, 3)
```

**Feedback boost:** If a semantically similar past query had a correct result,
apply a `+0.05` feedback bonus (capped at 0.95).

---

### 3.9 ExplanationGenerator (LLM — Final Call)

```python
EXPLANATION_PROMPT = """
You are explaining an analytics query result to a business user.

ORIGINAL QUESTION: "{query}"
WHAT THE SYSTEM UNDERSTOOD:
  - Primary operation: {intent.primary_operation}
  - Tables used: {intent.tables_needed}
  - Filters applied: {intent.entities}
  - Time range: {intent.time_constraint}

GENERATED CODE ({track} track):
{generated_logic}

RESULT PREVIEW (first 5 rows):
{result_preview}

CONFIDENCE SCORE: {score} ({score_label})

Write a 3-part explanation:
1. UNDERSTOOD: What did the system interpret from the question?
2. APPROACH: How was the result computed (in plain English, no code)?
3. CAVEATS: Any assumptions made or things to watch out for?

Keep it concise. Business user audience. No jargon.
"""
```

---

### 3.10 FeedbackLogger

Persists every query's outcome for the feedback loop.

```python
@dataclass
class FeedbackEntry:
    timestamp:        str
    query:            str
    generated_logic:  str    # The winning SQL or Pandas code
    result_preview:   str    # JSON of first 10 rows
    confidence_score: float
    track_used:       str    # "SQL" or "Pandas"
    intent_json:      str    # Serialized QueryIntent
    feedback:         str    # "pending" | "correct" | "incorrect"
    feedback_note:    str    # Optional human correction
```

The `feedback` column is initially `"pending"`. A simple CLI command lets a
human reviewer mark entries as `"correct"` or `"incorrect"` with an optional note.
These notes are used as negative examples in future CodeGenerator prompts.

---

## 4. Data Flow — Request Lifecycle

```
USER submits NL query: "Which sales rep had the highest revenue in Q1 in the North region?"

Step 1 — QueryPreprocessor
  IN:  raw string
  OUT: normalized="which sales rep had the highest revenue in q1 in the north region"
       complexity=COMPLEX (contains "highest" + time + geo filter)

Step 2 — DataContextBuilder
  IN:  CSV paths, data_dictionary.json
  OUT: schema_digest (injected into all subsequent LLM calls)
       DuckDB connection with sales_data + targets loaded

Step 3 — IntentClassifier  [LLM CALL #1 — ~200 tokens]
  IN:  preprocessed query + schema_digest
  OUT: QueryIntent {
         primary_operation: "RANK",
         secondary_operations: ["FILTER", "TOP_1"],
         entities: {"sales_rep": "*", "region": "North"},
         time_constraint: "Q1 2024",
         tables_needed: ["sales_data"],
         n_value: 1
       }

Step 4 — FeedbackRetriever
  IN:  preprocessed query
  OUT: [FeedbackEntry(query="top rep in East Q2", logic="SELECT...", was_correct=True, sim=0.73)]
       → This past query becomes a few-shot example

Step 5 — CodeGenerator  [LLM CALL #2 — ~800 tokens]
  IN:  intent + schema_digest + few-shot example
  OUT: {
    "sql": "SELECT sales_rep, SUM(sale_amount) AS total_revenue
            FROM sales_data
            WHERE region = 'North'
              AND sale_date BETWEEN '2024-01-01' AND '2024-03-31'
            GROUP BY sales_rep
            ORDER BY total_revenue DESC
            LIMIT 1",
    "pandas": "result = (sales_df[...]...)",
    "self_confidence": 0.92,
    "reasoning": "Filter by region and date, aggregate by rep, rank by revenue, take top 1"
  }

Step 6 — SafeExecutor
  IN:  generated SQL
  OUT: DataFrame:
       ┌─────────────┬───────────────┐
       │ sales_rep   │ total_revenue │
       ├─────────────┼───────────────┤
       │ Alice Chen  │ 187,432.00    │
       └─────────────┴───────────────┘
       track_used="SQL", success=True

Step 7 — ResultValidator
  IN:  result DataFrame + intent
  OUT: ValidationReport {
         checks: [not_empty✓, top_n_count✓(1≤1), numeric_sanity✓],
         pass_rate: 1.0
       }

Step 8 — ConfidenceScorer
  IN:  signals { schema_match=0.95, code_validity=1.0,
                 execution_success=1.0, result_validation=1.0, llm_self_score=0.92 }
  OUT: confidence = 0.977 (+0.05 feedback bonus → capped at 0.95)
       final_score = 0.95

Step 9 — ExplanationGenerator  [LLM CALL #3 — ~400 tokens]
  IN:  intent + code + result + confidence
  OUT: {
    "understood": "You asked for the single best-performing sales representative by 
                   total revenue, restricted to the North region during Q1 (Jan–Mar 2024).",
    "approach":   "Sales transactions in the North region for Q1 were summed per 
                   representative, then ranked by total revenue descending. The top row was returned.",
    "caveats":    "Only closed/recorded sales are included. Assumed Q1 = calendar Q1 2024."
  }

Step 10 — FeedbackLogger
  IN:  complete result object
  OUT: Appended row to feedback_log.csv with status="pending"

FINAL OUTPUT:
{
  "query": "Which sales rep had the highest revenue in Q1 in the North region?",
  "generated_logic": "SELECT sales_rep, SUM(sale_amount) AS total_revenue\nFROM sales_data\nWHERE region = 'North'\n  AND sale_date BETWEEN '2024-01-01' AND '2024-03-31'\nGROUP BY sales_rep\nORDER BY total_revenue DESC\nLIMIT 1",
  "result": [{"sales_rep": "Alice Chen", "total_revenue": 187432.00}],
  "confidence_score": 0.95,
  "explanation": {
    "understood": "...",
    "approach": "...",
    "caveats": "..."
  },
  "metadata": {
    "track_used": "SQL",
    "intent": {...},
    "processing_time_ms": 2341
  }
}
```

---

## 5. Tech Stack & Justification

| Layer | Choice | Why |
|-------|--------|-----|
| **LLM Engine** | `inclusionai/ling-3.0-flash-vl:free` via OpenRouter (OpenAI SDK) | High-speed multi-modal LLM; standard OpenAI SDK client interface; easily swappable to GPT-4o or Claude |
| **SQL Engine** | DuckDB (in-process) | Zero-infra, reads CSVs natively, full analytical SQL (window functions, CTEs), blazing fast |
| **DataFrames** | Pandas | Universal; fallback execution; output shape formatting & CSV exporting |
| **Semantic Search** | `sklearn.TfidfVectorizer` + cosine | Lightweight, zero-external-service semantic feedback retrieval; fast in-memory indexing |
| **Data Validation** | Pydantic v2 | All inter-component contracts enforced at runtime; strict JSON serialization |
| **Sandboxing** | `concurrent.futures` + Restricted builtins | 30s timeout enforcement + restricted exec namespace blocking dangerous system calls |
| **CLI** | Typer + Rich | Beautiful terminal output; easy to batch-process `nl_queries.json` and review feedback |
| **Web Dashboard** | FastAPI + Vue 3 (CDN) + Vanilla CSS | Zero-build, responsive web dashboard styled after `innovationalofficesolution.com` |
| **Config** | Pydantic Settings + `.env` | API keys, thresholds, port configuration — no hardcoded values |
| **Testing** | Pytest (106 tests) | 96 fast unit tests + 10 live LLM end-to-end integration tests |
| **Logging** | Structured feedback logger | CSV-persisted query telemetry with deterministic row index tracking for feedback |

**Deliberately NOT used:**
- LangChain / LlamaIndex — over-abstracted; hides prompt logic that matters here
- Vector DBs (Pinecone/Chroma/Qdrant) — overkill for feedback retrieval at this scale; TF-IDF is instant and runs locally without network overhead
- Heavy Node/Webpack build pipelines — Vue 3 via CDN and Vanilla CSS eliminate npm dependency bloat while providing a rich reactive UI
- Docker — adds unnecessary virtualization overhead for a lightweight embedded DuckDB setup

---

## 6. Prompt Engineering Strategy

### 6.1 The Three-LLM-Call Architecture

```
Call 1 (IntentClassifier):   ~200 input + ~150 output tokens  — cheap, fast
Call 2 (CodeGenerator):      ~800 input + ~400 output tokens  — main cost
Call 3 (ExplanationGenerator):~600 input + ~300 output tokens  — human-facing
```

**Total per query: ~2,450 tokens ≈ ~$0.002 at Sonnet pricing**

### 6.2 Schema Injection Always On

The schema digest is **always** in the system prompt of all three calls.
This prevents hallucinated column names — the #1 failure mode in NL-to-SQL.

```
System prompt fragment:
"You have access ONLY to these tables and columns: [schema_digest].
 Never reference a column that is not listed above.
 If the query cannot be answered with available data, say so explicitly
 rather than inventing a column."
```

### 6.3 Chain-of-Thought for COMPLEX queries

When `complexity == COMPLEX`, the CodeGenerator prompt includes:

```
Before writing any code, think through these steps:
<thinking>
1. What is the "grain" of the output? (one row = one what?)
2. What filtering must happen first?
3. Is a subquery or CTE needed?
4. What aggregation comes after filtering?
5. What post-aggregation filter is needed (HAVING)?
6. What ordering / limiting?
</thinking>

Then write the SQL.
```

This dramatically improves accuracy on "top N per group" and
"contribution percentage" queries that require window functions or CTEs.

### 6.4 Negative Examples from Feedback

When a past incorrect answer has a feedback note:
```
PREVIOUS ATTEMPT (DO NOT REPEAT):
  Query: "sales by region Q1"
  Wrong code: SELECT region, SUM(revenue) ... 
  ← Note: column is 'sale_amount', not 'revenue'
  Correct approach: use sale_amount
```

### 6.5 Output Format Enforcement

All LLM calls use `response_format: json_object` (or equivalent structured output).
The system prompt specifies the exact JSON schema the model must return.
Pydantic validates every response — if validation fails, we retry once with the
validation error appended.

---

## 7. Confidence Scoring System

### Score Bands

| Score | Label | Interpretation |
|-------|-------|----------------|
| 0.85 – 1.00 | ✅ High | Trust the result; safe to use |
| 0.65 – 0.84 | 🟡 Medium | Likely correct; verify for decisions |
| 0.40 – 0.64 | 🟠 Low | Partial success; review the generated code |
| 0.00 – 0.39 | 🔴 Unreliable | Do not use; likely misunderstood query |

### Score Computation Detail

```python
def compute_confidence(
    intent:          QueryIntent,
    generated_code:  GeneratedCode,
    execution:       ExecutionResult,
    validation:      ValidationReport,
    feedback_boost:  float,
) -> float:

    # Signal 1: Schema match
    # How many entities in intent are actual column names?
    entity_hits = sum(1 for v in intent.entities.values() 
                      if v in schema.all_columns)
    schema_match = entity_hits / max(len(intent.entities), 1)
    
    # Signal 2: Code validity
    code_validity = 0.0 if generated_code.had_syntax_error else 1.0
    
    # Signal 3: Execution success
    exec_map = {"SQL": 1.0, "Pandas": 0.7, "Both_failed": 0.0}
    execution_score = exec_map[execution.track or "Both_failed"]
    
    # Signal 4: Result validation
    result_score = validation.pass_rate
    
    # Signal 5: LLM self-confidence
    llm_score = generated_code.self_confidence
    
    weights = [0.25, 0.20, 0.25, 0.20, 0.10]
    signals = [schema_match, code_validity, execution_score, result_score, llm_score]
    raw = sum(s * w for s, w in zip(signals, weights))
    
    # Hard penalties
    if execution_score == 0.0: raw = min(raw, 0.15)
    if schema_match   < 0.50:  raw = min(raw, 0.30)
    if len(intent.ambiguities) > 2: raw *= 0.85
    
    # Feedback boost
    raw = min(raw + feedback_boost, 0.97)
    
    return round(raw, 3)
```

---

## 8. Feedback Loop Design

### 8.1 Feedback Sources

```
feedback_log.csv schema:
┌─────────────────┬──────────────────┬──────────┬──────────────────┬──────────┐
│ query           │ generated_logic  │ confidence│ feedback        │ note     │
├─────────────────┼──────────────────┼──────────┼──────────────────┼──────────┤
│ "top 5 reps..." │ "SELECT..."      │ 0.91     │ correct          │ ""       │
│ "avg sale Q2.." │ "result = df..." │ 0.63     │ incorrect        │ "wrong   │
│                 │                  │          │                  │  filter" │
│ "new query..."  │ "SELECT..."      │ 0.88     │ pending          │ ""       │
└─────────────────┴──────────────────┴──────────┴──────────────────┴──────────┘
```

### 8.2 How Feedback Improves Future Queries

```
                 ┌────────────────────────────────┐
                 │      FEEDBACK LOOP ENGINE       │
                 │                                 │
  New Query ────▶│  1. Embed query (TF-IDF)        │
                 │  2. Find top-K similar past Q   │
                 │  3. Classify each past Q:       │
                 │     • feedback="correct"        │──▶ Few-shot POSITIVE example
                 │     • feedback="incorrect"      │──▶ Few-shot NEGATIVE example
                 │     • feedback="pending"        │──▶ Ignored                   
                 │  4. Score similarity:           │
                 │     > 0.85 → strong match       │──▶ Confidence boost +0.05
                 │     0.6-0.85 → weak match       │──▶ Confidence boost +0.02
                 │     < 0.6  → no match           │──▶ No boost
                 └────────────────────────────────┘
```

### 8.3 Cold Start Handling

When `feedback_log.csv` is absent or empty:
- System works without feedback (degrades gracefully)
- Logs all results immediately
- After 10+ queries are marked, feedback retrieval activates
- The `nl_queries.json` provided in the dataset can be **pre-seeded** as
  training examples if they include expected outputs

---

## 9. Error Recovery Strategy

### Decision Tree

```
                     ┌─────────────────────┐
                     │  Execute SQL query   │
                     └──────────┬──────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │         Success?                   │
              └────────────┬────────────┬──────────┘
                         YES           NO
                           │           │
                      Return result   ┌▼─────────────────────────────┐
                                      │     What kind of failure?     │
                                      └──┬───────────────────────┬────┘
                                 Syntax Error              Runtime Error
                                       │                        │
                    ┌──────────────────▼──┐         ┌──────────▼────────────┐
                    │ Retry SQL with      │         │ Try Pandas fallback   │
                    │ error appended      │         └──────────┬────────────┘
                    │ to prompt           │                    │
                    └──────────────────┬──┘               Success?
                                       │              ┌──────┴──────┐
                                  Success?           YES            NO
                               ┌────┴─────┐          │             │
                              YES        NO      Return with    ┌──▼──────────────┐
                               │          │      lower conf     │ Return with     │
                          Return result  Try                    │ conf=0.1 +      │
                                         Pandas                │ explain failure  │
                                                               └─────────────────┘
```

**Retry budget:** Max 2 retries per track, 1 cross-track fallback.
After 3 total attempts, return a failure response with confidence=0.0
and the best partial result if available.

### Common Failure Patterns & Mitigations

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Column not found | LLM invented a column | Schema digest in prompt; validate against schema before exec |
| Empty result | Over-filtered | Validator flags it; mention in explanation |
| Wrong aggregation | Ambiguous "total" | Disambiguate in explanation; low confidence |
| Division by zero | Pct contribution on zero | Pandas `.fillna(0)` guard in Pandas track |
| Timeout | Complex nested query | 30s timeout; suggest query simplification |
| Type mismatch | DATE vs. TEXT | schema digest includes inferred types |

---

## 10. Repository Layout

```
Office Solutions AI Lab - Assignment/
├── README.md                      ← Top-level landing page with quickstart & badges
├── SYSTEM_DESIGN.md               ← This architectural blueprint document
├── USER_GUIDE.md                  ← Operator, runner, and modification guide
├── CODEBASE_ANALYSIS.md           ← Deep architectural and function-by-function analysis
├── REPO_PUBLISHING_GUIDE.md       ← Git hygiene rules and repository publishing guide
├── requirements.txt               ← Frozen Python dependencies
├── .env.example                   ← Environment configuration template
├── .gitignore                     ← Git exclusion rules
│
├── dataset/                       ← Test dataset (fixed, unedited)
│   ├── sales_data.csv             ← Primary transactions table
│   ├── targets.csv                ← Regional monthly targets table
│   ├── data_dictionary.json       ← Column schemas and metrics definitions
│   └── nl_queries.json            ← 8 canonical evaluation benchmark queries
│
├── feedback_log.csv               ← Persisted query telemetry & few-shot memory
│
├── IAQE/                          ← Core package
│   ├── __init__.py
│   ├── engine.py                  ← QueryEngine: orchestrates all 10 steps + self-healing retry
│   │
│   └── pipeline/                  ← Modular pipeline components
│       ├── __init__.py
│       ├── preprocessor.py        ← Step 1: Normalization, domain spell-check, complexity
│       ├── context_builder.py     ← Step 2: CSV ingestion, DuckDB in-memory table registration
│       ├── intent.py              ← Step 3: LLM Intent classification & entity extraction
│       ├── feedback.py            ← Step 4 & 10: TF-IDF few-shot retrieval & feedback logging
│       ├── generator.py           ← Step 5: Dual-track SQL & Pandas code generation
│       ├── executor.py            ← Step 6: SafeExecutor (DuckDB -> Pandas fallback, sandbox)
│       ├── validator.py           ← Step 7: ResultValidator (5 deterministic sanity checks)
│       ├── scorer.py              ← Step 8: ConfidenceScorer (5-signal composite matrix)
│       └── explainer.py           ← Step 9: ExplanationGenerator (3-part plain English format)
│
├── cli.py                         ← Typer CLI (single query, batch, stats, feedback review)
├── server.py                      ← FastAPI backend server for web dashboard
│
├── static/                        ← Zero-build Web Dashboard (Vue 3 CDN + Vanilla CSS)
│   ├── index.html                 ← Dashboard markup matching innovationalofficesolution.com
│   ├── style.css                  ← Enterprise brand theme (#1A1D1C, #FFB401, #016064)
│   └── app.js                     ← Vue 3 reactive controller & client telemetry
│
└── tests/                         ← 106 automated tests (100% pass rate)
    ├── unit/
    │   ├── test_preprocessor.py   ← 21 unit tests: normalization, domain vocab, complexity
    │   ├── test_scorer.py         ← 16 unit tests: signal weights, hard penalties, boosts
    │   ├── test_validator.py      ← 19 unit tests: all 5 sanity checks & contract rules
    │   ├── test_context_builder.py← 14 unit tests: DuckDB in-memory registration & schemas
    │   ├── test_executor.py       ← 13 unit tests: SQL/Pandas execution & security sandbox
    │   └── test_feedback.py       ← 13 unit tests: TF-IDF cosine retrieval & persistence
    └── integration/
        └── test_pipeline.py       ← 10 live LLM integration tests end-to-end
```

---

## 11. Sample Output

### Query: "What is each region's contribution to total revenue?"

```json
{
  "query": "What is each region's contribution to total revenue?",
  "generated_logic": "SELECT\n  region,\n  SUM(sale_amount) AS regional_revenue,\n  ROUND(100.0 * SUM(sale_amount) / SUM(SUM(sale_amount)) OVER (), 2) AS pct_of_total\nFROM sales_data\nGROUP BY region\nORDER BY regional_revenue DESC",
  "result": [
    {"region": "North", "regional_revenue": 542300.00, "pct_of_total": 34.12},
    {"region": "South", "regional_revenue": 483100.00, "pct_of_total": 30.40},
    {"region": "West",  "regional_revenue": 310200.00, "pct_of_total": 19.52},
    {"region": "East",  "regional_revenue": 253400.00, "pct_of_total": 15.96}
  ],
  "confidence_score": 0.93,
  "explanation": {
    "understood": "You asked for a breakdown of how much each geographic region contributes to the company's total sales revenue, expressed as a percentage.",
    "approach": "Sales amounts were summed by region, then each region's sum was divided by the grand total using a SQL window function to compute the percentage share. Results are sorted from highest to lowest contributor.",
    "caveats": "Percentages are based on total recorded sales with no date filter applied — this covers the full dataset time range. If a specific period is intended, please re-query with a date range."
  },
  "metadata": {
    "track_used": "SQL",
    "complexity": "MEDIUM",
    "intent": {
      "primary_operation": "AGGREGATE",
      "secondary_operations": ["GROUP_BY", "PCT_CONTRIBUTION"],
      "tables_needed": ["sales_data"]
    },
    "processing_time_ms": 1872,
    "llm_calls": 3
  }
}
```

---

### Query: "Which rep is underperforming vs target by the largest margin in Q3?"

```json
{
  "query": "Which rep is underperforming vs target by the largest margin in Q3?",
  "generated_logic": "WITH rep_actuals AS (\n  SELECT sales_rep,\n         SUM(sale_amount) AS actual\n  FROM sales_data\n  WHERE sale_date BETWEEN '2024-07-01' AND '2024-09-30'\n  GROUP BY sales_rep\n),\nrep_targets AS (\n  SELECT sales_rep, SUM(target_amount) AS target\n  FROM targets\n  WHERE quarter = 'Q3'\n  GROUP BY sales_rep\n)\nSELECT r.sales_rep,\n       r.actual,\n       t.target,\n       (r.actual - t.target) AS gap,\n       ROUND(100.0*(r.actual - t.target)/t.target, 1) AS pct_gap\nFROM rep_actuals r\nJOIN rep_targets t USING (sales_rep)\nWHERE r.actual < t.target\nORDER BY gap ASC\nLIMIT 1",
  "result": [
    {"sales_rep": "Bob Martin", "actual": 43200.00, "target": 90000.00, "gap": -46800.00, "pct_gap": -52.0}
  ],
  "confidence_score": 0.88,
  "explanation": {
    "understood": "You want to find the single sales representative who fell furthest below their assigned Q3 target, measured in absolute revenue gap.",
    "approach": "Q3 actual sales were aggregated per representative, then joined with the targets table on rep name and quarter. The gap (actual minus target) was computed, filtered to only underperformers (negative gap), and sorted to find the largest shortfall.",
    "caveats": "Assumed join between sales_data and targets is on 'sales_rep' column — verify if a rep_id key exists. Q3 defined as July 1 – September 30, 2024."
  },
  "confidence_score": 0.88,
  "metadata": {
    "track_used": "SQL",
    "complexity": "COMPLEX",
    "processing_time_ms": 3104
  }
}
```

---

## 12. Design Trade-offs

### Decisions Made & Why

| Decision | Alternative Considered | Why This Choice Won |
|----------|----------------------|---------------------|
| Three separate LLM calls | Single mega-call | Separation of concerns; each call is testable, cacheable, retryable independently |
| DuckDB (not Pandas-only) | Pandas-only | SQL is declarative, easier to validate, window functions are cleaner |
| TF-IDF for feedback retrieval | Dense embeddings (e.g., `all-MiniLM`) | No external model load at startup; good enough at this scale |
| No LangChain | LangChain/LlamaIndex | Prompt logic is the core IP here — hiding it in abstractions hurts debuggability |
| Pydantic for all contracts | Plain dicts | Validation at every boundary; type safety catches bugs early |
| No streaming | SSE streaming output | Batch use case; correctness > latency |
| CSV logging | SQLite / Postgres | Zero infrastructure; feedback_log.csv readable in Excel/Sheets |

### Known Trade-offs

| Trade-off | Impact | Mitigation |
|-----------|--------|------------|
| ~3 LLM calls per query = ~3-5 seconds latency | Slower than single-call | Acceptable for analytics workload; IntentClassifier call is cacheable |
| TF-IDF misses semantic similarity ("revenue" ≠ "sales") | Feedback retrieval misses | Upgrade path: swap in `sentence-transformers` with minimal code change |
| Pandas exec with `__builtins__={}` can still run slow loops | DoS potential | 30s hard timeout + row-count cap |
| Confidence score is heuristic, not calibrated | Score may not match probability | Add calibration dataset after 100+ labeled queries |

---

## 13. Scalability Path

### Current Design Target
- Single machine, CSV-sized datasets (< 10M rows)
- Batch processing of `nl_queries.json`
- Single user CLI

### Scale-Up Path (without redesign)

| Bottleneck | Solution at Scale |
|------------|-------------------|
| DuckDB in-process | Move to DuckDB-WASM or MotherDuck (cloud DuckDB); same SQL dialect |
| TF-IDF feedback search | Replace with FAISS + `sentence-transformers`; plug-in interface unchanged |
| CSV feedback log | Migrate to SQLite or Postgres; change only `FeedbackLogger.persist()` |
| LLM API latency | Add async batch processing; cache IntentClassifier results for repeated queries |
| Large datasets | Partition CSVs by date; DuckDB's Parquet support + predicate pushdown |

### API Mode (optional)

The `QueryEngine` class is pure Python with no I/O coupling.
Wrap with FastAPI in < 50 lines:

```python
@app.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    return await engine.run(request.query)
```

---

## 14. If Given More Time & Implemented Milestones

### Completed Milestones (Ahead of Schedule)
- [x] **Comprehensive Test Suites**: Implemented 106 tests (96 unit tests covering preprocessor, scorer, validator, executor, feedback + 10 live LLM integration tests with 100% pass rate).
- [x] **Interactive Human Feedback UI**: Built into both the CLI (`cli.py review` / `cli.py feedback-mark`) and the live Web Dashboard.
- [x] **Full-Featured Web Dashboard**: Implemented zero-build reactive Web Dashboard (FastAPI + Vue 3) styled after `innovationalofficesolution.com`.
- [x] **Self-Healing Execution Retry**: Automatic reflection loop re-prompting the code generator with runtime executor errors.
- [x] **Robust Data Ingestion**: Custom resilient parser (`_load_json_robust`) for escaped quotes in recruiter test files.

### Next Roadmap Iterations
**Priority 1 — Advanced Modeling**
- [ ] Calibrate confidence scores against labeled dataset (Platt scaling / temperature scaling)
- [ ] Add multi-agent query decomposition: break complex multi-part questions into sub-DAGs
- [ ] Dynamic relative date grounding ("last quarter", "QTD", "prior month") relative to dataset date ceiling

**Priority 2 — Feedback & Scalability**
- [ ] Upgrade feedback retrieval from TF-IDF to dense vector store (FAISS + sentence-transformers) for very large feedback bases
- [ ] Auto-promote high-confidence queries (confidence > 0.94) to active few-shot exemplar pools
- [ ] Semantic query caching per `(query_embedding, schema_hash)` to cut LLM latency and cost by ~40%

---

## Appendix A — Handling Each Query Type

| Query Pattern | Example | Strategy |
|---------------|---------|----------|
| Simple aggregate | "Total revenue" | Single SELECT SUM |
| Grouped aggregate | "Revenue by region" | GROUP BY |
| Filtered | "Sales in Q1" | WHERE + DATE range |
| Top N | "Top 5 reps" | ORDER BY + LIMIT |
| Top N per group | "Top rep per region" | PARTITION BY window function |
| Contribution % | "Each region's share" | SUM() OVER() window |
| Comparison vs target | "Who missed target?" | JOIN + gap calc |
| Time trend | "Monthly revenue trend" | DATE_TRUNC + GROUP BY |
| YoY comparison | "Q1 2024 vs Q1 2023" | Self-join or CASE WHEN |
| Nested / multi-step | "Reps below avg in worst region" | CTE pipeline |

---

## Appendix B — CLI Reference

```bash
# Single query
python cli.py query "Which product category had highest margin in Q2?"

# Batch process all queries from nl_queries.json
python cli.py batch dataset/nl_queries.json --output outputs/results.json

# Review pending feedback entries
python cli.py review

# Mark entry as correct/incorrect
python cli.py feedback mark <id> --correct --note "Verified in BI tool"

# Show confidence distribution across a batch run
python cli.py stats outputs/results.json
```

---

*"The measure of a good analytics system is not whether it answers the easy questions —
it's whether it knows when it doesn't know the answer."*

