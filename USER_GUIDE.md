# Intelligent Analytic Query Engine (IAQE) — User & Developer Guide

Welcome to the **Intelligent Analytic Query Engine (IAQE)**! This guide provides end-to-end instructions on how to set up, operate, test, and customize the system via both the Command-Line Interface (CLI) and the interactive Web Dashboard.

---

## Table of Contents

1. [System Overview & Prerequisites](#1-system-overview--prerequisites)
2. [Quickstart & Environment Setup](#2-quickstart--environment-setup)
3. [Running via Command-Line Interface (CLI)](#3-running-via-command-line-interface-cli)
   - [3.1 Single Query Execution](#31-single-query-execution)
   - [3.2 Batch Processing](#32-batch-processing)
   - [3.3 Human Feedback Review & Labeling](#33-human-feedback-review--labeling)
   - [3.4 Batch Statistics & Confidence Distribution](#34-batch-statistics--confidence-distribution)
4. [Running the Web Dashboard](#4-running-the-web-dashboard)
   - [4.1 Starting the Server](#41-starting-the-server)
   - [4.2 Operating Dashboard Features](#42-operating-dashboard-features)
5. [Running Test Suites](#5-running-test-suites)
   - [5.1 Unit Tests (Pure Python / Offline)](#51-unit-tests-pure-python--offline)
   - [5.2 Live Integration Tests (Live LLM)](#52-live-integration-tests-live-llm)
6. [How to Modify & Extend the System](#6-how-to-modify--extend-the-system)
   - [6.1 Adding New Datasets & CSV Tables](#61-adding-new-datasets--csv-tables)
   - [6.2 Extending the Business Glossary & Metrics](#62-extending-the-business-glossary--metrics)
   - [6.3 Tuning the Confidence Scorer Matrix](#63-tuning-the-confidence-scorer-matrix)
   - [6.4 Customizing LLM Prompts](#64-customizing-llm-prompts)
   - [6.5 Swapping LLM Providers or Models](#65-swapping-llm-providers-or-models)
   - [6.6 Customizing Web Dashboard Styling & Logic](#66-customizing-web-dashboard-styling--logic)

---

## 1. System Overview & Prerequisites

IAQE translates natural language business questions into verified SQL (DuckDB) and Pandas execution logic, scores query reliability via a 5-signal composite confidence matrix, produces 3-part plain-English explanations, and feeds back verified queries for continuous few-shot reinforcement.

### Prerequisites
- **Python 3.10+** (tested on Python 3.10 and 3.11).
- **OpenRouter API Key** (or direct OpenAI API key) with access to `inclusionai/ling-3.0-flash-vl:free` or equivalent model.
- Windows, macOS, or Linux operating system.

---

## 2. Quickstart & Environment Setup

### Step 1: Clone or Navigate to Project Root
```bash
cd "Office Solutions AI Lab - Assignment"
```

### Step 2: Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and configure your API key:
```ini
OPEN_ROUTER_KEY=sk-or-v1-your-openrouter-key-here
DEFAULT_MODEL=inclusionai/ling-3.0-flash-vl:free
PORT=8000
```

---

## 3. Running via Command-Line Interface (CLI)

The engine features a full-featured CLI built using **Typer** and **Rich** in `cli.py`.

### 3.1 Single Query Execution
Execute any ad-hoc natural language question directly in your terminal:

```bash
python cli.py query "Total sales in India for March"
```

**Sample Output:**
```text
╭─ Query ─────────────────────────────────────────────────────────────╮
│ Total sales in India for March                                      │
╰─────────────────────────────────────────────────────────────────────╯

Confidence: [HIGH] (0.96)
Track: SQL  |  Complexity: SIMPLE  |  Latency: 1,420 ms

── Generated Logic (SQL) ──────────────────────────────────────────────
SELECT SUM(revenue) AS total_sales
FROM sales_data
WHERE country = 'India' AND strftime('%Y-%m', order_date) = '2024-03';

── Result (1 row) ────────────────────────────────────────────────────
┏━━━━━━━━━━━━━┓
┃ total_sales ┃
┡━━━━━━━━━━━━━┩
│ 142500.00   │
└─────────────┘

── Explanation ───────────────────────────────────────────────────────
Understood: You asked for total revenue in India specifically for March 2024.
Approach:   Filtered sales_data where country is India and date matches March 2024, then aggregated revenue.
Caveats:    Assumes order_date follows ISO date format.
```

### 3.2 Batch Processing
Process an entire JSON file containing multiple natural language queries (such as `dataset/nl_queries.json`):

```bash
python cli.py batch dataset/nl_queries.json --output outputs/results.json
```
*Note:* The batch runner automatically handles both standard JSON and non-standard quoted JSON formats.

### 3.3 Human Feedback Review & Labeling
Inspect queries logged to `feedback_log.csv` and label pending queries:

**List pending queries:**
```bash
python cli.py review
```

**Mark a query as correct:**
```bash
python cli.py feedback-mark 0 --correct --note "Verified against BI dashboard"
```

**Mark a query as incorrect (acts as a negative guardrail):**
```bash
python cli.py feedback-mark 1 --incorrect --note "Incorrect column used: should be profit not revenue"
```

### 3.4 Batch Statistics & Confidence Distribution
Inspect the distribution of confidence bands across a completed batch run:

```bash
python cli.py stats outputs/results.json
```

---

## 4. Running the Web Dashboard

The application includes a zero-build, responsive web dashboard built with **FastAPI** (backend) and **Vue 3 + Vanilla CSS** (frontend) matching the **Office Solution AI Labs** enterprise aesthetic.

### 4.1 Starting the Server
Start the local web server:

```bash
python server.py
```
Or with Uvicorn live-reload:
```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Once started, open your browser at:
👉 **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

### 4.2 Operating Dashboard Features

1. **Natural Language Query Composer**:
   - Type any question or click any of the **Sample Enterprise Queries** chips (e.g. *Top 2 cities by profit*, *Average order value by region*, *Sales contribution % by category*).
   - Press <kbd>Ctrl</kbd> + <kbd>Enter</kbd> or click **Run Query**.

2. **Execution Track Overrides**:
   - **Auto (Dual-Track)**: Attempts DuckDB SQL first; falls back to sandboxed Pandas if SQL fails.
   - **SQL Only**: Restricts execution strictly to DuckDB SQL.
   - **Pandas Only**: Executes directly in the sandboxed Pandas DataFrame pipeline.

3. **Multi-Signal Metrics Row**:
   - **Confidence Score Pill**: Displays composite reliability percentage color-coded:
     - `High Confidence` ($\ge 0.85$): Emerald Green
     - `Medium Confidence` ($0.65 - 0.84$): Amber Gold
     - `Low Confidence` ($0.40 - 0.64$): Orange
     - `Unreliable` ($< 0.40$): Crimson Red
   - **Track Used Badge**: Shows whether SQL or Pandas succeeded.
   - **Complexity Badge**: Displays query difficulty (`SIMPLE`, `MEDIUM`, `COMPLEX`).
   - **Latency Badge**: Wall-clock execution time in seconds.

4. **3-Part Business Explanation Cards**:
   - **What Was Understood**: Plain-language translation of intent.
   - **Analytical Approach**: Steps and operations taken.
   - **Assumptions & Limitations**: Identified edge cases, time boundaries, or default joins.

5. **Interactive Data Table & Export**:
   - Clean tabular view of returned rows.
   - **Export to CSV**: Instant 1-click download of the result dataset as a CSV file.

6. **Generated Code Inspector**:
   - Monospace view of the generated SQL or Python code.
   - **Copy Code**: 1-click clipboard copy.

7. **Accuracy Feedback Loop**:
   - Click **Accurate** or **Inaccurate** to log feedback directly into `feedback_log.csv`.
   - Positive feedback grants confidence boosts to future similar queries; negative feedback creates few-shot negative guardrails.

---

## 5. Running Test Suites

The project features a comprehensive suite of **106 automated tests** (96 unit tests + 10 live LLM integration tests).

### 5.1 Unit Tests (Pure Python / Offline)
Unit tests run entirely in-memory with zero network calls, testing all mathematical models, regex preprocessors, validators, executor security sandbox, and duckdb loaders:

```bash
python -m pytest tests/unit/ -v
```
*(Expected: 96 passed in ~4–6 seconds)*

### 5.2 Live Integration Tests (Live LLM)
Integration tests run the full 10-step pipeline against the live OpenRouter model (`inclusionai/ling-3.0-flash-vl:free`):

```bash
python -m pytest tests/integration/ -v
```
*(Requires `OPEN_ROUTER_KEY` in `.env`. Expected: 10 passed in ~2–3 minutes)*

---

## 6. How to Modify & Extend the System

### 6.1 Adding New Datasets & CSV Tables
1. Place your new CSV file in `dataset/` (e.g. `dataset/inventory.csv`).
2. Update `DEFAULT_CSV_PATHS` in `cli.py` and `server.py`:
   ```python
   DEFAULT_CSV_PATHS = [
       "dataset/sales_data.csv",
       "dataset/targets.csv",
       "dataset/inventory.csv",
   ]
   ```
3. Update `dataset/data_dictionary.json` with the new table's column definitions and data types.
4. Restart `server.py`. The `DataContextBuilder` will automatically register the new table in DuckDB and inject its schema into the LLM system prompt.

### 6.2 Extending the Business Glossary & Metrics
To add custom enterprise business metrics (such as Gross Margin, Churn Rate, or LTV):
1. Open `IAQE/pipeline/context_builder.py`.
2. Locate `build_glossary()`:
   ```python
   glossary = {
       "revenue": "unit_price * quantity * (1 - discount)",
       "profit": "revenue - (unit_price * quantity * cost_pct) - shipping_cost",
       "aov": "Average Order Value = SUM(revenue) / COUNT(DISTINCT order_id)",
       # Add your custom business formula here:
       "gross_margin": "(revenue - cogs) / revenue * 100",
   }
   ```
3. The formula will be automatically injected into `CodeGenerator` prompts.

### 6.3 Tuning the Confidence Scorer Matrix
The confidence scorer weights are defined in `IAQE/pipeline/scorer.py`.
To alter the relative importance of schema matching vs execution vs validation:
```python
# Default weights:
self.weights = {
    "schema_match": 0.25,
    "code_validity": 0.20,
    "execution_success": 0.25,
    "result_validation": 0.20,
    "llm_self_score": 0.10,
}
```
You can also adjust hard failure penalties or the feedback boost:
```python
# In ConfidenceScorer.score():
if signals.execution_success == 0.0:
    raw_score = min(raw_score, 0.15)  # Cap score on runtime failure
```

### 6.4 Customizing LLM Prompts
Each LLM call's prompt template is encapsulated in its respective module:
- **Intent Classifier Prompt**: In `IAQE/pipeline/intent.py` (`IntentClassifier.classify`).
- **Code Generation Prompt**: In `IAQE/pipeline/generator.py` (`CodeGenerator.generate`). Controls the system prompt, DuckDB SQL dialect instructions, and few-shot formatting.
- **3-Part Explainer Prompt**: In `IAQE/pipeline/explainer.py` (`ExplanationGenerator.explain`). Governs how plain-English explanations are phrased for non-technical stakeholders.

### 6.5 Swapping LLM Providers or Models
The engine uses the standard `openai.OpenAI` client interface:
- To switch to **GPT-4o**, update `.env`:
  ```ini
  OPENAI_API_KEY=sk-...
  DEFAULT_MODEL=gpt-4o
  ```
  And in `IAQE/engine.py`, instantiate `OpenAI()` without `base_url`.
- To change OpenRouter model, simply update `DEFAULT_MODEL` in `.env`:
  ```ini
  DEFAULT_MODEL=anthropic/claude-3.5-sonnet
  ```

### 6.6 Customizing Web Dashboard Styling & Logic
- **Structure & Layout**: Edit `static/index.html`.
- **Styling & Color Themes**: Edit `static/style.css`. All colors, button variants, and fonts use standard CSS variables (`:root`) at the top of the file for instant theme re-coloring.
- **Client-Side Behavior**: Edit `static/app.js` (Vue 3 reactive methods for CSV export, keyboard shortcuts, and animations).
