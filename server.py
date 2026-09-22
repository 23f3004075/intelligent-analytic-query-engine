"""
Lightweight FastAPI Server for IAQE Web Dashboard.

Run with:
    python server.py
or:
    uvicorn server:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from IAQE.engine import QueryEngine
from cli import _load_json_robust, DEFAULT_CSV_PATHS, DEFAULT_DATA_DICT, DEFAULT_FEEDBACK_LOG

app = FastAPI(
    title="IAQE Dashboard API",
    description="Intelligent Analytic Query Engine REST API",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize QueryEngine singleton
print("[IAQE Server] Initializing QueryEngine and loading DuckDB tables...")
engine = QueryEngine(
    csv_paths=DEFAULT_CSV_PATHS,
    data_dict_path=DEFAULT_DATA_DICT,
    feedback_log_path=DEFAULT_FEEDBACK_LOG,
)
print("[IAQE Server] Engine initialized successfully.")

# Preload sample queries
SAMPLE_QUERIES_PATH = "dataset/nl_queries.json"
sample_queries = []
if Path(SAMPLE_QUERIES_PATH).exists():
    try:
        sample_queries = _load_json_robust(SAMPLE_QUERIES_PATH)
    except Exception as e:
        print(f"[IAQE Server] Warning: Could not load sample queries: {e}")


class QueryRequest(BaseModel):
    query: str
    track: Optional[str] = "auto"


class FeedbackRequest(BaseModel):
    entry_id: Optional[int] = None
    was_correct: bool = True
    note: Optional[str] = ""


@app.get("/api/info")
def get_info():
    """Return engine status, loaded tables, and column metadata."""
    tables_info = {}
    for name, df in engine.context_builder.dataframes.items():
        tables_info[name] = {
            "row_count": len(df),
            "columns": list(df.columns),
        }

    return {
        "status": "ready",
        "model": engine.model,
        "tables": tables_info,
        "column_count": len(engine.all_columns),
    }


@app.get("/api/samples")
def get_samples():
    """Return the pre-configured natural language sample queries."""
    return sample_queries


@app.post("/api/query")
def run_query(payload: QueryRequest):
    """Execute a natural language query through the 10-step IAQE pipeline."""
    q = payload.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = engine.run(query=q, track=payload.track or "auto")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
def submit_feedback(payload: FeedbackRequest):
    """Mark feedback for a past query run."""
    try:
        success = engine.mark_feedback(
            entry_id=payload.entry_id,
            was_correct=payload.was_correct,
            note=payload.note or "",
        )
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend static assets
STATIC_DIR = Path(__file__).parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "IAQE Server is running. Static index.html not yet found."}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[IAQE Server] Starting web dashboard at http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
