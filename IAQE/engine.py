"""
QueryEngine — Orchestrates all 10 pipeline steps.

This is the central coordinator that wires together:
  1. QueryPreprocessor
  2. DataContextBuilder  (schema digest built at init)
  3. IntentClassifier     (LLM Call #1)
  4. FeedbackRetriever
  5. CodeGenerator        (LLM Call #2)
  6. SafeExecutor         (DuckDB → Pandas fallback)
  7. ResultValidator
  8. ConfidenceScorer
  9. ExplanationGenerator (LLM Call #3)
 10. FeedbackLogger
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from IAQE.pipeline.preprocessor import QueryPreprocessor
from IAQE.pipeline.context_builder import DataContextBuilder
from IAQE.pipeline.intent import IntentClassifier, QueryIntent
from IAQE.pipeline.feedback import (
    FeedbackRetriever, FeedbackLogger, FeedbackEntry, RetrievedFeedback
)
from IAQE.pipeline.generator import CodeGenerator, GeneratedCode
from IAQE.pipeline.executor import SafeExecutor, ExecutionContext, ExecutionResult
from IAQE.pipeline.validator import ResultValidator, ValidationReport
from IAQE.pipeline.scorer import ConfidenceScorer, ConfidenceSignal
from IAQE.pipeline.explainer import ExplanationGenerator, Explanation

load_dotenv()

DEFAULT_MODEL = "inclusionai/ling-3.0-flash-vl:free"


class QueryEngine:
    """Orchestrates all pipeline steps to answer a natural language query."""

    def __init__(
        self,
        csv_paths: list,
        data_dict_path: str = None,
        feedback_log_path: str = "feedback_log.csv",
        model: str = DEFAULT_MODEL,
    ):
        # ── Shared LLM client ──
        api_key = os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPENAI_API_KEY") or "not_set"
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

        # ── Pipeline components ──
        self.preprocessor = QueryPreprocessor()

        self.context_builder = DataContextBuilder(csv_paths, data_dict_path)
        self.schema_digest = self.context_builder.build_schema_digest()
        self.glossary = self.context_builder.build_glossary()
        self.all_columns = self.context_builder.get_all_columns()

        self.intent_classifier = IntentClassifier(self.client, self.model)
        self.code_generator = CodeGenerator(self.client, self.model)
        self.executor = SafeExecutor()
        self.validator = ResultValidator()
        self.scorer = ConfidenceScorer()
        self.explainer = ExplanationGenerator(self.client, self.model)

        self.feedback_retriever = FeedbackRetriever(feedback_log_path)
        self.feedback_logger = FeedbackLogger(feedback_log_path)

        # Execution context (DuckDB + Pandas DataFrames) 
        self.exec_context = ExecutionContext(
            duckdb_conn=self.context_builder.get_connection(),
            dataframes=self.context_builder.get_dataframes(),
        )










    def run(self, query: str, track: str = "auto") -> dict:
        """Execute the full 10-step pipeline for a natural language query."""
        start = time.time()

        # ── Step 1: Preprocess ──
        preprocessed = self.preprocessor.process(query)

        # ── Step 3: Intent Classification (LLM Call #1) ──
        try:
            intent = self.intent_classifier.classify(
                preprocessed.normalized_q, self.schema_digest
            )
        except Exception as e:
            intent = QueryIntent(
                primary_operation="AGGREGATE",
                tables_needed=list(self.context_builder.dataframes.keys()),
                ambiguities=[f"Intent classification failed: {str(e)}"],
            )

        # ── Step 4: Feedback Retrieval ──
        past_examples = self.feedback_retriever.retrieve(preprocessed.normalized_q)
        feedback_str = self._format_feedback_examples(past_examples)
        feedback_boost = self._compute_feedback_boost(past_examples)

        # ── Step 5: Code Generation (LLM Call #2) ──
        try:
            generated = self.code_generator.generate(
                query=preprocessed.normalized_q,
                intent=intent,
                schema_digest=self.schema_digest,
                glossary=self.glossary,
                feedback_examples=feedback_str,
                complexity=preprocessed.complexity,
            )
        except Exception as e:
            generated = GeneratedCode(
                sql=None, pandas=None,
                self_confidence=0.0,
                reasoning=f"Code generation failed: {str(e)}",
                had_syntax_error=True,
            )

        # ── Step 6: Safe Execution ──
        gen_to_exec = generated
        track_lower = (track or "auto").lower()
        if track_lower == "sql":
            gen_to_exec = GeneratedCode(
                sql=generated.sql, pandas=None,
                self_confidence=generated.self_confidence,
                reasoning=generated.reasoning,
                had_syntax_error=generated.had_syntax_error,
            )
        elif track_lower == "pandas":
            gen_to_exec = GeneratedCode(
                sql=None, pandas=generated.pandas,
                self_confidence=generated.self_confidence,
                reasoning=generated.reasoning,
                had_syntax_error=generated.had_syntax_error,
            )

        exec_result = self.executor.execute(gen_to_exec, self.exec_context)

        # ── Self-Healing Retry (SYSTEM_DESIGN.md Section 9) ──
        if not exec_result.success and not getattr(generated, "had_syntax_error", False):
            retry_feedback = (
                f"{feedback_str}\n\n"
                f"[PREVIOUS CODE ATTEMPT FAILED WITH ERROR]: {exec_result.error}\n"
                f"Please fix the error and generate valid DuckDB SQL or Pandas code."
            )
            try:
                retry_generated = self.code_generator.generate(
                    query=preprocessed.normalized_q,
                    intent=intent,
                    schema_digest=self.schema_digest,
                    glossary=self.glossary,
                    feedback_examples=retry_feedback,
                    complexity=preprocessed.complexity,
                )
                retry_to_exec = retry_generated
                if track_lower == "sql":
                    retry_to_exec = GeneratedCode(
                        sql=retry_generated.sql, pandas=None,
                        self_confidence=retry_generated.self_confidence,
                        reasoning=retry_generated.reasoning,
                        had_syntax_error=retry_generated.had_syntax_error,
                    )
                elif track_lower == "pandas":
                    retry_to_exec = GeneratedCode(
                        sql=None, pandas=retry_generated.pandas,
                        self_confidence=retry_generated.self_confidence,
                        reasoning=retry_generated.reasoning,
                        had_syntax_error=retry_generated.had_syntax_error,
                    )
                retry_exec = self.executor.execute(retry_to_exec, self.exec_context)
                if retry_exec.success:
                    generated = retry_generated
                    exec_result = retry_exec
            except Exception:
                pass  # Fallback to original result if retry fails

        # ── Step 7: Result Validation ──
        if exec_result.success and exec_result.result is not None:
            validation = self.validator.validate(exec_result.result, intent)
        else:
            validation = ValidationReport(
                checks=[], pass_rate=0.0,
                issues=[exec_result.error or "Execution failed"],
            )

        # ── Step 8: Confidence Scoring ──
        signals = ConfidenceSignal(
            schema_match=self._compute_schema_match(intent),
            code_validity=0.0 if generated.had_syntax_error else 1.0,
            execution_success=(
                1.0 if exec_result.track == "SQL"
                else 0.7 if exec_result.track == "Pandas"
                else 0.0
            ),
            result_validation=validation.pass_rate,
            llm_self_score=generated.self_confidence,
        )
        score = self.scorer.score(
            signals,
            feedback_boost=feedback_boost,
            ambiguities_count=len(intent.ambiguities) if hasattr(intent, "ambiguities") else 0,
        )

        # ── Step 9: Explanation Generation (LLM Call #3) ──
        generated_logic = generated.sql or generated.pandas or ""
        if exec_result.success and exec_result.result is not None:
            try:
                explanation = self.explainer.explain(
                    query=query, intent=intent,
                    generated_logic=generated_logic,
                    track=exec_result.track,
                    result_df=exec_result.result,
                    score=score,
                )
            except Exception:
                explanation = Explanation(
                    understood="The system processed your query.",
                    approach="Generated and executed analytical code.",
                    caveats="Unable to generate detailed explanation.",
                )
        else:
            explanation = Explanation(
                understood="The system attempted to interpret your query but execution failed.",
                approach="N/A — no result was produced.",
                caveats=f"Error: {exec_result.error}",
            )

        # ── Step 10: Feedback Logging ──
        elapsed_ms = int((time.time() - start) * 1000)
        entry_id = self._log_feedback(query, generated, exec_result, intent, score)

        # ── Build final output ──
        result_data = []
        if exec_result.success and exec_result.result is not None:
            result_data = exec_result.result.to_dict(orient="records")

        return {
            "query": query,
            "generated_logic": generated_logic,
            "result": result_data,
            "confidence_score": score,
            "explanation": {
                "understood": explanation.understood,
                "approach": explanation.approach,
                "caveats": explanation.caveats,
            },
            "metadata": {
                "track_used": exec_result.track or "None",
                "complexity": preprocessed.complexity,
                "intent": intent.model_dump(),
                "processing_time_ms": elapsed_ms,
                "llm_calls": 3,
                "entry_id": entry_id,
            },
        }

    # ═════════════════════════════════════════════════════════════════════
    #  Internal helpers
    # ═════════════════════════════════════════════════════════════════════

    def _compute_schema_match(self, intent: QueryIntent) -> float:
        """Compute how well intent entities match actual schema columns."""
        if not intent.entities:
            return 1.0  # No entities to mismatch

        hits = 0
        for key, value in intent.entities.items():
            if key in self.all_columns or value in self.all_columns:
                hits += 1

        return hits / max(len(intent.entities), 1)

    def _format_feedback_examples(self, past: list) -> str:
        """Format past feedback entries as few-shot examples for the prompt."""
        if not past:
            return "No similar past queries found."

        parts = []
        for entry in past:
            label = "CORRECT" if entry.was_correct else "INCORRECT — DO NOT REPEAT"
            note = f"\n  Note: {entry.feedback_note}" if entry.feedback_note else ""
            parts.append(
                f"[{label}] (similarity: {entry.score:.2f})\n"
                f'  Query: "{entry.query}"\n'
                f"  Code: {entry.generated_logic}{note}"
            )
        return "\n\n".join(parts)

    def _compute_feedback_boost(self, past: list) -> float:
        """Compute confidence boost from similar correct past queries.

        > 0.85 similarity → +0.05
        0.6-0.85 similarity → +0.02
        """
        for entry in past:
            if entry.was_correct:
                if entry.score > 0.85:
                    return 0.05
                elif entry.score > 0.6:
                    return 0.02
        return 0.0

    def _log_feedback(self, query, generated, exec_result, intent, score) -> int:
        """Persist query result to feedback_log.csv and return row index."""
        result_preview = "[]"
        if exec_result.success and exec_result.result is not None:
            result_preview = exec_result.result.head(10).to_json(orient="records")

        entry = FeedbackEntry(
            timestamp=datetime.utcnow().isoformat(),
            query=query,
            generated_logic=generated.sql or generated.pandas or "",
            result_preview=result_preview,
            confidence_score=score,
            track_used=exec_result.track or "None",
            intent_json=intent.model_dump_json(),
            feedback="pending",
            feedback_note="",
        )
        return self.feedback_logger.log(entry)

    def mark_feedback(self, entry_id: int = None, was_correct: bool = True, note: str = "") -> bool:
        """Mark feedback for a past query run and refresh vector store."""
        if entry_id is not None:
            ok = self.feedback_logger.mark_feedback(entry_id, was_correct, note)
        else:
            ok = self.feedback_logger.mark_last(was_correct, note)
        if ok and self.feedback_log_path:
            self.feedback_retriever = FeedbackRetriever(self.feedback_log_path)
        return ok
