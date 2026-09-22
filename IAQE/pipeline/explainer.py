import json
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

DEFAULT_MODEL = "inclusionai/ling-3.0-flash-vl:free"


class Explanation(BaseModel):
    """3-part explanation for business users."""
    understood: str = Field(..., description="What did the system interpret from the question?")
    approach: str = Field(..., description="How was the result computed in plain English?")
    caveats: str = Field(..., description="Any assumptions made or things to watch out for?")


class ExplanationGenerator:
    """Generates human-readable explanations of query results (LLM Call #3)."""

    def __init__(self, client: OpenAI = None, model: str = DEFAULT_MODEL):
        api_key = os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPENAI_API_KEY") or "not_set"
        self.client = client or OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def explain(self, query: str, intent, generated_logic: str, track: str,
                result_df: pd.DataFrame, score: float) -> Explanation:
        score_label = self._get_score_label(score)
        result_preview = result_df.head(5).to_json(orient='records')

        prompt = f"""You are explaining an analytics query result to a business user.

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
Respond ONLY with a valid JSON matching this schema:
{json.dumps(Explanation.model_json_schema(), indent=2)}"""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )

        raw_json = self._extract_json(response.choices[0].message.content)

        try:
            return Explanation.model_validate_json(raw_json)
        except Exception:
            # Graceful fallback if LLM response is unparseable
            return Explanation(
                understood="The system processed your query.",
                approach="Generated and executed analytical code.",
                caveats="Unable to generate detailed explanation."
            )

    def _get_score_label(self, score: float) -> str:
        """Score bands from Section 7 of SYSTEM_DESIGN."""
        if score >= 0.85: return "High"
        if score >= 0.65: return "Medium"
        if score >= 0.40: return "Low"
        return "Unreliable"

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text