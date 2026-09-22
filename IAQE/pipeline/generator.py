import json
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "inclusionai/ling-3.0-flash-vl:free"

# Import QueryIntent — supports both package and direct execution
try:
    from IAQE.pipeline.intent import QueryIntent
except ImportError:
    from intent import QueryIntent


class GeneratedCode(BaseModel):
    """Output from the CodeGenerator LLM call."""
    sql: Optional[str] = Field(None, description="SQL query or null if not possible")
    pandas: Optional[str] = Field(None, description="Pandas code string")
    self_confidence: float = Field(..., description="0.0-1.0 confidence score")
    reasoning: str = Field(..., description="Brief chain of thought")
    had_syntax_error: bool = Field(False, exclude=True)  # Used downstream by scorer


class CodeGenerator:
    """Core generation step — uses Chain-of-Thought for COMPLEX queries."""

    def __init__(self, client: OpenAI = None, model: str = DEFAULT_MODEL):
        api_key = os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPENAI_API_KEY") or "not_set"
        self.client = client or OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def generate(self, query: str, intent, schema_digest: str,
                 glossary: str = "", feedback_examples: str = "",
                 complexity: str = "SIMPLE") -> GeneratedCode:
        # Chain-of-Thought for COMPLEX queries
        cot_prompt = ""
        if complexity == "COMPLEX":
            cot_prompt = """
Before writing any code, think through these steps:
<thinking>
1. What is the "grain" of the output? (one row = one what?)
2. What filtering must happen first?
3. Is a subquery or CTE needed?
4. What aggregation comes after filtering?
5. What post-aggregation filter is needed (HAVING)?
6. What ordering / limiting?
</thinking>
"""

        system_prompt = f"""You are an expert SQL and Pandas code generator for business analytics.

--- DATABASE SCHEMA ---
{schema_digest}

--- BUSINESS GLOSSARY ---
{glossary}

--- SIMILAR PAST QUERIES (few-shot) ---
{feedback_examples}

Rules:
1. Use ONLY column names that exist in the schema. Never invent columns.
2. For SQL: tables are already registered in DuckDB as their file names.
3. For Pandas: DataFrames are pre-loaded as variables named <table_name>_df (e.g., sales_data_df, targets_df).
4. Handle NULL values explicitly.
5. Return results as a table (not scalar unless query asks for single value).
6. If joining tables, always specify join keys explicitly.
7. For computed metrics like revenue, use the formula from the glossary.
8. For Pandas code, always assign the final output to a variable called `result`.

Output ONLY valid JSON matching this schema:
{json.dumps(GeneratedCode.model_json_schema(), indent=2)}"""

        user_prompt = f"""--- QUERY INTENT ---
{intent.model_dump_json()}

{cot_prompt}
TASK: Generate executable code to answer: "{query}"
"""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        raw_json = self._extract_json(response.choices[0].message.content)

        try:
            return GeneratedCode.model_validate_json(raw_json)
        except Exception:
            # If LLM response doesn't validate, return a failure signal
            return GeneratedCode(
                sql=None,
                pandas=None,
                self_confidence=0.0,
                reasoning="Failed to parse LLM response",
                had_syntax_error=True
            )

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