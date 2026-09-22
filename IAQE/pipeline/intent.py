import json
import os
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "inclusionai/ling-3.0-flash-vl:free"


class QueryIntent(BaseModel):
    """Structured intent extracted from a natural language query."""
    primary_operation: str = Field(
        ...,
        description="AGGREGATE | RANK | FILTER | COMPARE | TIME_SERIES"
    )
    secondary_operations: List[str] = Field(
        default_factory=list,
        description='e.g., ["GROUP_BY", "TOP_N", "PCT_CONTRIBUTION"]'
    )
    entities: Dict[str, str] = Field(
        default_factory=dict,
        description='Map of entity to value, e.g., {"region": "North", "metric": "sale_amount"}'
    )
    time_constraint: Optional[str] = Field(
        None,
        description='e.g., "Q1 2024", "last 3 months", or null'
    )
    n_value: Optional[int] = Field(
        None,
        description="For 'top N' queries, the N value"
    )
    tables_needed: List[str] = Field(
        ...,
        description="List of table names required to answer the query"
    )
    ambiguities: List[str] = Field(
        default_factory=list,
        description="Things the system is uncertain about based on the prompt"
    )


class IntentClassifier:
    """Lightweight, structured LLM call that classifies intent before code generation."""

    def __init__(self, client: OpenAI = None, model: str = DEFAULT_MODEL):
        api_key = os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPENAI_API_KEY") or "not_set"
        self.client = client or OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def classify(self, query: str, schema_digest: str) -> QueryIntent:
        system_prompt = f"""You are a query intent parser. Given a natural language analytics question 
and a database schema, extract a structured JSON intent.

SCHEMA:
{schema_digest}

Respond ONLY with a valid JSON matching this schema: 
{json.dumps(QueryIntent.model_json_schema(), indent=2)}

Never guess column names — use exactly what the schema lists."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=300,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'QUERY: "{query}"'}
            ]
        )

        raw_json = self._extract_json(response.choices[0].message.content)
        intent = QueryIntent.model_validate_json(raw_json)
        return intent

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # Remove opening ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove closing ```
            text = "\n".join(lines).strip()
        return text