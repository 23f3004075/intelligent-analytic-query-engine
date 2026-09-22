import json
import os
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "inclusionai/ling-3.0-flash-vl:free"


class QueryIntent(BaseModel):
    # Structured intent extracted from a natural language query.
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
    # LLM call that classifies intent before code generation

    def __init__(self, client: OpenAI = None, model: str = None):
        openrouter_key = os.getenv("OPEN_ROUTER_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if client is not None:
            self.client = client
            self.model = model or DEFAULT_MODEL
        elif openrouter_key:
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
            self.model = model or os.getenv("OPEN_ROUTER_MODEL", DEFAULT_MODEL)
        elif openai_key:
            self.client = OpenAI(api_key=openai_key)
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="not_set")
            self.model = model or DEFAULT_MODEL

    def classify(self, query: str, schema_digest: str) -> QueryIntent:
        system_prompt = f"""You are a query intent parser. Given a natural language analytics question 
and a database schema, extract a structured JSON intent.

SCHEMA:
{schema_digest}

Respond ONLY with a valid JSON matching this schema: 
{json.dumps(QueryIntent.model_json_schema(), indent=2)}

Never guess column names — use exactly what the schema lists."""

        try:
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
            return QueryIntent.model_validate_json(raw_json)
        except Exception:
            return self._heuristic_classify(query)

    def _heuristic_classify(self, query: str) -> QueryIntent:
        import re
        q = query.lower()
        primary = "AGGREGATE"
        secondaries = []
        n_val = None
        tables = ["sales_data"]
        entities = {}

        if any(w in q for w in ["target", "quota", "missed"]):
            tables.append("targets")
            primary = "COMPARE"
            secondaries.append("JOIN")

        if any(w in q for w in ["loss", "losses", "least profit", "lowest profit"]):
            primary = "RANK"
            secondaries.append("BOTTOM_N")
            entities["metric"] = "profit"
            entities["order"] = "asc"

        if any(w in q for w in ["top", "highest", "lowest", "bottom", "rank", "best", "worst"]):
            primary = "RANK"
            secondaries.append("TOP_N")
            m = re.search(r"(?:top|first|limit|bottom)\s+(\d+)", q)
            if m:
                n_val = int(m.group(1))

        if any(w in q for w in ["%", "contribution", "share", "percentage"]):
            secondaries.append("PCT_CONTRIBUTION")

        if any(w in q for w in ["yoy", "growth", "year over year"]):
            primary = "TIME_SERIES"

        for col in ["city", "region", "country", "product_category", "product_name", "customer_segment"]:
            if col in q or col.replace("_", " ") in q or col + "s" in q or (col == "city" and "cities" in q):
                entities[col] = "group"

        time_c = None
        for m in ["january", "february", "march", "jan", "feb", "mar", "q1", "q2", "2023", "2024"]:
            if m in q:
                time_c = m
                break

        return QueryIntent(
            primary_operation=primary,
            secondary_operations=secondaries,
            entities=entities,
            time_constraint=time_c,
            n_value=n_val,
            tables_needed=tables,
            ambiguities=[],
        )

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  
            text = "\n".join(lines).strip()
        return text