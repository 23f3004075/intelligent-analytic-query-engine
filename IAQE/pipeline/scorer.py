from dataclasses import dataclass

@dataclass
class ConfidenceSignal:
    schema_match: float
    code_validity: float
    execution_success: float
    result_validation: float
    llm_self_score: float

class ConfidenceScorer:
    WEIGHTS = [0.25, 0.20, 0.25, 0.20, 0.10]

    def score(self, signals: ConfidenceSignal, feedback_boost: float = 0.0, ambiguities_count: int = 0) -> float:
        values = [
            signals.schema_match,
            signals.code_validity,
            signals.execution_success,
            signals.result_validation,
            signals.llm_self_score,
        ]
        
        raw = sum(v * w for v, w in zip(values, self.WEIGHTS))
        
        # Hard Penalties
        if signals.execution_success == 0.0:
            raw = min(raw, 0.15)
        if signals.schema_match < 0.5:
            raw = min(raw, 0.30)
        if ambiguities_count > 2:
            raw *= 0.85
            
        # Apply feedback boost (if a semantically similar past query was correct)
        raw = min(raw + feedback_boost, 0.97)
        
        return round(raw, 3)