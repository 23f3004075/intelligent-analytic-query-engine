import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



@dataclass
class RetrievedFeedback:
    query: str
    generated_logic: str
    was_correct: bool
    score: float
    feedback_note: str = ""

@dataclass
class FeedbackEntry:
    timestamp: str
    query: str
    generated_logic: str
    result_preview: str
    confidence_score: float
    track_used: str
    intent_json: str
    feedback: str      # "pending" | "correct" | "incorrect"
    feedback_note: str 


class FeedbackRetriever:
    def __init__(self, feedback_log_path: str):
        self.log_path = Path(feedback_log_path)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.is_active = False
        
        if self.log_path.exists():
            self.log = pd.read_csv(self.log_path)
            evaluated_logs = self.log[self.log["feedback"] != "pending"]
            
            if not evaluated_logs.empty:
                self.log = evaluated_logs.reset_index(drop=True)
                self.vectors = self.vectorizer.fit_transform(self.log["query"])
                self.is_active = True

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedFeedback]:
        
        if not self.is_active:
            return []

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.vectors).flatten()
        
        top_idx = scores.argsort()[-top_k:][::-1]
        
        entries = []
        for i in top_idx:
            if scores[i] > 0.6:
                row = self.log.iloc[i]
                entries.append(RetrievedFeedback(
                    query=row["query"],
                    generated_logic=row["generated_logic"],
                    was_correct=(row["feedback"] == "correct"),
                    score=float(scores[i]),
                    feedback_note=row.get("feedback_note", "")
                ))
        return entries

class FeedbackLogger:
    def __init__(self, feedback_log_path: str):
        self.log_path = Path(feedback_log_path)

    def log(self, entry: FeedbackEntry) -> int:
        is_new = not self.log_path.exists()
        current_len = 0
        if not is_new:
            try:
                current_len = len(pd.read_csv(self.log_path))
            except Exception:
                current_len = 0
        df = pd.DataFrame([asdict(entry)])
        df.to_csv(
            self.log_path, 
            mode='a', 
            index=False, 
            header=is_new
        )
        return current_len

    def mark_feedback(self, entry_id: int, was_correct: bool, note: str = "") -> bool:
        if not self.log_path.exists():
            return False
        df = pd.read_csv(self.log_path, dtype={"feedback": str, "feedback_note": str})
        if entry_id < 0 or entry_id >= len(df):
            return False
        df.at[entry_id, "feedback"] = "correct" if was_correct else "incorrect"
        if note:
            df.at[entry_id, "feedback_note"] = str(note)
        df.to_csv(self.log_path, index=False)
        return True

    def mark_last(self, was_correct: bool, note: str = "") -> bool:
        if not self.log_path.exists():
            return False
        df = pd.read_csv(self.log_path, dtype={"feedback": str, "feedback_note": str})
        if df.empty:
            return False
        last_idx = len(df) - 1
        df.at[last_idx, "feedback"] = "correct" if was_correct else "incorrect"
        if note:
            df.at[last_idx, "feedback_note"] = str(note)
        df.to_csv(self.log_path, index=False)
        return True