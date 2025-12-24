# src/schemas.py
from pydantic import BaseModel
from typing import Optional, Dict, List

class EvaluationRequest(BaseModel):
    code: str
    language: str = "python"
    human_comment: Optional[str] = None  # Optional: Provide this if you want BLEU/BERT scores

class EvaluationResponse(BaseModel):
    generated_comment: str
    metrics: Dict[str, float] = {}
    hallucination_flags: List[str] = []