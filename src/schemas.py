# src/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

class EvaluationRequest(BaseModel):
    code: str
    language: str = "python"
    human_comment: Optional[str] = None  # Optional: Provide this if you want BLEU/BERT scores

class EvaluationResponse(BaseModel):
    generated_comment: str
    metrics: Dict[str, float] = {}
    hallucination_flags: List[str] = []


# --- NEW: Validation System Schemas ---
class ValidationStatisticsResponse(BaseModel):
    """Response model for validation statistics."""
    status: str
    statistics: Dict[str, Any] = Field(..., description="Validation statistics including violations, rejections, retries")


class ValidationReportResponse(BaseModel):
    """Response model for validation report."""
    status: str
    report: str = Field(..., description="Text report with validation statistics and insights")


class ReviewFlagResponse(BaseModel):
    """Response model for a review flag."""
    comment_id: str
    comment: str
    confidence: float
    reason: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PendingReviewsResponse(BaseModel):
    """Response model for pending reviews."""
    status: str
    pending_count: int
    reviews: List[ReviewFlagResponse]


class ReviewActionResponse(BaseModel):
    """Response model for review actions (approve/reject)."""
    status: str
    message: str


class ReviewStatisticsResponse(BaseModel):
    """Response model for review statistics."""
    status: str
    statistics: Dict[str, Any] = Field(..., description="Review statistics including approval rates, pending counts")