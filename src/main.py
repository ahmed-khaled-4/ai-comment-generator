"""FastAPI application for AI Comment Generator."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import logging

# Existing imports
from src.models.ai_client import OllamaService
from src.models.model_config import ModelConfig

# --- NEW: Import Evaluation Tools ---
# We use try/except to handle cases where these might not be fully set up yet
try:
    from src.evaluation.metrics import MetricsCalculator
    from src.evaluation.hallucination import HallucinationDetector
    EVALUATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Evaluation modules not found: {e}")
    EVALUATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Comment Generator API",
    description="API for generating code comments using Ollama LLM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama service with default config
default_config = ModelConfig()
ollama_service = OllamaService(model=default_config.model_name, config=default_config)

# Initialize Evaluation Tools (Global)
metrics_calc = None
hallucination_detector = None
if EVALUATION_AVAILABLE:
    try:
        metrics_calc = MetricsCalculator()
        hallucination_detector = HallucinationDetector()
        logger.info("Evaluation tools initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize evaluation tools: {e}")
        EVALUATION_AVAILABLE = False


# ==========================================
# Pydantic Models
# ==========================================

class CommentRequest(BaseModel):
    """Request model for comment generation."""
    code: str = Field(..., description="Source code to generate comment for")
    language: str = Field(default="python", description="Programming language")
    comment_type: str = Field(
        default="function", 
        description="Type of comment: function, class, or inline"
    )
    temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0, 
        description="Sampling temperature (0.0-2.0)"
    )
    max_tokens: int = Field(
        default=400, 
        ge=1, 
        le=2000, 
        description="Maximum tokens to generate"
    )
    model: Optional[str] = Field(
        default=None, 
        description="Ollama model to use (optional, overrides default)"
    )


class CommentResponse(BaseModel):
    """Response model for comment generation."""
    comment: str = Field(..., description="Generated comment")
    model: str = Field(..., description="Model used for generation")
    language: str = Field(..., description="Programming language")
    comment_type: str = Field(..., description="Type of comment generated")
    metadata: dict = Field(..., description="Generation metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    model: str


# --- NEW: Evaluation Models ---
class EvaluationRequest(BaseModel):
    """Request model for evaluation."""
    code: str = Field(..., description="Code to evaluate")
    language: str = Field(default="python", description="Programming language")
    human_comment: Optional[str] = Field(None, description="Human reference comment for calculating metrics (BLEU, etc.)")

class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    generated_comment: str
    metrics: Dict[str, float] = Field(default_factory=dict, description="Evaluation scores (BLEU, BERTScore, etc.)")
    hallucination_flags: List[str] = Field(default_factory=list, description="Detected issues (e.g., 'TOO_SHORT', 'HALLUCINATION')")


# ==========================================
# Endpoints
# ==========================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    endpoints = {
        "generate": "/generate_comment",
        "health": "/health",
        "docs": "/docs"
    }
    if EVALUATION_AVAILABLE:
        endpoints["evaluate"] = "/evaluate"
        
    return {
        "message": "AI Comment Generator API",
        "version": "1.0.0",
        "endpoints": endpoints
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        # Verify Ollama connection
        ollama_service._verify_connection()
        return {
            "status": "healthy",
            "service": "ollama",
            "model": ollama_service.model
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


@app.post("/generate_comment", response_model=CommentResponse, tags=["Generation"])
async def generate_comment(request: CommentRequest):
    """
    Generate a comment for the given code.
    """
    try:
        # Use custom model if provided
        service = ollama_service
        if request.model:
            service = OllamaService(model=request.model)
        
        # Generate comment
        result = service.generate_comment(
            code=request.code,
            language=request.language,
            comment_type=request.comment_type,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return CommentResponse(**result)
        
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama. Make sure Ollama is running: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error generating comment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating comment: {str(e)}"
        )


# --- NEW: Evaluation Endpoint ---
@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_code_sample(request: EvaluationRequest):
    """
    Generates a comment, calculates quality metrics, and checks for hallucinations.
    """
    if not EVALUATION_AVAILABLE:
        raise HTTPException(status_code=501, detail="Evaluation modules are not available on this server.")

    try:
        # 1. Generate Comment using the existing Ollama Service
        # We assume standard parameters (temperature 0.7) for evaluation
        gen_result = ollama_service.generate_comment(
            code=request.code,
            language=request.language,
            comment_type="function", # Default to function comments
            temperature=0.7
        )
        generated_comment = gen_result["comment"]

        # 2. Calculate Metrics (if human reference provided)
        metrics = {}
        if request.human_comment and metrics_calc:
            # Wrap in list because calculator expects batch input
            scores = metrics_calc.compute_per_sample_metrics_safe(
                [request.human_comment], 
                [generated_comment]
            )
            if scores:
                metrics = scores[0]

        # 3. Run Hallucination Detection
        flags = []
        if hallucination_detector:
            flags = hallucination_detector.analyze(
                request.code, 
                generated_comment, 
                score_metrics=metrics if metrics else None
            )

        return EvaluationResponse(
            generated_comment=generated_comment,
            metrics=metrics,
            hallucination_flags=flags
        )

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)