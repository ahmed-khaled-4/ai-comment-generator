"""FastAPI application for AI Comment Generator."""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import logging
import uuid

# Existing imports
from src.models.ai_client import OllamaService
from src.models.model_config import ModelConfig
from src.app_logging.logger import get_logger

# --- NEW: Import Evaluation Tools ---
# We use try/except to handle cases where these might not be fully set up yet
try:
    from src.evaluation.metrics import MetricsCalculator
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

# Initialize model logger
model_logger = get_logger()


# Exception handler for better error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )

# Initialize Evaluation Tools (Global)
metrics_calc = None
if EVALUATION_AVAILABLE:
    try:
        metrics_calc = MetricsCalculator()
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


# --- NEW: Multi-Agent Models ---
class MultiAgentRequest(BaseModel):
    """Request model for multi-agent generation."""
    code: str = Field(..., description="Source code to generate comment for")
    language: str = Field(default="python", description="Programming language")
    comment_type: str = Field(default="function", description="Type of comment: function, class, or inline")
    temperature: float = Field(default=0.4, ge=0.0, le=2.0, description="Sampling temperature")
    max_retries: int = Field(default=2, ge=1, le=5, description="Maximum retry attempts")
    governance_level: str = Field(default="standard", description="Governance level: permissive, standard, strict")

class MultiAgentResponse(BaseModel):
    """Response model for multi-agent generation."""
    comment: str
    approved: bool
    attempts: int
    generator_result: Optional[Dict[str, Any]]
    validator_result: Optional[Dict[str, Any]]
    governance_statistics: Dict[str, Any]
    requires_human_review: bool
    audit_trail: List[Dict[str, Any]]

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
    Generate a comment for the given code with comprehensive logging and error handling.
    
    Args:
        request: CommentRequest with code and parameters
    
    Returns:
        CommentResponse with generated comment and metadata
    
    Raises:
        HTTPException: If generation fails or service is unavailable
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    logger.info(f"Request {request_id}: Generating {request.comment_type} comment for {request.language} code")
    
    try:
        # Validate configuration
        temp_config = ModelConfig(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        temp_config.validate()
        
        # Use custom model if provided
        service = ollama_service
        if request.model:
            logger.info(f"Request {request_id}: Using custom model {request.model}")
            service = OllamaService(model=request.model)
        
        # Generate comment with request ID for logging
        result = service.generate_comment(
            code=request.code,
            language=request.language,
            comment_type=request.comment_type,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            request_id=request_id
        )
        
        logger.info(f"Request {request_id}: Successfully generated comment")
        return CommentResponse(**result)
        
    except ValueError as e:
        logger.error(f"Request {request_id}: Validation error - {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request parameters: {str(e)}"
        )
    except ConnectionError as e:
        logger.error(f"Request {request_id}: Connection error - {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama. Make sure Ollama is running: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Request {request_id}: Generation error - {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating comment: {str(e)}"
        )


@app.get("/logs/statistics", tags=["Logging"])
async def get_log_statistics():
    """
    Get statistics about logged generations.
    
    Returns:
        Statistics including total requests, success rate, etc.
    """
    try:
        stats = model_logger.get_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error retrieving log statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving statistics: {str(e)}"
        )


# --- NEW: Multi-Agent Endpoint ---
@app.post("/multi_agent/generate", response_model=MultiAgentResponse, tags=["Multi-Agent"])
async def multi_agent_generate(request: MultiAgentRequest):
    """
    Generate code comment using CrewAI multi-agent system with ethical governance.
    
    This endpoint uses CrewAI framework with a two-agent workflow:
    1. Generator Agent: Creates the comment
    2. Validator Agent: Validates quality and checks for hallucinations
    
    Both agents operate under governance controls (Safety, Transparency, Explainability, Accountability).
    
    Args:
        request: MultiAgentRequest with code and parameters
    
    Returns:
        MultiAgentResponse with comment, governance decisions, and audit trail
    """
    try:
        # Import CrewAI multi-agent system
        from src.multi_agent.crewai_agents import CrewAICommentGenerator
        from src.multi_agent.governance import GovernanceConfig, GovernanceLevel
        
        # Configure governance
        governance_level_map = {
            "permissive": GovernanceLevel.PERMISSIVE,
            "standard": GovernanceLevel.STANDARD,
            "strict": GovernanceLevel.STRICT
        }
        
        governance_config = GovernanceConfig(
            level=governance_level_map.get(request.governance_level.lower(), GovernanceLevel.STANDARD)
        )
        
        # Create CrewAI generator
        crew_generator = CrewAICommentGenerator(
            model_service=ollama_service,
            governance_config=governance_config
        )
        
        # Run CrewAI workflow
        crew_result = crew_generator.generate_and_validate(
            code=request.code,
            language=request.language,
            comment_type=request.comment_type,
            temperature=request.temperature,
            max_retries=request.max_retries
        )
        
        # Convert CrewAI result to MultiAgentResponse format
        result = {
            "comment": crew_result.comment,
            "approved": crew_result.approved,
            "attempts": crew_result.attempts,
            "generator_result": {
                "confidence": crew_result.generator_confidence,
                "reasoning": crew_result.governance_decisions[0]["reasoning"] if crew_result.governance_decisions else "",
                "governance_approved": crew_result.governance_decisions[0]["approved"] if crew_result.governance_decisions else False
            },
            "validator_result": {
                "confidence": crew_result.validator_confidence,
                "reasoning": crew_result.governance_decisions[1]["reasoning"] if len(crew_result.governance_decisions) > 1 else "",
                "governance_approved": crew_result.governance_decisions[1]["approved"] if len(crew_result.governance_decisions) > 1 else False
            },
            "governance_statistics": crew_generator.get_governance_statistics(),
            "requires_human_review": crew_result.requires_human_review,
            "audit_trail": crew_generator.get_audit_trail()
        }
        
        return MultiAgentResponse(**result)
        
    except ImportError as e:
        logger.error(f"CrewAI multi-agent system not available: {e}")
        raise HTTPException(
            status_code=501,
            detail="CrewAI is not installed. Install with: pip install crewai"
        )
    except Exception as e:
        logger.error(f"Error in CrewAI multi-agent generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error in multi-agent generation: {str(e)}"
        )


@app.get("/multi_agent/governance/statistics", tags=["Multi-Agent"])
async def get_governance_statistics():
    """
    Get governance framework statistics from CrewAI system.
    
    Returns statistics about governance decisions including approval rates,
    rejections, and human review requirements.
    """
    try:
        from src.multi_agent.crewai_agents import CrewAICommentGenerator
        
        # Create temporary generator to get statistics
        crew_generator = CrewAICommentGenerator(model_service=ollama_service)
        stats = crew_generator.get_governance_statistics()
        
        return {
            "status": "success",
            "statistics": stats,
            "framework": "CrewAI"
        }
    except Exception as e:
        logger.error(f"Error retrieving governance statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving statistics: {str(e)}"
        )


# --- NEW: Evaluation Endpoint ---
@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_code_sample(request: EvaluationRequest):
    """
    Generates a comment and calculates quality metrics.
    """
    if not EVALUATION_AVAILABLE:
        raise HTTPException(status_code=501, detail="Evaluation modules are not available on this server.")

    try:
        # 1. Generate Comment using the existing Ollama Service
        gen_result = ollama_service.generate_comment(
            code=request.code,
            language=request.language,
            comment_type="function",
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

        return EvaluationResponse(
            generated_comment=generated_comment,
            metrics=metrics
        )

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)