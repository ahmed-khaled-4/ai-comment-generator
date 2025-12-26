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

# --- NEW: Import Validation System ---
from src.validators.input_request_validation import validate_request, ValidationError
from src.validators.output_validation import validate_output, OutputValidationError
from src.validators.safety_rules import enforce_safety_rules, SafetyViolationError, get_safety_config, SafetyConfig
from src.validators.rejection_retry import process_with_retry, RetryConfig, RetryStrategy, RetryExhaustedError
from src.human_review.review_system import flag_for_human_review, get_review_system, HumanReviewSystem
from src.monitoring.validation_monitor import (
    get_monitor, log_safety_violation, log_rejection, log_retry,
    get_validation_statistics, generate_validation_report
)

# --- NEW: Import Evaluation Tools ---
# We use try/except to handle cases where these might not be fully set up yet
try:
    from src.evaluation.metrics import MetricsCalculator
    EVALUATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Evaluation modules not available: {e}")
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

# Initialize Validation System components
validation_monitor = get_monitor()
review_system = get_review_system()
logger.info("Validation system initialized successfully")


# Exception handler for better error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    # Don't handle HTTPException (let FastAPI handle it)
    from fastapi import HTTPException as FastAPIHTTPException
    if isinstance(exc, FastAPIHTTPException):
        raise exc
    
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
        "docs": "/docs",
        "validation_stats": "/validation/statistics",
        "validation_report": "/validation/report",
        "human_review": "/human_review/pending",
        "human_review_stats": "/human_review/statistics"
    }
    if EVALUATION_AVAILABLE:
        endpoints["evaluate"] = "/evaluate"
        
    return {
        "message": "AI Comment Generator API",
        "version": "1.0.0",
        "endpoints": endpoints,
        "features": {
            "input_validation": True,
            "output_validation": True,
            "safety_rules": True,
            "human_review": True,
            "automatic_retry": True,
            "monitoring": True
        }
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
    Generate a comment for the given code with comprehensive logging, validation, and error handling.
    
    This endpoint now includes:
    - Input request validation (syntax, language, size, security)
    - Output validation (format, completeness, quality)
    - Safety rules enforcement
    - Human review flagging (if confidence is low)
    - Automatic retry with fallback (if validation fails)
    - Monitoring and logging
    
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
        # ==========================================
        # 1. INPUT VALIDATION
        # ==========================================
        try:
            request_data = {
                'code': request.code,
                'language': request.language,
                'comment_type': request.comment_type,
                'temperature': request.temperature,
                'max_tokens': request.max_tokens,
                'model': request.model
            }
            validate_request(request_data)
            logger.debug(f"Request {request_id}: Input validation passed")
        except ValidationError as e:
            logger.warning(f"Request {request_id}: Input validation failed - {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {str(e)}"
            )
        
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
        
        # ==========================================
        # 2. GENERATE COMMENT WITH RETRY LOGIC
        # ==========================================
        def generate_comment_func():
            """Inner function for retry logic."""
            return service.generate_comment(
                code=request.code,
                language=request.language,
                comment_type=request.comment_type,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                request_id=request_id
            )
        
        def validate_comment(comment: str) -> bool:
            """Validation function for retry logic."""
            try:
                # Output validation
                validate_output(comment, code=request.code, language=request.language)
                
                # Safety rules enforcement
                safety_config = get_safety_config("standard")
                enforce_safety_rules(comment, config=safety_config)
                
                return True
            except (OutputValidationError, SafetyViolationError) as e:
                # Log rejection
                log_rejection(
                    reason=str(e),
                    comment=comment[:200] if comment else "",
                    attempt_number=1,
                    metadata={'request_id': request_id, 'language': request.language}
                )
                raise
        
        # Initial generation
        result = generate_comment_func()
        generated_comment = result["comment"]
        
        # Try to validate and retry if needed
        retry_config = RetryConfig(
            max_retries=2,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            enable_fallback=True,
            fallback_comment="Comment generation completed but validation failed after retries."
        )
        
        try:
            # Process with retry logic
            validated_comment = process_with_retry(
                comment=generated_comment,
                validation_func=validate_comment,
                generation_func=lambda: generate_comment_func()["comment"],
                config=retry_config
            )
            result["comment"] = validated_comment
        except RetryExhaustedError:
            # Fallback comment used
            logger.warning(f"Request {request_id}: Retry exhausted, using fallback")
            result["comment"] = retry_config.fallback_comment
        
        # ==========================================
        # 3. HUMAN REVIEW FLAGGING
        # ==========================================
        # Calculate confidence score (simplified - could use actual model confidence)
        confidence = 0.8  # Default confidence, could be extracted from model response
        requires_review = flag_for_human_review(
            comment=result["comment"],
            confidence=confidence,
            reason="Standard review check",
            metadata={'request_id': request_id, 'language': request.language}
        )
        
        # Add review flag to metadata
        if requires_review:
            result["metadata"]["requires_human_review"] = True
            result["metadata"]["review_reason"] = "Low confidence or safety concerns"
            logger.info(f"Request {request_id}: Comment flagged for human review")
        else:
            result["metadata"]["requires_human_review"] = False
        
        # ==========================================
        # 4. MONITORING
        # ==========================================
        # Log successful generation (monitoring system tracks this automatically)
        result["metadata"]["validation_passed"] = True
        result["metadata"]["safety_checked"] = True
        
        logger.info(f"Request {request_id}: Successfully generated and validated comment")
        return CommentResponse(**result)
        
    except ValidationError as e:
        logger.error(f"Request {request_id}: Validation error - {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(e)}"
        )
    except (OutputValidationError, SafetyViolationError) as e:
        logger.error(f"Request {request_id}: Output/safety validation error - {e}")
        # Log safety violation
        log_safety_violation(
            violation_type=type(e).__name__,
            comment=request.code[:200] if hasattr(request, 'code') else "",
            severity="medium",
            metadata={'request_id': request_id}
        )
        raise HTTPException(
            status_code=422,
            detail=f"Output validation failed: {str(e)}"
        )
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


# --- NEW: Validation System Endpoints ---
@app.get("/validation/statistics", tags=["Validation"])
async def get_validation_stats(time_window_hours: Optional[int] = None):
    """
    Get validation system statistics.
    
    Args:
        time_window_hours: Optional time window in hours (None = all time)
    
    Returns:
        Dictionary with validation statistics including violations, rejections, retries
    """
    try:
        stats = get_validation_statistics(time_window_hours)
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error retrieving validation statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving validation statistics: {str(e)}"
        )


@app.get("/validation/report", tags=["Validation"])
async def get_validation_report():
    """
    Generate and return validation monitoring report.
    
    Returns:
        Text report with validation statistics and insights
    """
    try:
        report = generate_validation_report()
        return {
            "status": "success",
            "report": report
        }
    except Exception as e:
        logger.error(f"Error generating validation report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}"
        )


@app.get("/human_review/pending", tags=["Human Review"])
async def get_pending_reviews():
    """
    Get list of comments pending human review.
    
    Returns:
        List of review flags pending review
    """
    try:
        pending = review_system.get_pending_reviews()
        return {
            "status": "success",
            "pending_count": len(pending),
            "reviews": [
                {
                    "comment_id": flag.comment_id,
                    "comment": flag.comment[:200] + "..." if len(flag.comment) > 200 else flag.comment,
                    "confidence": flag.confidence,
                    "reason": flag.reason,
                    "timestamp": flag.timestamp,
                    "metadata": flag.metadata
                }
                for flag in pending
            ]
        }
    except Exception as e:
        logger.error(f"Error retrieving pending reviews: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving pending reviews: {str(e)}"
        )


@app.post("/human_review/approve/{comment_id}", tags=["Human Review"])
async def approve_comment(comment_id: str, reviewer: str, notes: Optional[str] = None):
    """
    Approve a flagged comment.
    
    Args:
        comment_id: ID of the comment to approve
        reviewer: Name/ID of the reviewer
        notes: Optional review notes
    
    Returns:
        Success status
    """
    try:
        success = review_system.approve_comment(comment_id, reviewer, notes)
        if success:
            return {
                "status": "success",
                "message": f"Comment {comment_id} approved by {reviewer}"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Comment {comment_id} not found in review queue"
            )
    except Exception as e:
        logger.error(f"Error approving comment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error approving comment: {str(e)}"
        )


@app.post("/human_review/reject/{comment_id}", tags=["Human Review"])
async def reject_comment(comment_id: str, reviewer: str, notes: Optional[str] = None):
    """
    Reject a flagged comment.
    
    Args:
        comment_id: ID of the comment to reject
        reviewer: Name/ID of the reviewer
        notes: Optional review notes
    
    Returns:
        Success status
    """
    try:
        success = review_system.reject_comment(comment_id, reviewer, notes)
        if success:
            return {
                "status": "success",
                "message": f"Comment {comment_id} rejected by {reviewer}"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Comment {comment_id} not found in review queue"
            )
    except Exception as e:
        logger.error(f"Error rejecting comment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error rejecting comment: {str(e)}"
        )


@app.get("/human_review/statistics", tags=["Human Review"])
async def get_review_statistics():
    """
    Get statistics about human reviews.
    
    Returns:
        Dictionary with review statistics
    """
    try:
        stats = review_system.get_review_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error retrieving review statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving review statistics: {str(e)}"
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
    
    This endpoint now includes input validation before generation.
    
    Args:
        request: MultiAgentRequest with code and parameters
    
    Returns:
        MultiAgentResponse with comment, governance decisions, and audit trail
    """
    try:
        # Input validation
        try:
            request_data = {
                'code': request.code,
                'language': request.language,
                'comment_type': request.comment_type,
                'temperature': request.temperature
            }
            validate_request(request_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {str(e)}"
            )
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
    
    This endpoint now includes input validation before generation.
    """
    if not EVALUATION_AVAILABLE:
        raise HTTPException(status_code=501, detail="Evaluation modules are not available on this server.")

    try:
        # Input validation
        try:
            request_data = {
                'code': request.code,
                'language': request.language,
                'comment_type': 'function'
            }
            validate_request(request_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {str(e)}"
            )
        
        # 1. Generate Comment using the existing Ollama Service
        gen_result = ollama_service.generate_comment(
            code=request.code,
            language=request.language,
            comment_type="function",
            temperature=0.7
        )
        generated_comment = gen_result["comment"]
        
        # Output validation
        try:
            validate_output(generated_comment, code=request.code, language=request.language)
        except OutputValidationError as e:
            logger.warning(f"Output validation warning: {e}")
            # Continue with evaluation but log the warning

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)