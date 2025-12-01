"""FastAPI application for AI Comment Generator."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
from models.ai_client import OllamaService
from models.model_config import ModelConfig

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


# Request/Response Models
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


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "AI Comment Generator API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/generate_comment",
            "health": "/health",
            "docs": "/docs"
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
    Generate a comment for the given code.
    
    Args:
        request: CommentRequest with code and parameters
    
    Returns:
        CommentResponse with generated comment
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
