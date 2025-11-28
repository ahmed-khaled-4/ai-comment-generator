"""Model configuration management."""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for AI model parameters."""
    
    # Model selection
    model_name: str = "deepseek-coder:6.7b"
    base_url: str = "http://localhost:11434"
    
    # Generation parameters (optimized for deepseek-coder:6.7b)
    temperature: float = 0.4  # Lower for more focused output
    max_tokens: int = 600  # Increased for better generation
    top_p: float = 0.9
    top_k: Optional[int] = None
    repeat_penalty: float = 1.1
    
    # Comment generation defaults
    default_language: str = "python"
    default_comment_type: str = "function"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelConfig to dictionary."""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "default_language": self.default_language,
            "default_comment_type": self.default_comment_type
        }
    
    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Create ModelConfig from environment variables."""
        return cls(
            model_name=os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.4")),
            max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "600")),
            top_p=float(os.getenv("OLLAMA_TOP_P", "0.9")),
        )
    
    @classmethod
    def get_default_configs(cls) -> Dict[str, "ModelConfig"]:
        """Get predefined configurations for different use cases."""
        return {
            "default": cls(
                model_name="deepseek-coder:6.7b",
                temperature=0.4,
                max_tokens=600
            ),
            "fast": cls(
                model_name="llama3.2",
                temperature=0.5,
                max_tokens=300
            ),
            "high_quality": cls(
                model_name="deepseek-coder",
                temperature=0.3,
                max_tokens=800
            ),
            "balanced": cls(
                model_name="qwen2.5-coder",
                temperature=0.7,
                max_tokens=500
            )
        }
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")
        return True
