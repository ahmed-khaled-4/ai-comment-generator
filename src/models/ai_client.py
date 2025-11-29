"""AI Client for model integration (Ollama, etc.)."""
import ollama
import logging
import re
from typing import Optional, Dict, Any
from pathlib import Path
from .model_config import ModelConfig

logger = logging.getLogger(__name__)


class OllamaService:
    """Service for interacting with Ollama LLM."""
    
    def __init__(self, model: str = "deepseek-coder:6.7b", base_url: str = "http://localhost:11434", config: Optional[ModelConfig] = None):
        """
        Initialize Ollama service.
        
        Args:
            model: Model name (default: deepseek-coder:6.7b)
            base_url: Ollama server URL
            config: Optional ModelConfig for default parameters
        """
        self.client = ollama.Client(host=base_url)
        self.base_url = base_url
        self.model = model
        self.config = config or ModelConfig()
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama connection and model availability."""
        try:
            models_response = self.client.list()
            
            # Handle Ollama API response structure
            # Response is a ListResponse object with a 'models' attribute
            if hasattr(models_response, 'models'):
                available_models = [m.model if hasattr(m, 'model') else str(m) for m in models_response.models]
            elif isinstance(models_response, dict):
                available_models = [m.get('name', m.get('model', str(m))) for m in models_response.get('models', [])]
            else:
                available_models = [str(m) for m in models_response] if hasattr(models_response, '__iter__') else []
            
            # Check if model exists (handle model:tag format)
            model_base = self.model.split(':')[0]
            model_found = any(model_base in m for m in available_models)
            
            if not model_found:
                logger.warning(
                    f"Model '{self.model}' not found. Available models: {available_models}\n"
                    f"Run: ollama pull {model_base}"
                )
            else:
                logger.info(f"Ollama service initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Could not verify Ollama connection: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
    
    def generate_comment(
        self, 
        code: str, 
        language: str = "python",
        comment_type: str = "function",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a comment for the given code.
        
        Args:
            code: Source code to generate comment for
            language: Programming language
            comment_type: Type of comment (function, class, inline)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Dictionary with generated comment and metadata
        """
        try:
            # Use config defaults if not provided
            temperature = temperature if temperature is not None else self.config.temperature
            max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
            
            # Format prompt based on comment type
            prompt = self._format_prompt(code, language, comment_type)
            
            # Generate using Ollama
            import time
            start_time = time.time()
            
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": self.config.top_p,
                    "stop": [
                         '"""', "'''",
                        "```", "###", "##",
                        "<|file_separator|>", "<|fim_prefix|>", "<|fim_suffix|>",
                        "Example:", "Output:",
                        "Docstring:", "Comment:",
                        "Code:",
                        "def ",
                        "class ",
                    ]

                },
                stream=False  # Disable streaming to prevent repetition
            )
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Extract content from response (ChatResponse supports attribute access)
            generated_comment = response.message.content.strip()
            
            # Clean up the generated comment
            cleaned_comment = self._clean_comment(generated_comment)
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'prompt_eval_count') and response.prompt_eval_count:
                token_usage['prompt_tokens'] = response.prompt_eval_count
            if hasattr(response, 'eval_count') and response.eval_count:
                token_usage['completion_tokens'] = response.eval_count
            if hasattr(response, 'prompt_eval_count') and hasattr(response, 'eval_count'):
                if response.prompt_eval_count and response.eval_count:
                    token_usage['total_tokens'] = response.prompt_eval_count + response.eval_count
            
            return {
                "comment": cleaned_comment,
                "model": self.model,
                "language": language,
                "comment_type": comment_type,
                "metadata": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": self.config.top_p,
                    "latency": round(latency, 3),
                    "prompt_tokens": token_usage.get('prompt_tokens'),
                    "completion_tokens": token_usage.get('completion_tokens'),
                    "total_tokens": token_usage.get('total_tokens')
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating comment: {e}")
            raise
    
    def _clean_comment(self, comment: str) -> str:
        """
        Clean generated comment: remove quotes, duplicates, noise, and special tokens.
        """
        if not comment:
            return ""

        # Remove special tokens (some models use these)
        special_tokens = [
            "<|file_separator|>",
            "<|fim_prefix|>",
            "<|fim_suffix|>",
            "<|fim_middle|>",
            "<|endoftext|>",
            "<|end|>",
            "<|startoftext|>"
        ]
        for token in special_tokens:
            comment = comment.replace(token, "")

        # Remove any quotes the model added
        comment = comment.replace('"""', '').replace("'''", "").strip()

        # Remove unwanted prefixes
        bad_prefixes = ["docstring:", "output:", "comment:", "here is", "here's"]
        for p in bad_prefixes:
            if comment.lower().startswith(p):
                comment = comment[len(p):].strip()

        # Remove lines that are just special tokens or noise
        lines = comment.split("\n")
        cleaned = []
        seen = set()

        for line in lines:
            normalized = line.strip().lower()
            
            # Skip lines that are just special tokens or noise
            if any(token.lower() in normalized for token in special_tokens):
                continue
            if normalized.startswith("you are an expert"):
                continue
            if ".ipynb" in normalized or ".py" in normalized:
                continue
            # Skip lines that start with code patterns
            if normalized.startswith("code:") or normalized.startswith("def ") or normalized.startswith("class "):
                continue
            if normalized.startswith("do not modify") or normalized.startswith("import "):
                continue
            
            if normalized and normalized not in seen:
                cleaned.append(line)
                seen.add(normalized)

        # Remove double blank lines
        final = []
        for line in cleaned:
            if line.strip() == "" and (not final or final[-1].strip() == ""):
                continue
            final.append(line)

        result = "\n".join(final).strip()
        
        # If result is empty or only contains noise, return empty
        if not result or result in ["<|file_separator|>", "<|fim_prefix|>", "<|fim_suffix|>"]:
            return ""
        
        return result
    
    def _format_prompt(self, code: str, language: str, comment_type: str) -> str:
        """
        Format prompt based on comment type.
        """
        if comment_type == "function":
            return f"""You are an expert Python documentation generator.

Generate ONE clean Python docstring for the following code.

STRICT FORMAT (DO NOT MODIFY):

Short description line.

(blank line)

Args:
    parameter_name: description

(blank line)

Returns:
    type: description

RULES:
- Follow EXACTLY the format above.
- Output ONLY the docstring BODY WITHOUT triple quotes.
- NEVER repeat the code.
- NEVER generate more than one docstring.
- NEVER include explanations.
- NEVER say "Docstring:" or anything outside the content.
- Keep it concise and correct.

Code:

{code}

Return ONLY the formatted docstring body."""

        elif comment_type == "class":
            return f"""You are an expert Python documentation generator.

Write a single, complete Python docstring for the following class, including its methods.

Class:
{code}

Docstring format:
Short sentence describing what the class represents.

Attributes:
    attribute_name: description

Methods:
    method_name(args): short description

Rules:
- Output ONLY the docstring text (no triple quotes, no code).
- Infer attributes from __init__ parameters and documented instance variables.
- List important public methods under Methods with a brief description.
- If there are no attributes, write: "Attributes: None"
- If there are no public methods, write: "Methods: None"
- Do NOT repeat the class code or write \"Code:\".

Docstring:"""
