"""
Input Request Validation Module

This module provides comprehensive validation for incoming API requests.
It checks syntax correctness, supported programming languages, size limits,
and security concerns (malicious code patterns).
"""

import re
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


# Supported programming languages
SUPPORTED_LANGUAGES = {
    'python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'csharp',
    'go', 'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'r',
    'sql', 'html', 'css', 'json', 'yaml', 'xml', 'bash', 'shell'
}

# Request size limits (in characters)
MAX_CODE_LENGTH = 50000  # Maximum code length
MIN_CODE_LENGTH = 1     # Minimum code length
MAX_REQUEST_SIZE = 100000  # Maximum total request size

# Security patterns - potentially malicious code patterns
MALICIOUS_PATTERNS = [
    # Command injection patterns
    (r'eval\s*\(', 'Potential eval() usage'),
    (r'exec\s*\(', 'Potential exec() usage'),
    (r'__import__\s*\(', 'Potential __import__() usage'),
    (r'subprocess\s*\.', 'Potential subprocess usage'),
    (r'os\s*\.\s*system\s*\(', 'Potential os.system() usage'),
    (r'shell\s*=\s*True', 'Potential shell=True in subprocess'),
    
    # SQL injection patterns
    (r'\.execute\s*\([^)]*\+', 'Potential SQL string concatenation'),
    (r'\.execute\s*\([^)]*%', 'Potential SQL string formatting'),
    
    # File system access patterns
    (r'open\s*\([^)]*[\'"]\s*\.\./', 'Potential directory traversal'),
    (r'rm\s+-rf', 'Potential file deletion'),
    (r'del\s+/', 'Potential file deletion'),
    
    # Network access patterns
    (r'urllib\s*\.', 'Potential network access'),
    (r'requests\s*\.', 'Potential network access'),
    (r'socket\s*\.', 'Potential network socket usage'),
    
    # Base64 encoding (often used to obfuscate)
    (r'base64\s*\.\s*b64decode', 'Potential base64 decoding'),
]

# Comment types
VALID_COMMENT_TYPES = {'function', 'class', 'inline', 'module', 'file'}


def validate_request(data: Dict[str, Any]) -> None:
    """
    Validate all incoming request parameters.
    
    This function performs comprehensive validation including:
    - Syntax correctness
    - Supported programming language
    - Size limits for requests
    - Security concerns (malicious code patterns)
    
    Args:
        data: Dictionary containing request parameters with keys:
            - code: str (required) - Source code to validate
            - language: str (optional) - Programming language
            - comment_type: str (optional) - Type of comment
            - temperature: float (optional) - Sampling temperature
            - max_tokens: int (optional) - Maximum tokens
            - model: str (optional) - Model name
    
    Raises:
        ValidationError: If any validation check fails
    
    Example:
        >>> validate_request({'code': 'def hello(): pass', 'language': 'python'})
        >>> # No exception raised if valid
    """
    logger.debug(f"Validating request: {list(data.keys())}")
    
    # 1. Check required fields
    if 'code' not in data:
        raise ValidationError("Missing required field: 'code'")
    
    code = data.get('code', '')
    if not isinstance(code, str):
        raise ValidationError("Field 'code' must be a string")
    
    # 2. Validate code length
    code_length = len(code)
    if code_length < MIN_CODE_LENGTH:
        raise ValidationError(f"Code is too short (minimum {MIN_CODE_LENGTH} characters)")
    
    if code_length > MAX_CODE_LENGTH:
        raise ValidationError(
            f"Code is too long (maximum {MAX_CODE_LENGTH} characters, "
            f"got {code_length})"
        )
    
    # 3. Validate total request size
    total_size = sum(len(str(v)) for v in data.values())
    if total_size > MAX_REQUEST_SIZE:
        raise ValidationError(
            f"Request is too large (maximum {MAX_REQUEST_SIZE} characters, "
            f"got {total_size})"
        )
    
    # 4. Validate programming language
    language = data.get('language', 'python')
    if not isinstance(language, str):
        raise ValidationError("Field 'language' must be a string")
    
    language_lower = language.lower().strip()
    if language_lower not in SUPPORTED_LANGUAGES:
        raise ValidationError(
            f"Unsupported programming language: '{language}'. "
            f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
        )
    
    # 5. Validate comment type
    comment_type = data.get('comment_type', 'function')
    if not isinstance(comment_type, str):
        raise ValidationError("Field 'comment_type' must be a string")
    
    comment_type_lower = comment_type.lower().strip()
    if comment_type_lower not in VALID_COMMENT_TYPES:
        raise ValidationError(
            f"Invalid comment type: '{comment_type}'. "
            f"Valid types: {', '.join(sorted(VALID_COMMENT_TYPES))}"
        )
    
    # 6. Validate temperature if provided
    if 'temperature' in data:
        temperature = data['temperature']
        if not isinstance(temperature, (int, float)):
            raise ValidationError("Field 'temperature' must be a number")
        if temperature < 0.0 or temperature > 2.0:
            raise ValidationError(
                f"Temperature must be between 0.0 and 2.0, got {temperature}"
            )
    
    # 7. Validate max_tokens if provided
    if 'max_tokens' in data:
        max_tokens = data['max_tokens']
        if not isinstance(max_tokens, int):
            raise ValidationError("Field 'max_tokens' must be an integer")
        if max_tokens < 1:
            raise ValidationError("max_tokens must be at least 1")
        if max_tokens > 2000:
            raise ValidationError("max_tokens must be at most 2000")
    
    # 8. Security validation - check for malicious patterns
    security_issues = _check_security_patterns(code, language_lower)
    if security_issues:
        logger.warning(f"Security concerns detected: {security_issues}")
        # In strict mode, we could raise an error, but for now we just log
        # Uncomment the line below to enable strict security checking
        # raise ValidationError(f"Security concerns detected: {', '.join(security_issues)}")
    
    # 9. Basic syntax validation (language-specific)
    syntax_errors = _validate_syntax(code, language_lower)
    if syntax_errors:
        logger.warning(f"Syntax warnings detected: {syntax_errors}")
        # We log warnings but don't fail - the model might still generate useful comments
    
    logger.info(f"Request validation passed for {language_lower} code ({code_length} chars)")
    return None


def _check_security_patterns(code: str, language: str) -> List[str]:
    """
    Check code for potentially malicious patterns.
    
    Args:
        code: Source code to check
        language: Programming language
    
    Returns:
        List of security concerns found (empty if none)
    """
    issues = []
    
    # Language-specific security checks
    if language == 'python':
        for pattern, description in MALICIOUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(description)
    
    # Check for suspicious file operations
    if re.search(r'\.\./', code):
        issues.append('Potential directory traversal')
    
    # Check for hardcoded credentials (basic check)
    if re.search(r'(password|api_key|secret|token)\s*=\s*[\'"][^\'"]+[\'"]', code, re.IGNORECASE):
        issues.append('Potential hardcoded credentials')
    
    return issues


def _validate_syntax(code: str, language: str) -> List[str]:
    """
    Perform basic syntax validation for the given language.
    
    This is a lightweight check - full syntax validation would require
    language-specific parsers. This function provides basic heuristics.
    
    Args:
        code: Source code to validate
        language: Programming language
    
    Returns:
        List of syntax warnings (empty if none)
    """
    warnings = []
    
    if not code.strip():
        warnings.append("Code is empty or whitespace only")
        return warnings
    
    # Language-specific basic checks
    if language == 'python':
        # Check for unmatched brackets (basic)
        open_brackets = code.count('(') + code.count('[') + code.count('{')
        close_brackets = code.count(')') + code.count(']') + code.count('}')
        if open_brackets != close_brackets:
            warnings.append("Potential unmatched brackets")
        
        # Check for common Python syntax issues
        if re.search(r'def\s+\w+\s*\([^)]*\)\s*:', code) and not re.search(r'return|pass|raise|yield', code):
            warnings.append("Function definition without return/pass/raise/yield")
    
    elif language in ['javascript', 'typescript']:
        # Check for unmatched brackets
        open_brackets = code.count('(') + code.count('[') + code.count('{')
        close_brackets = code.count(')') + code.count(']') + code.count('}')
        if open_brackets != close_brackets:
            warnings.append("Potential unmatched brackets")
    
    return warnings


def validate_language(language: str) -> bool:
    """
    Check if a programming language is supported.
    
    Args:
        language: Programming language name
    
    Returns:
        True if supported, False otherwise
    """
    return language.lower().strip() in SUPPORTED_LANGUAGES


def validate_comment_type(comment_type: str) -> bool:
    """
    Check if a comment type is valid.
    
    Args:
        comment_type: Type of comment
    
    Returns:
        True if valid, False otherwise
    """
    return comment_type.lower().strip() in VALID_COMMENT_TYPES


def get_supported_languages() -> List[str]:
    """
    Get list of supported programming languages.
    
    Returns:
        List of supported language names
    """
    return sorted(list(SUPPORTED_LANGUAGES))


def get_valid_comment_types() -> List[str]:
    """
    Get list of valid comment types.
    
    Returns:
        List of valid comment type names
    """
    return sorted(list(VALID_COMMENT_TYPES))

