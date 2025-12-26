"""
Output Validation Module

This module validates generated comment outputs to ensure:
- Format correctness
- Completeness
- Basic quality metrics
"""

import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputValidationError(Exception):
    """Custom exception for output validation errors."""
    pass


# Quality thresholds
MIN_COMMENT_LENGTH = 10  # Minimum comment length in characters
MAX_COMMENT_LENGTH = 5000  # Maximum comment length in characters
MIN_WORD_COUNT = 3  # Minimum number of words
MAX_WORD_COUNT = 500  # Maximum number of words

# Quality indicators
QUALITY_KEYWORDS = {
    'positive': ['function', 'class', 'method', 'parameter', 'return', 'description', 'example'],
    'negative': ['TODO', 'FIXME', 'XXX', 'HACK', 'BUG', 'ERROR']
}

# Invalid patterns in comments
INVALID_PATTERNS = [
    r'^```',  # Code blocks
    r'```$',  # Closing code blocks
    r'^def\s+',  # Function definitions
    r'^class\s+',  # Class definitions
    r'^import\s+',  # Import statements
    r'^#\s*',  # Comment markers (for some languages)
    r'<\|.*?\|>',  # Special tokens
]


def validate_output(comment: str, code: Optional[str] = None, 
                   language: Optional[str] = None) -> bool:
    """
    Validate generated comment output.
    
    This function checks:
    - Format correctness
    - Completeness
    - Basic quality metrics
    
    Args:
        comment: Generated comment string to validate
        code: Optional source code for context validation
        language: Optional programming language for format validation
    
    Returns:
        True if valid, otherwise raises OutputValidationError
    
    Raises:
        OutputValidationError: If validation fails
    
    Example:
        >>> validate_output("This function calculates the sum of two numbers.")
        True
        >>> validate_output("")  # Too short
        OutputValidationError: Comment is too short
    """
    logger.debug(f"Validating output comment (length: {len(comment) if comment else 0})")
    
    # 1. Check if comment is None or empty
    if comment is None:
        raise OutputValidationError("Generated comment is None")
    
    if not isinstance(comment, str):
        raise OutputValidationError(f"Generated comment must be a string, got {type(comment).__name__}")
    
    # 2. Check minimum length
    comment_stripped = comment.strip()
    if len(comment_stripped) < MIN_COMMENT_LENGTH:
        raise OutputValidationError(
            f"Comment is too short (minimum {MIN_COMMENT_LENGTH} characters, "
            f"got {len(comment_stripped)})"
        )
    
    # 3. Check maximum length
    if len(comment_stripped) > MAX_COMMENT_LENGTH:
        raise OutputValidationError(
            f"Comment is too long (maximum {MAX_COMMENT_LENGTH} characters, "
            f"got {len(comment_stripped)})"
        )
    
    # 4. Check word count
    words = comment_stripped.split()
    word_count = len(words)
    if word_count < MIN_WORD_COUNT:
        raise OutputValidationError(
            f"Comment has too few words (minimum {MIN_WORD_COUNT} words, "
            f"got {word_count})"
        )
    
    if word_count > MAX_WORD_COUNT:
        raise OutputValidationError(
            f"Comment has too many words (maximum {MAX_WORD_COUNT} words, "
            f"got {word_count})"
        )
    
    # 5. Check for invalid patterns (code artifacts)
    for pattern in INVALID_PATTERNS:
        if re.search(pattern, comment_stripped, re.MULTILINE | re.IGNORECASE):
            raise OutputValidationError(
                f"Comment contains invalid pattern: {pattern}. "
                "Comment should not contain code artifacts."
            )
    
    # 6. Check for special tokens
    special_tokens = ['<|file_separator|>', '<|fim_prefix|>', '<|fim_suffix|>', 
                      '<|endoftext|>', '<|end|>', '<|startoftext|>']
    for token in special_tokens:
        if token in comment_stripped:
            raise OutputValidationError(
                f"Comment contains special token: {token}. "
                "Comment should not contain model artifacts."
            )
    
    # 7. Check for markdown code fences
    if '```' in comment_stripped:
        raise OutputValidationError(
            "Comment contains markdown code fences. "
            "Comment should be plain text, not code blocks."
        )
    
    # 8. Check for excessive repetition
    if _has_excessive_repetition(comment_stripped):
        raise OutputValidationError(
            "Comment contains excessive repetition. "
            "This may indicate a generation error."
        )
    
    # 9. Check for basic quality indicators
    quality_score = _calculate_quality_score(comment_stripped)
    if quality_score < 0.3:  # Threshold for minimum quality
        logger.warning(f"Low quality score detected: {quality_score}")
        # We log but don't fail - this is a warning, not an error
    
    # 10. Validate format based on language (if provided)
    if language:
        format_issues = _validate_language_format(comment_stripped, language)
        if format_issues:
            logger.warning(f"Format issues for {language}: {format_issues}")
    
    logger.info(f"Output validation passed (length: {len(comment_stripped)}, words: {word_count})")
    return True


def _has_excessive_repetition(text: str, threshold: float = 0.5) -> bool:
    """
    Check if text has excessive repetition.
    
    Args:
        text: Text to check
        threshold: Maximum ratio of repeated content (0.0-1.0)
    
    Returns:
        True if excessive repetition detected
    """
    words = text.split()
    if len(words) < 10:
        return False
    
    # Check for repeated phrases
    word_sequences = {}
    for i in range(len(words) - 2):
        phrase = ' '.join(words[i:i+3])
        word_sequences[phrase] = word_sequences.get(phrase, 0) + 1
    
    if word_sequences:
        max_repetition = max(word_sequences.values())
        repetition_ratio = max_repetition / len(words)
        return repetition_ratio > threshold
    
    return False


def _calculate_quality_score(comment: str) -> float:
    """
    Calculate a basic quality score for the comment.
    
    Args:
        comment: Comment text
    
    Returns:
        Quality score between 0.0 and 1.0
    """
    score = 0.0
    
    # Check for positive keywords
    comment_lower = comment.lower()
    positive_count = sum(1 for keyword in QUALITY_KEYWORDS['positive'] 
                        if keyword in comment_lower)
    negative_count = sum(1 for keyword in QUALITY_KEYWORDS['negative'] 
                        if keyword in comment_lower)
    
    # Positive keywords increase score
    score += min(positive_count * 0.1, 0.5)
    
    # Negative keywords decrease score
    score -= min(negative_count * 0.2, 0.3)
    
    # Length appropriateness (not too short, not too long)
    length_ratio = len(comment) / MAX_COMMENT_LENGTH
    if 0.1 <= length_ratio <= 0.8:
        score += 0.2
    
    # Sentence structure (has periods, not just one long sentence)
    sentence_count = comment.count('.') + comment.count('!') + comment.count('?')
    if sentence_count >= 1:
        score += 0.1
    
    return max(0.0, min(1.0, score))


def _validate_language_format(comment: str, language: str) -> List[str]:
    """
    Validate comment format for specific programming language.
    
    Args:
        comment: Comment text
        language: Programming language
    
    Returns:
        List of format issues (empty if none)
    """
    issues = []
    
    if language == 'python':
        # Python docstrings should not have triple quotes in the content
        if '"""' in comment or "'''" in comment:
            issues.append("Python docstring should not contain triple quotes")
        
        # Check for proper docstring structure
        if comment.count('\n') > 0:
            # Multi-line docstring should have proper structure
            lines = comment.split('\n')
            if len(lines) > 1 and not lines[0].strip():
                issues.append("First line of multi-line docstring should not be empty")
    
    return issues


def validate_output_with_metrics(comment: str, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate output and return validation metrics.
    
    Args:
        comment: Generated comment
        metrics: Optional additional metrics to include
    
    Returns:
        Dictionary with validation results and metrics
    
    Raises:
        OutputValidationError: If validation fails
    """
    # Perform basic validation
    validate_output(comment)
    
    # Calculate additional metrics
    validation_metrics = {
        'length': len(comment),
        'word_count': len(comment.split()),
        'quality_score': _calculate_quality_score(comment),
        'has_repetition': _has_excessive_repetition(comment),
        'timestamp': datetime.now().isoformat(),
    }
    
    if metrics:
        validation_metrics.update(metrics)
    
    return validation_metrics

