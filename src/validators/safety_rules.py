"""
Safety Rules and Constraints Module

This module defines and enforces safety constraints including:
- Content filters
- Length limits
- Quality thresholds
"""

import re
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyViolationError(Exception):
    """Custom exception for safety rule violations."""
    pass


class SafetyLevel(Enum):
    """Safety enforcement levels."""
    PERMISSIVE = "permissive"  # Minimal checks
    STANDARD = "standard"      # Default safety checks
    STRICT = "strict"          # Maximum safety checks


@dataclass
class SafetyConfig:
    """Configuration for safety rules."""
    level: SafetyLevel = SafetyLevel.STANDARD
    max_length: int = 5000
    min_length: int = 10
    block_profanity: bool = True
    block_pii: bool = True
    require_quality_threshold: float = 0.3
    block_code_injection: bool = True
    block_urls: bool = False  # Set to True to block URLs in comments
    block_emails: bool = False  # Set to True to block emails in comments


# Profanity filter (basic list - can be expanded)
PROFANITY_WORDS = {
    'damn', 'hell',  # Add more as required
}

# PII patterns
PII_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN pattern'),
    (r'\b\d{16}\b', 'Credit card pattern'),
    (r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b', 'Brazilian CPF pattern'),
    (r'\b[A-Z]{2}\d{6}\b', 'Passport pattern'),
]

# Code injection patterns
CODE_INJECTION_PATTERNS = [
    (r'<script[^>]*>', 'Script tag'),
    (r'javascript:', 'JavaScript protocol'),
    (r'on\w+\s*=', 'Event handler'),
    (r'eval\s*\(', 'Eval function'),
    (r'exec\s*\(', 'Exec function'),
]


def sanitize_comment(comment: str) -> str:
    """
    Replace forbidden words in a comment with [redacted].
    
    Args:
        comment: Original comment text
    
    Returns:
        Sanitized comment
    """
    if not comment:
        return comment

    # Replace each profanity word with [redacted]
    pattern = re.compile(r'\b(' + '|'.join(PROFANITY_WORDS) + r')\b', re.IGNORECASE)
    return pattern.sub("[redacted]", comment)


def enforce_safety_rules(comment: str, config: Optional[SafetyConfig] = None) -> bool:
    """
    Enforce safety rules on a comment.
    
    This function checks:
    - Content filters (profanity, PII)
    - Length limits
    - Quality thresholds
    - Code injection patterns
    
    Args:
        comment: Comment text to validate
        config: Optional SafetyConfig (uses default if not provided)
    
    Returns:
        True if comment passes all safety rules
    
    Raises:
        SafetyViolationError: If any safety rule is violated
    """
    if config is None:
        config = SafetyConfig()
    
    # --- Sanitize comment first ---
    comment = sanitize_comment(comment)
    
    logger.debug(f"Enforcing safety rules (level: {config.level.value})")
    
    violations = []
    
    # 1. Length checks
    comment_length = len(comment.strip())
    if comment_length < config.min_length:
        violations.append(f"Comment too short (minimum {config.min_length} characters)")
    
    if comment_length > config.max_length:
        violations.append(f"Comment too long (maximum {config.max_length} characters)")
    
    # 2. Profanity filter (if enabled)
    if config.block_profanity:
        profanity_found = _check_profanity(comment)
        if profanity_found:
            violations.append(f"Profanity detected: {profanity_found}")
    
    # 3. PII detection (if enabled)
    if config.block_pii:
        pii_found = _check_pii(comment)
        if pii_found:
            violations.append(f"PII detected: {pii_found}")
    
    # 4. Code injection patterns (if enabled)
    if config.block_code_injection:
        injection_found = _check_code_injection(comment)
        if injection_found:
            violations.append(f"Code injection pattern detected: {injection_found}")
    
    # 5. URL blocking (if enabled)
    if config.block_urls:
        urls_found = _check_urls(comment)
        if urls_found:
            violations.append(f"URLs detected: {urls_found}")
    
    # 6. Email blocking (if enabled)
    if config.block_emails:
        emails_found = _check_emails(comment)
        if emails_found:
            violations.append(f"Email addresses detected: {emails_found}")
    
    # 7. Quality threshold (if required)
    if config.require_quality_threshold > 0:
        quality_score = _calculate_quality_score(comment)
        if quality_score < config.require_quality_threshold:
            violations.append(
                f"Quality score too low (minimum {config.require_quality_threshold}, "
                f"got {quality_score:.2f})"
            )
    
    # 8. Level-specific checks
    if config.level == SafetyLevel.STRICT:
        # Additional strict checks
        if _has_suspicious_patterns(comment):
            violations.append("Suspicious patterns detected (strict mode)")
    
    # Raise error if violations found
    if violations:
        error_message = "Safety rule violations: " + "; ".join(violations)
        logger.warning(f"Safety violations: {error_message}")
        raise SafetyViolationError(error_message)
    
    logger.info("Safety rules passed")
    return True


def _check_profanity(text: str) -> Optional[str]:
    # Use word boundaries (\b) to match whole words only
    text_lower = text.lower()
    found = []
    for word in PROFANITY_WORDS:
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text_lower):
            found.append(word)
    return ', '.join(found) if found else None


def _check_pii(text: str) -> Optional[str]:
    found = []
    for pattern, description in PII_PATTERNS:
        if re.search(pattern, text):
            found.append(description)
    return ', '.join(found) if found else None


def _check_code_injection(text: str) -> Optional[str]:
    found = []
    for pattern, description in CODE_INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(description)
    return ', '.join(found) if found else None


def _check_urls(text: str) -> Optional[str]:
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    urls = re.findall(url_pattern, text)
    return ', '.join(urls[:3]) if urls else None


def _check_emails(text: str) -> Optional[str]:
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return ', '.join(emails[:3]) if emails else None


def _has_suspicious_patterns(text: str) -> bool:
    special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0
    if special_char_ratio > 0.3:
        return True
    if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text):
        return True
    return False


def _calculate_quality_score(comment: str) -> float:
    if not comment or not comment.strip():
        return 0.0
    
    score = 0.5  # Base score
    
    # Length appropriateness
    length = len(comment)
    if 50 <= length <= 500:
        score += 0.2
    elif 20 <= length < 50 or 500 < length <= 1000:
        score += 0.1
    
    # Sentence structure
    sentence_count = comment.count('.') + comment.count('!') + comment.count('?')
    if sentence_count >= 1:
        score += 0.1
    
    # Word diversity
    words = comment.split()
    unique_words = len(set(words))
    if len(words) > 0:
        diversity = unique_words / len(words)
        score += diversity * 0.2
    
    return min(1.0, max(0.0, score))


def get_safety_config(level: str = "standard") -> SafetyConfig:
    level_map = {
        "permissive": SafetyLevel.PERMISSIVE,
        "standard": SafetyLevel.STANDARD,
        "strict": SafetyLevel.STRICT,
    }
    
    safety_level = level_map.get(level.lower(), SafetyLevel.STANDARD)
    
    if safety_level == SafetyLevel.PERMISSIVE:
        return SafetyConfig(
            level=safety_level,
            block_profanity=False,
            block_pii=False,
            require_quality_threshold=0.1,
        )
    elif safety_level == SafetyLevel.STRICT:
        return SafetyConfig(
            level=safety_level,
            block_profanity=True,
            block_pii=True,
            require_quality_threshold=0.5,
            block_urls=True,
            block_emails=True,
        )
    else:  # STANDARD
        return SafetyConfig(level=safety_level)
