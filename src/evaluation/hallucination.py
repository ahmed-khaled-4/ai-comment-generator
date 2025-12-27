# src/evaluation/hallucination.py - Hallucination detection for generated comments
"""
Hallucination Detection Module

This module detects various types of hallucinations in AI-generated code comments:
- Empty or too short comments
- Code repetition (comment copies code)
- Low quality scores (BLEU, ROUGE, BERTScore)
- Suspicious patterns (markdown, code fences, etc.)
- Inconsistencies between code and comment
"""

from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Detects hallucinations and quality issues in generated code comments.
    
    This detector identifies:
    - Empty or insufficient comments
    - Comments that repeat too much of the original code
    - Low-quality outputs based on metrics
    - Suspicious patterns indicating model artifacts
    - Inconsistencies between code and comment content
    """
    
    def __init__(
        self,
        min_comment_length: int = 10,
        max_code_overlap_ratio: float = 0.7,
        min_bleu_threshold: float = 0.1,
        min_rouge_threshold: float = 0.1,
        min_bertscore_threshold: float = 0.3
    ):
        """
        Initialize hallucination detector with thresholds.
        
        Args:
            min_comment_length: Minimum acceptable comment length
            max_code_overlap_ratio: Maximum ratio of code words in comment (0.0-1.0)
            min_bleu_threshold: Minimum BLEU score to consider valid
            min_rouge_threshold: Minimum ROUGE score to consider valid
            min_bertscore_threshold: Minimum BERTScore to consider valid
        """
        self.min_comment_length = min_comment_length
        self.max_code_overlap_ratio = max_code_overlap_ratio
        self.min_bleu_threshold = min_bleu_threshold
        self.min_rouge_threshold = min_rouge_threshold
        self.min_bertscore_threshold = min_bertscore_threshold
        
        # Suspicious patterns that indicate model artifacts
        self.suspicious_patterns = [
            r'```',  # Code fences
            r'<\|',  # Special tokens
            r'\|>',  # Special tokens
            r'###\s+',  # Markdown headers
            r'you are an expert',
            r'as an ai',
            r'here is the',
            r'here\'s the',
            r'this is a',
            r'note:',
            r'important:',
        ]
    
    def analyze(
        self,
        code: str,
        comment: str,
        score_metrics: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Analyze a generated comment for hallucinations and quality issues.
        
        Args:
            code: Original source code
            comment: Generated comment to analyze
            score_metrics: Optional dict with metric scores (bleu, rouge1, rougeL, bertscore_f1, etc.)
        
        Returns:
            List of flag strings indicating detected issues (e.g., ["TOO_SHORT", "LOW_BLEU"])
            Empty list if no issues detected.
        """
        flags = []
        
        if not comment or not comment.strip():
            flags.append("EMPTY_COMMENT")
            return flags
        
        comment = comment.strip()
        
        # Check 1: Comment length
        if len(comment) < self.min_comment_length:
            flags.append("TOO_SHORT")
        
        # Check 2: Code repetition (hallucination indicator)
        code_overlap = self._check_code_repetition(code, comment)
        if code_overlap > self.max_code_overlap_ratio:
            flags.append("HIGH_CODE_OVERLAP")
        
        # Check 3: Suspicious patterns (model artifacts)
        if self._check_suspicious_patterns(comment):
            flags.append("SUSPICIOUS_PATTERNS")
        
        # Check 4: Low quality metrics (if provided)
        if score_metrics:
            if score_metrics.get('bleu', 1.0) < self.min_bleu_threshold:
                flags.append("LOW_BLEU")
            
            if score_metrics.get('rouge1', 1.0) < self.min_rouge_threshold:
                flags.append("LOW_ROUGE")
            
            if score_metrics.get('bertscore_f1', 1.0) < self.min_bertscore_threshold:
                flags.append("LOW_BERTSCORE")
        
        # Check 5: Inconsistency between code and comment
        if self._check_inconsistency(code, comment):
            flags.append("CODE_COMMENT_MISMATCH")
        
        # Check 6: Comment appears to be incomplete or truncated
        if self._check_incomplete(comment):
            flags.append("INCOMPLETE_COMMENT")
        
        return flags
    
    def _check_code_repetition(self, code: str, comment: str) -> float:
        """
        Check if comment repeats too much of the original code.
        
        Returns:
            Ratio of code words found in comment (0.0-1.0)
        """
        # Normalize both code and comment
        code_lower = code.lower()
        comment_lower = comment.lower()
        
        # Extract words (remove punctuation, split on whitespace)
        code_words = set(re.findall(r'\b\w+\b', code_lower))
        comment_words = set(re.findall(r'\b\w+\b', comment_lower))
        
        if not comment_words:
            return 0.0
        
        # Calculate overlap ratio
        overlap = len(code_words & comment_words)
        overlap_ratio = overlap / len(comment_words) if comment_words else 0.0
        
        return overlap_ratio
    
    def _check_suspicious_patterns(self, comment: str) -> bool:
        """
        Check for suspicious patterns that indicate model artifacts.
        
        Returns:
            True if suspicious patterns found
        """
        comment_lower = comment.lower()
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, comment_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _check_inconsistency(self, code: str, comment: str) -> bool:
        """
        Check for inconsistencies between code and comment.
        
        For example:
        - Code has function but comment doesn't mention it
        - Code has parameters but comment doesn't document them
        - Code returns something but comment says it doesn't
        
        Returns:
            True if inconsistency detected
        """
        code_lower = code.lower()
        comment_lower = comment.lower()
        
        # Check if code has "def" but comment doesn't mention function/method
        if 'def ' in code_lower and 'function' not in comment_lower and 'method' not in comment_lower:
            # But allow if comment has "Args" or "Returns" (docstring format)
            if 'args' not in comment_lower and 'returns' not in comment_lower:
                return True
        
        # Check if code has "class" but comment doesn't mention class
        if 'class ' in code_lower and 'class' not in comment_lower:
            return True
        
        # Check if code has parameters but comment doesn't mention them
        # This is a simple heuristic - could be improved
        if 'def ' in code_lower:
            # Extract function parameters
            func_match = re.search(r'def\s+\w+\s*\((.*?)\)', code_lower)
            if func_match:
                params = func_match.group(1).strip()
                if params and params != 'self':
                    # Check if comment mentions parameters
                    if 'args' not in comment_lower and 'param' not in comment_lower:
                        # But only flag if comment is substantial (not just a short description)
                        if len(comment) > 50:
                            return True
        
        return False
    
    def _check_incomplete(self, comment: str) -> bool:
        """
        Check if comment appears incomplete or truncated.
        
        Returns:
            True if comment seems incomplete
        """
        # Check for trailing ellipsis or incomplete sentences
        if comment.endswith('...') or comment.endswith('..'):
            return True
        
        # Check if comment ends mid-sentence (no punctuation)
        if len(comment) > 20 and not comment[-1] in '.!?:':
            # But allow if it's a list format (Args/Returns)
            if not any(keyword in comment.lower() for keyword in ['args:', 'returns:', 'attributes:']):
                return True
        
        # Check for very short comments that seem incomplete
        if len(comment) < 30 and not any(keyword in comment.lower() for keyword in ['args', 'returns', 'description']):
            return True
        
        return False

