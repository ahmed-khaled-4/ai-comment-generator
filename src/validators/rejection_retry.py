"""
Automatic Rejection and Retry Module

This module implements automatic rejection of unsafe or low-quality outputs:
- Configurable retry policies
- Fallback strategies
- Retry until safe or return fallback
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RetryExhaustedError(Exception):
    """Exception raised when retry attempts are exhausted."""
    pass


class RetryStrategy(Enum):
    """Retry strategies."""
    IMMEDIATE = "immediate"  # Retry immediately
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential backoff
    LINEAR_BACKOFF = "linear_backoff"  # Linear backoff
    FIXED_DELAY = "fixed_delay"  # Fixed delay between retries


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_delay: float = 0.1  # Initial delay in seconds
    max_delay: float = 5.0  # Maximum delay in seconds
    backoff_multiplier: float = 2.0  # For exponential backoff
    fallback_comment: str = "Comment generation failed after retries."
    enable_fallback: bool = True


class RejectionRetrySystem:
    """
    System for handling automatic rejection and retry of unsafe or low-quality outputs.
    
    This class manages:
    - Retry policies
    - Fallback strategies
    - Retry attempts tracking
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize rejection retry system.
        
        Args:
            config: Optional RetryConfig (uses default if not provided)
        """
        self.config = config or RetryConfig()
        self.retry_history: List[Dict[str, Any]] = []
        logger.info(f"RejectionRetrySystem initialized (max_retries: {self.config.max_retries})")
    
    def process_with_retry(self, 
                           comment: str,
                           validation_func: Callable[[str], bool],
                           generation_func: Optional[Callable[[], str]] = None) -> str:
        """
        Process comment with retry logic until safe or return fallback.
        
        This function:
        1. Validates the comment
        2. If invalid, retries generation (if generation_func provided)
        3. Returns safe comment or fallback after max retries
        
        Args:
            comment: Initial comment to validate
            validation_func: Function that validates comment and returns True if valid,
                           raises exception if invalid
            generation_func: Optional function to generate new comment on retry
        
        Returns:
            Valid comment string (or fallback if retries exhausted)
        
        Raises:
            RetryExhaustedError: If retries exhausted and fallback disabled
        
        Example:
            >>> def validate(comment):
            ...     if len(comment) < 10:
            ...         raise ValueError("Too short")
            ...     return True
            >>> system = RejectionRetrySystem()
            >>> result = system.process_with_retry("short", validate)
            >>> # Returns fallback or retries
        """
        attempt = 0
        current_comment = comment
        
        while attempt <= self.config.max_retries:
            try:
                # Validate current comment
                validation_func(current_comment)
                logger.info(f"Comment validated successfully on attempt {attempt + 1}")
                
                # Record successful validation
                self._record_attempt(attempt, current_comment, success=True)
                return current_comment
                
            except Exception as e:
                logger.warning(f"Validation failed on attempt {attempt + 1}: {e}")
                self._record_attempt(attempt, current_comment, success=False, error=str(e))
                
                # Check if we have retries left
                if attempt < self.config.max_retries:
                    # Wait before retry (if not first attempt)
                    if attempt > 0:
                        delay = self._calculate_delay(attempt)
                        logger.debug(f"Waiting {delay:.2f}s before retry {attempt + 1}")
                        time.sleep(delay)
                    
                    # Generate new comment if generation function provided
                    if generation_func:
                        try:
                            current_comment = generation_func()
                            logger.info(f"Generated new comment for retry {attempt + 1}")
                        except Exception as gen_error:
                            logger.error(f"Generation failed on retry {attempt + 1}: {gen_error}")
                            # Continue with previous comment
                    else:
                        # No generation function, can't retry
                        logger.warning("No generation function provided, cannot retry")
                        break
                    
                    attempt += 1
                else:
                    # Retries exhausted
                    logger.error(f"All {self.config.max_retries} retry attempts exhausted")
                    break
        
        # All retries exhausted
        if self.config.enable_fallback:
            logger.info("Returning fallback comment")
            return self.config.fallback_comment
        else:
            raise RetryExhaustedError(
                f"Failed to generate valid comment after {self.config.max_retries} retries"
            )
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry based on strategy.
        
        Args:
            attempt: Current attempt number (0-indexed)
        
        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.initial_delay * (self.config.backoff_multiplier ** attempt)
            return min(delay, self.config.max_delay)
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.initial_delay * (attempt + 1)
            return min(delay, self.config.max_delay)
        
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            return min(self.config.initial_delay, self.config.max_delay)
        
        else:
            return 0.0
    
    def _record_attempt(self, attempt: int, comment: str, success: bool, error: Optional[str] = None):
        """Record a retry attempt."""
        record = {
            'attempt': attempt + 1,
            'comment': comment[:100] + '...' if len(comment) > 100 else comment,
            'success': success,
            'error': error,
            'timestamp': time.time()
        }
        self.retry_history.append(record)
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about retry attempts.
        
        Returns:
            Dictionary with retry statistics
        """
        if not self.retry_history:
            return {
                'total_attempts': 0,
                'successful_attempts': 0,
                'failed_attempts': 0,
                'success_rate': 0.0,
                'average_attempts_per_retry': 0.0
            }
        
        total = len(self.retry_history)
        successful = sum(1 for r in self.retry_history if r['success'])
        failed = total - successful
        
        # Calculate average attempts per retry cycle
        # Group by retry cycles (consecutive attempts)
        cycles = []
        current_cycle = []
        for record in self.retry_history:
            if record['success']:
                current_cycle.append(record)
                cycles.append(current_cycle)
                current_cycle = []
            else:
                current_cycle.append(record)
        
        if current_cycle:
            cycles.append(current_cycle)
        
        avg_attempts = sum(len(cycle) for cycle in cycles) / len(cycles) if cycles else 0
        
        return {
            'total_attempts': total,
            'successful_attempts': successful,
            'failed_attempts': failed,
            'success_rate': round((successful / total * 100) if total > 0 else 0, 2),
            'average_attempts_per_retry': round(avg_attempts, 2),
            'retry_cycles': len(cycles)
        }


# Global instance
_retry_system_instance: Optional[RejectionRetrySystem] = None


def get_retry_system(config: Optional[RetryConfig] = None) -> RejectionRetrySystem:
    """
    Get singleton instance of RejectionRetrySystem.
    
    Args:
        config: Optional RetryConfig (only used on first call)
    
    Returns:
        RejectionRetrySystem instance
    """
    global _retry_system_instance
    if _retry_system_instance is None:
        _retry_system_instance = RejectionRetrySystem(config)
    return _retry_system_instance


def process_with_retry(comment: str,
                      validation_func: Callable[[str], bool],
                      generation_func: Optional[Callable[[], str]] = None,
                      config: Optional[RetryConfig] = None) -> str:
    """
    Process comment with retry logic (convenience function).
    
    Args:
        comment: Initial comment to validate
        validation_func: Function that validates comment
        generation_func: Optional function to generate new comment
        config: Optional RetryConfig
    
    Returns:
        Valid comment string
    """
    if config:
        system = RejectionRetrySystem(config)
    else:
        system = get_retry_system()
    
    return system.process_with_retry(comment, validation_func, generation_func)

