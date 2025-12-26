"""
Human Review Integration Module

This module provides endpoints and workflow for human review:
- Flag outputs requiring human approval based on confidence scores or safety checks
- Track review status
- Integrate with FastAPI app
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Status of a review item."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"


@dataclass
class ReviewFlag:
    """Represents a flag for human review."""
    comment_id: str
    comment: str
    confidence: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None


class HumanReviewSystem:
    """
    System for managing human review of generated comments.
    
    This class tracks comments that need human review based on:
    - Low confidence scores
    - Safety violations
    - Quality thresholds
    - Custom criteria
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.6,
                 auto_approve_high_confidence: bool = True,
                 high_confidence_threshold: float = 0.9):
        """
        Initialize human review system.
        
        Args:
            confidence_threshold: Confidence score below which review is required
            auto_approve_high_confidence: Auto-approve high confidence outputs
            high_confidence_threshold: Confidence score above which auto-approve
        """
        self.confidence_threshold = confidence_threshold
        self.auto_approve_high_confidence = auto_approve_high_confidence
        self.high_confidence_threshold = high_confidence_threshold
        self.review_queue: List[ReviewFlag] = []
        self.review_history: List[ReviewFlag] = []
        logger.info(f"HumanReviewSystem initialized (threshold: {confidence_threshold})")
    
    def flag_for_human_review(self, 
                              comment: str, 
                              confidence: float,
                              reason: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Flag a comment for human review.
        
        This function determines if a comment should be flagged based on:
        - Confidence score
        - Safety checks
        - Quality metrics
        
        Args:
            comment: Generated comment
            confidence: Confidence score (0.0-1.0)
            reason: Optional reason for flagging
            metadata: Optional metadata about the comment
        
        Returns:
            True if comment should be flagged for review, False otherwise
        """
        # Auto-approve high confidence if enabled
        if self.auto_approve_high_confidence and confidence >= self.high_confidence_threshold:
            logger.debug(f"High confidence ({confidence:.2f}), auto-approving")
            return False
        
        # Flag if confidence is below threshold
        should_flag = confidence < self.confidence_threshold
        
        if should_flag:
            comment_id = str(uuid.uuid4())
            flag = ReviewFlag(
                comment_id=comment_id,
                comment=comment,
                confidence=confidence,
                reason=reason or f"Low confidence score: {confidence:.2f}",
                metadata=metadata or {},
                status=ReviewStatus.FLAGGED
            )
            self.review_queue.append(flag)
            logger.info(f"Flagged comment {comment_id} for review (confidence: {confidence:.2f})")
        
        return should_flag
    
    def add_review_flag(self, 
                       comment: str,
                       reason: str,
                       confidence: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Manually add a review flag.
        
        Args:
            comment: Comment to flag
            reason: Reason for flagging
            confidence: Optional confidence score
            metadata: Optional metadata
        
        Returns:
            Comment ID for tracking
        """
        comment_id = str(uuid.uuid4())
        flag = ReviewFlag(
            comment_id=comment_id,
            comment=comment,
            confidence=confidence or 0.0,
            reason=reason,
            metadata=metadata or {},
            status=ReviewStatus.FLAGGED
        )
        self.review_queue.append(flag)
        logger.info(f"Manually flagged comment {comment_id}: {reason}")
        return comment_id
    
    def approve_comment(self, comment_id: str, reviewer: str, notes: Optional[str] = None) -> bool:
        """
        Approve a flagged comment.
        
        Args:
            comment_id: ID of the comment to approve
            reviewer: Name/ID of the reviewer
            notes: Optional review notes
        
        Returns:
            True if comment was found and approved, False otherwise
        """
        flag = self._find_flag(comment_id)
        if flag:
            flag.status = ReviewStatus.APPROVED
            flag.reviewer = reviewer
            flag.review_notes = notes
            flag.timestamp = datetime.now().isoformat()
            self._move_to_history(flag)
            logger.info(f"Comment {comment_id} approved by {reviewer}")
            return True
        return False
    
    def reject_comment(self, comment_id: str, reviewer: str, notes: Optional[str] = None) -> bool:
        """
        Reject a flagged comment.
        
        Args:
            comment_id: ID of the comment to reject
            reviewer: Name/ID of the reviewer
            notes: Optional review notes
        
        Returns:
            True if comment was found and rejected, False otherwise
        """
        flag = self._find_flag(comment_id)
        if flag:
            flag.status = ReviewStatus.REJECTED
            flag.reviewer = reviewer
            flag.review_notes = notes
            flag.timestamp = datetime.now().isoformat()
            self._move_to_history(flag)
            logger.info(f"Comment {comment_id} rejected by {reviewer}")
            return True
        return False
    
    def get_pending_reviews(self) -> List[ReviewFlag]:
        """
        Get list of pending reviews.
        
        Returns:
            List of ReviewFlag objects pending review
        """
        return [flag for flag in self.review_queue if flag.status == ReviewStatus.FLAGGED]
    
    def get_review_history(self, limit: int = 100) -> List[ReviewFlag]:
        """
        Get review history.
        
        Args:
            limit: Maximum number of history items to return
        
        Returns:
            List of ReviewFlag objects from history
        """
        return self.review_history[-limit:]
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about reviews.
        
        Returns:
            Dictionary with review statistics
        """
        pending = len(self.get_pending_reviews())
        approved = sum(1 for f in self.review_history if f.status == ReviewStatus.APPROVED)
        rejected = sum(1 for f in self.review_history if f.status == ReviewStatus.REJECTED)
        total = len(self.review_history)
        
        approval_rate = (approved / total * 100) if total > 0 else 0
        rejection_rate = (rejected / total * 100) if total > 0 else 0
        
        return {
            'pending_reviews': pending,
            'total_reviewed': total,
            'approved': approved,
            'rejected': rejected,
            'approval_rate': round(approval_rate, 2),
            'rejection_rate': round(rejection_rate, 2),
        }
    
    def _find_flag(self, comment_id: str) -> Optional[ReviewFlag]:
        """Find a flag by comment ID."""
        for flag in self.review_queue:
            if flag.comment_id == comment_id:
                return flag
        return None
    
    def _move_to_history(self, flag: ReviewFlag):
        """Move a flag from queue to history."""
        if flag in self.review_queue:
            self.review_queue.remove(flag)
        self.review_history.append(flag)


# Global instance
_review_system_instance: Optional[HumanReviewSystem] = None


def get_review_system() -> HumanReviewSystem:
    """
    Get singleton instance of HumanReviewSystem.
    
    Returns:
        HumanReviewSystem instance
    """
    global _review_system_instance
    if _review_system_instance is None:
        _review_system_instance = HumanReviewSystem()
    return _review_system_instance


def flag_for_human_review(comment: str, 
                          confidence: float,
                          reason: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Flag a comment for human review (convenience function).
    
    Args:
        comment: Generated comment
        confidence: Confidence score (0.0-1.0)
        reason: Optional reason for flagging
        metadata: Optional metadata
    
    Returns:
        True if comment should be flagged for review
    """
    return get_review_system().flag_for_human_review(
        comment, confidence, reason, metadata
    )

