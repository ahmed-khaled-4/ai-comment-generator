"""Ethical governance framework for multi-agent system.

This module implements the STEA framework:
- Safety: Ensuring outputs are safe and appropriate
- Transparency: Making agent decisions visible and understandable
- Explainability: Providing reasoning for agent actions
- Accountability: Tracking and logging all agent decisions
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class GovernanceLevel(Enum):
    """Governance strictness levels."""
    PERMISSIVE = "permissive"  # Minimal checks
    STANDARD = "standard"      # Normal operation
    STRICT = "strict"          # Maximum safety checks


class SafetyViolationType(Enum):
    """Types of safety violations."""
    EMPTY_OUTPUT = "empty_output"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    HALLUCINATION = "hallucination"
    QUALITY_THRESHOLD = "quality_threshold"
    CONTENT_FILTER = "content_filter"
    INCONSISTENCY = "inconsistency"


@dataclass
class GovernanceConfig:
    """Configuration for governance controls."""
    level: GovernanceLevel = GovernanceLevel.STANDARD
    
    # Safety thresholds
    min_comment_length: int = 10
    max_comment_length: int = 2000
    min_quality_score: float = 0.3
    
    # Transparency settings
    log_all_decisions: bool = True
    provide_explanations: bool = True
    
    # Accountability
    require_approval_threshold: float = 0.5  # If confidence < this, require human approval
    track_agent_performance: bool = True


@dataclass
class SafetyCheck:
    """Result of a safety check."""
    passed: bool
    violation_type: Optional[SafetyViolationType]
    severity: str  # "low", "medium", "high"
    message: str
    details: Dict[str, Any]


@dataclass
class GovernanceDecision:
    """A governance decision with full transparency."""
    decision_id: str
    timestamp: str
    agent_name: str
    action: str
    approved: bool
    reasoning: str
    safety_checks: List[SafetyCheck]
    confidence_score: float
    requires_human_review: bool
    metadata: Dict[str, Any]


class GovernanceFramework:
    """
    Ethical governance framework implementing STEA principles.
    
    This framework ensures:
    1. Safety: All outputs pass safety checks
    2. Transparency: All decisions are logged and visible
    3. Explainability: Reasoning is provided for each decision
    4. Accountability: Full audit trail of agent actions
    """
    
    def __init__(self, config: Optional[GovernanceConfig] = None):
        """
        Initialize governance framework.
        
        Args:
            config: Governance configuration (uses defaults if not provided)
        """
        self.config = config or GovernanceConfig()
        self.decisions: List[GovernanceDecision] = []
        logger.info(f"Governance framework initialized with level: {self.config.level.value}")
    
    def check_safety(
        self,
        output: str,
        code: str,
        metadata: Dict[str, Any]
    ) -> List[SafetyCheck]:
        """
        Perform comprehensive safety checks on generated output.
        
        Args:
            output: Generated comment
            code: Original code
            metadata: Additional metadata (quality scores, etc.)
        
        Returns:
            List of safety check results
        """
        checks = []
        
        # Check 1: Empty output
        if not output or not output.strip():
            checks.append(SafetyCheck(
                passed=False,
                violation_type=SafetyViolationType.EMPTY_OUTPUT,
                severity="high",
                message="Generated output is empty",
                details={"output_length": 0}
            ))
        
        # Check 2: Length constraints
        output_length = len(output)
        if output_length < self.config.min_comment_length:
            checks.append(SafetyCheck(
                passed=False,
                violation_type=SafetyViolationType.TOO_SHORT,
                severity="medium",
                message=f"Output too short ({output_length} < {self.config.min_comment_length})",
                details={"output_length": output_length, "min_required": self.config.min_comment_length}
            ))
        
        if output_length > self.config.max_comment_length:
            checks.append(SafetyCheck(
                passed=False,
                violation_type=SafetyViolationType.TOO_LONG,
                severity="medium",
                message=f"Output too long ({output_length} > {self.config.max_comment_length})",
                details={"output_length": output_length, "max_allowed": self.config.max_comment_length}
            ))
        
        # Check 3: Quality threshold
        quality_score = metadata.get("quality_score", 1.0)
        if quality_score < self.config.min_quality_score:
            checks.append(SafetyCheck(
                passed=False,
                violation_type=SafetyViolationType.QUALITY_THRESHOLD,
                severity="high",
                message=f"Quality score below threshold ({quality_score:.2f} < {self.config.min_quality_score})",
                details={"quality_score": quality_score, "threshold": self.config.min_quality_score}
            ))
        
        # Check 4: Content filters (basic)
        suspicious_patterns = [
            "you are an expert",
            "as an ai",
            "i cannot",
            "i don't have",
            "<|",
            "|>",
            "###",
            "```"
        ]
        for pattern in suspicious_patterns:
            if pattern in output.lower():
                checks.append(SafetyCheck(
                    passed=False,
                    violation_type=SafetyViolationType.CONTENT_FILTER,
                    severity="medium",
                    message=f"Suspicious pattern detected: '{pattern}'",
                    details={"pattern": pattern}
                ))
                break
        
        # If no violations found, add a passing check
        if not checks:
            checks.append(SafetyCheck(
                passed=True,
                violation_type=None,
                severity="none",
                message="All safety checks passed",
                details={}
            ))
        
        return checks
    
    def make_decision(
        self,
        agent_name: str,
        action: str,
        output: str,
        code: str,
        confidence_score: float,
        metadata: Dict[str, Any]
    ) -> GovernanceDecision:
        """
        Make a governance decision with full transparency and explainability.
        
        Args:
            agent_name: Name of the agent making the action
            action: Action being performed
            output: Generated output
            code: Original code
            confidence_score: Agent's confidence in the output
            metadata: Additional metadata
        
        Returns:
            GovernanceDecision with full reasoning and safety checks
        """
        decision_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.now().isoformat()
        
        # Perform safety checks
        safety_checks = self.check_safety(output, code, metadata)
        
        # Determine if approved
        has_violations = any(not check.passed for check in safety_checks)
        approved = not has_violations
        
        # Determine if human review required
        requires_human_review = (
            confidence_score < self.config.require_approval_threshold or
            any(check.severity == "high" and not check.passed for check in safety_checks)
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            approved, confidence_score, safety_checks, requires_human_review
        )
        
        # Create decision
        decision = GovernanceDecision(
            decision_id=decision_id,
            timestamp=timestamp,
            agent_name=agent_name,
            action=action,
            approved=approved,
            reasoning=reasoning,
            safety_checks=safety_checks,
            confidence_score=confidence_score,
            requires_human_review=requires_human_review,
            metadata=metadata
        )
        
        # Log decision (Accountability)
        if self.config.log_all_decisions:
            self.decisions.append(decision)
            logger.info(f"Decision {decision_id}: {action} by {agent_name} - {'APPROVED' if approved else 'REJECTED'}")
        
        return decision
    
    def _generate_reasoning(
        self,
        approved: bool,
        confidence_score: float,
        safety_checks: List[SafetyCheck],
        requires_human_review: bool
    ) -> str:
        """Generate human-readable reasoning for the decision."""
        if approved:
            reasoning = f"Output approved with confidence {confidence_score:.2f}. "
            reasoning += "All safety checks passed. "
        else:
            violations = [check for check in safety_checks if not check.passed]
            reasoning = f"Output rejected with confidence {confidence_score:.2f}. "
            reasoning += f"Found {len(violations)} safety violation(s): "
            reasoning += ", ".join([v.message for v in violations[:3]])  # First 3
            if len(violations) > 3:
                reasoning += f" and {len(violations) - 3} more."
        
        if requires_human_review:
            reasoning += " Human review required due to low confidence or high-severity violations."
        
        return reasoning
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """
        Get full audit trail of all decisions (Accountability).
        
        Returns:
            List of all decisions in dictionary format
        """
        return [
            {
                "decision_id": d.decision_id,
                "timestamp": d.timestamp,
                "agent_name": d.agent_name,
                "action": d.action,
                "approved": d.approved,
                "reasoning": d.reasoning,
                "confidence_score": d.confidence_score,
                "requires_human_review": d.requires_human_review,
                "safety_violations": len([c for c in d.safety_checks if not c.passed])
            }
            for d in self.decisions
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get governance statistics."""
        if not self.decisions:
            return {
                "total_decisions": 0,
                "approved": 0,
                "rejected": 0,
                "human_review_required": 0
            }
        
        total = len(self.decisions)
        approved = sum(1 for d in self.decisions if d.approved)
        rejected = total - approved
        human_review = sum(1 for d in self.decisions if d.requires_human_review)
        
        return {
            "total_decisions": total,
            "approved": approved,
            "rejected": rejected,
            "human_review_required": human_review,
            "approval_rate": round(approved / total * 100, 2) if total > 0 else 0
        }


