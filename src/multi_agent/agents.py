"""Multi-agent system with Generator and Validator agents.

This module implements a two-agent workflow:
1. Generator Agent: Creates code comments
2. Validator Agent: Checks quality and hallucinations

Both agents operate under ethical governance controls.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from datetime import datetime

from src.models.ai_client import OllamaService
from src.models.model_config import ModelConfig
from .governance import GovernanceFramework, GovernanceConfig, GovernanceDecision

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent action."""
    agent_name: str
    action: str
    output: str
    confidence_score: float
    reasoning: str
    metadata: Dict[str, Any]
    governance_decision: Optional[GovernanceDecision] = None


class GeneratorAgent:
    """
    Generator Agent: Creates code comments using LLM.
    
    This agent is responsible for:
    - Generating high-quality code comments
    - Providing confidence scores
    - Explaining its generation process
    """
    
    def __init__(
        self,
        model_service: OllamaService,
        governance: GovernanceFramework,
        name: str = "GeneratorAgent"
    ):
        """
        Initialize Generator Agent.
        
        Args:
            model_service: Ollama service for LLM access
            governance: Governance framework for ethical controls
            name: Agent name for identification
        """
        self.name = name
        self.model_service = model_service
        self.governance = governance
        logger.info(f"{self.name} initialized")
    
    def generate(
        self,
        code: str,
        language: str = "python",
        comment_type: str = "function",
        temperature: float = 0.4
    ) -> AgentResult:
        """
        Generate a code comment with governance oversight.
        
        Args:
            code: Source code to comment
            language: Programming language
            comment_type: Type of comment (function, class, inline)
            temperature: Generation temperature
        
        Returns:
            AgentResult with generated comment and governance decision
        """
        logger.info(f"{self.name}: Generating {comment_type} comment for {language} code")
        
        try:
            # Generate comment using LLM
            result = self.model_service.generate_comment(
                code=code,
                language=language,
                comment_type=comment_type,
                temperature=temperature
            )
            
            generated_comment = result["comment"]
            metadata = result["metadata"]
            
            # Calculate confidence score based on generation quality indicators
            confidence_score = self._calculate_confidence(generated_comment, metadata)
            
            # Generate reasoning for transparency
            reasoning = self._generate_reasoning(
                code, generated_comment, confidence_score, metadata
            )
            
            # Submit to governance for approval
            governance_decision = self.governance.make_decision(
                agent_name=self.name,
                action="generate_comment",
                output=generated_comment,
                code=code,
                confidence_score=confidence_score,
                metadata={
                    **metadata,
                    "quality_score": confidence_score,
                    "language": language,
                    "comment_type": comment_type
                }
            )
            
            logger.info(
                f"{self.name}: Generation {'approved' if governance_decision.approved else 'rejected'} "
                f"(confidence: {confidence_score:.2f})"
            )
            
            return AgentResult(
                agent_name=self.name,
                action="generate_comment",
                output=generated_comment,
                confidence_score=confidence_score,
                reasoning=reasoning,
                metadata=metadata,
                governance_decision=governance_decision
            )
            
        except Exception as e:
            logger.error(f"{self.name}: Generation failed - {e}")
            raise
    
    def _calculate_confidence(self, comment: str, metadata: Dict[str, Any]) -> float:
        """
        Calculate confidence score for generated comment.
        
        Factors:
        - Comment length (too short or too long reduces confidence)
        - Latency (very slow generation may indicate issues)
        - Token usage (unusual patterns reduce confidence)
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 1.0
        
        # Length check
        comment_length = len(comment)
        if comment_length < 20:
            confidence *= 0.5  # Very short comments are suspicious
        elif comment_length < 50:
            confidence *= 0.8
        elif comment_length > 1500:
            confidence *= 0.7  # Very long comments may be verbose
        
        # Latency check
        latency = metadata.get("latency", 0)
        if latency > 30:  # Very slow
            confidence *= 0.8
        
        # Check for empty or placeholder patterns
        if not comment or comment.strip() in ["", "[NO COMMENT GENERATED]", "[GENERATION FAILED]"]:
            confidence = 0.0
        
        # Check for suspicious patterns
        suspicious_patterns = ["<|", "|>", "###", "```", "you are an expert"]
        if any(pattern in comment.lower() for pattern in suspicious_patterns):
            confidence *= 0.6
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_reasoning(
        self,
        code: str,
        comment: str,
        confidence: float,
        metadata: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for transparency."""
        reasoning = f"Generated comment with {len(comment)} characters. "
        reasoning += f"Confidence score: {confidence:.2f}. "
        
        if confidence >= 0.8:
            reasoning += "High confidence - comment appears well-formed and appropriate."
        elif confidence >= 0.5:
            reasoning += "Medium confidence - comment is acceptable but may have minor issues."
        else:
            reasoning += "Low confidence - comment may have quality issues or be inappropriate."
        
        latency = metadata.get("latency", 0)
        reasoning += f" Generation took {latency:.2f}s."
        
        return reasoning


class ValidatorAgent:
    """
    Validator Agent: Validates generated comments for quality and hallucinations.
    
    This agent is responsible for:
    - Checking comment quality
    - Detecting hallucinations
    - Verifying consistency with code
    - Providing validation reasoning
    """
    
    def __init__(
        self,
        governance: GovernanceFramework,
        name: str = "ValidatorAgent"
    ):
        """
        Initialize Validator Agent.
        
        Args:
            governance: Governance framework for ethical controls
            name: Agent name for identification
        """
        self.name = name
        self.governance = governance
        logger.info(f"{self.name} initialized")
    
    def validate(
        self,
        code: str,
        comment: str,
        generator_result: AgentResult
    ) -> AgentResult:
        """
        Validate a generated comment with governance oversight.
        
        Args:
            code: Original source code
            comment: Generated comment to validate
            generator_result: Result from generator agent
        
        Returns:
            AgentResult with validation decision
        """
        logger.info(f"{self.name}: Validating comment")
        
        # Perform validation checks
        validation_checks = self._perform_validation(code, comment)
        
        # Calculate validation confidence
        confidence_score = self._calculate_validation_confidence(validation_checks)
        
        # Generate reasoning
        reasoning = self._generate_validation_reasoning(validation_checks, confidence_score)
        
        # Submit to governance
        governance_decision = self.governance.make_decision(
            agent_name=self.name,
            action="validate_comment",
            output=comment,
            code=code,
            confidence_score=confidence_score,
            metadata={
                "validation_checks": validation_checks,
                "generator_confidence": generator_result.confidence_score,
                "combined_confidence": (confidence_score + generator_result.confidence_score) / 2
            }
        )
        
        logger.info(
            f"{self.name}: Validation {'passed' if governance_decision.approved else 'failed'} "
            f"(confidence: {confidence_score:.2f})"
        )
        
        return AgentResult(
            agent_name=self.name,
            action="validate_comment",
            output=comment,
            confidence_score=confidence_score,
            reasoning=reasoning,
            metadata={"validation_checks": validation_checks},
            governance_decision=governance_decision
        )
    
    def _perform_validation(self, code: str, comment: str) -> Dict[str, Any]:
        """
        Perform comprehensive validation checks.
        
        Returns:
            Dictionary with validation results
        """
        checks = {}
        
        # Check 1: Comment is not empty
        checks["not_empty"] = bool(comment and comment.strip())
        
        # Check 2: Reasonable length
        comment_length = len(comment)
        checks["reasonable_length"] = 20 <= comment_length <= 1500
        checks["comment_length"] = comment_length
        
        # Check 3: No code repetition (hallucination check)
        code_lower = code.lower()
        comment_lower = comment.lower()
        
        # Check if comment contains large chunks of the original code
        code_words = set(code_lower.split())
        comment_words = set(comment_lower.split())
        overlap = len(code_words & comment_words)
        overlap_ratio = overlap / len(comment_words) if comment_words else 0
        
        checks["no_code_repetition"] = overlap_ratio < 0.7  # Less than 70% overlap
        checks["code_overlap_ratio"] = overlap_ratio
        
        # Check 4: Contains documentation keywords
        doc_keywords = ["args", "returns", "parameters", "description", "attributes", "methods"]
        has_doc_keywords = any(keyword in comment_lower for keyword in doc_keywords)
        checks["has_documentation_structure"] = has_doc_keywords
        
        # Check 5: No suspicious patterns
        suspicious_patterns = ["<|", "|>", "###", "```", "you are an expert", "as an ai"]
        has_suspicious = any(pattern in comment_lower for pattern in suspicious_patterns)
        checks["no_suspicious_patterns"] = not has_suspicious
        
        # Check 6: Consistency check (basic)
        # If code has "def", comment should mention function/method
        if "def " in code_lower:
            checks["mentions_function"] = "function" in comment_lower or "method" in comment_lower
        else:
            checks["mentions_function"] = True  # Not applicable
        
        # If code has "class", comment should mention class
        if "class " in code_lower:
            checks["mentions_class"] = "class" in comment_lower
        else:
            checks["mentions_class"] = True  # Not applicable
        
        return checks
    
    def _calculate_validation_confidence(self, checks: Dict[str, Any]) -> float:
        """Calculate overall validation confidence from checks."""
        # Weight different checks
        weights = {
            "not_empty": 0.25,
            "reasonable_length": 0.15,
            "no_code_repetition": 0.20,
            "has_documentation_structure": 0.15,
            "no_suspicious_patterns": 0.15,
            "mentions_function": 0.05,
            "mentions_class": 0.05
        }
        
        confidence = 0.0
        for check_name, weight in weights.items():
            if check_name in checks and checks[check_name]:
                confidence += weight
        
        return min(1.0, confidence)
    
    def _generate_validation_reasoning(
        self,
        checks: Dict[str, Any],
        confidence: float
    ) -> str:
        """Generate human-readable validation reasoning."""
        passed_checks = sum(1 for k, v in checks.items() if isinstance(v, bool) and v)
        total_checks = sum(1 for v in checks.values() if isinstance(v, bool))
        
        reasoning = f"Validation completed: {passed_checks}/{total_checks} checks passed. "
        reasoning += f"Overall confidence: {confidence:.2f}. "
        
        # Mention specific issues
        issues = []
        if not checks.get("not_empty"):
            issues.append("empty comment")
        if not checks.get("reasonable_length"):
            issues.append("inappropriate length")
        if not checks.get("no_code_repetition"):
            issues.append("code repetition detected")
        if not checks.get("no_suspicious_patterns"):
            issues.append("suspicious patterns found")
        
        if issues:
            reasoning += "Issues: " + ", ".join(issues) + "."
        else:
            reasoning += "No major issues detected."
        
        return reasoning


class MultiAgentOrchestrator:
    """
    Orchestrator for multi-agent workflow with ethical governance.
    
    Coordinates Generator and Validator agents to produce high-quality,
    validated code comments under governance oversight.
    """
    
    def __init__(
        self,
        model_service: OllamaService,
        governance_config: Optional[GovernanceConfig] = None
    ):
        """
        Initialize multi-agent orchestrator.
        
        Args:
            model_service: Ollama service for LLM access
            governance_config: Optional governance configuration
        """
        self.governance = GovernanceFramework(governance_config)
        self.generator = GeneratorAgent(model_service, self.governance)
        self.validator = ValidatorAgent(self.governance)
        logger.info("Multi-agent orchestrator initialized")
    
    def generate_and_validate(
        self,
        code: str,
        language: str = "python",
        comment_type: str = "function",
        temperature: float = 0.4,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Generate and validate a code comment using multi-agent workflow.
        
        Args:
            code: Source code to comment
            language: Programming language
            comment_type: Type of comment
            temperature: Generation temperature
            max_retries: Maximum retry attempts if validation fails
        
        Returns:
            Dictionary with final comment, governance decisions, and full audit trail
        """
        logger.info("Starting multi-agent workflow")
        
        attempt = 0
        generator_result = None
        validator_result = None
        
        while attempt < max_retries:
            attempt += 1
            logger.info(f"Attempt {attempt}/{max_retries}")
            
            # Step 1: Generator creates comment
            generator_result = self.generator.generate(
                code=code,
                language=language,
                comment_type=comment_type,
                temperature=temperature
            )
            
            # Check if generator was approved by governance
            if not generator_result.governance_decision.approved:
                logger.warning(f"Generator output rejected by governance: {generator_result.governance_decision.reasoning}")
                if attempt < max_retries:
                    temperature += 0.1  # Adjust temperature for retry
                    continue
                else:
                    break
            
            # Step 2: Validator checks comment
            validator_result = self.validator.validate(
                code=code,
                comment=generator_result.output,
                generator_result=generator_result
            )
            
            # Check if validator approved
            if validator_result.governance_decision.approved:
                logger.info("Comment approved by both generator and validator")
                break
            else:
                logger.warning(f"Validator rejected comment: {validator_result.governance_decision.reasoning}")
                if attempt < max_retries:
                    temperature += 0.1  # Adjust for retry
                    continue
        
        # Prepare final result
        final_approved = (
            generator_result is not None and
            generator_result.governance_decision.approved and
            validator_result is not None and
            validator_result.governance_decision.approved
        )
        
        return {
            "comment": generator_result.output if generator_result else "",
            "approved": final_approved,
            "attempts": attempt,
            "generator_result": {
                "confidence": generator_result.confidence_score if generator_result else 0.0,
                "reasoning": generator_result.reasoning if generator_result else "",
                "governance_approved": generator_result.governance_decision.approved if generator_result else False
            } if generator_result else None,
            "validator_result": {
                "confidence": validator_result.confidence_score if validator_result else 0.0,
                "reasoning": validator_result.reasoning if validator_result else "",
                "governance_approved": validator_result.governance_decision.approved if validator_result else False
            } if validator_result else None,
            "governance_statistics": self.governance.get_statistics(),
            "audit_trail": self.governance.get_audit_trail(),
            "requires_human_review": (
                generator_result.governance_decision.requires_human_review if generator_result else True
            ) or (
                validator_result.governance_decision.requires_human_review if validator_result else True
            )
        }
    
    def get_governance_statistics(self) -> Dict[str, Any]:
        """Get governance framework statistics."""
        return self.governance.get_statistics()
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get full audit trail of all governance decisions."""
        return self.governance.get_audit_trail()


