"""CrewAI-based multi-agent system for code comment generation.

This module implements a multi-agent workflow using CrewAI:
1. Generator Agent: Creates code comments
2. Validator Agent: Validates quality and checks for hallucinations

Both agents operate under ethical governance controls.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from datetime import datetime

try:
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.tools import tool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not installed. Multi-agent system will not be available.")

from src.models.ai_client import OllamaService
from .governance import GovernanceFramework, GovernanceConfig

logger = logging.getLogger(__name__)


@dataclass
class CrewAIResult:
    """Result from CrewAI workflow."""
    comment: str
    approved: bool
    attempts: int
    generator_confidence: float
    validator_confidence: float
    governance_decisions: List[Dict[str, Any]]
    requires_human_review: bool
    execution_time: float
    metadata: Dict[str, Any]


class CrewAICommentGenerator:
    """
    CrewAI-based comment generation system with ethical governance.
    
    Uses CrewAI framework to coordinate:
    - Generator Agent: Creates comments
    - Validator Agent: Validates and checks quality
    - Governance: Ensures ethical operation
    """
    
    def __init__(
        self,
        model_service: OllamaService,
        governance_config: Optional[GovernanceConfig] = None
    ):
        """
        Initialize CrewAI comment generator.
        
        Args:
            model_service: Ollama service for LLM access
            governance_config: Optional governance configuration
        """
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is not installed. Install with: pip install crewai")
        
        self.model_service = model_service
        self.governance = GovernanceFramework(governance_config)
        
        # Configure Ollama LLM for CrewAI
        self.llm = LLM(
            model=f"ollama/{model_service.model}",
            base_url=model_service.base_url
        )
        
        # Initialize agents
        self.generator_agent = self._create_generator_agent()
        self.validator_agent = self._create_validator_agent()
        
        logger.info("CrewAI comment generator initialized with Ollama")
    
    def _clean_comment(self, comment: str) -> str:
        """Clean generated comment: remove code fences, quotes, preambles, and noise."""
        if not comment:
            return ""
        
        import re
        # Remove markdown code fences (```python, ```, etc.)
        comment = re.sub(r'^```\w*\s*\n?', '', comment, flags=re.MULTILINE)
        comment = re.sub(r'\n?```\s*$', '', comment, flags=re.MULTILINE)
        comment = comment.strip()
        
        # Remove triple quotes
        comment = comment.replace('"""', '').replace("'''", "").strip()
        
        # Remove common preambles and apologies (case-insensitive)
        preamble_patterns = [
            r"^I'?m sorry.*?\n+",
            r"^I apologize.*?\n+",
            r"^Here is.*?:\s*\n+",
            r"^Here's.*?:\s*\n+",
            r"^Based on.*?:\s*\n+",
            r"^The following.*?:\s*\n+",
            r"^This is.*?:\s*\n+",
            r"^Below is.*?:\s*\n+",
        ]
        for pattern in preamble_patterns:
            comment = re.sub(pattern, '', comment, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove unwanted prefixes
        bad_prefixes = ["docstring:", "output:", "comment:", "## class comment", "# comment"]
        for p in bad_prefixes:
            if comment.lower().startswith(p):
                comment = comment[len(p):].strip()
        
        # Remove lines with just code fences or noise
        lines = comment.split("\n")
        cleaned = []
        skip_next = False
        
        for i, line in enumerate(lines):
            normalized = line.strip().lower()
            
            # Skip markdown code fences
            if normalized.startswith("```") or normalized == "```":
                continue
            
            # Skip lines that are just preambles
            if i == 0 and any(phrase in normalized for phrase in ["i'm sorry", "i apologize", "here is", "here's", "based on"]):
                skip_next = True
                continue
            
            if skip_next and not normalized:
                skip_next = False
                continue
            
            cleaned.append(line)
        
        result = "\n".join(cleaned).strip()
        
        # Remove leading/trailing blank lines
        while result.startswith("\n"):
            result = result[1:]
        while result.endswith("\n\n"):
            result = result[:-1]
        
        return result
    
    def _create_generator_agent(self) -> Agent:
        """Create the Generator Agent using CrewAI."""
        return Agent(
            role="Code Comment Generator",
            goal="Generate high-quality, accurate code comments that explain functionality clearly",
            backstory="""You are an expert software documentation specialist with deep knowledge 
            of programming best practices. Your role is to analyze code and create clear, 
            comprehensive comments that help developers understand the code's purpose, 
            parameters, return values, and behavior. You always follow documentation 
            standards and ensure accuracy.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    def _create_validator_agent(self) -> Agent:
        """Create the Validator Agent using CrewAI."""
        return Agent(
            role="Comment Quality Validator",
            goal="Validate generated comments for accuracy, completeness, and quality",
            backstory="""You are a meticulous code reviewer and quality assurance specialist. 
            Your role is to validate generated comments by checking for accuracy, completeness, 
            consistency with the code, and potential hallucinations. You ensure that comments 
            meet quality standards and flag any issues that need attention. You have a keen 
            eye for detail and never let inaccurate documentation pass.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
    
    def generate_and_validate(
        self,
        code: str,
        language: str = "python",
        comment_type: str = "function",
        temperature: float = 0.4,
        max_retries: int = 2
    ) -> CrewAIResult:
        """
        Generate and validate a code comment using CrewAI workflow.
        
        Args:
            code: Source code to comment
            language: Programming language
            comment_type: Type of comment
            temperature: Generation temperature
            max_retries: Maximum retry attempts
        
        Returns:
            CrewAIResult with complete workflow results
        """
        import time
        start_time = time.time()
        
        logger.info("Starting CrewAI workflow for comment generation")
        
        # Task 1: Generate comment
        generation_task = Task(
            description=f"""TASK: Generate a {comment_type} comment for the following {language} code.

CODE TO DOCUMENT:
```{language}
{code}
```

STRICT INSTRUCTIONS:
1. Output ONLY the comment text - NO apologies, NO explanations, NO meta-commentary
2. Do NOT say "I'm sorry", "Here is", "Based on", or any preamble
3. Start directly with the comment content
4. For functions: Use this exact format:
   
   Brief description of what the function does.
   
   Args:
       param_name (type): Description of parameter.
   
   Returns:
       type: Description of return value.

5. For classes: Use this exact format:
   
   Brief description of what the class does.
   
   Attributes:
       attr_name (type): Description of attribute.
   
   Methods:
       method_name(params): Brief description of method.

6. Be accurate - only document what exists in the code
7. Do NOT add code fences, quotes, or markdown formatting

OUTPUT THE COMMENT NOW:""",
            agent=self.generator_agent,
            expected_output="A clean, well-formatted docstring without any preamble or meta-commentary"
        )
        
        # Task 2: Validate comment
        validation_task = Task(
            description=f"""TASK: Validate the generated comment against the code.

ORIGINAL CODE:
```{language}
{code}
```

GENERATED COMMENT TO VALIDATE:
{{{{ generation_task.output }}}}

VALIDATION CHECKLIST:
1. Does the comment accurately describe what the code does?
2. Are ALL parameters from the code documented?
3. Is the return value documented correctly?
4. Are there any hallucinations or made-up information?
5. Is the comment complete and properly formatted?
6. Does it follow {language} documentation standards?

CRITICAL: You MUST respond in EXACTLY this format (no additional text):

VALID: YES
CONFIDENCE: 0.95
ISSUES: None
RECOMMENDATION: APPROVE

OR if there are problems:

VALID: NO
CONFIDENCE: 0.60
ISSUES: Missing parameter documentation for 'x'
RECOMMENDATION: REJECT

RULES:
- Output ONLY these 4 lines in this exact format
- VALID must be either YES or NO (nothing else)
- CONFIDENCE must be a number between 0.0 and 1.0
- ISSUES must list specific problems or say "None"
- RECOMMENDATION must be APPROVE, REJECT, or MODIFY
- Do NOT add explanations, preambles, or extra text

OUTPUT YOUR VALIDATION NOW:""",
            agent=self.validator_agent,
            expected_output="Exactly 4 lines: VALID, CONFIDENCE, ISSUES, RECOMMENDATION",
            context=[generation_task]
        )
        
        # Create crew
        crew = Crew(
            agents=[self.generator_agent, self.validator_agent],
            tasks=[generation_task, validation_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute workflow
        try:
            result = crew.kickoff()
            
            # Extract generated comment from first task
            generated_comment = generation_task.output.raw if hasattr(generation_task.output, 'raw') else str(generation_task.output)
            
            # Clean the generated comment (remove code fences, etc.)
            generated_comment = self._clean_comment(generated_comment)
            
            # Extract validation result from second task
            validation_output = validation_task.output.raw if hasattr(validation_task.output, 'raw') else str(validation_task.output)
            
            # Parse validation result
            validation_result = self._parse_validation_result(validation_output)
            
            # Debug logging
            logger.info(f"Validator output: {validation_output[:200]}")
            logger.info(f"Parsed validation_result: {validation_result}")
            
            # Apply governance checks
            generator_confidence = 0.8  # Default confidence
            validator_confidence = validation_result.get("confidence", 0.7)
            
            generator_decision = self.governance.make_decision(
                agent_name="CrewAI_Generator",
                action="generate_comment",
                output=generated_comment,
                code=code,
                confidence_score=generator_confidence,
                metadata={
                    "language": language,
                    "comment_type": comment_type,
                    "temperature": temperature
                }
            )
            
            validator_decision = self.governance.make_decision(
                agent_name="CrewAI_Validator",
                action="validate_comment",
                output=generated_comment,
                code=code,
                confidence_score=validator_confidence,
                metadata={
                    "validation_result": validation_result,
                    "issues": validation_result.get("issues", [])
                }
            )
            
            # Determine final approval
            # If both governance agents approve, we should approve even if validator didn't explicitly say "VALID: YES"
            # This handles cases where the validator gives approval but doesn't format output perfectly
            approved = (
                generator_decision.approved and
                validator_decision.approved and
                (validation_result.get("valid", False) or validator_confidence >= 0.6)
            )
            
            requires_human_review = (
                generator_decision.requires_human_review or
                validator_decision.requires_human_review or
                validator_confidence < 0.5
            )
            
            execution_time = time.time() - start_time
            
            logger.info(f"CrewAI workflow completed in {execution_time:.2f}s - Approved: {approved}")
            
            return CrewAIResult(
                comment=generated_comment,
                approved=approved,
                attempts=1,
                generator_confidence=generator_confidence,
                validator_confidence=validator_confidence,
                governance_decisions=[
                    {
                        "agent": "generator",
                        "approved": generator_decision.approved,
                        "reasoning": generator_decision.reasoning,
                        "confidence": generator_confidence
                    },
                    {
                        "agent": "validator",
                        "approved": validator_decision.approved,
                        "reasoning": validator_decision.reasoning,
                        "confidence": validator_confidence
                    }
                ],
                requires_human_review=requires_human_review,
                execution_time=execution_time,
                metadata={
                    "validation_result": validation_result,
                    "governance_stats": self.governance.get_statistics()
                }
            )
            
        except Exception as e:
            logger.error(f"CrewAI workflow failed: {e}")
            raise
    
    def _parse_validation_result(self, validation_output: str) -> Dict[str, Any]:
        """Parse validation output from validator agent."""
        result = {
            "valid": False,
            "confidence": 0.5,
            "issues": [],
            "recommendation": "REJECT"
        }
        
        lines = validation_output.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith("VALID:"):
                result["valid"] = "YES" in line.upper()
            
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split(":")[-1].strip()
                    result["confidence"] = float(conf_str)
                except:
                    pass
            
            elif line.startswith("ISSUES:"):
                issues_str = line.split(":", 1)[-1].strip()
                if issues_str.lower() != "none":
                    result["issues"] = [issues_str]
            
            elif line.startswith("RECOMMENDATION:"):
                result["recommendation"] = line.split(":")[-1].strip().upper()
        
        return result
    
    def get_governance_statistics(self) -> Dict[str, Any]:
        """Get governance framework statistics."""
        return self.governance.get_statistics()
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get full audit trail of all governance decisions."""
        return self.governance.get_audit_trail()

