"""
Comprehensive Test Suite for Input Validation System

This test script verifies that all 6 validation system modules work correctly
and integrate properly with the FastAPI application.

Modules Tested:
1. Input Request Validation (validate_request)
2. Output Validation (validate_output)
3. Safety Rules (enforce_safety_rules)
4. Human Review System (flag_for_human_review)
5. Rejection and Retry (process_with_retry)
6. Monitoring System (log_safety_violation)

Usage:
    python -m pytest tests/test_validation_system.py -v
    or
    python tests/test_validation_system.py
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import validation modules
from validators.input_request_validation import (
    validate_request,
    ValidationError,
    get_supported_languages,
    get_valid_comment_types
)
from validators.output_validation import (
    validate_output,
    OutputValidationError,
    validate_output_with_metrics
)
from validators.safety_rules import (
    enforce_safety_rules,
    SafetyViolationError,
    get_safety_config,
    SafetyLevel
)
from human_review.review_system import (
    flag_for_human_review,
    get_review_system,
    HumanReviewSystem
)
from validators.rejection_retry import (
    process_with_retry,
    RetryConfig,
    RetryStrategy,
    RetryExhaustedError,
    get_retry_system
)
from monitoring.validation_monitor import (
    get_monitor,
    log_safety_violation,
    log_rejection,
    log_retry,
    get_validation_statistics,
    generate_validation_report
)


# ==========================================
# Test Data
# ==========================================

# Valid request data
VALID_REQUEST = {
    "code": "def hello_world():\n    print('Hello, World!')\n    return True",
    "language": "python",
    "comment_type": "function",
    "temperature": 0.7,
    "max_tokens": 400
}

# Valid comment output
VALID_COMMENT = "This function prints 'Hello, World!' and returns True."

# Invalid test cases
INVALID_REQUESTS = {
    "missing_code": {
        "language": "python"
    },
    "invalid_language": {
        "code": "print('test')",
        "language": "invalid_lang"
    },
    "code_too_long": {
        "code": "x" * 60000,  # Exceeds MAX_CODE_LENGTH
        "language": "python"
    },
    "invalid_temperature": {
        "code": "print('test')",
        "language": "python",
        "temperature": 3.0  # Exceeds max (2.0)
    },
    "invalid_max_tokens": {
        "code": "print('test')",
        "language": "python",
        "max_tokens": 3000  # Exceeds max (2000)
    }
}

INVALID_COMMENTS = {
    "too_short": "Hi",  # Less than MIN_COMMENT_LENGTH
    "too_long": "x" * 6000,  # Exceeds MAX_COMMENT_LENGTH
    "contains_code": "```python\ndef test(): pass\n```",
    "contains_special_tokens": "This is a comment <|file_separator|> with tokens"
}


# ==========================================
# Test Functions
# ==========================================

def test_input_request_validation():
    """
    Test input request validation module.
    
    Tests:
    - Valid request passes validation
    - Invalid requests raise ValidationError
    - Supported languages are recognized
    - Comment types are validated
    """
    print("\n" + "="*60)
    print("TEST 1: Input Request Validation")
    print("="*60)
    
    try:
        # Test 1.1: Valid request
        print("\n[1.1] Testing valid request...")
        validate_request(VALID_REQUEST)
        print("✓ Valid request passed validation")
        
        # Test 1.2: Missing required field
        print("\n[1.2] Testing missing 'code' field...")
        try:
            validate_request(INVALID_REQUESTS["missing_code"])
            print("✗ FAILED: Should have raised ValidationError")
            return False
        except ValidationError as e:
            print(f"✓ Correctly raised ValidationError: {str(e)[:50]}...")
        
        # Test 1.3: Invalid language
        print("\n[1.3] Testing invalid language...")
        try:
            validate_request(INVALID_REQUESTS["invalid_language"])
            print("✗ FAILED: Should have raised ValidationError")
            return False
        except ValidationError as e:
            print(f"✓ Correctly raised ValidationError: {str(e)[:50]}...")
        
        # Test 1.4: Code too long
        print("\n[1.4] Testing code length limit...")
        try:
            validate_request(INVALID_REQUESTS["code_too_long"])
            print("✗ FAILED: Should have raised ValidationError")
            return False
        except ValidationError as e:
            print(f"✓ Correctly raised ValidationError: {str(e)[:50]}...")
        
        # Test 1.5: Invalid temperature
        print("\n[1.5] Testing temperature validation...")
        try:
            validate_request(INVALID_REQUESTS["invalid_temperature"])
            print("✗ FAILED: Should have raised ValidationError")
            return False
        except ValidationError as e:
            print(f"✓ Correctly raised ValidationError: {str(e)[:50]}...")
        
        # Test 1.6: Supported languages
        print("\n[1.6] Testing supported languages list...")
        languages = get_supported_languages()
        assert "python" in languages, "Python should be in supported languages"
        assert "javascript" in languages, "JavaScript should be in supported languages"
        print(f"✓ Found {len(languages)} supported languages")
        
        # Test 1.7: Valid comment types
        print("\n[1.7] Testing valid comment types...")
        comment_types = get_valid_comment_types()
        assert "function" in comment_types, "function should be a valid comment type"
        print(f"✓ Found {len(comment_types)} valid comment types")
        
        print("\n✓ All input request validation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_validation():
    """
    Test output validation module.
    
    Tests:
    - Valid comment passes validation
    - Invalid comments raise OutputValidationError
    - Quality metrics are calculated
    """
    print("\n" + "="*60)
    print("TEST 2: Output Validation")
    print("="*60)
    
    try:
        # Test 2.1: Valid comment
        print("\n[2.1] Testing valid comment...")
        result = validate_output(VALID_COMMENT)
        assert result is True, "validate_output should return True for valid comment"
        print("✓ Valid comment passed validation")
        
        # Test 2.2: Comment too short
        print("\n[2.2] Testing comment too short...")
        try:
            validate_output(INVALID_COMMENTS["too_short"])
            print("✗ FAILED: Should have raised OutputValidationError")
            return False
        except OutputValidationError as e:
            print(f"✓ Correctly raised OutputValidationError: {str(e)[:50]}...")
        
        # Test 2.3: Comment too long
        print("\n[2.3] Testing comment too long...")
        try:
            validate_output(INVALID_COMMENTS["too_long"])
            print("✗ FAILED: Should have raised OutputValidationError")
            return False
        except OutputValidationError as e:
            print(f"✓ Correctly raised OutputValidationError: {str(e)[:50]}...")
        
        # Test 2.4: Comment contains code artifacts
        print("\n[2.4] Testing comment with code artifacts...")
        try:
            validate_output(INVALID_COMMENTS["contains_code"])
            print("✗ FAILED: Should have raised OutputValidationError")
            return False
        except OutputValidationError as e:
            print(f"✓ Correctly raised OutputValidationError: {str(e)[:50]}...")
        
        # Test 2.5: Comment with special tokens
        print("\n[2.5] Testing comment with special tokens...")
        try:
            validate_output(INVALID_COMMENTS["contains_special_tokens"])
            print("✗ FAILED: Should have raised OutputValidationError")
            return False
        except OutputValidationError as e:
            print(f"✓ Correctly raised OutputValidationError: {str(e)[:50]}...")
        
        # Test 2.6: Validation with metrics
        print("\n[2.6] Testing validation with metrics...")
        metrics = validate_output_with_metrics(VALID_COMMENT)
        assert "length" in metrics, "Metrics should include length"
        assert "word_count" in metrics, "Metrics should include word_count"
        assert "quality_score" in metrics, "Metrics should include quality_score"
        print(f"✓ Metrics calculated: length={metrics['length']}, quality={metrics['quality_score']:.2f}")
        
        print("\n✓ All output validation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_safety_rules():
    """
    Test safety rules enforcement module.
    
    Tests:
    - Safe comments pass safety rules
    - Unsafe comments raise SafetyViolationError
    - Different safety levels work correctly
    """
    print("\n" + "="*60)
    print("TEST 3: Safety Rules Enforcement")
    print("="*60)
    
    try:
        # Test 3.1: Safe comment
        print("\n[3.1] Testing safe comment...")
        result = enforce_safety_rules(VALID_COMMENT)
        assert result is True, "enforce_safety_rules should return True for safe comment"
        print("✓ Safe comment passed safety rules")
        
        # Test 3.2: Comment too short (safety violation)
        print("\n[3.2] Testing comment too short (safety violation)...")
        try:
            enforce_safety_rules("Hi")
            print("✗ FAILED: Should have raised SafetyViolationError")
            return False
        except SafetyViolationError as e:
            print(f"✓ Correctly raised SafetyViolationError: {str(e)[:50]}...")
        
        # Test 3.3: Different safety levels
        print("\n[3.3] Testing different safety levels...")
        permissive_config = get_safety_config("permissive")
        standard_config = get_safety_config("standard")
        strict_config = get_safety_config("strict")
        
        assert permissive_config.level == SafetyLevel.PERMISSIVE
        assert standard_config.level == SafetyLevel.STANDARD
        assert strict_config.level == SafetyLevel.STRICT
        print("✓ Safety level configurations work correctly")
        
        # Test 3.4: Safety rules with custom config
        print("\n[3.4] Testing safety rules with custom config...")
        custom_config = get_safety_config("standard")
        result = enforce_safety_rules(VALID_COMMENT, config=custom_config)
        assert result is True
        print("✓ Custom safety config works correctly")
        
        print("\n✓ All safety rules tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_human_review_system():
    """
    Test human review system module.
    
    Tests:
    - High confidence comments are not flagged
    - Low confidence comments are flagged
    - Review system statistics work
    """
    print("\n" + "="*60)
    print("TEST 4: Human Review System")
    print("="*60)
    
    try:
        # Test 4.1: High confidence (should not flag)
        print("\n[4.1] Testing high confidence comment...")
        should_flag = flag_for_human_review(
            comment=VALID_COMMENT,
            confidence=0.95,
            reason="Test high confidence"
        )
        assert should_flag is False, "High confidence should not be flagged"
        print("✓ High confidence comment not flagged (correct)")
        
        # Test 4.2: Low confidence (should flag)
        print("\n[4.2] Testing low confidence comment...")
        should_flag = flag_for_human_review(
            comment=VALID_COMMENT,
            confidence=0.4,
            reason="Test low confidence"
        )
        assert should_flag is True, "Low confidence should be flagged"
        print("✓ Low confidence comment flagged (correct)")
        
        # Test 4.3: Review system statistics
        print("\n[4.3] Testing review system statistics...")
        review_system = get_review_system()
        stats = review_system.get_review_statistics()
        assert "pending_reviews" in stats, "Stats should include pending_reviews"
        assert "total_reviewed" in stats, "Stats should include total_reviewed"
        print(f"✓ Review statistics: {stats['pending_reviews']} pending, {stats['total_reviewed']} total")
        
        # Test 4.4: Get pending reviews
        print("\n[4.4] Testing pending reviews retrieval...")
        pending = review_system.get_pending_reviews()
        assert isinstance(pending, list), "Pending reviews should be a list"
        print(f"✓ Found {len(pending)} pending reviews")
        
        print("\n✓ All human review system tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rejection_retry():
    """
    Test rejection and retry module.
    
    Tests:
    - Valid comments pass without retry
    - Invalid comments trigger retry logic
    - Retry exhaustion uses fallback
    """
    print("\n" + "="*60)
    print("TEST 5: Rejection and Retry System")
    print("="*60)
    
    try:
        # Test 5.1: Valid comment (no retry needed)
        print("\n[5.1] Testing valid comment (no retry)...")
        def always_valid(comment: str) -> bool:
            return True
        
        result = process_with_retry(
            comment=VALID_COMMENT,
            validation_func=always_valid
        )
        assert result == VALID_COMMENT, "Valid comment should be returned as-is"
        print("✓ Valid comment processed without retry")
        
        # Test 5.2: Invalid comment with retry
        print("\n[5.2] Testing invalid comment with retry...")
        attempt_count = [0]
        
        def failing_then_valid(comment: str) -> bool:
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise OutputValidationError("Simulated validation failure")
            return True
        
        def generate_new():
            return "This is a retried comment that should pass validation."
        
        retry_config = RetryConfig(
            max_retries=3,
            strategy=RetryStrategy.IMMEDIATE,
            enable_fallback=True
        )
        
        result = process_with_retry(
            comment="Invalid",
            validation_func=failing_then_valid,
            generation_func=generate_new,
            config=retry_config
        )
        assert attempt_count[0] >= 2, "Should have retried at least once"
        print(f"✓ Retry logic worked (attempts: {attempt_count[0]})")
        
        # Test 5.3: Retry exhaustion with fallback
        print("\n[5.3] Testing retry exhaustion with fallback...")
        def always_fail(comment: str) -> bool:
            raise OutputValidationError("Always fails")
        
        retry_config_fallback = RetryConfig(
            max_retries=2,
            strategy=RetryStrategy.IMMEDIATE,
            enable_fallback=True,
            fallback_comment="Fallback comment used."
        )
        
        result = process_with_retry(
            comment="Invalid",
            validation_func=always_fail,
            generation_func=lambda: "Still invalid",
            config=retry_config_fallback
        )
        assert result == retry_config_fallback.fallback_comment
        print("✓ Fallback comment used after retry exhaustion")
        
        # Test 5.4: Retry statistics
        print("\n[5.4] Testing retry statistics...")
        retry_system = get_retry_system()
        stats = retry_system.get_retry_statistics()
        assert "total_attempts" in stats, "Stats should include total_attempts"
        print(f"✓ Retry statistics: {stats['total_attempts']} total attempts")
        
        print("\n✓ All rejection and retry tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitoring_system():
    """
    Test monitoring system module.
    
    Tests:
    - Safety violations are logged
    - Rejections are logged
    - Retries are logged
    - Statistics are generated
    """
    print("\n" + "="*60)
    print("TEST 6: Monitoring System")
    print("="*60)
    
    try:
        # Test 6.1: Log safety violation
        print("\n[6.1] Testing safety violation logging...")
        log_safety_violation(
            violation_type="test_violation",
            comment="Test comment with violation",
            severity="medium",
            metadata={"test": True}
        )
        print("✓ Safety violation logged successfully")
        
        # Test 6.2: Log rejection
        print("\n[6.2] Testing rejection logging...")
        log_rejection(
            reason="Test rejection",
            comment="Test comment",
            attempt_number=1,
            metadata={"test": True}
        )
        print("✓ Rejection logged successfully")
        
        # Test 6.3: Log retry
        print("\n[6.3] Testing retry logging...")
        log_retry(
            attempt_number=1,
            success=True,
            delay_seconds=0.1,
            metadata={"test": True}
        )
        print("✓ Retry logged successfully")
        
        # Test 6.4: Get validation statistics
        print("\n[6.4] Testing validation statistics...")
        stats = get_validation_statistics()
        assert "total_violations" in stats, "Stats should include total_violations"
        assert "total_rejections" in stats, "Stats should include total_rejections"
        assert "total_retries" in stats, "Stats should include total_retries"
        print(f"✓ Statistics: {stats['total_violations']} violations, "
              f"{stats['total_rejections']} rejections, {stats['total_retries']} retries")
        
        # Test 6.5: Generate validation report
        print("\n[6.5] Testing validation report generation...")
        report = generate_validation_report()
        assert "VALIDATION MONITORING REPORT" in report, "Report should contain header"
        assert "SAFETY VIOLATIONS" in report, "Report should contain violations section"
        print("✓ Validation report generated successfully")
        print(f"  Report length: {len(report)} characters")
        
        print("\n✓ All monitoring system tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_example():
    """
    Integration test showing how validation functions work together.
    
    This simulates a typical FastAPI endpoint flow.
    """
    print("\n" + "="*60)
    print("INTEGRATION TEST: FastAPI Endpoint Simulation")
    print("="*60)
    
    try:
        # Simulate request data
        request_data = {
            "code": "def calculate_sum(a, b):\n    return a + b",
            "language": "python",
            "comment_type": "function",
            "temperature": 0.7,
            "max_tokens": 400
        }
        
        print("\n[Integration] Step 1: Input validation...")
        validate_request(request_data)
        print("  ✓ Input validation passed")
        
        print("\n[Integration] Step 2: Simulate comment generation...")
        generated_comment = "Calculates the sum of two numbers and returns the result."
        print(f"  ✓ Generated comment: {generated_comment[:50]}...")
        
        print("\n[Integration] Step 3: Output validation...")
        validate_output(generated_comment, code=request_data["code"], language=request_data["language"])
        print("  ✓ Output validation passed")
        
        print("\n[Integration] Step 4: Safety rules enforcement...")
        enforce_safety_rules(generated_comment)
        print("  ✓ Safety rules passed")
        
        print("\n[Integration] Step 5: Human review check...")
        confidence = 0.85
        requires_review = flag_for_human_review(
            comment=generated_comment,
            confidence=confidence,
            reason="Integration test"
        )
        print(f"  ✓ Review check: {'Flagged' if requires_review else 'Not flagged'} (confidence: {confidence})")
        
        print("\n[Integration] Step 6: Monitoring...")
        log_safety_violation(
            violation_type="none",
            comment=generated_comment,
            severity="low",
            metadata={"integration_test": True}
        )
        print("  ✓ Monitoring logged")
        
        print("\n✓ Integration test completed successfully!")
        print("  This demonstrates the complete validation flow in a FastAPI endpoint.")
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==========================================
# Main Test Runner
# ==========================================

def run_all_tests():
    """
    Run all validation system tests.
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    print("\n" + "="*60)
    print("VALIDATION SYSTEM TEST SUITE")
    print("="*60)
    print("\nRunning comprehensive tests for all 6 validation modules...")
    
    test_results = []
    
    # Run all tests
    test_results.append(("Input Request Validation", test_input_request_validation()))
    test_results.append(("Output Validation", test_output_validation()))
    test_results.append(("Safety Rules", test_safety_rules()))
    test_results.append(("Human Review System", test_human_review_system()))
    test_results.append(("Rejection and Retry", test_rejection_retry()))
    test_results.append(("Monitoring System", test_monitoring_system()))
    test_results.append(("Integration Example", test_integration_example()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*60)
    print(f"Total: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! Validation system is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    """
    Run tests when executed directly.
    
    Usage:
        python tests/test_validation_system.py
    """
    success = run_all_tests()
    sys.exit(0 if success else 1)

