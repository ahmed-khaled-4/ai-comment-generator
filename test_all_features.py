"""
Comprehensive test script for AI Comment Generator
Tests all Ahmed's implemented features (Tasks 1-5)
"""
import requests
import json
import time
from datetime import datetime

# API Base URL
BASE_URL = "http://localhost:8000"

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.RESET}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_info(text):
    print(f"{Colors.YELLOW}ℹ {text}{Colors.RESET}")

# Test 1: Health Check
def test_health():
    print_header("TEST 1: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Server is healthy")
            print_info(f"Service: {data.get('service')}")
            print_info(f"Model: {data.get('model')}")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

# Test 2: Basic Comment Generation (Task 1 - AI Service Integration)
def test_basic_generation():
    print_header("TEST 2: Basic Comment Generation (Task 1)")
    
    test_code = """def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"""
    
    try:
        response = requests.post(
            f"{BASE_URL}/generate_comment",
            json={
                "code": test_code,
                "language": "python",
                "comment_type": "function",
                "temperature": 0.4,
                "max_tokens": 400
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Comment generated successfully")
            print_info(f"Comment length: {len(data['comment'])} characters")
            print_info(f"Model used: {data['model']}")
            print_info(f"Latency: {data['metadata'].get('latency', 'N/A')}s")
            print_info(f"Tokens: {data['metadata'].get('total_tokens', 'N/A')}")
            print("\nGenerated Comment:")
            print(f"{Colors.YELLOW}{data['comment'][:200]}...{Colors.RESET}")
            return True
        else:
            print_error(f"Generation failed: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    except Exception as e:
        print_error(f"Generation error: {e}")
        return False

# Test 3: Logging System (Task 2)
def test_logging_statistics():
    print_header("TEST 3: Logging System (Task 2)")
    try:
        response = requests.get(f"{BASE_URL}/logs/statistics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            stats = data.get('statistics', {})
            print_success("Logging system is working")
            print_info(f"Total requests: {stats.get('total_requests', 0)}")
            print_info(f"Successful: {stats.get('successful_generations', 0)}")
            print_info(f"Failed: {stats.get('failed_generations', 0)}")
            print_info(f"Success rate: {stats.get('success_rate', 0)}%")
            return True
        else:
            print_error(f"Logging stats failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Logging stats error: {e}")
        return False

# Test 4: Multi-Agent System (Tasks 3, 4, 5)
def test_multi_agent_generation():
    print_header("TEST 4: CrewAI Multi-Agent System (Tasks 3, 4, 5)")
    
    test_code = """class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        return [x * 2 for x in self.data]"""
    
    try:
        print_info("Sending request to CrewAI multi-agent endpoint...")
        response = requests.post(
            f"{BASE_URL}/multi_agent/generate",
            json={
                "code": test_code,
                "language": "python",
                "comment_type": "class",
                "temperature": 0.4,
                "max_retries": 2,
                "governance_level": "standard"
            },
            timeout=60  # Multi-agent takes longer
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Multi-agent generation successful")
            print_info(f"Approved: {data.get('approved', False)}")
            print_info(f"Attempts: {data.get('attempts', 0)}")
            print_info(f"Requires human review: {data.get('requires_human_review', False)}")
            
            # Generator result
            gen_result = data.get('generator_result', {})
            print_info(f"Generator confidence: {gen_result.get('confidence', 0):.2f}")
            
            # Validator result
            val_result = data.get('validator_result', {})
            print_info(f"Validator confidence: {val_result.get('confidence', 0):.2f}")
            
            # Governance statistics
            gov_stats = data.get('governance_statistics', {})
            print_info(f"Governance decisions: {gov_stats.get('total_decisions', 0)}")
            print_info(f"Approval rate: {gov_stats.get('approval_rate', 0):.1f}%")
            
            print("\nGenerated Comment:")
            print(f"{Colors.YELLOW}{data['comment'][:200]}...{Colors.RESET}")
            return True
        elif response.status_code == 501:
            print_error("CrewAI not installed - Multi-agent system unavailable")
            print_info("Install with: pip install crewai")
            return False
        else:
            print_error(f"Multi-agent generation failed: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print_error("Multi-agent request timed out (this is normal for first run)")
        print_info("Try running the test again")
        return False
    except Exception as e:
        print_error(f"Multi-agent error: {e}")
        return False

# Test 5: Governance Statistics (Task 3 - Governance Framework)
def test_governance_statistics():
    print_header("TEST 5: Governance Framework Statistics (Task 3)")
    try:
        response = requests.get(f"{BASE_URL}/multi_agent/governance/statistics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            stats = data.get('statistics', {})
            print_success("Governance framework is working")
            print_info(f"Framework: {data.get('framework', 'Unknown')}")
            print_info(f"Total decisions: {stats.get('total_decisions', 0)}")
            print_info(f"Approved: {stats.get('approved', 0)}")
            print_info(f"Rejected: {stats.get('rejected', 0)}")
            print_info(f"Human review required: {stats.get('human_review_required', 0)}")
            return True
        elif response.status_code == 501:
            print_error("CrewAI not installed - Governance stats unavailable")
            return False
        else:
            print_error(f"Governance stats failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Governance stats error: {e}")
        return False

# Test 6: Error Handling (Task 1 - Enhanced Error Handling)
def test_error_handling():
    print_header("TEST 6: Error Handling (Task 1)")
    
    # Test with invalid parameters
    try:
        response = requests.post(
            f"{BASE_URL}/generate_comment",
            json={
                "code": "",  # Empty code
                "language": "python",
                "temperature": 5.0  # Invalid temperature
            },
            timeout=10
        )
        
        if response.status_code == 400:
            print_success("Error handling works correctly (400 for invalid input)")
            return True
        elif response.status_code == 422:
            print_success("Validation works correctly (422 for validation error)")
            return True
        else:
            print_error(f"Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error handling test failed: {e}")
        return False

# Test 7: Different Comment Types
def test_different_comment_types():
    print_header("TEST 7: Different Comment Types")
    
    test_cases = [
        ("function", "def add(a, b):\n    return a + b"),
        ("class", "class Calculator:\n    def add(self, a, b):\n        return a + b"),
    ]
    
    results = []
    for comment_type, code in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/generate_comment",
                json={
                    "code": code,
                    "language": "python",
                    "comment_type": comment_type,
                    "temperature": 0.4
                },
                timeout=30
            )
            
            if response.status_code == 200:
                print_success(f"{comment_type.capitalize()} comment generated")
                results.append(True)
            else:
                print_error(f"{comment_type.capitalize()} comment failed")
                results.append(False)
        except Exception as e:
            print_error(f"{comment_type.capitalize()} error: {e}")
            results.append(False)
    
    return all(results)

# Main test runner
def run_all_tests():
    print_header("AI COMMENT GENERATOR - COMPREHENSIVE TEST SUITE")
    print_info(f"Testing server at: {BASE_URL}")
    print_info(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "Health Check": test_health(),
        "Basic Generation (Task 1)": test_basic_generation(),
        "Logging System (Task 2)": test_logging_statistics(),
        "Multi-Agent System (Tasks 3-5)": test_multi_agent_generation(),
        "Governance Framework (Task 3)": test_governance_statistics(),
        "Error Handling (Task 1)": test_error_handling(),
        "Different Comment Types": test_different_comment_types(),
    }
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{Colors.BLUE}{'='*60}")
    if passed == total:
        print(f"{Colors.GREEN}ALL TESTS PASSED! ({passed}/{total}){Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}TESTS PASSED: {passed}/{total}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    # Task completion summary
    print_header("AI TASKS VERIFICATION")
    print_success("Task 1: AI Service Integration - WORKING" if results["Basic Generation (Task 1)"] else "Task 1: FAILED")
    print_success("Task 2: Logging System - WORKING" if results["Logging System (Task 2)"] else "Task 2: FAILED")
    print_success("Task 3: Governance Framework - WORKING" if results["Governance Framework (Task 3)"] else "Task 3: FAILED")
    print_success("Task 4: Multi-Agent Implementation - WORKING" if results["Multi-Agent System (Tasks 3-5)"] else "Task 4: FAILED")
    print_success("Task 5: FastAPI Integration - WORKING" if results["Multi-Agent System (Tasks 3-5)"] else "Task 5: FAILED")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Test suite error: {e}{Colors.RESET}")

