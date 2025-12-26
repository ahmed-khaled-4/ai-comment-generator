"""
Comprehensive Multi-Agent Test Suite
Tests multiple functions and classes to validate the entire workflow
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

# Test cases with various complexity levels
TEST_CASES = [
    {
        "name": "Simple Addition Function",
        "code": """def add(a, b):
    return a + b""",
        "language": "python",
        "comment_type": "function"
    },
    {
        "name": "Factorial Function (Recursive)",
        "code": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)""",
        "language": "python",
        "comment_type": "function"
    },
    {
        "name": "Fibonacci Function",
        "code": """def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)""",
        "language": "python",
        "comment_type": "function"
    },
    {
        "name": "List Processing Function",
        "code": """def process_list(items, multiplier=2):
    result = []
    for item in items:
        if isinstance(item, (int, float)):
            result.append(item * multiplier)
    return result""",
        "language": "python",
        "comment_type": "function"
    },
    {
        "name": "String Manipulation Function",
        "code": """def reverse_words(text):
    words = text.split()
    reversed_words = [word[::-1] for word in words]
    return ' '.join(reversed_words)""",
        "language": "python",
        "comment_type": "function"
    },
    {
        "name": "Simple Calculator Class",
        "code": """class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(result)
        return result
    
    def subtract(self, a, b):
        result = a - b
        self.history.append(result)
        return result""",
        "language": "python",
        "comment_type": "class"
    },
    {
        "name": "Data Processor Class",
        "code": """class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed = False
    
    def process(self):
        self.data = [x * 2 for x in self.data]
        self.processed = True
        return self.data
    
    def reset(self):
        self.processed = False""",
        "language": "python",
        "comment_type": "class"
    },
    {
        "name": "User Manager Class",
        "code": """class UserManager:
    def __init__(self):
        self.users = {}
    
    def add_user(self, username, email):
        if username not in self.users:
            self.users[username] = {'email': email, 'active': True}
            return True
        return False
    
    def get_user(self, username):
        return self.users.get(username)
    
    def deactivate_user(self, username):
        if username in self.users:
            self.users[username]['active'] = False""",
        "language": "python",
        "comment_type": "class"
    },
    {
        "name": "Error Handling Function",
        "code": """def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
    except TypeError:
        return None""",
        "language": "python",
        "comment_type": "function"
    },
    {
        "name": "Dictionary Merge Function",
        "code": """def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result""",
        "language": "python",
        "comment_type": "function"
    }
]


def print_header(text, char="="):
    """Print a formatted header."""
    print(f"\n{char * 80}")
    print(f" {text}")
    print(f"{char * 80}\n")


def print_section(text):
    """Print a section divider."""
    print(f"\n{'-' * 80}")
    print(f" {text}")
    print(f"{'-' * 80}\n")


def test_multi_agent_system():
    """Run comprehensive multi-agent tests."""
    
    print_header("COMPREHENSIVE MULTI-AGENT SYSTEM TEST", "=")
    print(f"Testing {len(TEST_CASES)} different code samples")
    print(f"Each will go through: Generator Agent -> Validator Agent -> Governance")
    print()
    
    results = []
    total_time = 0
    
    for idx, test_case in enumerate(TEST_CASES, 1):
        print_section(f"TEST {idx}/{len(TEST_CASES)}: {test_case['name']}")
        
        print("Code:")
        print("-" * 80)
        print(test_case['code'])
        print("-" * 80)
        print()
        
        request_data = {
            "code": test_case["code"],
            "language": test_case["language"],
            "comment_type": test_case["comment_type"],
            "temperature": 0.4,
            "max_retries": 2,
            "governance_level": "standard"
        }
        
        print(f"Sending to multi-agent system ({test_case['comment_type']})...")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_URL}/multi_agent/generate",
                json=request_data,
                timeout=120
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if response.status_code == 200:
                result = response.json()
                
                # Store result
                results.append({
                    "name": test_case["name"],
                    "approved": result["approved"],
                    "attempts": result["attempts"],
                    "generator_confidence": result["generator_result"]["confidence"],
                    "validator_confidence": result["validator_result"]["confidence"],
                    "time": elapsed,
                    "comment": result["comment"]
                })
                
                # Display results
                print(f"\n[{'APPROVED' if result['approved'] else 'REJECTED'}] Status: {result['approved']}")
                print(f"Time: {elapsed:.2f}s | Attempts: {result['attempts']}")
                
                gen_result = result.get('generator_result', {})
                val_result = result.get('validator_result', {})
                
                print(f"\nGenerator: {gen_result.get('confidence', 0)*100:.0f}% confidence | " +
                      f"{'APPROVED' if gen_result.get('governance_approved') else 'REJECTED'}")
                print(f"Validator: {val_result.get('confidence', 0)*100:.0f}% confidence | " +
                      f"{'APPROVED' if val_result.get('governance_approved') else 'REJECTED'}")
                
                gov_stats = result.get('governance_statistics', {})
                print(f"\nGovernance: {gov_stats.get('approved', 0)}/{gov_stats.get('total_decisions', 0)} approved " +
                      f"({gov_stats.get('approval_rate', 0):.0f}%)")
                
                print(f"\nGenerated Comment:")
                print("=" * 80)
                print(result['comment'])
                print("=" * 80)
                
                if not result['approved']:
                    print("\n[WARNING] Comment was REJECTED by governance!")
                    print(f"Generator Reasoning: {gen_result.get('reasoning', 'N/A')}")
                    print(f"Validator Reasoning: {val_result.get('reasoning', 'N/A')}")
                
            else:
                print(f"\n[ERROR] HTTP {response.status_code}")
                print(f"Response: {response.text}")
                results.append({
                    "name": test_case["name"],
                    "approved": False,
                    "error": f"HTTP {response.status_code}"
                })
                
        except requests.exceptions.Timeout:
            print(f"\n[ERROR] Request timed out (>120s)")
            results.append({
                "name": test_case["name"],
                "approved": False,
                "error": "Timeout"
            })
        except Exception as e:
            print(f"\n[ERROR] {e}")
            results.append({
                "name": test_case["name"],
                "approved": False,
                "error": str(e)
            })
        
        # Small delay between tests
        if idx < len(TEST_CASES):
            print("\n[Waiting 2 seconds before next test...]")
            time.sleep(2)
    
    # Print summary
    print_header("TEST SUMMARY", "=")
    
    successful = [r for r in results if r.get("approved", False)]
    failed = [r for r in results if not r.get("approved", False)]
    
    print(f"Total Tests: {len(results)}")
    print(f"Approved: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Rejected/Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time: {total_time/len(results):.2f}s per test")
    print()
    
    if successful:
        print("APPROVED TESTS:")
        print("-" * 80)
        for r in successful:
            print(f"  [{r['attempts']} attempt(s)] {r['name']}")
            print(f"    Generator: {r['generator_confidence']*100:.0f}% | " +
                  f"Validator: {r['validator_confidence']*100:.0f}% | " +
                  f"Time: {r['time']:.2f}s")
        print()
    
    if failed:
        print("REJECTED/FAILED TESTS:")
        print("-" * 80)
        for r in failed:
            error = r.get('error', 'Governance rejection')
            print(f"  [FAILED] {r['name']}: {error}")
        print()
    
    # Overall statistics
    print_header("OVERALL STATISTICS", "=")
    
    if successful:
        avg_gen_conf = sum(r['generator_confidence'] for r in successful) / len(successful)
        avg_val_conf = sum(r['validator_confidence'] for r in successful) / len(successful)
        avg_attempts = sum(r['attempts'] for r in successful) / len(successful)
        
        print(f"Average Generator Confidence: {avg_gen_conf*100:.1f}%")
        print(f"Average Validator Confidence: {avg_val_conf*100:.1f}%")
        print(f"Average Attempts: {avg_attempts:.2f}")
        print()
    
    print_header("MULTI-AGENT SYSTEM TEST COMPLETE", "=")
    
    if len(successful) == len(results):
        print("[SUCCESS] All tests passed! Multi-agent system is working perfectly!")
    elif len(successful) > len(results) * 0.8:
        print("[GOOD] Most tests passed. Multi-agent system is working well.")
    else:
        print("[WARNING] Some tests failed. Review the results above.")
    
    print()


if __name__ == "__main__":
    try:
        # Check if server is running
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("[OK] Server is running")
            test_multi_agent_system()
        else:
            print("[ERROR] Server returned unexpected status")
    except:
        print("[ERROR] Server is not running!")
        print("Please start the server first: python -m src.main")

