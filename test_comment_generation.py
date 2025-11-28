"""Test script for code comment generation."""
import requests
import json
import time
from pathlib import Path
import inspect
import ast
from datetime import datetime

BASE_URL = "http://localhost:8000"
RESULTS_FILE = "results.json"


def get_code_samples():
    """Extract code samples from test_code_samples.py - only top-level functions and classes."""
    samples = []
    
    try:
        # Read the test file
        with open("test_code_samples.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse the file with error handling
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Warning: Syntax error in test_code_samples.py: {e}")
            return samples
        except Exception as e:
            print(f"Warning: Error parsing test_code_samples.py: {e}")
            return samples
        
        # Only get top-level nodes (not methods inside classes)
        for node in tree.body:
            try:
                if isinstance(node, ast.FunctionDef):
                    # Get function code
                    func_code = ast.get_source_segment(content, node)
                    if func_code:
                        samples.append({
                            "name": node.name,
                            "code": func_code,
                            "type": "function",
                            "has_docstring": ast.get_docstring(node) is not None
                        })
                elif isinstance(node, ast.ClassDef):
                    # Get class code
                    class_code = ast.get_source_segment(content, node)
                    if class_code:
                        samples.append({
                            "name": node.name,
                            "code": class_code,
                            "type": "class",
                            "has_docstring": ast.get_docstring(node) is not None
                        })
            except Exception as e:
                print(f"Warning: Error processing {node.__class__.__name__}: {e}")
                continue
    
    except FileNotFoundError:
        print(f"Error: test_code_samples.py not found")
        return samples
    except Exception as e:
        print(f"Error reading test_code_samples.py: {e}")
        return samples
    
    return samples


def test_health():
    """Test health endpoint."""
    print("=" * 70)
    print("Testing Health Endpoint")
    print("=" * 70)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Status: {data['status']}")
            print(f"✓ Service: {data['service']}")
            print(f"✓ Model: {data['model']}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Cannot connect to server!")
        print("\nPlease start the server first:")
        print("  python -m src.main")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_comment_generation(code, name, code_type):
    """Test comment generation for a code sample."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name} ({code_type})")
    print("=" * 70)
    print(f"\nCode:\n{code}\n")
    
    # Prepare request payload
    payload = {
        "code": code,
        "language": "python",
        "comment_type": code_type,
        "temperature": 0.4,
        "max_tokens": 600
    }
    
    try:
        # Measure latency
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/generate_comment",
            json=payload,
            timeout=60
        )
        
        # Calculate latency
        latency = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Generated Comment:\n{result['comment']}\n")
            print(f"Model: {result['model']}")
            print(f"Type: {result['comment_type']}")
            print(f"Latency: {latency:.2f}s")
            
            # Extract token usage and latency from metadata
            token_usage = {}
            result_latency = latency  # Use measured latency
            if 'metadata' in result:
                metadata = result['metadata']
                # Use latency from metadata if available, otherwise use measured
                result_latency = metadata.get('latency', latency)
                # Extract token usage
                if 'prompt_tokens' in metadata:
                    token_usage['prompt_tokens'] = metadata['prompt_tokens']
                if 'completion_tokens' in metadata:
                    token_usage['completion_tokens'] = metadata['completion_tokens']
                if 'total_tokens' in metadata:
                    token_usage['total_tokens'] = metadata['total_tokens']
            
            # Return result with additional metadata (cleaner format)
            return {
                "success": True,
                "name": name,
                "type": code_type,
                "code": code,
                "comment": result['comment'],
                "model": result['model'],
                "language": result.get('language', 'python'),
                "prompt": payload,  # Raw prompt payload
                "raw_response": response.text,  # Raw response text
                "latency": round(result_latency, 3) if result_latency else None,  # Latency in seconds
                "tokens": token_usage if token_usage else None,
                "metadata": result.get('metadata', {})
            }
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
            return {
                "success": False,
                "name": name,
                "type": code_type,
                "code": code,
                "prompt": payload,
                "raw_response": response.text,
                "latency": round(latency, 3),
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        print(f"✗ Error: {e}")
        latency = time.time() - start_time if 'start_time' in locals() else 0
        return {
            "success": False,
            "name": name,
            "type": code_type,
            "code": code,
            "prompt": payload,
            "raw_response": None,
            "latency": round(latency, 3),
            "error": str(e)
        }


def save_results_json(all_results, summary):
    """Save test results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "test_results": all_results
    }
    
    json_file = "results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return json_file


def save_results_markdown(all_results, summary):
    """Save test results to Markdown file."""
    md_file = "results.md"
    
    with open(md_file, "w", encoding="utf-8") as f:
        # Header
        f.write("# AI Comment Generator - Test Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Total Tests:** {summary['total_tests']}\n")
        f.write(f"- **Successful:** {summary['successful']} ✓\n")
        f.write(f"- **Failed:** {summary['failed']} ✗\n")
        f.write(f"- **Success Rate:** {summary['success_rate']}\n\n")
        f.write("---\n\n")
        
        # Test Results
        f.write("## Test Results\n\n")
        for i, result in enumerate(all_results, 1):
            f.write(f"### {i}. {result['name']} ({result['type']})\n\n")
            
            # Code
            f.write("**Code:**\n```python\n")
            f.write(result['code'])
            f.write("\n```\n\n")
            
            # Comment or Error
            if result.get('success', False):
                f.write("**Generated Comment:**\n\n")
                comment = result.get('comment', 'N/A')
                # Format multi-line comments
                if '\n' in comment:
                    f.write("```\n")
                    f.write(comment)
                    f.write("\n```\n\n")
                else:
                    f.write(f"{comment}\n\n")
                
                f.write(f"*Model: {result.get('model', 'N/A')} | Language: {result.get('language', 'python')}*\n\n")
            else:
                f.write(f"**Error:** {result.get('error', 'Unknown error')}\n\n")
            
            f.write("---\n\n")
    
    return md_file


def save_results_html(all_results, summary):
    """Save test results to HTML file."""
    html_file = "results.html"
    
    with open(html_file, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Comment Generator - Test Results</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        h3 {
            color: #555;
            margin-top: 25px;
            margin-bottom: 10px;
        }
        .summary {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .summary-item {
            display: inline-block;
            margin-right: 30px;
            font-size: 1.1em;
        }
        .summary-item strong {
            color: #2c3e50;
        }
        .success {
            color: #27ae60;
        }
        .failed {
            color: #e74c3c;
        }
        .test-item {
            background: #fafafa;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin-bottom: 25px;
            border-radius: 4px;
        }
        .test-item.failed {
            border-left-color: #e74c3c;
        }
        .code-block {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .comment-block {
            background: #e8f5e9;
            border-left: 3px solid #4caf50;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            white-space: pre-wrap;
        }
        .error-block {
            background: #ffebee;
            border-left: 3px solid #f44336;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .metadata {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 10px;
        }
        hr {
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }
        .timestamp {
            color: #95a5a6;
            font-size: 0.9em;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Comment Generator - Test Results</h1>
        <div class="timestamp">Generated: """)
        
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.write("""</div>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-item">
                <strong>Total Tests:</strong> """)
        f.write(str(summary['total_tests']))
        f.write("""</div>
            <div class="summary-item success">
                <strong>Successful:</strong> """)
        f.write(str(summary['successful']))
        f.write(""" ✓</div>
            <div class="summary-item failed">
                <strong>Failed:</strong> """)
        f.write(str(summary['failed']))
        f.write(""" ✗</div>
            <div class="summary-item">
                <strong>Success Rate:</strong> """)
        f.write(summary['success_rate'])
        f.write("""</div>
        </div>
        
        <h2>Test Results</h2>
""")
        
        # Test Results
        for i, result in enumerate(all_results, 1):
            status_class = "" if result.get('success', False) else "failed"
            f.write(f"""        <div class="test-item {status_class}">
            <h3>{i}. {result['name']} ({result['type']})</h3>
            
            <strong>Code:</strong>
            <div class="code-block">""")
            # Escape HTML in code
            code_escaped = result['code'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            f.write(code_escaped)
            f.write("""</div>
""")
            
            if result.get('success', False):
                f.write("""            <strong>Generated Comment:</strong>
            <div class="comment-block">""")
                comment = result.get('comment', 'N/A')
                comment_escaped = comment.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                f.write(comment_escaped)
                f.write("""</div>
            <div class="metadata">
                Model: """)
                f.write(result.get('model', 'N/A'))
                f.write(" | Language: ")
                f.write(result.get('language', 'python'))
                f.write("""</div>
""")
            else:
                f.write("""            <div class="error-block">
                <strong>Error:</strong> """)
                error_escaped = result.get('error', 'Unknown error').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                f.write(error_escaped)
                f.write("""</div>
""")
            
            f.write("""        </div>
""")
        
        f.write("""    </div>
</body>
</html>""")
    
    return html_file


def save_results(all_results, summary):
    """Save test results to JSON, Markdown, and HTML files."""
    json_file = save_results_json(all_results, summary)
    md_file = save_results_markdown(all_results, summary)
    html_file = save_results_html(all_results, summary)
    
    print(f"\n✓ Results saved to:")
    print(f"  - {json_file} (JSON)")
    print(f"  - {md_file} (Markdown)")
    print(f"  - {html_file} (HTML)")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("AI Comment Generator - Test Suite")
    print("=" * 70)
    
    # Test health first
    if not test_health():
        return
    
    # Wait a moment
    print("\nWaiting 2 seconds before starting tests...")
    time.sleep(2)
    
    # Get code samples
    print("\n" + "=" * 70)
    print("Extracting Code Samples")
    print("=" * 70)
    samples = get_code_samples()
    print(f"Found {len(samples)} code samples to test\n")
    
    # Test each sample
    results = {"success": 0, "failed": 0}
    all_test_results = []
    
    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] ", end="")
        result = test_comment_generation(
            sample["code"],
            sample["name"],
            sample["type"]
        )
        
        all_test_results.append(result)
        
        if result.get("success", False):
            results["success"] += 1
        else:
            results["failed"] += 1
        
        # Small delay between requests
        if i < len(samples):
            time.sleep(1)
    
    # Summary
    summary = {
        "total_tests": len(samples),
        "successful": results["success"],
        "failed": results["failed"],
        "success_rate": f"{(results['success']/len(samples)*100):.1f}%"
    }
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Total tests: {summary['total_tests']}")
    print(f"✓ Successful: {summary['successful']}")
    print(f"✗ Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']}")
    print("=" * 70)
    
    # Save results to file
    save_results(all_test_results, summary)


if __name__ == "__main__":
    main()

