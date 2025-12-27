# AI Comment Generator

An AI-powered system for automatically generating high-quality code comments using large language models. This project implements a FastAPI-based REST API that leverages Ollama's DeepSeek-Coder 6.7B model to generate Python docstrings and code documentation with comprehensive validation, safety controls, and multi-agent governance.

##  Project Description

The AI Comment Generator is a production-ready system that automatically generates code comments and documentation for source code. It features:

- **Multi-layer validation** to ensure output quality and safety
- **Multi-agent framework** using CrewAI for generator-validator workflows
- **Ethical governance** with STEA (Safety, Transparency, Explainability, Accountability) principles
- **Comprehensive monitoring** and human-in-the-loop review capabilities
- **Privacy-first design** with local LLM deployment via Ollama

The system supports multiple programming languages, comment types (function, class, inline), and includes automated retry mechanisms with fallback strategies.

##  Features

### Core Functionality
-  **AI-Powered Comment Generation**: Automatically generates docstrings using DeepSeek-Coder 6.7B
- **FastAPI REST API**: Easy-to-use HTTP API with interactive Swagger documentation
-  **Multiple Comment Types**: Supports function, class, and inline comments
- **Multi-Language Support**: 20+ programming languages (Python, JavaScript, Java, C++, Go, Rust, etc.)
-  **Configurable Parameters**: Adjustable temperature, max tokens, and model selection

### Validation & Safety
-  **Input Validation**: Syntax checking, language support, size limits, security pattern detection
-  **Output Validation**: Format correctness, completeness checks, quality metrics
-  **Safety Rules**: Content filtering, PII detection, code injection prevention
-  **Automatic Retry**: Exponential backoff retry with fallback strategies
-  **Human Review Integration**: Confidence-based flagging for manual review

### Advanced Features
-  **Multi-Agent System**: CrewAI-based generator-validator workflow
-  **Comprehensive Monitoring**: Real-time tracking of violations, rejections, and quality metrics
-  **Performance Metrics**: Tracks latency, token usage, and generation quality
-  **Privacy-First**: Local Ollama deployment ensures code stays on-premises
-  **Evaluation Tools**: BLEU, ROUGE, and BERTScore metrics for quality assessment

##  Installation / Setup Instructions

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running

### Step 1: Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download the installer from [https://ollama.com](https://ollama.com)

### Step 2: Pull the Code Model

```bash
ollama pull deepseek-coder:6.7b
```

Alternative models you can use:
- `codellama`
- `qwen2.5-coder`
- `llama3.2`

### Step 3: Start Ollama Server

The Ollama server usually starts automatically. If not, run:

```bash
ollama serve
```

Verify it's running:
```bash
ollama list
```

### Step 4: Clone the Repository

```bash
git clone <repository-url>
cd ai-comment-generator
```

### Step 5: Install Python Dependencies

Create a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 6: Verify Installation

Check that all dependencies are installed:

```bash
python -c "import fastapi, ollama; print('Installation successful!')"
```

## Running the Project

### Start the API Server

From the project root directory:

```bash
# Option 1: Using Python module
python -m src.main

# Option 2: Using uvicorn directly
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive API Docs (Swagger)**: http://localhost:8000/docs
- **Alternative API Docs (ReDoc)**: http://localhost:8000/redoc

### Verify the Server is Running

```bash
# Test health endpoint
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "ollama",
  "model": "deepseek-coder:6.7b"
}
```

## Example Requests / Usage

### 1. Generate a Function Comment

**Request:**
```bash
curl -X POST "http://localhost:8000/generate_comment" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
    "language": "python",
    "comment_type": "function",
    "temperature": 0.4,
    "max_tokens": 600
  }'
```

**Response:**
```json
{
  "comment": "Calculates the nth Fibonacci number using recursion.\n\nArgs:\n    n (int): The position in the Fibonacci sequence.\n\nReturns:\n    int: The nth Fibonacci number.",
  "model": "deepseek-coder:6.7b",
  "language": "python",
  "comment_type": "function",
  "metadata": {
    "request_id": "abc123...",
    "temperature": 0.4,
    "max_tokens": 600,
    "latency": 2.57,
    "prompt_tokens": 266,
    "completion_tokens": 37,
    "total_tokens": 303,
    "validation_passed": true,
    "safety_checked": true,
    "requires_human_review": false
  }
}
```

### 2. Generate a Class Comment

**Request:**
```bash
curl -X POST "http://localhost:8000/generate_comment" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "class DataProcessor:\n    def __init__(self, data):\n        self.data = data\n    def process(self):\n        return sorted(self.data)",
    "language": "python",
    "comment_type": "class"
  }'
```

### 3. Using Python Client

```python
import requests

url = "http://localhost:8000/generate_comment"
payload = {
    "code": "def add(a, b):\n    return a + b",
    "language": "python",
    "comment_type": "function",
    "temperature": 0.4,
    "max_tokens": 600
}

response = requests.post(url, json=payload)
result = response.json()

print("Generated Comment:")
print(result["comment"])
print("\nMetadata:")
print(f"Latency: {result['metadata']['latency']}s")
print(f"Tokens: {result['metadata']['total_tokens']}")
```

### 4. Using JavaScript/Node.js

```javascript
const response = await fetch('http://localhost:8000/generate_comment', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    code: 'def multiply(x, y):\n    return x * y',
    language: 'python',
    comment_type: 'function',
    temperature: 0.4,
    max_tokens: 600
  })
});

const result = await response.json();
console.log('Generated Comment:', result.comment);
console.log('Latency:', result.metadata.latency, 'seconds');
```

### 5. Using the Interactive API Documentation

1. Start the server (see [Running the Project](#-running-the-project))
2. Open your browser and navigate to http://localhost:8000/docs
3. Click on the `/generate_comment` endpoint
4. Click the "Try it out" button
5. Fill in the request body with your code
6. Click "Execute" to see the response

### 6. Multi-Agent Generation (Advanced)

```bash
curl -X POST "http://localhost:8000/multi_agent/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def process_data(data):\n    return sorted(data)",
    "language": "python",
    "comment_type": "function",
    "governance_level": "standard",
    "max_retries": 2
  }'
```

### 7. Get Validation Statistics

```bash
curl http://localhost:8000/validation/statistics
```

### 8. View Pending Human Reviews

```bash
curl http://localhost:8000/human_review/pending
```

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API information |
| `/health` | GET | Health check and service status |
| `/generate_comment` | POST | Generate comment with full validation |
| `/multi_agent/generate` | POST | Multi-agent generation with governance |
| `/evaluate` | POST | Generate comment and calculate metrics |

### Validation & Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/validation/statistics` | GET | Get validation system statistics |
| `/validation/report` | GET | Generate validation monitoring report |
| `/human_review/pending` | GET | Get pending reviews |
| `/human_review/approve/{comment_id}` | POST | Approve a flagged comment |
| `/human_review/reject/{comment_id}` | POST | Reject a flagged comment |
| `/human_review/statistics` | GET | Get review statistics |

### Request Parameters

**POST `/generate_comment` Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `code` | string | Yes | - | Source code to generate comment for |
| `language` | string | No | "python" | Programming language |
| `comment_type` | string | No | "function" | Type: `function`, `class`, or `inline` |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `max_tokens` | integer | No | 400 | Maximum tokens to generate (1-2000) |
| `model` | string | No | "deepseek-coder:6.7b" | Ollama model to use |

## Testing

### Run Validation System Tests

```bash
python tests/test_validation_system.py
```

### Run Evaluation Tests

```bash
python tests/test_evaluation.py
```

### Test All Features

```bash
python test_all_features.py
```

## Technologies / Tools Used

### Core Framework & Libraries
- **FastAPI** (0.123.0) - Modern, fast web framework for building APIs
- **Uvicorn** (0.38.0) - ASGI server for FastAPI
- **Pydantic** (2.12.5) - Data validation using Python type annotations

### AI & Machine Learning
- **Ollama** (0.6.1) - Local LLM deployment and management
- **DeepSeek-Coder 6.7B** - Code-specialized language model
- **CrewAI** (≥1.0.0) - Multi-agent framework for AI workflows
- **Transformers** (4.57.3) - HuggingFace transformers library
- **Torch** (2.9.1) - PyTorch for model inference

### Evaluation & Metrics
- **HuggingFace Evaluate** (0.4.6) - Evaluation metrics library
- **BERTScore** (0.3.13) - Semantic similarity evaluation
- **NLTK** (3.9.2) - Natural language processing toolkit
- **ROUGE-Score** (0.1.2) - ROUGE metric implementation

### Data Processing
- **Pandas** (2.3.3) - Data manipulation and analysis
- **NumPy** (2.3.5) - Numerical computing
- **Datasets** (4.4.2) - HuggingFace datasets library

### Development & Utilities
- **Python 3.8+** - Programming language
- **Requests** (2.32.5) - HTTP library for API calls
- **PyYAML** (6.0.3) - YAML parser for configuration files
- **Regex** (2025.11.3) - Advanced regular expressions

## Project Structure

```
ai-comment-generator/
├── src/                          # Source code
│   ├── main.py                   # FastAPI application entry point
│   ├── models/                   # AI model integration
│   │   ├── ai_client.py         # Ollama service wrapper
│   │   └── model_config.py      # Model configuration
│   ├── validators/              # Validation system
│   │   ├── input_request_validation.py
│   │   ├── output_validation.py
│   │   ├── safety_rules.py
│   │   └── rejection_retry.py
│   ├── human_review/            # Human review system
│   │   └── review_system.py
│   ├── monitoring/              # Monitoring and logging
│   │   └── validation_monitor.py
│   ├── multi_agent/             # Multi-agent framework
│   │   ├── crewai_agents.py
│   │   └── governance.py
│   ├── evaluation/              # Evaluation metrics
│   │   ├── metrics.py
│   │   └── evaluator.py
│   └── app_logging/            # Logging utilities
│       └── logger.py
├── dataset/                     # Evaluation dataset
│   ├── clean_dataset.json      # 100 code-comment pairs
│   └── dataset_description.md
├── tests/                       # Test suites
│   ├── test_validation_system.py
│   └── test_evaluation.py
├── logs/                        # Application logs
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Performance Metrics

Based on evaluation with DeepSeek-Coder 6.7B:

| Metric | Value |
|--------|-------|
| Success Rate | 100% (12/12 test samples) |
| Average Latency | 4.2 seconds |
| Average Prompt Tokens | 280 tokens |
| Average Completion Tokens | 48 tokens |
| Average Total Tokens | 328 tokens |
| Validation Overhead | <100ms per request |
| Retry Success Rate | 68.9% |

## Security & Privacy

- **Local Deployment**: All processing happens locally via Ollama
- **No External APIs**: Code never leaves your machine
- **Input Validation**: Security pattern detection prevents malicious code
- **PII Detection**: Automatic detection and filtering of sensitive information
- **Safety Rules**: Content filtering and quality enforcement


## Acknowledgments

- [Ollama](https://ollama.com) for providing local LLM infrastructure
- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder) for the DeepSeek-Coder model
- [CrewAI](https://github.com/joaomdmoura/crewAI) for the multi-agent framework
- Open-source contributors whose code samples are included in the dataset

## Additional Resources

- **Dataset Description**: See `dataset/dataset_description.md` for dataset information
- **API Documentation**: Interactive docs available at http://localhost:8000/docs when server is running

## Contributing

This is a research project. For questions, issues, or contributions:

1. Open an issue on GitHub
2. Contact the project maintainers
3. Submit a pull request with detailed description

---

**Note**: Make sure Ollama is running and the model is downloaded before starting the API server. The system requires at least 8GB RAM for optimal performance with DeepSeek-Coder 6.7B.
