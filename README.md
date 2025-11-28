# AI Comment Generator

An AI-powered system for automatically generating code comments using large language models.

## Project Structure

```
ai-comment-generator/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ai_client.py          # AI model integration
│   │   └── model_config.py       # Model configuration management
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── templates.py          # Prompt templates
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py     # Dataset loading/preprocessing
│   │   └── preprocessor.py       # Code preprocessing utilities
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # BLEU, ROUGE, cosine similarity
│   │   └── evaluator.py          # Evaluation orchestration
│   ├── logging/
│   │   ├── __init__.py
│   │   └── logger.py             # Prompt/response logging
│   └── main.py                   # Main entry point
├── experiments/
│   ├── run_experiments.py        # Batch experiment runner
│   ├── configs/
│   │   └── experiment_config.yaml # Experiment configurations
│   └── notebooks/
│       └── analysis.ipynb        # Results analysis
├── data/
│   ├── raw/                      # Raw dataset
│   ├── processed/                # Preprocessed dataset
│   └── results/                  # Experiment results
├── logs/
│   └── prompts/                  # Logged prompts/responses
├── reports/
│   └── phase2_report.md          # Phase 2 research report
└── tests/
    └── test_evaluation.py
```

## Setup & Installation

### Prerequisites

1. **Install Ollama** (if not already installed):
   - Download from: https://ollama.com
   - Or use: `brew install ollama` (macOS) / `curl -fsSL https://ollama.com/install.sh | sh` (Linux)

2. **Pull the code model**:
   ```bash
   ollama pull deepseek-coder:6.7b
   # Or other models: codellama, deepseek-coder, qwen2.5-coder, llama3.2
   ```

3. **Start Ollama server** (usually runs automatically):
   ```bash
   ollama serve
   ```

### Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env if needed (Ollama doesn't require API keys)
   ```

## Usage

### Starting the API Server

Run the FastAPI server:

```bash
# From project root
python -m src.main

# Or using uvicorn directly
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check

**GET** `/health`

Check if the service is running and Ollama is connected.

**Response:**
```json
{
  "status": "healthy",
  "service": "ollama",
  "model": "deepseek-coder:6.7b"
}
```

#### 2. Generate Comment

**POST** `/generate_comment`

Generate a code comment for the provided code.

**Request Body:**
```json
{
  "code": "def add(a, b):\n    return a + b",
  "language": "python",
  "comment_type": "function",
  "temperature": 0.7,
  "max_tokens": 400,
  "model": "deepseek-coder:6.7b"
}
```

**Parameters:**
- `code` (required): Source code to generate comment for
- `language` (optional, default: "python"): Programming language
- `comment_type` (optional, default: "function"): Type of comment - `function`, `class`, or `inline`
- `temperature` (optional, default: 0.7): Sampling temperature (0.0-2.0)
- `max_tokens` (optional, default: 400): Maximum tokens to generate (1-2000)
- `model` (optional): Ollama model to use (overrides default)

**Response:**
```json
{
  "comment": "\"\"\"Add two numbers together.\n\nArgs:\n    a: First number\n    b: Second number\n\nReturns:\n    Sum of a and b\n\"\"\"",
  "model": "deepseek-coder:6.7b",
  "language": "python",
  "comment_type": "function",
  "metadata": {
    "temperature": 0.7,
    "max_tokens": 400
  }
}
```

### Example Requests

#### Using cURL

```bash
# Generate function comment
curl -X POST "http://localhost:8000/generate_comment" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
    "language": "python",
    "comment_type": "function"
  }'
```

#### Using Python

```python
import requests

url = "http://localhost:8000/generate_comment"
payload = {
    "code": "def add(a, b):\n    return a + b",
    "language": "python",
    "comment_type": "function",
    "temperature": 0.7
}

response = requests.post(url, json=payload)
result = response.json()
print(result["comment"])
```

#### Using JavaScript/Node.js

```javascript
const response = await fetch('http://localhost:8000/generate_comment', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    code: 'def add(a, b):\n    return a + b',
    language: 'python',
    comment_type: 'function'
  })
});

const result = await response.json();
console.log(result.comment);
```

#### Using the Interactive Docs

1. Start the server
2. Navigate to http://localhost:8000/docs
3. Click on `/generate_comment` endpoint
4. Click "Try it out"
5. Fill in the request body
6. Click "Execute"

### Example: Different Comment Types

**Function Comment:**
```json
{
  "code": "def process_data(data):\n    return sorted(data)",
  "comment_type": "function"
}
```

**Class Comment:**
```json
{
  "code": "class DataProcessor:\n    def __init__(self):\n        self.data = []",
  "comment_type": "class"
}
```

**Inline Comment:**
```json
{
  "code": "result = [x*2 for x in range(10)]",
  "comment_type": "inline"
}
```

### Testing the API

Test with a simple Python script:

```python
# test_api.py
import requests

# Test health endpoint
health = requests.get("http://localhost:8000/health")
print("Health:", health.json())

# Test comment generation
response = requests.post(
    "http://localhost:8000/generate_comment",
    json={
        "code": "def multiply(x, y):\n    return x * y",
        "language": "python",
        "comment_type": "function"
    }
)
print("\nGenerated Comment:")
print(response.json()["comment"])
```

Run: `python test_api.py`

## Phase 2 Deliverables

- [ ] Prototype Implementation
- [ ] Experimental Setup & Dataset
- [ ] Early Experimental Results
- [ ] Hallucination & Error Analysis
- [ ] Early Research Report
- [ ] GitHub Repository