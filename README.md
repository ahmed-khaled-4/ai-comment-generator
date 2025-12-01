# AI Comment Generator

An AI-powered system for automatically generating code comments using large language models. This project implements a FastAPI-based prototype that leverages Ollama's DeepSeek-Coder 6.7B model to generate Python docstrings for functions and classes.

## Features

- 🤖 **AI-Powered Comment Generation**: Automatically generates docstrings using DeepSeek-Coder 6.7B
- 🚀 **FastAPI REST API**: Easy-to-use HTTP API with interactive documentation
- 📝 **Multiple Comment Types**: Supports function, class, and inline comments
- 🔧 **Configurable Parameters**: Adjustable temperature, max tokens, and model selection
- 🧹 **Smart Post-Processing**: Automatic cleaning of generated output
- 📊 **Comprehensive Testing**: Test suite with 12 code samples (100% success rate)
- 📈 **Performance Metrics**: Tracks latency, token usage, and generation quality
- 🔒 **Privacy-First**: Local Ollama deployment ensures code stays on-premises

## Project Structure

```
ai-comment-generator/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
├── test_code_samples.py           # Code samples for testing
├── test_comment_generation.py    # Test suite runner
├── results.json                   # Test results (JSON)
├── results.md                     # Test results (Markdown)
├── results.html                   # Test results (HTML)
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ai_client.py          # OllamaService - LLM integration
│   │   └── model_config.py       # Model configuration management
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── templates.py          # Prompt templates (placeholder)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py     # Dataset loading (placeholder)
│   │   └── preprocessor.py      # Code preprocessing (placeholder)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Evaluation metrics (placeholder)
│   │   └── evaluator.py         # Evaluation orchestration (placeholder)
│   ├── logging/
│   │   ├── __init__.py
│   │   └── logger.py             # Logging utilities (placeholder)
│   └── main.py                   # FastAPI application entry point
├── dataset/
│   ├── clean_dataset.json        # 100 code-comment pairs
│   └── dataset_description.md    # Dataset documentation
├── experiments/
│   ├── run_experiments.py        # Batch experiment runner (placeholder)
│   ├── configs/
│   │   └── experiment_config.yaml # Experiment configs (placeholder)
│   └── notebooks/
│       └── analysis.ipynb        # Results analysis (placeholder)
├── data/
│   ├── raw/                      # Raw dataset storage
│   ├── processed/                # Preprocessed dataset storage
│   └── results/                  # Experiment results storage
├── logs/
│   └── prompts/                  # Logged prompts/responses
├── reports/
│   └── phase2_report.md          # Phase 2 research report (IEEE format)
└── tests/
    └── test_evaluation.py        # Evaluation tests (placeholder)
```

### Key Files

- **`src/main.py`**: FastAPI application with `/generate_comment` and `/health` endpoints
- **`src/models/ai_client.py`**: Core LLM integration using Ollama, includes prompt formatting and response cleaning
- **`src/models/model_config.py`**: Configuration management for model parameters
- **`test_comment_generation.py`**: Comprehensive test suite that evaluates the API on 12 code samples
- **`test_code_samples.py`**: Test dataset with 12 Python code samples (functions and classes)
- **`reports/phase2_report.md`**: Detailed research report in IEEE format

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
  "temperature": 0.4,
  "max_tokens": 600,
  "model": "deepseek-coder:6.7b"
}
```

**Parameters:**
- `code` (required): Source code to generate comment for
- `language` (optional, default: "python"): Programming language
- `comment_type` (optional, default: "function"): Type of comment - `function`, `class`, or `inline`
- `temperature` (optional, default: 0.4): Sampling temperature (0.0-2.0). Lower values produce more focused output
- `max_tokens` (optional, default: 600): Maximum tokens to generate (1-2000)
- `model` (optional): Ollama model to use (overrides default)

**Response:**
```json
{
  "comment": "Args:\n    a: first number for addition\n    b: second number for addition\nReturns:\n    int: sum of two numbers",
  "model": "deepseek-coder:6.7b",
  "language": "python",
  "comment_type": "function",
  "metadata": {
    "temperature": 0.4,
    "max_tokens": 600,
    "top_p": 0.9,
    "latency": 2.57,
    "prompt_tokens": 266,
    "completion_tokens": 37,
    "total_tokens": 303
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
    "temperature": 0.4,
    "max_tokens": 600
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

### Running the Test Suite

The project includes a comprehensive test suite that evaluates the API on 12 code samples:

```bash
# Make sure the API server is running first
python -m src.main

# In another terminal, run the test suite
python test_comment_generation.py
```

The test suite will:
- Test all code samples from `test_code_samples.py`
- Generate comments for functions and classes
- Save results in multiple formats:
  - `results.json` - Machine-readable JSON format
  - `results.md` - Human-readable Markdown format
  - `results.html` - Interactive HTML report

**Test Results Summary:**
- ✅ **Success Rate**: 100% (12/12 tests passed)
- ⚡ **Average Latency**: 4.2 seconds
- 📊 **Average Tokens**: 328 tokens per generation
- 🎯 **Coverage**: 9 functions, 3 classes (simple to complex)

## Dataset

The project includes a curated dataset of 100 high-quality code-comment pairs:

- **Location**: `dataset/clean_dataset.json`
- **Source**: Open-source GitHub repositories
- **Languages**: Python (primary), Java
- **Licenses**: MIT, Apache-2.0, BSD
- **Quality**: Human-written docstrings, deduplicated, quality-checked

See `dataset/dataset_description.md` for more details.

## Performance Metrics

Based on test results with DeepSeek-Coder 6.7B:

| Metric | Value |
|--------|-------|
| Success Rate | 100% (12/12) |
| Average Latency | 4.2 seconds |
| Average Prompt Tokens | 280 tokens |
| Average Completion Tokens | 48 tokens |
| Average Total Tokens | 328 tokens |
| Latency Range | 2.2s - 12.8s |

## Phase 2 Deliverables

- [x] ✅ Prototype Implementation
- [x] ✅ Experimental Setup & Dataset
- [x] ✅ Early Experimental Results
- [x] ✅ Hallucination & Error Analysis
- [x] ✅ Early Research Report
- [x] ✅ GitHub Repository

**Phase 2 Status**: ✅ **COMPLETE**

See `reports/phase2_report.md` for the detailed research report (IEEE format).

## Project Status

**Current Phase**: Phase 2 Complete ✅

**Next Steps (Phase 3)**:
- Enhanced post-processing and format validation
- Multi-language support (Java, JavaScript, C++)
- Expanded evaluation with automated metrics (BLEU, ROUGE)
- Human evaluation study
- Comparative analysis across multiple models

## Contributing

This is a research project. For questions or contributions, please open an issue or contact the project maintainers.

## License

[Add your license here]

## Acknowledgments

- [Ollama](https://ollama.com) for providing local LLM infrastructure
- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder) for the DeepSeek-Coder model
- Open-source contributors whose code samples are included in the dataset