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

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

Coming soon...

## Phase 2 Deliverables

- [ ] Prototype Implementation
- [ ] Experimental Setup & Dataset
- [ ] Early Experimental Results
- [ ] Hallucination & Error Analysis
- [ ] Early Research Report
- [ ] GitHub Repository