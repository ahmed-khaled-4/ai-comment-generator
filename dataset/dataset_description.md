# Dataset Description

This dataset contains 100 high–quality code–comment pairs
automatically extracted from open–source GitHub repositories.

## Key Features
- Languages: Python and Java
- Only human-written function-level comments (docstrings / Javadoc)
- Functions filtered for clarity and meaningful documentation
- Fully open-source licenses (MIT, Apache-2.0, BSD)
- Deduplicated and quality-checked samples
- Extracted using automated GitHub API pipeline

## Files
- `clean_dataset.json` — The dataset containing all collected samples
- `dataset_description.md` — This file (short description)

## Purpose
Designed for training and evaluating AI models that generate or improve
code comments from real-world open-source projects.