# Human Evaluation Guide

## For Evaluators

Thank you for participating in this evaluation!

### What You'll Do
Evaluate AI-generated code samples on:
- **Correctness**: Does the code work? Does it solve the problem?
- **Readability**: Is the code well-formatted and clear?
- **Usefulness**: Would you use this code in a real project?
- **Hallucinations**: Does the code contain fabricated APIs, functions, or logic?

### Time Required
- ~30-45 minutes
- 40 samples to evaluate

### Instructions

1. **Download the template**:
   - File: `human_eval_template.xlsx`
   - Download from this repository

2. **Fill out your evaluation**:
   - Open the Excel file
   - Read the "Instructions" sheet
   - Go to "Evaluation" sheet
   - Rate each sample (columns F-L)
   - Save your changes

3. **Submit your responses**:
   - Rename file to: `human_eval_[YOUR_NAME].xlsx`
   - Email to: [your-email]
   - Or upload to shared drive: [link]
   - **Deadline**: [date]

### Questions?
Contact: [your-contact-info]

## For Project Owner (Mariam)

### Collecting Responses

1. Receive Excel files from evaluators
2. Save them in `evaluation/responses/` folder:
```
   evaluation/responses/
   ├── human_eval_evaluator1.xlsx
   ├── human_eval_evaluator2.xlsx
   └── human_eval_evaluator3.xlsx
```

3. Run consolidation script (see next step)