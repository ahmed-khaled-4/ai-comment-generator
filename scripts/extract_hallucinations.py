import pandas as pd
import glob
import os

RESPONSES_DIR = "evaluation/responses/"
OUTPUT_PATH = "results/hallucinations.csv"

def load_all_responses():
    """Load all evaluator responses (CSV or Excel)"""
    all_files = glob.glob(os.path.join(RESPONSES_DIR, "*.*"))
    if not all_files:
        raise FileNotFoundError(f"No evaluator files found in {RESPONSES_DIR}")
    
    all_responses = []
    for file_path in all_files:
        evaluator_name = os.path.basename(file_path).replace('human_eval_', '').split('.')[0]
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, sheet_name='Evaluation')
        else:
            continue
        df.columns = [col.strip() for col in df.columns]
        df['evaluator_id'] = evaluator_name
        all_responses.append(df)
    
    return pd.concat(all_responses, ignore_index=True)

def categorize_hallucination(description):
    if pd.isna(description) or description.strip() == '':
        return 'Unknown'
    desc_lower = description.lower()
    if any(word in desc_lower for word in ['api', 'function', 'method', 'library', 'module', 'import']):
        return 'Non-existent API/Function'
    elif any(word in desc_lower for word in ['syntax', 'error', 'invalid', 'typo']):
        return 'Syntax Error'
    elif any(word in desc_lower for word in ['logic', 'wrong', 'incorrect', 'bug', 'calculation']):
        return 'Logic Error'
    elif any(word in desc_lower for word in ['variable', 'name', 'undefined', 'reference']):
        return 'Incorrect Variable Reference'
    elif any(word in desc_lower for word in ['parameter', 'argument', 'type']):
        return 'Incorrect Parameters'
    elif any(word in desc_lower for word in ['deprecated', 'outdated', 'old']):
        return 'Deprecated Usage'
    else:
        return 'Other'

def extract_root_cause(description, category):
    causes = {
        'Non-existent API/Function': 'Model hallucinated non-existent library functions.',
        'Syntax Error': 'Model generated syntactically invalid code.',
        'Logic Error': 'Model misunderstood problem requirements or generated wrong logic.',
        'Incorrect Variable Reference': 'Model referenced undefined or incorrectly named variables.',
        'Incorrect Parameters': 'Model used wrong function parameters.',
        'Deprecated Usage': 'Model used outdated API patterns.',
        'Other': 'Unclear hallucination type.'
    }
    return causes.get(category, 'Unclear hallucination type.')

def main():
    responses = load_all_responses()
    
    # Filter for hallucinations
    hallucination_col = 'Has Hallucination'
    severity_col = 'Severity'
    description_col = 'Hallucination Description'
    
    hallucinations = responses[responses[hallucination_col].str.lower() == 'yes'].copy()
    
    print(f"Found {len(hallucinations)} hallucination instances ({hallucinations['id'].nunique()} unique samples)")
    
    # Categorize and add root cause
    hallucinations['Error_Type'] = hallucinations[description_col].apply(categorize_hallucination)
    hallucinations['Root_Cause_Hypothesis'] = hallucinations['Error_Type'].apply(lambda c: extract_root_cause('', c))
    hallucinations['Hallucination_Severity'] = hallucinations[severity_col]
    
    # Select final columns
    final_columns = ['id', 'input', 'model_output', 'expected', 'Error_Type', 'Hallucination_Severity',
                     'Root_Cause_Hypothesis', description_col, 'evaluator_id']
    output_df = hallucinations[[col for col in final_columns if col in hallucinations.columns]].copy()
    output_df = output_df.rename(columns={
        'id': 'ID',
        'input': 'Input',
        'model_output': 'Model Output',
        'expected': 'Expected',
        description_col: 'Detailed Description',
        'evaluator_id': 'Reported By'
    })
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    
    # Summary
    print(f"Saved {len(output_df)} hallucinations to {OUTPUT_PATH}")
    print("\n=== Hallucination Categories ===")
    print(output_df['Error_Type'].value_counts())
    print("\n=== Severity Distribution ===")
    print(output_df['Hallucination_Severity'].value_counts())
    
    rate = (output_df['ID'].nunique() / responses['id'].nunique()) * 100
    print(f"\nHallucination Rate: {rate:.1f}% ({output_df['ID'].nunique()}/{responses['id'].nunique()} samples)")

if __name__ == "__main__":
    main()
