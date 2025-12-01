import pandas as pd
import glob
import os
from scipy.stats import pearsonr

RESPONSES_DIR = "evaluation/responses/"
OUTPUT_PATH = "results/human_ratings.csv"

def load_all_responses():
    all_files = glob.glob(os.path.join(RESPONSES_DIR, "*.*"))
    if not all_files:
        raise FileNotFoundError(f"No evaluator files found in {RESPONSES_DIR}")
    
    print(f"Found {len(all_files)} evaluator responses")
    
    all_responses = []
    for file_path in all_files:
        evaluator_name = os.path.basename(file_path).replace('human_eval_', '').split('.')[0]
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, sheet_name='Evaluation')
        else:
            print(f"Skipping unsupported file type: {file_path}")
            continue
        
        df.columns = [col.strip() for col in df.columns]
        df['evaluator_id'] = evaluator_name
        all_responses.append(df)
        print(f"  ✓ Loaded {evaluator_name}: {len(df)} samples")
    
    combined = pd.concat(all_responses, ignore_index=True)
    return combined

def calculate_agreement(responses_df):
    print("\n=== Inter-Rater Agreement ===")
    evaluators = responses_df['evaluator_id'].unique()
    if len(evaluators) < 2:
        print("Only one evaluator - skipping agreement calculation")
        return
    
    metrics = ['Correctness', 'Readability', 'Usefulness']
    for metric in metrics:
        print(f"\n{metric}:")
        for i in range(len(evaluators)):
            for j in range(i+1, len(evaluators)):
                eval1_data = responses_df[responses_df['evaluator_id'] == evaluators[i]]
                eval2_data = responses_df[responses_df['evaluator_id'] == evaluators[j]]
                
                merged = eval1_data[['id', metric]].merge(
                    eval2_data[['id', metric]], 
                    on='id', 
                    suffixes=('_1', '_2')
                )
                
                scores1 = pd.to_numeric(merged[f'{metric}_1'], errors='coerce')
                scores2 = pd.to_numeric(merged[f'{metric}_2'], errors='coerce')
                
                valid_mask = ~(scores1.isna() | scores2.isna())
                scores1 = scores1[valid_mask]
                scores2 = scores2[valid_mask]
                
                if len(scores1) > 1:
                    corr, p_value = pearsonr(scores1, scores2)
                    print(f"  {evaluators[i]} vs {evaluators[j]}: r={corr:.3f} (p={p_value:.3f})")

def aggregate_ratings(responses_df):
    """Aggregate ratings including hallucination columns"""
    # Numeric rating columns
    rating_cols = ['Correctness', 'Readability', 'Usefulness']
    
    for col in rating_cols:
        responses_df[col] = pd.to_numeric(responses_df[col], errors='coerce')
    
    # Hallucination columns
    if 'Has Hallucination' in responses_df.columns:
        responses_df['Has Hallucination'] = responses_df['Has Hallucination'].map({'Yes': 1, 'No': 0}).fillna(0)
    if 'Severity' in responses_df.columns:
        responses_df['Severity'] = pd.to_numeric(responses_df['Severity'], errors='coerce')
    
    aggregated = responses_df.groupby('id').agg({
        'Correctness': ['mean', 'std', 'count'],
        'Readability': ['mean', 'std', 'count'],
        'Usefulness': ['mean', 'std', 'count'],
        'Has Hallucination': 'mean',
        'Severity': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = [
        'id',
        'correctness_mean', 'correctness_std', 'correctness_count',
        'readability_mean', 'readability_std', 'readability_count',
        'usefulness_mean', 'usefulness_std', 'usefulness_count',
        'has_hallucination_mean',
        'severity_mean', 'severity_std', 'severity_count'
    ]
    
    return aggregated

def main():
    print("Loading evaluator responses...")
    responses = load_all_responses()
    
    print(f"\nTotal responses collected: {len(responses)}")
    print(f"Unique samples evaluated: {responses['id'].nunique()}")
    print(f"Number of evaluators: {responses['evaluator_id'].nunique()}")
    
    calculate_agreement(responses)
    
    print("\n\nAggregating ratings...")
    aggregated = aggregate_ratings(responses)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    aggregated.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved aggregated ratings to {OUTPUT_PATH}")
    
    print("\n=== Rating Summary ===")
    print(aggregated[['correctness_mean', 'readability_mean', 'usefulness_mean',
                      'has_hallucination_mean', 'severity_mean']].describe())

if __name__ == "__main__":
    main()
