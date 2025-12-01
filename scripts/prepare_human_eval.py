"""
Select samples for human evaluation using stratified sampling
Run: python scripts/prepare_human_eval.py
"""
import pandas as pd
import numpy as np
import os

#Paths
OUTPUTS_PATH = "results/model_outputs.csv"
METRICS_PATH = "results/metrics.csv"
EVAL_SAMPLES_PATH = "results/human_eval_samples.csv"

def stratified_sampling(df, metrics_df=None, n_samples=40):
    """
    Select diverse samples for human evaluation
    """
    samples = []
    
    # 1. Random samples (10)
    random_samples = df.sample(min(10, len(df)), random_state=42)
    samples.append(random_samples)
    print(f"✓ Selected 10 random samples")
    
    # 2. Samples with automated metrics (if available)
    
    if metrics_df is not None:
        merged = df.merge(metrics_df, on='id', how='left')
        
        # Lowest metric samples (10)
        if 'bleu_score' in merged.columns:
            worst = merged.nsmallest(10, 'bleu_score')
            samples.append(worst[df.columns])
            print(f"✓ Selected 10 lowest BLEU score samples")
        
        # Highest metric samples (10)
        if 'bleu_score' in merged.columns:
            best = merged.nlargest(10, 'bleu_score')
            samples.append(best[df.columns])
            print(f"✓ Selected 10 highest BLEU score samples")
    
    # 3. Edge cases (10)
    # Long inputs
    df['input_length'] = df['input'].astype(str).str.len()
    long_inputs = df.nlargest(5, 'input_length')
    samples.append(long_inputs)
    
    # Failed generations
    if 'success' in df.columns:
        failed = df[df['success'] == False].sample(min(5, len(df[df['success'] == False])))
        samples.append(failed)
    
    print(f"✓ Selected 10 edge case samples")
    
    # Combine and remove duplicates
    eval_df = pd.concat(samples, ignore_index=True)
    eval_df = eval_df.drop_duplicates(subset=['id'])
    eval_df = eval_df.head(n_samples)
    
    # Add evaluation columns
    eval_df['correctness_rating'] = ''
    eval_df['readability_rating'] = ''
    eval_df['usefulness_rating'] = ''
    eval_df['has_hallucination'] = ''
    eval_df['hallucination_severity'] = ''
    eval_df['hallucination_description'] = ''
    eval_df['evaluator_notes'] = ''
    
    return eval_df

def main():
    print("Loading model outputs...")
    df = pd.read_csv(OUTPUTS_PATH)
    print(f"Total outputs: {len(df)}")
    
    # Load metrics if available
    metrics_df = None
    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH, on_bad_lines='skip') 
        print(f"Loaded metrics for {len(metrics_df)} samples")

        if 'id' not in metrics_df.columns:
            metrics_df['id'] = range(len(metrics_df))

    # Perform sampling
    print("\nPerforming stratified sampling...")
    eval_samples = stratified_sampling(df, metrics_df, n_samples=40)
    
    # Save
    eval_samples.to_csv(EVAL_SAMPLES_PATH, index=False)
    print(f"\n✓ Saved {len(eval_samples)} samples to {EVAL_SAMPLES_PATH}")
    
    # Print summary
    print("\nSample Distribution:")
    if 'task' in eval_samples.columns:
        print(eval_samples['task'].value_counts())

if __name__ == "__main__":
    main()