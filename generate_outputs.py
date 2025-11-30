import requests
import json
import pandas as pd
from tqdm import tqdm
import time
import os

# Configuration
API_URL = "http://localhost:8000"  
DATASET_PATH = r"C:\Users\laptop.house\ai-comment-generator\dataset\clean_dataset.json"  
OUTPUT_PATH = "results/model_outputs.csv"

def load_dataset():
    """Load dataset from CSV or JSON"""
    if DATASET_PATH.endswith('.csv'):
        return pd.read_csv(DATASET_PATH)
    elif DATASET_PATH.endswith('.json'):
        with open(DATASET_PATH, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError("Dataset must be CSV or JSON")

def generate_output(row, endpoint="/generate_comment"):
    """Call API to generate output for one sample"""
    try:
        payload = {
            'code': row.get('input_code', row.get('input', '')),
            'language': row.get('language', 'python'),
            'comment_type': row.get('comment_type', 'function'),
            'temperature': 0.7,
            'max_tokens': 400
        }

        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'output': result.get('comment', ''),
                'model': result.get('model', 'unknown'),
                'prompt': ''
            }
        else:
            return {
                'success': False,
                'output': f"Error: {response.status_code}",
                'model': 'error',
                'prompt': ''
            }
    except Exception as e:
        return {
            'success': False,
            'output': f"Exception: {str(e)}",
            'model': 'error',
            'prompt': ''
        }

def main():
    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} samples")
    
    print("\nGenerating outputs (this may take a while)...")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        result = generate_output(row)
        
        results.append({
            'id': idx,
            'input': row.get('input_code', row.get('input', '')),
            'expected': row.get('expected_output', row.get('expected', '')),
            'task': row.get('task', 'unknown'),
            'model_output': result['output'],
            'model': result['model'],
            'prompt_used': result['prompt'],
            'success': result['success']
        })
        
        # Rate limiting - adjust as needed
        time.sleep(0.5)
    
    # Save results
    output_df = pd.DataFrame(results)
    os.makedirs('data', exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✓ Saved {len(output_df)} outputs to {OUTPUT_PATH}")
    print(f"  Success rate: {output_df['success'].sum()}/{len(output_df)}")
    print(f"  Failed: {(~output_df['success']).sum()}")

if __name__ == "__main__":
    main()