import requests
import json
import pandas as pd
from tqdm import tqdm
import time
import os

# --- Correct Paths for Your Project ---
DATASET_PATH = "dataset/clean_dataset.json"
OUTPUT_PATH = "results/model_outputs.csv"
API_URL = "http://localhost:8000"

def load_dataset():
    """Load dataset from CSV or JSON."""
    if DATASET_PATH.endswith(".json"):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError("Dataset must be JSON")

def generate_output(row):
    """Send request to FastAPI model server."""
    try:
        payload = {
            "code": row.get("code", ""),
            "language": row.get("language", "python"),
            # you can remove comment_type or keep default
            "comment_type": "function"
        }

        response = requests.post(
            f"{API_URL}/generate_comment",
            json=payload,
            timeout=40
        )

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "output": result.get("comment", ""),
                "model": result.get("model", "unknown"),
                "prompt": result.get("prompt", "")
            }

        return {
            "success": False,
            "output": f"Error: {response.status_code}",
            "model": "error",
            "prompt": ""
        }

    except Exception as e:
        return {
            "success": False,
            "output": f"Exception: {str(e)}",
            "model": "error",
            "prompt": ""
        }

def main():
    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} samples")

    os.makedirs("results", exist_ok=True)

    results = []

    print("\nGenerating outputs…")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        result = generate_output(row)

        results.append({
            "id": idx,
            "input": row.get("code", ""),
            "expected": row.get("human_comment", ""),
            "task": "comment_generation",
            "model_output": result["output"],
            "model": result["model"],
            "prompt_used": result["prompt"],
            "success": result["success"]
        })

        time.sleep(0.3)  # avoid server overload

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✓ Saved {len(out_df)} outputs to {OUTPUT_PATH}")
    print(f"✓ Success rate: {out_df['success'].sum()}/{len(out_df)}")


if __name__ == "__main__":
    main()
