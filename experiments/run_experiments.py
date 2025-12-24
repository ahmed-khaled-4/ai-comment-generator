# Batch experiment runner
import sys
import os

# Add the project root to the python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TO THIS:
from src.evaluation.evaluator import EvaluationPipeline

def main():
    # 1. Start the evaluator
    evaluator = EvaluationPipeline()

    # 2. Define paths
    # Note: Adjust '../dataset/clean_dataset.json' if your folder structure is different
    dataset_file = os.path.join("dataset", "clean_dataset.json")
    output_file = "phase3_evaluation_results.json"

    # 3. Check if dataset exists
    if not os.path.exists(dataset_file):
        print(f"Error: Could not find dataset at {dataset_file}")
        return

    # 4. Run!
    print("Starting Phase 3 Batch Evaluation...")
    evaluator.evaluate()

if __name__ == "__main__":
    main()