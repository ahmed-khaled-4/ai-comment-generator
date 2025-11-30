import evaluate
import pandas as pd
from typing import List, Dict


class MetricsCalculator:
    def __init__(self):
        print("Loading metrics...")
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        print("Metrics loaded successfully.")

    def compute_bleu(self, references: List[str], predictions: List[str]) -> Dict:
        return self.bleu.compute(predictions=predictions, references=references)

    def compute_rouge(self, references: List[str], predictions: List[str]) -> Dict:
        return self.rouge.compute(predictions=predictions, references=references)

    def compute_all(self, references: List[str], predictions: List[str]) -> Dict:
        bleu = self.compute_bleu(references, predictions)
        rouge = self.compute_rouge(references, predictions)

        # Print results to terminal
        print("\n--- Overall Metrics ---")
        print(f"BLEU: {bleu['bleu']:.4f}")
        print(f"ROUGE-1: {rouge['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge['rougeL']:.4f}")
        print("----------------------\n")

        return {
            "bleu": bleu["bleu"],
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
        }

    def save_to_csv(self, results: List[Dict], output_path="results/metrics.csv"):
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f" Metrics saved to {output_path}")
