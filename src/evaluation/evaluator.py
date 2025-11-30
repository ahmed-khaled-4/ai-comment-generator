import json
from pathlib import Path
from src.models.api_client import APIClient
from src.evaluation.metrics import MetricsCalculator


class EvaluationPipeline:
    def __init__(self, dataset_path="dataset/clean_dataset.json"):
        self.dataset_path = dataset_path
        self.client = APIClient()
        self.metrics_calc = MetricsCalculator()

    def load_dataset(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self):
        data = self.load_dataset()

        references = []
        predictions = []
        rows = []

        print("\n--- Generating comments ---\n")
        for idx, item in enumerate(data, start=1):
            lang = item["language"]
            code = item["code"]
            gold = item["human_comment"]

            # Generate model comment
            model_comment = self.client.generate_comment(code, lang)

            references.append(gold)
            predictions.append(model_comment)

            # Store dataset row
            row = {
                "language": lang,
                "code": code,
                "human_comment": gold,
                "model_comment": model_comment
            }
            rows.append(row)

            # Print preview
            preview = model_comment.replace("\n", " ")[:200]
            print(f"[{idx}/{len(data)}] Generated comment preview:\n{preview}\n")

        # Compute overall metrics (BLEU + ROUGE)
        scores = self.metrics_calc.compute_all(references, predictions)

        # Add same metrics to each row for CSV
        for r in rows:
            r.update(scores)

        # Save CSV
        Path("results").mkdir(exist_ok=True)
        self.metrics_calc.save_to_csv(rows)

        print("\n Evaluation completed successfully.\n")
        return scores


if __name__ == "__main__":
    EvaluationPipeline().evaluate()
