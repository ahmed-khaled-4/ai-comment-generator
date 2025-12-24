import json
import difflib
import requests
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score

class AutomatedEvaluator:
    def __init__(self, api_url="http://localhost:8000/generate_comment"):
        self.api_url = api_url
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def generate_comment(self, code, language="python"):
        """Calls the local FastAPI to get the AI generation."""
        try:
            payload = {
                "code": code,
                "language": language,
                "comment_type": "function", # Defaulting to function for now
                "temperature": 0.4,
                "max_tokens": 600
            }
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json().get("comment", "")
        except Exception as e:
            print(f"Error generating comment: {e}")
            return ""

    def calculate_metrics(self, reference, candidate):
        """Computes BLEU and ROUGE for a single pair."""
        metrics = {}
        
        # BLEU Score
        smooth = SmoothingFunction().method1
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        metrics['bleu'] = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth)

        # ROUGE Score
        rouge_scores = self.rouge_scorer.score(reference, candidate)
        metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
        metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
        metrics['rougeL'] = rouge_scores['rougeL'].fmeasure

        return metrics

    def generate_diff(self, reference, candidate):
        """Creates a readable text diff between human and AI comments."""
        diff = difflib.ndiff(reference.splitlines(), candidate.splitlines())
        return '\n'.join(diff)

    def evaluate_dataset(self, dataset_path, output_path="evaluation_results.json"):
        """
        1. Loads dataset
        2. Generates AI comments (if missing)
        3. Computes metrics
        4. Saves detailed report
        """
        print(f"Loading dataset from {dataset_path}...")
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        results = []
        references = []
        candidates = []

        print(f"Processing {len(data)} samples...")
        
        for i, item in enumerate(data):
            code = item.get('code', '')
            ref = item.get('human_docstring', '')
            
            # Generate AI comment if it doesn't exist in the dataset
            print(f"[{i+1}/{len(data)}] Generating comment...")
            cand = self.generate_comment(code)
            
            references.append(ref)
            candidates.append(cand)

            # Calculate individual metrics
            metrics = self.calculate_metrics(ref, cand)
            diff_view = self.generate_diff(ref, cand)

            results.append({
                "id": item.get('id', i),
                "code_snippet": code[:50] + "...",
                "human_ref": ref,
                "ai_candidate": cand,
                "metrics": metrics,
                "diff": diff_view
            })

        # Batch BERTScore (More efficient to run at once)
        print("Computing BERTScore (this might take a minute)...")
        try:
            P, R, F1 = score(candidates, references, lang="en", verbose=True)
            f1_scores = F1.numpy().tolist()
            
            # Update results with BERTScore
            for i, f1 in enumerate(f1_scores):
                results[i]['metrics']['bert_score_f1'] = f1
        except Exception as e:
            print(f"BERTScore failed (is pytorch installed?): {e}")

        # Summary Statistics
        avg_bleu = np.mean([r['metrics']['bleu'] for r in results])
        avg_rouge = np.mean([r['metrics']['rougeL'] for r in results])
        avg_bert = np.mean([r.get('metrics', {}).get('bert_score_f1', 0) for r in results])

        final_report = {
            "summary": {
                "avg_bleu": avg_bleu,
                "avg_rougeL": avg_rouge,
                "avg_bert_score_f1": avg_bert
            },
            "details": results
        }

        with open(output_path, 'w') as f:
            json.dump(final_report, f, indent=4)
        
        print(f"✅ Evaluation Complete! Report saved to {output_path}")
        print(f"Average BLEU: {avg_bleu:.4f} | ROUGE-L: {avg_rouge:.4f} | BERTScore: {avg_bert:.4f}")