# metrics.py - Complete metrics calculation for comment generation evaluation
# Implements: BLEU, ROUGE, BERTScore with robust error handling

import evaluate
import pandas as pd
from typing import List, Dict
import warnings

class MetricsCalculator:
    """
    Calculator for NLP evaluation metrics: BLEU, ROUGE, and BERTScore.
    Handles empty comments and provides both per-sample and aggregate metrics.
    """
    
    def __init__(self):
        """Initialize all metric calculators."""
        print("Loading metrics...")
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        
        # Load BERTScore with small model
        self.bertscore = self._load_bertscore()
        print(" All metrics loaded successfully.\n")
    
    def _load_bertscore(self):
        """
        Load BERTScore with a small model that fits in limited disk space.
        
        Returns:
            Loaded BERTScore metric evaluator
        
        Raises:
            RuntimeError: If no model can be loaded due to disk space
        """
        # Try models in order from smallest to largest
        models_to_try = [
            "distilbert-base-uncased",     
            "bert-base-uncased",            
            "microsoft/deberta-v3-xsmall",   
        ]
        
        for model_name in models_to_try:
            try:
                print(f"  Attempting BERTScore with model: {model_name}...")
            
                bertscore = evaluate.load("bertscore")
            
                test_result = bertscore.compute(
                    predictions=["test"],
                    references=["test"],
                    model_type=model_name, 
                    device="cpu",
                    lang="en"
                )
                
                print(f"  Successfully loaded BERTScore with {model_name}\n")
                
                # Store the model name for later use
                self.bertscore_model = model_name
                return bertscore
                
            except Exception as e:
                error_msg = str(e)[:200]
                print(f"  Failed with {model_name}: {error_msg}...")
                continue
        
        # If all models fail, raise informative error
        raise RuntimeError(
            "\n ERROR: Could not load any BERTScore model!\n"
            "SOLUTIONS:\n"
            "1. Free up at least 500MB disk space on C: drive\n"
            "2. Run: pip cache purge\n"
            "3. Empty your Recycle Bin\n"
            "4. Delete temp files from %TEMP% folder\n"
        )
    
    def compute_bleu(self, references: List[str], predictions: List[str]) -> Dict:
        """
        Compute BLEU score for a set of predictions.
        
        Args:
            references: List of reference (gold standard) texts
            predictions: List of predicted (generated) texts
        
        Returns:
            Dict containing BLEU score
        """
        return self.bleu.compute(predictions=predictions, references=references)
    
    def compute_rouge(self, references: List[str], predictions: List[str]) -> Dict:
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Args:
            references: List of reference (gold standard) texts
            predictions: List of predicted (generated) texts
        
        Returns:
            Dict containing rouge1, rouge2, rougeL scores
        """
        return self.rouge.compute(predictions=predictions, references=references)
    
    def compute_bertscore(self, references: List[str], predictions: List[str]) -> Dict:
        """
        Compute BERTScore (Precision, Recall, F1).
        
        Args:
            references: List of reference (gold standard) texts
            predictions: List of predicted (generated) texts
        
        Returns:
            Dict containing lists of precision, recall, f1 scores
        """
        # Use the model that was successfully loaded
        return self.bertscore.compute(
            predictions=predictions,
            references=references,
            model_type=self.bertscore_model,  # ← Use stored model name
            device="cpu",
            lang="en"
        )
    
    def compute_all(self, references: List[str], predictions: List[str]) -> Dict:
        """
        Compute all metrics (BLEU, ROUGE, BERTScore) and return aggregated scores.
        
        Args:
            references: List of gold standard (human) comments
            predictions: List of model-generated comments
        
        Returns:
            Dict containing all metric scores (averaged for BERTScore)
        """
        print("Computing overall metrics...")
        
        # Compute individual metrics
        bleu = self.compute_bleu(references, predictions)
        rouge = self.compute_rouge(references, predictions)
        bert = self.compute_bertscore(references, predictions)
        
        # Average BERTScore metrics across all samples
        bert_precision = sum(bert["precision"]) / len(bert["precision"])
        bert_recall = sum(bert["recall"]) / len(bert["recall"])
        bert_f1 = sum(bert["f1"]) / len(bert["f1"])
        
        # Print summary
        print("           EVALUATION RESULTS")
        print(f"BLEU Score:        {bleu['bleu']:.4f}")
        print(f"ROUGE-1:           {rouge['rouge1']:.4f}")
        print(f"ROUGE-2:           {rouge['rouge2']:.4f}")
        print(f"ROUGE-L:           {rouge['rougeL']:.4f}")
        print(f"BERTScore-P:       {bert_precision:.4f}")
        print(f"BERTScore-R:       {bert_recall:.4f}")
        print(f"BERTScore-F1:      {bert_f1:.4f}")
        print(f"Model used:        {self.bertscore_model}")
        
        return {
            "bleu": bleu["bleu"],
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "bertscore_precision": bert_precision,
            "bertscore_recall": bert_recall,
            "bertscore_f1": bert_f1,
        }
    
    def compute_per_sample_metrics(self, references: List[str], predictions: List[str]) -> List[Dict]:
        """
        Compute metrics for each individual sample.
        
        Args:
            references: List of gold standard comments
            predictions: List of model-generated comments
        
        Returns:
            List of dicts, one per sample, containing all metrics
        
        Note:
            This method may fail with division by zero if predictions contain empty strings.
            Use compute_per_sample_metrics_safe() for robust handling.
        """
        print(f"  Computing BERTScore for {len(references)} samples...")
        bert = self.compute_bertscore(references, predictions)
        
        print(f"  Computing BLEU and ROUGE for each sample...")
        per_sample = []
        for i, (ref, pred) in enumerate(zip(references, predictions)):
            # BLEU for single sample
            bleu_score = self.bleu.compute(predictions=[pred], references=[ref])
            
            # ROUGE for single sample
            rouge_scores = self.rouge.compute(predictions=[pred], references=[ref])
            
            per_sample.append({
                "bleu": bleu_score["bleu"],
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "bertscore_precision": float(bert["precision"][i]),
                "bertscore_recall": float(bert["recall"][i]),
                "bertscore_f1": float(bert["f1"][i]),
            })
            
            # Progress indicator
            if (i + 1) % 25 == 0:
                print(f"    Processed {i + 1}/{len(references)} samples...")
        
        print(f"  Computed metrics for all {len(references)} samples\n")
        return per_sample
    
    def compute_per_sample_metrics_safe(self, references: List[str], predictions: List[str]) -> List[Dict]:
        """
        Compute metrics for each individual sample with error handling for empty comments.
        
        This is the SAFE version that handles:
        - Empty predictions
        - Division by zero errors
        - Placeholder text like "[NO COMMENT GENERATED]"
        
        Args:
            references: List of gold standard comments
            predictions: List of model-generated comments
        
        Returns:
            List of dicts, one per sample, containing all metrics (0.0 for failed samples)
        """
        print(f"  Computing BERTScore for {len(references)} samples...")
        bert = self.compute_bertscore(references, predictions)
        
        print(f"  Computing BLEU and ROUGE for each sample...")
        per_sample = []
        errors = 0
        
        for i, (ref, pred) in enumerate(zip(references, predictions)):
            # Handle empty or placeholder predictions
            if not pred or not pred.strip() or pred in ["[NO COMMENT GENERATED]", "[GENERATION FAILED]"]:
                per_sample.append({
                    "bleu": 0.0,
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "bertscore_precision": float(bert["precision"][i]),
                    "bertscore_recall": float(bert["recall"][i]),
                    "bertscore_f1": float(bert["f1"][i]),
                })
                errors += 1
                continue
            
            try:
                # BLEU for single sample
                bleu_score = self.bleu.compute(predictions=[pred], references=[ref])
                
                # ROUGE for single sample
                rouge_scores = self.rouge.compute(predictions=[pred], references=[ref])
                
                per_sample.append({
                    "bleu": bleu_score["bleu"],
                    "rouge1": rouge_scores["rouge1"],
                    "rouge2": rouge_scores["rouge2"],
                    "rougeL": rouge_scores["rougeL"],
                    "bertscore_precision": float(bert["precision"][i]),
                    "bertscore_recall": float(bert["recall"][i]),
                    "bertscore_f1": float(bert["f1"][i]),
                })
            except (ZeroDivisionError, ValueError) as e:
                # Handle division by zero or other calculation errors
                per_sample.append({
                    "bleu": 0.0,
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "bertscore_precision": float(bert["precision"][i]),
                    "bertscore_recall": float(bert["recall"][i]),
                    "bertscore_f1": float(bert["f1"][i]),
                })
                errors += 1
            
            # Progress indicator
            if (i + 1) % 25 == 0:
                print(f"    Processed {i + 1}/{len(references)} samples...")
        
        if errors > 0:
            print(f"   {errors} samples had metric calculation errors (set to 0.0)")
        print(f"  Computed metrics for all {len(references)} samples\n")
        return per_sample
    
    def save_to_csv(self, results: List[Dict], output_path="results/metrics.csv"):
        """
        Save evaluation results to CSV file with summary statistics.
        
        Args:
            results: List of dicts containing evaluation data
            output_path: Path to save CSV file
        """
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        metric_cols = ["bleu", "rouge1", "rouge2", "rougeL", 
                       "bertscore_precision", "bertscore_recall", "bertscore_f1"]
        other_cols = [c for c in df.columns if c not in metric_cols]
        
        # Only reorder if metric columns exist
        if all(col in df.columns for col in metric_cols):
            df = df[other_cols + metric_cols]
        
            # Round metrics to 4 decimal places
            for col in metric_cols:
                if col in df.columns:
                    df[col] = df[col].round(4)
        
        df.to_csv(output_path, index=False)
        print(f" Detailed results saved to: {output_path}")
        
        # Also save summary statistics if metrics exist
        if all(col in df.columns for col in metric_cols):
            summary_path = output_path.replace(".csv", "_summary.csv")
            self._save_summary_stats(df, summary_path)
    
    def _save_summary_stats(self, df: pd.DataFrame, output_path: str):
        """
        Save summary statistics (mean, std, min, max) for all metrics.
        
        Args:
            df: DataFrame containing evaluation results
            output_path: Path to save summary CSV
        """
        metric_cols = ["bleu", "rouge1", "rouge2", "rougeL", 
                       "bertscore_precision", "bertscore_recall", "bertscore_f1"]
        
        # Only compute stats for columns that exist
        existing_metrics = [col for col in metric_cols if col in df.columns]
        
        if existing_metrics:
            summary = df[existing_metrics].describe().T
            summary = summary[["mean", "std", "min", "max"]]
            summary = summary.round(4)
            summary.to_csv(output_path)
            print(f" Summary statistics saved to: {output_path}")
        else:
            print(f"  No metrics to summarize, skipping {output_path}")