# src/evaluation/evaluator.py - Complete evaluation pipeline with hallucination detection
import json
from pathlib import Path
from src.models.api_client import APIClient
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.hallucination import HallucinationDetector

class EvaluationPipeline:
    """
    Main evaluation pipeline for AI-powered comment generation.
    
    This class handles:
    1. Loading dataset
    2. Generating comments for each code sample
    3. Computing per-sample metrics
    4. Detecting hallucinations and failure modes
    5. Saving results to multiple output files
    """
    
    def __init__(self, dataset_path="dataset/clean_dataset.json"):
        """
        Initialize the evaluation pipeline.
        
        Args:
            dataset_path: Path to the JSON dataset file
        """
        self.dataset_path = dataset_path
        self.client = APIClient()
        self.metrics_calc = MetricsCalculator()
        self.hallucination_detector = HallucinationDetector()
    
    def load_dataset(self):
        """
        Load dataset from JSON file.
        
        Returns:
            List of dicts containing code samples and human comments
        """
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def evaluate(self):
        """
        Main evaluation pipeline:
        1. Generate comments for all code samples
        2. Compute per-sample metrics
        3. Detect hallucinations
        4. Compute overall metrics
        5. Save results to CSV and JSON files
        
        Returns:
            Dict containing overall metric scores
        """
        data = self.load_dataset()
        references = []  # Human gold standard comments
        predictions = []  # Model-generated comments
        rows = []  # Detailed results for CSV
        
        print("\n" + "="*60)
        print("          COMMENT GENERATION PHASE")
        print("="*60)
        print(f"Dataset: {len(data)} code samples\n")
        
        # Phase 1: Generate all comments
        empty_count = 0
        for idx, item in enumerate(data, start=1):
            lang = item["language"]
            code = item["code"]
            gold = item["human_comment"]
            
            # Generate model output
            try:
                model_comment = self.client.generate_comment(code, lang)
                # Handle empty or whitespace-only comments
                if not model_comment or not model_comment.strip():
                    model_comment = "[NO COMMENT GENERATED]"
                    empty_count += 1
            except Exception as e:
                print(f"[{idx}/{len(data)}] ERROR generating comment: {e}")
                model_comment = "[GENERATION FAILED]"
                empty_count += 1
            
            references.append(gold)
            predictions.append(model_comment)
            
            # Store basic data (metrics will be added later)
            rows.append({
                "id": idx,
                "language": lang,
                "code": code,
                "human_comment": gold,
                "model_comment": model_comment
            })
            
            # Show progress with preview
            preview = model_comment.replace("\n", " ")[:150]
            print(f"[{idx}/{len(data)}] {lang}: {preview}...")
        
        if empty_count > 0:
            print(f"\n Warning: {empty_count} samples had empty/failed generation")
        
        print("          METRICS & HALLUCINATION PHASE")
        
        # Phase 2: Compute per-sample metrics & Detect Hallucinations
        print("Computing per-sample metrics and detecting hallucinations...")
        per_sample_computed = False
        try:
            per_sample_metrics = self.metrics_calc.compute_per_sample_metrics_safe(
                references, predictions
            )
            
            # Add metrics and run hallucination detection for each row
            for row, metrics in zip(rows, per_sample_metrics):
                row.update(metrics)
                
                # --- NEW: Run Hallucination Detector ---
                flags = self.hallucination_detector.analyze(
                    row['code'], 
                    row['model_comment'], 
                    score_metrics=metrics
                )
                # Join flags into a single string for CSV (e.g., "TOO_SHORT;LOW_TEXT_OVERLAP")
                row['hallucination_flags'] = ";".join(flags)
            
            print(" Per-sample metrics & hallucination flags computed successfully\n")
            per_sample_computed = True
            
        except Exception as e:
            print(f" Error computing per-sample metrics: {e}")
            print("Adding default metric values...\n")
            # Add zero metrics so CSV can still be saved
            default_metrics = {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "hallucination_flags": "ERROR_COMPUTING"
            }
            for row in rows:
                row.update(default_metrics)
        
        # Phase 3: Compute overall metrics (for display and summary)
        print("Computing overall metrics...")
        try:
            overall_scores = self.metrics_calc.compute_all(references, predictions)
            print(" Overall metrics computed successfully\n")
        except Exception as e:
            print(f" Error computing overall metrics: {e}\n")
            overall_scores = {}
        
        # Phase 4: Save all results
        print("          SAVING RESULTS")
        
        # Create results directory if it doesn't exist
        Path("results").mkdir(exist_ok=True)
        
        # Save detailed per-sample results to CSV
        try:
            self.metrics_calc.save_to_csv(rows)
        except Exception as e:
            print(f" Error saving CSV: {e}")
            # Try saving without summary stats
            try:
                import pandas as pd
                df = pd.DataFrame(rows)
                df.to_csv("results/metrics.csv", index=False)
                print(" Basic CSV saved (without summary stats)")
            except Exception as e2:
                print(f" Could not save even basic CSV: {e2}")
        
        # Save overall metrics to JSON file
        try:
            self._save_overall_metrics(overall_scores)
        except Exception as e:
            print(f" Error saving overall metrics: {e}")
        
        # Save analysis report
        self._save_analysis_report(rows, overall_scores, empty_count)
        
        # Print final summary
        print("          EVALUATION COMPLETE ")
        print(f"Generated comments for {len(data)} code samples")
        if empty_count > 0:
            print(f"  {empty_count} samples had generation issues")
        print(f"\nResults saved in 'results/' directory:")
        print(f"  • metrics.csv - Detailed per-sample results")
        if per_sample_computed:
            print(f"  • metrics_summary.csv - Summary statistics")
        print(f"  • overall_metrics.json - Average metrics")
        print(f"  • analysis_report.txt - Detailed analysis")
        
        return overall_scores
    
    def _save_overall_metrics(self, scores: dict, output_path="results/overall_metrics.json"):
        """
        Save overall/average metrics to JSON file.
        
        Args:
            scores: Dict containing metric scores
            output_path: Path to save JSON file
        """
        with open(output_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f" Overall metrics saved to: {output_path}")
    
    def _save_analysis_report(self, rows, overall_scores, empty_count):
        """
        Save a text report with analysis and insights.
        
        Args:
            rows: List of all evaluation rows
            overall_scores: Dict of overall metrics
            empty_count: Number of samples with empty generation
        """
        try:
            report_path = "results/analysis_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("AUTOMATED EVALUATION REPORT\n")
                f.write("AI-Powered Comment Generation System\n")
                
                # Overall metrics
                f.write("OVERALL METRICS\n")
                for metric, value in overall_scores.items():
                    f.write(f"{metric:20s}: {value:.4f}\n")
                f.write("\n")
                
                # Dataset statistics
                f.write("DATASET STATISTICS\n")
                f.write(f"Total samples: {len(rows)}\n")
                f.write(f"Empty/failed generations: {empty_count}\n")
                f.write(f"Success rate: {((len(rows)-empty_count)/len(rows)*100):.2f}%\n")
                f.write("\n")
                
                # Language breakdown
                from collections import Counter
                lang_counts = Counter(r["language"] for r in rows)
                f.write("LANGUAGE DISTRIBUTION\n")
                for lang, count in lang_counts.items():
                    f.write(f"{lang}: {count} samples\n")
                f.write("\n")
                
                # --- NEW: Hallucination Stats ---
                f.write("HALLUCINATION & FAILURE ANALYSIS\n")
                all_flags = []
                for r in rows:
                    if 'hallucination_flags' in r and r['hallucination_flags']:
                        all_flags.extend(r['hallucination_flags'].split(';'))
                
                flag_counts = Counter(all_flags)
                for flag, count in flag_counts.most_common():
                    f.write(f"{flag}: {count} occurrences\n")
                f.write("\n")

                # Best and worst samples (if metrics available)
                if "bleu" in rows[0]:
                    sorted_by_bleu = sorted(rows, key=lambda x: x.get("bleu", 0), reverse=True)
                    
                    f.write("TOP 5 SAMPLES (by BLEU)\n")
                    for i, row in enumerate(sorted_by_bleu[:5], 1):
                        f.write(f"{i}. ID={row['id']}, BLEU={row.get('bleu', 0):.4f}, "
                               f"ROUGE-1={row.get('rouge1', 0):.4f}, "
                               f"BERTScore-F1={row.get('bertscore_f1', 0):.4f}\n")
                    f.write("\n")
                    
                    f.write("BOTTOM 5 SAMPLES (by BLEU)\n")
                    for i, row in enumerate(sorted_by_bleu[-5:], 1):
                        flags = row.get('hallucination_flags', 'N/A')
                        f.write(f"{i}. ID={row['id']}, BLEU={row.get('bleu', 0):.4f}, "
                               f"Flags={flags}\n")
                    f.write("\n")
                
                f.write("End of Report\n")
            
            print(f" Analysis report saved to: {report_path}")
        except Exception as e:
            print(f" Could not save analysis report: {e}")

if __name__ == "__main__":
    """
    Run the evaluation pipeline when executed as main script.
    Usage: python -m src.evaluation.evaluator
    """
    try:
        pipeline = EvaluationPipeline()
        results = pipeline.evaluate()
        
        # Print final summary of metrics
        if results:
            print("\n Final Metrics Summary:")
            for metric, value in results.items():
                print(f"   {metric:20s}: {value:.4f}")
        
    except KeyboardInterrupt:
        print("\n\n Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n Fatal error: {e}")
        import traceback
        traceback.print_exc()