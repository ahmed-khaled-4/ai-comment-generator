import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_metrics(results_file="results/metrics.csv", output_dir="results/plots"):
    """
    Reads the evaluation metrics CSV and generates visualization plots.
    """
    # 1. Setup
    if not os.path.exists(results_file):
        print(f"Error: File '{results_file}' not found. Run experiments first.")
        return

    print(f"Loading data from {results_file}...")
    df = pd.read_csv(results_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.dpi': 150}) # Higher resolution

    # ---------------------------------------------------------
    # PLOT 1: Score Distribution (Histogram)
    # ---------------------------------------------------------
    print("Generating Score Distribution plot...")
    plt.figure(figsize=(10, 6))
    
    metrics_to_plot = ['bleu', 'rougeL', 'bertscore_f1']
    labels = ['BLEU', 'ROUGE-L', 'BERTScore-F1']
    colors = ['#3498db', '#e74c3c', '#2ecc71'] # Blue, Red, Green

    for metric, label, color in zip(metrics_to_plot, labels, colors):
        if metric in df.columns:
            sns.kdeplot(df[metric], label=label, fill=True, alpha=0.2, color=color, linewidth=2)
            # Also plot rugplot for individual data points
            sns.rugplot(df[metric], color=color, alpha=0.5, height=0.1)

    plt.title('Distribution of Evaluation Scores', fontsize=14, pad=15)
    plt.xlabel('Score (0.0 to 1.0)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_distribution.png")
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Box Plot Comparison
    # ---------------------------------------------------------
    print("Generating Metric Comparison plot...")
    plt.figure(figsize=(10, 6))
    
    # Melt dataframe for seaborn boxplot
    plot_metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
    readable_labels = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']
    
    # Filter columns that actually exist
    existing_metrics = [m for m in plot_metrics if m in df.columns]
    
    if existing_metrics:
        df_melted = df[existing_metrics].melt(var_name='Metric', value_name='Score')
        
        # Create Box Plot
        sns.boxplot(x='Metric', y='Score', data=df_melted, palette="viridis", width=0.5)
        
        # Add swarmplot to show individual points on top of boxplot
        sns.stripplot(x='Metric', y='Score', data=df_melted, 
                      color='black', alpha=0.3, size=3, jitter=True)

        plt.title('Comparison of Metric Ranges', fontsize=14, pad=15)
        plt.ylim(0, 1.05) # Fix y-axis to 0-1
        plt.ylabel('Score', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metric_comparison.png")
    
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: Performance vs Code Length
    # ---------------------------------------------------------
    # We approximate code complexity by the length of the code string
    if 'code' in df.columns and 'bertscore_f1' in df.columns:
        print("Generating Length vs Performance plot...")
        df['code_length'] = df['code'].str.len()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='code_length', y='bertscore_f1', data=df, 
                        color='purple', alpha=0.6, s=60)
        
        # Add trend line
        sns.regplot(x='code_length', y='bertscore_f1', data=df, 
                    scatter=False, color='gray', line_kws={'linestyle': '--'})

        plt.title('Does Code Length Affect Model Performance?', fontsize=14)
        plt.xlabel('Code Length (characters)', fontsize=12)
        plt.ylabel('BERTScore F1', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/length_correlation.png")
        plt.close()

    print(f"\nDone! Plots saved in '{output_dir}/'")

if __name__ == "__main__":
    visualize_metrics()