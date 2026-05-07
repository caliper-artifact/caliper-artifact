

import argparse
import logging
import gc
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import ijson # For memory-efficient JSON parsing

try:
    from scipy import stats
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    ADVANCED_LIBS_AVAILABLE = True
except ImportError:
    ADVANCED_LIBS_AVAILABLE = False

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

METRIC_NAMES = [
    "Task Fulfilment / Relevance", "Usefulness & Actionability", "Factual Accuracy & Verifiabiliy",
    "Efficiency / Depth & Completeness", "Reasoning Quality / Transparency", "Tone & Likeability",
    "Adaptation to Context", "Safety & Bias Avoidance", "Structure & Formatting & UX Extras", "Creativity"
]
METRIC_COLS_SHORT = [f"m{i}" for i in range(1, 11)]
METRIC_MAP = {old: new for old, new in zip(METRIC_NAMES, METRIC_COLS_SHORT)}
METRIC_MAP_REVERSE = {v: k for k, v in METRIC_MAP.items()}


def setup_logging(log_dir: Path):
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analysis_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")

def stream_and_parse_main_data(data_path: Path):
    """Generator to stream a large JSON and yield processed records."""
    logging.info(f"Streaming main data from: {data_path}")
    dataset_name = data_path.stem.split('_')[1]
    model_name = data_path.stem.split('_')[-1]

    with open(data_path, 'r', encoding='utf-8') as f:
        parser = ijson.items(f, 'item')
        for prompt_obj in parser:
            for para_obj in prompt_obj.get("paraphrases", []):
                record = {
                    "prompt_count": prompt_obj["prompt_count"],
                    "dataset": dataset_name,
                    "model": f"{model_name}_baseline",
                    "instruct_type": para_obj["instruct_type"],
                    METRIC_COLS_SHORT[0]: para_obj.get("task_score"),
                    "p_content_score": para_obj.get("paraphrase_content_score"),
                    "bucket": para_obj.get("bucket"),
                    "tags": tuple(para_obj.get("tags", [])),
                    "perplexity": para_obj.get("perplexity"),
                    "p_len": len(para_obj.get("paraphrase", "").split())
                }
                scores = para_obj.get("answer_scores", [np.nan] * 10)
                for i, score in enumerate(scores):
                    record[METRIC_COLS_SHORT[i]] = score
                yield record

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and convert strings to categories to save memory."""
    logging.info("Optimizing DataFrame data types...")
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
        elif 'm' in col or 'bucket' in col or 'p_content_score' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')

    mem_usage = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info(f"Optimized DataFrame memory usage: {mem_usage:.2f} MB")
    return df

def load_and_prepare_data(args: argparse.Namespace) -> pd.DataFrame:
    """Loads all data sources using streaming and merges them."""
    main_files = list(Path(args.data_dir).glob("*.json"))
    if not main_files:
        raise FileNotFoundError(f"No main data files found in {args.data_dir}")

    all_records = (record for f in main_files for record in stream_and_parse_main_data(f))
    baseline_df = pd.DataFrame(all_records)

    ft_files = list(Path(args.ft_dir).glob("**/*buckets*.json")) if args.ft_dir else []
    if not ft_files:
        logging.warning("No fine-tuning data found or directory not provided. Continuing with baseline data only.")
        return optimize_dtypes(baseline_df)

    ft_df_list = [pd.read_json(f) for f in ft_files] # Placeholder for efficient FT loading
    ft_df = pd.concat(ft_df_list, ignore_index=True)

    metadata_cols = [
        "prompt_count", "instruct_type", "dataset", "p_content_score",
        "bucket", "tags", "perplexity", "p_len"
    ]
    merged_df = pd.merge(
        ft_df,
        baseline_df[metadata_cols].drop_duplicates(subset=["prompt_count", "instruct_type"]),
        on=["prompt_count", "instruct_type"],
        how="left"
    )

    full_df = pd.concat([baseline_df, merged_df], ignore_index=True)

    full_df = optimize_dtypes(full_df)

    logging.info(f"Data loading complete. Final DataFrame shape: {full_df.shape}")
    logging.info(f"Models found: {full_df['model'].unique().tolist()}")
    gc.collect() # Force garbage collection
    return full_df


class Analyzer:
    def __init__(self, df: pd.DataFrame, output_dir: Path, args: argparse.Namespace):
        self.df = df
        self.output_dir = output_dir
        self.args = args
        self.results_log = []
        self.graphics_log = {}
        (self.output_dir / "plots").mkdir(exist_ok=True)

    def run_all(self):
        """Runs all selected analysis modules."""
        logging.info("Starting analysis modules...")
        if self.args.do_descriptive: self.run_descriptive_stats()
        if self.args.do_perplexity: self.run_perplexity_analysis()
        if self.args.do_advanced and ADVANCED_LIBS_AVAILABLE: self.run_advanced_stats()
        self.save_summary_report()

    def _add_result(self, title, content):
        self.results_log.append(f"\n{'='*80}\n## {title}\n{'='*80}\n\n{content}\n")

    def _save_plot(self, fig, name, description):
        filepath = self.output_dir / "plots" / f"{name}.png"
        fig.tight_layout()
        fig.savefig(filepath, dpi=120) # Lower DPI for smaller files
        plt.close(fig)
        self.graphics_log[f"plots/{name}.png"] = description
        logging.info(f"Saved plot: {filepath}")

    def run_descriptive_stats(self):
        logging.info("Running Descriptive Statistics...")
        df_high_eq = self.df[self.df['p_content_score'].isin([4, 5])].copy()
        if df_high_eq.empty: return

        title = "Descriptive Stats (Equivalence Score 4-5)"
        content = "TF stands for 'Task Fulfilment / Relevance'.\n"

        tf_col = METRIC_COLS_SHORT[0]
        content += "### Overall TF Score:\n" + df_high_eq[tf_col].describe().to_string() + "\n\n"
        content += "### TF Score by Model:\n" + df_high_eq.groupby('model')[tf_col].describe().to_string() + "\n\n"
        content += "### TF Score by Dataset:\n" + df_high_eq.groupby('dataset')[tf_col].describe().to_string() + "\n\n"

        df_tags_small = df_high_eq[['tags', tf_col]].explode('tags')
        content += "### TF Score by Tag:\n" + df_tags_small.groupby('tags')[tf_col].describe().sort_values('mean', ascending=False).to_string() + "\n\n"
        self._add_result(title, content)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=df_high_eq, x=tf_col, y='model', ax=ax)
        ax.set_title("TF Score Distribution by Model (Equiv. 4-5)")
        ax.set_xlabel(METRIC_MAP_REVERSE[tf_col])
        self._save_plot(fig, "tf_boxplot_by_model", "Box plots of Task Fulfilment scores for each model.")
        del df_tags_small, df_high_eq
        gc.collect()

    def run_perplexity_analysis(self):
        logging.info("Running Perplexity Analysis...")
        df_ppl = self.df.dropna(subset=['perplexity', METRIC_COLS_SHORT[0]]).copy()

        corr_cols = ['perplexity', METRIC_COLS_SHORT[0], 'p_content_score', 'p_len']
        corr_matrix = df_ppl[corr_cols].corr(method='spearman')
        self._add_result("Perplexity Spearman Correlations", corr_matrix.to_string())

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='vlag', center=0, ax=ax)
        ax.set_title('Spearman Correlation Heatmap with Perplexity')
        self._save_plot(fig, "ppl_correlation_heatmap", "Heatmap of Spearman correlations.")

        fig, ax = plt.subplots(figsize=(12, 7))
        sample_df = df_ppl.sample(n=min(5000, len(df_ppl)), random_state=42)
        sns.scatterplot(data=sample_df, x='perplexity', y=METRIC_COLS_SHORT[0], hue='bucket', palette='viridis', alpha=0.6, ax=ax)
        ax.set_title('Perplexity vs. Task Fulfilment (Colored by Bucket) [Sampled]')
        ax.set_xlabel('Perplexity')
        ax.set_ylabel(METRIC_MAP_REVERSE[METRIC_COLS_SHORT[0]])
        ax.set_xscale('log')
        self._save_plot(fig, "ppl_vs_tf_scatter", "Scatter plot of Perplexity vs. TF score.")
        del df_ppl, sample_df
        gc.collect()

    def run_advanced_stats(self):
        logging.info("Running Advanced Statistics...")

        tf_col = METRIC_COLS_SHORT[0]
        model = ols(f'{tf_col} ~ C(bucket)', data=self.df.dropna(subset=[tf_col, 'bucket'])).fit()
        self._add_result("ANOVA: Task Fulfilment vs. Bucket", str(model.summary()))

        df_pca = self.df.dropna(subset=METRIC_COLS_SHORT).copy()
        if len(df_pca) > 20000:
            df_pca = df_pca.sample(n=20000, random_state=42)

        X_scaled = StandardScaler().fit_transform(df_pca[METRIC_COLS_SHORT])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_pca[['PC1', 'PC2']] = X_pca

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='bucket', palette='Set2', alpha=0.7, ax=ax)
        ax.set_title(f'PCA of 10 Performance Metrics [Sampled n={len(df_pca)}]')
        ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        self._save_plot(fig, "pca_of_metrics", "2D PCA plot of all 10 performance metrics.")
        del df_pca, X_scaled, X_pca
        gc.collect()

    def save_summary_report(self):
        report_path = self.output_dir / "results_summary.txt"
        logging.info(f"Saving summary report to: {report_path}")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            for result in self.results_log: f.write(result)
            f.write(f"\n{'='*80}\n## Generated Graphics\n{'='*80}\n\n")
            for filename, description in self.graphics_log.items():
                f.write(f"- **File:** `{filename}`\n  **Description:** {description}\n\n")

def main():
    parser = argparse.ArgumentParser(description="Memory-efficient analysis script for LLM prompt robustness.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with main data JSONs.")
    parser.add_argument("--ft_dir", type=str, default=None, help="Optional root directory for FT outputs.")
    parser.add_argument("--output_dir", type=str, default="e_eval/output", help="Directory for results.")
    parser.add_argument("--run_all", action='store_true', help="Run all modules.")
    parser.add_argument("--do_descriptive", action='store_true')
    parser.add_argument("--do_perplexity", action='store_true')
    parser.add_argument("--do_advanced", action='store_true')
    args = parser.parse_args()

    if args.run_all:
        args.do_descriptive, args.do_perplexity, args.do_advanced = True, True, True

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path / "logs")

    try:
        full_df = load_and_prepare_data(args)
        analyzer = Analyzer(full_df, output_path, args)
        analyzer.run_all()
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)

    logging.info("Analysis script finished.")

if __name__ == "__main__":
    main()