"""
Cross-experiment analyzer module.

This module provides functionality to analyze and compare results from multiple experiment runs.
"""
import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats


class CrossExperimentAnalyzer:
    """Analyzer for comparing results across multiple experiments."""
    
    def __init__(self, experiment_dir: str = "experiment_results", output_dir: str = None):
        """
        Initialize the cross-experiment analyzer.
        
        Args:
            experiment_dir: Base directory containing experiment results
            output_dir: Directory to save analysis results (defaults to experiment_dir/cross_experiment_analysis)
        """
        self.experiment_dir = experiment_dir
        if output_dir is None:
            self.output_dir = os.path.join(experiment_dir, "cross_experiment_analysis")
        else:
            self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
        self.experiment_data = {}
        self.experiment_metrics = {}
        self.analysis_results = {}
    
    def _convert_numpy_types_to_python(self, data: Any) -> Any:
        """Recursively convert NumPy types (like np.bool_) to native Python types for JSON serialization."""
        if isinstance(data, list):
            return [self._convert_numpy_types_to_python(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._convert_numpy_types_to_python(value) for key, value in data.items()}
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, (np.integer, np.int_)):
            return int(data)
        elif isinstance(data, (np.floating, np.float_)):
            return float(data)
        elif pd.isna(data): # Handle Pandas NaT or other NA types if they sneak in
            return None
        return data
    
    def find_experiment_directories(self, experiment_types: Optional[List[str]] = None) -> List[str]:
        """
        Find experiment result directories.
        
        Args:
            experiment_types: Optional list of experiment types to filter by (e.g., ["Scientific_Hypothesis"])
            
        Returns:
            List of experiment directory paths
        """
        # Find all experiment directories in the base directory
        all_dirs = glob.glob(os.path.join(self.experiment_dir, "*"))
        
        # Filter to only include directories that are actual experiment directories
        # by checking for required experiment files
        exp_dirs = []
        for d in all_dirs:
            if os.path.isdir(d):
                # Check if this is a valid experiment directory with required files
                has_metadata = os.path.exists(os.path.join(d, "metadata.json"))
                has_results = os.path.exists(os.path.join(d, "results.json"))
                
                if has_metadata and has_results:
                    exp_dirs.append(d)
                else:
                    # Skip directories like "analysis" and "cross_experiment_analysis"
                    # that aren't actual experiment results
                    if not (d.endswith("cross_experiment_analysis") or 
                           d.endswith("analysis") or
                           d.endswith("integrated_analysis")):
                        print(f"[DEBUG] Skipping directory without experiment files: {d}")
        
        # Filter by experiment type if specified
        if experiment_types:
            filtered_dirs = []
            for exp_dir in exp_dirs:
                metadata_path = os.path.join(exp_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    # Check if experiment name starts with any of the specified types
                    if any(metadata.get("name", "").startswith(exp_type) for exp_type in experiment_types):
                        filtered_dirs.append(exp_dir)
            
            return filtered_dirs
        
        return exp_dirs
    
    def load_experiment_results(self, experiment_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load results from experiment directories.
        
        Args:
            experiment_types: Optional list of experiment types to filter by
            
        Returns:
            Dictionary of loaded experiment data
        """
        # Find experiment directories
        exp_dirs = self.find_experiment_directories(experiment_types)
        
        # Load data from each experiment
        for exp_dir in exp_dirs:
            exp_name = os.path.basename(exp_dir)
            
            # Load metadata
            metadata_path = os.path.join(exp_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                print(f"[WARNING] No metadata.json found in {exp_dir}")
                continue
            
            # Load results.json
            results_path = os.path.join(exp_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    results = json.load(f)
            else:
                print(f"[WARNING] No results.json found in {exp_dir}")
                continue
            
            # Load results.csv
            csv_path = os.path.join(exp_dir, "results.csv")
            if os.path.exists(csv_path):
                csv_data = pd.read_csv(csv_path)
            else:
                print(f"[WARNING] No results.csv found in {exp_dir}")
                csv_data = None
            
            # Store experiment data
            self.experiment_data[exp_name] = {
                "metadata": metadata,
                "results": results,
                "csv_data": csv_data,
                "directory": exp_dir
            }
        
        print(f"[INFO] Loaded data from {len(self.experiment_data)} experiments")
        return self.experiment_data
    
    def extract_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract key metrics from experiment results.
        
        Returns:
            Dictionary of metrics by experiment
        """
        metrics = {}
        
        for exp_name, exp_data in self.experiment_data.items():
            # Get experiment metadata for context
            metadata = exp_data["metadata"]
            experiment_type = metadata.get("type", "unknown")
            model = metadata.get("model", "unknown")
            config = metadata.get("config", {})
            
            # Get summary metrics from results
            results = exp_data["results"]
            summary = results.get("summary", {})
            
            # Extract metrics
            exp_metrics = {
                "name": metadata.get("name", exp_name),
                "type": experiment_type,
                "model": model,
                "run_date": metadata.get("start_time", ""),
                "config": {
                    k: v for k, v in config.items() 
                    if k not in ["start_time", "experiment_name", "experiment_type", "model_name"]
                }
            }
            
            # Extract context similarities
            if "Context Similarities" in summary:
                context_sims = summary["Context Similarities"]
                for metric, stats in context_sims.items():
                    exp_metrics[f"context_{metric}_mean"] = stats.get("mean", 0)
                    exp_metrics[f"context_{metric}_median"] = stats.get("median", 0)
                    exp_metrics[f"context_{metric}_std"] = stats.get("std", 0)
            
            # Extract pairwise similarities
            if "Pairwise Similarities" in summary:
                pairwise_sims = summary["Pairwise Similarities"]
                for metric, stats in pairwise_sims.items():
                    exp_metrics[f"pairwise_{metric}_mean"] = stats.get("mean", 0)
                    exp_metrics[f"pairwise_{metric}_median"] = stats.get("median", 0)
                    exp_metrics[f"pairwise_{metric}_std"] = stats.get("std", 0)
            
            # Add runtime information
            if "Runtime" in summary:
                runtime = summary["Runtime"]
                exp_metrics["runtime_seconds"] = runtime.get("Seconds", 0)
                exp_metrics["ideas_per_minute"] = (
                    summary.get("Total Ideas", 0) / (runtime.get("Seconds", 0) / 60)
                    if runtime.get("Seconds", 0) > 0 else 0
                )
            
            # Extract raw similarity scores from CSV data if available
            if exp_data["csv_data"] is not None:
                csv_data = exp_data["csv_data"]
                
                # Map CSV column name to internal raw metric key name for pairwise scores
                # The CSV stores lists of pairwise scores under these 'avg_...' names.
                # These are actually the raw lists if `log_result` in ExperimentTracker uses these keys for raw lists.
                # Let's ensure consistency: `run_idea_generation_batch` logs e.g. "cosine_similarities" (raw list)
                # and "avg_pairwise_cosine_similarity" (float). `ExperimentTracker` should save both if dtypes are defined.
                # The dtypes now reflect this separation. `extract_metrics` should read the raw list columns.
                csv_to_raw_pairwise_metric_map = {
                    "cosine_similarities": "raw_pairwise_cosine", # Key in CSV : Key in exp_metrics
                    "self_bleu_scores": "raw_pairwise_self_bleu",
                    "bertscore_scores": "raw_pairwise_bertscore"
                }

                for csv_col_name, raw_metric_key in csv_to_raw_pairwise_metric_map.items():
                    if csv_col_name in csv_data.columns:
                        try:
                            raw_scores_str = csv_data[csv_col_name].iloc[0]
                            if pd.isna(raw_scores_str):
                                print(f"[DEBUG] Raw pairwise scores for '{csv_col_name}' in {exp_name} are NaN. Storing as empty list.")
                                raw_scores = []
                            elif isinstance(raw_scores_str, str):
                                raw_scores = eval(raw_scores_str)
                            elif isinstance(raw_scores_str, list): # Already a list (less likely from CSV but handle)
                                raw_scores = raw_scores_str
                            else:
                                print(f"[WARNING] Unexpected type for raw pairwise scores '{csv_col_name}' in {exp_name}: {type(raw_scores_str)}. Storing as empty list.")
                                raw_scores = []
                            
                            if isinstance(raw_scores, list):
                                exp_metrics[raw_metric_key] = raw_scores
                                if not raw_scores:
                                    print(f"[DEBUG] Parsed raw pairwise scores for '{csv_col_name}' in {exp_name} is an empty list.")
                            else:
                                print(f"[WARNING] Failed to parse raw pairwise scores for '{csv_col_name}' in {exp_name} into a list. Storing as empty list.")
                                exp_metrics[raw_metric_key] = []
                        except Exception as e:
                            print(f"[WARNING] Error parsing raw pairwise scores for '{csv_col_name}' in {exp_name}: {e}. Storing as empty list.")
                            exp_metrics[raw_metric_key] = []
                    else:
                        print(f"[DEBUG] Pairwise raw score column '{csv_col_name}' not found in CSV for {exp_name}. Storing empty list for {raw_metric_key}.")
                        exp_metrics[raw_metric_key] = []
                
                # NEW: Extract raw CONTEXT similarity scores from CSV data
                csv_to_raw_context_metric_map = {
                    "context_cosine_scores_raw": "raw_context_cosine",
                    "context_self_bleu_scores_raw": "raw_context_self_bleu",
                    "context_bertscore_scores_raw": "raw_context_bertscore"
                }
                for csv_col_name, raw_metric_key in csv_to_raw_context_metric_map.items():
                    if csv_col_name in csv_data.columns:
                        try:
                            raw_scores_str = csv_data[csv_col_name].iloc[0]
                            if pd.isna(raw_scores_str):
                                print(f"[DEBUG] Raw context scores for '{csv_col_name}' in {exp_name} are NaN. Storing as empty list.")
                                raw_scores = []
                            elif isinstance(raw_scores_str, str):
                                raw_scores = eval(raw_scores_str)
                            elif isinstance(raw_scores_str, list):
                                raw_scores = raw_scores_str
                            else:
                                print(f"[WARNING] Unexpected type for raw context scores '{csv_col_name}' in {exp_name}: {type(raw_scores_str)}. Storing as empty list.")
                                raw_scores = []
                            
                            if isinstance(raw_scores, list):
                                exp_metrics[raw_metric_key] = raw_scores
                                if not raw_scores:
                                    print(f"[DEBUG] Parsed raw context scores for '{csv_col_name}' in {exp_name} is an empty list.")
                            else:
                                print(f"[WARNING] Failed to parse raw context scores for '{csv_col_name}' in {exp_name} into a list. Storing as empty list.")
                                exp_metrics[raw_metric_key] = []
                        except Exception as e:
                            print(f"[WARNING] Error parsing raw context scores '{csv_col_name}' in {exp_name}: {e}. Storing as empty list.")
                            exp_metrics[raw_metric_key] = []
                    else:
                        print(f"[DEBUG] Context raw score column '{csv_col_name}' not found in CSV for {exp_name}. Storing empty list for {raw_metric_key}.")
                        exp_metrics[raw_metric_key] = []

                # Get KDE data if available
                if "kde_values" in csv_data.columns:
                    try:
                        kde_values = eval(csv_data["kde_values"].iloc[0])
                        exp_metrics["kde_values"] = kde_values
                    except (SyntaxError, ValueError, IndexError) as e:
                        print(f"[WARNING] Could not parse KDE values from CSV: {e}")
            
            metrics[exp_name] = exp_metrics
        
        self.experiment_metrics = metrics
        return metrics
    
    def compare_metrics(self) -> Dict[str, Any]:
        """
        Compare metrics across experiments.
        
        Returns:
            Dictionary of comparative analysis results
        """
        if not self.experiment_metrics:
            self.extract_metrics()
        
        # Create a DataFrame from metrics for easier comparison
        metrics_df = pd.DataFrame.from_dict(self.experiment_metrics, orient="index")
        
        # Define metrics to compare
        context_metrics = [col for col in metrics_df.columns if col.startswith("context_")]
        pairwise_metrics = [col for col in metrics_df.columns if col.startswith("pairwise_")]
        
        # Group experiments by type
        exp_types = metrics_df["type"].unique()
        
        comparison_results = {
            "metrics_df": metrics_df,
            "context_metrics": {},
            "pairwise_metrics": {},
            "runtime_comparison": {},
            "statistical_tests": {}
        }
        
        # Compare context metrics
        for metric in context_metrics:
            base_metric = metric.replace("context_", "").replace("_mean", "").replace("_median", "").replace("_std", "")
            if base_metric not in comparison_results["context_metrics"]:
                comparison_results["context_metrics"][base_metric] = {
                    "mean": {},
                    "median": {},
                    "std": {}
                }
            
            if metric.endswith("_mean"):
                for exp_type in exp_types:
                    type_df = metrics_df[metrics_df["type"] == exp_type]
                    comparison_results["context_metrics"][base_metric]["mean"][exp_type] = type_df[metric].mean()
            elif metric.endswith("_median"):
                for exp_type in exp_types:
                    type_df = metrics_df[metrics_df["type"] == exp_type]
                    comparison_results["context_metrics"][base_metric]["median"][exp_type] = type_df[metric].mean()
            elif metric.endswith("_std"):
                for exp_type in exp_types:
                    type_df = metrics_df[metrics_df["type"] == exp_type]
                    comparison_results["context_metrics"][base_metric]["std"][exp_type] = type_df[metric].mean()
        
        # Compare pairwise metrics
        for metric in pairwise_metrics:
            base_metric = metric.replace("pairwise_", "").replace("_mean", "").replace("_median", "").replace("_std", "")
            if base_metric not in comparison_results["pairwise_metrics"]:
                comparison_results["pairwise_metrics"][base_metric] = {
                    "mean": {},
                    "median": {},
                    "std": {}
                }
            
            if metric.endswith("_mean"):
                for exp_type in exp_types:
                    type_df = metrics_df[metrics_df["type"] == exp_type]
                    comparison_results["pairwise_metrics"][base_metric]["mean"][exp_type] = type_df[metric].mean()
            elif metric.endswith("_median"):
                for exp_type in exp_types:
                    type_df = metrics_df[metrics_df["type"] == exp_type]
                    comparison_results["pairwise_metrics"][base_metric]["median"][exp_type] = type_df[metric].mean()
            elif metric.endswith("_std"):
                for exp_type in exp_types:
                    type_df = metrics_df[metrics_df["type"] == exp_type]
                    comparison_results["pairwise_metrics"][base_metric]["std"][exp_type] = type_df[metric].mean()
        
        # Compare runtime and ideas per minute
        comparison_results["runtime_comparison"] = {
            "runtime_seconds": {},
            "ideas_per_minute": {}
        }
        
        for exp_type in exp_types:
            type_df = metrics_df[metrics_df["type"] == exp_type]
            comparison_results["runtime_comparison"]["runtime_seconds"][exp_type] = type_df["runtime_seconds"].mean()
            comparison_results["runtime_comparison"]["ideas_per_minute"][exp_type] = type_df["ideas_per_minute"].mean()
        
        # Perform statistical tests for raw similarity scores
        raw_metrics = {
            "cosine": "raw_pairwise_cosine",
            "self_bleu": "raw_pairwise_self_bleu",
            "bertscore": "raw_pairwise_bertscore"
        }
        
        for metric_name, metric_key in raw_metrics.items():
            # Check which experiments have raw scores for this metric
            exps_with_data = [exp for exp, data in self.experiment_metrics.items() 
                             if metric_key in data and isinstance(data[metric_key], list)]
            
            if len(exps_with_data) >= 2:
                # Perform pairwise tests
                test_results = {}
                
                for i in range(len(exps_with_data)):
                    for j in range(i+1, len(exps_with_data)):
                        exp1 = exps_with_data[i]
                        exp2 = exps_with_data[j]
                        
                        # Get the raw scores
                        scores1 = self.experiment_metrics[exp1][metric_key]
                        scores2 = self.experiment_metrics[exp2][metric_key]
                        
                        # Perform Mann-Whitney U test
                        try:
                            mw_stat, mw_p = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                            
                            # Perform t-test
                            t_stat, t_p = stats.ttest_ind(scores1, scores2, equal_var=False)
                            
                            # Calculate effect size (Cohen's d)
                            mean1, std1 = np.mean(scores1), np.std(scores1)
                            mean2, std2 = np.mean(scores2), np.std(scores2)
                            
                            pooled_std = np.sqrt(((len(scores1) - 1) * std1**2 + 
                                                 (len(scores2) - 1) * std2**2) / 
                                                (len(scores1) + len(scores2) - 2))
                            
                            cohen_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
                            
                            # Perform 2-sample Kolmogorov-Smirnov test
                            ks_stat, ks_p = stats.ks_2samp(scores1, scores2)
                            
                            # Store results
                            test_results[f"{exp1} vs {exp2}"] = {
                                "mann_whitney": {
                                    "statistic": float(mw_stat),
                                    "p_value": float(mw_p),
                                    "significant": mw_p < 0.05
                                },
                                "t_test": {
                                    "statistic": float(t_stat),
                                    "p_value": float(t_p),
                                    "significant": t_p < 0.05
                                },
                                "ks_test": {
                                    "statistic": float(ks_stat),
                                    "p_value": float(ks_p),
                                    "significant": ks_p < 0.05
                                },
                                "effect_size": {
                                    "cohen_d": float(cohen_d),
                                    "interpretation": self._interpret_cohens_d(cohen_d)
                                }
                            }
                        except Exception as e:
                            print(f"[WARNING] Statistical test failed for {metric_name} ({exp1} vs {exp2}): {e}")
                
                comparison_results["statistical_tests"][metric_name] = test_results
        
        self.analysis_results = comparison_results
        return comparison_results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            d: Cohen's d value
            
        Returns:
            String interpretation of the effect size
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_comparison_plots(self) -> Dict[str, str]:
        """
        Generate comparison plots for key metrics.
        
        Returns:
            Dictionary of plot paths
        """
        if not self.analysis_results:
            self.compare_metrics()
        
        plot_paths = {}
        
        # Create box plots for raw similarity scores - RESTORING THIS BLOCK
        raw_pairwise_metrics_config = {
            "cosine": "raw_pairwise_cosine",
            "self_bleu": "raw_pairwise_self_bleu",
            "bertscore": "raw_pairwise_bertscore"
        }
        
        for metric_name, metric_key in raw_pairwise_metrics_config.items():
            data_for_box = []
            for exp_name, exp_metrics_data in self.experiment_metrics.items():
                if metric_key in exp_metrics_data and isinstance(exp_metrics_data[metric_key], list):
                    # Additional check and debug for box plots
                    raw_scores_list_for_box = exp_metrics_data[metric_key]
                    if not raw_scores_list_for_box:
                        print(f"[DEBUG] Boxplot: No scores for metric '{metric_key}' in experiment '{exp_name}'. Skipping.")
                        continue # Skip if list is empty

                    exp_type = exp_metrics_data["type"]
                    for score in raw_scores_list_for_box:
                        data_for_box.append({
                            "Experiment Type": exp_type,
                            "Score": score,
                            "Metric": metric_name.title()
                        })
                elif metric_key not in exp_metrics_data:
                     print(f"[DEBUG] Boxplot: Metric key '{metric_key}' not found in experiment_metrics for '{exp_name}'. Skipping.")
            
            if data_for_box:
                box_df = pd.DataFrame(data_for_box)
                fig = px.box(
                    box_df, 
                    x="Experiment Type", 
                    y="Score", 
                    color="Experiment Type",
                    title=f"{metric_name.title()} Distribution Comparison",
                    labels={"Score": f"{metric_name.title()} Score"}
                )
                fig.add_trace(
                    go.Scatter(
                        x=box_df["Experiment Type"],
                        y=box_df["Score"],
                        mode="markers",
                        marker=dict(color="rgba(0, 0, 0, 0.3)", size=4),
                        name="Individual Scores"
                    )
                )
                plot_name = f"raw_{metric_name}_boxplot.html"
                plot_path = os.path.join(self.output_dir, "plots", plot_name)
                fig.write_html(plot_path)
                plot_paths[f"raw_{metric_name}_boxplot"] = plot_path
        
        # Create KDE plots for (pairwise) similarity distributions - RESTORING THIS BLOCK
        for metric_name, metric_key in raw_pairwise_metrics_config.items():
            exps_with_data_for_kde = []
            for exp_name, data in self.experiment_metrics.items():
                if metric_key in data and isinstance(data[metric_key], list) and len(data[metric_key]) > 1:
                    exps_with_data_for_kde.append(exp_name)
                elif metric_key not in data:
                    print(f"[DEBUG] Pairwise KDE: Metric key '{metric_key}' not found for experiment '{exp_name}'. Skipping KDE.")
                elif not isinstance(data.get(metric_key), list):
                    print(f"[DEBUG] Pairwise KDE: Data for '{metric_key}' in experiment '{exp_name}' is not a list ({type(data.get(metric_key))}). Skipping KDE.")
                elif len(data.get(metric_key, [])) <= 1:
                    print(f"[DEBUG] Pairwise KDE: Not enough data points ({len(data.get(metric_key, []))}) for '{metric_key}' in '{exp_name}'. Skipping KDE.")
            
            if exps_with_data_for_kde:
                fig = go.Figure()
                for exp_name in exps_with_data_for_kde:
                    exp_metrics_data = self.experiment_metrics[exp_name]
                    raw_scores_list = exp_metrics_data[metric_key]
                    if len(raw_scores_list) > 1:
                        kde = stats.gaussian_kde(raw_scores_list)
                        x_vals = np.linspace(min(raw_scores_list), max(raw_scores_list), 1000)
                        y_vals = kde(x_vals)
                        legend_label = self.experiment_metrics[exp_name].get("name", exp_name)
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="lines",
                            name=legend_label,
                            line=dict(width=2)
                        ))
                fig.update_layout(
                    title=f"{metric_name.title()} Density Distribution Comparison",
                    xaxis_title=f"{metric_name.title()} Score",
                    yaxis_title="Density",
                    legend_title="Prompt Strategy",
                    legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"),
                    margin=dict(t=60, r=40, b=100, l=40)
                )
                plot_name = f"{metric_name}_kde_comparison.html"
                plot_path = os.path.join(self.output_dir, "plots", plot_name)
                fig.write_html(plot_path)
                plot_paths[f"{metric_name}_kde"] = plot_path
        
        # NEW: Create KDE plots for RAW CONTEXT similarity distributions
        raw_context_metrics_config = {
            "context_cosine": "raw_context_cosine",
            "context_self_bleu": "raw_context_self_bleu",
            "context_bertscore": "raw_context_bertscore"
        }

        for metric_name_display, metric_key_data in raw_context_metrics_config.items():
            exps_with_data_for_context_kde = []
            for exp_name, data in self.experiment_metrics.items():
                if metric_key_data in data and isinstance(data[metric_key_data], list) and len(data[metric_key_data]) > 1:
                    exps_with_data_for_context_kde.append(exp_name)
                elif metric_key_data not in data:
                    print(f"[DEBUG] Context KDE: Metric key '{metric_key_data}' not found for experiment '{exp_name}'. Skipping KDE.")
                elif not isinstance(data.get(metric_key_data), list):
                    print(f"[DEBUG] Context KDE: Data for '{metric_key_data}' in experiment '{exp_name}' is not a list ({type(data.get(metric_key_data))}). Skipping KDE.")
                elif len(data.get(metric_key_data, [])) <= 1:
                    print(f"[DEBUG] Context KDE: Not enough data points ({len(data.get(metric_key_data, []))}) for '{metric_key_data}' in '{exp_name}'. Skipping KDE.")

            if exps_with_data_for_context_kde:
                fig = go.Figure()
                for exp_name in exps_with_data_for_context_kde:
                    exp_metrics_data = self.experiment_metrics[exp_name]
                    raw_scores_list = exp_metrics_data[metric_key_data]
                    if len(raw_scores_list) > 1:
                        kde = stats.gaussian_kde(raw_scores_list)
                        x_vals = np.linspace(min(raw_scores_list), max(raw_scores_list), 1000)
                        y_vals = kde(x_vals)
                        legend_label = self.experiment_metrics[exp_name].get("name", exp_name)
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="lines",
                            name=legend_label,
                            line=dict(width=2)
                        ))
                fig.update_layout(
                    title=f"{metric_name_display.replace('_', ' ').title()} Density (Output vs. Input Context)",
                    xaxis_title=f"{metric_name_display.replace('_', ' ').title()} Score",
                    yaxis_title="Density",
                    legend_title="Prompt Strategy",
                    legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"),
                    margin=dict(t=60, r=40, b=100, l=40)
                )
                plot_name = f"{metric_name_display}_context_kde_comparison.html"
                plot_path = os.path.join(self.output_dir, "plots", plot_name)
                fig.write_html(plot_path)
                plot_paths[f"{metric_name_display}_context_kde"] = plot_path
        
        return plot_paths
    
    def generate_comparison_dashboard(self) -> str:
        """
        Generate a comprehensive dashboard for cross-experiment comparison.
        
        Returns:
            Path to the generated dashboard HTML file
        """
        if not self.analysis_results:
            self.compare_metrics()
        
        # Generate plots (performance plot generation is now removed)
        plot_paths = self.generate_comparison_plots()
        
        # Create the dashboard HTML
        dashboard_path = os.path.join(self.output_dir, "comparison_dashboard.html")
        
        # HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cross-Experiment Comparison Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .section-description {{
                    font-size: 0.9em;
                    color: #555;
                    margin-bottom: 15px;
                }}
                .summary, .experiment-details-section {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .plot-container {{
                    margin: 20px 0;
                    padding: 15px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .plot-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .stat-table th, .stat-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .stat-table th {{
                    background-color: #f2f2f2;
                }}
                .stat-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .iframe-container {{
                    height: 500px; /* Adjust as needed */
                    width: 100%;
                    overflow: auto; /* Allow scroll if content overflows */
                }}
                .iframe-container iframe {{
                    width: 100%;
                    height: 100%;
                    border: none;
                }}
                /* Removed tab-specific CSS */
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Cross-Experiment Comparison Dashboard</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p class="section-description">This dashboard provides a high-level overview and comparison of results across different experiment types.</p>
                    <p>Total experiments analyzed: {len(self.experiment_data)}</p>
                    <p>Experiment types: {", ".join(sorted(set(exp["type"] for exp in self.experiment_metrics.values())))}</p>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <!-- Experiment Details Section (Moved to top) -->
                <div class="experiment-details-section">
                    <h2>Experiment Details</h2>
                    <p class="section-description">This table summarizes key parameters and performance metrics for each individual experiment run included in this analysis.</p>
                    <table class="stat-table">
                        <tr>
                            <th>Experiment</th>
                            <th>Type</th>
                            <th>Model</th>
                            <th>Run Date</th>
                            <th>Ideas</th>
                            <th>Runtime (s)</th>
                            <th>Ideas/Minute</th>
                        </tr>
        """
        
        for exp_name, exp_metrics in self.experiment_metrics.items():
            run_date = exp_metrics.get("run_date", "N/A")
            if isinstance(run_date, str) and run_date:
                try:
                    date_obj = datetime.fromisoformat(run_date.replace("Z", "+00:00"))
                    run_date = date_obj.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    pass # Keep original string if parsing fails
            
            html_content += f"""
                <tr>
                    <td>{exp_metrics.get("name", exp_name)}</td>
                    <td>{exp_metrics.get("type", "N/A")}</td>
                    <td>{exp_metrics.get("model", "N/A")}</td>
                    <td>{run_date}</td>
                    <td>{self.experiment_data[exp_name]["results"].get("summary", {}).get("Total Ideas", "N/A")}</td>
                    <td>{exp_metrics.get("runtime_seconds", 0):.1f}</td>
                    <td>{exp_metrics.get("ideas_per_minute", 0):.1f}</td>
                </tr>
            """
        html_content += """
                    </table>
                </div>
                
                <!-- Performance Metrics Section (Removed) -->
                
        """
        # STATISTICAL TESTS SECTION STARTS HERE (MOVED)
        html_content += """
                <h2>Statistical Tests</h2>
                <p class="section-description">This section presents the results of statistical tests (Mann-Whitney U, T-test, Kolmogorov-Smirnov) comparing the distributions of scores for different metrics between pairs of experiment types. It helps determine if observed differences are statistically significant.</p>
        """ # First part of Statistical Tests HTML is now a clean, self-contained string.
        
        # The 'if' block for detailed statistical tests starts here, correctly indented.
        if "statistical_tests" in self.analysis_results and self.analysis_results["statistical_tests"]:
            html_content += '<div style="max-height: 400px; overflow-y: auto;">' # Start scrollable div
            for metric, tests in self.analysis_results["statistical_tests"].items():
                html_content += f"""
                <h3>{metric.title()} Comparison Tests</h3>
                <table class="stat-table">
                    <tr>
                        <th>Comparison</th>
                        <th>Mann-Whitney p-value</th>
                        <th>Significant?</th>
                        <th>t-test p-value</th>
                        <th>Significant?</th>
                        <th>KS Test p-value</th>
                        <th>Significant?</th>
                        <th>Effect Size (Cohen's d)</th>
                        <th>Interpretation</th>
                    </tr>
                """
                
                for comparison, test_results in tests.items():
                    mw_test = test_results["mann_whitney"]
                    t_test = test_results["t_test"]
                    ks_test = test_results.get("ks_test", {"p_value": np.nan, "significant": False}) 
                    effect_size = test_results["effect_size"]
                    
                    html_content += f"""
                    <tr>
                        <td>{comparison}</td>
                        <td>{mw_test['p_value']:.4f}</td>
                        <td>{"Yes" if mw_test['significant'] else "No"}</td>
                        <td>{t_test['p_value']:.4f}</td>
                        <td>{"Yes" if t_test['significant'] else "No"}</td>
                        <td>{ks_test['p_value']:.4f}</td>
                        <td>{"Yes" if ks_test['significant'] else "No"}</td>
                        <td>{effect_size['cohen_d']:.4f}</td>
                        <td>{effect_size['interpretation']}</td>
                    </tr>
                    """
                html_content += "</table>\n"
            html_content += '</div>' # End scrollable div
        else:
            html_content += "<p>No statistical test results available.</p>"
        # STATISTICAL TESTS SECTION ENDS HERE (MOVED)

        # NEW: Conclusion Section - REMOVING THIS AS PER DECISION TO USE STATISTICALANALYZER

        # Context Similarity Metrics Section - REMOVING THIS ENTIRE BLOCK
        # (This comment is from a previous step, the block itself is already removed)

        # Distribution Comparison Section - RENAMING AND KEEPING for Pairwise (Output vs Output)
        html_content += """
                <h2>Pairwise Similarity Distributions (Output vs. Output)</h2>
                <p class="section-description">
                    These plots visualize the distribution of <strong>pairwise similarity scores</strong> (Cosine, Self-BLEU, BERTScore) calculated between all unique pairs of ideas generated <em>within each experiment type</em> (i.e., how similar generated ideas are to each other).
                    This helps assess the internal diversity of ideas produced by each prompting strategy. Each experiment type is aggregated across all its runs.
                    <br><strong>KDE Plots:</strong> Each colored line represents an experiment type (prompting strategy). The X-axis is the similarity score, and the Y-axis shows the density (concentration) of scores. Peaks indicate common similarity values for that strategy.
                    <br><strong>Box Plots:</strong> The X-axis shows the different Experiment Types. Each box plot summarizes the distribution of pairwise similarity scores for all ideas generated by that strategy. It shows the median (central line), interquartile range (the box), whiskers (typically 1.5x IQR), and any outliers (individual points). This allows for comparing the central tendency and spread of pairwise scores between strategies.
                </p>
        """ # End of Pairwise Distribution Comparison intro

        distribution_metrics = ["cosine", "self_bleu", "bertscore"] # These are keys for pairwise metrics in plot_paths
        for metric in distribution_metrics:
            html_content += f"<h3>{metric.title()} Pairwise Score Distributions</h3>\n"
            kde_key = f"{metric}_kde" # Key for pairwise KDE plot
            boxplot_key = f"raw_{metric}_boxplot" # Key for pairwise box plot
            
            plot_grid_items = ""
            if kde_key in plot_paths:
                kde_file = os.path.basename(plot_paths[kde_key])
                plot_grid_items += f'''
                    <div class="plot-container">
                        <h4>Density Distribution (KDE) - Output vs. Output</h4>
                        <div class="iframe-container">
                            <iframe src="plots/{kde_file}"></iframe>
                        </div>
                    </div>
                '''
            
            if boxplot_key in plot_paths:
                boxplot_file = os.path.basename(plot_paths[boxplot_key])
                plot_grid_items += f'''
                    <div class="plot-container">
                        <h4>Box Plot - Output vs. Output</h4>
                        <div class="iframe-container">
                            <iframe src="plots/{boxplot_file}"></iframe>
                        </div>
                    </div>
                '''
            
            if plot_grid_items:
                 html_content += f'<div class="plot-grid">{plot_grid_items}</div>'
            else:
                html_content += f"<p>Pairwise distribution plots not available for {metric.title()}.</p>"
        # END OF Pairwise Distribution Comparison Section

        # NEW: Context Similarity Distributions (Output vs. Input) Section
        html_content += """
                <h2>Context Similarity Distributions (Output vs. Input)</h2>
                <p class="section-description">
                    These plots visualize the distribution of similarity scores comparing each <strong>generated idea to the original input context</strong> (e.g., paper abstract/methods).
                    This helps assess how relevant or aligned the generated ideas are to the source material for each prompting strategy.
                    <br><strong>KDE Plots:</strong> Each colored line represents an experiment type (prompting strategy). The X-axis is the similarity score (idea vs. context), and the Y-axis shows the density.
                </p>
        """
        context_kde_metrics = ["context_cosine", "context_self_bleu", "context_bertscore"] # Keys for context KDE plots
        for metric_display_key in context_kde_metrics:
            # Format for display, e.g., "Context Cosine" -> "Cosine"
            clean_metric_name = metric_display_key.replace("context_", "").replace("_", " ").title()
            html_content += f"<h3>{clean_metric_name} Density (Output vs. Input)</h3>\n"
            
            plot_key = f"{metric_display_key}_context_kde" # Key used when saving the plot
            if plot_key in plot_paths:
                plot_file = os.path.basename(plot_paths[plot_key])
                html_content += f'''
                    <div class="plot-container">
                        <div class="iframe-container">
                            <iframe src="plots/{plot_file}"></iframe>
                        </div>
                    </div>
                '''
            else:
                html_content += f"<p>Context KDE plot not available for {clean_metric_name}.</p>"
        # END OF Context Similarity Distributions Section

        # Ensure the final part of HTML is also correctly appended.
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write the HTML file
        with open(dashboard_path, "w") as f:
            f.write(html_content)
        
        # Save analysis results as JSON
        json_path = os.path.join(self.output_dir, "analysis_results.json")
        
        # Convert analysis results to JSON-serializable format
        serializable_results = {}
        for key, value in self.analysis_results.items():
            if key == "metrics_df":
                # Skip DataFrame
                continue
            elif isinstance(value, dict):
                serializable_results[key] = value
        
        # Convert any lingering numpy types (especially np.bool_) to Python native types
        serializable_results = self._convert_numpy_types_to_python(serializable_results)

        with open(json_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        return dashboard_path 