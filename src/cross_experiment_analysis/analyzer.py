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
                csv_to_raw_metric_map = {
                    "avg_cosine_similarity": "raw_cosine_similarities",
                    "avg_self_bleu": "raw_self_bleu_scores",
                    "avg_bertscore": "raw_bertscore_scores"
                }

                for csv_col_name, raw_metric_key in csv_to_raw_metric_map.items():
                    if csv_col_name in csv_data.columns:
                        # These might be stored as string representations of lists
                        try:
                            # If the column contains string representations of lists
                            if isinstance(csv_data[csv_col_name].iloc[0], str):
                                raw_scores = eval(csv_data[csv_col_name].iloc[0])
                            # If the column contains actual lists
                            else:
                                raw_scores = csv_data[csv_col_name].iloc[0]
                            
                            if isinstance(raw_scores, list):
                                exp_metrics[raw_metric_key] = raw_scores # Store as "raw_cosine_similarities", etc.
                        except (SyntaxError, ValueError, IndexError, TypeError) as e:
                            print(f"[WARNING] Could not parse {csv_col_name} from CSV for {exp_name}: {e}")
                
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
            "cosine": "raw_cosine_similarities",
            "self_bleu": "raw_self_bleu_scores",
            "bertscore": "raw_bertscore_scores"
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
        
        # Create bar charts for context and pairwise metrics
        for metric_type in ["context_metrics", "pairwise_metrics"]:
            for metric_name, metric_data in self.analysis_results[metric_type].items():
                # Create bar chart comparing means
                if metric_data["mean"]:
                    fig = go.Figure()
                    
                    # Add bars for each experiment type
                    for exp_type, mean_value in metric_data["mean"].items():
                        std_value = metric_data["std"].get(exp_type, 0)
                        
                        fig.add_trace(go.Bar(
                            x=[exp_type],
                            y=[mean_value],
                            name=exp_type,
                            error_y=dict(
                                type='data',
                                array=[std_value],
                                visible=True
                            )
                        ))
                    
                    # Update layout
                    metric_type_display = "Context" if metric_type == "context_metrics" else "Pairwise"
                    fig.update_layout(
                        title=f"{metric_type_display} {metric_name.title()} Comparison",
                        xaxis_title="Experiment Type",
                        yaxis_title=f"{metric_name.title()} Score",
                        barmode='group',
                        legend_title="Experiment Type"
                    )
                    
                    # Save the plot
                    plot_name = f"{metric_type.replace('_metrics', '')}_{metric_name}_comparison.html"
                    plot_path = os.path.join(self.output_dir, "plots", plot_name)
                    fig.write_html(plot_path)
                    plot_paths[f"{metric_type}_{metric_name}"] = plot_path
        
        # Create box plots for raw similarity scores
        raw_metrics = {
            "cosine": "raw_cosine_similarities",
            "self_bleu": "raw_self_bleu_scores",
            "bertscore": "raw_bertscore_scores"
        }
        
        for metric_name, metric_key in raw_metrics.items():
            # Check which experiments have raw scores for this metric
            data_for_box = []
            
            for exp_name, exp_metrics in self.experiment_metrics.items():
                if metric_key in exp_metrics and isinstance(exp_metrics[metric_key], list):
                    # Create data for box plot
                    exp_type = exp_metrics["type"]
                    raw_scores = exp_metrics[metric_key]
                    
                    for score in raw_scores:
                        data_for_box.append({
                            "Experiment Type": exp_type,
                            "Score": score,
                            "Metric": metric_name.title()
                        })
            
            if data_for_box:
                # Create DataFrame for box plot
                box_df = pd.DataFrame(data_for_box)
                
                # Create box plot
                fig = px.box(
                    box_df, 
                    x="Experiment Type", 
                    y="Score", 
                    color="Experiment Type",
                    title=f"{metric_name.title()} Distribution Comparison",
                    labels={"Score": f"{metric_name.title()} Score"}
                )
                
                # Add individual points as a scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=box_df["Experiment Type"],
                        y=box_df["Score"],
                        mode="markers",
                        marker=dict(
                            color="rgba(0, 0, 0, 0.3)",
                            size=4
                        ),
                        name="Individual Scores"
                    )
                )
                
                # Save the plot
                plot_name = f"raw_{metric_name}_boxplot.html"
                plot_path = os.path.join(self.output_dir, "plots", plot_name)
                fig.write_html(plot_path)
                plot_paths[f"raw_{metric_name}_boxplot"] = plot_path
        
        # Create KDE plots for similarity distributions
        for metric_name, metric_key in raw_metrics.items():
            # Check which experiments have raw scores for this metric
            exps_with_data = [exp for exp, data in self.experiment_metrics.items() 
                             if metric_key in data and isinstance(data[metric_key], list)]
            
            if exps_with_data:
                fig = go.Figure()
                
                for exp_name in exps_with_data:
                    exp_metrics = self.experiment_metrics[exp_name]
                    exp_type = exp_metrics["type"]
                    raw_scores = exp_metrics[metric_key]
                    
                    # Calculate KDE
                    if len(raw_scores) > 1:
                        kde = stats.gaussian_kde(raw_scores)
                        x = np.linspace(min(raw_scores), max(raw_scores), 1000)
                        y = kde(x)
                        
                        # Add KDE trace
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            name=exp_type,
                            line=dict(width=2)
                        ))
                
                # Update layout
                fig.update_layout(
                    title=f"{metric_name.title()} Density Distribution Comparison",
                    xaxis_title=f"{metric_name.title()} Score",
                    yaxis_title="Density",
                    legend_title="Experiment Type"
                )
                
                # Save the plot
                plot_name = f"{metric_name}_kde_comparison.html"
                plot_path = os.path.join(self.output_dir, "plots", plot_name)
                fig.write_html(plot_path)
                plot_paths[f"{metric_name}_kde"] = plot_path
        
        # Create runtime comparison plot
        if "runtime_comparison" in self.analysis_results:
            runtime_data = self.analysis_results["runtime_comparison"]
            
            # Create grouped bar chart for runtime and ideas per minute
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Runtime (seconds)", "Ideas per Minute"))
            
            # Add runtime bars
            exp_types = list(runtime_data["runtime_seconds"].keys())
            runtime_values = list(runtime_data["runtime_seconds"].values())
            
            fig.add_trace(
                go.Bar(
                    x=exp_types,
                    y=runtime_values,
                    name="Runtime (seconds)"
                ),
                row=1, col=1
            )
            
            # Add ideas per minute bars
            ideas_per_min_values = list(runtime_data["ideas_per_minute"].values())
            
            fig.add_trace(
                go.Bar(
                    x=exp_types,
                    y=ideas_per_min_values,
                    name="Ideas per Minute"
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Performance Comparison",
                showlegend=False
            )
            
            # Save the plot
            plot_name = "performance_comparison.html"
            plot_path = os.path.join(self.output_dir, "plots", plot_name)
            fig.write_html(plot_path)
            plot_paths["performance"] = plot_path
        
        return plot_paths
    
    def generate_comparison_dashboard(self) -> str:
        """
        Generate a comprehensive dashboard for cross-experiment comparison.
        
        Returns:
            Path to the generated dashboard HTML file
        """
        if not self.analysis_results:
            self.compare_metrics()
        
        # Generate plots
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
                .summary {{
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
                    height: 500px;
                    width: 100%;
                    overflow: hidden;
                }}
                .iframe-container iframe {{
                    width: 100%;
                    height: 100%;
                    border: none;
                }}
                .tabset {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: 20px 0;
                }}
                .tab-label {{
                    padding: 10px 15px;
                    background-color: #f2f2f2;
                    border: 1px solid #ddd;
                    cursor: pointer;
                    border-radius: 5px 5px 0 0;
                    margin-right: 5px;
                }}
                .tab-label.active {{
                    background-color: #3498db;
                    color: white;
                    border-color: #3498db;
                }}
                .tab-content {{
                    display: none;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 0 5px 5px 5px;
                    width: 100%;
                }}
                .tab-content.active {{
                    display: block;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Cross-Experiment Comparison Dashboard</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>This dashboard provides a comparison of results across different experiment types.</p>
                    <p>Total experiments analyzed: {len(self.experiment_data)}</p>
                    <p>Experiment types: {", ".join(sorted(set(exp["type"] for exp in self.experiment_metrics.values())))}</p>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <!-- Performance Metrics Section -->
                <h2>Performance Metrics</h2>
                <div class="iframe-container">
                    <iframe src="plots/performance_comparison.html"></iframe>
                </div>
                
                <!-- Context Similarity Metrics Section -->
                <h2>Context Similarity Metrics</h2>
                <div class="tabset" id="context-tabs">
        """
        
        # Add tabs for context metrics
        context_metrics = list(self.analysis_results.get("context_metrics", {}).keys())
        for i, metric in enumerate(context_metrics):
            active_class = "active" if i == 0 else ""
            html_content += f'<div class="tab-label {active_class}" onclick="openTab(\'context-tab-{i}\', \'context-tabs\')">{metric.title()}</div>\n'
        
        # Add tab content for context metrics
        html_content += "</div>\n"  # Close tab labels
        
        for i, metric in enumerate(context_metrics):
            active_class = "active" if i == 0 else ""
            plot_key = f"context_metrics_{metric}"
            if plot_key in plot_paths:
                plot_file = os.path.basename(plot_paths[plot_key])
                html_content += f'''
                <div id="context-tab-{i}" class="tab-content {active_class}">
                    <div class="iframe-container">
                        <iframe src="plots/{plot_file}"></iframe>
                    </div>
                </div>
                '''
        
        # Pairwise Similarity Metrics Section
        html_content += """
                <!-- Pairwise Similarity Metrics Section -->
                <h2>Pairwise Similarity Metrics</h2>
                <div class="tabset" id="pairwise-tabs">
        """
        
        # Add tabs for pairwise metrics
        pairwise_metrics = list(self.analysis_results.get("pairwise_metrics", {}).keys())
        for i, metric in enumerate(pairwise_metrics):
            active_class = "active" if i == 0 else ""
            html_content += f'<div class="tab-label {active_class}" onclick="openTab(\'pairwise-tab-{i}\', \'pairwise-tabs\')">{metric.title()}</div>\n'
        
        # Add tab content for pairwise metrics
        html_content += "</div>\n"  # Close tab labels
        
        for i, metric in enumerate(pairwise_metrics):
            active_class = "active" if i == 0 else ""
            plot_key = f"pairwise_metrics_{metric}"
            if plot_key in plot_paths:
                plot_file = os.path.basename(plot_paths[plot_key])
                html_content += f'''
                <div id="pairwise-tab-{i}" class="tab-content {active_class}">
                    <div class="iframe-container">
                        <iframe src="plots/{plot_file}"></iframe>
                    </div>
                </div>
                '''
        
        # Distribution Comparison Section
        html_content += """
                <!-- Distribution Comparison Section -->
                <h2>Distribution Comparison</h2>
                <div class="tabset" id="distribution-tabs">
        """
        
        # Add tabs for distribution metrics
        distribution_metrics = ["cosine", "self_bleu", "bertscore"]
        for i, metric in enumerate(distribution_metrics):
            active_class = "active" if i == 0 else ""
            html_content += f'<div class="tab-label {active_class}" onclick="openTab(\'dist-tab-{i}\', \'distribution-tabs\')">{metric.title()}</div>\n'
        
        # Add tab content for distribution metrics
        html_content += "</div>\n"  # Close tab labels
        
        for i, metric in enumerate(distribution_metrics):
            active_class = "active" if i == 0 else ""
            kde_key = f"{metric}_kde"
            boxplot_key = f"raw_{metric}_boxplot"
            
            html_content += f'<div id="dist-tab-{i}" class="tab-content {active_class}">\n'
            
            if kde_key in plot_paths:
                kde_file = os.path.basename(plot_paths[kde_key])
                html_content += f'''
                    <h3>Density Distribution</h3>
                    <div class="iframe-container">
                        <iframe src="plots/{kde_file}"></iframe>
                    </div>
                '''
            
            if boxplot_key in plot_paths:
                boxplot_file = os.path.basename(plot_paths[boxplot_key])
                html_content += f'''
                    <h3>Box Plot</h3>
                    <div class="iframe-container">
                        <iframe src="plots/{boxplot_file}"></iframe>
                    </div>
                '''
            
            html_content += "</div>\n"
        
        # Statistical Tests Section
        html_content += """
                <!-- Statistical Tests Section -->
                <h2>Statistical Tests</h2>
        """
        
        # Add tables for statistical tests
        if "statistical_tests" in self.analysis_results:
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
                    ks_test = test_results.get("ks_test", {"p_value": np.nan, "significant": False}) # Handle missing ks_test for older data
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
        
        # Experiment Detail Section
        html_content += """
                <!-- Experiment Details Section -->
                <h2>Experiment Details</h2>
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
                # Try to parse and format the date
                try:
                    date_obj = datetime.fromisoformat(run_date.replace("Z", "+00:00"))
                    run_date = date_obj.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    pass
            
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
                
                <script>
                function openTab(tabId, tabsetId) {
                    // Hide all tab content
                    var tabContents = document.getElementById(tabsetId).parentNode.querySelectorAll('.tab-content');
                    for (var i = 0; i < tabContents.length; i++) {
                        tabContents[i].classList.remove('active');
                    }
                    
                    // Remove active class from all tab labels
                    var tabLabels = document.getElementById(tabsetId).querySelectorAll('.tab-label');
                    for (var i = 0; i < tabLabels.length; i++) {
                        tabLabels[i].classList.remove('active');
                    }
                    
                    // Show the selected tab content
                    document.getElementById(tabId).classList.add('active');
                    
                    // Add active class to the clicked tab label
                    event.currentTarget.classList.add('active');
                }
                </script>
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