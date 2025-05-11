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
                
                # Enhanced debug for CSV columns
                print(f"[DEBUG] Available CSV columns for {exp_name}: {csv_data.columns.tolist()}")
                
                # Map CSV column name to internal raw metric key name for pairwise scores
                csv_to_raw_pairwise_metric_map = {
                    "cosine_similarities": "raw_pairwise_cosine", # Key in CSV : Key in exp_metrics
                    "self_bleu_scores": "raw_pairwise_self_bleu",
                    "bertscore_scores": "raw_pairwise_bertscore"
                }
                
                # NEW: Extract raw CONTEXT similarity scores from CSV data
                csv_to_raw_context_metric_map = {
                    "context_cosine_scores_raw": "raw_context_cosine", 
                    "context_self_bleu_scores_raw": "raw_context_self_bleu", # Added for context self-bleu
                    "context_bertscore_scores_raw": "raw_context_bertscore"
                }
                
                # Process all pairwise metrics columns
                for csv_col_name, raw_metric_key in csv_to_raw_pairwise_metric_map.items():
                    if csv_col_name in csv_data.columns:
                        try:
                            # Check if there's any data
                            if csv_data[csv_col_name].empty:
                                print(f"[DEBUG] Empty column for '{csv_col_name}' in {exp_name}. Storing as empty list.")
                                exp_metrics[raw_metric_key] = []
                                continue
                                
                            # Try different parsing methods
                            raw_scores = None
                            raw_scores_str = csv_data[csv_col_name].iloc[0]
                            print(f"[DEBUG] Raw data type for '{csv_col_name}' in {exp_name}: {type(raw_scores_str)}")
                            
                            # 1. Try direct eval if it's a string representation of a list
                            if isinstance(raw_scores_str, str):
                                try:
                                    # Fix: Make str representation safer for eval
                                    if raw_scores_str.strip().startswith('[') and raw_scores_str.strip().endswith(']'):
                                        raw_scores = eval(raw_scores_str)
                                    else:
                                        # Handle case where string might be formatted differently
                                        import re
                                        # Extract numbers from string
                                        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw_scores_str)
                                        raw_scores = [float(n) for n in numbers]
                                except (SyntaxError, ValueError) as e:
                                    print(f"[DEBUG] Failed to parse with eval due to {e}: {raw_scores_str[:50]}...")
                                    
                                    # 2. Try json.loads if it's a JSON string
                                    if raw_scores is None:
                                        try:
                                            import json
                                            # Ensure the string is properly JSON formatted
                                            clean_str = raw_scores_str.replace("'", '"')
                                            raw_scores = json.loads(clean_str)
                                        except (json.JSONDecodeError, ValueError) as e:
                                            print(f"[DEBUG] Failed to parse with json due to {e}: {raw_scores_str[:50]}...")
                                    
                                    # 3. Try literal_eval as a safer alternative to eval
                                    if raw_scores is None:
                                        try:
                                            from ast import literal_eval
                                            raw_scores = literal_eval(raw_scores_str)
                                        except (SyntaxError, ValueError) as e:
                                            print(f"[DEBUG] Failed to parse with literal_eval due to {e}: {raw_scores_str[:50]}...")
                                    
                                    # 4. Try extracting list directly from string representation
                                    if raw_scores is None and '[' in raw_scores_str and ']' in raw_scores_str:
                                        try:
                                            import re
                                            # Extract the list content between the first '[' and the last ']'
                                            list_content = raw_scores_str[raw_scores_str.find('[')+1:raw_scores_str.rfind(']')]
                                            # Split by comma and convert to float
                                            raw_scores = [float(x.strip()) for x in list_content.split(',') if x.strip()]
                                        except (ValueError, TypeError) as e:
                                            print(f"[DEBUG] Failed to extract list directly due to {e}: {raw_scores_str[:50]}...")
                            
                            if isinstance(raw_scores, list):
                                # Validate that all items are numeric
                                validated_scores = []
                                for score in raw_scores:
                                    try:
                                        validated_scores.append(float(score))
                                    except (ValueError, TypeError):
                                        print(f"[WARNING] Skipping non-numeric value in '{csv_col_name}' for {exp_name}: {score}")
                                
                                if validated_scores:
                                    print(f"[DEBUG] Successfully parsed {len(validated_scores)} scores for '{csv_col_name}' in {exp_name}")
                                    exp_metrics[raw_metric_key] = validated_scores
                                else:
                                    print(f"[DEBUG] No valid scores found for '{csv_col_name}' in {exp_name}. Storing empty list.")
                                    exp_metrics[raw_metric_key] = []
                            else:
                                if raw_scores is None:
                                    print(f"[WARNING] Failed to parse raw pairwise scores for '{csv_col_name}' in {exp_name} into a valid format. Storing as empty list.")
                                    exp_metrics[raw_metric_key] = []
                                else:
                                    # Try to handle case where we got a single scalar value
                                    try:
                                        single_value = float(raw_scores)
                                        print(f"[DEBUG] Got single numeric value for '{csv_col_name}' in {exp_name}. Converting to list.")
                                        exp_metrics[raw_metric_key] = [single_value]
                                    except (ValueError, TypeError):
                                        print(f"[WARNING] Failed to parse raw pairwise scores for '{csv_col_name}' in {exp_name} into a list. Storing as empty list.")
                                        exp_metrics[raw_metric_key] = []
                        except Exception as e:
                            print(f"[WARNING] Error parsing raw pairwise scores for '{csv_col_name}' in {exp_name}: {e}. Storing as empty list.")
                            exp_metrics[raw_metric_key] = []
                    else:
                        print(f"[DEBUG] Pairwise raw score column '{csv_col_name}' not found in CSV for {exp_name}. Storing empty list for {raw_metric_key}.")
                        exp_metrics[raw_metric_key] = []
                
                # Process all context metrics columns using the same extraction logic
                for csv_col_name, raw_metric_key in csv_to_raw_context_metric_map.items():
                    if csv_col_name in csv_data.columns:
                        try:
                            # Check if there's any data
                            if csv_data[csv_col_name].empty:
                                print(f"[DEBUG] Empty column for context metric '{csv_col_name}' in {exp_name}. Storing as empty list.")
                                exp_metrics[raw_metric_key] = []
                                continue
                                
                            # Try different parsing methods
                            raw_scores = None
                            raw_scores_str = csv_data[csv_col_name].iloc[0]
                            print(f"[DEBUG] Raw data type for context metric '{csv_col_name}' in {exp_name}: {type(raw_scores_str)}")
                            
                            # 1. Try direct eval if it's a string representation of a list
                            if isinstance(raw_scores_str, str):
                                try:
                                    # Fix: Make str representation safer for eval
                                    if raw_scores_str.strip().startswith('[') and raw_scores_str.strip().endswith(']'):
                                        raw_scores = eval(raw_scores_str)
                                    else:
                                        # Handle case where string might be formatted differently
                                        import re
                                        # Extract numbers from string
                                        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw_scores_str)
                                        raw_scores = [float(n) for n in numbers]
                                except (SyntaxError, ValueError) as e:
                                    print(f"[DEBUG] Failed to parse with eval due to {e}: {raw_scores_str[:50]}...")
                                    
                                    # 2. Try json.loads if it's a JSON string
                                    if raw_scores is None:
                                        try:
                                            import json
                                            # Ensure the string is properly JSON formatted
                                            clean_str = raw_scores_str.replace("'", '"')
                                            raw_scores = json.loads(clean_str)
                                        except (json.JSONDecodeError, ValueError) as e:
                                            print(f"[DEBUG] Failed to parse with json due to {e}: {raw_scores_str[:50]}...")
                                    
                                    # 3. Try literal_eval as a safer alternative to eval
                                    if raw_scores is None:
                                        try:
                                            from ast import literal_eval
                                            raw_scores = literal_eval(raw_scores_str)
                                        except (SyntaxError, ValueError) as e:
                                            print(f"[DEBUG] Failed to parse with literal_eval due to {e}: {raw_scores_str[:50]}...")
                                    
                                    # 4. Try extracting list directly from string representation
                                    if raw_scores is None and '[' in raw_scores_str and ']' in raw_scores_str:
                                        try:
                                            import re
                                            # Extract the list content between the first '[' and the last ']'
                                            list_content = raw_scores_str[raw_scores_str.find('[')+1:raw_scores_str.rfind(']')]
                                            # Split by comma and convert to float
                                            raw_scores = [float(x.strip()) for x in list_content.split(',') if x.strip()]
                                        except (ValueError, TypeError) as e:
                                            print(f"[DEBUG] Failed to extract list directly due to {e}: {raw_scores_str[:50]}...")
                            
                            if isinstance(raw_scores, list):
                                # Validate that all items are numeric
                                validated_scores = []
                                for score in raw_scores:
                                    try:
                                        validated_scores.append(float(score))
                                    except (ValueError, TypeError):
                                        print(f"[WARNING] Skipping non-numeric value in context metric '{csv_col_name}' for {exp_name}: {score}")
                                
                                if validated_scores:
                                    print(f"[DEBUG] Successfully parsed {len(validated_scores)} context scores for '{csv_col_name}' in {exp_name}")
                                    exp_metrics[raw_metric_key] = validated_scores
                                else:
                                    print(f"[DEBUG] No valid context scores found for '{csv_col_name}' in {exp_name}. Storing empty list.")
                                    exp_metrics[raw_metric_key] = []
                            else:
                                if raw_scores is None:
                                    print(f"[WARNING] Failed to parse raw context scores for '{csv_col_name}' in {exp_name} into a valid format. Storing as empty list.")
                                    exp_metrics[raw_metric_key] = []
                                else:
                                    # Try to handle case where we got a single scalar value
                                    try:
                                        single_value = float(raw_scores)
                                        print(f"[DEBUG] Got single numeric value for '{csv_col_name}' in {exp_name}. Converting to list.")
                                        exp_metrics[raw_metric_key] = [single_value]
                                    except (ValueError, TypeError):
                                        print(f"[WARNING] Failed to parse raw context scores for '{csv_col_name}' in {exp_name} into a list. Storing as empty list.")
                                        exp_metrics[raw_metric_key] = []
                        except Exception as e:
                            print(f"[WARNING] Error parsing raw context scores for '{csv_col_name}' in {exp_name}: {e}. Storing as empty list.")
                            exp_metrics[raw_metric_key] = []
                    else:
                        print(f"[DEBUG] Context metric column '{csv_col_name}' not found in CSV for {exp_name}. Storing empty list for {raw_metric_key}.")
                        exp_metrics[raw_metric_key] = []
                
                # Get KDE data if available
                if "kde_values" in csv_data.columns:
                    try:
                        kde_values = eval(csv_data["kde_values"].iloc[0])
                        exp_metrics["kde_values"] = kde_values
                    except (SyntaxError, ValueError, IndexError) as e:
                        print(f"[WARNING] Could not parse KDE values from CSV: {e}")
                
                # Check if we've obtained valid raw scores; if not, try to extract from results.json
                pairwise_metrics_to_check = ["raw_pairwise_cosine", "raw_pairwise_self_bleu", "raw_pairwise_bertscore"]
                context_metrics_to_check = ["raw_context_cosine", "raw_context_self_bleu", "raw_context_bertscore"]
                
                # Flag to check if any pairwise metrics were missing
                missing_pairwise_metrics = any(
                    exp_metrics.get(metric_key, []) == [] for metric_key in pairwise_metrics_to_check
                )
                
                # Flag to check if any context metrics were missing
                missing_context_metrics = any(
                    exp_metrics.get(metric_key, []) == [] for metric_key in context_metrics_to_check
                )
                
                # If we're missing any metrics, try to extract from results.json
                if missing_pairwise_metrics or missing_context_metrics:
                    print(f"[INFO] Some raw scores missing from CSV for {exp_name}. Trying to extract from results.json...")
                    results = exp_data["results"]
                    
                    # Extract individual experiment results
                    if "results" in results and isinstance(results["results"], list):
                        for result in results["results"]:
                            # Extract pairwise metrics if missing
                            if missing_pairwise_metrics:
                                for csv_key, raw_key in csv_to_raw_pairwise_metric_map.items():
                                    if raw_key in pairwise_metrics_to_check and exp_metrics.get(raw_key, []) == [] and csv_key in result:
                                        scores = result[csv_key]
                                        if isinstance(scores, list) and scores:
                                            print(f"[INFO] Found {len(scores)} {csv_key} scores in results.json for {exp_name}")
                                            # Ensure all values are float
                                            try:
                                                validated_scores = [float(score) for score in scores if pd.notna(score)]
                                                exp_metrics[raw_key] = validated_scores
                                            except (ValueError, TypeError) as e:
                                                print(f"[WARNING] Error converting scores to float in results.json: {e}")
                            
                            # Extract context metrics if missing
                            if missing_context_metrics:
                                for csv_key, raw_key in csv_to_raw_context_metric_map.items():
                                    if raw_key in context_metrics_to_check and exp_metrics.get(raw_key, []) == [] and csv_key in result:
                                        scores = result[csv_key]
                                        if isinstance(scores, list) and scores:
                                            print(f"[INFO] Found {len(scores)} {csv_key} scores in results.json for {exp_name}")
                                            # Ensure all values are float
                                            try:
                                                validated_scores = [float(score) for score in scores if pd.notna(score)]
                                                exp_metrics[raw_key] = validated_scores
                                            except (ValueError, TypeError) as e:
                                                print(f"[WARNING] Error converting scores to float in results.json: {e}")
                            
                            # Extract KDE values if missing
                            if "kde_values" not in exp_metrics and "kde_values" in result:
                                exp_metrics["kde_values"] = result["kde_values"]
                                print(f"[INFO] Found KDE values in results.json for {exp_name}")
                    
                    # Try a different approach - look directly at the original log_data if results extraction failed
                    if (missing_pairwise_metrics or missing_context_metrics) and "summary" in results:
                        print(f"[INFO] Attempting to extract from summary in results.json for {exp_name}...")
                        try:
                            # Look for raw scores in Context Similarities and Pairwise Similarities
                            if missing_context_metrics and "Context Similarities" in results["summary"]:
                                context_sims = results["summary"]["Context Similarities"]
                                # Raw data might be stored under _raw suffix
                                for metric in ["cosine_raw", "bertscore_raw"]:
                                    if metric in context_sims:
                                        raw_key = f"raw_context_{metric.replace('_raw', '')}"
                                        if raw_key in context_metrics_to_check and exp_metrics.get(raw_key, []) == []:
                                            if isinstance(context_sims[metric], list) and context_sims[metric]:
                                                print(f"[INFO] Found {len(context_sims[metric])} scores for {raw_key} in summary.")
                                                exp_metrics[raw_key] = context_sims[metric]
                            
                            if missing_pairwise_metrics and "Pairwise Similarities" in results["summary"]:
                                pairwise_sims = results["summary"]["Pairwise Similarities"]
                                # Raw data might be stored under _raw suffix
                                for metric in ["cosine_raw", "self_bleu_raw", "bertscore_raw"]:
                                    if metric in pairwise_sims:
                                        raw_key = f"raw_pairwise_{metric.replace('_raw', '')}"
                                        if raw_key in pairwise_metrics_to_check and exp_metrics.get(raw_key, []) == []:
                                            if isinstance(pairwise_sims[metric], list) and pairwise_sims[metric]:
                                                print(f"[INFO] Found {len(pairwise_sims[metric])} scores for {raw_key} in summary.")
                                                exp_metrics[raw_key] = pairwise_sims[metric]
                        except Exception as e:
                            print(f"[WARNING] Error extracting from summary: {e}")

                # Last chance fallback - if still missing data, log warning but continue with empty arrays
                still_missing_pairwise = any(
                    exp_metrics.get(metric_key, []) == [] for metric_key in pairwise_metrics_to_check
                )
                still_missing_context = any(
                    exp_metrics.get(metric_key, []) == [] for metric_key in context_metrics_to_check
                )
                
                if still_missing_pairwise:
                    missing_metrics = [key for key in pairwise_metrics_to_check if exp_metrics.get(key, []) == []]
                    print(f"[WARNING] Still missing pairwise metrics after all extraction attempts: {missing_metrics}")
                
                if still_missing_context:
                    missing_metrics = [key for key in context_metrics_to_check if exp_metrics.get(key, []) == []]
                    print(f"[WARNING] Still missing context metrics after all extraction attempts: {missing_metrics}")
            
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
        # Define a mapping for PAIRWISE raw metrics
        pairwise_raw_metrics_map = {
            "pairwise_cosine": "raw_pairwise_cosine",
            "pairwise_self_bleu": "raw_pairwise_self_bleu",
            "pairwise_bertscore": "raw_pairwise_bertscore"
        }

        # Define a mapping for CONTEXT raw metrics
        context_raw_metrics_map = {
            "context_cosine": "raw_context_cosine",
            "context_self_bleu": "raw_context_self_bleu", # Assuming you will add/ensure this key exists in extract_metrics
            "context_bertscore": "raw_context_bertscore"
        }

        # Combine both maps for iteration
        all_raw_metrics_to_test = {**pairwise_raw_metrics_map, **context_raw_metrics_map}
        
        for display_metric_name, data_metric_key in all_raw_metrics_to_test.items():
            # Check which experiments have raw scores for this metric
            exps_with_data = [exp for exp, data in self.experiment_metrics.items() 
                             if data_metric_key in data and isinstance(data[data_metric_key], list) and data[data_metric_key]]
            
            if len(exps_with_data) >= 2:
                test_results = {}
                
                for i in range(len(exps_with_data)):
                    for j in range(i + 1, len(exps_with_data)):
                        exp1 = exps_with_data[i]
                        exp2 = exps_with_data[j]
                        
                        scores1 = self.experiment_metrics[exp1][data_metric_key]
                        scores2 = self.experiment_metrics[exp2][data_metric_key]
                        
                        try:
                            ks_stat, ks_p = stats.ks_2samp(scores1, scores2)
                            mean1, std1 = np.mean(scores1), np.std(scores1)
                            mean2, std2 = np.mean(scores2), np.std(scores2)
                            
                            # Ensure std1 and std2 are not zero and lengths are sufficient for pooled_std
                            if len(scores1) > 1 and len(scores2) > 1 and std1 > 0 and std2 > 0:
                                pooled_std = np.sqrt(((len(scores1) - 1) * std1**2 + 
                                                     (len(scores2) - 1) * std2**2) / 
                                                    (len(scores1) + len(scores2) - 2))
                                cohen_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
                            else: # Fallback if cannot compute pooled_std or if std is zero
                                cohen_d = 0 
                                if np.std(np.concatenate([scores1, scores2])) > 0: # Check if combined data has variance
                                   cohen_d = (mean1 - mean2) / np.std(np.concatenate([scores1, scores2]))
                                else: # True zero variance case for combined data
                                   cohen_d = 0
                            
                            test_results[f"{exp1} vs {exp2}"] = {
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
                            print(f"[WARNING] Statistical test failed for {display_metric_name} ({exp1} vs {exp2}): {e}")
                
                try:
                    all_exp_scores = []
                    group_labels = []
                    for exp_name in exps_with_data:
                        if data_metric_key in self.experiment_metrics[exp_name]:
                            scores = self.experiment_metrics[exp_name][data_metric_key]
                            if scores: # Ensure scores list is not empty
                                all_exp_scores.append(scores)
                                group_labels.append(exp_name)

                    if len(all_exp_scores) >= 2:
                        # Filter out empty lists from all_exp_scores before passing to kruskal
                        valid_scores_for_kruskal = [s for s in all_exp_scores if len(s) > 0]
                        if len(valid_scores_for_kruskal) >=2: # Need at least two groups with data
                            kw_stat, kw_p = stats.kruskal(*valid_scores_for_kruskal)
                            test_results["kruskal_wallis"] = {
                                "statistic": float(kw_stat),
                                "p_value": float(kw_p),
                                "significant": kw_p < 0.05,
                                "groups": group_labels # groups should correspond to all_exp_scores used
                            }
                        else:
                            print(f"[INFO] Kruskal-Wallis skipped for {display_metric_name}: Not enough groups with data after filtering empty lists.")
                except Exception as e:
                    print(f"[WARNING] Kruskal-Wallis test failed for {display_metric_name}: {e}")
                
                comparison_results["statistical_tests"][display_metric_name] = test_results
        
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
                        <th>KS Test p-value</th>
                        <th>Significant?</th>
                        <th>Effect Size (Cohen's d)</th>
                        <th>Interpretation</th>
                    </tr>
                """
                
                for comparison, test_results in tests.items():
                    # Skip the Kruskal-Wallis test entry when iterating through pairwise comparisons
                    if comparison == "kruskal_wallis":
                        continue
                        
                    ks_test = test_results["ks_test"]
                    effect_size = test_results["effect_size"]
                    
                    html_content += f"""
                    <tr>
                        <td>{comparison}</td>
                        <td>{ks_test['p_value']:.4f}</td>
                        <td>{"Yes" if ks_test['significant'] else "No"}</td>
                        <td>{effect_size['cohen_d']:.4f}</td>
                        <td>{effect_size['interpretation']}</td>
                    </tr>
                    """
                html_content += "</table>\n"
                
                # Add Kruskal-Wallis test results if available
                if "kruskal_wallis" in tests:
                    kw_test = tests["kruskal_wallis"]
                    html_content += f"""
                    <h4>Kruskal-Wallis Test (Overall Comparison)</h4>
                    <p>Tests whether the distributions of all experiment types are identical or at least one differs.</p>
                    <table class="stat-table">
                        <tr>
                            <th>Test Statistic</th>
                            <th>p-value</th>
                            <th>Significant?</th>
                            <th>Groups Compared</th>
                        </tr>
                        <tr>
                            <td>{kw_test['statistic']:.4f}</td>
                            <td>{kw_test['p_value']:.4f}</td>
                            <td>{"Yes" if kw_test['significant'] else "No"}</td>
                            <td>{", ".join(kw_test['groups'])}</td>
                        </tr>
                    </table>
                    """
            html_content += '</div>' # End scrollable div
        else:
            html_content += "<p>No statistical test results available.</p>"
        # STATISTICAL TESTS SECTION ENDS HERE (MOVED)

        # NEW: Conclusion Section - REMOVING THIS AS PER DECISION TO USE STATISTICALANALYZER

        # Context Similarity Metrics Section - REMOVING THIS ENTIRE BLOCK
        # (This comment is from a previous step, the block itself is already removed)

        # Distribution Comparison Section - RENAMING AND KEEPING for Pairwise (Output vs Output)
        html_content += """
                <h2>Similarity Distributions Comparison</h2>
                <p class="section-description">
                    These plots visualize the distributions of similarity scores, comparing both pairwise scores between generated ideas and similarity to the input context.
                </p>
        """

        # Group related metric plots together
        distribution_metrics = ["cosine", "bertscore"] # Remove "self_bleu" from this list
        
        for metric in distribution_metrics:
            html_content += f"<h3>{metric.title()} Similarity Distributions</h3>\n"
            
            # Add Pairwise (Output vs. Output) plots first
            kde_pairwise_key = f"{metric}_kde" # Key for pairwise KDE plot
            boxplot_key = f"raw_{metric}_boxplot" # Key for pairwise box plot
            
            html_content += """
                <div class="plot-grid">
            """
            
            # Pairwise (output vs output) plots
            html_content += """
                        <div class="plot-container">
                    <h4>Output vs. Output Distribution (KDE)</h4>
                    <p>Similarity between different generated ideas (pairwise comparison)</p>
            """
            
            if kde_pairwise_key in plot_paths:
                kde_file = os.path.basename(plot_paths[kde_pairwise_key])
                html_content += f'''
                            <div class="iframe-container">
                        <iframe src="plots/{kde_file}"></iframe>
                            </div>
                '''
            else:
                html_content += f"<p>Pairwise KDE plot not available for {metric.title()}.</p>"
                
            html_content += """
                </div>
            """
            
            # Output vs. Input plots for the same metric
            context_metric_display = f"context_{metric}"
            kde_context_key = f"{context_metric_display}_context_kde"
            
            html_content += """
                <div class="plot-container">
                    <h4>Output vs. Input Distribution (KDE)</h4>
                    <p>Similarity between each generated idea and the original input context</p>
            """
            
            if kde_context_key in plot_paths:
                context_kde_file = os.path.basename(plot_paths[kde_context_key])
                html_content += f'''
                    <div class="iframe-container">
                        <iframe src="plots/{context_kde_file}"></iframe>
                        </div>
                    '''
            else:
                html_content += f"<p>Context KDE plot not available for {metric.title()}.</p>"
        
            html_content += """
                </div>
            """
            
            # Add boxplot in a new row if available
            if boxplot_key in plot_paths:
                boxplot_file = os.path.basename(plot_paths[boxplot_key])
                html_content += f'''
                <div class="plot-container">
                    <h4>Box Plot Distribution (Output vs. Output)</h4>
                    <div class="iframe-container">
                        <iframe src="plots/{boxplot_file}"></iframe>
                    </div>
                </div>
                '''
            
            html_content += """
            </div>
            """
            
            # Add Self-BLEU section separately (only Output vs. Output)
            html_content += """
                <h3>Self-BLEU Distribution (Output Diversity Only)</h3>
                <p>Self-BLEU measures text overlap between pairs of generated ideas. Lower values indicate more diverse outputs.</p>
            """
            
            metric = "self_bleu"
            kde_key = f"{metric}_kde"
            boxplot_key = f"raw_{metric}_boxplot"
            
            html_content += """
                <div class="plot-grid">
            """
            
            if kde_key in plot_paths:
                kde_file = os.path.basename(plot_paths[kde_key])
                html_content += f'''
                    <div class="plot-container">
                    <h4>Output vs. Output Distribution (KDE)</h4>
                        <div class="iframe-container">
                            <iframe src="plots/{kde_file}"></iframe>
                        </div>
                    </div>
                '''
            
            if boxplot_key in plot_paths:
                boxplot_file = os.path.basename(plot_paths[boxplot_key])
                html_content += f'''
                    <div class="plot-container">
                    <h4>Box Plot Distribution</h4>
                        <div class="iframe-container">
                            <iframe src="plots/{boxplot_file}"></iframe>
                        </div>
                    </div>
                '''
            
            html_content += """
                </div>
            """
            # End of new distribution section layout

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