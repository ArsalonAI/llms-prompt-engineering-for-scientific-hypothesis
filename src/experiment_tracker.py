import os
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
import time

class ExperimentTracker:
    # =====================
    # Core Experiment Management Methods
    # =====================
    def __init__(self, output_dir: str = "experiment_results"):
        """
        Initialize the experiment tracker.
        
        Args:
            output_dir: Directory to store experiment results
        """
        self.output_dir = output_dir
        self.ensure_output_dir()
        self.current_experiment = None
        self._experiment_type = None
        self._step = 0
        self._config = None
        self._results_df = None
        self._start_time = None
        
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_run_id(self, base_name: str) -> str:
        """Generate a unique run ID based on experiment config and timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{base_name}_{timestamp}"

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Safely get a configuration value"""
        return self._config.get(key, default) if self._config else default

    def start_experiment(self, 
                        experiment_name: str, 
                        experiment_type: str = "idea_generation",
                        model_name: str = None,
                        config: Dict[str, Any] = None):
        """Start tracking a new experiment."""
        if self.current_experiment is not None:
            print(f"[INFO] Automatically ending previous experiment: {self.current_experiment['name']} before starting {experiment_name}")
            self.end_experiment()

        self._experiment_type = experiment_type
        self._step = 0
        self._start_time = time.time()  # Record start time for runtime tracking
        
        # Define dtypes for a more robust DataFrame initialization
        # This helps prevent dtype-related warnings during pd.concat in log_result
        results_df_dtypes = {
            'step': pd.Int64Dtype(), 
            'run_id': str,
            'idea': str,            
            'batch_prompt': str,  
            'evaluation': str,      
            'evaluation_full': str, 
            'avg_cosine_similarity': object, # Stores list of raw pairwise scores
            'avg_self_bleu': object,       # Stores list of raw pairwise scores
            'avg_bertscore': object,       # Stores list of raw pairwise scores
            'context_cosine': object,       # Stores list of raw context scores for cosine
            'context_self_bleu': object,    # Stores list of raw context scores for self_bleu
            'context_bertscore': object,    # Stores list of raw context scores for bertscore
            'timestamp': str, 
            'model': str, 
            'prompt': str, 
            'context': str, # This is the main paper context string
            'num_ideas': pd.Int64Dtype(), 
            'quality_scores': object, 
            # Raw score lists (if run_idea_generation_batch provides them with these exact keys for df population)
            # These are the keys `log_result` will look for in the `result` dict to populate df_row.
            # `run_idea_generation_batch` now provides `context_..._scores_raw` and the pairwise lists directly.
            'cosine_similarities': object, # Raw pairwise list
            'self_bleu_scores': object,    # Raw pairwise list
            'bertscore_scores': object,    # Raw pairwise list
            'context_cosine_scores_raw': object, 
            'context_self_bleu_scores_raw': object, 
            'context_bertscore_scores_raw': object, 
            'has_kde_data': bool,
            'runtime_seconds': float,  # Add tracking for runtime
            'pairwise_pairs_compared': object # Stores the list of (i,j) tuples for actual pairwise comparisons
        }
        # Initialize results DataFrame with all required columns and dtypes
        self._results_df = pd.DataFrame(columns=list(results_df_dtypes.keys()))
        for col, dtype in results_df_dtypes.items():
            # Ensure column exists before trying to astype; pd.DataFrame(columns=...) creates them
            if col in self._results_df.columns:
                self._results_df[col] = self._results_df[col].astype(dtype)
            else: # Should not happen if columns=list(results_df_dtypes.keys())
                print(f"[WARN] Column '{col}' defined in dtypes but not in DataFrame during init.")
        
        # Ensure config has required fields
        self._config = config or {}
        self._config.update({
            "experiment_name": experiment_name,
            "experiment_type": experiment_type,
            "model_name": model_name,
            "start_time": datetime.now().isoformat()
        })
        
        self.current_experiment = {
            'name': experiment_name,
            'type': experiment_type,
            'model': model_name,
            'config': self._config,
            'start_time': self._config["start_time"],
            'results': []
        }
        
        print(f"[INFO] Started experiment: {experiment_name}")
    
    def log_result(self, run_id: str, result: Dict[str, Any]):
        """Log a result for the current experiment."""
        if self.current_experiment is None:
            raise ValueError("No experiment currently running. Call start_experiment first.")
        
        self._step += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Clean and validate numeric values
        cleaned_result = result.copy()
        for key in ['cosine_similarity', 'self_bleu', 'bertscore']:
            if key in cleaned_result:
                try:
                    cleaned_result[key] = float(cleaned_result[key])
                except (TypeError, ValueError):
                    cleaned_result[key] = None
        
        # Add metadata to result
        result_with_metadata = {
            'timestamp': timestamp,
            'step': self._step,
            'run_id': run_id,
            **cleaned_result
        }
        
        # Store locally
        self.current_experiment['results'].append(result_with_metadata)
        
        # Prepare data for DataFrame
        df_row = {
            'timestamp': timestamp,
            'step': self._step,
            'run_id': run_id,
            'model': result.get('model', 'N/A'),
            'prompt': result.get('prompt', 'N/A'),
            'context': result.get('context', 'N/A'),
            'num_ideas': result.get('num_ideas', 0),
            
            # Store the raw lists of scores directly. _calculate_similarity_stats will process these.
            'avg_cosine_similarity': result.get('cosine_similarities'), # Was avg_pairwise_cosine_similarity
            'avg_self_bleu': result.get('self_bleu_scores'),       # Was avg_pairwise_self_bleu
            'avg_bertscore': result.get('bertscore_scores'),       # Was avg_pairwise_bertscore
            
            'context_cosine': result.get('context_cosine_scores_raw'), # List of context cosine for each idea
            'context_self_bleu': result.get('context_self_bleu_scores_raw', []), # Expecting a list, default to empty if not present
            'context_bertscore': result.get('context_bertscore_scores_raw', [])  # Expecting a list, default to empty if not present
        }
        
        # Add other raw lists and specific items like quality_scores, kde_values etc.
        # This part populates columns that might be lists (e.g. quality_scores can be list of dicts)
        # or other objects.
        # The keys here should match column names defined in results_df_dtypes in start_experiment.
        for key in ['ideas', 'quality_scores', 'has_kde_data', 'batch_prompt', 'evaluation', 'evaluation_full', 'pairwise_pairs_compared']:
            if key in result:
                df_row[key] = result.get(key)
            elif key == 'has_kde_data':
                df_row[key] = True if 'kde_values' in result else False
        
        # Note: 'cosine_similarities', 'self_bleu_scores', 'bertscore_scores', 'context_cosine_scores_raw'
        # are now directly assigned to avg_cosine_similarity, avg_self_bleu, etc. above.
        # If you need to keep *both* the averaged values (if calculated in run_idea_generation_batch)
        # AND the raw lists under different column names in the CSV, the df_row and results_df_dtypes need adjustment.
        # Current approach: the "avg_" columns now hold lists for _calculate_similarity_stats to process.

        # The following loop for adding lists to df_row might be redundant if already handled above
        # for key in ['quality_scores', 'cosine_similarities', 'self_bleu_scores', 'bertscore_scores', 'context_cosine_scores_raw']:
        #     if key in result:
        #         df_row[key] = result[key]

        # Ensure all columns defined in dtypes are present in df_row, adding NaN if missing
        # This is important if the `result` dict from run_idea_generation_batch is sparse.
        expected_cols = self._results_df.columns # Get columns from the pre-defined DataFrame structure
        for col in expected_cols:
            if col not in df_row:
                df_row[col] = np.nan # Or an appropriate default like [] for object types meant to be lists

        # Update DataFrame
        self._results_df = pd.concat([
            self._results_df,
            pd.DataFrame([df_row])
        ], ignore_index=True)
        
        # Print concise progress
        metrics_avg = {}
        # Only average metrics that are lists of numbers
        for metric_key_in_result_dict, display_name in [
            ('cosine_similarities', 'cosine'),
            ('self_bleu_scores', 'self_bleu'),
            ('bertscore_scores', 'bertscore')
        ]:
            if metric_key_in_result_dict in result and result[metric_key_in_result_dict] and isinstance(result[metric_key_in_result_dict], list):
                numeric_values = [v for v in result[metric_key_in_result_dict] if isinstance(v, (int, float))]
                if numeric_values:
                    metrics_avg[display_name] = sum(numeric_values) / len(numeric_values)
        
        # Handle quality summary separately (e.g., count ACCEPT/PRUNE)
        quality_summary = "N/A"
        if 'quality_scores' in result and result['quality_scores'] and isinstance(result['quality_scores'], list):
            try:
                accept_count = sum(1 for q in result['quality_scores'] if isinstance(q, dict) and q.get('evaluation') == 'ACCEPT')
                total_quality = len(result['quality_scores'])
                quality_summary = f"{accept_count}/{total_quality} ACCEPT"
            except TypeError: # Handle case where quality_scores might not contain dicts as expected
                quality_summary = "Error processing quality"

        print(f"\n[STEP {self._step}]")
        print(f"Run ID: {run_id} | Ideas: {result.get('num_ideas', 0)} | Quality: {quality_summary}")
        if metrics_avg:
            print("Average Pairwise Metrics: " + " | ".join([
                f"{key}={val:.3f}" for key, val in metrics_avg.items()
            ]))
        else:
            print("Average Pairwise Metrics: N/A")
    
    def end_experiment(self):
        """End the current experiment and save results."""
        if self.current_experiment is None:
            raise ValueError("No experiment currently running.")
        
        # Calculate runtime
        end_time = time.time()
        runtime_seconds = end_time - self._start_time
        
        self.current_experiment['end_time'] = datetime.now().isoformat()
        self.current_experiment['runtime_seconds'] = runtime_seconds
        
        # Create experiment directory with absolute path and detailed timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # New format: YYYY-MM-DD_HH-MM-SS
        experiment_dir = os.path.abspath(os.path.join(
            self.output_dir,
            f"{self.current_experiment['name']}_{timestamp}"
        ))
        os.makedirs(experiment_dir, exist_ok=True)
        
        print(f"\n[INFO] Creating experiment directory: {os.path.basename(experiment_dir)}")
        
        # Save experiment metadata
        metadata = {
            'name': self.current_experiment['name'],
            'type': self.current_experiment['type'],
            'model': self.current_experiment['model'],
            'config': self.current_experiment['config'],
            'start_time': self.current_experiment['start_time'],
            'end_time': self.current_experiment['end_time'],
            'runtime_seconds': runtime_seconds
        }
        metadata_path = os.path.join(experiment_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Calculate the actual number of ideas generated
        total_ideas = 0
        if 'ideas' in self._results_df.columns and not self._results_df.empty:
            # If we have direct access to the ideas, count them
            for _, row in self._results_df.iterrows():
                ideas_list = row.get('ideas', [])
                if isinstance(ideas_list, list):
                    total_ideas += len(ideas_list)
        elif 'num_ideas' in self._results_df.columns and not self._results_df.empty:
            # Otherwise use the num_ideas column
            total_ideas = self._results_df['num_ideas'].sum()
        else:
            # Fallback to length of results DataFrame (likely inaccurate for batched experiments)
            total_ideas = len(self._results_df)
        
        # Create summary statistics
        summary = {
            "Total Ideas": total_ideas,
            "Context Similarities": self._get_similarity_summary('context'),
            "Pairwise Similarities": self._get_similarity_summary('pairwise'),
            "Runtime": {
                "Seconds": runtime_seconds,
                "Minutes": runtime_seconds / 60,
                "Hours": runtime_seconds / 3600,
                "Format": self._format_runtime(runtime_seconds)
            }
        }
        
        # Save summary.json (quick overview)
        summary_path = os.path.join(experiment_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results.json
        results_data = {
            'results': self.current_experiment['results'],
            'summary': summary,  # Include summary in results for completeness
            'metadata': metadata  # Include metadata for context
        }
        results_path = os.path.join(experiment_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save CSV for easy analysis
        csv_path = os.path.join(experiment_dir, "results.csv")
        self._results_df.to_csv(csv_path, index=False)
        
        # Create and save dashboard
        try:
            print("\n[INFO] Creating experiment dashboard...")
            dashboard_path = self._create_dashboard_html(experiment_dir, metadata, summary)
            print("\n" + "="*50)
            print("EXPERIMENT RESULTS SAVED!")
            print("="*50)
            print(f"📁 Results Directory:")
            print(f"   {experiment_dir}")
            print(f"\n⏰ Experiment Timing:")
            start_time = datetime.fromisoformat(self.current_experiment['start_time'])
            end_time = datetime.fromisoformat(self.current_experiment['end_time'])
            duration = end_time - start_time
            print(f"   Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Runtime: {self._format_runtime(runtime_seconds)} ({runtime_seconds:.1f} seconds)")
            
            print("\n📊 Generated Files:")
            print(f"   1. metadata.json - Experiment configuration and timing")
            print(f"   2. summary.json  - Quick overview of metrics")
            print(f"   3. results.json  - Detailed results and analysis")
            print(f"   4. results.csv   - Results in CSV format")
            print(f"   5. dashboard.html - Interactive visualization")
            print("\n🌐 View Dashboard:")
            print(f"   File Protocol: file://{os.path.abspath(dashboard_path)}")
            web_path = os.path.relpath(dashboard_path, os.path.dirname(os.path.dirname(experiment_dir)))
            print(f"   HTTP Protocol: http://localhost:8000/{web_path}")
            print("\nQuick Start:")
            print("1. Start local server:  python -m http.server 8000")
            print("2. Open either URL in your browser")
            print("="*50 + "\n")
            
            # Print summary to console with both context and pairwise metrics
            print("\n=== Experiment Summary ===")
            print(f"Total Ideas Generated: {summary['Total Ideas']}")
            print(f"Runtime: {self._format_runtime(runtime_seconds)}")
            
            print("\nContext Similarities (with original text):")
            for metric, stats in summary["Context Similarities"].items():
                print(f"{metric}:")
                print(f"  Mean: {stats['mean']:.3f}")
                print(f"  Median: {stats['median']:.3f}")
                print(f"  Std: {stats['std']:.3f}")
            
            print("\nPairwise Similarities (between generated ideas):")
            for metric, stats in summary["Pairwise Similarities"].items():
                print(f"{metric}:")
                print(f"  Mean: {stats['mean']:.3f}")
                print(f"  Median: {stats['median']:.3f}")
                print(f"  Std: {stats['std']:.3f}")
            
        except Exception as e:
            print("\n" + "="*50)
            print("[ERROR] Dashboard Creation Failed!")
            print("="*50)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nOther result files were saved successfully:")
            print(f"📁 Results Directory: {experiment_dir}")
            print("   - metadata.json")
            print("   - summary.json")
            print("   - results.json")
            print("   - results.csv")
            print("\nStack trace:")
            import traceback
            print(traceback.format_exc())
            print("="*50 + "\n")
        
        # Reset experiment state
        self.current_experiment = None
        self._step = 0
        self._results_df = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current_experiment is not None:
            self.end_experiment()

    # =====================
    # Data Analysis Methods
    # =====================
    def _calculate_similarity_stats(self, metric_data: pd.Series) -> Dict[str, float]:
        """Helper function to calculate statistics for a similarity metric.
           The input Series is expected to contain one entry per run (log_result call),
           and that entry is now a list of raw scores for that run.
        """
        # metric_data is a Series. If one log_result call, it has one element: the list of scores.
        if metric_data.empty or metric_data.iloc[0] is None or not isinstance(metric_data.iloc[0], list) or not metric_data.iloc[0]:
            # Handle cases where the data is missing, not a list, or an empty list for the first (and only) run.
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0, # Or np.nan if preferred for undefined std of empty/single list
                "min": 0.0,
                "max": 0.0
            }

        # Extract the list of scores from the first (and typically only) element of the Series
        scores_list = metric_data.iloc[0]
        
        # Ensure all elements in the list are numeric, coercing errors
        data = pd.to_numeric(pd.Series(scores_list), errors='coerce').dropna()
        
        if len(data) > 0:
            return {
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()) if len(data) > 1 else 0.0, # Std dev is NaN for < 2 elements, return 0.0 or np.nan
                "min": float(data.min()),
                "max": float(data.max())
            }
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
        }

    def _calculate_kde(self, metric_data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate kernel density estimation for a given metric with adaptive bandwidth.
        
        Args:
            metric_data: Pandas Series containing the metric values
            
        Returns:
            Tuple of (x_values, y_values) for the KDE curve
        """
        # Clean the data
        data = pd.to_numeric(metric_data, errors='coerce').dropna()
        if len(data) < 2:
            return np.array([]), np.array([])
        
        # Calculate data range to adapt bandwidth
        data_range = data.max() - data.min()
        
        # Use adaptive bandwidth - wider for tightly clustered data
        bw_method = max(0.05, data_range)
        
        try:    
            # Calculate KDE with adaptive bandwidth
            kde = gaussian_kde(data, bw_method=bw_method)
            
            # Generate x values with higher density for smoother curve
            x_min = max(0, data.min() - max(0.05, data_range * 0.2))
            x_max = min(1, data.max() + max(0.05, data_range * 0.2))
            x_values = np.linspace(x_min, x_max, 200)  # More points for smoother curve
            
            # Calculate y values
            y_values = kde(x_values)
            
            return x_values, y_values
        except Exception as e:
            print(f"[WARN] KDE calculation error: {str(e)}")
            return np.array([]), np.array([])

    def _calculate_avg_pairwise_per_idea(self, num_ideas: int, pairwise_scores: List[float]) -> List[float]:
        """Calculates the average pairwise similarity for each idea against all others.

        Args:
            num_ideas: The total number of ideas generated.
            pairwise_scores: A flat list of pairwise scores, assumed to be in 
                             itertools.combinations order (e.g., (0,1), (0,2), ..., (0,N-1), (1,2), ...).

        Returns:
            A list of average scores, one per idea. Returns empty list if input is invalid.
        """
        if num_ideas < 2 or not pairwise_scores:
            return [0.0] * num_ideas

        expected_num_scores = num_ideas * (num_ideas - 1) // 2
        if len(pairwise_scores) != expected_num_scores:
            print(f"[WARN] _calculate_avg_pairwise_per_idea: Expected {expected_num_scores} scores for {num_ideas} ideas, but got {len(pairwise_scores)}.")
            
            # If we have fewer scores than expected, we'll use what we have and treat missing comparisons as zeros/NA
            # This is better than returning all zeros when we do have some valid scores
            if len(pairwise_scores) < expected_num_scores:
                print(f"[INFO] Will use available {len(pairwise_scores)} scores and impute NA for missing comparisons")
            else:
                print(f"[WARN] Extra scores will be ignored")
                pairwise_scores = pairwise_scores[:expected_num_scores]
        
        # Create a structure to hold scores involving each idea index
        # Using a list of lists: scores_per_idea[i] holds scores involving idea i
        scores_per_idea = [[] for _ in range(num_ideas)]
        
        # Map the flat list back to pairs
        # Try to map as many scores as possible, even if incomplete
        current_score_index = 0
        for i in range(num_ideas):
            for j in range(i + 1, num_ideas):
                if current_score_index < len(pairwise_scores):
                    score = pairwise_scores[current_score_index]
                    # Score applies to both idea i and idea j
                    scores_per_idea[i].append(score)
                    scores_per_idea[j].append(score)
                    current_score_index += 1
                else:
                    # We've run out of scores - leave the remaining comparisons as empty
                    break
            if current_score_index >= len(pairwise_scores):
                break
        
        # Calculate the average for each idea
        avg_scores = []
        for i in range(num_ideas):
            idea_scores = scores_per_idea[i]
            if idea_scores:
                # Convert to Series to handle NaN values properly
                idea_score_series = pd.to_numeric(pd.Series(idea_scores), errors='coerce').dropna()
                if not idea_score_series.empty:
                    avg_scores.append(float(idea_score_series.mean()))
                else:
                    avg_scores.append(0.0)
            else:
                # No scores for this idea
                avg_scores.append(0.0)
                
        return avg_scores

    def _get_similarity_summary(self, similarity_type: str) -> Dict[str, Dict[str, float]]:
        """Helper function to get summary statistics for either context or pairwise similarities."""
        metrics = {
            'context': ['context_cosine', 'context_self_bleu', 'context_bertscore'],
            'pairwise': ['avg_cosine_similarity', 'avg_self_bleu', 'avg_bertscore']
        }
        
        summary = {}
        for metric in metrics[similarity_type]:
            if metric in self._results_df.columns:
                display_name = metric.replace('context_', '').replace('avg_', '')
                summary[display_name] = self._calculate_similarity_stats(self._results_df[metric])
        
        return summary

    # =====================
    # Visualization Methods
    # =====================
    def _create_distribution_plot(self, metric_name: str) -> go.Figure:
        """Create a distribution plot for a given metric that shows all individual data points."""
        if self._results_df is None or self._results_df.empty or metric_name not in self._results_df.columns:
            return None
        
        # Get the list of scores from the first row of the DataFrame
        scores_list = []
        if not self._results_df[metric_name].empty:
            first_item = self._results_df[metric_name].iloc[0]
            if isinstance(first_item, list):
                scores_list = first_item
            elif pd.api.types.is_scalar(first_item) and pd.notna(first_item):
                scores_list = [float(first_item)]

        if not scores_list:
            print(f"[WARN] No data points found for metric '{metric_name}'")
            return None
        
        # Clean the data - remove None values and convert to float
        data = pd.to_numeric(pd.Series(scores_list), errors='coerce').dropna()
        
        if len(data) < 1:
            print(f"[WARN] No valid numeric data for metric '{metric_name}'")
            return None
        
        # Create a figure with a suitable layout    
        fig = go.Figure()
        
        # Calculate statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std() if len(data) > 1 else 0
        min_val = data.min()
        max_val = data.max()
        data_range = max_val - min_val
        
        # For extremely tight distributions, we need to adjust the scale
        is_tight_distribution = data_range < 0.05
        
        # Find appropriate x-axis range with padding
        if is_tight_distribution:
            # For extremely clustered data, create a wider view centered on the mean
            center = (min_val + max_val) / 2
            half_range = max(0.05, data_range * 5)
            x_min = max(0, center - half_range)
            x_max = min(1, center + half_range)
        else:
            # Normal padding
            x_padding = data_range * 0.2
            x_min = max(0, min_val - x_padding)
            x_max = min(1, max_val + x_padding)
        
        # Create histogram-like representation for the data points
        bin_size = max(0.001, data_range / 30) if data_range > 0 else 0.01
        bin_counts = {}
        for val in data:
            bin_key = round(val / bin_size) * bin_size
            bin_counts[bin_key] = bin_counts.get(bin_key, 0) + 1
        
        # Normalize bin heights
        max_bin_count = max(bin_counts.values()) if bin_counts else 1
        normalized_bins = {k: v / max_bin_count * 0.8 for k, v in bin_counts.items()}
        
        # Add scatter plot of individual points with jittered y values
        for val in data:
            bin_key = round(val / bin_size) * bin_size
            norm_height = normalized_bins[bin_key]
            y_pos = np.random.uniform(0, norm_height)
            
            fig.add_trace(go.Scatter(
                x=[val],
                y=[y_pos],
                mode='markers',
                marker=dict(
                    size=8,
                    color='rgba(50, 100, 200, 0.7)',
                    line=dict(color='black', width=1)
                ),
                hoverinfo='x',
                hovertemplate=f'Value: {val:.4f}<extra></extra>',
                showlegend=False
            ))
        
        # Only calculate KDE if we have enough points and distribution isn't too tight
        if len(data) >= 3 and (not is_tight_distribution or len(data) > 10):
            try:
                # Generate KDE curve - use custom bandwidth for tightly clustered data
                kde_bandwidth = max(0.01, data_range) if is_tight_distribution else None
                kde = gaussian_kde(data, bw_method=kde_bandwidth)
                
                # Generate more points for smoother curves
                x_kde = np.linspace(x_min, x_max, 200)
                y_kde = kde(x_kde)
                
                # Scale KDE y values for visibility (up to 1.0)
                y_kde_scaled = y_kde / np.max(y_kde) if np.max(y_kde) > 0 else y_kde
                
                # Add KDE curve
                fig.add_trace(go.Scatter(
                    x=x_kde,
                    y=y_kde_scaled,
                    mode='lines',
                    name='Density',
                    line=dict(color='blue', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0, 0, 255, 0.1)'
                ))
            except Exception as e:
                print(f"[WARN] KDE generation failed for {metric_name}: {str(e)}")
        else:
            # For very tight distributions, just add a vertical line to indicate the cluster
            if is_tight_distribution:
                fig.add_shape(
                    type="line",
                    x0=mean_val, x1=mean_val,
                    y0=0, y1=1,
                    line=dict(color="blue", width=2, dash="solid"),
                    opacity=0.5
                )

        # Add mean and median lines
        fig.add_trace(go.Scatter(
            x=[mean_val, mean_val],
            y=[0, 1],
            mode='lines',
            name=f'Mean: {mean_val:.4f}',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[median_val, median_val],
            y=[0, 1],
            mode='lines',
            name=f'Median: {median_val:.4f}',
            line=dict(color='green', dash='dash', width=2)
        ))
        
        # Add background color bands to show standard deviation range if data has variance
        if len(data) > 1 and std_val > 0.0001:
            # Add 1 std dev range
            fig.add_shape(
                type="rect",
                x0=mean_val-std_val, x1=mean_val+std_val,
                y0=0, y1=1,
                fillcolor="rgba(200,200,200,0.2)",
                line=dict(width=0),
                layer="below"
            )
        
        # Add text annotations for key statistics
        stats_text = (
            f"Mean: {mean_val:.4f}<br>"
            f"Median: {median_val:.4f}<br>"
            f"Std: {std_val:.4f}<br>"
            f"N={len(data)} points<br>"
            f"Range: [{min_val:.4f}, {max_val:.4f}]"
        )
        
        fig.add_annotation(
            x=x_min + (x_max - x_min) * 0.05,
            y=0.9,
            text=stats_text,
            showarrow=False,
            font=dict(color="black", size=12),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            align="left"
        )
        
        # Update layout
        base_metric_name = metric_name.replace("avg_", "").replace("context_", "Context ")
        plot_title = f"Distribution of {base_metric_name.replace('_', ' ').title()} Scores"
        xaxis_title = base_metric_name.replace('_', ' ').title()
        
        fig.update_layout(
            title=dict(text=plot_title, x=0.5, xanchor='center'),
            xaxis=dict(
                title=xaxis_title,
                range=[x_min, x_max],
                tickformat='.4f',
                gridcolor='rgba(200, 200, 200, 0.2)'
            ),
            yaxis=dict(
                title="Density",
                range=[0, 1.1],
                showticklabels=False,
                gridcolor='rgba(200, 200, 200, 0.2)'
            ),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25, # Adjusted y to give more space for x-axis title
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=80, b=100) # Increased bottom margin
        )
        
        return fig
        
    def _create_results_table(self) -> go.Figure:
        """Create an interactive table visualization of results, showing individual ideas."""
        if self._results_df is None or self._results_df.empty:
            return None

        # We expect one row in _results_df representing the entire experiment run's aggregated data (lists of scores).
        run_data = self._results_df.iloc[0]

        ideas_list = run_data.get('ideas', []) # Assuming 'ideas' key holds list of idea strings
        quality_evals_list = run_data.get('quality_scores', []) # List of dicts from HypothesisEvaluator
        
        # BUGFIX: Make sure quality_evals_list contains dictionaries rather than float values
        # This handles cases where quality scores might be simple numbers instead of dictionaries
        if quality_evals_list and not all(isinstance(item, dict) for item in quality_evals_list if item is not None):
            # Convert non-dictionary values to a standard format with evaluation and evaluation_full keys
            quality_evals_list = [
                item if isinstance(item, dict) else {'evaluation': str(item), 'evaluation_full': str(item)}
                for item in quality_evals_list
            ]
        
        # Context scores are per idea, matching the order in ideas_list
        # Fetch from the correct columns in run_data (self._results_df.iloc[0])
        # These columns ('context_cosine', etc.) should already contain lists of raw scores.
        ctx_cosine_scores_list = run_data.get('context_cosine', [])
        ctx_self_bleu_scores_list = run_data.get('context_self_bleu', [])
        ctx_bertscore_scores_list = run_data.get('context_bertscore', [])
        
        # Safeguard: Ensure they are lists, especially if only one idea was generated
        # and a single float might have been stored (though current pipeline should make them lists).
        if not isinstance(ctx_cosine_scores_list, list):
            ctx_cosine_scores_list = [ctx_cosine_scores_list] if pd.notna(ctx_cosine_scores_list) else []
        
        if not isinstance(ctx_self_bleu_scores_list, list):
            ctx_self_bleu_scores_list = [ctx_self_bleu_scores_list] if pd.notna(ctx_self_bleu_scores_list) else []
        
        if not isinstance(ctx_bertscore_scores_list, list):
            ctx_bertscore_scores_list = [ctx_bertscore_scores_list] if pd.notna(ctx_bertscore_scores_list) else []

        if not ideas_list:
            return None # No ideas to display

        num_ideas = len(ideas_list)

        # Prepare data for table cells
        idea_indices = [i + 1 for i in range(num_ideas)]
        eval_summaries = [quality_evals_list[i].get('evaluation', 'N/A') if i < len(quality_evals_list) and quality_evals_list[i] is not None else 'N/A' for i in range(num_ideas)]
        eval_full_texts = [quality_evals_list[i].get('evaluation_full', 'N/A').replace('\n', '<br>') if i < len(quality_evals_list) and quality_evals_list[i] is not None else 'N/A' for i in range(num_ideas)]
        
        context_similarity_strings = []
        for i in range(num_ideas):
            cos_sim = ctx_cosine_scores_list[i] if i < len(ctx_cosine_scores_list) and pd.notna(ctx_cosine_scores_list[i]) else 0.0
            bleu_sim = ctx_self_bleu_scores_list[i] if i < len(ctx_self_bleu_scores_list) and pd.notna(ctx_self_bleu_scores_list[i]) else 0.0
            bert_sim = ctx_bertscore_scores_list[i] if i < len(ctx_bertscore_scores_list) and pd.notna(ctx_bertscore_scores_list[i]) else 0.0
            context_similarity_strings.append(
                f"cos: {cos_sim:.3f}<br>" 
                f"bleu: {bleu_sim:.3f}<br>" 
                f"bert: {bert_sim:.3f}"
            )

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[
                    'Idea #', 'Generated Idea Text', 'Evaluation',
                    'Similarity to Original Context',
                    'Full Evaluation Text'
                ],
                fill_color='royalblue',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    idea_indices,
                    ideas_list,
                    eval_summaries,
                    context_similarity_strings,
                    eval_full_texts
                ],
                fill_color='lavender',
                align='left',
                height=30
            )
        )])
        
        fig.update_layout(
            title=dict(text="Detailed Experiment Results: Per Idea Analysis", x=0.5, xanchor='center'),
            height=max(400, 35 * num_ideas + 100), # Adjust height based on number of ideas
            margin=dict(t=50, l=10, r=10, b=10)
        )
        
        return fig

    def _create_heatmap(self, metric_col_name: str, plot_title_suffix: str) -> go.Figure:
        """Create a heatmap visualization for pairwise comparisons for a given metric."""
        if self._results_df is None or self._results_df.empty:
            print(f"[WARN] Heatmap: _results_df is None or empty.")
            return None

        run_data = self._results_df.iloc[0]
        num_ideas = run_data.get('num_ideas', 0)
        
        # The metric_col_name (e.g., 'avg_cosine_similarity') contains a flat list of pairwise scores
        pairwise_scores_flat = run_data.get(metric_col_name, [])

        if num_ideas < 2 or not isinstance(pairwise_scores_flat, list) or not pairwise_scores_flat:
            print(f"[WARN] Heatmap: Not enough ideas ({num_ideas}) or missing/invalid pairwise scores for '{metric_col_name}'.")
            return None
        
        # Check if we have the actual pairs used (if sampling occurred)
        # The 'pairwise_pairs' key should store the list of (i,j) tuples that were actually compared
        pairs_actually_compared = run_data.get('pairwise_pairs_compared', None)

        # Initialize an N x N matrix with NaNs (or a value indicating no comparison)
        similarity_matrix = np.full((num_ideas, num_ideas), np.nan)
        
        # Populate the diagonal with 1.0 (or a distinct value for self-similarity)
        for i in range(num_ideas):
            similarity_matrix[i, i] = 1.0 

        if pairs_actually_compared:
            # If we have the specific pairs that were compared (due to sampling)
            if len(pairs_actually_compared) != len(pairwise_scores_flat):
                print(f"[WARN] Heatmap: Mismatch between number of compared pairs ({len(pairs_actually_compared)}) and scores ({len(pairwise_scores_flat)}) for '{metric_col_name}'. Heatmap may be incomplete.")
                # Proceed with caution, might lead to errors if lengths truly mismatch logic
            
            for k, (i, j) in enumerate(pairs_actually_compared):
                if k < len(pairwise_scores_flat):
                    score = pairwise_scores_flat[k]
                    similarity_matrix[i, j] = score
                    similarity_matrix[j, i] = score # Symmetric matrix
                else:
                    # This case implies more pairs than scores, which is problematic
                    break 
        else:
            # If no specific pairs list, assume all pairs were compared in itertools.combinations order
            expected_num_scores = num_ideas * (num_ideas - 1) // 2
            if len(pairwise_scores_flat) != expected_num_scores:
                print(f"[WARN] Heatmap: Expected {expected_num_scores} scores for {num_ideas} ideas (full comparison), but got {len(pairwise_scores_flat)} for '{metric_col_name}'. Heatmap may be incomplete.")
                # We'll fill what we can

            current_score_index = 0
            for i in range(num_ideas):
                for j in range(i + 1, num_ideas):
                    if current_score_index < len(pairwise_scores_flat):
                        score = pairwise_scores_flat[current_score_index]
                        similarity_matrix[i, j] = score
                        similarity_matrix[j, i] = score # Symmetric matrix
                        current_score_index += 1
                    else:
                        # Ran out of scores, remaining pairs in matrix will be NaN
                        break
                if current_score_index >= len(pairwise_scores_flat):
                    break

        idea_labels = [f"Idea {i+1}" for i in range(num_ideas)]

        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=idea_labels,
            y=idea_labels,
            colorscale='Viridis', 
            zmin=0, 
            zmax=1,
            text=np.around(similarity_matrix, decimals=3).astype(str),
            hoverongaps=False, # Do not show hover for NaN values
            hovertemplate='<b>Idea X</b>: %{y}<br>' +
                          '<b>Idea Y</b>: %{x}<br>' +
                          '<b>Similarity</b>: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=f"Pairwise {plot_title_suffix} Similarity Heatmap", x=0.5, xanchor='center'),
            xaxis_title="Compared Idea",
            yaxis_title="Reference Idea",
            xaxis_side="top", # Move x-axis labels to top for standard matrix view
            width=min(800, 40 * num_ideas + 200), # Dynamic width
            height=min(700, 40 * num_ideas + 150), # Dynamic height
            margin=dict(t=100, b=50, l=100, r=50)
        )
        
        return fig

    def _create_context_similarity_plot(self) -> go.Figure:
        """Create a plot showing similarities with original context over time."""
        if self._results_df is None or len(self._results_df) == 0:
            return None
            
        fig = go.Figure()
        
        # metrics_to_plot maps the column name (which contains a list of scores per run)
        # to the display name for the legend.
        metrics_to_plot = {
            'context_cosine': 'Context Cosine Sim.',
            'context_self_bleu': 'Context Self-BLEU',
            'context_bertscore': 'Context BERTScore'
        }
        
        # Assuming self._results_df has one row per experiment run/log_result call.
        # For each metric, we extract the list of scores, calculate its mean, and plot that mean.
        # If there were multiple rows (multiple runs), this would plot a mean for each run over time.
        
        for col, name in metrics_to_plot.items():
            if col in self._results_df.columns:
                # Define a helper to calculate mean from a list in a cell, defaulting to np.nan
                def get_mean_from_list_cell(cell_value):
                    if isinstance(cell_value, list) and cell_value:
                        numeric_scores = pd.to_numeric(pd.Series(cell_value), errors='coerce').dropna()
                        return numeric_scores.mean() if not numeric_scores.empty else np.nan
                    return np.nan
                
                # Apply the helper to the column to get a Series of means (or NaNs)
                y_values = self._results_df[col].apply(get_mean_from_list_cell)
                
                if not y_values.dropna().empty:
                    fig.add_trace(go.Scatter(
                        x=self._results_df['run_id'], # Or step, or a proper time series index
                        y=y_values,
                        name=name,
                        mode='lines+markers'
                    ))
        
        fig.update_layout(
            title="Average Similarity to Original Context per Run",
            xaxis_title="Idea",
            yaxis_title="Similarity Score",
            yaxis_range=[0, 1],
            showlegend=True
        )
        
        return fig
        
    def _create_similarity_scatter_plot(self, context_metric_col: str = 'context_cosine', 
                                        pairwise_metric_col: str = 'avg_cosine_similarity',
                                        plot_title_suffix: str = 'Cosine') -> go.Figure:
        """Create a scatter plot comparing context similarity vs. average pairwise similarity for each idea."""
        if self._results_df is None or self._results_df.empty:
            return None
        
        # Check if we have at least some data to plot
        has_context_data = context_metric_col in self._results_df.columns
        has_pairwise_data = pairwise_metric_col in self._results_df.columns
        has_ideas_data = 'ideas' in self._results_df.columns
        
        if not (has_context_data or has_pairwise_data):
            print(f"[WARN] Missing both context and pairwise data columns for scatter plot.")
            return None
        
        # Get data from first row 
        if len(self._results_df) > 0:
            run_data = self._results_df.iloc[0]
        else:
            # This case should ideally not be reached if the initial checks pass,
            # but it's a safeguard.
            print(f"[WARN] Scatter plot: _results_df is empty despite initial checks.")
            return None
        
        # Try to get number of ideas and idea texts
        num_ideas = run_data.get('num_ideas', 0)
        ideas_list = run_data.get('ideas', [])
        
        # If num_ideas is 0 but we have ideas list, use its length
        if num_ideas == 0 and ideas_list:
            num_ideas = len(ideas_list)
        # If we have num_ideas but no ideas list, create placeholder texts
        elif num_ideas > 0 and not ideas_list:
            ideas_list = [f"Idea {i+1}" for i in range(num_ideas)]
        
        # If we still have no ideas, check if we can infer from metrics
        if num_ideas == 0:
            # Try to infer from context metric length if available
            if has_context_data:
                context_scores = run_data.get(context_metric_col, [])
                if isinstance(context_scores, list) and context_scores:
                    num_ideas = len(context_scores)
                    ideas_list = [f"Idea {i+1}" for i in range(num_ideas)]
        
        # If we still have no ideas to plot, we can't create the scatter plot
        if num_ideas == 0:
            print(f"[WARN] Could not determine number of ideas for scatter plot.")
            return None
        
        # Prepare x-axis data (context scores)
        x_values_raw = []
        if has_context_data:
            # Correctly fetch the list of context scores for the specified metric column
            # e.g., context_metric_col could be 'context_cosine' or 'context_bertscore'
            context_scores_for_metric = run_data.get(context_metric_col, [])
            if isinstance(context_scores_for_metric, list) and context_scores_for_metric:
                x_values_raw = context_scores_for_metric
        
        # Convert to numeric, coercing errors. This will be a pd.Series.
        x_values = pd.to_numeric(pd.Series(x_values_raw), errors='coerce')

        # If no context data but we know num_ideas, use dummy values and note in title
        if x_values.empty and num_ideas > 0:
            x_values = pd.Series([np.nan] * num_ideas) # Use NaNs so they don't plot if other axis has data
            context_available = False
        else:
            context_available = not x_values.isnull().all()

        # Prepare y-axis data (pairwise scores)
        y_values_raw = []
        if has_pairwise_data:
            # pairwise_metric_col is e.g. 'avg_cosine_similarity' which holds the flat list of ALL pairwise scores
            pairwise_scores_flat = run_data.get(pairwise_metric_col, [])
            if isinstance(pairwise_scores_flat, list) and pairwise_scores_flat and num_ideas > 0:
                # This function calculates the average pairwise score FOR EACH IDEA
                avg_pairwise_per_idea = self._calculate_avg_pairwise_per_idea(num_ideas, pairwise_scores_flat)
                y_values_raw = avg_pairwise_per_idea
            
        # Convert to numeric, coercing errors. This will be a pd.Series.
        y_values = pd.to_numeric(pd.Series(y_values_raw), errors='coerce')
        
        # If no pairwise data but we know num_ideas, use dummy values and note in title
        if y_values.empty and num_ideas > 0:
            y_values = pd.Series([np.nan] * num_ideas) # Use NaNs
            pairwise_available = False
        else:
            pairwise_available = not y_values.isnull().all()

        # Ensure x_values and y_values are Series of the same length as num_ideas, padding with NaN if necessary
        # This is crucial if one metric was partially available.
        if len(x_values) < num_ideas:
            x_values = pd.concat([x_values, pd.Series([np.nan]*(num_ideas - len(x_values)))], ignore_index=True)
        if len(y_values) < num_ideas:
            y_values = pd.concat([y_values, pd.Series([np.nan]*(num_ideas - len(y_values)))], ignore_index=True)

        # Prepare hover text
        if has_ideas_data and ideas_list and len(ideas_list) == num_ideas:
            hover_texts = [f"Idea {i+1}: {ideas_list[i][:50]}..." if i < len(ideas_list) else f"Idea {i+1}" 
                           for i in range(min(len(x_values), len(y_values)))]
        else:
            hover_texts = [f"Idea {i+1}" for i in range(min(len(x_values), len(y_values)))]

        # Create the figure with available data
        fig = go.Figure(data=go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            marker=dict(
                size=12, 
                color='rgba(50, 100, 200, 0.7)',
                line=dict(width=1, color='black')
            ),
            text=hover_texts,
            hoverinfo='x+y+text'
        ))
        
        # Adjust title based on available data
        scatter_plot_title = f'Similarity Analysis'
        if context_available and pairwise_available:
            scatter_plot_title = f'Context vs. Pairwise {plot_title_suffix} Similarity'
        elif context_available:
            scatter_plot_title = f'Distribution of Context {plot_title_suffix} Scores'
        elif pairwise_available:
            scatter_plot_title = f'Distribution of Pairwise {plot_title_suffix} Scores'
        
        # Set axis titles based on available data
        x_title = f'Similarity to Context ({plot_title_suffix})' if context_available else 'Position'
        y_title = f'Avg Pairwise {plot_title_suffix} Similarity' if pairwise_available else 'Idea Index'

        fig.update_layout(
            title=dict(text=scatter_plot_title, x=0.5, xanchor='center'),
            xaxis_title=x_title,
            yaxis_title=y_title,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(t=80, b=60, l=60, r=40)
        )
        
        # Add a note if using placeholder data
        if not (context_available and pairwise_available):
            fig.add_annotation(
                x=0.5, y=0.05,
                text="Note: Some data unavailable, showing partial visualization",
                showarrow=False,
                font=dict(color="red", size=12)
            )

        return fig

    def _create_dashboard_html(self, experiment_dir: str, metadata: dict, summary: dict):
        """Create a comprehensive HTML dashboard."""
        dashboard_path = os.path.join(experiment_dir, "dashboard.html")
        
        # Format runtime for display
        runtime_display = summary.get("Runtime", {}).get("Format", "N/A")
        runtime_seconds = summary.get("Runtime", {}).get("Seconds", 0)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Results - {metadata['name']}</title>
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
                h1, h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .metadata {{
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
                .plot-description {{
                    font-size: 0.9em;
                    color: #555;
                    margin-top: 5px;
                    margin-bottom: 15px;
                    text-align: center;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2980b9;
                    margin: 10px 0;
                }}
                .stat-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                }}
                .summary-metrics {{
                    margin-bottom: 20px;
                }}
                .metric-tables-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .metric-table {{
                    flex: 1;
                    min-width: 300px;
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .metric-table h3 {{
                    margin-top: 0;
                    color: #34495e;
                }}
                .metric-table table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .metric-table th, .metric-table td {{
                    text-align: left;
                    padding: 8px;
                    border-bottom: 1px solid #ddd;
                }}
                .metric-table th {{
                    background-color: #e9ecef;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Experiment Results Dashboard</h1>
                
                <div class="metadata">
                    <h2>Experiment Details</h2>
                    <p><strong>Name:</strong> {metadata['name']}</p>
                    <p><strong>Type:</strong> {metadata['type']}</p>
                    <p><strong>Model:</strong> {metadata['model']}</p>
                    <p><strong>Start Time:</strong> {metadata['start_time']}</p>
                    <p><strong>End Time:</strong> {metadata['end_time']}</p>
                    <p><strong>Runtime:</strong> {runtime_display}</p>
                </div>

                <!-- Section 1: Key Statistics -->
                <h2>Key Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Ideas Generated</div>
                        <div class="stat-value">{summary['Total Ideas']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Runtime</div>
                        <div class="stat-value">{runtime_display}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Ideas per Minute</div>
                        <div class="stat-value">{(summary['Total Ideas'] / (runtime_seconds / 60)):.1f}</div>
                    </div>
                </div>

                <!-- Section 2: Overall Run Summary -->
                <div class="summary-metrics">
                     <h2>Overall Run Metrics</h2>
                     <div class="metric-tables-container">
                        <div class="metric-table">
                            <h3>Context Similarity (vs. Original Paper)</h3>
                            <table>
                                <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std</th></tr>
                                <tr>
                                    <td>Cosine</td>
                                    <td>{summary.get('Context Similarities', {}).get('cosine', {}).get('mean', 0.0):.3f}</td>
                                    <td>{summary.get('Context Similarities', {}).get('cosine', {}).get('median', 0.0):.3f}</td>
                                    <td>{summary.get('Context Similarities', {}).get('cosine', {}).get('std', 0.0):.3f}</td>
                                </tr>
                                <tr>
                                    <td>Self-BLEU</td>
                                    <td>{summary.get('Context Similarities', {}).get('self_bleu', {}).get('mean', 0.0):.3f}</td>
                                    <td>{summary.get('Context Similarities', {}).get('self_bleu', {}).get('median', 0.0):.3f}</td>
                                    <td>{summary.get('Context Similarities', {}).get('self_bleu', {}).get('std', 0.0):.3f}</td>
                                </tr>
                                <tr>
                                    <td>BERTScore</td>
                                    <td>{summary.get('Context Similarities', {}).get('bertscore', {}).get('mean', 0.0):.3f}</td>
                                    <td>{summary.get('Context Similarities', {}).get('bertscore', {}).get('median', 0.0):.3f}</td>
                                    <td>{summary.get('Context Similarities', {}).get('bertscore', {}).get('std', 0.0):.3f}</td>
                                </tr>
                            </table>
                        </div>
                        <div class="metric-table">
                            <h3>Pairwise Similarity (Between Generated Ideas)</h3>
                            <table>
                                <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std</th></tr>
                                <tr>
                                    <td>Cosine</td>
                                    <td>{summary.get('Pairwise Similarities', {}).get('cosine_similarity', {}).get('mean', 0.0):.3f}</td>
                                    <td>{summary.get('Pairwise Similarities', {}).get('cosine_similarity', {}).get('median', 0.0):.3f}</td>
                                    <td>{summary.get('Pairwise Similarities', {}).get('cosine_similarity', {}).get('std', 0.0):.3f}</td>
                                </tr>
                                <tr>
                                    <td>Self-BLEU</td>
                                    <td>{summary.get('Pairwise Similarities', {}).get('self_bleu', {}).get('mean', 0.0):.3f}</td>
                                    <td>{summary.get('Pairwise Similarities', {}).get('self_bleu', {}).get('median', 0.0):.3f}</td>
                                    <td>{summary.get('Pairwise Similarities', {}).get('self_bleu', {}).get('std', 0.0):.3f}</td>
                                </tr>
                                <tr>
                                    <td>BERTScore</td>
                                    <td>{summary.get('Pairwise Similarities', {}).get('bertscore', {}).get('mean', 0.0):.3f}</td>
                                    <td>{summary.get('Pairwise Similarities', {}).get('bertscore', {}).get('median', 0.0):.3f}</td>
                                    <td>{summary.get('Pairwise Similarities', {}).get('bertscore', {}).get('std', 0.0):.3f}</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- Section 3: Idea Details Table -->
                <h2>Generated Idea Details</h2>
                <div class="plot-container" id="results-table"></div>

                <!-- Section 4: Similarity Analysis Scatter Plot -->
                <h2>Context vs. Pairwise Similarity Analysis</h2>
                <div class="plot-grid">
                    <div class="plot-container" id="similarity-scatter-cosine"></div>
                    <div class="plot-container" id="similarity-scatter-bertscore"></div>
                </div>

                <!-- Section 5: Metric Distributions -->
                <h2>Metric Distributions</h2>
                <div class="plot-grid">
                    <div class="plot-container">
                        <div id="context-cosine-dist"></div>
                        <p class="plot-description">Distribution of semantic similarity scores (Cosine) between each generated idea and the original input context. Higher scores indicate closer relevance to the source.</p>
                    </div>
                    <div class="plot-container">
                        <div id="context-bleu-dist"></div>
                        <p class="plot-description">Distribution of n-gram overlap scores (Self-BLEU) between each generated idea and the original input context. Higher scores mean more shared phrases with the source.</p>
                    </div>
                    <div class="plot-container">
                        <div id="context-bert-dist"></div>
                        <p class="plot-description">Distribution of semantic similarity scores (BERTScore) between each generated idea and the original input context. Higher scores indicate closer semantic meaning to the source.</p>
                    </div>
                    <div class="plot-container">
                        <div id="pairwise-cosine-dist"></div>
                        <p class="plot-description">Distribution of semantic similarity scores (Cosine) between all unique pairs of generated ideas. Higher scores indicate generated ideas are similar to each other (less diversity).</p>
                    </div>
                    <div class="plot-container">
                        <div id="pairwise-bleu-dist"></div>
                        <p class="plot-description">Distribution of n-gram overlap scores (Self-BLEU) between all unique pairs of generated ideas. Higher scores indicate generated ideas share more common phrases (less diversity).</p>
                    </div>
                    <div class="plot-container">
                        <div id="pairwise-bert-dist"></div>
                        <p class="plot-description">Distribution of semantic similarity scores (BERTScore) between all unique pairs of generated ideas. Higher scores indicate generated ideas are semantically similar to each other (less diversity).</p>
                    </div>
                </div>
                
                <!-- New Heatmap Section -->
                <h2>Pairwise Similarity Heatmaps</h2>
                <p class="plot-description">These heatmaps visualize the similarity between each pair of generated ideas. Darker cells indicate higher similarity. The diagonal is always 1.0 (idea compared to itself).</p>
                <div class="plot-grid">
                    <div class="plot-container" id="cosine-heatmap"></div>
                    <div class="plot-container" id="self-bleu-heatmap"></div>
                    <div class="plot-container" id="bertscore-heatmap"></div>
                </div>

            </div>
            <script>
        """
        
        # Add the table plot
        table_fig = self._create_results_table()
        if table_fig:
            html_content += f"Plotly.newPlot('results-table', {table_fig.to_json()});\n"
        
        # Add distribution plots with KDE
        # Use the column names that actually contain the lists of scores in _results_df
        metric_cols_for_kde = {
            # Context Metrics
            'context_cosine': 'context-cosine-dist',
            'context_self_bleu': 'context-bleu-dist',
            'context_bertscore': 'context-bert-dist',
            # Pairwise Metrics
            'avg_cosine_similarity': 'pairwise-cosine-dist',
            'avg_self_bleu': 'pairwise-bleu-dist',
            'avg_bertscore': 'pairwise-bert-dist'
        }
        for metric_col_name, div_id in metric_cols_for_kde.items():
            if metric_col_name in self._results_df.columns:
                # Pass the correct DataFrame column name to the plot function
                dist_fig = self._create_distribution_plot(metric_col_name)
                if dist_fig:
                    # Update title dynamically based on metric type
                    plot_title = f"Distribution: {metric_col_name.replace('_', ' ').replace('avg', 'Pairwise').replace('similarity', 'Sim.').title()}"
                    dist_fig.update_layout(title=plot_title)
                    html_content += f"Plotly.newPlot('{div_id}', {dist_fig.to_json()});\n"
                # else: # Optional: add placeholder if plot fails
                    # html_content += f'<p>Could not generate distribution plot for {metric_col_name}.</p>\n'
        
        # Add context similarity plot (shows average per run)
        context_fig = self._create_context_similarity_plot()
        if context_fig:
            html_content += f"Plotly.newPlot('context-similarity', {context_fig.to_json()});\n"
            
        # Add the new Similarity Scatter Plots
        scatter_cosine_fig = self._create_similarity_scatter_plot(
            context_metric_col='context_cosine',
            pairwise_metric_col='avg_cosine_similarity',
            plot_title_suffix='Cosine'
        )
        if scatter_cosine_fig:
            html_content += f"Plotly.newPlot('similarity-scatter-cosine', {scatter_cosine_fig.to_json()});\n"

        scatter_bert_fig = self._create_similarity_scatter_plot(
            context_metric_col='context_bertscore',
            pairwise_metric_col='avg_bertscore',
            plot_title_suffix='BERTScore'
        )
        if scatter_bert_fig:
            html_content += f"Plotly.newPlot('similarity-scatter-bertscore', {scatter_bert_fig.to_json()});\n"

        # Add Heatmap plots
        heatmap_cosine_fig = self._create_heatmap(metric_col_name='avg_cosine_similarity', plot_title_suffix='Cosine')
        if heatmap_cosine_fig:
            html_content += f"Plotly.newPlot('cosine-heatmap', {heatmap_cosine_fig.to_json()});\n"
        
        heatmap_bleu_fig = self._create_heatmap(metric_col_name='avg_self_bleu', plot_title_suffix='Self-BLEU')
        if heatmap_bleu_fig:
            html_content += f"Plotly.newPlot('self-bleu-heatmap', {heatmap_bleu_fig.to_json()});\n"

        heatmap_bertscore_fig = self._create_heatmap(metric_col_name='avg_bertscore', plot_title_suffix='BERTScore')
        if heatmap_bertscore_fig:
            html_content += f"Plotly.newPlot('bertscore-heatmap', {heatmap_bertscore_fig.to_json()});\n"
        
        html_content += """
            </script>
        </body>
        </html>
        """
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return dashboard_path

    def _format_runtime(self, seconds: float) -> str:
        """Format seconds into a human-readable duration string."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{int(minutes)} minutes {int(seconds % 60)} seconds"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{int(hours)} hours {int(minutes)} minutes"

