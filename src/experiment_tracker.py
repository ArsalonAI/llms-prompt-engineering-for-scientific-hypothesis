import os
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde

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
        self._experiment_type = experiment_type
        self._step = 0
        
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
            'has_kde_data': bool
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
        for key in ['quality_scores', 'has_kde_data', 'idea', 'batch_prompt', 'evaluation', 'evaluation_full']:
            if key in result: # 'idea', 'batch_prompt', etc. might not be in a typical result dict for log_result
                df_row[key] = result.get(key)
            elif key == 'has_kde_data': # Ensure this column is always present
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
        
        self.current_experiment['end_time'] = datetime.now().isoformat()
        
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
            'end_time': self.current_experiment['end_time']
        }
        metadata_path = os.path.join(experiment_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create summary statistics
        summary = {
            "Total Ideas": len(self._results_df),
            "Context Similarities": self._get_similarity_summary('context'),
            "Pairwise Similarities": self._get_similarity_summary('pairwise')
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
            print(f"   Duration: {duration.total_seconds():.1f} seconds")
            
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
        """Calculate kernel density estimation for a given metric.
        
        Args:
            metric_data: Pandas Series containing the metric values
            
        Returns:
            Tuple of (x_values, y_values) for the KDE curve
        """
        # Clean the data
        data = pd.to_numeric(metric_data, errors='coerce').dropna()
        if len(data) == 0:
            return np.array([]), np.array([])
            
        # Calculate KDE
        kde = gaussian_kde(data)
        
        # Generate x values for the plot
        x_min = max(0, data.min() - 0.1)
        x_max = min(1, data.max() + 0.1)
        x_values = np.linspace(x_min, x_max, 100)
        
        # Calculate y values
        y_values = kde(x_values)
        
        return x_values, y_values

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
        """Create a distribution plot for a given metric."""
        if self._results_df is None or self._results_df.empty or metric_name not in self._results_df.columns:
            return None
            
        # metric_name refers to a column in _results_df (e.g., 'avg_cosine_similarity')
        # This column now contains a list of scores in its first (and only) row.
        # Or, for older compatibility, it might be a column like 'cosine_similarities' that holds the list.

        scores_list = []
        if not self._results_df[metric_name].empty:
            first_item = self._results_df[metric_name].iloc[0]
            if isinstance(first_item, list):
                scores_list = first_item
            elif pd.api.types.is_scalar(first_item) and pd.notna(first_item):
                # Fallback if by some chance a scalar is logged - treat as single score
                # Though _calculate_kde expects a series that can have variance.
                # This case is unlikely with current logic but makes it robust.
                scores_list = [first_item]
            # If it's np.nan or other, scores_list remains empty.

        if not scores_list:
            # print(f"[Debug] No scores_list found or empty for metric '{metric_name}' in _create_distribution_plot")
            return None
            
        # Clean the data - remove None values and convert to float
        data = pd.to_numeric(pd.Series(scores_list), errors='coerce').dropna()
        
        if len(data) < 2: # KDE needs at least 2 points to estimate density meaningfully (ideally more)
            # print(f"[Debug] Not enough data points ({len(data)}) for KDE for metric '{metric_name}'")
            return None # Or create a simple histogram without KDE for <2 points
            
        # Create histogram
        hist = go.Histogram(x=data, nbinsx=30, name=metric_name)
        fig = go.Figure(data=[hist])
        
        # Get histogram y values
        hist_data = fig.data[0]
        max_y = 1  # Default height
        if hist_data.y is not None and len(hist_data.y) > 0:
            max_y = max(hist_data.y)
        
        # Calculate statistics
        mean_val = data.mean()
        median_val = data.median()
        
        # Add mean and median lines
        fig.add_trace(go.Scatter(
            x=[mean_val, mean_val],
            y=[0, max_y],
            mode='lines',
            name='Mean',
            line=dict(color='red', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=[median_val, median_val],
            y=[0, max_y],
            mode='lines',
            name='Median',
            line=dict(color='green', dash='dash')
        ))
        
        # Calculate and add KDE
        # The _calculate_kde method itself expects a Series of numeric data
        x_kde, y_kde = self._calculate_kde(data) # Pass the cleaned Series of scores
        if len(x_kde) > 0:
            # Normalize KDE to match histogram scale
            y_kde = y_kde * (max_y / y_kde.max())
            fig.add_trace(go.Scatter(
                x=x_kde,
                y=y_kde,
                mode='lines',
                name='KDE',
                line=dict(color='blue', width=2)
            ))
        
        fig.update_layout(
            title=f'{metric_name} Distribution',
            xaxis_title=metric_name,
            yaxis_title='Count',
            showlegend=True
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
        
        # Context scores are per idea, matching the order in ideas_list
        ctx_cosine_scores = run_data.get('context_cosine_scores_raw', [np.nan] * len(ideas_list))
        ctx_self_bleu_scores = run_data.get('context_self_bleu_scores_raw', [np.nan] * len(ideas_list))
        ctx_bertscore_scores = run_data.get('context_bertscore_scores_raw', [np.nan] * len(ideas_list))

        if not ideas_list:
            return None # No ideas to display

        num_ideas = len(ideas_list)

        # Prepare data for table cells
        idea_indices = [i + 1 for i in range(num_ideas)]
        eval_summaries = [quality_evals_list[i].get('evaluation', 'N/A') if i < len(quality_evals_list) else 'N/A' for i in range(num_ideas)]
        eval_full_texts = [quality_evals_list[i].get('evaluation_full', 'N/A').replace('\n', '<br>') if i < len(quality_evals_list) else 'N/A' for i in range(num_ideas)]
        
        context_similarity_strings = []
        for i in range(num_ideas):
            cos_sim = ctx_cosine_scores[i] if i < len(ctx_cosine_scores) and pd.notna(ctx_cosine_scores[i]) else 0.0
            bleu_sim = ctx_self_bleu_scores[i] if i < len(ctx_self_bleu_scores) and pd.notna(ctx_self_bleu_scores[i]) else 0.0
            bert_sim = ctx_bertscore_scores[i] if i < len(ctx_bertscore_scores) and pd.notna(ctx_bertscore_scores[i]) else 0.0
            context_similarity_strings.append(
                f"cos: {cos_sim:.3f}<br>" 
                f"bleu: {bleu_sim:.3f}<br>" 
                f"bert: {bert_sim:.3f}"
            )

        # For pairwise, we can show the average pairwise for the whole batch, or it gets complex per idea here.
        # The current dashboard structure shows overall pairwise in a separate summary.
        # For simplicity, we can omit per-idea pairwise from this table, or show the overall batch avg.
        # Let's fetch the overall average pairwise scores (which are lists of scores in run_data)
        # and display their mean here for context, though it's not per-idea.
        # This is just for display in this table, true summary stats are elsewhere.
        
        # avg_pairwise_cosine_list = run_data.get('avg_cosine_similarity', [])
        # avg_pairwise_bleu_list = run_data.get('avg_self_bleu', [])
        # avg_pairwise_bert_list = run_data.get('avg_bertscore', [])

        # mean_pairwise_cos = np.mean(pd.to_numeric(pd.Series(avg_pairwise_cosine_list), errors='coerce').dropna()) if avg_pairwise_cosine_list else 0.0
        # mean_pairwise_bleu = np.mean(pd.to_numeric(pd.Series(avg_pairwise_bleu_list), errors='coerce').dropna()) if avg_pairwise_bleu_list else 0.0
        # mean_pairwise_bert = np.mean(pd.to_numeric(pd.Series(avg_pairwise_bert_list), errors='coerce').dropna()) if avg_pairwise_bert_list else 0.0
        
        # pairwise_similarity_strings = [
        #     f"cos: {mean_pairwise_cos:.3f}<br>bleu: {mean_pairwise_bleu:.3f}<br>bert: {mean_pairwise_bert:.3f}"
        # ] * num_ideas # Repeat for each row, as it's a batch summary

        # Simpler: Let's remove pairwise from this per-idea table for now to avoid confusion.
        # Pairwise stats are better in the summary table or dedicated pairwise plots.

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[
                    'Idea #', 'Generated Idea Text', 'Evaluation',
                    'Similarity to Original Context',
                    # 'Avg. Pairwise Similarity (Batch)', # Removed for clarity
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
                    # pairwise_similarity_strings, # Removed for clarity
                    eval_full_texts
                ],
                fill_color='lavender',
                align='left',
                height=30
            )
        )])
        
        fig.update_layout(
            title="Detailed Experiment Results: Per Idea",
            height=max(400, 35 * num_ideas + 100), # Adjust height based on number of ideas
            margin=dict(t=50, l=10, r=10, b=10)
        )
        
        return fig

    def _create_heatmap(self, metric_name: str) -> go.Figure:
        """Create a heatmap visualization for pairwise comparisons."""
        if self._results_df is None or len(self._results_df) == 0:
            return None
            
        # Extract pairwise comparisons
        matrix = []
        run_ids = []
        
        for _, row in self._results_df.iterrows():
            run_ids.append(row['run_id'])
            row_values = []
            
            # Get pairwise similarities for this idea
            pairwise = row.get('pairwise_similarities', {}).get(metric_name, [])
            
            # Create row in matrix
            for other_id in run_ids[:-1]:  # Exclude current run_id
                score = next(
                    (item['score'] for item in pairwise if item['compared_to'] == other_id),
                    None
                )
                row_values.append(score if score is not None else 0.0)
            
            # Add 1.0 for self-comparison
            row_values.append(1.0)
            
            # Pad with zeros for future comparisons
            row_values.extend([0.0] * (len(self._results_df) - len(row_values)))
            
            matrix.append(row_values)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=run_ids,
            y=run_ids,
            colorscale='Viridis',
            zmin=0,
            zmax=1
        ))
        
        fig.update_layout(
            title=f"Pairwise {metric_name.replace('_', ' ').title()} Comparison",
            xaxis_title="Idea",
            yaxis_title="Idea",
            width=600,
            height=600
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
        
    def _create_dashboard_html(self, experiment_dir: str, metadata: dict, summary: dict):
        """Create a comprehensive HTML dashboard."""
        dashboard_path = os.path.join(experiment_dir, "dashboard.html")
        
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
                </div>

                <h2>Results Table</h2>
                <div class="plot-container" id="results-table"></div>
                
                <h2>Metric Distributions with KDE</h2>
                <div class="plot-grid">
                    <div class="plot-container" id="cosine-sim-dist"></div>
                    <div class="plot-container" id="self-bleu-dist"></div>
                    <div class="plot-container" id="bertscore-dist"></div>
                </div>
                
                <h2>Context Similarity Over Time</h2>
                <div class="plot-container" id="context-similarity"></div>
                
                <h2>Pairwise Similarity Heatmaps</h2>
                <div class="plot-grid">
                    <div class="plot-container" id="cosine-heatmap"></div>
                    <div class="plot-container" id="bleu-heatmap"></div>
                    <div class="plot-container" id="bert-heatmap"></div>
                </div>
            </div>
            <script>
        """
        
        # Add the table plot
        table_fig = self._create_results_table()
        if table_fig:
            html_content += f"Plotly.newPlot('results-table', {table_fig.to_json()});\n"
        
        # Add distribution plots with KDE
        for metric, div_id in [
            ('cosine_similarities', 'cosine-sim-dist'),
            ('self_bleu_scores', 'self-bleu-dist'),
            ('bertscore_scores', 'bertscore-dist')
        ]:
            if metric in self._results_df.columns:
                dist_fig = self._create_distribution_plot(metric)
                if dist_fig:
                    html_content += f"Plotly.newPlot('{div_id}', {dist_fig.to_json()});\n"
        
        # Add context similarity plot
        context_fig = self._create_context_similarity_plot()
        if context_fig:
            html_content += f"Plotly.newPlot('context-similarity', {context_fig.to_json()});\n"
        
        # Add heatmaps
        for metric, div_id in [
            ('cosine', 'cosine-heatmap'),
            ('self_bleu', 'bleu-heatmap'),
            ('bertscore', 'bert-heatmap')
        ]:
            heatmap = self._create_heatmap(metric)
            if heatmap:
                html_content += f"Plotly.newPlot('{div_id}', {heatmap.to_json()});\n"
        
        html_content += """
            </script>
        </body>
        </html>
        """
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return dashboard_path

