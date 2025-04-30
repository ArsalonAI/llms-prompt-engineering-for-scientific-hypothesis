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
        
        # Initialize results DataFrame with all required columns
        self._results_df = pd.DataFrame(columns=[
            'step',
            'run_id', 
            'idea', 
            'batch_prompt', 
            'evaluation',
            'evaluation_full',
            # Pairwise similarity metrics (between generated ideas)
            'avg_cosine_similarity',
            'avg_self_bleu',
            'avg_bertscore',
            # Context similarity metrics (with original text)
            'context_cosine',
            'context_self_bleu',
            'context_bertscore',
            'timestamp'
        ])
        
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
            'num_ideas': result.get('num_ideas', 0)
        }
        
        # Add arrays as separate columns when appropriate
        for key in ['quality_scores', 'cosine_similarities', 'self_bleu_scores', 'bertscore_scores']:
            if key in result:
                df_row[key] = result[key]
        
        # Store KDE data separately to avoid bloating the DataFrame
        if 'kde_values' in result:
            df_row['has_kde_data'] = True
        
        # Update DataFrame
        self._results_df = pd.concat([
            self._results_df,
            pd.DataFrame([df_row])
        ], ignore_index=True)
        
        # Print concise progress
        metrics_avg = {}
        # Only average metrics that are lists of numbers
        for metric in ['cosine_similarities', 'self_bleu_scores', 'bertscore_scores']:
            if metric in result and result[metric] and isinstance(result[metric], list):
                # Filter out non-numeric types just in case, though they should be numbers
                numeric_values = [v for v in result[metric] if isinstance(v, (int, float))]
                if numeric_values:
                    metrics_avg[metric.replace('_scores', '').replace('_similarities', '')] = \
                        sum(numeric_values) / len(numeric_values)
        
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Format: YYYYMMDD_HHMMSS
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
        """Helper function to calculate statistics for a similarity metric."""
        data = pd.to_numeric(metric_data, errors='coerce').dropna()
        if len(data) > 0:
            return {
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
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
        if self._results_df is None or metric_name not in self._results_df.columns:
            return None
            
        # Clean the data - remove None values and convert to float
        data = pd.to_numeric(self._results_df[metric_name], errors='coerce').dropna()
        
        if len(data) == 0:
            return None
            
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
        x_kde, y_kde = self._calculate_kde(data)
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
        """Create an interactive table visualization of results."""
        if self._results_df is None or len(self._results_df) == 0:
            return None
            
        # Create a formatted version of the DataFrame for display
        display_df = self._results_df.copy()
        
        # Format numeric columns to 3 decimal places
        numeric_cols = [
            'avg_cosine_similarity', 'avg_self_bleu', 'avg_bertscore',
            'context_cosine', 'context_self_bleu', 'context_bertscore'
        ]
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(3)
        
        # Create the table with separate sections for different metrics
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[
                    'Step', 'Run ID', 'Evaluation',
                    'Similarity to Original Context',
                    'Similarity to Other Ideas',
                    'Full Analysis'
                ],
                fill_color='royalblue',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    display_df['step'].fillna('N/A'),
                    display_df['run_id'].fillna('N/A'),
                    display_df['evaluation'].fillna('N/A'),
                    # Context similarities
                    display_df.apply(lambda row: (
                        f"cos: {row.get('context_cosine', 0.0):.3f}<br>"
                        f"bleu: {row.get('context_self_bleu', 0.0):.3f}<br>"
                        f"bert: {row.get('context_bertscore', 0.0):.3f}"
                    ), axis=1),
                    # Pairwise similarities
                    display_df.apply(lambda row: (
                        f"cos: {row.get('avg_cosine_similarity', 0.0):.3f}<br>"
                        f"bleu: {row.get('avg_self_bleu', 0.0):.3f}<br>"
                        f"bert: {row.get('avg_bertscore', 0.0):.3f}"
                    ), axis=1),
                    display_df['evaluation_full'].fillna('N/A').apply(
                        lambda x: x.replace('\n', '<br>') if pd.notnull(x) else ''
                    )
                ],
                fill_color='lavender',
                align='left',
                height=30
            )
        )])
        
        fig.update_layout(
            title="Experiment Results",
            height=400 * (len(display_df) // 10 + 1),
            margin=dict(t=30, l=10, r=10, b=10)
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
        
        metrics = [
            ('context_cosine', 'Cosine'),
            ('context_self_bleu', 'Self-BLEU'),
            ('context_bertscore', 'BERTScore')
        ]
        
        for col, name in metrics:
            if col in self._results_df.columns:
                fig.add_trace(go.Scatter(
                    x=self._results_df['run_id'],
                    y=self._results_df[col],
                    name=name,
                    mode='lines+markers'
                ))
        
        fig.update_layout(
            title="Similarity to Original Context",
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
                    <div class="plot-container" id="quality-dist"></div>
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
            ('quality_scores', 'quality-dist'),
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

