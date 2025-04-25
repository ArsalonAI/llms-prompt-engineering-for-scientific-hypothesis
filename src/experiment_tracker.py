import os
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

class ExperimentTracker:
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
        std_val = data.std()
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=[mean_val, mean_val],
            y=[0, max_y],
            mode='lines',
            name=f'Mean: {mean_val:.3f}',
            line=dict(color='red', dash='dash')
        ))
        
        # Add median line
        fig.add_trace(go.Scatter(
            x=[median_val, median_val],
            y=[0, max_y],
            mode='lines',
            name=f'Median: {median_val:.3f}',
            line=dict(color='green', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{metric_name} Distribution (std: {std_val:.3f})',
            xaxis_title=metric_name,
            yaxis_title='Count',
            showlegend=True,
            template='plotly_white',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig
        
    def _create_results_table(self) -> go.Figure:
        """Create an interactive table visualization of results."""
        if self._results_df is None or len(self._results_df) == 0:
            return None
            
        # Create a formatted version of the DataFrame for display
        display_df = self._results_df.copy()
        
        # Ensure required columns exist
        required_cols = ['step', 'run_id', 'evaluation', 'evaluation_full']
        for col in required_cols:
            if col not in display_df.columns:
                display_df[col] = None
        
        # Format numeric columns
        numeric_cols = ['cosine_similarity', 'self_bleu', 'bertscore']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(3)
        
        # Create the table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Step', 'Run ID', 'Evaluation', 'Metrics', 'Full Analysis'],
                fill_color='royalblue',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    display_df['step'].fillna('N/A'),
                    display_df['run_id'].fillna('N/A'),
                    display_df['evaluation'].fillna('N/A'),
                    display_df.apply(lambda row: f"cos_sim: {row.get('cosine_similarity', 0.0):.3f}<br>"
                                               f"self_bleu: {row.get('self_bleu', 0.0):.3f}<br>"
                                               f"bert: {row.get('bertscore', 0.0):.3f}", axis=1),
                    display_df['evaluation_full'].fillna('N/A').apply(lambda x: x.replace('\n', '<br>') if pd.notnull(x) else '')
                ],
                fill_color='lavender',
                align='left',
                height=30
            )
        )])
        
        fig.update_layout(
            title="Experiment Results",
            height=400 * (len(display_df) // 10 + 1),  # Adjust height based on number of rows
            margin=dict(t=30, l=10, r=10, b=10)
        )
        
        return fig
        
    def _create_dashboard_html(self, experiment_dir: str, metadata: dict, summary: dict):
        """Create a comprehensive HTML dashboard."""
        try:
            dashboard_path = os.path.join(experiment_dir, "dashboard.html")
            print(f"[DEBUG] Creating dashboard at: {dashboard_path}")
            
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
                        max-width: 1200px;
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
                    .summary {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin: 20px 0;
                    }}
                    .metric-card {{
                        background-color: #ffffff;
                        padding: 15px;
                        border-radius: 5px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    .plot-container {{
                        margin: 20px 0;
                        padding: 15px;
                        background-color: white;
                        border-radius: 5px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    .no-data {{
                        text-align: center;
                        padding: 20px;
                        color: #666;
                        font-style: italic;
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

                    <h2>Summary Statistics</h2>
                    <div class="summary">
                        <div class="metric-card">
                            <h3>Total Ideas Generated</h3>
                            <p>{summary['Total Ideas']}</p>
                        </div>
            """
            
            # Add metric cards
            for metric, stats in summary['Metrics'].items():
                html_content += f"""
                        <div class="metric-card">
                            <h3>{metric}</h3>
                            <p>Mean: {stats['mean']:.3f}</p>
                            <p>Median: {stats['median']:.3f}</p>
                            <p>Std: {stats['std']:.3f}</p>
                        </div>
                """
            
            html_content += """
                    </div>
                    
                    <h2>Results Table</h2>
                    <div class="plot-container" id="results-table"></div>
                    
                    <h2>Metric Distributions</h2>
            """
            
            # Add distribution plots
            metrics = ['cosine_similarity', 'self_bleu', 'bertscore']
            for metric in metrics:
                html_content += f"""
                    <div class="plot-container" id="{metric}-plot">
                        <div class="no-data" id="{metric}-no-data" style="display: none;">
                            No data available for {metric}
                        </div>
                    </div>
                """
            
            html_content += """
                </div>
                <script>
            """
            
            # Add the table plot
            table_fig = self._create_results_table()
            if table_fig:
                html_content += f"Plotly.newPlot('results-table', {table_fig.to_json()});\n"
            else:
                html_content += """
                    document.getElementById('results-table').innerHTML = '<div class="no-data">No results available</div>';
                """
            
            # Add distribution plots
            for metric in metrics:
                fig = self._create_distribution_plot(metric)
                if fig:
                    html_content += f"Plotly.newPlot('{metric}-plot', {fig.to_json()});\n"
                else:
                    html_content += f"""
                        document.getElementById('{metric}-no-data').style.display = 'block';
                    """
            
            html_content += """
                </script>
            </body>
            </html>
            """
            
            print(f"[DEBUG] Writing dashboard HTML file ({len(html_content)} bytes)")
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"[DEBUG] Dashboard created successfully")
            return dashboard_path
        except Exception as e:
            print(f"[ERROR] Failed to create dashboard HTML: {str(e)}")
            print("[ERROR] Error details:")
            import traceback
            traceback.print_exc()
            # Re-raise to be caught by the outer try-catch
            raise
    
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
            'step',  # Add step column
            'run_id', 
            'idea', 
            'batch_prompt', 
            'evaluation',  # Add evaluation column
            'evaluation_full',  # Add evaluation_full column
            'judged_quality',
            'cosine_similarity', 
            'self_bleu', 
            'bertscore', 
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
    
    def log_result(self, result: Dict[str, Any]):
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
            **cleaned_result
        }
        
        # Store locally
        self.current_experiment['results'].append(result_with_metadata)
        
        # Update DataFrame
        self._results_df = pd.concat([
            self._results_df,
            pd.DataFrame([result_with_metadata])
        ], ignore_index=True)
        
        # Print concise progress
        print(f"\n[STEP {self._step}]")
        print(f"Run ID: {result.get('run_id', 'N/A')} | "
              f"Evaluation: {result.get('evaluation', 'N/A')} | "
              f"Metrics: [cos_sim={result.get('cosine_similarity', 0.0):.3f}, "
              f"self_bleu={result.get('self_bleu', 0.0):.3f}, "
              f"bert={result.get('bertscore', 0.0):.3f}]")
    
    def end_experiment(self):
        """End the current experiment and save results."""
        if self.current_experiment is None:
            raise ValueError("No experiment currently running.")
        
        self.current_experiment['end_time'] = datetime.now().isoformat()
        
        # Create experiment directory with absolute path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = os.path.abspath(os.path.join(
            self.output_dir,
            f"{self.current_experiment['name']}_{timestamp}"
        ))
        os.makedirs(experiment_dir, exist_ok=True)
        
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
            "Metrics": {}
        }
        
        # Calculate metrics for both summary and detailed results
        metrics = ['cosine_similarity', 'self_bleu', 'bertscore']
        for metric in metrics:
            if metric in self._results_df.columns:
                data = pd.to_numeric(self._results_df[metric], errors='coerce').dropna()
                if len(data) > 0:
                    metric_stats = {
                        "mean": float(data.mean()),
                        "median": float(data.median()),
                        "std": float(data.std())
                    }
                    summary["Metrics"][metric] = metric_stats
                else:
                    metric_stats = {
                        "mean": 0.0,
                        "median": 0.0,
                        "std": 0.0
                    }
                    summary["Metrics"][metric] = metric_stats
        
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
            print(f"📁 Results Directory:\n   {experiment_dir}")
            print("\n📊 Generated Files:")
            print(f"   1. metadata.json - Experiment configuration and timing")
            print(f"   2. summary.json  - Quick overview of metrics")
            print(f"   3. results.json  - Detailed results and analysis")
            print(f"   4. results.csv   - Results in CSV format")
            print(f"   5. dashboard.html - Interactive visualization")
            print("\n🌐 View Dashboard:")
            print(f"   File Protocol: file://{os.path.abspath(dashboard_path)}")
            # Convert absolute path to relative web path
            web_path = os.path.relpath(dashboard_path, os.path.dirname(os.path.dirname(experiment_dir)))
            print(f"   HTTP Protocol: http://localhost:8000/{web_path}")
            print("\nQuick Start:")
            print("1. Start local server:  python -m http.server 8000")
            print("2. Open either URL in your browser")
            print("="*50 + "\n")
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
        
        # Print summary to console
        print("\n=== Experiment Summary ===")
        print(f"Total Ideas Generated: {summary['Total Ideas']}")
        for metric, stats in summary["Metrics"].items():
            print(f"{metric}:")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Median: {stats['median']:.3f}")
            print(f"  Std: {stats['std']:.3f}")
        
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