import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from datetime import datetime
from pathlib import Path

class LocalExperimentLogger:
    def __init__(self, output_dir="experiment_results"):
        """Initialize the local experiment logger.
        
        Args:
            output_dir: Directory where experiment results will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_data = []
        
        # Create experiment directory
        self.experiment_dir = self.output_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
    def create_distribution_plot(self, data, metric_name):
        """Create a distribution plot for a given metric."""
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
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=[mean_val, mean_val],
            y=[0, max_y],
            mode='lines',
            name='Mean',
            line=dict(color='red', dash='dash')
        ))
        
        # Add median line
        fig.add_trace(go.Scatter(
            x=[median_val, median_val],
            y=[0, max_y],
            mode='lines',
            name='Median',
            line=dict(color='green', dash='dash')
        ))
        
        fig.update_layout(
            title=f'{metric_name} Distribution',
            xaxis_title=metric_name,
            yaxis_title='Count',
            showlegend=True
        )
        return fig
    
    def log_experiment(self, experiment_data):
        """Log experiment data and create visualizations.
        
        Args:
            experiment_data: Dictionary containing the experiment data
        """
        # Store the result
        self.results_data.append(experiment_data)
        
        # Save raw data
        self._save_raw_data()
        
        # Update visualizations
        self._update_visualizations()
        
        # Create and update summary HTML
        self._create_summary_html()
        
    def _save_raw_data(self):
        """Save raw data as JSON."""
        with open(self.experiment_dir / "raw_data.json", "w") as f:
            json.dump(self.results_data, f, indent=2)
            
    def _update_visualizations(self):
        """Update all visualizations in the HTML file."""
        if not self.results_data:
            return
        
        df = pd.DataFrame(self.results_data)
        
        # Only create plots if we have data
        if len(df) > 0:
            # Create and save a simple HTML file
            html_content = """
            <html>
            <head>
                <title>Experiment Results</title>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <h1>Experiment Results</h1>
                <h2>Results Table</h2>
            """
            
            # Add the DataFrame as a table
            html_content += df.to_html()
            
            html_content += """
            </body>
            </html>
            """
            
            # Save the HTML file
            with open(self.experiment_dir / "dashboard.html", "w") as f:
                f.write(html_content)

    def _create_summary_html(self):
        """Create a simple summary HTML page."""
        if not self.results_data:
            return
            
        df = pd.DataFrame(self.results_data)
        
        html_content = """
        <html>
        <head>
            <title>Experiment Summary</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
            </style>
        </head>
        <body>
            <h1>Experiment Summary</h1>
        """
        
        # Add basic statistics
        html_content += f"<p>Total Experiments: {len(df)}</p>"
        
        # Add the data table
        html_content += "<h2>Latest Results</h2>"
        html_content += df.tail(10).to_html()
        
        html_content += """
        </body>
        </html>
        """
        
        with open(self.experiment_dir / "summary.html", "w") as f:
            f.write(html_content)
            
    def get_experiment_url(self):
        """Get the URL to the experiment summary page."""
        return str(self.experiment_dir / "index.html") 