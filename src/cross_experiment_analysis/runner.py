"""
Cross-experiment analysis runner module.

This module provides functionality to run cross-experiment analysis.
"""
import os
from cross_experiment_analysis.analyzer import CrossExperimentAnalyzer

def run_cross_experiment_analysis(experiment_dir=None, output_dir=None):
    """
    Run cross-experiment analysis to compare results across different experiment runs.
    
    Args:
        experiment_dir: Base directory for experiment results (optional)
        output_dir: Directory to save analysis results (optional)
    
    Returns:
        Path to the generated dashboard
    """
    print("\n=== Running Cross-Experiment Analysis ===")
    
    # Resolve default directories if not provided
    if experiment_dir is None:
        # Try to import from main, but provide a fallback
        try:
            from main import EXPERIMENT_RESULTS_DIR
            experiment_dir = EXPERIMENT_RESULTS_DIR
        except ImportError:
            # Fallback: use a relative path based on the current file
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            experiment_dir = os.path.join(current_dir, "experiment_results")
            print(f"[INFO] Using fallback experiment directory: {experiment_dir}")
        
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, "cross_experiment_analysis")
    
    # Initialize the cross-experiment analyzer
    analyzer = CrossExperimentAnalyzer(
        experiment_dir=experiment_dir,
        output_dir=output_dir
    )
    
    # Load experiment results
    analyzer.load_experiment_results()
    
    # Extract metrics
    analyzer.extract_metrics()
    
    # Compare metrics
    analyzer.compare_metrics()
    
    # Generate the comparison dashboard
    dashboard_path = analyzer.generate_comparison_dashboard()
    
    print(f"\n[INFO] Cross-experiment analysis complete.")
    print(f"[INFO] Dashboard available at: {dashboard_path}")
    print(f"[INFO] Open with: file://{os.path.abspath(dashboard_path)}")
    
    return dashboard_path 