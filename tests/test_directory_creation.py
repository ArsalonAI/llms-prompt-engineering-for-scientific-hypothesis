#!/usr/bin/env python3
"""
Test script to verify directory creation for statistical analysis.
"""
import os
import sys

# Define directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiment_results")

def create_directories():
    """Create necessary directories for the experiment."""
    # Create main experiment results directory
    os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)
    print(f"Created main directory: {EXPERIMENT_RESULTS_DIR}")
    
    # Create subdirectories
    subdirs = [
        "cross_experiment_analysis",
        "statistical_analysis",
        "cross_experiment_analysis/plots"
    ]
    
    for subdir in subdirs:
        full_path = os.path.join(EXPERIMENT_RESULTS_DIR, subdir)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created subdirectory: {full_path}")
    
    # Verify directories exist
    for subdir in subdirs:
        full_path = os.path.join(EXPERIMENT_RESULTS_DIR, subdir)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            print(f"✅ Verified: {full_path}")
        else:
            print(f"❌ Failed to create: {full_path}")

def main():
    create_directories()
    print("\nDirectory structure is correctly set up for the statistical analysis.")

if __name__ == "__main__":
    main() 