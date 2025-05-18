#!/usr/bin/env python3
"""
Test script to verify that our statistical analyzer fixes work in a real-world scenario.
This script simulates the experiment workflow with mock data, focusing only on the
statistical analysis components.
"""
import os
import sys
import json
import unittest
import numpy as np
from datetime import datetime
from src.statistical_analysis import StatisticalAnalyzer

class TestStatisticalWorkflow(unittest.TestCase):
    """Test case for statistical analysis workflow."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temp experiment_results directory
        self.experiment_dir = os.path.join(os.getcwd(), "test_experiment_results")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create mock current results with mixed data types
        self.current_results = {
            "Scientific_Hypothesis": {
                "cosine_similarities": [0.88, 0.92, 0.90, "non-numeric"],
                "self_bleu_scores": [0.32, 0.42, 0.35],
                "bertscore_scores": [0.67, 0.72, 0.69],
            },
            "Role_Based_Hypothesis": {
                "cosine_similarities": [0.85, 0.89, 0.87],
                "self_bleu_scores": [0.29, 0.39, 0.32],
                "bertscore_scores": [0.64, 0.68, 0.66],
            },
            "Few_Shot_Hypothesis": {
                "cosine_similarities": [0.82, 0.87, 0.84],
                "self_bleu_scores": [0.27, 0.35, 0.30],
                "bertscore_scores": [0.62, 0.67, 0.64],
            }
        }
    
    def tearDown(self):
        """Clean up after the test."""
        # Uncomment to clean up test directory
        # import shutil
        # shutil.rmtree(self.experiment_dir)
        pass
    
    def create_mock_experiment_data(self, experiment_type):
        """Create mock experiment results similar to what would be generated in a real run."""
        
        # Create a unique experiment folder name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_dir = os.path.join(self.experiment_dir, f"{experiment_type}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Generate some mock metric data
        cosine_values = np.random.uniform(0.8, 0.95, size=10).tolist()
        self_bleu_values = np.random.uniform(0.2, 0.6, size=10).tolist()
        bertscore_values = np.random.uniform(0.6, 0.9, size=10).tolist()
        
        # Some non-numeric values mixed in
        cosine_values.append("non-numeric")
        
        # Create a mock metadata file
        metadata = {
            "name": experiment_type,
            "type": "idea_generation",
            "model": "llama-3-3-70b",
            "start_time": timestamp,
            "config": {
                "domain": "genetic engineering",
                "focus_area": "CRISPR gene editing",
                "num_ideas": 10,
                "batch_size": 5,
                "system_prompt": "You are a helpful AI...",
                "main_prompt": "Generate hypotheses about..."
            }
        }
        
        # Create a mock results file
        results = {
            "ideas": ["Hypothesis 1", "Hypothesis 2", "Hypothesis 3"],
            "quality_scores": [0.8, 0.7, 0.9],
            "cosine_similarities": cosine_values,
            "self_bleu_scores": self_bleu_values,
            "bertscore_scores": bertscore_values,
            "summary": {
                "Context Similarities": {
                    "cosine": {"mean": 0.85, "median": 0.86, "std": 0.05},
                    "self_bleu": {"mean": 0.15, "median": 0.14, "std": 0.08},
                    "bertscore": {"mean": 0.61, "median": 0.62, "std": 0.09},
                },
                "Pairwise Similarities": {
                    "cosine_similarity": {"mean": 0.92, "median": 0.93, "std": 0.03},
                    "self_bleu": {"mean": 0.38, "median": 0.37, "std": 0.15},
                    "bertscore": {"mean": 0.72, "median": 0.73, "std": 0.11},
                }
            },
            "individual_metrics": {
                "idea1": {"cosine": 0.91, "self_bleu": 0.35, "bertscore": 0.68},
                "idea2": {"cosine": 0.89, "self_bleu": "not_a_number", "bertscore": 0.75},
                "idea3": {"cosine": 0.94, "self_bleu": 0.42, "bertscore": 0.69}
            }
        }
        
        # Save the mock data
        with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        with open(os.path.join(exp_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return exp_dir
    
    def test_statistical_analysis_with_mixed_data(self):
        """Test the statistical analyzer with mixed data types."""
        print(f"\n=== Testing Statistical Analysis Workflow ===")
        print(f"Creating mock experiment data in: {self.experiment_dir}")
        
        # Create a few mock experiments
        exp1_dir = self.create_mock_experiment_data("Scientific_Hypothesis")
        exp2_dir = self.create_mock_experiment_data("Role_Based_Hypothesis")
        exp3_dir = self.create_mock_experiment_data("Few_Shot_Hypothesis")
        
        print(f"Created mock experiments:")
        print(f"  - {os.path.basename(exp1_dir)}")
        print(f"  - {os.path.basename(exp2_dir)}")
        print(f"  - {os.path.basename(exp3_dir)}")
        
        # Create and run the statistical analyzer
        print("\nRunning Statistical Analysis...")
        try:
            analyzer = StatisticalAnalyzer(
                current_results=self.current_results,
                experiment_dir=self.experiment_dir
            )
            
            analysis_results = analyzer.perform_analysis()
            
            print("\n✅ Statistical analysis completed successfully!")
            
            # Print some results to verify
            if "conclusions" in analysis_results and "key_findings" in analysis_results["conclusions"]:
                print("\nKey Findings:")
                for finding in analysis_results["conclusions"]["key_findings"]:
                    print(f"- {finding}")
            
            dashboard_path = analysis_results.get("dashboard_path", "")
            if dashboard_path:
                print(f"\nDashboard created at: {dashboard_path}")
            
            # Make assertions to verify the test passed
            self.assertIn("metrics", analysis_results)
            self.assertIn("extended_statistics", analysis_results)
            self.assertIn("significant_differences", analysis_results)
            self.assertIn("conclusions", analysis_results)
            
            print("\nTest Successful - The statistical analyzer now handles mixed data types correctly!")
            
        except Exception as e:
            print(f"\n❌ Error in statistical analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            self.fail(f"Statistical analysis failed: {str(e)}")

if __name__ == "__main__":
    unittest.main() 