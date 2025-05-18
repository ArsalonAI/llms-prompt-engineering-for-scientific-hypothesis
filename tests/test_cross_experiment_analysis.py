import unittest
import os
import shutil
import json
import tempfile
from unittest.mock import patch, MagicMock

from src.cross_experiment_analysis.analyzer import CrossExperimentAnalyzer
from src.statistical_analysis import StatisticalAnalyzer


class TestCrossExperimentAnalysis(unittest.TestCase):
    """Test the cross-experiment analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "cross_experiment_analysis")
        self.integrated_dir = os.path.join(self.test_dir, "integrated_analysis")
        
        # Create two mock experiment directories
        self.exp1_dir = os.path.join(self.test_dir, "Scientific_Experiment_20220101_120000")
        self.exp2_dir = os.path.join(self.test_dir, "Role_Based_Experiment_20220101_120000")
        
        # Create required files in each experiment directory
        os.makedirs(self.exp1_dir, exist_ok=True)
        os.makedirs(self.exp2_dir, exist_ok=True)
        
        # Create mock metadata files
        with open(os.path.join(self.exp1_dir, "metadata.json"), 'w') as f:
            json.dump({
                "name": "Scientific_Experiment",
                "type": "Scientific_Hypothesis",
                "model": "test-model",
                "config": {},
                "start_time": "2022-01-01T12:00:00",
                "end_time": "2022-01-01T12:10:00"
            }, f)
        
        with open(os.path.join(self.exp2_dir, "metadata.json"), 'w') as f:
            json.dump({
                "name": "Role_Based_Experiment",
                "type": "Role_Based_Hypothesis",
                "model": "test-model",
                "config": {},
                "start_time": "2022-01-01T12:00:00",
                "end_time": "2022-01-01T12:10:00"
            }, f)
        
        # Create mock results.json files with statistical data
        with open(os.path.join(self.exp1_dir, "results.json"), 'w') as f:
            json.dump({
                "results": [
                    {
                        "run_id": "run1",
                        "cosine_similarities": [0.7, 0.8, 0.9],
                        "self_bleu_scores": [0.3, 0.4, 0.5],
                        "bertscore_scores": [0.6, 0.7, 0.8]
                    }
                ],
                "summary": {
                    "Total Ideas": 10,
                    "Context Similarities": {
                        "cosine": {"mean": 0.8, "median": 0.8, "std": 0.1}
                    },
                    "Pairwise Similarities": {
                        "cosine_similarity": {"mean": 0.8, "median": 0.8, "std": 0.1},
                        "self_bleu": {"mean": 0.4, "median": 0.4, "std": 0.1},
                        "bertscore": {"mean": 0.7, "median": 0.7, "std": 0.1}
                    }
                }
            }, f)
        
        with open(os.path.join(self.exp2_dir, "results.json"), 'w') as f:
            json.dump({
                "results": [
                    {
                        "run_id": "run2",
                        "cosine_similarities": [0.6, 0.7, 0.8],
                        "self_bleu_scores": [0.2, 0.3, 0.4],
                        "bertscore_scores": [0.5, 0.6, 0.7]
                    }
                ],
                "summary": {
                    "Total Ideas": 10,
                    "Context Similarities": {
                        "cosine": {"mean": 0.7, "median": 0.7, "std": 0.1}
                    },
                    "Pairwise Similarities": {
                        "cosine_similarity": {"mean": 0.7, "median": 0.7, "std": 0.1},
                        "self_bleu": {"mean": 0.3, "median": 0.3, "std": 0.1},
                        "bertscore": {"mean": 0.6, "median": 0.6, "std": 0.1}
                    }
                }
            }, f)
        
        # Create mock results.csv files (empty but present)
        with open(os.path.join(self.exp1_dir, "results.csv"), 'w') as f:
            f.write("run_id,cosine_similarities,self_bleu_scores,bertscore_scores\n")
        
        with open(os.path.join(self.exp2_dir, "results.csv"), 'w') as f:
            f.write("run_id,cosine_similarities,self_bleu_scores,bertscore_scores\n")
    
    def test_cross_analyzer_load_experiment_results(self):
        """Test CrossExperimentAnalyzer's ability to load experiment results."""
        analyzer = CrossExperimentAnalyzer(
            experiment_dir=self.test_dir,
            output_dir=self.output_dir
        )
        
        experiment_data = analyzer.load_experiment_results()
        
        # Check that both experiments were loaded
        self.assertEqual(len(experiment_data), 2)
        
        # Verify experiment names were parsed correctly
        exp_names = sorted(experiment_data.keys())
        self.assertIn("Scientific_Experiment_20220101_120000", exp_names)
        self.assertIn("Role_Based_Experiment_20220101_120000", exp_names)
        
        # Verify metadata was loaded
        self.assertEqual(experiment_data["Scientific_Experiment_20220101_120000"]["metadata"]["name"], "Scientific_Experiment")
        self.assertEqual(experiment_data["Role_Based_Experiment_20220101_120000"]["metadata"]["name"], "Role_Based_Experiment")
    
    def test_cross_analyzer_extract_metrics(self):
        """Test CrossExperimentAnalyzer's ability to extract metrics."""
        analyzer = CrossExperimentAnalyzer(
            experiment_dir=self.test_dir,
            output_dir=self.output_dir
        )
        
        analyzer.load_experiment_results()
        metrics = analyzer.extract_metrics()
        
        # Check that metrics were extracted for both experiments
        self.assertEqual(len(metrics), 2)
        
        # Check that metrics contain experiment metadata
        for exp_name, exp_metrics in metrics.items():
            self.assertIn("name", exp_metrics)
            self.assertIn("type", exp_metrics)
            
            # Check that metrics contain similarity data
            if "Scientific_Experiment" in exp_metrics["name"]:
                self.assertAlmostEqual(exp_metrics["pairwise_cosine_similarity_mean"], 0.8, places=1)
            elif "Role_Based_Experiment" in exp_metrics["name"]:
                self.assertAlmostEqual(exp_metrics["pairwise_cosine_similarity_mean"], 0.7, places=1)
    
    def test_cross_analyzer_compare_metrics(self):
        """Test CrossExperimentAnalyzer's ability to compare metrics."""
        analyzer = CrossExperimentAnalyzer(
            experiment_dir=self.test_dir,
            output_dir=self.output_dir
        )
        
        analyzer.load_experiment_results()
        analyzer.extract_metrics()
        comparison_results = analyzer.compare_metrics()
        
        # Check that comparison results include key sections
        self.assertIn("context_metrics", comparison_results)
        self.assertIn("pairwise_metrics", comparison_results)
        self.assertIn("runtime_comparison", comparison_results)
    
    @patch('src.cross_experiment_analysis.analyzer.CrossExperimentAnalyzer.generate_comparison_plots')
    def test_cross_analyzer_generate_dashboard(self, mock_generate_plots):
        """Test CrossExperimentAnalyzer's ability to generate a dashboard."""
        # Configure the mock
        mock_generate_plots.return_value = {
            "context_metrics_cosine": os.path.join(self.output_dir, "plots", "context_cosine_comparison.html"),
            "pairwise_metrics_cosine_similarity": os.path.join(self.output_dir, "plots", "pairwise_cosine_comparison.html")
        }
        
        analyzer = CrossExperimentAnalyzer(
            experiment_dir=self.test_dir,
            output_dir=self.output_dir
        )
        
        analyzer.load_experiment_results()
        analyzer.extract_metrics()
        analyzer.compare_metrics()
        
        # Create the plots directory that would normally be created by generate_comparison_plots
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
        # Mock HTML generation
        with open(os.path.join(self.output_dir, "plots", "context_cosine_comparison.html"), 'w') as f:
            f.write("<html><body>Test plot</body></html>")
        
        with open(os.path.join(self.output_dir, "plots", "pairwise_cosine_comparison.html"), 'w') as f:
            f.write("<html><body>Test plot</body></html>")
        
        dashboard_path = analyzer.generate_comparison_dashboard()
        
        # Verify the dashboard path
        self.assertEqual(dashboard_path, os.path.join(self.output_dir, "comparison_dashboard.html"))
        
        # Verify the dashboard file was created
        self.assertTrue(os.path.exists(dashboard_path))
    
    def test_integrated_analyzer_initialization(self):
        """Test initialization of StatisticalAnalyzer."""
        mock_results = {"test": "data"}
        
        analyzer = StatisticalAnalyzer(
            current_results=mock_results,
            experiment_dir=self.test_dir
        )
        
        # Verify analyzer was initialized successfully
        self.assertEqual(analyzer.experiment_dir, self.test_dir)
        self.assertEqual(analyzer.current_results, mock_results)
    
    @patch('src.cross_experiment_analysis.analyzer.CrossExperimentAnalyzer.load_experiment_results')
    @patch('src.cross_experiment_analysis.analyzer.CrossExperimentAnalyzer.extract_metrics')
    @patch('src.cross_experiment_analysis.analyzer.CrossExperimentAnalyzer.compare_metrics')
    @patch('src.statistical_analysis.StatisticalAnalyzer.generate_research_dashboard')
    def test_integrated_analyzer_perform_analysis(self, mock_generate_dashboard, mock_compare_metrics, 
                                                mock_extract_metrics, mock_load_results):
        """Test StatisticalAnalyzer's perform_analysis method."""
        # Configure mocks
        mock_load_results.return_value = {"results": "some results"}
        mock_extract_metrics.return_value = {"metrics": "some metrics"}
        mock_compare_metrics.return_value = {"comparison": "some comparison"}
        mock_generate_dashboard.return_value = os.path.join(self.integrated_dir, "analysis_dashboard.html")
        
        # Create mock results
        mock_current_results = {
            "Scientific_Hypothesis": {
                "ideas": ["Idea 1", "Idea 2"],
                "quality_scores": [
                    {"evaluation": "ACCEPT", "score": 0.8},
                    {"evaluation": "REJECT", "score": 0.3}
                ],
                "cosine_similarities": [0.7, 0.8],
                "self_bleu_scores": [0.3, 0.4],
                "bertscore_scores": [0.6, 0.7]
            }
        }
        
        # Initialize and run the analyzer
        analyzer = StatisticalAnalyzer(
            current_results=mock_current_results,
            experiment_dir=self.test_dir
        )
        
        results = analyzer.perform_analysis()
        
        # Verify analyze_cross_experiment_data was called
        self.assertIsNotNone(results)
        self.assertIn("metrics", results)
        self.assertIn("comparison_results", results)
        self.assertIn("dashboard_path", results)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main() 