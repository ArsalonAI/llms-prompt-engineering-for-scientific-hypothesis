import unittest
import os
import shutil
from unittest.mock import patch, MagicMock

from src.experiment_tracker import ExperimentTracker
from src.experiment_runners.base_runner import BaseExperimentRunner
from src.experiment_runners.scientific_runner import ScientificHypothesisRunner
from src.experiment_runners.role_based_runner import RoleBasedHypothesisRunner
from src.experiment_runners.few_shot_runner import FewShotHypothesisRunner
from src.cross_experiment_analysis.analyzer import CrossExperimentAnalyzer
from src.statistical_analysis import StatisticalAnalyzer


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the experiment pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_output_dir = "test_results_integration"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create a real ExperimentTracker
        self.tracker = ExperimentTracker(output_dir=self.test_output_dir)
        
        # Mock the LLM function
        self.mock_llama_fn = MagicMock(return_value="This is a test hypothesis idea.")
        
        # Create mock paper content
        self.paper_content = {
            "abstract": "This is a test abstract about CRISPR gene editing.",
            "methods": "This paper used advanced methods in genetic engineering."
        }
    
    @patch('src.experiment_runners.scientific_runner.ScientificHypothesisRunner._run_idea_generation_batch')
    def test_scientific_runner_to_dashboard(self, mock_run_batch):
        """
        Test the full pipeline from ScientificHypothesisRunner to dashboard creation.
        
        This tests that the quality_scores format is compatible between the runner and
        the dashboard creation process.
        """
        # Create a mock HypothesisEvaluator to inject
        mock_evaluator = MagicMock()
        mock_evaluator.get_evaluation_criteria.return_value = ["Criterion 1", "Criterion 2"]
        mock_evaluator.evaluate_hypothesis.return_value = {
            "evaluation": "ACCEPT",
            "evaluation_full": "Detailed scientific evaluation...",
            "is_accepted": True,
            "cosine_similarity": 0.75,
            "self_bleu": 0.3,
            "bertscore": 0.8
        }
        
        # Create ScientificHypothesisRunner
        scientific_runner = ScientificHypothesisRunner(
            tracker=self.tracker,
            llama_fn=self.mock_llama_fn,
            model_name="test-model",
            domain="test-domain",
            focus_area="test-focus",
            num_ideas=2,  # Just use 2 ideas for faster testing
            batch_size=2
        )
        
        # Inject our mock evaluator
        scientific_runner.evaluator = mock_evaluator
        
        # Set up the mock return value
        mock_run_batch.return_value = {
            "ideas": ["Idea 1", "Idea 2"],
            "quality_scores": [
                {"evaluation": "ACCEPT", "evaluation_full": "Good idea 1"},
                {"evaluation": "PRUNE", "evaluation_full": "Bad idea 2"}
            ],
            "cosine_similarities": [0.5, 0.6],
            "self_bleu_scores": [0.4, 0.45],
            "bertscore_scores": [0.8, 0.85],
            "num_ideas": 2
        }
        
        # Run the experiment
        results = scientific_runner.run("Test_Scientific", self.paper_content)
        
        # Verify the experiment ran successfully
        self.assertIsNotNone(results)
        self.assertEqual(len(results.get("ideas", [])), 2)
    
    @patch('src.experiment_runners.role_based_runner.RoleBasedHypothesisRunner._run_idea_generation_batch')
    def test_role_based_runner_to_dashboard(self, mock_run_batch):
        """
        Test the full pipeline from RoleBasedHypothesisRunner to dashboard creation.
        
        This tests that the quality_scores format from BaseExperimentRunner is compatible
        with the dashboard creation process.
        """
        # Create RoleBasedHypothesisRunner
        role_runner = RoleBasedHypothesisRunner(
            tracker=self.tracker,
            llama_fn=self.mock_llama_fn,
            model_name="test-model",
            domain="test-domain",
            focus_area="test-focus",
            num_ideas=2,
            batch_size=2
        )
        
        # Set up the mock return value
        mock_run_batch.return_value = {
            "ideas": ["Role-based Idea 1", "Role-based Idea 2"],
            "quality_scores": [
                # These will be the dictionary returned by BaseExperimentRunner._evaluate_quality
                {"evaluation": "SCORE: 0.7", "evaluation_full": "Placeholder evaluation", "score": 0.7},
                {"evaluation": "SCORE: 0.7", "evaluation_full": "Placeholder evaluation", "score": 0.7}
            ],
            "cosine_similarities": [0.5, 0.6],
            "self_bleu_scores": [0.4, 0.45],
            "bertscore_scores": [0.8, 0.85],
            "num_ideas": 2
        }
        
        # Run the experiment
        results = role_runner.run("Test_Role_Based", self.paper_content)
        
        # Verify the experiment ran successfully
        self.assertIsNotNone(results)
        self.assertEqual(len(results.get("ideas", [])), 2)
            
    @patch('src.experiment_runners.few_shot_runner.FewShotHypothesisRunner._run_idea_generation_batch')
    def test_few_shot_runner_to_dashboard(self, mock_run_batch):
        """
        Test the full pipeline from FewShotHypothesisRunner to dashboard creation.
        
        This tests that the quality_scores format from BaseExperimentRunner is compatible
        with the dashboard creation process.
        """
        # Create FewShotHypothesisRunner
        few_shot_runner = FewShotHypothesisRunner(
            tracker=self.tracker,
            llama_fn=self.mock_llama_fn,
            model_name="test-model",
            domain="test-domain",
            focus_area="test-focus",
            num_ideas=2,
            batch_size=2
        )
        
        # Set up the mock return value
        mock_run_batch.return_value = {
            "ideas": ["Few-shot Idea 1", "Few-shot Idea 2"],
            "quality_scores": [
                {"evaluation": "SCORE: 0.7", "evaluation_full": "Placeholder evaluation", "score": 0.7},
                {"evaluation": "SCORE: 0.7", "evaluation_full": "Placeholder evaluation", "score": 0.7}
            ],
            "cosine_similarities": [0.5, 0.6],
            "self_bleu_scores": [0.4, 0.45],
            "bertscore_scores": [0.8, 0.85],
            "num_ideas": 2
        }
        
        # Run the experiment
        results = few_shot_runner.run("Test_Few_Shot", self.paper_content)
        
        # Verify the experiment ran successfully
        self.assertIsNotNone(results)
        self.assertEqual(len(results.get("ideas", [])), 2)
    
    @patch('src.experiment_tracker.ExperimentTracker._create_dashboard_html')
    def test_dashboard_creation_with_mixed_quality_score_types(self, mock_create_dashboard):
        """Test dashboard creation compatibility with mixed quality score types."""
        # Configure the mock
        mock_create_dashboard.return_value = os.path.join(self.test_output_dir, "dashboard.html")
        
        # Prepare mixed quality_scores data
        mixed_quality_scores = [
            {"evaluation": "ACCEPT", "evaluation_full": "Good scientific idea"},
            0.7,  # Float from base runner implementation
            {"evaluation": "SCORE: 0.8", "evaluation_full": "Role-based score", "score": 0.8}
        ]
        
        # Manual experiment setup
        self.tracker.start_experiment(
            experiment_name="Mixed_Quality_Types",
            experiment_type="test_type",
            model_name="test-model"
        )
        
        # Log a result with mixed quality score types
        self.tracker.log_result("test_mixed", {
            "ideas": ["Idea 1", "Idea 2", "Idea 3"],
            "quality_scores": mixed_quality_scores,
            "cosine_similarities": [0.5, 0.6, 0.7],
            "self_bleu_scores": [0.4, 0.5, 0.6],
            "bertscore_scores": [0.8, 0.7, 0.9],
            "num_ideas": 3
        })
        
        # End the experiment which triggers dashboard creation
        self.tracker.end_experiment()
        
        # Verify mock was called
        self.assertTrue(mock_create_dashboard.called)
    
    @patch('src.cross_experiment_analysis.analyzer.CrossExperimentAnalyzer.load_experiment_results')
    @patch('src.cross_experiment_analysis.analyzer.CrossExperimentAnalyzer.extract_metrics')
    @patch('src.cross_experiment_analysis.analyzer.CrossExperimentAnalyzer.compare_metrics')
    @patch('src.cross_experiment_analysis.analyzer.CrossExperimentAnalyzer.generate_comparison_dashboard')
    def test_statistical_analyzer_integration(self, mock_generate_dashboard, mock_compare_metrics, 
                                             mock_extract_metrics, mock_load_results):
        """
        Test the integration of StatisticalAnalyzer with experiment results.
        
        This tests that the StatisticalAnalyzer can process experiment results
        and generate a research-grade dashboard with statistical analysis.
        """
        # Configure mocks
        mock_dashboard_path = os.path.join(self.test_output_dir, "mock_comparison_dashboard.html")
        with open(mock_dashboard_path, "w") as f:
            f.write("<html><body><h1>Comparison Dashboard</h1></body></html>")
        
        mock_generate_dashboard.return_value = mock_dashboard_path
        
        mock_extract_metrics.return_value = {
            "cosine": {
                "Scientific_Hypothesis": [0.6, 0.65, 0.7],
                "Role_Based_Hypothesis": [0.5, 0.55, 0.6],
                "Few_Shot_Hypothesis": [0.7, 0.75, 0.8]
            },
            "self_bleu": {
                "Scientific_Hypothesis": [0.5, 0.55, 0.6],
                "Role_Based_Hypothesis": [0.4, 0.45, 0.5],
                "Few_Shot_Hypothesis": [0.6, 0.65, 0.7]
            },
            "bertscore": {
                "Scientific_Hypothesis": [0.8, 0.85, 0.9],
                "Role_Based_Hypothesis": [0.7, 0.75, 0.8],
                "Few_Shot_Hypothesis": [0.9, 0.95, 1.0]
            }
        }
        
        mock_compare_metrics.return_value = {
            "statistical_tests": {
                "cosine": {
                    "Scientific_Hypothesis vs Role_Based_Hypothesis": {
                        "mann_whitney": {"p_value": 0.03, "significant": True},
                        "t_test": {"p_value": 0.04, "significant": True},
                        "effect_size": {"cohen_d": 0.8, "interpretation": "large"}
                    },
                    "Scientific_Hypothesis vs Few_Shot_Hypothesis": {
                        "mann_whitney": {"p_value": 0.01, "significant": True},
                        "t_test": {"p_value": 0.02, "significant": True},
                        "effect_size": {"cohen_d": -0.7, "interpretation": "large"}
                    }
                }
            }
        }
        
        # Create experiment results
        experiment_results = {
            "Scientific_Hypothesis": {
                "ideas": ["Idea 1", "Idea 2", "Idea 3"],
                "quality_scores": [
                    {"evaluation": "SCORE: 0.8", "evaluation_full": "Full eval", "score": 0.8},
                    {"evaluation": "SCORE: 0.9", "evaluation_full": "Full eval", "score": 0.9},
                    {"evaluation": "SCORE: 0.7", "evaluation_full": "Full eval", "score": 0.7}
                ],
                "cosine_similarities": [0.6, 0.65, 0.7],
                "self_bleu_scores": [0.5, 0.55, 0.6],
                "bertscore_scores": [0.8, 0.85, 0.9]
            },
            "Role_Based_Hypothesis": {
                "ideas": ["Idea 4", "Idea 5", "Idea 6"],
                "quality_scores": [
                    {"evaluation": "SCORE: 0.7", "evaluation_full": "Full eval", "score": 0.7},
                    {"evaluation": "SCORE: 0.8", "evaluation_full": "Full eval", "score": 0.8},
                    {"evaluation": "SCORE: 0.6", "evaluation_full": "Full eval", "score": 0.6}
                ],
                "cosine_similarities": [0.5, 0.55, 0.6],
                "self_bleu_scores": [0.4, 0.45, 0.5],
                "bertscore_scores": [0.7, 0.75, 0.8]
            },
            "Few_Shot_Hypothesis": {
                "ideas": ["Idea 7", "Idea 8", "Idea 9"],
                "quality_scores": [
                    {"evaluation": "SCORE: 0.9", "evaluation_full": "Full eval", "score": 0.9},
                    {"evaluation": "SCORE: 0.8", "evaluation_full": "Full eval", "score": 0.8},
                    {"evaluation": "SCORE: 0.7", "evaluation_full": "Full eval", "score": 0.7}
                ],
                "cosine_similarities": [0.7, 0.75, 0.8],
                "self_bleu_scores": [0.6, 0.65, 0.7],
                "bertscore_scores": [0.9, 0.95, 1.0]
            }
        }
        
        # Create the statistical analyzer
        analyzer = StatisticalAnalyzer(
            current_results=experiment_results,
            experiment_dir=self.test_output_dir
        )
        
        # Run the analysis
        results = analyzer.perform_analysis()
        
        # Verify mocks were called
        mock_load_results.assert_called_once()
        mock_extract_metrics.assert_called_once()
        mock_compare_metrics.assert_called_once()
        mock_generate_dashboard.assert_called_once()
        
        # Check results structure
        self.assertIsNotNone(results)
        self.assertIn("metrics", results)
        self.assertIn("comparison_results", results)
        self.assertIn("extended_statistics", results)
        self.assertIn("significant_differences", results)
        self.assertIn("conclusions", results)
        self.assertIn("dashboard_path", results)
        
        # Check that the dashboard was created
        dashboard_path = results["dashboard_path"]
        self.assertTrue(os.path.exists(dashboard_path))
        
        # Check content of dashboard
        with open(dashboard_path, 'r') as f:
            content = f.read()
        self.assertIn("Statistical Evidence and Conclusions", content)
        self.assertIn("Extended Statistical Measures", content)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)


if __name__ == '__main__':
    unittest.main() 