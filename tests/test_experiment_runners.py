import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from src.experiment_runners.base_runner import BaseExperimentRunner
from src.experiment_runners.scientific_runner import ScientificHypothesisRunner
from src.experiment_runners.role_based_runner import RoleBasedHypothesisRunner
from src.experiment_runners.few_shot_runner import FewShotHypothesisRunner
from src.experiment_tracker import ExperimentTracker


class TestBaseExperimentRunner(unittest.TestCase):
    """Test cases for the BaseExperimentRunner class and its _evaluate_quality method."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock tracker
        self.mock_tracker = MagicMock(spec=ExperimentTracker)
        
        # Create mock LLM function
        self.mock_llama_fn = MagicMock(return_value="This is a mock response")
        
        # Initialize BaseExperimentRunner
        self.base_runner = BaseExperimentRunner(
            tracker=self.mock_tracker,
            llama_fn=self.mock_llama_fn,
            model_name="test-model",
            domain="test-domain",
            focus_area="test-focus",
            num_ideas=5,
            batch_size=2
        )
    
    def test_evaluate_quality_returns_dictionary(self):
        """Test that the base _evaluate_quality method returns a dictionary with the correct structure."""
        idea = "This is a test idea"
        context = "This is a test context"
        
        result = self.base_runner._evaluate_quality(idea, context)
        
        # Verify result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Verify required keys are present
        self.assertIn("evaluation", result)
        self.assertIn("evaluation_full", result)
        self.assertIn("score", result)
        
        # Verify evaluation contains the expected placeholder score format
        self.assertEqual(result["evaluation"], "SCORE: 0.7")
        
        # Verify evaluation_full contains explanatory text
        self.assertIn("placeholder evaluation", result["evaluation_full"])
        
        # Verify score is a float
        self.assertIsInstance(result["score"], float)
        self.assertEqual(result["score"], 0.7)
    
    @patch("src.experiment_runners.base_runner.run_idea_generation_batch")
    def test_run_idea_generation_batch_with_skip_parameter(self, mock_run_batch):
        """Test that the skip_intermediate_calculations parameter is correctly passed to run_idea_generation_batch."""
        # Setup mock return value
        mock_run_batch.return_value = {"ideas": ["idea1", "idea2"], "quality_scores": [0.7, 0.8]}
        
        # Call with skip_intermediate_calculations=True
        self.base_runner._run_idea_generation_batch(
            prompt="test prompt",
            system_prompt="test system prompt",
            context="test context",
            run_id="test_run_id",
            skip_intermediate_calculations=True
        )
        
        # Verify the parameter was passed correctly to the imported function
        mock_run_batch.assert_called_once()
        args, kwargs = mock_run_batch.call_args
        self.assertTrue("skip_intermediate_calculations" in kwargs)
        self.assertTrue(kwargs["skip_intermediate_calculations"])
        
        # Reset mock
        mock_run_batch.reset_mock()
        
        # Call with skip_intermediate_calculations=False
        self.base_runner._run_idea_generation_batch(
            prompt="test prompt",
            system_prompt="test system prompt",
            context="test context",
            run_id="test_run_id",
            skip_intermediate_calculations=False
        )
        
        # Verify the parameter was passed correctly
        mock_run_batch.assert_called_once()
        args, kwargs = mock_run_batch.call_args
        self.assertTrue("skip_intermediate_calculations" in kwargs)
        self.assertFalse(kwargs["skip_intermediate_calculations"])
    
    @patch("src.experiment_runners.base_runner.BaseExperimentRunner._run_idea_generation_batch")
    def test_run_passes_skip_parameter(self, mock_run_batch):
        """Test that the run method correctly passes skip_intermediate_calculations parameter."""
        # Setup mock
        mock_run_batch.return_value = {"ideas": ["idea1", "idea2"], "quality_scores": [0.7, 0.8]}
        
        # Mock prepare_experiment to avoid NotImplementedError
        self.base_runner.prepare_experiment = MagicMock(return_value={
            "main_prompt": "test prompt",
            "system_prompt": "test system prompt"
        })
        
        # Setup mock tracker method
        self.mock_tracker.generate_run_id.return_value = "test_run_id"
        
        # Call with skip_intermediate_calculations=True
        paper_content = {"abstract": "test abstract", "methods": "test methods"}
        self.base_runner.run("TestExperiment", paper_content, skip_intermediate_calculations=True)
        
        # Verify parameter is passed to _run_idea_generation_batch
        mock_run_batch.assert_called_once()
        args, kwargs = mock_run_batch.call_args
        self.assertTrue("skip_intermediate_calculations" in kwargs)
        self.assertTrue(kwargs["skip_intermediate_calculations"])
        
        # Reset mock
        mock_run_batch.reset_mock()
        
        # Call with skip_intermediate_calculations=False
        self.base_runner.run("TestExperiment", paper_content, skip_intermediate_calculations=False)
        
        # Verify parameter is passed correctly
        mock_run_batch.assert_called_once()
        args, kwargs = mock_run_batch.call_args
        self.assertTrue("skip_intermediate_calculations" in kwargs)
        self.assertFalse(kwargs["skip_intermediate_calculations"])


class TestScientificHypothesisRunner(unittest.TestCase):
    """Test cases for the ScientificHypothesisRunner class and its _evaluate_quality method."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock tracker
        self.mock_tracker = MagicMock(spec=ExperimentTracker)
        
        # Create mock LLM function
        self.mock_llama_fn = MagicMock(return_value="This is a mock response")
        
        # Mock the imported HypothesisEvaluator
        mock_hypothesis_evaluator = MagicMock()
        mock_hypothesis_evaluator.evaluate_hypothesis.return_value = {
            "evaluation": "ACCEPT",
            "evaluation_full": "Detailed evaluation...",
            "is_accepted": True,
            "cosine_similarity": 0.75,
            "self_bleu": 0.3,
            "bertscore": 0.8
        }
        
        # Initialize ScientificHypothesisRunner with mocked evaluator
        self.scientific_runner = ScientificHypothesisRunner(
            tracker=self.mock_tracker,
            llama_fn=self.mock_llama_fn,
            model_name="test-model",
            domain="test-domain",
            focus_area="test-focus",
            num_ideas=5,
            batch_size=2
        )
        
        # Manually set the evaluator
        self.scientific_runner.evaluator = mock_hypothesis_evaluator
    
    def tearDown(self):
        """Clean up after each test."""
        pass
    
    def test_evaluate_quality_returns_dictionary(self):
        """Test that ScientificHypothesisRunner._evaluate_quality returns a dictionary with the correct structure."""
        idea = "This is a test scientific hypothesis"
        context = "This is a test context"
        
        result = self.scientific_runner._evaluate_quality(idea, context)
        
        # Verify result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Verify HypothesisEvaluator was called
        self.scientific_runner.evaluator.evaluate_hypothesis.assert_called_once_with(idea, context=context)
        
        # Verify required keys are present (from the mock return value)
        self.assertIn("evaluation", result)
        self.assertIn("evaluation_full", result)
        self.assertIn("is_accepted", result)
        
        # Verify values match what the evaluator returned
        self.assertEqual(result["evaluation"], "ACCEPT")
        self.assertEqual(result["evaluation_full"], "Detailed evaluation...")
        self.assertTrue(result["is_accepted"])


class TestRoleBasedHypothesisRunner(unittest.TestCase):
    """Test cases for the RoleBasedHypothesisRunner to ensure it inherits the correct _evaluate_quality behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock tracker
        self.mock_tracker = MagicMock(spec=ExperimentTracker)
        
        # Create mock LLM function
        self.mock_llama_fn = MagicMock(return_value="This is a mock response")
        
        # Initialize RoleBasedHypothesisRunner
        self.role_runner = RoleBasedHypothesisRunner(
            tracker=self.mock_tracker,
            llama_fn=self.mock_llama_fn,
            model_name="test-model",
            domain="test-domain",
            focus_area="test-focus",
            num_ideas=5,
            batch_size=2
        )
    
    def test_evaluate_quality_returns_dictionary(self):
        """Test that RoleBasedHypothesisRunner inherits the base _evaluate_quality method that returns a dictionary."""
        idea = "This is a test role-based hypothesis"
        context = "This is a test context"
        
        result = self.role_runner._evaluate_quality(idea, context)
        
        # Verify result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Verify required keys are present
        self.assertIn("evaluation", result)
        self.assertIn("evaluation_full", result)
        self.assertIn("score", result)
        
        # Verify evaluation contains the expected placeholder score format
        self.assertEqual(result["evaluation"], "SCORE: 0.7")


class TestFewShotHypothesisRunner(unittest.TestCase):
    """Test cases for the FewShotHypothesisRunner to ensure it inherits the correct _evaluate_quality behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock tracker
        self.mock_tracker = MagicMock(spec=ExperimentTracker)
        
        # Create mock LLM function
        self.mock_llama_fn = MagicMock(return_value="This is a mock response")
        
        # Initialize FewShotHypothesisRunner
        self.few_shot_runner = FewShotHypothesisRunner(
            tracker=self.mock_tracker,
            llama_fn=self.mock_llama_fn,
            model_name="test-model",
            domain="test-domain",
            focus_area="test-focus",
            num_ideas=5,
            batch_size=2
        )
    
    def test_evaluate_quality_returns_dictionary(self):
        """Test that FewShotHypothesisRunner inherits the base _evaluate_quality method that returns a dictionary."""
        idea = "This is a test few-shot hypothesis"
        context = "This is a test context"
        
        result = self.few_shot_runner._evaluate_quality(idea, context)
        
        # Verify result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Verify required keys are present
        self.assertIn("evaluation", result)
        self.assertIn("evaluation_full", result)
        self.assertIn("score", result)
        
        # Verify evaluation contains the expected placeholder score format
        self.assertEqual(result["evaluation"], "SCORE: 0.7")


if __name__ == '__main__':
    unittest.main() 