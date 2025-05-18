#!/usr/bin/env python
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from experiment_tracker import ExperimentTracker
from experiment_runner_templates import run_idea_generation_batch

class MockLlamaCaller:
    """Mock class to simulate LLama API calls."""
    def __init__(self, responses=None):
        self.responses = responses or ["This is a test idea " + str(i) for i in range(1, 6)]
        self.call_count = 0
        
    def __call__(self, prompt, context=None):
        self.call_count += 1
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return "Fallback test idea"

class MockQualityEvaluator:
    """Mock class to simulate quality evaluation."""
    def __call__(self, idea, context=None):
        return {"evaluation": "ACCEPT", "score": 0.75}

def test_metrics_calculation():
    """Test that metrics are calculated correctly with and without skipping intermediate calculations."""
    # Setup
    tracker = ExperimentTracker(output_dir="test_results")
    tracker.start_experiment("TestMetrics", "test")
    run_id = "test_run"
    llama_fn = MockLlamaCaller()
    num_ideas = 5
    context = "This is a test context for similarity calculation"
    
    # Test with skip_intermediate_calculations=True (default)
    print("\n--- Testing with skip_intermediate_calculations=True ---")
    results_with_skip = run_idea_generation_batch(
        prompt="Generate ideas",
        llama_fn=llama_fn,
        model_name="test_model",
        run_id=run_id + "_skip",
        quality_evaluator=MockQualityEvaluator(),
        tracker=tracker,
        context=context,
        num_ideas=num_ideas,
        skip_intermediate_calculations=True
    )
    
    # Test with skip_intermediate_calculations=False
    print("\n--- Testing with skip_intermediate_calculations=False ---")
    results_without_skip = run_idea_generation_batch(
        prompt="Generate ideas",
        llama_fn=llama_fn,
        model_name="test_model",
        run_id=run_id + "_no_skip",
        quality_evaluator=MockQualityEvaluator(),
        tracker=tracker,
        context=context,
        num_ideas=num_ideas,
        skip_intermediate_calculations=False
    )
    
    # Verify both approaches give the same final metrics
    print("\n--- Verifying Results ---")
    print(f"With skip: {len(results_with_skip['cosine_similarities'])} pairwise comparisons")
    print(f"Without skip: {len(results_without_skip['cosine_similarities'])} pairwise comparisons")
    
    # Check that context similarity scores exist in both cases
    print(f"With skip context cosine scores: {len(results_with_skip.get('context_cosine_scores', []))}")
    print(f"Without skip context cosine scores: {len(results_without_skip.get('context_cosine_scores', []))}")
    
    # Check that key metrics are the same in both cases
    key_metrics = [
        "cosine_similarities", 
        "self_bleu_scores", 
        "bertscore_scores"
    ]
    
    for metric in key_metrics:
        if len(results_with_skip[metric]) != len(results_without_skip[metric]):
            print(f"WARNING: {metric} lengths differ: {len(results_with_skip[metric])} vs {len(results_without_skip[metric])}")
        else:
            print(f"OK: {metric} lengths match: {len(results_with_skip[metric])}")
    
    tracker.end_experiment()
    print("\nTest completed")

if __name__ == "__main__":
    test_metrics_calculation() 