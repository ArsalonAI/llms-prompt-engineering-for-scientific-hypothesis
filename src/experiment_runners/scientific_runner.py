"""
Scientific hypothesis experiment runner module.
"""
from typing import Dict, Any, Optional

from .base_runner import BaseExperimentRunner
from prompts.scientific_prompts import generate_scientific_system_prompt, generate_scientific_hypothesis_prompt


class ScientificHypothesisRunner(BaseExperimentRunner):
    """Runner for scientific hypothesis generation experiments."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize HypothesisEvaluator if needed
        try:
            from HypothesisEvaluator import HypothesisEvaluator
            self.evaluator = HypothesisEvaluator()
        except ImportError:
            print("[WARNING] HypothesisEvaluator not available. Using default quality evaluation.")
            self.evaluator = None
    
    def prepare_experiment(self, experiment_name: str, paper_content: Dict[str, str]) -> Dict[str, Any]:
        """
        Prepare scientific hypothesis experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            paper_content: Dictionary containing paper content
            
        Returns:
            Dictionary containing experiment configuration
        """
        # Call super to get base config with LLM hyperparams
        config = super().prepare_experiment(experiment_name, paper_content)
        
        system_prompt = generate_scientific_system_prompt()
        main_prompt = generate_scientific_hypothesis_prompt(
            domain=self.domain,
            focus_area=self.focus_area,
            context = {
                'abstract': paper_content.get('abstract', ''),
                'methods': paper_content.get('methods', '')
            }
        )
        
        # Update config with specific prompts and strategy type
        config.update({
            "system_prompt": system_prompt,
            "main_prompt": main_prompt,
            "prompt_strategy_type": "Scientific_Hypothesis", # For experiment grouping
            "evaluation_criteria": self.evaluator.get_evaluation_criteria() if self.evaluator else []
        })
        return config
    
    def _evaluate_quality(self, idea: str, context: str = None) -> Dict[str, Any]:
        """
        Evaluate the quality of a hypothesis.
        
        Args:
            idea: The generated hypothesis to evaluate
            context: Optional context information
            
        Returns:
            Dictionary containing quality evaluation results
        """
        if self.evaluator:
            return self.evaluator.evaluate_hypothesis(idea, context=context)
        return super()._evaluate_quality(idea, context) 