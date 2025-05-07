"""
Scientific hypothesis experiment runner module.
"""
from typing import Dict, Any, Optional

from .base_runner import BaseExperimentRunner
from src.prompts.scientific_prompts import generate_scientific_system_prompt, generate_scientific_hypothesis_prompt


class ScientificHypothesisRunner(BaseExperimentRunner):
    """Runner for scientific hypothesis generation experiments."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize HypothesisEvaluator if needed
        try:
            from src.HypothesisEvaluator import HypothesisEvaluator
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
        # Generate experiment-specific prompts
        system_prompt = generate_scientific_system_prompt(
            abstract=paper_content['abstract']
        )
        
        main_prompt = generate_scientific_hypothesis_prompt(
            domain=self.domain,
            focus_area=self.focus_area,
            context={
                'abstract': paper_content['abstract'],
                'methods': paper_content['methods']
            }
        )
        
        # Create experiment configuration
        config = {
            "domain": self.domain,
            "focus_area": self.focus_area,
            "num_ideas": self.num_ideas,
            "batch_size": self.batch_size,
            "system_prompt": system_prompt,
            "main_prompt": main_prompt,
            "evaluation_criteria": self.evaluator.get_evaluation_criteria() if self.evaluator else []
        }
        
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