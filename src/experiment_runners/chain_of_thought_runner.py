"""
Chain-of-thought hypothesis experiment runner module.
"""
from typing import Dict, Any

from .base_runner import BaseExperimentRunner
from prompts.chain_of_thought_prompts import generate_cot_prompt


class ChainOfThoughtHypothesisRunner(BaseExperimentRunner):
    """Runner for chain-of-thought hypothesis generation experiments."""
    
    def prepare_experiment(self, experiment_name: str, paper_content: Dict[str, str]) -> Dict[str, Any]:
        """
        Prepare chain-of-thought hypothesis experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            paper_content: Dictionary containing paper content
            
        Returns:
            Dictionary containing experiment configuration
        """
        config = super().prepare_experiment(experiment_name, paper_content)
        
        system_prompt = "You are an AI assistant skilled in step-by-step reasoning to generate scientific hypotheses."
        main_prompt = generate_cot_prompt(
            combined_text=f"{paper_content.get('abstract', '')}\n\n{paper_content.get('methods', '')}",
            domain=self.domain
        )
        
        config.update({
            "system_prompt": system_prompt,
            "main_prompt": main_prompt,
            "prompt_strategy_type": "Chain_Of_Thought_Hypothesis"
        })
        return config 