"""
Role-based hypothesis experiment runner module.
"""
from typing import Dict, Any

from .base_runner import BaseExperimentRunner
from prompts.scientific_prompts import generate_scientific_system_prompt
from prompts.role_based_prompts import generate_role_based_prompt


class RoleBasedHypothesisRunner(BaseExperimentRunner):
    """Runner for role-based hypothesis generation experiments."""
    
    def prepare_experiment(self, experiment_name: str, paper_content: Dict[str, str]) -> Dict[str, Any]:
        """
        Prepare role-based hypothesis experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            paper_content: Dictionary containing paper content
            
        Returns:
            Dictionary containing experiment configuration
        """
        config = super().prepare_experiment(experiment_name, paper_content)
        
        role_description = f"You are an experienced researcher in {self.domain} with a specialization in {self.focus_area}. You are known for your ability to synthesize information and propose innovative research directions."
        
        system_prompt = "You are an AI assistant adopting a specific research persona to generate hypotheses."
        main_prompt = generate_role_based_prompt(
            combined_text=f"{paper_content.get('abstract', '')}\n\n{paper_content.get('methods', '')}",
            role_description=role_description,
            domain=self.domain
        )
        
        config.update({
            "system_prompt": system_prompt,
            "main_prompt": main_prompt,
            "prompt_strategy_type": "Role_Based_Hypothesis",
            "role_description_used": role_description
        })
        return config 