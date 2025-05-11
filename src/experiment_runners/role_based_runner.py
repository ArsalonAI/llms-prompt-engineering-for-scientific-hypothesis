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
        # Generate system prompt similar to scientific method
        system_prompt = generate_scientific_system_prompt()
        
        # Create combined text for role-based prompt
        combined_text = f"Abstract: {paper_content['abstract']}\n\nMethods: {paper_content['methods']}"
        
        # Generate the role-based prompt
        main_prompt = generate_role_based_prompt(
            combined_text=combined_text,
            domain=self.domain
        )
        
        # Create experiment configuration
        config = {
            "domain": self.domain,
            "focus_area": self.focus_area,
            "num_ideas": self.num_ideas,
            "batch_size": self.batch_size,
            "system_prompt": system_prompt,
            "main_prompt": main_prompt,
        }
        
        return config 