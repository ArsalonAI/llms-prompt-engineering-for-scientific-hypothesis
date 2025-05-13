"""
Few-shot hypothesis experiment runner module.
"""
from typing import Dict, Any, List

from .base_runner import BaseExperimentRunner
from prompts.scientific_prompts import generate_scientific_system_prompt
from prompts.few_shot_prompts import generate_few_shot_prompt
from prompts.types import FewShotExample


class FewShotHypothesisRunner(BaseExperimentRunner):
    """Runner for few-shot hypothesis generation experiments."""
    
    def prepare_experiment(self, experiment_name: str, paper_content: Dict[str, str]) -> Dict[str, Any]:
        """
        Prepare few-shot hypothesis experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            paper_content: Dictionary containing paper content
            
        Returns:
            Dictionary containing experiment configuration
        """
        config = super().prepare_experiment(experiment_name, paper_content)
        
        # Define few-shot examples (these could be loaded from a file or defined elsewhere)
        # For now, using placeholder examples. These should be high-quality and relevant.
        examples: List[FewShotExample] = [
            FewShotExample(abstract_methods="Existing research on X shows Y...", evaluation="Good hypothesis, testable.", novel_hypothesis="Therefore, we hypothesize Z..."),
            FewShotExample(abstract_methods="Given findings A and B...", evaluation="Plausible but needs refinement.", novel_hypothesis="It is plausible that C will occur if conditions D are met...")
        ]
        
        system_prompt = "You are an AI assistant skilled in learning from examples to generate scientific hypotheses."
        main_prompt = generate_few_shot_prompt(
            combined_text=f"{paper_content.get('abstract', '')}\n\n{paper_content.get('methods', '')}",
            examples=examples, 
            domain=self.domain
        )
        
        # Log the actual examples used for traceability
        # Converting examples to dicts if they are objects for better JSON serialization if needed
        examples_for_log = []
        for ex in examples:
            if hasattr(ex, '__dict__'):
                 examples_for_log.append(ex.__dict__)
            elif isinstance(ex, dict):
                 examples_for_log.append(ex)
            # else: handle other types or raise error

        config.update({
            "system_prompt": system_prompt,
            "main_prompt": main_prompt,
            "prompt_strategy_type": "Few_Shot_Hypothesis",
            "few_shot_examples_used": examples_for_log 
        })
        return config 