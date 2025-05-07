"""
Experiment runner module for different prompt engineering techniques.
This module provides a base class and specialized implementations for running
various prompt engineering experiments on scientific hypothesis generation.
"""
import os
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Callable, Union

from experiment_tracker import ExperimentTracker
from similarity_metrics_utils import get_cosine_similarity, get_self_bleu, get_bertscore
from prompts.scientific_prompts import generate_scientific_system_prompt, generate_scientific_hypothesis_prompt
from prompts.role_based_prompts import generate_role_based_prompt
from prompts.few_shot_prompts import generate_few_shot_prompt


class BaseExperimentRunner:
    """Base class for experiment runners."""
    
    def __init__(
        self,
        tracker: ExperimentTracker,
        llama_fn: Callable,
        model_name: str,
        domain: str = "genetic engineering",
        focus_area: str = "CRISPR gene editing",
        num_ideas: int = 10,
        batch_size: int = 5,
    ):
        """
        Initialize the experiment runner.
        
        Args:
            tracker: ExperimentTracker instance for logging results
            llama_fn: Function to call the language model
            model_name: Name of the model being used
            domain: Scientific domain for specialization
            focus_area: Specific area within the domain
            num_ideas: Number of ideas to generate
            batch_size: Number of ideas to generate in each batch
        """
        self.tracker = tracker
        self.llama_fn = llama_fn
        self.model_name = model_name
        self.domain = domain
        self.focus_area = focus_area
        self.num_ideas = num_ideas
        self.batch_size = batch_size
        self.experiment_results = {}
    
    def prepare_experiment(self, experiment_name: str, paper_content: Dict[str, str]) -> Dict[str, Any]:
        """
        Prepare experiment configuration. To be implemented by subclasses.
        
        Args:
            experiment_name: Name of the experiment
            paper_content: Dictionary containing paper content
            
        Returns:
            Dictionary containing experiment configuration
        """
        raise NotImplementedError("Subclasses must implement prepare_experiment")
    
    def run(self, experiment_name: str, paper_content: Dict[str, str]) -> Dict[str, Any]:
        """
        Run the experiment.
        
        Args:
            experiment_name: Name of the experiment
            paper_content: Dictionary containing paper content
            
        Returns:
            Dictionary containing experiment results
        """
        # Prepare experiment configuration
        config = self.prepare_experiment(experiment_name, paper_content)
        
        # Start the experiment with configuration
        self.tracker.start_experiment(
            experiment_name=experiment_name,
            experiment_type="idea_generation",
            model_name=self.model_name,
            config=config
        )
        
        # Generate a unique run ID
        run_id = self.tracker.generate_run_id("run")
        
        # Create context from paper content
        context = f"Abstract: {paper_content['abstract']}\nMethods: {paper_content['methods']}"
        
        # Run the experiment batch
        start_time = time.time()
        print(f"\n[INFO] Starting experiment: {experiment_name}")
        
        results = self._run_idea_generation_batch(
            prompt=config["main_prompt"],
            system_prompt=config["system_prompt"],
            context=context,
            run_id=run_id
        )
        
        # Store results for this experiment
        self.experiment_results[experiment_name] = results
        
        total_time = time.time() - start_time
        print(f"[INFO] Experiment completed in {total_time:.1f} seconds")
        
        return results
    
    def _run_idea_generation_batch(
        self, 
        prompt: str, 
        system_prompt: str, 
        context: str, 
        run_id: str
    ) -> Dict[str, Any]:
        """
        Run the idea generation batch using the imported function.
        
        Args:
            prompt: Main prompt for idea generation
            system_prompt: System prompt to use
            context: Context information for the model
            run_id: Unique identifier for this run
            
        Returns:
            Dictionary containing experiment results
        """
        from experiment_runner_templates import run_idea_generation_batch
        
        # Run the experiment using the existing function
        results = run_idea_generation_batch(
            prompt=prompt,
            llama_fn=lambda p, context=None: self.llama_fn(
                prompt=f"{context}\n\n{p}" if context else p, 
                system_prompt=system_prompt
            ),
            model_name=self.model_name,
            run_id=run_id,
            quality_evaluator=self._evaluate_quality,
            tracker=self.tracker,
            context=context,
            num_ideas=self.num_ideas
        )
        
        return results
    
    def _evaluate_quality(self, idea: str, context: str = None) -> float:
        """
        Dummy quality evaluation method to be overridden by subclasses.
        
        Args:
            idea: The generated idea to evaluate
            context: Optional context information
            
        Returns:
            Quality score between 0 and 1
        """
        # Default implementation - can be overridden by subclasses
        # For now, just return a placeholder score
        # In a real implementation, you would want to use an actual evaluator
        return 0.7
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the results of all experiments.
        
        Returns:
            Dictionary containing analysis results
        """
        if not self.experiment_results:
            print("[WARNING] No experiment results to analyze")
            return {}
        
        analysis = {}
        
        # Compare distributions across experiments
        metrics = ["cosine_similarities", "self_bleu_scores", "bertscore_scores"]
        
        for metric in metrics:
            metric_data = {}
            for exp_name, results in self.experiment_results.items():
                if metric in results and results[metric]:
                    metric_data[exp_name] = {
                        "values": results[metric],
                        "mean": np.mean(results[metric]),
                        "std": np.std(results[metric]),
                        "min": min(results[metric]),
                        "max": max(results[metric])
                    }
            
            analysis[metric] = metric_data
        
        # Compare KDE distributions
        kde_metrics = ["cosine", "self_bleu", "bertscore"]
        kde_analysis = {}
        
        for kde_metric in kde_metrics:
            kde_data = {}
            for exp_name, results in self.experiment_results.items():
                if "kde_values" in results and kde_metric in results["kde_values"]:
                    kde_values = results["kde_values"][kde_metric]
                    if kde_values["x"] and kde_values["y"]:
                        # Find the mode (peak density)
                        peak_idx = kde_values["y"].index(max(kde_values["y"]))
                        mode_value = kde_values["x"][peak_idx]
                        
                        kde_data[exp_name] = {
                            "x": kde_values["x"],
                            "y": kde_values["y"],
                            "mode": mode_value,
                            "range": [min(kde_values["x"]), max(kde_values["x"])]
                        }
            
            kde_analysis[kde_metric] = kde_data
        
        analysis["kde"] = kde_analysis
        
        # Print a summary of the analysis
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _print_analysis_summary(self, analysis: Dict[str, Any]) -> None:
        """
        Print a summary of the analysis results.
        
        Args:
            analysis: Dictionary containing analysis results
        """
        print("\n===== Experiment Analysis Summary =====")
        
        # Print summary statistics for each metric
        for metric_name, metric_data in analysis.items():
            if metric_name != "kde":
                print(f"\n{metric_name.upper()} COMPARISON:")
                for exp_name, stats in metric_data.items():
                    print(f"  {exp_name}:")
                    print(f"    Mean: {stats['mean']:.4f}")
                    print(f"    Std:  {stats['std']:.4f}")
                    print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Print KDE mode values for comparison
        if "kde" in analysis:
            print("\nKDE MODE VALUES (peak density point):")
            for kde_metric, kde_data in analysis["kde"].items():
                print(f"  {kde_metric}:")
                for exp_name, stats in kde_data.items():
                    print(f"    {exp_name}: {stats['mode']:.4f} (range: {stats['range'][0]:.4f} - {stats['range'][1]:.4f})")
        
        print("\n=======================================")


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
    
    def _evaluate_quality(self, idea: str, context: str = None) -> float:
        """
        Evaluate the quality of a hypothesis.
        
        Args:
            idea: The generated hypothesis to evaluate
            context: Optional context information
            
        Returns:
            Quality score between 0 and 1
        """
        if self.evaluator:
            return self.evaluator.evaluate_hypothesis(idea, context=context)
        return super()._evaluate_quality(idea, context)


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
        system_prompt = generate_scientific_system_prompt(
            abstract=paper_content['abstract']
        )
        
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
        # Generate system prompt similar to scientific method
        system_prompt = generate_scientific_system_prompt(
            abstract=paper_content['abstract']
        )
        
        # Create combined text for few-shot prompt
        combined_text = f"Abstract: {paper_content['abstract']}\n\nMethods: {paper_content['methods']}"
        
        # Generate the few-shot prompt
        main_prompt = generate_few_shot_prompt(
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