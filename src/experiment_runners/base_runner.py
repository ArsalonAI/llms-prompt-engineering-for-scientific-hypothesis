"""
Base experiment runner module.
This module provides a base class for experiment runners.
"""
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Callable, Union

from experiment_tracker import ExperimentTracker


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
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        max_tokens: int = 1024
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
            temperature: Temperature for the language model
            top_p: Top_p for the language model
            top_k: Top_k for the language model
            repetition_penalty: Repetition penalty for the language model
            max_tokens: Maximum tokens for the language model
        """
        self.tracker = tracker
        self.llama_fn = llama_fn
        self.model_name = model_name
        self.domain = domain
        self.focus_area = focus_area
        self.num_ideas = num_ideas
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
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
        base_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens
        }
        print(f"[BaseRunner] prepare_experiment returning base LLM hyperparams in config: {base_config}")
        return base_config
    
    def run(self, experiment_name: str, paper_content: Dict[str, str], 
             skip_intermediate_calculations: bool = True,
             temperature: Optional[float] = None,
             top_p: Optional[float] = None,
             top_k: Optional[int] = None,
             repetition_penalty: Optional[float] = None,
             max_tokens: Optional[int] = None
            ) -> Dict[str, Any]:
        """
        Run the experiment.
        
        Args:
            experiment_name: Name of the experiment
            paper_content: Dictionary containing paper content
            skip_intermediate_calculations: Whether to skip intermediate calculations for speed
            temperature: Temperature for the language model
            top_p: Top_p for the language model
            top_k: Top_k for the language model
            repetition_penalty: Repetition penalty for the language model
            max_tokens: Maximum tokens for the language model
            
        Returns:
            Dictionary containing experiment results
        """
        config = self.prepare_experiment(experiment_name, paper_content)
        
        current_temp = temperature if temperature is not None else self.temperature
        current_top_p = top_p if top_p is not None else self.top_p
        current_top_k = top_k if top_k is not None else self.top_k
        current_rep_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        config.update({
            "temperature": current_temp,
            "top_p": current_top_p,
            "top_k": current_top_k,
            "repetition_penalty": current_rep_penalty,
            "max_tokens": current_max_tokens,
            "base_model_name": self.model_name,
            "num_ideas_requested": self.num_ideas
        })
        
        self.tracker.start_experiment(
            experiment_name=experiment_name,
            experiment_type=config.get("prompt_strategy_type", "idea_generation"),
            model_name=self.model_name,
            config=config
        )
        
        run_id = self.tracker.generate_run_id("run")
        
        abstract_content = paper_content.get('abstract', '')
        methods_content = paper_content.get('methods', '')
        context = f"Abstract: {abstract_content}\nMethods: {methods_content}"
        preview = context[:100].replace('\n', ' ')
        print(f"[DEBUG] BaseExperimentRunner: Generated context string. Length: {len(context)}. Preview: '{preview}...'" )
        if not abstract_content:
            print("[WARNING] BaseExperimentRunner: Abstract missing or empty from paper_content.")
        if not methods_content:
            print("[WARNING] BaseExperimentRunner: Methods missing or empty from paper_content.")
        
        start_time = time.time()
        print(f"\n[INFO] Starting experiment: {experiment_name}")
        
        results = self._run_idea_generation_batch(
            prompt=config["main_prompt"],
            system_prompt=config["system_prompt"],
            context=context,
            run_id=run_id,
            skip_intermediate_calculations=skip_intermediate_calculations,
            temperature=current_temp,
            top_p=current_top_p,
            top_k=current_top_k,
            repetition_penalty=current_rep_penalty,
            max_tokens=current_max_tokens
        )
        
        self.experiment_results[experiment_name] = results
        
        total_time = time.time() - start_time
        print(f"[INFO] Experiment completed in {total_time:.1f} seconds")
        
        return results
    
    def _run_idea_generation_batch(
        self, 
        prompt: str, 
        system_prompt: str, 
        context: str, 
        run_id: str,
        skip_intermediate_calculations: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Run the idea generation batch using the imported function.
        
        Args:
            prompt: Main prompt for idea generation
            system_prompt: System prompt to use
            context: Context information for the model
            run_id: Unique identifier for this run
            skip_intermediate_calculations: Whether to skip intermediate calculations for speed
            temperature: Temperature for the language model
            top_p: Top_p for the language model
            top_k: Top_k for the language model
            repetition_penalty: Repetition penalty for the language model
            max_tokens: Maximum tokens for the language model
            
        Returns:
            Dictionary containing experiment results
        """
        from experiment_runner_templates import run_idea_generation_batch
        
        results = run_idea_generation_batch(
            prompt=prompt,
            llama_fn=lambda p, context=None: self.llama_fn(
                prompt=f"{context}\n\n{p}" if context else p, 
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens
            ),
            model_name=self.model_name,
            run_id=run_id,
            quality_evaluator=self._evaluate_quality,
            tracker=self.tracker,
            context=context,
            num_ideas=self.num_ideas,
            skip_intermediate_calculations=skip_intermediate_calculations
        )
        
        return results
    
    def _evaluate_quality(self, idea: str, context: str = None) -> Dict[str, Any]:
        """
        Dummy quality evaluation method to be overridden by subclasses.
        
        Args:
            idea: The generated idea to evaluate
            context: Optional context information
            
        Returns:
            Dictionary containing quality evaluation results with at least 'evaluation'
            and 'evaluation_full' keys
        """
        # Default implementation - can be overridden by subclasses
        # Always return a dictionary with a consistent format
        placeholder_score = 0.7
        return {
            "evaluation": f"SCORE: {placeholder_score:.1f}",
            "evaluation_full": f"Evaluation score: {placeholder_score:.1f}\n\nThis is a placeholder evaluation because no specific evaluator is available for this runner type.",
            "score": placeholder_score
        }
    
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