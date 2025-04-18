import os
import json
import time
from datetime import datetime
from wandb_utils import log_experiment_to_wandb
from similarity_metrics_utils import get_cosine_similarity, get_llm_similarity_category
from typing import Dict, Any, List, Optional

_previous_completions = []


def run_experiment(prompt, experiment_type, llama_fn, model_name, system_prompt=None, run_id=None):
    global _previous_completions

    start_time = time.time()
    completion = llama_fn(prompt, system_prompt=system_prompt)
    elapsed_time = time.time() - start_time

    cosine_sim = get_cosine_similarity(completion, _previous_completions)
    llm_cat, llm_score = get_llm_similarity_category(llama_fn, prompt, completion, _previous_completions)
    _previous_completions.append(completion)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_identifier = f"run_{run_id}" if run_id else datetime.now().strftime('%H%M%S_%f')

    experiment_data = [
        timestamp, experiment_type, system_prompt or "", prompt, completion,
        f"{elapsed_time:.2f}", model_name, run_identifier,
        f"{cosine_sim:.3f}" if cosine_sim is not None else "",
        llm_cat or "", f"{llm_score:.2f}" if llm_score is not None else "",
        {
            "elapsed_time": elapsed_time,
            "prompt_length": len(prompt),
            "completion_length": len(completion),
            "experiment_type": experiment_type,
            "run_id": run_identifier,
            "cosine_similarity": cosine_sim or 0,
            "llm_similarity_score": llm_score or 0,
            "llm_similarity_category": llm_cat or ""
        }
    ]

    log_experiment_to_wandb(timestamp, experiment_data)
    return completion


class ExperimentTracker:
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = output_dir
        self.ensure_output_dir()
        self.current_experiment = None
        
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def start_experiment(self, experiment_name: str, config: Dict[str, Any]):
        """Start tracking a new experiment."""
        self.current_experiment = {
            'name': experiment_name,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'results': []
        }
    
    def log_result(self, result: Dict[str, Any]):
        """Log a result for the current experiment."""
        if self.current_experiment is None:
            raise ValueError("No experiment currently running. Call start_experiment first.")
        
        self.current_experiment['results'].append({
            'timestamp': datetime.now().isoformat(),
            **result
        })
    
    def end_experiment(self):
        """End the current experiment and save results."""
        if self.current_experiment is None:
            raise ValueError("No experiment currently running.")
        
        self.current_experiment['end_time'] = datetime.now().isoformat()
        
        # Save to file
        filename = f"{self.current_experiment['name']}_{self.current_experiment['start_time']}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.current_experiment, f, indent=2)
        
        self.current_experiment = None