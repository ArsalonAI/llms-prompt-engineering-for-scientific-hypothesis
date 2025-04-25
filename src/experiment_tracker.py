import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from wandb_utils import init_wandb, log_experiment_to_wandb, finish_wandb_run
from experiment_schemas import LLM_IDEA_GENERATION_COLUMNS, LLM_ITERATIVE_SYNTHESIS_COLUMNS

class ExperimentTracker:
    def __init__(self, output_dir: str = "experiment_results", use_wandb: bool = True):
        """
        Initialize the experiment tracker.
        
        Args:
            output_dir: Directory to store experiment results
            use_wandb: Whether to use Weights & Biases for tracking
        """
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.ensure_output_dir()
        self.current_experiment = None
        self._experiment_type = None
        
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def start_experiment(self, 
                        experiment_name: str, 
                        experiment_type: str = "idea_generation",
                        model_name: str = None,
                        config: Dict[str, Any] = None):
        """
        Start tracking a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            experiment_type: Type of experiment (idea_generation or iterative_synthesis)
            model_name: Name of the model being used
            config: Additional configuration parameters
        """
        self._experiment_type = experiment_type
        self.current_experiment = {
            'name': experiment_name,
            'type': experiment_type,
            'model': model_name,
            'config': config or {},
            'start_time': datetime.now().isoformat(),
            'results': []
        }
        
        if self.use_wandb:
            # Initialize W&B with appropriate columns based on experiment type
            columns = (LLM_IDEA_GENERATION_COLUMNS if experiment_type == "idea_generation" 
                      else LLM_ITERATIVE_SYNTHESIS_COLUMNS)
            init_wandb(
                project_name=experiment_name,
                model_name=model_name,
                table_columns=columns
            )
    
    def log_result(self, result: Dict[str, Any]):
        """
        Log a result for the current experiment.
        
        Args:
            result: Dictionary containing the result data
        """
        if self.current_experiment is None:
            raise ValueError("No experiment currently running. Call start_experiment first.")
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add timestamp to result
        result_with_timestamp = {
            'timestamp': timestamp,
            **result
        }
        
        # Store locally
        self.current_experiment['results'].append(result_with_timestamp)
        
        # Log to W&B if enabled
        if self.use_wandb:
            log_experiment_to_wandb(timestamp, result_with_timestamp)
    
    def end_experiment(self):
        """End the current experiment and save results."""
        if self.current_experiment is None:
            raise ValueError("No experiment currently running.")
        
        self.current_experiment['end_time'] = datetime.now().isoformat()
        
        # Save to local file
        filename = f"{self.current_experiment['name']}_{self.current_experiment['start_time']}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.current_experiment, f, indent=2)
        
        # Finish W&B run if enabled
        if self.use_wandb:
            finish_wandb_run()
        
        self.current_experiment = None
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current_experiment is not None:
            self.end_experiment()