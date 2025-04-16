import os
import wandb
from datetime import datetime

_experiments_table = None
_previous_completions = []

# Define default experiment table columns directly in this file
DEFAULT_TABLE_COLUMNS = [
    "timestamp", "experiment_type", "system_prompt", "user_prompt",
    "completion", "elapsed_time", "model", "run_id",
    "cosine_similarity", "llm_similarity_category", "llm_similarity_score"
]

def init_wandb(project_name=None, entity_name=None, model_name=None, table_columns=None):
    """
    Initialize Weights & Biases (wandb) logging.

    Args:
        project_name (str): Name of the wandb project.
        entity_name (str): Name of the wandb entity (user or team).
        model_name (str): Optional model name to log.
        table_columns (list): Optional custom columns for the experiment table.
    """
    global _experiments_table, _previous_completions
    _previous_completions = []

    try:
        config_dict = {"model": model_name} if model_name else {}

        run = wandb.init(
            project=project_name or os.getenv('WANDB_PROJECT', 'prompt-engineering-experiments'),
            entity=entity_name or os.getenv('WANDB_ENTITY'),
            name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config_dict,
            settings=wandb.Settings(_disable_stats=True, mode="online")
        )

        _experiments_table = wandb.Table(columns=table_columns or DEFAULT_TABLE_COLUMNS)

        print(f"Initialized wandb run: {run.name}")
        return True

    except Exception as e:
        print(f"Warning: Could not initialize wandb: {str(e)}")
        return False


def log_experiment_to_wandb(timestamp, experiment_data):
    """
    Log experiment data to the wandb table.

    Args:
        timestamp (str): Timestamp of the experiment.
        experiment_data (list): Data row to log. The final element should be a dictionary of metrics.
    """
    if wandb.run and _experiments_table:
        try:
            # All elements except the last are added to the wandb table
            _experiments_table.add_data(*experiment_data[:-1])
            wandb.log({"experiments": _experiments_table})
            # The last element is expected to be a dict of additional metrics
            wandb.log(experiment_data[-1])
        except Exception as e:
            print(f"Warning: Could not log to wandb: {str(e)}")