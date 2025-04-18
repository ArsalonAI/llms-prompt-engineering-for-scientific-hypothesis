import os
import wandb
import pandas as pd
from datetime import datetime

_experiments_table = None
_results_data = []  # Store all results for final table

# Define default experiment table columns directly in this file
DEFAULT_TABLE_COLUMNS = [
    "run_id",              # ID of the run or experiment
    "idea",                # LLM-generated idea
    "batch_prompt",        # Prompt that generated the batch
    "judged_quality",      # Output of LLM judge (stupid / not stupid)
    "is_pruned",          # Boolean - whether idea passed filter
    "cosine_sim",         # Avg similarity with other ideas
    "self_bleu",          # Diversity metric
    "bertscore"           # Semantic token alignment score (novelty/quality)
]

def init_wandb(project_name=None, entity_name=None, model_name=None, table_columns=None):
    """
    Initialize Weights & Biases (wandb) logging.
    """
    global _experiments_table, _results_data
    _results_data = []  # Reset results data
    
    try:
        config_dict = {"model": model_name} if model_name else {}
        
        run = wandb.init(
            project=project_name or os.getenv('WANDB_PROJECT', 'prompt-engineering-experiments'),
            entity=entity_name or os.getenv('WANDB_ENTITY'),
            name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config_dict,
            settings=wandb.Settings(_disable_stats=True, mode="online")
        )
        
        # Create the table with the specified columns
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
        timestamp: Current timestamp
        experiment_data: List containing the data to log, matching the table columns
    """
    global _results_data
    
    if wandb.run and _experiments_table:
        try:
            # Print results to terminal for visibility
            print("\n=== Experiment Results ===")
            print(f"Run ID: {experiment_data[0]}")
            print(f"Idea: {experiment_data[1][:200]}...")  # Truncate long ideas for display
            print(f"Judged Quality: {experiment_data[3]}")
            print(f"Is Pruned: {experiment_data[4]}")
            print(f"Cosine Similarity: {experiment_data[5]}")
            print(f"Self BLEU: {experiment_data[6]}")
            print(f"BERTScore: {experiment_data[7]}")
            print("========================\n")
            
            # Store the result
            result_dict = {
                "Run ID": experiment_data[0],
                "Idea": experiment_data[1],
                "Batch Prompt": experiment_data[2],
                "Judged Quality": experiment_data[3],
                "Is Pruned": experiment_data[4],
                "Cosine Similarity": float(experiment_data[5]),
                "Self BLEU": float(experiment_data[6]),
                "BERTScore": float(experiment_data[7]),
                "Timestamp": timestamp
            }
            _results_data.append(result_dict)
            
            # Create a pandas DataFrame for better visualization
            df = pd.DataFrame(_results_data)
            
            # Log as a new table for better visualization
            wandb.log({"results_table": wandb.Table(dataframe=df)})
            
            # Also log individual metrics
            wandb.log({
                "avg_cosine_similarity": df["Cosine Similarity"].mean(),
                "avg_self_bleu": df["Self BLEU"].mean(),
                "avg_bertscore": df["BERTScore"].mean(),
                "total_ideas": len(df),
                "pruned_ideas": df["Is Pruned"].sum()
            })
            
        except Exception as e:
            print(f"Warning: Could not log to wandb: {str(e)}")
            print(f"Data that failed to log: {experiment_data}")

def finish_wandb_run():
    """
    Finalize the W&B run and create a final summary table.
    """
    if wandb.run and _results_data:
        try:
            # Create final DataFrame
            df = pd.DataFrame(_results_data)
            
            # Create summary statistics
            summary = {
                "Total Ideas Generated": len(df),
                "Ideas Pruned": df["Is Pruned"].sum(),
                "Average Cosine Similarity": df["Cosine Similarity"].mean(),
                "Average Self BLEU": df["Self BLEU"].mean(),
                "Average BERTScore": df["BERTScore"].mean()
            }
            
            # Log final artifacts
            final_table = wandb.Table(dataframe=df)
            summary_table = wandb.Table(data=[[k, v] for k, v in summary.items()],
                                      columns=["Metric", "Value"])
            
            wandb.log({
                "final_results": final_table,
                "summary_statistics": summary_table
            })
            
            # Print summary to terminal
            print("\n=== Experiment Summary ===")
            for metric, value in summary.items():
                print(f"{metric}: {value:.3f}")
            print("========================\n")
            
        except Exception as e:
            print(f"Warning: Could not create final summary: {str(e)}")
        
        finally:
            wandb.finish()