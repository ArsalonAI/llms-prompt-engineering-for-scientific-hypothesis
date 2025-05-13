import os
import wandb
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.figure_factory as ff
import plotly.graph_objects as go

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

def create_distribution_plot(data, metric_name):
    """Create a distribution plot for a given metric."""
    fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30, name=metric_name)])
    fig.add_trace(go.Scatter(
        x=[np.mean(data), np.mean(data)],
        y=[0, fig.data[0].y.max()],
        mode='lines',
        name='Mean',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=[np.median(data), np.median(data)],
        y=[0, fig.data[0].y.max()],
        mode='lines',
        name='Median',
        line=dict(color='green', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{metric_name} Distribution',
        xaxis_title=metric_name,
        yaxis_title='Count',
        showlegend=True
    )
    return fig

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
        print(f"[INFO] Started experiment: {run.name}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Could not initialize wandb: {str(e)}")
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
            # Print minimal console output
            print(f"[STEP] Run ID: {experiment_data[0]} | "
                  f"Quality: {experiment_data[3]} | "
                  f"Metrics: cos_sim={experiment_data[5]:.3f}, "
                  f"self_bleu={experiment_data[6]:.3f}, "
                  f"bert_score={experiment_data[7]:.3f}")
            
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
            
            # Create a pandas DataFrame
            df = pd.DataFrame(_results_data)
            
            # Log metrics
            wandb.log({
                "metrics/cosine_similarity": float(experiment_data[5]),
                "metrics/self_bleu": float(experiment_data[6]),
                "metrics/bertscore": float(experiment_data[7]),
                "metrics/running_avg_cosine": df["Cosine Similarity"].mean(),
                "metrics/running_avg_self_bleu": df["Self BLEU"].mean(),
                "metrics/running_avg_bertscore": df["BERTScore"].mean(),
                "metrics/total_ideas": len(df),
                "metrics/pruned_ideas": df["Is Pruned"].sum()
            })
            
            # Update distribution plots every 10 steps
            if len(df) % 10 == 0:
                cosine_dist = create_distribution_plot(df["Cosine Similarity"].values, "Cosine Similarity")
                self_bleu_dist = create_distribution_plot(df["Self BLEU"].values, "Self BLEU")
                
                wandb.log({
                    "distributions/cosine_similarity": wandb.Plotly(cosine_dist),
                    "distributions/self_bleu": wandb.Plotly(self_bleu_dist)
                })
            
        except Exception as e:
            print(f"[ERROR] Could not log to wandb: {str(e)}")

def finish_wandb_run():
    """
    Finalize the W&B run and create a final summary table.
    """
    if wandb.run and _results_data:
        try:
            # Create final DataFrame
            df = pd.DataFrame(_results_data)
            
            # Calculate summary statistics
            summary = {
                "Total Ideas Generated": len(df),
                "Ideas Pruned": df["Is Pruned"].sum(),
                "Cosine Similarity": {
                    "mean": df["Cosine Similarity"].mean(),
                    "median": df["Cosine Similarity"].median(),
                    "std": df["Cosine Similarity"].std()
                },
                "Self BLEU": {
                    "mean": df["Self BLEU"].mean(),
                    "median": df["Self BLEU"].median(),
                    "std": df["Self BLEU"].std()
                },
                "BERTScore": {
                    "mean": df["BERTScore"].mean(),
                    "median": df["BERTScore"].median(),
                    "std": df["BERTScore"].std()
                }
            }
            
            # Create final distribution plots
            final_cosine_dist = create_distribution_plot(df["Cosine Similarity"].values, "Final Cosine Similarity")
            final_self_bleu_dist = create_distribution_plot(df["Self BLEU"].values, "Final Self BLEU")
            
            # Log final artifacts
            wandb.log({
                "final/results_table": wandb.Table(dataframe=df),
                "final/summary": wandb.Table(
                    columns=["Metric", "Value"],
                    data=[
                        ["Total Ideas", summary["Total Ideas Generated"]],
                        ["Pruned Ideas", summary["Ideas Pruned"]],
                        ["Cosine Similarity (mean ± std)", f"{summary['Cosine Similarity']['mean']:.3f} ± {summary['Cosine Similarity']['std']:.3f}"],
                        ["Self BLEU (mean ± std)", f"{summary['Self BLEU']['mean']:.3f} ± {summary['Self BLEU']['std']:.3f}"],
                        ["BERTScore (mean ± std)", f"{summary['BERTScore']['mean']:.3f} ± {summary['BERTScore']['std']:.3f}"]
                    ]
                ),
                "final/cosine_distribution": wandb.Plotly(final_cosine_dist),
                "final/self_bleu_distribution": wandb.Plotly(final_self_bleu_dist)
            })
            
            # Print minimal summary to terminal
            print("\n[SUMMARY]")
            print(f"Total Ideas: {summary['Total Ideas Generated']}")
            print(f"Pruned Ideas: {summary['Ideas Pruned']}")
            print(f"Cosine Similarity: {summary['Cosine Similarity']['mean']:.3f} ± {summary['Cosine Similarity']['std']:.3f}")
            print(f"Self BLEU: {summary['Self BLEU']['mean']:.3f} ± {summary['Self BLEU']['std']:.3f}")
            print(f"BERTScore: {summary['BERTScore']['mean']:.3f} ± {summary['BERTScore']['std']:.3f}\n")
            
        except Exception as e:
            print(f"[ERROR] Could not create final summary: {str(e)}")
        
        finally:
            wandb.finish()