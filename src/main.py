# main.py
import warnings
# Suppress specific Hugging Face warnings for RoBERTa model initialization
# Attempting to place these as early as possible.
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized:.*", category=UserWarning)
warnings.filterwarnings("ignore", message="You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.", category=UserWarning)

import os
from completion_util import llama_3_3_70B_completion
from experiment_tracker import ExperimentTracker
from pdf_util import extract_paper_content
from experiment_runners import (
    ScientificHypothesisRunner,
    RoleBasedHypothesisRunner,
    FewShotHypothesisRunner
)
from src.statistical_analysis import StatisticalAnalyzer
from src.cross_experiment_analysis.runner import run_cross_experiment_analysis

# Get the absolute path to the src directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
EXPERIMENT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiment_results")

# Process the research paper using absolute path
def load_paper_content():
    """Load and process the paper content."""
    pdf_path = os.path.join(PROJECT_ROOT, 'src', 'research_papers', 'nihms-1761240.pdf')
    print(f"Loading PDF from: {pdf_path}")
    return extract_paper_content(pdf_path)

def run_experiments(paper_content, num_ideas=10):
    """
    Run multiple experiments with different prompt techniques.
    
    Args:
        paper_content: Dictionary containing paper content
        num_ideas: Number of ideas to generate per experiment
    
    Returns:
        Dictionary containing experiment results
    """
    experiment_results = {}
    
    # Initialize experiment tracker
    with ExperimentTracker(output_dir=EXPERIMENT_RESULTS_DIR) as tracker:
        # Initialize experiment runners
        scientific_runner = ScientificHypothesisRunner(
            tracker=tracker,
            llama_fn=llama_3_3_70B_completion,
            model_name="llama-3-3-70b",
            domain="genetic engineering",
            focus_area="CRISPR gene editing",
            num_ideas=num_ideas
        )
        
        role_based_runner = RoleBasedHypothesisRunner(
            tracker=tracker,
            llama_fn=llama_3_3_70B_completion,
            model_name="llama-3-3-70b",
            domain="genetic engineering",
            focus_area="CRISPR gene editing",
            num_ideas=num_ideas
        )
        
        few_shot_runner = FewShotHypothesisRunner(
            tracker=tracker,
            llama_fn=llama_3_3_70B_completion,
            model_name="llama-3-3-70b",
            domain="genetic engineering",
            focus_area="CRISPR gene editing",
            num_ideas=num_ideas
        )
        
        # Run experiments
        print("\n=== Running Scientific Hypothesis Experiment ===")
        scientific_results = scientific_runner.run("Scientific_Hypothesis", paper_content)
        experiment_results["Scientific_Hypothesis"] = scientific_results
        
        print("\n=== Running Role-Based Hypothesis Experiment ===")
        role_based_results = role_based_runner.run("Role_Based_Hypothesis", paper_content)
        experiment_results["Role_Based_Hypothesis"] = role_based_results
        
        print("\n=== Running Few-Shot Hypothesis Experiment ===")
        few_shot_results = few_shot_runner.run("Few_Shot_Hypothesis", paper_content)
        experiment_results["Few_Shot_Hypothesis"] = few_shot_results
        
        # Collect all experiment results
        all_experiment_results = {
            "Scientific_Hypothesis": scientific_results,
            "Role_Based_Hypothesis": role_based_results,
            "Few_Shot_Hypothesis": few_shot_results
        }
        
    return all_experiment_results

def main():
    """Main function to run all experiments."""
    # Create experiment results directory and all subdirectories if they don't exist
    os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)
    
    # Create required subdirectories
    subdirs = [
        "cross_experiment_analysis",
        "statistical_analysis",
        "cross_experiment_analysis/plots"
    ]
    
    for subdir in subdirs:
        full_path = os.path.join(EXPERIMENT_RESULTS_DIR, subdir)
        os.makedirs(full_path, exist_ok=True)
        print(f"[INFO] Ensured directory exists: {full_path}")
    
    # Load paper content
    paper_content = load_paper_content()
    
    # Run experiments
    print("\n=== Starting Prompt Engineering Experiments ===")
    experiment_results = run_experiments(paper_content, num_ideas=10)
    
    print("\n=== Experiments Completed Successfully ===")
    
    # Perform statistical analysis (combines current and historical data)
    print("\n=== Running Statistical Analysis ===")
    analyzer = StatisticalAnalyzer(
        current_results=experiment_results,
        experiment_dir=EXPERIMENT_RESULTS_DIR
    )
    
    analysis_results = analyzer.perform_analysis()
    conclusions = analysis_results["conclusions"]
    
    # Output key findings
    print("\n=== Evidence-Based Conclusions ===")
    for finding in conclusions["key_findings"]:
        print(f"- {finding}")
    
    # Print method scores
    print("\n=== Method Effectiveness ===")
    for method, score in sorted(conclusions["method_scores"].items(), key=lambda x: x[1], reverse=True):
        print(f"- {method}: {score} statistical wins")
    
    # Print recommendations
    print("\n=== Recommendations ===")
    for recommendation in conclusions["recommendations"]:
        print(f"- {recommendation}")
    
    # Print dashboard path
    dashboard_path = analysis_results["dashboard_path"]
    print(f"\nView statistical analysis dashboard at: file://{os.path.abspath(dashboard_path)}")
    
if __name__ == "__main__":
    main()

