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
    FewShotHypothesisRunner,
    ChainOfThoughtHypothesisRunner
)
from statistical_analysis import StatisticalAnalyzer
from cross_experiment_analysis.runner import run_cross_experiment_analysis

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

def run_experiments(paper_content, num_ideas=5):
    all_run_results = {} # To store results from all hyperparameter variants

    # Define hyperparameter configurations to test
    hyperparam_configs = [
        {"temp": 0.7, "top_p": 0.7, "top_k": 50, "label": "default"}, # Baseline
        {"temp": 0.9, "top_p": 0.7, "top_k": 50, "label": "temp0.9"},
        {"temp": 0.5, "top_p": 0.7, "top_k": 50, "label": "temp0.5"},
        {"temp": 0.7, "top_p": 0.9, "top_k": 50, "label": "top_p0.9"},
        # Add more configurations as needed
    ]

    with ExperimentTracker(output_dir=EXPERIMENT_RESULTS_DIR) as tracker:
        for hp_config in hyperparam_configs:
            print(f"\n=== Running Experiments with Hyperparams: {hp_config['label']} ===")
            current_temp = hp_config['temp']
            current_top_p = hp_config['top_p']
            current_top_k = hp_config['top_k']
            # Assuming repetition_penalty and max_tokens remain default for now, or add them to hp_config

            # Instantiate runners with current hyperparameters
            scientific_runner = ScientificHypothesisRunner(
                tracker=tracker, llama_fn=llama_3_3_70B_completion, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                num_ideas=num_ideas, temperature=current_temp, top_p=current_top_p, top_k=current_top_k
            )
            role_based_runner = RoleBasedHypothesisRunner(
                tracker=tracker, llama_fn=llama_3_3_70B_completion, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                num_ideas=num_ideas, temperature=current_temp, top_p=current_top_p, top_k=current_top_k
            )
            few_shot_runner = FewShotHypothesisRunner(
                tracker=tracker, llama_fn=llama_3_3_70B_completion, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                num_ideas=num_ideas, temperature=current_temp, top_p=current_top_p, top_k=current_top_k
            )
            chain_of_thought_runner = ChainOfThoughtHypothesisRunner(
                tracker=tracker, llama_fn=llama_3_3_70B_completion, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                num_ideas=num_ideas, temperature=current_temp, top_p=current_top_p, top_k=current_top_k
            )

            runner_setups = [
                {"runner": scientific_runner, "name_prefix": "Scientific_Hypothesis"},
                {"runner": role_based_runner, "name_prefix": "Role_Based_Hypothesis"},
                {"runner": few_shot_runner, "name_prefix": "Few_Shot_Hypothesis"},
                {"runner": chain_of_thought_runner, "name_prefix": "Chain_Of_Thought_Hypothesis"}
            ]

            for setup in runner_setups:
                # Construct a unique experiment name including the hyperparameter label
                experiment_name_full = f"{setup['name_prefix']}__{hp_config['label']}"
                print(f"\n=== Running: {experiment_name_full} ===")
                
                results = setup["runner"].run(
                    experiment_name_full, 
                    paper_content,
                    skip_intermediate_calculations=True
                    # The runner.run() will use the hyperparams it was initialized with
                )
                all_run_results[experiment_name_full] = results
        
    return all_run_results

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
    print("\n=== Starting Prompt Engineering Experiments with Hyperparameter Variations ===")
    all_experiment_results_with_hparams = run_experiments(paper_content, num_ideas=50)
    
    print("\n=== All Experiments Completed Successfully ===")
    
    # Perform statistical analysis (combines current and historical data)
    print("\n=== Running Statistical Analysis ===")
    analyzer = StatisticalAnalyzer(
        current_results=all_experiment_results_with_hparams,
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
    
    # Print detailed statistical test results
    print("\n=== Detailed Statistical Test Results ===")
    if "statistical_tests" in analysis_results.get("comparison_results", {}):
        for metric, tests in analysis_results["comparison_results"]["statistical_tests"].items():
            print(f"\n--- {metric.title()} --- ")
            header = "Comparison".ljust(50) + "KS p-val".ljust(12) + "Effect Size".ljust(15) + "Interpretation"
            print(header)
            print("-" * len(header))
            
            if not tests:
                print("No statistical test results available for this metric.")
                continue
            
            # Print Kruskal-Wallis test result first if available
            if "kruskal_wallis" in tests:
                kw_test = tests["kruskal_wallis"]
                kw_line = "Kruskal-Wallis (overall)".ljust(50)
                kw_line += f"{kw_test['p_value']:.4f}".ljust(12)
                kw_line += "N/A".ljust(15)
                kw_line += "N/A"
                print(kw_line)
            
            # Print pairwise KS test results
            for comparison, result in tests.items():
                if comparison == "kruskal_wallis":  # Skip, already printed
                    continue
                    
                ks_p = result.get("ks_test", {}).get("p_value", float('nan'))
                effect_size = result.get("effect_size", {}).get("cohen_d", float('nan'))
                interpretation = result.get("effect_size", {}).get("interpretation", "unknown")
                
                line = comparison.ljust(50)
                line += f"{ks_p:.4f}".ljust(12)
                line += f"{effect_size:.4f}".ljust(15)
                line += interpretation
                print(line)
    else:
        print("No statistical test results available.")

    # Print dashboard path
    dashboard_path = analysis_results["dashboard_path"]
    print(f"\nView statistical analysis dashboard at: file://{os.path.abspath(dashboard_path)}")
    
if __name__ == "__main__":
    main()

