# main.py
from experiment_runner_templates import run_idea_generation_batch
from completion_util import llama_3_3_70B_completion
from experiment_tracker import ExperimentTracker
from evaluation_utils import HypothesisEvaluator
from pdf_util import extract_paper_content
from prompts.scientific_prompts import generate_scientific_system_prompt, generate_scientific_hypothesis_prompt
import os

# Get the absolute path to the src directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Process the research paper using absolute path
pdf_path = os.path.join(PROJECT_ROOT, 'src', 'research_papers', 'nihms-1761240.pdf')
print(f"Loading PDF from: {pdf_path}")
paper_content = extract_paper_content(pdf_path)

# First experiment: Novel Hypothesis Generation
with ExperimentTracker() as tracker:
    # Initialize experiment-specific evaluator
    evaluator = HypothesisEvaluator()
    
    # Generate experiment-specific prompts
    system_prompt = generate_scientific_system_prompt(
        abstract=paper_content['abstract']
    )
    main_prompt = generate_scientific_hypothesis_prompt(
        domain="genetic engineering",
        focus_area="CRISPR gene editing",
        context={
            'abstract': paper_content['abstract'],
            'methods': paper_content['methods']
        }
    )

    # Experiment #1 - 

    # Create experiment configuration
    idea_generation_config = {
        "domain": "genetic engineering",
        "focus_area": "CRISPR gene editing",
        "num_ideas": 15,
        "batch_size": 5,  # Process 5 ideas at a time
        "system_prompt": system_prompt,
        "main_prompt": main_prompt,
        "evaluation_criteria": evaluator.get_evaluation_criteria()
    }

    # Start the experiment with configuration
    tracker.start_experiment(
        experiment_name="llm-scientific-hypothesis-generation",
        experiment_type="idea_generation",
        model_name="llama-3-3-70b",
        config=idea_generation_config
    )
    
    # Generate a unique run ID
    run_id = tracker.generate_run_id("run")
    
    # Run the experiment
    results = run_idea_generation_batch(
        prompt=main_prompt,
        llama_fn=lambda p, context=None: llama_3_3_70B_completion(
            prompt=f"{context}\n\n{p}" if context else p, 
            system_prompt=system_prompt
        ),
        model_name="llama-3-3-70b",
        run_id=run_id,
        quality_evaluator=evaluator.evaluate_hypothesis,
        tracker=tracker,
        context=f"Abstract: {paper_content['abstract']}\nMethods: {paper_content['methods']}",
        num_ideas=idea_generation_config["num_ideas"]
    )

    # Print KDE results summary
    print("\nKDE Analysis Results:")
    for metric, kde_data in results["kde_values"].items():
        if kde_data["x"] and kde_data["y"]:
            try:
                # Find the peak of the KDE curve (mode of the distribution)
                peak_idx = kde_data["y"].index(max(kde_data["y"]))
                mode_value = kde_data["x"][peak_idx]
                
                # Calculate the range
                x_min = min(kde_data["x"])
                x_max = max(kde_data["x"])
                
                # Calculate the area under the KDE curve (approximation)
                area = sum(kde_data["y"]) * (x_max - x_min) / len(kde_data["x"])
                
                print(f"\n{metric.title()} Distribution:")
                print(f"  - Sample size: {len(results[metric+'_scores']) if metric+'_scores' in results else 'N/A'}")
                print(f"  - KDE points: {len(kde_data['x'])}")
                print(f"  - Mode (peak density): {mode_value:.4f}")
                print(f"  - Range: [{x_min:.4f}, {x_max:.4f}]")
                print(f"  - Span: {x_max - x_min:.4f}")
                print(f"  - Relative density (area under curve): {area:.4f}")
            except (IndexError, ValueError) as e:
                print(f"\n{metric.title()} Distribution: Error analyzing KDE - {str(e)}")
        else:
            print(f"\n{metric.title()} Distribution: No KDE data available")

