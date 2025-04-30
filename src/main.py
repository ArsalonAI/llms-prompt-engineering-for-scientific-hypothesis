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

# First experiment: CRISPR gene editing hypothesis generation
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

    # Start the experiment with all configuration
    tracker.start_experiment(
        experiment_name="llm-scientific-hypothesis-generation",
        experiment_type="idea_generation",
        model_name="llama-3-3-70b",
        config={
            "domain": "genetic engineering",
            "focus_area": "CRISPR gene editing",
            "num_ideas": 25,
            "system_prompt": system_prompt,
            "main_prompt": main_prompt,
            "evaluation_criteria": evaluator.get_evaluation_criteria()
        }
    )
    
    # Run the experiment
    results = run_idea_generation_batch(
        prompt=main_prompt,
        llama_fn=lambda p: llama_3_3_70B_completion(p, system_prompt=system_prompt),
        model_name="llama-3-3-70b",
        run_id="crispr_gene_editing_001",
        quality_evaluator=evaluator.evaluate_hypothesis,
        tracker=tracker,
        context={
            'abstract': paper_content['abstract'],
            'methods': paper_content['methods']
        }
    )

