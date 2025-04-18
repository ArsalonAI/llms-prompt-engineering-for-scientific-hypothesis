# main.py
from wandb_utils import init_wandb
from experiment_schemas import LLM_IDEA_GENERATION_COLUMNS
from experiment_runner_templates import run_idea_generation_batch
from completion_util import llama_3_3_70B_completion
from experiment_tracker import ExperimentTracker
from evaluation_utils import HypothesisEvaluator
from pdf_util import extract_text_from_pdf, extract_sections, clean_text
from prompts.scientific_prompts import generate_scientific_system_prompt, generate_scientific_hypothesis_prompt
import os

# Extract and process research paper
def extract_paper_content(pdf_path):
    """Extract and process content from a research paper PDF."""
    # Extract raw text
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    
    # Extract sections
    sections = extract_sections(cleaned_text)
    
    # Get abstract and methods sections
    abstract = sections.get('Abstract', [{'chunk': ''}])[0]['chunk']
    methods = sections.get('Methods', [{'chunk': ''}])[0]['chunk']
    
    return {
        'abstract': abstract,
        'methods': methods,
        'sections': sections
    }

# Process the research paper
pdf_path = os.path.join('research_papers', 'nihms-1761240.pdf')
paper_content = extract_paper_content(pdf_path)

# Initialize components
tracker = ExperimentTracker()
evaluator = HypothesisEvaluator()

# Generate prompts
system_prompt = generate_scientific_system_prompt(abstract=paper_content['abstract'])
main_prompt = generate_scientific_hypothesis_prompt(
    domain="genetic engineering",
    focus_area="CRISPR gene editing",
    context={
        'abstract': paper_content['abstract'],
        'methods': paper_content['methods']
    }
)

# Init W&B run
init_wandb(
    project_name="llm-scientific-hypothesis-generation",
    model_name="llama-3-3-70b",
    table_columns=LLM_IDEA_GENERATION_COLUMNS
)

# Run the experiment
run_idea_generation_batch(
    prompt=main_prompt,
    llama_fn=lambda p: llama_3_3_70B_completion(p, system_prompt=system_prompt),
    model_name="llama-3-3-70b",
    run_id="crispr_gene_editing_001",
    num_ideas=10,  # Generate 10 hypotheses
    quality_evaluator=evaluator.evaluate_hypothesis
)
