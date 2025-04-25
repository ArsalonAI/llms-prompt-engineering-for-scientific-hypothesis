import time
from datetime import datetime
from similarity_metrics_utils import get_cosine_similarity, get_self_bleu, get_bertscore
from wandb_utils import log_experiment_to_wandb
from typing import Optional, List, Callable
from prompts.types import FewShotExample, EvaluationCriteria
from prompts.few_shot_prompts import generate_few_shot_prompt
from prompts.role_based_prompts import generate_role_based_prompt, generate_expert_critique_prompt
from prompts.chain_of_thought_prompts import generate_cot_prompt
from prompts.evaluator_prompts import generate_evaluator_prompt
from experiment_tracker import ExperimentTracker


_previous_ideas = []

def run_idea_generation_batch(
    prompt: str,
    llama_fn: Callable,
    model_name: str,
    run_id: Optional[str] = None,
    num_ideas: int = 10,
    quality_evaluator: Optional[Callable] = None,
    tracker: Optional[ExperimentTracker] = None
):
    """
    Run a batch of idea generations and evaluations.
    
    Args:
        prompt: The prompt to use for generation
        llama_fn: Function to call the language model
        model_name: Name of the model being used
        run_id: Optional identifier for the experiment run
        num_ideas: Number of ideas to generate
        quality_evaluator: Optional function to evaluate idea quality
        tracker: Optional ExperimentTracker instance for logging results
    """
    global _previous_ideas
    
    # Generate ideas
    for _ in range(num_ideas):
        # Generate and evaluate idea
        idea = llama_fn(prompt)
        evaluation_results = quality_evaluator(idea) if quality_evaluator else {"is_accepted": True}
        
        # Calculate similarity metrics
        cosine_sim = get_cosine_similarity(idea, _previous_ideas)
        self_bleu_score = get_self_bleu(idea, _previous_ideas)
        bert_score = get_bertscore(idea, _previous_ideas)
        
        # Track for future similarity comparisons
        _previous_ideas.append(idea)
        
        # Format run identifier
        run_identifier = f"run_{run_id}" if run_id else datetime.now().strftime('%H%M%S_%f')
        
        # Create experiment data
        result_data = {
            "run_id": run_identifier,
            "idea": idea,
            "batch_prompt": prompt,
            "judged_quality": evaluation_results.get("evaluation", ""),
            "is_pruned": not evaluation_results.get("is_accepted", True),
            "cosine_sim": f"{cosine_sim:.3f}",
            "self_bleu": f"{self_bleu_score:.3f}",
            "bertscore": f"{bert_score:.3f}",
            "metadata": {
                "model": model_name,
                "elapsed_time": time.time(),
                "evaluation_full": evaluation_results
            }
        }
        
        # Log result using tracker if provided
        if tracker:
            tracker.log_result(result_data)
    
    return _previous_ideas  # Return the list of ideas generated


def run_iterative_synthesis(source_paper_id, paper_title, domain, reference_abstracts, llama_fn, model_name, prompt, run_id=None):
    generated_idea = llama_fn(prompt)

    similarity_to_refs = get_cosine_similarity(generated_idea, reference_abstracts)
    similarity_to_source = get_cosine_similarity(generated_idea, [prompt])

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_identifier = f"run_{run_id}" if run_id else datetime.now().strftime('%H%M%S_%f')

    experiment_data = [
        timestamp, run_identifier, "llm_iterative_synthesis", source_paper_id,
        reference_abstracts, generated_idea, similarity_to_source, similarity_to_refs,
        prompt, paper_title, domain,
        {
            "similarity_to_source": similarity_to_source,
            "similarity_to_references": similarity_to_refs,
            "domain": domain,
            "run_id": run_identifier,
        }
    ]

    log_experiment_to_wandb(timestamp, experiment_data)
    return generated_idea

def run_prompt_engineering_experiment(
    combined_text: str,
    llama_fn: Callable,
    model_name: str,
    domain: str = "genetic engineering",
    run_id: Optional[str] = None,
    custom_examples: Optional[List[FewShotExample]] = None,
    custom_role_description: Optional[str] = None,
    custom_evaluation_criteria: Optional[List[EvaluationCriteria]] = None
) -> dict:
    """
    Run a comprehensive prompt engineering experiment using different prompting strategies.
    
    Args:
        combined_text: The combined abstract and methods text to analyze
        llama_fn: Function to call the language model
        model_name: Name of the model being used
        domain: Scientific domain for specialization
        run_id: Optional identifier for the experiment run
        custom_examples: Optional list of custom few-shot examples
        custom_role_description: Optional custom role description
        custom_evaluation_criteria: Optional list of custom evaluation criteria
    """
    # Generate prompts using different strategies
    few_shot_prompt = generate_few_shot_prompt(
        combined_text=combined_text,
        examples=custom_examples,
        domain=domain
    )
    
    role_prompt = generate_role_based_prompt(
        combined_text=combined_text,
        role_description=custom_role_description,
        domain=domain
    )
    
    cot_prompt = generate_cot_prompt(
        combined_text=combined_text,
        domain=domain
    )
    
    # Get responses from the model
    few_shot_response = llama_fn(few_shot_prompt)
    role_response = llama_fn(role_prompt)
    cot_response = llama_fn(cot_prompt)
    
    # Generate evaluation
    evaluator_prompt = generate_evaluator_prompt(
        combined_text=combined_text,
        fewshot_response=few_shot_response,
        role_response=role_response,
        cot_response=cot_response,
        evaluation_criteria=custom_evaluation_criteria,
        domain=domain
    )
    evaluation_report = llama_fn(evaluator_prompt)
    
    # Log experiment data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_identifier = f"run_{run_id}" if run_id else datetime.now().strftime('%H%M%S_%f')
    
    experiment_data = {
        "timestamp": timestamp,
        "run_id": run_identifier,
        "experiment_type": "prompt_engineering",
        "model_name": model_name,
        "domain": domain,
        "combined_text": combined_text,
        "few_shot_response": few_shot_response,
        "role_response": role_response,
        "cot_response": cot_response,
        "evaluation_report": evaluation_report,
        "prompts": {
            "few_shot": few_shot_prompt,
            "role_based": role_prompt,
            "chain_of_thought": cot_prompt,
            "evaluator": evaluator_prompt
        }
    }
    
    log_experiment_to_wandb(timestamp, experiment_data)
    
    return {
        "few_shot_response": few_shot_response,
        "role_response": role_response,
        "cot_response": cot_response,
        "evaluation_report": evaluation_report
    }