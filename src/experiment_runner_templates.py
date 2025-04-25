import time
from datetime import datetime
from similarity_metrics_utils import get_cosine_similarity, get_self_bleu, get_bertscore
from local_logger import LocalExperimentLogger
from typing import Optional, List, Callable, Dict, Any
from prompts.types import FewShotExample, EvaluationCriteria
from prompts.few_shot_prompts import generate_few_shot_prompt
from prompts.role_based_prompts import generate_role_based_prompt, generate_expert_critique_prompt
from prompts.chain_of_thought_prompts import generate_cot_prompt
from prompts.evaluator_prompts import generate_evaluator_prompt
from experiment_tracker import ExperimentTracker

# Initialize the local logger
_local_logger = LocalExperimentLogger()
_previous_ideas = []

def run_idea_generation_batch(
    prompt: str,
    llama_fn: callable,
    model_name: str,
    run_id: str,
    quality_evaluator: callable,
    tracker: 'ExperimentTracker',
    num_ideas: int = 5  # Default to 5 if not specified
) -> List[Dict[str, Any]]:
    """
    Run a batch of idea generations and track results.
    
    Args:
        prompt: The prompt to use for generation
        llama_fn: Function to call for LLM completion
        model_name: Name of the model being used
        run_id: Unique identifier for this run
        quality_evaluator: Function to evaluate idea quality
        tracker: ExperimentTracker instance
        num_ideas: Number of ideas to generate (defaults to 5)
    """
    results = []
    for i in range(num_ideas):
        # Generate idea
        response = llama_fn(prompt)
        
        # Evaluate quality
        quality_score = quality_evaluator(response)
        
        # Calculate similarity metrics
        cosine_similarity = get_cosine_similarity(response, [idea["idea"] for idea in _previous_ideas]) if _previous_ideas else 0.0
        self_bleu = get_self_bleu(response, [idea["idea"] for idea in _previous_ideas]) if _previous_ideas else 1.0
        bertscore = get_bertscore(response, [idea["idea"] for idea in _previous_ideas]) if _previous_ideas else 1.0
        
        # Create result entry
        result = {
            "run_id": f"{run_id}_{i+1}",
            "idea": response,
            "batch_prompt": prompt,
            "judged_quality": quality_score,
            "model": model_name,
            "cosine_similarity": cosine_similarity,
            "self_bleu": self_bleu,
            "bertscore": bertscore
        }
        
        # Add to previous ideas for future similarity calculations
        _previous_ideas.append(result)
        
        # Create experiment data for local logging
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        experiment_data = [
            timestamp, result["run_id"], "batch_generation", None,
            [], response, cosine_similarity, self_bleu,
            prompt, None, None,
            {
                "model": model_name,
                "judged_quality": quality_score,
                "cosine_similarity": cosine_similarity,
                "self_bleu": self_bleu,
                "bertscore": bertscore
            }
        ]
        
        # Log to local HTML
        _local_logger.log_experiment(experiment_data)
        
        # Log result to tracker
        tracker.log_result(result)
        results.append(result)
        
        # Print minimal console output
        print(f"[STEP] Run ID: {result['run_id']} | "
              f"Quality: {quality_score} | "
              f"Metrics: cos_sim={cosine_similarity:.3f}, "
              f"self_bleu={self_bleu:.3f}, "
              f"bert_score={bertscore:.3f}")
    
    print(f"[INFO] Batch results logged to: {_local_logger.get_experiment_url()}")
    return results


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

    # Log to local HTML instead of W&B
    _local_logger.log_experiment(experiment_data)
    print(f"[INFO] Results logged to: {_local_logger.get_experiment_url()}")
    
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
    
    experiment_data = [
        timestamp, run_identifier, "prompt_engineering", None,
        [], None, 0.0, 0.0,  # Placeholders for reference data and similarities
        combined_text, None, domain,
        {
            "model_name": model_name,
            "domain": domain,
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
    ]
    
    # Log to local HTML
    _local_logger.log_experiment(experiment_data)
    print(f"[INFO] Results logged to: {_local_logger.get_experiment_url()}")
    
    return {
        "few_shot_response": few_shot_response,
        "role_response": role_response,
        "cot_response": cot_response,
        "evaluation_report": evaluation_report
    }