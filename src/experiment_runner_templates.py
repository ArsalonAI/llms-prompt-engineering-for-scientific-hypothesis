import time
from datetime import datetime
from similarity_metrics_utils import get_cosine_similarity, get_self_bleu, get_bertscore
from typing import Optional, List, Callable, Dict, Any
from prompts.types import FewShotExample, EvaluationCriteria
from prompts.few_shot_prompts import generate_few_shot_prompt
from prompts.role_based_prompts import generate_role_based_prompt, generate_expert_critique_prompt
from prompts.chain_of_thought_prompts import generate_cot_prompt
from prompts.evaluator_prompts import generate_evaluator_prompt
from experiment_tracker import ExperimentTracker
import pandas as pd

def run_idea_generation_batch(
    prompt: str,
    llama_fn: Callable,
    model_name: str,
    run_id: str,
    quality_evaluator: Callable,
    tracker: ExperimentTracker,
    context: Optional[str] = None,
    num_ideas: int = 10
) -> Dict[str, Any]:
    """Run a batch of idea generation experiments.
    
    Args:
        prompt: The prompt to use for idea generation
        llama_fn: Function to call the LLaMA model
        model_name: Name of the model being used
        run_id: Unique identifier for this run
        quality_evaluator: Function to evaluate idea quality
        tracker: ExperimentTracker instance
        context: Optional context to provide to the model
        num_ideas: Number of ideas to generate
        
    Returns:
        Dictionary containing the results
    """
    # Initialize results dictionary
    results = {
        "ideas": [],
        "quality_scores": [],
        "cosine_similarities": [],
        "self_bleu_scores": [],
        "bertscore_scores": [],
        "kde_values": {
            "quality": {"x": [], "y": []},
            "cosine": {"x": [], "y": []},
            "self_bleu": {"x": [], "y": []},
            "bertscore": {"x": [], "y": []}
        }
    }
    
    # Generate ideas in batches
    batch_size = 5
    for i in range(0, num_ideas, batch_size):
        current_batch_size = min(batch_size, num_ideas - i)
        
        # Generate ideas
        ideas = []
        for _ in range(current_batch_size):
            idea = llama_fn(prompt, context=context)
            ideas.append(idea)
        
        # Evaluate quality
        quality_scores = [quality_evaluator(idea) for idea in ideas]
        
        # Calculate similarities
        cosine_sims = []
        self_bleu_scores = []
        bertscore_scores = []
        
        for j in range(len(ideas)):
            for k in range(j + 1, len(ideas)):
                # Cosine similarity
                cosine_sim = get_cosine_similarity(ideas[j], [ideas[k]])
                cosine_sims.append(cosine_sim)
                
                # Self-BLEU - expects (candidate, others) - first arg is string, second is list
                self_bleu = get_self_bleu(ideas[j], [ideas[k]])
                self_bleu_scores.append(self_bleu)
                
                # BERTScore - expects (candidate, others) - first arg is string, second is list
                bertscore = get_bertscore(ideas[j], [ideas[k]])
                bertscore_scores.append(bertscore)
        
        # Store results
        results["ideas"].extend(ideas)
        results["quality_scores"].extend(quality_scores)
        results["cosine_similarities"].extend(cosine_sims)
        results["self_bleu_scores"].extend(self_bleu_scores)
        results["bertscore_scores"].extend(bertscore_scores)
        
        # Calculate KDE values for each metric
        for metric, values in [
            ("quality", quality_scores),
            ("cosine", cosine_sims),
            ("self_bleu", self_bleu_scores),
            ("bertscore", bertscore_scores)
        ]:
            if values:
                x_kde, y_kde = tracker._calculate_kde(pd.Series(values))
                results["kde_values"][metric]["x"].extend(x_kde.tolist())
                results["kde_values"][metric]["y"].extend(y_kde.tolist())
    
    # Log results
    tracker.log_result(run_id, {
        "model": model_name,
        "prompt": prompt,
        "context": context,
        "num_ideas": num_ideas,
        "quality_scores": results["quality_scores"],
        "cosine_similarities": results["cosine_similarities"],
        "self_bleu_scores": results["self_bleu_scores"],
        "bertscore_scores": results["bertscore_scores"],
        "kde_values": results["kde_values"]
    })
    
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
    print(f"[INFO] Results logged to: {_local_logger.get_experiment_url()}")
    
    return {
        "few_shot_response": few_shot_response,
        "role_response": role_response,
        "cot_response": cot_response,
        "evaluation_report": evaluation_report
    }