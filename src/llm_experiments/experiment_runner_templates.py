import time
from datetime import datetime
from similarity_metrics_utils import get_cosine_similarity, get_self_bleu, get_bertscore
from wandb_utils import log_experiment_to_wandb
from typing import Optional, List


_previous_ideas = []

def run_idea_generation_batch(prompt, llama_fn, model_name, run_id=None):
    global _previous_ideas

    generated_ideas = [llama_fn(prompt) for _ in range(100)]
    pruned_ideas = []

    for idea in generated_ideas:
        judgment_prompt = f"Is this idea smart or stupid?\n\nIdea: {idea}\n\nAnswer with 'smart' or 'stupid'."
        judged_quality = llama_fn(judgment_prompt).strip().lower()
        is_pruned = judged_quality != "stupid"

        if is_pruned:
            pruned_ideas.append(idea)

        cosine_sim = get_cosine_similarity(idea, _previous_ideas)
        self_bleu = get_self_bleu(idea, _previous_ideas)
        bertscore = get_bertscore(idea, _previous_ideas)

        _previous_ideas.append(idea)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_identifier = f"run_{run_id}" if run_id else datetime.now().strftime('%H%M%S_%f')

        experiment_data = [
            timestamp, run_identifier, "llm_idea_generation", prompt, idea,
            judged_quality, is_pruned,
            cosine_sim, self_bleu, bertscore,
            {
                "judged_quality": judged_quality,
                "cosine_similarity": cosine_sim,
                "self_bleu": self_bleu,
                "bertscore": bertscore,
                "run_id": run_identifier,
            }
        ]

        log_experiment_to_wandb(timestamp, experiment_data)


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
    llama_fn,
    model_name: str,
    domain: str = "genetic engineering",
    run_id: Optional[str] = None,
    custom_examples: Optional[List[FewShotExample]] = None,
    custom_role_description: Optional[str] = None,
    custom_evaluation_criteria: Optional[List[str]] = None
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
        custom_role_description: Optional custom role description for role-based prompting
        custom_evaluation_criteria: Optional list of custom evaluation criteria
        
    Returns:
        Dictionary containing the responses and evaluation for each prompting strategy
    """
    from prompt_templates import PromptTemplates
    
    # Generate responses using different prompting strategies
    few_shot_prompt = PromptTemplates.few_shot_prompt(combined_text, examples=custom_examples)
    role_prompt = PromptTemplates.role_based_prompt(combined_text, role_description=custom_role_description)
    cot_prompt = PromptTemplates.chain_of_thought_prompt(combined_text, domain=domain)
    
    few_shot_response = llama_fn(few_shot_prompt)
    role_response = llama_fn(role_prompt)
    cot_response = llama_fn(cot_prompt)
    
    # Generate evaluation of the responses
    evaluator_prompt = PromptTemplates.evaluator_prompt(
        combined_text=combined_text,
        fewshot_response=few_shot_response,
        role_response=role_response,
        cot_response=cot_response,
        evaluation_criteria=custom_evaluation_criteria
    )
    evaluation_report = llama_fn(evaluator_prompt)
    
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