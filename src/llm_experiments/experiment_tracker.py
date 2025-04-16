
import time
from datetime import datetime
from wandb_utils import log_experiment_to_wandb
from similarity_utils import get_cosine_similarity, get_llm_similarity_category

_previous_completions = []


def run_experiment(prompt, experiment_type, llama_fn, model_name, system_prompt=None, run_id=None):
    global _previous_completions

    start_time = time.time()
    completion = llama_fn(prompt, system_prompt=system_prompt)
    elapsed_time = time.time() - start_time

    cosine_sim = get_cosine_similarity(completion, _previous_completions)
    llm_cat, llm_score = get_llm_similarity_category(llama_fn, prompt, completion, _previous_completions)
    _previous_completions.append(completion)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_identifier = f"run_{run_id}" if run_id else datetime.now().strftime('%H%M%S_%f')

    experiment_data = [
        timestamp, experiment_type, system_prompt or "", prompt, completion,
        f"{elapsed_time:.2f}", model_name, run_identifier,
        f"{cosine_sim:.3f}" if cosine_sim is not None else "",
        llm_cat or "", f"{llm_score:.2f}" if llm_score is not None else "",
        {
            "elapsed_time": elapsed_time,
            "prompt_length": len(prompt),
            "completion_length": len(completion),
            "experiment_type": experiment_type,
            "run_id": run_identifier,
            "cosine_similarity": cosine_sim or 0,
            "llm_similarity_score": llm_score or 0,
            "llm_similarity_category": llm_cat or ""
        }
    ]

    log_experiment_to_wandb(timestamp, experiment_data)
    return completion