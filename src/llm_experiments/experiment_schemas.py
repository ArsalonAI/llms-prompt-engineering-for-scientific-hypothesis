LLM_IDEA_GENERATION_COLUMNS = [
    "timestamp", "run_id", "experiment_type", "prompt", "idea", "judged_quality",
    "is_pruned", "cosine_similarity", "self_bleu", "bertscore"
]

LLM_ITERATIVE_SYNTHESIS_COLUMNS = [
    "timestamp", "run_id", "experiment_type", "source_paper_id", "reference_ids",
    "generated_idea", "similarity_to_source", "similarity_to_references",
    "prompt", "paper_title", "domain"
]