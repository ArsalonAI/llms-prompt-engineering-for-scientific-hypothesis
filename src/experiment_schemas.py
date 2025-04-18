LLM_IDEA_GENERATION_COLUMNS = [
    "run_id",              # ID of the run or experiment
    "idea",                # LLM-generated idea
    "batch_prompt",        # Prompt that generated the batch
    "judged_quality",      # Output of LLM judge (stupid / not stupid)
    "is_pruned",          # Boolean - whether idea passed filter
    "cosine_sim",         # Avg similarity with other ideas
    "self_bleu",          # Diversity metric
    "bertscore"           # Semantic token alignment score (novelty/quality)
]

LLM_ITERATIVE_SYNTHESIS_COLUMNS = [
    "timestamp", "run_id", "experiment_type", "source_paper_id", "reference_ids",
    "generated_idea", "similarity_to_source", "similarity_to_references",
    "prompt", "paper_title", "domain"
]