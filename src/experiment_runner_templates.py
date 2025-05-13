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
import numpy as np

def run_idea_generation_batch(
    prompt: str,
    llama_fn: Callable,
    model_name: str,
    run_id: str,
    quality_evaluator: Callable,
    tracker: ExperimentTracker,
    context: Optional[str] = None,
    num_ideas: int = 10,
    max_pairwise_comparisons: int = 500,  # Limit pairwise comparisons for large batches
    sampling_strategy: str = "auto",  # Can be "all", "random", "stratified" or "auto"
    skip_intermediate_calculations: bool = True  # Whether to skip intermediate similarity calculations
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
        max_pairwise_comparisons: Maximum number of pairwise comparisons to perform (set to -1 for all)
        sampling_strategy: How to sample pairwise comparisons: "all", "random", "stratified", or "auto"
        skip_intermediate_calculations: Whether to skip intermediate similarity calculations (faster)
        
    Returns:
        Dictionary containing the results
    """
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "ideas": [],
        "quality_scores": [],
        "cosine_similarities": [],
        "self_bleu_scores": [],
        "bertscore_scores": [],
        "kde_values": {
            "cosine": {"x": [], "y": []},
            "self_bleu": {"x": [], "y": []},
            "bertscore": {"x": [], "y": []}
        }
    }
    
    # Generate ideas in batches
    batch_size = 5
    total_ideas_generated = 0 # Counter for tracking progress
    for i in range(0, num_ideas, batch_size):
        current_batch_size = min(batch_size, num_ideas - i)
        print(f"\n[INFO] Generating ideas {i+1}-{i+current_batch_size} of {num_ideas}...")
        
        batch_start = time.time()
        # Generate ideas
        ideas = []
        for j in range(current_batch_size):
            print(f"  Generating idea {i+j+1}/{num_ideas}... ({total_ideas_generated} completed)", end="\r")
            idea = llama_fn(prompt, context=context)
            ideas.append(idea)
            total_ideas_generated += 1
        
        # Evaluate quality
        print(f"  Evaluating quality of {len(ideas)} ideas...", end="\r")
        quality_scores = [quality_evaluator(idea, context=context) for idea in ideas]
        
        # Only calculate within-batch similarities for progress tracking if not skipping
        # Skip this for large batches to save time
        if not skip_intermediate_calculations and len(ideas) <= 10:
            cosine_sims = []
            self_bleu_scores = []
            bertscore_scores = []
            
            print(f"  Calculating intermediate similarity metrics...", end="\r")
            for j in range(len(ideas)):
                for k in range(j + 1, len(ideas)):
                    # Cosine similarity
                    cosine_sim = get_cosine_similarity(ideas[j], [ideas[k]])
                    cosine_sims.append(cosine_sim)
                    
                    # Self-BLEU
                    self_bleu = get_self_bleu(ideas[j], [ideas[k]])
                    self_bleu_scores.append(self_bleu)
                    
                    # BERTScore
                    bertscore = get_bertscore(ideas[j], [ideas[k]])
                    bertscore_scores.append(bertscore)
                    
            # Only track interim KDE values for small batches
            for metric, values in [
                ("cosine", cosine_sims),
                ("self_bleu", self_bleu_scores),
                ("bertscore", bertscore_scores)
            ]:
                if values:
                    x_kde, y_kde = tracker._calculate_kde(pd.Series(values))
                    results["kde_values"][metric]["x"].extend(x_kde.tolist())
                    results["kde_values"][metric]["y"].extend(y_kde.tolist())
            
            # Store batch results for progress tracking
            results["cosine_similarities"].extend(cosine_sims)
            results["self_bleu_scores"].extend(self_bleu_scores)
            results["bertscore_scores"].extend(bertscore_scores)
        
        # Store ideas and quality scores
        results["ideas"].extend(ideas)
        results["quality_scores"].extend(quality_scores)
        
        batch_time = time.time() - batch_start
        print(f"  Batch {i//batch_size + 1} completed in {batch_time:.1f} seconds ({total_ideas_generated}/{num_ideas} ideas total)")
    
    # Now that all ideas are generated, calculate complete pairwise similarity metrics
    # This ensures we compare all ideas to each other, including those across different batches
    all_ideas = results["ideas"]
    n_ideas = len(all_ideas)
    
    # Clear any existing similarity scores from batch calculations
    complete_cosine_sims = []
    complete_self_bleu_scores = []
    complete_bertscore_scores = []
    
    # Calculate expected number of comparisons
    full_comparisons = n_ideas * (n_ideas - 1) // 2
    
    # Dynamically adjust max_pairwise_comparisons based on number of ideas if "auto" mode
    if sampling_strategy == "auto":
        if n_ideas <= 10:
            # For small numbers of ideas, compare all pairs
            sampling_strategy = "all"
        elif n_ideas <= 25:
            # For medium numbers, use stratified sampling to ensure all ideas are included
            sampling_strategy = "stratified"
        else:
            # For large numbers, use random sampling with a cap
            sampling_strategy = "random"
            if max_pairwise_comparisons < 0 or max_pairwise_comparisons > 1000:
                max_pairwise_comparisons = min(1000, full_comparisons)
    
    # Determine if we need to sample and which strategy to use
    do_sampling = max_pairwise_comparisons > 0 and full_comparisons > max_pairwise_comparisons
    
    if sampling_strategy == "all" or not do_sampling:
        print(f"\n[INFO] Calculating all {full_comparisons} pairwise comparisons for {n_ideas} ideas")
        total_comparisons = full_comparisons
        pairs_to_compare = [(i, j) for i in range(n_ideas) for j in range(i+1, n_ideas)]
    elif sampling_strategy == "random":
        print(f"\n[INFO] Random sampling {max_pairwise_comparisons} out of {full_comparisons} possible pairwise comparisons")
        # Create all possible pairs and sample randomly
        all_pairs = [(i, j) for i in range(n_ideas) for j in range(i+1, n_ideas)]
        import random
        random.shuffle(all_pairs)
        pairs_to_compare = all_pairs[:max_pairwise_comparisons]
        total_comparisons = len(pairs_to_compare)
    elif sampling_strategy == "stratified":
        print(f"\n[INFO] Using stratified sampling to ensure each idea is compared")
        # Ensure each idea is included in at least one comparison
        pairs_per_idea = {}
        for i in range(n_ideas):
            for j in range(i+1, n_ideas):
                if i not in pairs_per_idea:
                    pairs_per_idea[i] = []
                if j not in pairs_per_idea:
                    pairs_per_idea[j] = []
                pairs_per_idea[i].append((i, j))
                pairs_per_idea[j].append((i, j))
        
        # Start by selecting one comparison for each idea
        selected_pairs = set()
        for i in range(n_ideas):
            if i in pairs_per_idea and pairs_per_idea[i]:
                # Select random pair involving this idea
                import random
                pair = random.choice(pairs_per_idea[i])
                selected_pairs.add(pair)
        
        # If we need more comparisons and have a limit, add more pairs randomly
        remaining = max_pairwise_comparisons - len(selected_pairs)
        if remaining > 0:
            all_remaining_pairs = [(i, j) for i in range(n_ideas) for j in range(i+1, n_ideas) 
                                  if (i, j) not in selected_pairs]
            import random
            random.shuffle(all_remaining_pairs)
            selected_pairs.update(all_remaining_pairs[:remaining])
        
        pairs_to_compare = list(selected_pairs)
        total_comparisons = len(pairs_to_compare)
        print(f"[INFO] Selected {total_comparisons} pairs using stratified sampling")
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    # Calculate pairwise similarities with progress tracking
    start_pairwise = time.time()
    for idx, (i, j) in enumerate(pairs_to_compare):
        if idx % 10 == 0 or idx == total_comparisons - 1:
            elapsed = time.time() - start_pairwise
            progress = (idx + 1) / total_comparisons
            est_remaining = (elapsed / progress) - elapsed if progress > 0 else 0
            print(f"  Progress: {idx+1}/{total_comparisons} ({progress*100:.1f}%) - ETA: {est_remaining:.1f}s   ", end="\r")
        
        # Calculate similarity metrics
        cosine_sim = get_cosine_similarity(all_ideas[i], [all_ideas[j]])
        self_bleu = get_self_bleu(all_ideas[i], [all_ideas[j]])
        bertscore = get_bertscore(all_ideas[i], [all_ideas[j]])
        
        # Store results
        complete_cosine_sims.append(cosine_sim)
        complete_self_bleu_scores.append(self_bleu)
        complete_bertscore_scores.append(bertscore)
    
    print(f"\n[INFO] Pairwise comparison completed in {time.time() - start_pairwise:.1f} seconds                 ")
    
    # Replace with complete pairwise metrics
    results["cosine_similarities"] = complete_cosine_sims
    results["self_bleu_scores"] = complete_self_bleu_scores
    results["bertscore_scores"] = complete_bertscore_scores
    results["pairwise_pairs"] = pairs_to_compare  # Store the actual pairs we compared
    
    # Calculate KDE for the complete metrics
    print(f"[INFO] Calculating KDE values...")
    results["kde_values"] = {
        "cosine": {"x": [], "y": []},
        "self_bleu": {"x": [], "y": []},
        "bertscore": {"x": [], "y": []}
    }
    
    for metric, values in [
        ("cosine", complete_cosine_sims),
        ("self_bleu", complete_self_bleu_scores),
        ("bertscore", complete_bertscore_scores)
    ]:
        if values:
            x_kde, y_kde = tracker._calculate_kde(pd.Series(values))
            results["kde_values"][metric]["x"] = x_kde.tolist()
            results["kde_values"][metric]["y"] = y_kde.tolist()
    
    # Calculate context similarities for all generated ideas
    print(f"[INFO] Calculating context similarities...")
    context_cosine_scores = []
    context_self_bleu_scores = []
    context_bertscore_scores = []

    # Add DEBUG logging for context and ideas before calculation
    print(f"[DEBUG] run_idea_generation_batch: Attempting context similarity. Context provided: {bool(context)}. Number of ideas: {len(results.get('ideas', []))}.")
    if context:
        preview = context[:100].replace('\n', ' ')
        print(f"[DEBUG] run_idea_generation_batch: Context preview for similarity calc: '{preview}...'")

    if context and results["ideas"]:
        context_self_bleu_scores = [] # Initialize list for context Self-BLEU
        for idx, idea_text in enumerate(results["ideas"]):
            if idx % 5 == 0 or idx == len(results["ideas"]) - 1:
                print(f"  Progress: {idx+1}/{len(results['ideas'])}   ", end="\r")
            cosine_score = get_cosine_similarity(idea_text, [context])
            # Calculate context Self-BLEU (idea vs. context)
            self_bleu_score = get_self_bleu(idea_text, [context]) 
            bertscore_score = get_bertscore(idea_text, [context])
            context_cosine_scores.append(cosine_score)
            context_self_bleu_scores.append(self_bleu_score) # Store context Self-BLEU
            context_bertscore_scores.append(bertscore_score)
            
        print(f"\n[DEBUG] Context similarity calculation complete")
        # Add DEBUG logging after calculations
        print(f"[DEBUG] run_idea_generation_batch: Calculated context_cosine_scores. Count: {len(context_cosine_scores)}. Preview: {context_cosine_scores[:3]}")
        print(f"[DEBUG] run_idea_generation_batch: Calculated context_self_bleu_scores. Count: {len(context_self_bleu_scores)}. Preview: {context_self_bleu_scores[:3]}") # Log Self-BLEU
        print(f"[DEBUG] run_idea_generation_batch: Calculated context_bertscore_scores. Count: {len(context_bertscore_scores)}. Preview: {context_bertscore_scores[:3]}")
    else:
        # Ensure this print statement is properly newlined after the progress indicator from the loop might have used \r
        print(f"\n[INFO] run_idea_generation_batch: Skipping context similarity calculation. Context present: {bool(context)}, Ideas present: {bool(results.get('ideas', []))}")
    
    # Calculate overall averages
    avg_pairwise_cosine = np.mean(results["cosine_similarities"]) if results["cosine_similarities"] else 0.0
    avg_pairwise_self_bleu = np.mean(results["self_bleu_scores"]) if results["self_bleu_scores"] else 0.0
    avg_pairwise_bertscore = np.mean(results["bertscore_scores"]) if results["bertscore_scores"] else 0.0
    
    avg_context_cosine = np.mean(context_cosine_scores) if context_cosine_scores else 0.0
    avg_context_self_bleu = np.mean(context_self_bleu_scores) if context_self_bleu_scores else 0.0 # Average for Self-BLEU
    avg_context_bertscore = np.mean(context_bertscore_scores) if context_bertscore_scores else 0.0

    # Track performance stats
    total_time = time.time() - start_time
    time_per_idea = total_time / len(results["ideas"]) if results["ideas"] else 0
    
    print(f"[INFO] Batch completed in {total_time:.1f} seconds ({time_per_idea:.1f}s per idea)")
    
    # Calculate variance and other statistical measures for each metric
    cosine_std = np.std(results["cosine_similarities"]) if len(results["cosine_similarities"]) > 1 else 0.0
    self_bleu_std = np.std(results["self_bleu_scores"]) if len(results["self_bleu_scores"]) > 1 else 0.0
    bertscore_std = np.std(results["bertscore_scores"]) if len(results["bertscore_scores"]) > 1 else 0.0
    
    # Log metrics distribution characteristics
    print("\n=== Distribution Statistics ===")
    for metric_name, values, avg, std in [
        ("Cosine Similarity", results["cosine_similarities"], avg_pairwise_cosine, cosine_std),
        ("Self-BLEU", results["self_bleu_scores"], avg_pairwise_self_bleu, self_bleu_std),
        ("BERTScore", results["bertscore_scores"], avg_pairwise_bertscore, bertscore_std)
    ]:
        if values:
            print(f"{metric_name}:")
            print(f"  Mean: {avg:.4f}")
            print(f"  Std:  {std:.4f}")
            print(f"  Range: [{min(values):.4f}, {max(values):.4f}]")
            print(f"  Samples: {len(values)}")
    
    # Also log context metrics if available
    if context_cosine_scores or context_bertscore_scores or context_self_bleu_scores: # Added self_bleu
        print("\n=== Context Similarity Statistics ===")
        for metric_name, values, avg in [
            ("Context Cosine", context_cosine_scores, avg_context_cosine),
            ("Context Self-BLEU", context_self_bleu_scores, avg_context_self_bleu), # Added self_bleu
            ("Context BERTScore", context_bertscore_scores, avg_context_bertscore)
        ]:
            if values:
                print(f"{metric_name}:")
                print(f"  Mean: {avg:.4f}")
                print(f"  Std:  {np.std(values):.4f}")
                print(f"  Range: [{min(values):.4f}, {max(values):.4f}]")
                print(f"  Samples: {len(values)}")
    
    # Prepare data for logging
    log_data = {
        "model": model_name,
        "prompt": prompt,
        "context": context, # The original context string
        "num_ideas": len(results["ideas"]),
        "ideas": results["ideas"],  # Make sure to include the actual ideas
        "quality_scores": results["quality_scores"], # List of evaluations
        
        # Raw lists of pairwise scores
        "cosine_similarities": results["cosine_similarities"],
        "self_bleu_scores": results["self_bleu_scores"],
        "bertscore_scores": results["bertscore_scores"],
        
        # Information about pairwise comparisons
        "pairwise_sampled": do_sampling,
        "pairwise_sampling_strategy": sampling_strategy,
        "pairwise_total_possible": full_comparisons,
        "pairwise_actual": total_comparisons,
        "pairwise_pairs_compared": results.get("pairwise_pairs"), # Log the actual pairs
        
        # Overall averages of pairwise scores for this run
        "avg_pairwise_cosine_similarity": avg_pairwise_cosine,
        "avg_pairwise_self_bleu": avg_pairwise_self_bleu,
        "avg_pairwise_bertscore": avg_pairwise_bertscore,
        
        # Standard deviations
        "std_pairwise_cosine": cosine_std,
        "std_pairwise_self_bleu": self_bleu_std, 
        "std_pairwise_bertscore": bertscore_std,
        
        # Context-based scores (raw list and average for cosine)
        "context_cosine_scores_raw": context_cosine_scores, # List of context cosine for each idea
        "avg_context_cosine_similarity": avg_context_cosine,
        "context_self_bleu_scores_raw": context_self_bleu_scores, # Add raw Self-BLEU scores
        "avg_context_self_bleu_similarity": avg_context_self_bleu, # Add average Self-BLEU
        "context_bertscore_scores_raw": context_bertscore_scores,
        "avg_context_bertscore_similarity": avg_context_bertscore,
        
        "kde_values": results["kde_values"],
        
        # Performance stats
        "runtime_seconds": total_time,
        "seconds_per_idea": time_per_idea
    }
    # Add DEBUG logging before calling tracker.log_result
    print(f"[DEBUG] run_idea_generation_batch: About to log results. Checking context_cosine_scores_raw. Count: {len(log_data.get('context_cosine_scores_raw', []))}")
    if log_data.get('context_cosine_scores_raw'):
        print(f"[DEBUG] run_idea_generation_batch: First 3 context_cosine_scores: {log_data['context_cosine_scores_raw'][:3]}")
    print(f"[DEBUG] run_idea_generation_batch: Checking context_self_bleu_scores_raw. Count: {len(log_data.get('context_self_bleu_scores_raw', []))}") # Log Self-BLEU
    if log_data.get('context_self_bleu_scores_raw'):
        print(f"[DEBUG] run_idea_generation_batch: First 3 context_self_bleu_scores: {log_data['context_self_bleu_scores_raw'][:3]}") # Log Self-BLEU
    print(f"[DEBUG] run_idea_generation_batch: Checking context_bertscore_scores_raw. Count: {len(log_data.get('context_bertscore_scores_raw', []))}")
    if log_data.get('context_bertscore_scores_raw'):
        print(f"[DEBUG] run_idea_generation_batch: First 3 context_bertscore_scores: {log_data['context_bertscore_scores_raw'][:3]}")
    tracker.log_result(run_id, log_data)
    
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


# This experimental structure has been replaced by a better DOE (design of experiments)
# will consider to remove this method in the future
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