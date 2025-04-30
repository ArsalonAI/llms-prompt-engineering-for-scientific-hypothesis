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

def run_idea_generation_batch(
    prompt: str,
    llama_fn: callable,
    model_name: str,
    run_id: str,
    quality_evaluator: callable,
    tracker: 'ExperimentTracker',
    context: Dict[str, str] = None,  # Add context parameter
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
        context: Dictionary containing 'abstract' and 'methods' from original paper
        num_ideas: Number of ideas to generate (defaults to 5)
    """
    results = []
    previous_ideas = []
    
    # Combine context for similarity comparison
    original_context = ""
    if context:
        print(f"\n[DEBUG] Context received: {type(context)}")
        print(f"[DEBUG] Abstract present: {'abstract' in context}")
        print(f"[DEBUG] Methods present: {'methods' in context}")
        
        abstract = context.get('abstract', '')
        methods = context.get('methods', '')
        
        print(f"[DEBUG] Abstract length: {len(abstract)}")
        print(f"[DEBUG] Methods length: {len(methods)}")
        
        original_context = abstract + " " + methods
        
        print(f"[DEBUG] Combined context length: {len(original_context)}")
        if len(original_context) > 0:
            print(f"[DEBUG] Context sample: {original_context[:100]}...")
        else:
            print("[WARNING] Context is empty!")
    
    for i in range(num_ideas):
        # Generate and evaluate ideas
        response = llama_fn(prompt)
        print(f"\nGenerated response (length: {len(response)})")
        
        # Get quality evaluation
        eval_result = quality_evaluator(response)
        quality_score = eval_result.get('evaluation', 'N/A')
        
        # Initialize similarity metrics dictionaries
        similarity_metrics = {
            'pairwise': {
                'cosine': [],
                'self_bleu': [],
                'bertscore': []
            },
            'context': {
                'cosine': 0.0,
                'self_bleu': 0.0,
                'bertscore': 0.0
            }
        }
        
        # Calculate similarities with previous ideas
        if previous_ideas:
            for prev_idea in previous_ideas:
                # Compare only the generated ideas with each other
                cos_sim = get_cosine_similarity(response, [prev_idea["idea"]])
                similarity_metrics['pairwise']['cosine'].append({
                    'compared_to': prev_idea['run_id'],
                    'score': cos_sim
                })
                
                self_bleu_score = get_self_bleu(response, [prev_idea["idea"]])
                similarity_metrics['pairwise']['self_bleu'].append({
                    'compared_to': prev_idea['run_id'],
                    'score': self_bleu_score
                })
                
                bert_score = get_bertscore(response, [prev_idea["idea"]])
                similarity_metrics['pairwise']['bertscore'].append({
                    'compared_to': prev_idea['run_id'],
                    'score': bert_score
                })
        
        # Calculate similarities with original context (abstract + methods only)
        if original_context:
            print(f"\n[DEBUG] Calculating context similarities for response {i+1}:")
            print(f"[DEBUG] Response length: {len(response)}")
            print(f"[DEBUG] Context length: {len(original_context)}")
            
            # Print the first 100 chars of each to verify content
            print(f"[DEBUG] Response sample: {response[:100]}...")
            print(f"[DEBUG] Context sample: {original_context[:100]}...")
            
            try:
                # Compare only the generated idea with the original context
                context_cos = get_cosine_similarity(response, [original_context])
                print(f"[DEBUG] Cosine similarity calculated: {context_cos} (type: {type(context_cos)})")
                
                context_bleu = get_self_bleu(response, [original_context])
                print(f"[DEBUG] Self-BLEU calculated: {context_bleu} (type: {type(context_bleu)})")
                
                context_bert = get_bertscore(response, [original_context])
                print(f"[DEBUG] BERTScore calculated: {context_bert} (type: {type(context_bert)})")
                
                # Ensure values are valid floats
                context_cos = float(context_cos) if context_cos is not None else 0.0
                context_bleu = float(context_bleu) if context_bleu is not None else 0.0
                context_bert = float(context_bert) if context_bert is not None else 0.0
                
                print(f"[DEBUG] Final values after conversion:")
                print(f"  Cosine: {context_cos}")
                print(f"  BLEU: {context_bleu}")
                print(f"  BERT: {context_bert}")
                
                similarity_metrics['context'] = {
                    'cosine': context_cos,
                    'self_bleu': context_bleu,
                    'bertscore': context_bert
                }
            except Exception as e:
                print(f"[ERROR] Exception in similarity calculation: {str(e)}")
                import traceback
                print(traceback.format_exc())
                # Set defaults
                similarity_metrics['context'] = {
                    'cosine': 0.0,
                    'self_bleu': 0.0,
                    'bertscore': 0.0
                }
        
        # Calculate average similarities for summary metrics
        avg_cosine = (
            sum(item['score'] for item in similarity_metrics['pairwise']['cosine']) / len(similarity_metrics['pairwise']['cosine'])
            if similarity_metrics['pairwise']['cosine'] else 0.0
        )
        avg_self_bleu = (
            sum(item['score'] for item in similarity_metrics['pairwise']['self_bleu']) / len(similarity_metrics['pairwise']['self_bleu'])
            if similarity_metrics['pairwise']['self_bleu'] else 1.0
        )
        avg_bertscore = (
            sum(item['score'] for item in similarity_metrics['pairwise']['bertscore']) / len(similarity_metrics['pairwise']['bertscore'])
            if similarity_metrics['pairwise']['bertscore'] else 1.0
        )
        
        # Create result dictionary
        result = {
            'run_id': f"{run_id}_{i+1}",
            'idea': response,
            'batch_prompt': prompt,
            'evaluation': quality_score,
            'evaluation_full': eval_result.get('evaluation_full', ''),
            # Store pairwise similarities (with other ideas)
            'avg_cosine_similarity': avg_cosine,
            'avg_self_bleu': avg_self_bleu,
            'avg_bertscore': avg_bertscore,
            # Store context similarities (with original text)
            'context_cosine': similarity_metrics['context']['cosine'],
            'context_self_bleu': similarity_metrics['context']['self_bleu'],
            'context_bertscore': similarity_metrics['context']['bertscore'],
            # Store raw pairwise comparison data
            'pairwise_similarities': similarity_metrics['pairwise'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print debug information about stored metrics
        print("\nStored Metrics:")
        print("Context Similarities (with original text):")
        print(f"  Cosine: {result['context_cosine']:.3f}")
        print(f"  BLEU: {result['context_self_bleu']:.3f}")
        print(f"  BERT: {result['context_bertscore']:.3f}")
        print("\nPairwise Similarities (with other ideas):")
        print(f"  Avg Cosine: {result['avg_cosine_similarity']:.3f}")
        print(f"  Avg BLEU: {result['avg_self_bleu']:.3f}")
        print(f"  Avg BERT: {result['avg_bertscore']:.3f}")
        print("Similarities with Original Context:")
        print(f"  Cosine: {result['context_cosine']:.3f}")
        print(f"  Self-BLEU: {result['context_self_bleu']:.3f}")
        print(f"  BERTScore: {result['context_bertscore']:.3f}")
        
        # Add to previous ideas for future similarity calculations
        previous_ideas.append(result)
        
        # Log result to tracker
        tracker.log_result(result)
        results.append(result)
        
        # Print progress with detailed metrics
        print(f"\n[STEP {i+1}/{num_ideas}]")
        print(f"Run ID: {result['run_id']}")
        print(f"Quality: {quality_score}")
        print("Average Similarities with Other Ideas:")
        print(f"  Cosine: {avg_cosine:.3f}")
        print(f"  Self-BLEU: {avg_self_bleu:.3f}")
        print(f"  BERTScore: {avg_bertscore:.3f}")
        print("Similarities with Original Context:")
        print(f"  Cosine: {result['context_cosine']:.3f}")
        print(f"  Self-BLEU: {result['context_self_bleu']:.3f}")
        print(f"  BERTScore: {result['context_bertscore']:.3f}")
    
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