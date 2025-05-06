from typing import List, Dict, Optional
from completion_util import llama_3_3_70B_completion
from similarity_metrics_utils import get_cosine_similarity, get_self_bleu, get_bertscore
from prompts.evaluator_prompts import judge_scientific_hypothesis_quality

class HypothesisEvaluator:
    def __init__(self):
        self.previous_hypotheses: List[str] = []
        self._evaluation_criteria = [
            "Scientific Validity",
            "Novelty",
            "Testability",
            "Relevance to CRISPR",
            "Potential Impact"
        ]
    
    def get_evaluation_criteria(self) -> List[str]:
        """
        Get the list of criteria used for evaluating hypotheses.
        
        Returns:
            List of evaluation criteria strings
        """
        return self._evaluation_criteria
    
    def evaluate_hypothesis(self, hypothesis: str, system_prompt: Optional[str] = None, original_context: Optional[str] = None, context: Optional[str] = None) -> Dict:
        """
        Evaluate a scientific hypothesis using quality metrics and similarity comparisons.
        
        Args:
            hypothesis: The hypothesis to evaluate
            system_prompt: Optional system prompt to use for evaluation
            original_context: Optional original context text to compare with (for context similarity)
            context: Alternative name for original_context (for compatibility)
        
        Returns:
            Dictionary containing evaluation results
        """
        # Handle either context or original_context parameter
        context_text = original_context or context
        
        # Get quality evaluation
        eval_prompt = judge_scientific_hypothesis_quality(hypothesis)
        evaluation_full = llama_3_3_70B_completion(eval_prompt, system_prompt=system_prompt)
        
        # Calculate similarity metrics between this hypothesis and previous ones
        cosine_sim = get_cosine_similarity(hypothesis, self.previous_hypotheses) if self.previous_hypotheses else 0.0
        self_bleu = get_self_bleu(hypothesis, self.previous_hypotheses) if self.previous_hypotheses else 0.0
        bertscore = get_bertscore(hypothesis, self.previous_hypotheses) if self.previous_hypotheses else 0.0
        
        # Calculate similarity to original context if provided
        context_cosine = 0.0
        context_self_bleu = 0.0
        context_bertscore = 0.0
        
        if context_text:
            context_cosine = get_cosine_similarity(hypothesis, [context_text])
            context_self_bleu = get_self_bleu(hypothesis, [context_text])
            context_bertscore = get_bertscore(hypothesis, [context_text])
        
        # Track the hypothesis
        self.previous_hypotheses.append(hypothesis)
        
        # Extract final decision (ACCEPT or PRUNE)
        decision = evaluation_full.strip().split('\n')[-1].strip()
        is_accepted = decision == "ACCEPT"
        
        # Create a concise summary for console logging
        evaluation_summary = "ACCEPT" if is_accepted else "PRUNE"
        
        return {
            "evaluation": evaluation_summary,  # Concise version for console
            "evaluation_full": evaluation_full,  # Detailed version for HTML
            "is_accepted": is_accepted,
            "cosine_similarity": cosine_sim,
            "self_bleu": self_bleu,
            "bertscore": bertscore,
            "context_cosine": context_cosine,
            "context_self_bleu": context_self_bleu,
            "context_bertscore": context_bertscore
        } 