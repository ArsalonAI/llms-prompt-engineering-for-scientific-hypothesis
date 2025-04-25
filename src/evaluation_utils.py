from typing import List, Dict, Optional
from completion_util import llama_3_3_70B_completion
from similarity_metrics_utils import get_cosine_similarity, get_self_bleu, get_bertscore
from prompts.evaluator_prompts import judge_scientific_hypothesis_quality

class HypothesisEvaluator:
    def __init__(self):
        self.previous_hypotheses: List[str] = []
        self._evaluation_criteria = {
            "testability": "Can this hypothesis be tested through experimental methods?",
            "novelty": "Does this hypothesis propose something genuinely new to the field?",
            "scientific_grounding": "Is it well-grounded in existing scientific knowledge?",
            "specificity": "Is the hypothesis clear and specific about what it proposes?",
            "potential_impact": "If proven true, would this hypothesis significantly advance our understanding?"
        }
    
    def get_evaluation_criteria(self) -> Dict[str, str]:
        """Return the evaluation criteria used by this evaluator."""
        return self._evaluation_criteria.copy()
    
    def evaluate_hypothesis(self, hypothesis: str, system_prompt: Optional[str] = None) -> Dict:
        """
        Evaluate a scientific hypothesis using quality metrics and similarity comparisons.
        
        Args:
            hypothesis: The hypothesis to evaluate
            system_prompt: Optional system prompt to use for evaluation
        
        Returns:
            Dictionary containing evaluation results
        """
        # Get quality evaluation
        eval_prompt = judge_scientific_hypothesis_quality(hypothesis)
        evaluation = llama_3_3_70B_completion(eval_prompt, system_prompt=system_prompt)
        
        # Calculate similarity metrics
        cosine_sim = get_cosine_similarity(hypothesis, self.previous_hypotheses) if self.previous_hypotheses else 0.0
        self_bleu = get_self_bleu(hypothesis, self.previous_hypotheses) if self.previous_hypotheses else 0.0
        bertscore = get_bertscore(hypothesis, self.previous_hypotheses) if self.previous_hypotheses else 0.0
        
        # Track the hypothesis
        self.previous_hypotheses.append(hypothesis)
        
        # Extract final decision (ACCEPT or PRUNE)
        decision = evaluation.strip().split('\n')[-1].strip()
        is_accepted = decision == "ACCEPT"
        
        return {
            "evaluation": evaluation,
            "is_accepted": is_accepted,
            "cosine_similarity": cosine_sim,
            "self_bleu": self_bleu,
            "bertscore": bertscore
        } 