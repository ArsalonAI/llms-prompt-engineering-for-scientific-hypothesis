"""
Evaluator prompt templates for assessing research hypotheses and methodologies.
"""
from typing import List, Optional
from .types import EvaluationCriteria, DEFAULT_EVALUATION_CRITERIA

def generate_evaluator_prompt(
    combined_text: str,
    fewshot_response: str,
    role_response: str,
    cot_response: str,
    evaluation_criteria: Optional[List[EvaluationCriteria]] = None,
    domain: str = "genetic engineering"
) -> str:
    """
    Generate a prompt for evaluating different analysis approaches.
    
    Args:
        combined_text: The original abstract and methods text
        fewshot_response: Response from few-shot learning approach
        role_response: Response from role-based approach
        cot_response: Response from chain-of-thought approach
        evaluation_criteria: Optional custom evaluation criteria
        domain: The scientific domain for context
    """
    criteria = evaluation_criteria or DEFAULT_EVALUATION_CRITERIA
    criteria_text = "\n".join(
        f"{i+1}. {c.name}: {c.description}"
        for i, c in enumerate(criteria)
    )

    return (
        f"You are a scientific evaluator with expertise in {domain}. "
        "Your task is to assess the research evaluation outputs produced by three different prompting approaches: "
        "Few Shot, Role Based, and Chain of Thought. "
        "Each of these outputs includes an assessment of the research paper's hypothesis and a proposal for a "
        "novel hypothesis for future experimentation.\n\n"
        f"Please evaluate each of the provided outputs based on the following criteria:\n{criteria_text}\n\n"
        "Below is the Combined Abstract and Methods section for context:\n\n"
        f"{combined_text}\n\n"
        "Now, please review and evaluate the following outputs:\n\n"
        "----- Few Shot Output -----\n"
        f"{fewshot_response}\n\n"
        "----- Role Based Output -----\n"
        f"{role_response}\n\n"
        "----- Chain of Thought Output -----\n"
        f"{cot_response}\n\n"
        "For each output, provide a detailed evaluator report that includes:\n"
        "1. Numerical ratings (1-5) for each criterion\n"
        "2. Brief justification for each rating\n"
        "3. Overall assessment of the approach's effectiveness\n"
        "4. Suggestions for improving the output\n\n"
        "Structure your response clearly, addressing each output separately."
    )

def generate_comparative_analysis_prompt(
    hypothesis_list: List[str],
    evaluation_focus: Optional[List[str]] = None
) -> str:
    """
    Generate a prompt for comparing multiple hypotheses.
    
    Args:
        hypothesis_list: List of hypotheses to compare
        evaluation_focus: Optional specific aspects to focus on
    """
    hypotheses_text = "\n\n".join(
        f"Hypothesis {i+1}:\n{hypothesis}"
        for i, hypothesis in enumerate(hypothesis_list)
    )
    
    focus_text = ""
    if evaluation_focus:
        focus_text = "\n\nPlease pay particular attention to these aspects:\n" + \
                    "\n".join(f"- {aspect}" for aspect in evaluation_focus)
    
    return (
        "Compare and analyze the following research hypotheses:\n\n"
        f"{hypotheses_text}\n"
        f"{focus_text}\n\n"
        "For each hypothesis:\n"
        "1. Evaluate its originality and potential impact\n"
        "2. Assess its technical feasibility\n"
        "3. Identify its strengths and limitations\n\n"
        "Then:\n"
        "1. Rank the hypotheses from most to least promising\n"
        "2. Suggest how the best elements of different hypotheses might be combined\n"
        "3. Propose a refined hypothesis that builds on these insights"
    )

def judge_scientific_hypothesis_quality(hypothesis: str) -> str:
    """
    Generate a prompt for evaluating the quality of a scientific hypothesis.
    
    Args:
        hypothesis: The hypothesis to evaluate
    """
    return (
        "You are an expert scientific evaluator. Evaluate the following scientific hypothesis based on these criteria:\n\n"
        "1. Testability: Can this hypothesis be tested through experimental methods?\n"
        "2. Novelty: Does this hypothesis propose something genuinely new to the field?\n"
        "3. Scientific Grounding: Is it well-grounded in existing scientific knowledge?\n"
        "4. Specificity: Is the hypothesis clear and specific about what it proposes?\n"
        "5. Potential Impact: If proven true, would this hypothesis significantly advance our understanding?\n\n"
        f"Hypothesis to evaluate:\n{hypothesis}\n\n"
        "First, rate each criterion from 1-5, where 5 is excellent and 1 is poor.\n"
        "Then provide a brief justification for each rating.\n"
        "Finally, make a binary decision: Is this hypothesis worth pursuing (ACCEPT) or should it be pruned (PRUNE)?\n"
        "Output your decision as the last line, using only the word ACCEPT or PRUNE."
    ) 