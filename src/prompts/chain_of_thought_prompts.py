"""
Chain-of-thought prompt templates.
"""
from typing import Optional, List

def generate_cot_prompt(
    combined_text: str,
    domain: str = "genetic engineering",
    thinking_steps: Optional[List[str]] = None
) -> str:
    """
    Generate a chain-of-thought prompt for research paper analysis.
    
    Args:
        combined_text: The abstract and methods text to analyze
        domain: The scientific domain for context
        thinking_steps: Optional custom thinking steps to guide the analysis
    
    Returns:
        str: The formatted chain-of-thought prompt
    """
    if thinking_steps is None:
        thinking_steps = [
            "Identify the main hypothesis and methodology",
            "Analyze the strengths of the current approach",
            "Identify potential limitations or gaps",
            "Consider how the limitations could be addressed",
            "Brainstorm potential extensions of the research",
            "Formulate a novel hypothesis that builds on this work"
        ]
    
    steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(thinking_steps))
    
    return (
        f"You are an expert research scientist specializing in {domain} with extensive experience in evaluating research papers.\n\n"
        "When analyzing the following combined Abstract and Methods section, think through these steps:\n\n"
        f"{steps_text}\n\n"
        "For each step, explicitly write out your thinking process. Then, provide your final, concise evaluation "
        "along with a novel hypothesis for future experimentation.\n\n"
        "Combined Abstract and Methods section:\n\n"
        f"{combined_text}\n\n"
        "Start your analysis with Step 1 and clearly label each step of your thinking process."
    )

def generate_iterative_cot_prompt(
    combined_text: str,
    previous_analysis: str,
    focus_area: str
) -> str:
    """
    Generate a prompt for iterative chain-of-thought analysis.
    
    Args:
        combined_text: The abstract and methods text to analyze
        previous_analysis: Previous analysis to build upon
        focus_area: Specific aspect to focus on in this iteration
    """
    return (
        "Building on the previous analysis, let's dive deeper into a specific aspect of this research.\n\n"
        "Previous Analysis:\n"
        f"{previous_analysis}\n\n"
        f"Now, let's focus specifically on {focus_area}. Think through:\n\n"
        "1. What aspects of this area were addressed in the original research?\n"
        "2. What questions remain unanswered?\n"
        "3. What new methodologies or approaches could be applied?\n"
        "4. How could this lead to a novel research direction?\n\n"
        "Original Abstract and Methods:\n"
        f"{combined_text}\n\n"
        "Walk through your thinking process for each question, then synthesize your thoughts into "
        "a focused hypothesis for future research in this area."
    ) 