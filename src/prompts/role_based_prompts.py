"""
Role-based prompt templates.
"""
from typing import Optional

DEFAULT_ROLE_DESCRIPTION = (
    "You are an expert research scientist specializing in {domain} with extensive experience "
    "in analyzing experimental designs and research methodologies."
)

def generate_role_based_prompt(
    combined_text: str,
    role_description: Optional[str] = None,
    domain: str = "genetic engineering"
) -> str:
    """
    Generate a role-based prompt for research paper analysis.
    
    Args:
        combined_text: The abstract and methods text to analyze
        role_description: Optional custom role description
        domain: The scientific domain for context
    
    Returns:
        str: The formatted role-based prompt
    """
    if role_description is None:
        role_description = DEFAULT_ROLE_DESCRIPTION.format(domain=domain)

    return (
        f"{role_description}\n\n"
        "Your task is twofold:\n\n"
        "1. Evaluate the following combined Abstract and Methods section of a research paper by highlighting "
        "the key strengths and weaknesses in the existing hypothesis and experimental design.\n\n"
        "2. Based on your evaluation, propose a novel hypothesis for future experimentation that addresses "
        "any gaps or builds upon the findings.\n\n"
        "Combined Abstract and Methods section:\n"
        f"{combined_text}\n\n"
        "Please provide a concise evaluation followed by your novel hypothesis."
    )

def generate_expert_critique_prompt(
    combined_text: str,
    expertise_area: str,
    focus_aspects: Optional[list[str]] = None
) -> str:
    """
    Generate a prompt for expert critique of research.
    
    Args:
        combined_text: The abstract and methods text to analyze
        expertise_area: Specific area of expertise (e.g., "CRISPR technology", "gene therapy")
        focus_aspects: Optional list of aspects to focus on in the critique
    """
    aspects_text = ""
    if focus_aspects:
        aspects_text = "\n\nPlease pay particular attention to these aspects:\n" + \
                      "\n".join(f"- {aspect}" for aspect in focus_aspects)
    
    return (
        f"As a leading expert in {expertise_area}, review the following research abstract and methods.\n\n"
        "Provide a detailed critique that:\n"
        "1. Identifies methodological strengths and weaknesses\n"
        "2. Suggests potential improvements\n"
        "3. Highlights opportunities for future research\n"
        f"{aspects_text}\n\n"
        "Combined Abstract and Methods section:\n"
        f"{combined_text}\n\n"
        "Please structure your response with clear sections for each point."
    ) 