"""
Prompts specifically designed for scientific hypothesis generation and evaluation.
"""

def generate_scientific_system_prompt(abstract: str = "") -> str:
    """
    Generate a system prompt for scientific hypothesis generation.
    
    Args:
        abstract: Optional abstract from a research paper to provide context
    """
    base_prompt = """You are a creative scientific researcher tasked with generating novel research hypotheses. 
Your hypotheses should be:
1. Scientifically testable
2. Novel and not obvious
3. Well-grounded in existing scientific knowledge
4. Clear and specific
5. Potentially impactful if proven true"""

    if abstract:
        base_prompt += f"\n\nConsider this context from a relevant research paper:\n{abstract}"

    base_prompt += "\n\nFormat your response as a single, clear hypothesis statement."
    
    return base_prompt

def generate_scientific_hypothesis_prompt(domain: str, focus_area: str, context: dict = None) -> str:
    """
    Generate a domain-specific scientific hypothesis prompt.
    
    Args:
        domain: The scientific domain (e.g., "neuroscience")
        focus_area: Specific area of study within the domain
        context: Optional dictionary containing additional context (abstract, methods, etc.)
    """
    prompt = f"""Generate a novel scientific hypothesis in the field of {domain}, specifically focusing on {focus_area}.
Your hypothesis should propose a specific mechanism or relationship that hasn't been extensively studied before.
Make sure your hypothesis is testable and could be investigated through experimental methods."""

    if context:
        if context.get('abstract'):
            prompt += f"\n\nBase your hypothesis on, but go beyond, this research context:\n{context['abstract']}"
        if context.get('methods'):
            prompt += f"\n\nMethods used in current research:\n{context['methods']}"
            
    return prompt 