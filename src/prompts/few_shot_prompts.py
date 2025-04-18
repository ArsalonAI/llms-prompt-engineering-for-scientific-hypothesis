"""
Few-shot learning prompt templates.
"""
from typing import List, Optional
from .types import FewShotExample

DEFAULT_EXAMPLES = [
    FewShotExample(
        abstract_methods=(
            "Abstract: Recent advances in CRISPR/Cas9 have enabled precise modifications in crop genomes to enhance drought tolerance. "
            "Methods: The study used CRISPR vectors to introduce targeted mutations in drought-resistance genes of maize."
        ),
        evaluation="The research examines how targeted gene editing can improve crop resilience to environmental stress.",
        novel_hypothesis="Optimize CRISPR protocols for enhanced drought resilience in maize."
    ),
    FewShotExample(
        abstract_methods=(
            "Abstract: A novel gene therapy approach was explored to combat muscular dystrophy by correcting mutant genes. "
            "Methods: Viral vectors were employed to deliver corrected gene sequences to the affected muscle tissues."
        ),
        evaluation="The study tests the feasibility of gene therapy in alleviating symptoms of genetic muscle disorders.",
        novel_hypothesis="Enhance the specificity of viral vector delivery in gene therapy for muscular dystrophy."
    )
]

def generate_few_shot_prompt(
    combined_text: str,
    examples: Optional[List[FewShotExample]] = None,
    domain: str = "genetic engineering"
) -> str:
    """
    Generate a few-shot prompt for research paper analysis.
    
    Args:
        combined_text: The abstract and methods text to analyze
        examples: Optional list of examples for few-shot learning
        domain: The scientific domain for context
    
    Returns:
        str: The formatted few-shot prompt
    """
    examples = examples or DEFAULT_EXAMPLES
    
    examples_text = "\n\n".join(
        f"Example {i+1}:\n"
        f"Combined Abstract and Methods:\n"
        f"\"{example.abstract_methods}\"\n"
        f"Evaluation: {example.evaluation}\n"
        f"Novel hypothesis: {example.novel_hypothesis}"
        for i, example in enumerate(examples)
    )

    return (
        f"Below are a few example evaluations of research papers in the field of {domain}:\n\n"
        f"{examples_text}\n\n"
        f"Now, here is the combined Abstract and Methods section of a research paper:\n\n"
        f"{combined_text}\n\n"
        "Evaluate the research hypothesis and methods and generate a novel hypothesis for future experimentation."
    ) 