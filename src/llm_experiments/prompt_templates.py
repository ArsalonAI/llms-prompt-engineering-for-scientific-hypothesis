"""
Module containing prompt templates for different experimental approaches in scientific hypothesis generation.
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FewShotExample:
    abstract_methods: str
    evaluation: str
    novel_hypothesis: str

class PromptTemplates:
    @staticmethod
    def few_shot_prompt(combined_text: str, examples: Optional[List[FewShotExample]] = None) -> str:
        if examples is None:
            examples = [
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

        examples_text = "\n\n".join(
            f"Example {i+1}:\n"
            f"Combined Abstract and Methods:\n"
            f"\"{example.abstract_methods}\"\n"
            f"Evaluation: {example.evaluation}\n"
            f"Novel hypothesis: {example.novel_hypothesis}"
            for i, example in enumerate(examples)
        )

        return (
            f"Below are a few example evaluations of research papers in the field of genetic engineering:\n\n"
            f"{examples_text}\n\n"
            f"Now, here is the combined Abstract and Methods section of a research paper:\n\n"
            f"{combined_text}\n\n"
            "Evaluate the research hypothesis and methods and generate a novel hypothesis for future experimentation."
        )

    @staticmethod
    def role_based_prompt(combined_text: str, role_description: Optional[str] = None) -> str:
        if role_description is None:
            role_description = (
                "You are an expert research scientist specializing in genetic engineering with extensive experience "
                "in analyzing experimental designs and research methodologies."
            )

        return (
            f"{role_description} "
            "Your task is twofold:\n\n"
            "1. Evaluate the following combined Abstract and Methods section of a research paper by highlighting "
            "the key strengths and weaknesses in the existing hypothesis and experimental design.\n\n"
            "2. Based on your evaluation, propose a novel hypothesis for future experimentation that addresses "
            "any gaps or builds upon the findings.\n\n"
            "Combined Abstract and Methods section:\n"
            f"{combined_text}\n\n"
            "Please provide a concise evaluation followed by your novel hypothesis."
        )

    @staticmethod
    def chain_of_thought_prompt(combined_text: str, domain: str = "genetic engineering") -> str:
        return (
            f"You are an expert research scientist specializing in {domain} with extensive experience in evaluating research papers. "
            "When analyzing the following combined Abstract and Methods section, list your thoughts, using your internal chain-of-thought, "
            "to consider the strengths and weaknesses of the hypothesis and experimental design, as well as any gaps or opportunities for future research. "
            "Then, provide only your final, concise evaluation along with a novel hypothesis for future experimentation.\n\n"
            "Combined Abstract and Methods section:\n\n"
            f"{combined_text}\n\n"
            "Please provide your final evaluation and novel hypothesis."
        )

    @staticmethod
    def evaluator_prompt(
        combined_text: str,
        fewshot_response: str,
        role_response: str,
        cot_response: str,
        evaluation_criteria: Optional[List[str]] = None
    ) -> str:
        if evaluation_criteria is None:
            evaluation_criteria = [
                "Novelty and ingenuity: How original and innovative is the proposed hypothesis given the combined Abstract and Methods section?",
                "Experimental design feasibility: How feasible is the experimental design suggested for testing the hypothesis?",
                "Impact potential: What is the potential impact of this research on the field of genetic engineering?",
                "Cost and materials feasibility: Evaluate whether the required materials and overall cost make the experimental design practical."
            ]

        criteria_text = "\n".join(f"{i+1}. {criterion}" for i, criterion in enumerate(evaluation_criteria))

        return (
            "You are a scientific evaluator with expertise in genetic engineering. "
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
            "For each output, provide a detailed evaluator report that includes numerical ratings on a scale of 1 to 5 "
            "along with concise justifications for each of the four criteria. "
            "Your response should clearly indicate your evaluation for each of the three outputs."
        )

# Legacy templates maintained for backward compatibility
IDEA_GENERATION_PROMPT = "Generate a futuristic business idea related to {theme}."

def judge_quality_prompt(idea: str) -> str:
    return f"Is this idea smart or stupid?\n\nIdea: {idea}\n\nAnswer with 'smart' or 'stupid'."

def literature_prompt(reference_abstracts: list[str]) -> str:
    joined = "\n\n".join(reference_abstracts)
    return f"""Based on the following references, generate a novel research idea:

{joined}"""

def iterative_prompt(source_paper: str, reference_ids: list[str]) -> str:
    return f"""Based on the source paper and reference IDs, generate a novel research hypothesis:
    
Source Paper: {source_paper}
Reference IDs: {', '.join(reference_ids)}"""

