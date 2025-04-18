"""
Shared types for prompt templates.
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FewShotExample:
    """Example for few-shot learning in research paper analysis."""
    abstract_methods: str
    evaluation: str
    novel_hypothesis: str

@dataclass
class EvaluationCriteria:
    """Standard evaluation criteria for research hypotheses."""
    name: str
    description: str
    weight: float = 1.0

DEFAULT_EVALUATION_CRITERIA = [
    EvaluationCriteria(
        "Novelty and ingenuity",
        "How original and innovative is the proposed hypothesis given the combined Abstract and Methods section?"
    ),
    EvaluationCriteria(
        "Experimental design feasibility",
        "How feasible is the experimental design suggested for testing the hypothesis?"
    ),
    EvaluationCriteria(
        "Impact potential",
        "What is the potential impact of this research on the field?"
    ),
    EvaluationCriteria(
        "Cost and materials feasibility",
        "Evaluate whether the required materials and overall cost make the experimental design practical."
    )
] 