"""
Experiment runner module for different prompt engineering techniques.
This module provides implementations for running
various prompt engineering experiments on scientific hypothesis generation.
"""

from .base_runner import BaseExperimentRunner
from .scientific_runner import ScientificHypothesisRunner
from .role_based_runner import RoleBasedHypothesisRunner
from .few_shot_runner import FewShotHypothesisRunner

__all__ = [
    'BaseExperimentRunner',
    'ScientificHypothesisRunner',
    'RoleBasedHypothesisRunner',
    'FewShotHypothesisRunner'
] 