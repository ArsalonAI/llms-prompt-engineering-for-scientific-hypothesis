# LLMs for Scientific Hypothesis Generation

A research project investigating the potential of Large Language Models (LLMs) in augmenting the scientific process of hypothesis generation.

## Author
**Arsalon Amini**  
In collaboration with Milwaukee School of Engineering Computer Science Department

## Project Overview
This project explores whether Large Language Models (LLMs) can be effectively utilized to assist in the scientific process of hypothesis generation. The research aims to evaluate the capabilities of LLMs in understanding scientific contexts, generating meaningful hypotheses, and potentially accelerating the scientific discovery process.

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/ArsalonAI/llms-prompt-engineering-for-scientific-hypothesis.git
cd llms-prompt-engineering-for-scientific-hypothesis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

### Experiment Runners
The code uses a modular approach to experiment runners in the `experiment_runners` package:
- `BaseExperimentRunner`: Core functionality for running experiments
- `ScientificHypothesisRunner`: Runner for scientific hypothesis generation
- `RoleBasedHypothesisRunner`: Runner for role-based prompt experiments
- `FewShotHypothesisRunner`: Runner for few-shot prompt experiments

### Prompt Types
Different prompt engineering techniques are implemented to compare effectiveness:
- Scientific prompts (system + hypothesis)
- Role-based prompts (expert persona)
- Few-shot prompts (with examples)

### Analysis
- `statistical_analysis.py`: Functions for analyzing and comparing distributions
- `cross_experiment_analysis`: Module for comparing results across different experiment types
- `integrated_analysis`: Module that combines current results with historical data to draw evidence-based conclusions

## Running Experiments

Run all experiments and generate a comprehensive analysis:
```bash
python src/main.py
```

This will:
1. Run experiments with different prompt types
2. Generate individual experiment results in `src/experiment_results/[experiment_name]_[timestamp]/`
3. Perform integrated analysis that combines current and historical results
4. Generate an interactive unified dashboard at `src/experiment_results/integrated_analysis/unified_analysis_dashboard.html`

## Experiment Results
Each experiment generates:
- Raw data in JSON and CSV format
- Statistical metrics for cosine similarity, self-BLEU, and BERTScore
- Distribution analysis with KDE plots
- Interactive dashboard with visualizations

## Cross-Experiment Analysis
The `CrossExperimentAnalyzer` compares results across different prompt techniques:
- Statistical significance testing
- Distribution comparison with visualizations
- Performance metrics comparison
- Detailed comparison dashboard

## Integrated Analysis
The `IntegratedAnalyzer` combines current experiment results with historical data:
- Identifies statistically significant differences between prompt types
- Draws evidence-based conclusions using p-values and effect sizes
- Generates a comprehensive dashboard with tabs for different metrics
- Provides recommendations based on meaningful differences

## Testing

### Running All Tests

To run all tests, navigate to the project root directory and execute:

```bash
python -m unittest discover tests
```

### Unit Tests

Individual unit tests can be run for specific components:

```bash
# Test experiment runners
python -m unittest tests/test_experiment_runners.py

# Test experiment tracker functionality
python -m unittest tests/test_experiment_tracker.py

# Test similarity metrics
python -m unittest tests/test_similarity_metrics.py

# Test hypothesis evaluator
python -m unittest tests/test_hypothesis_evaluator.py
```

### Dashboard Tests

Tests for dashboard creation and visualization components:

```bash
# Test experiment tracker dashboard creation
python -m unittest tests/test_experiment_tracker_dashboard.py
```

### Cross-Experiment Analysis Tests

Tests for cross-experiment analysis and integrated analysis:

```bash
# Test cross-experiment analysis functionality
python -m unittest tests/test_cross_experiment_analysis.py
```

### Integration Tests

End-to-end integration tests that verify complete pipeline functionality:

```bash
# Run integration tests
python -m unittest tests/test_integration.py
```

## Requirements
The project dependencies are listed in `requirements.txt`. Make sure to install them using the instructions above.

## Contact
For questions, collaborations, or more information about this research, please contact:

Arsalon Amini  
Email: arsalon.ai@gmail.com

---
*Note: This README will be updated as the project progresses with additional information about methodologies, findings, and publications.*
