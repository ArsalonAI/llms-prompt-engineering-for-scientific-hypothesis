"""
Statistical analysis module.

This module provides functionality to perform rigorous statistical analysis on experiment results
with emphasis on statistical significance testing and effect size calculations.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
from ..cross_experiment_analysis.analyzer import CrossExperimentAnalyzer

class StatisticalAnalyzer:
    """Analyzer for performing statistical tests and generating research-grade reports."""
    
    def __init__(self, current_results: Dict[str, Any], experiment_dir: str):
        """
        Initialize the statistical analyzer.
        
        Args:
            current_results: Dictionary containing current experiment results
            experiment_dir: Base directory containing experiment results
        """
        self.current_results = current_results
        self.experiment_dir = experiment_dir
        self.output_dir = os.path.join(experiment_dir, "statistical_analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track whether current results were added to historical data
        self.current_results_added = False
        
    def perform_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on experiment results.
        
        Returns:
            Dictionary containing analysis results
        """
        # Initialize the cross-experiment analyzer
        cross_analyzer = CrossExperimentAnalyzer(
            experiment_dir=self.experiment_dir,
            output_dir=self.output_dir
        )
        
        # Load historical experiment results
        cross_analyzer.load_experiment_results()
        
        # Extract metrics from current and historical data
        metrics = cross_analyzer.extract_metrics()
        
        # Perform comparative analysis
        comparison_results = cross_analyzer.compare_metrics()
        
        # Perform additional statistical tests
        extended_stats = self.calculate_extended_statistics(metrics)
        
        # Identify statistically significant differences
        significant_differences = self.identify_significant_differences(comparison_results)
        
        # Draw evidence-based conclusions
        conclusions = self.draw_evidence_based_conclusions(metrics, significant_differences)
        
        # Generate research dashboard
        dashboard_path = self.generate_research_dashboard(cross_analyzer, conclusions, significant_differences, extended_stats)
        
        return {
            "metrics": metrics,
            "comparison_results": comparison_results,
            "extended_statistics": extended_stats,
            "significant_differences": significant_differences,
            "conclusions": conclusions,
            "dashboard_path": dashboard_path
        }
    
    def calculate_extended_statistics(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate extended statistical measures beyond simple means and standard deviations.
        
        Args:
            metrics: Extracted metrics from cross-experiment analysis
            
        Returns:
            Dictionary of extended statistical measures
        """
        extended_stats = {}
        
        for metric_name, metric_data in metrics.items():
            extended_stats[metric_name] = {}
            
            for exp_type, values in metric_data.items():
                if not values:
                    continue
                
                # Convert to numpy array if it's a list
                data = np.array(values)
                
                # Calculate extended statistics
                stats_dict = {
                    "mean": np.mean(data),
                    "median": np.median(data),
                    "std": np.std(data),
                    "min": np.min(data),
                    "max": np.max(data),
                    "q1": np.percentile(data, 25),
                    "q3": np.percentile(data, 75),
                    "iqr": np.percentile(data, 75) - np.percentile(data, 25),
                    "skewness": stats.skew(data),
                    "kurtosis": stats.kurtosis(data),
                    "shapiro_test": {
                        "statistic": stats.shapiro(data)[0],
                        "p_value": stats.shapiro(data)[1],
                        "is_normal": stats.shapiro(data)[1] > 0.05
                    },
                    "confidence_interval_95": stats.norm.interval(0.95, loc=np.mean(data), scale=stats.sem(data))
                }
                
                extended_stats[metric_name][exp_type] = stats_dict
        
        return extended_stats
    
    def identify_significant_differences(self, comparison_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify statistically significant differences between experiment types.
        
        Args:
            comparison_results: Results from cross-experiment analysis
            
        Returns:
            Dictionary of significant differences by metric
        """
        significant_diffs = {}
        
        if "statistical_tests" not in comparison_results:
            print("[WARNING] No statistical tests found in comparison results")
            return {}
        
        for metric, tests in comparison_results["statistical_tests"].items():
            significant_diffs[metric] = []
            
            for comparison, result in tests.items():
                # Check if difference is statistically significant
                is_significant = (
                    result["mann_whitney"]["significant"] or 
                    result["t_test"]["significant"]
                )
                
                # Check if effect size is meaningful
                effect_size = result["effect_size"]["cohen_d"]
                meaningful_effect = abs(effect_size) > 0.5
                
                if is_significant and meaningful_effect:
                    # Parse experiment types from comparison string (e.g., "Type1 vs Type2")
                    exp_types = comparison.split(" vs ")
                    if len(exp_types) != 2:
                        continue
                    
                    # Determine which method is better based on metric and effect size
                    # For cosine and self-bleu, lower is better (negative effect size means second method is better)
                    # For bertscore, higher is better (positive effect size means first method is better)
                    if metric in ["cosine", "self_bleu"]:
                        better_method = exp_types[1] if effect_size > 0 else exp_types[0]
                    else:  # bertscore
                        better_method = exp_types[0] if effect_size > 0 else exp_types[1]
                    
                    significant_diffs[metric].append({
                        "comparison": comparison,
                        "effect_size": effect_size,
                        "effect_interpretation": result["effect_size"]["interpretation"],
                        "p_value_mw": result["mann_whitney"]["p_value"],
                        "p_value_t": result["t_test"]["p_value"],
                        "better_method": better_method
                    })
        
        return significant_diffs
    
    def draw_evidence_based_conclusions(
        self, 
        metrics: Dict[str, Dict[str, Any]], 
        significant_differences: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Draw evidence-based conclusions from the data.
        
        Args:
            metrics: Extracted metrics from cross-experiment analysis
            significant_differences: Dictionary of significant differences
            
        Returns:
            Dictionary of conclusions
        """
        # Initialize conclusions
        conclusions = {
            "overall_best_method": None,
            "method_scores": {},
            "metric_conclusions": {},
            "key_findings": [],
            "detailed_comparisons": {},
            "recommendations": []
        }
        
        # Initialize scores for each method
        experiment_types = set()
        for metric, diffs in significant_differences.items():
            for diff in diffs:
                experiment_types.add(diff["better_method"])
        
        for exp_type in experiment_types:
            conclusions["method_scores"][exp_type] = 0
        
        # Score methods based on statistically significant advantages
        for metric, diffs in significant_differences.items():
            metric_conclusions = {
                "best_method": None,
                "significant_findings": []
            }
            
            for diff in diffs:
                better_method = diff["better_method"]
                worse_method = diff["comparison"].replace(" vs ", "|").replace(better_method, "").replace("|", "")
                
                # Add points to the better method
                conclusions["method_scores"][better_method] = conclusions["method_scores"].get(better_method, 0) + 1
                
                # Record the finding
                finding = f"{better_method} is significantly better than {worse_method} for {metric} (p={min(diff['p_value_mw'], diff['p_value_t']):.4f}, effect size={abs(diff['effect_size']):.2f})"
                metric_conclusions["significant_findings"].append(finding)
                
                # Add to detailed comparisons
                comparison_key = f"{better_method} vs {worse_method}"
                if comparison_key not in conclusions["detailed_comparisons"]:
                    conclusions["detailed_comparisons"][comparison_key] = {
                        "strengths": [],
                        "weaknesses": []
                    }
                
                conclusions["detailed_comparisons"][comparison_key]["strengths"].append(
                    f"Significantly better {metric} (effect size: {abs(diff['effect_size']):.2f}, {diff['effect_interpretation']})"
                )
            
            # Determine best method for this metric
            best_method = None
            best_score = -1
            
            for method, score in conclusions["method_scores"].items():
                if score > best_score:
                    best_score = score
                    best_method = method
            
            metric_conclusions["best_method"] = best_method
            conclusions["metric_conclusions"][metric] = metric_conclusions
        
        # Determine overall best method
        best_method = None
        best_score = -1
        
        for method, score in conclusions["method_scores"].items():
            if score > best_score:
                best_score = score
                best_method = method
        
        conclusions["overall_best_method"] = best_method
        
        # Generate key findings
        if best_method:
            conclusions["key_findings"].append(
                f"{best_method} is the most effective prompting technique overall based on statistically significant differences."
            )
        
        for metric, metric_conclusions in conclusions["metric_conclusions"].items():
            if metric_conclusions["best_method"]:
                metric_display = metric.replace("_", " ").title()
                conclusions["key_findings"].append(
                    f"{metric_conclusions['best_method']} performs best for {metric_display} with statistical significance."
                )
        
        # Generate recommendations
        if best_method:
            conclusions["recommendations"].append(
                f"Use {best_method} for general scientific hypothesis generation tasks."
            )
        
        conclusions["recommendations"].append(
            "Consider the specific strengths of each method for specialized tasks."
        )
        
        if len(conclusions["method_scores"]) > 1:
            conclusions["recommendations"].append(
                "For the broadest range of hypotheses, consider using multiple prompt techniques in combination."
            )
        
        return conclusions
    
    def generate_research_dashboard(
        self, 
        cross_analyzer: CrossExperimentAnalyzer,
        conclusions: Dict[str, Any],
        significant_differences: Dict[str, List[Dict[str, Any]]],
        extended_stats: Dict[str, Any]
    ) -> str:
        """
        Generate a research-grade dashboard with comprehensive statistical analysis.
        
        Args:
            cross_analyzer: CrossExperimentAnalyzer instance
            conclusions: Dictionary of conclusions
            significant_differences: Dictionary of significant differences
            extended_stats: Extended statistical measures
            
        Returns:
            Path to the generated dashboard
        """
        # First, generate the standard cross-experiment dashboard
        dashboard_path = cross_analyzer.generate_comparison_dashboard()
        
        # Load the generated HTML
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            dashboard_html = f.read()
        
        # Add statistical analysis sections
        evidence_section = self._generate_evidence_section(conclusions, significant_differences)
        stats_section = self._generate_extended_statistics_section(extended_stats)
        
        # Insert the sections before the closing body tag
        dashboard_html = dashboard_html.replace('</body>', f'{evidence_section}{stats_section}</body>')
        
        # Save the enhanced dashboard
        enhanced_path = os.path.join(self.output_dir, "statistical_analysis_dashboard.html")
        with open(enhanced_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        return enhanced_path
    
    def _generate_evidence_section(
        self,
        conclusions: Dict[str, Any],
        significant_differences: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Generate HTML for the evidence-based conclusions section.
        
        Args:
            conclusions: Dictionary of conclusions
            significant_differences: Dictionary of significant differences
            
        Returns:
            HTML string for the evidence section
        """
        html = """
            <h2>Statistical Evidence and Conclusions</h2>
            <div class="summary">
                <h3>Key Findings</h3>
                <ul>
        """
        
        for finding in conclusions["key_findings"]:
            html += f"<li>{finding}</li>\n"
        
        html += """
                </ul>
            </div>
            
            <h3>Statistical Significance Tests</h3>
            <div class="tabset" id="evidence-tabs">
        """
        
        # Add tabs for each metric
        metrics = list(significant_differences.keys())
        for i, metric in enumerate(metrics):
            active_class = "active" if i == 0 else ""
            html += f'<div class="tab-label {active_class}" onclick="openTab(\'evidence-tab-{i}\', \'evidence-tabs\')">{metric.title()}</div>\n'
        
        html += "</div>\n"  # Close tab labels
        
        # Add tab content for each metric
        for i, metric in enumerate(metrics):
            active_class = "active" if i == 0 else ""
            diffs = significant_differences[metric]
            
            html += f"""
                <div id="evidence-tab-{i}" class="tab-content {active_class}">
                    <h4>Significant Differences for {metric.title()}</h4>
            """
            
            if diffs:
                html += """
                    <table class="stat-table">
                        <tr>
                            <th>Comparison</th>
                            <th>Better Method</th>
                            <th>Effect Size</th>
                            <th>Interpretation</th>
                            <th>Mann-Whitney p-value</th>
                            <th>t-test p-value</th>
                        </tr>
                """
                
                for diff in diffs:
                    html += f"""
                        <tr>
                            <td>{diff["comparison"]}</td>
                            <td><strong>{diff["better_method"]}</strong></td>
                            <td>{diff["effect_size"]:.4f}</td>
                            <td>{diff["effect_interpretation"]}</td>
                            <td>{diff["p_value_mw"]:.4f}</td>
                            <td>{diff["p_value_t"]:.4f}</td>
                        </tr>
                    """
                
                html += "</table>\n"
            else:
                html += "<p>No statistically significant differences found.</p>\n"
            
            html += "</div>\n"  # Close tab content
        
        # Method scores section
        html += """
            <h3>Method Effectiveness (Based on Statistical Significance)</h3>
            <div class="method-scores">
                <table class="stat-table">
                    <tr>
                        <th>Method</th>
                        <th>Statistical Wins</th>
                        <th>Significant Advantages</th>
                    </tr>
        """
        
        method_scores = conclusions["method_scores"]
        sorted_methods = sorted(method_scores.keys(), key=lambda x: method_scores[x], reverse=True)
        
        for method in sorted_methods:
            score = method_scores[method]
            
            # Count advantages by metric
            advantages = {}
            for metric, diffs in significant_differences.items():
                for diff in diffs:
                    if diff["better_method"] == method:
                        advantages[metric] = advantages.get(metric, 0) + 1
            
            advantages_text = ", ".join([f"{count} in {metric}" for metric, count in advantages.items()])
            if not advantages_text:
                advantages_text = "None"
            
            html += f"""
                <tr>
                    <td><strong>{method}</strong></td>
                    <td>{score}</td>
                    <td>{advantages_text}</td>
                </tr>
            """
        
        html += """
                </table>
            </div>
            
            <h3>Recommendations</h3>
            <div class="recommendations">
                <ul>
        """
        
        for recommendation in conclusions["recommendations"]:
            html += f"<li>{recommendation}</li>\n"
        
        html += """
                </ul>
            </div>
        """
        
        return html
    
    def _generate_extended_statistics_section(self, extended_stats: Dict[str, Any]) -> str:
        """
        Generate HTML for the extended statistics section.
        
        Args:
            extended_stats: Extended statistical measures
            
        Returns:
            HTML string for the extended statistics section
        """
        html = """
            <h2>Extended Statistical Measures</h2>
            <div class="tabset" id="stats-tabs">
        """
        
        # Add tabs for each metric
        metrics = list(extended_stats.keys())
        for i, metric in enumerate(metrics):
            active_class = "active" if i == 0 else ""
            html += f'<div class="tab-label {active_class}" onclick="openTab(\'stats-tab-{i}\', \'stats-tabs\')">{metric.title()}</div>\n'
        
        html += "</div>\n"  # Close tab labels
        
        # Add tab content for each metric
        for i, metric in enumerate(metrics):
            active_class = "active" if i == 0 else ""
            exp_stats = extended_stats[metric]
            
            html += f"""
                <div id="stats-tab-{i}" class="tab-content {active_class}">
                    <h4>Extended Statistics for {metric.title()}</h4>
                    <table class="stat-table">
                        <tr>
                            <th>Experiment Type</th>
                            <th>Mean</th>
                            <th>Median</th>
                            <th>Std Dev</th>
                            <th>95% CI</th>
                            <th>Min/Max</th>
                            <th>IQR</th>
                            <th>Shapiro-Wilk (Normality)</th>
                            <th>Skewness</th>
                            <th>Kurtosis</th>
                        </tr>
            """
            
            for exp_type, stats_dict in exp_stats.items():
                # Format confidence interval
                ci_low, ci_high = stats_dict["confidence_interval_95"]
                ci_formatted = f"({ci_low:.4f}, {ci_high:.4f})"
                
                # Format min/max
                min_max = f"{stats_dict['min']:.4f} / {stats_dict['max']:.4f}"
                
                # Format Shapiro-Wilk test result
                shapiro_result = f"{stats_dict['shapiro_test']['statistic']:.4f}, p={stats_dict['shapiro_test']['p_value']:.4f}"
                if stats_dict['shapiro_test']['is_normal']:
                    shapiro_result += " (Normal)"
                else:
                    shapiro_result += " (Non-normal)"
                
                html += f"""
                    <tr>
                        <td><strong>{exp_type}</strong></td>
                        <td>{stats_dict['mean']:.4f}</td>
                        <td>{stats_dict['median']:.4f}</td>
                        <td>{stats_dict['std']:.4f}</td>
                        <td>{ci_formatted}</td>
                        <td>{min_max}</td>
                        <td>{stats_dict['iqr']:.4f}</td>
                        <td>{shapiro_result}</td>
                        <td>{stats_dict['skewness']:.4f}</td>
                        <td>{stats_dict['kurtosis']:.4f}</td>
                    </tr>
                """
            
            html += """
                    </table>
                    <div class="interpretation">
                        <h5>Statistical Interpretation Guide:</h5>
                        <ul>
                            <li><strong>95% CI</strong>: 95% confidence interval for the true mean</li>
                            <li><strong>IQR</strong>: Interquartile range (Q3-Q1), robust measure of dispersion</li>
                            <li><strong>Shapiro-Wilk</strong>: Test for normality. P > 0.05 suggests normal distribution</li>
                            <li><strong>Skewness</strong>: Measure of asymmetry. 0 = symmetric, >0 = right skew, <0 = left skew</li>
                            <li><strong>Kurtosis</strong>: Measure of "tailedness". >0 = heavy tails, <0 = light tails</li>
                        </ul>
                    </div>
                </div>
            """
        
        return html 