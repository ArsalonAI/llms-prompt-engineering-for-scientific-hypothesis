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
from cross_experiment_analysis.analyzer import CrossExperimentAnalyzer

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
        
        # Identify statistically significant differences
        significant_differences = self.identify_significant_differences(comparison_results)
        
        # Draw evidence-based conclusions
        conclusions = self.draw_evidence_based_conclusions(metrics, significant_differences)
        
        # Generate research dashboard
        dashboard_path = self.generate_research_dashboard(cross_analyzer, conclusions, significant_differences)
        
        return {
            "metrics": metrics,
            "comparison_results": comparison_results,
            "significant_differences": significant_differences,
            "conclusions": conclusions,
            "dashboard_path": dashboard_path
        }
    
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
                try:
                    # Skip the Kruskal-Wallis test entry
                    if comparison == "kruskal_wallis":
                        continue
                        
                    # Check if difference is statistically significant
                    is_significant = (
                        result.get("ks_test", {}).get("significant", False)  # KS test significance
                    )
                    
                    # Convert effect size to float if it's a string
                    effect_size_raw = result["effect_size"]["cohen_d"]
                    try:
                        effect_size = float(effect_size_raw)
                    except (ValueError, TypeError):
                        print(f"[WARNING] Invalid effect size in {metric} for {comparison}: {effect_size_raw}")
                        continue
                        
                    meaningful_effect = abs(effect_size) > 0.5
                    
                    if is_significant:
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
                        
                        # Ensure p-values are numeric
                        try:
                            p_value_ks = float(result.get("ks_test", {}).get("p_value", 1.0)) # KS p-value, default to 1.0 if not present
                        except (ValueError, TypeError):
                            print(f"[WARNING] Invalid p-values in {metric} for {comparison}")
                            p_value_ks = 1.0  # Default to non-significant
                        
                        significant_diffs[metric].append({
                            "comparison": comparison,
                            "effect_size": effect_size,
                            "effect_interpretation": result["effect_size"]["interpretation"],
                            "p_value_ks": p_value_ks, # KS p-value
                            "better_method": better_method
                        })
                except Exception as e:
                    print(f"[WARNING] Error processing {metric} comparison {comparison}: {str(e)}")
                    continue
        
        return significant_diffs
    
    def draw_evidence_based_conclusions(
        self, 
        metrics_data: Dict[str, Dict[str, List[float]]], 
        significant_differences: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Draw evidence-based conclusions from the data.
        
        Args:
            metrics_data: Extracted metrics from cross-experiment analysis
            significant_differences: Dictionary of significant differences
            
        Returns:
            Dictionary of conclusions
        """
        conclusions = {
            "overall_best_method": None,
            "method_scores": {},
            "metric_conclusions": {},
            "key_findings": [],
            "detailed_comparisons": {},
            "recommendations": [],
            "scenario_recommendations": {}
        }

        experiment_types = list(metrics_data.keys()) # Assumes top-level keys are experiment names/types
        if not experiment_types:
            conclusions["key_findings"].append("No experiment data found to draw conclusions.")
            return conclusions

        # --- Calculate overall method scores based on statistical wins (existing logic) ---
        # Initialize scores for each method based on presence in metrics_data
        for exp_type in experiment_types:
            if exp_type not in conclusions["method_scores"]:
                conclusions["method_scores"][exp_type] = 0
        
        # Consolidate unique experiment types mentioned in significant differences
        # (This ensures methods only in metrics_data but not in sig_diffs are initialized)
        sig_diff_exp_types = set()
        for metric, diffs in significant_differences.items():
            for diff in diffs:
                sig_diff_exp_types.add(diff["better_method"])
                comparison_parts = diff["comparison"].split(" vs ")
                sig_diff_exp_types.add(comparison_parts[0])
                if len(comparison_parts) > 1: sig_diff_exp_types.add(comparison_parts[1])
        
        for exp_type in sig_diff_exp_types:
            if exp_type not in conclusions["method_scores"]:
                conclusions["method_scores"][exp_type] = 0
        
        for metric, diffs in significant_differences.items():
            metric_conclusions_for_metric = {"best_method": None, "significant_findings": []}
            for diff in diffs:
                better_method = diff["better_method"]
                if better_method in conclusions["method_scores"]:
                    conclusions["method_scores"][better_method] += 1
                
                worse_method = diff["comparison"].replace(" vs ", "|").replace(better_method, "").replace("|", "")
                finding = f"{better_method} is significantly better than {worse_method} for {metric.replace('_', ' ').title()} (p_ks={diff['p_value_ks']:.4f}, effect: {diff['effect_interpretation']})"
                metric_conclusions_for_metric["significant_findings"].append(finding)
            
            current_best_method_for_metric = None
            current_max_score_for_metric = -1
            # This part needs refinement if we want per-metric best method based on its specific comparisons
            # For now, it's implicitly handled by overall method scores for key findings
            conclusions["metric_conclusions"][metric.replace('_', ' ').title()] = metric_conclusions_for_metric
        
        if conclusions["method_scores"]:
            sorted_method_scores = sorted(conclusions["method_scores"].items(), key=lambda item: item[1], reverse=True)
            if sorted_method_scores and sorted_method_scores[0][1] > 0:
                conclusions["overall_best_method"] = sorted_method_scores[0][0]
                conclusions["key_findings"].append(f"{conclusions['overall_best_method']} has the most statistical wins overall.")
        
        # --- Scenario-Based Recommendations --- 
        avg_context_cos_scores = {}
        avg_pairwise_cos_scores = {}
        
        # Calculate average scores for each experiment type (using raw_context_cosine and raw_pairwise_cosine)
        for exp_type in experiment_types:
            if exp_type in metrics_data:
                context_scores = metrics_data[exp_type].get("raw_context_cosine", [])
                pairwise_scores = metrics_data[exp_type].get("raw_pairwise_cosine", []) # Ensure this key exists from CrossExperimentAnalyzer
                
                avg_context_cos_scores[exp_type] = np.mean(context_scores) if context_scores else 0
                avg_pairwise_cos_scores[exp_type] = np.mean(pairwise_scores) if pairwise_scores else 0
            else: # Handle case where exp_type from sig_diffs might not be in top-level metrics_data keys
                avg_context_cos_scores[exp_type] = 0
                avg_pairwise_cos_scores[exp_type] = 0
        
        if not avg_context_cos_scores and not avg_pairwise_cos_scores:
            conclusions["scenario_recommendations"]["DataIssue"] = {
                "description": "Could not generate scenario-based recommendations due to missing average score data.",
                "methods": [],
                "justification": "Ensure 'raw_context_cosine' and 'raw_pairwise_cosine' lists are populated in metrics_data."
            }
        else:
            # Define thresholds (median of averages)
            median_context_cos = np.median(list(avg_context_cos_scores.values())) if avg_context_cos_scores else 0
            median_pairwise_cos = np.median(list(avg_pairwise_cos_scores.values())) if avg_pairwise_cos_scores else 0

            scenarios = {
                "Novel & Focused": {"desc": "Generate novel ideas (different from original) that are variations on a focused theme (similar to each other).", "methods": [], "details": []},
                "Incremental & Coherent": {"desc": "Generate ideas closely related to original input and also similar to each other.", "methods": [], "details": []},
                "Innovative & Diverse": {"desc": "Explore a wide range of innovative ideas that significantly depart from original input and are different from each other.", "methods": [], "details": []},
                "Relevant & Diverse": {"desc": "Generate ideas relevant to original input but explore diverse aspects.", "methods": [], "details": []}
            }

            for exp_type in experiment_types:
                ctx_score = avg_context_cos_scores.get(exp_type, 0)
                pwise_score = avg_pairwise_cos_scores.get(exp_type, 0)
                justification = f"{exp_type} (Context Cos: {ctx_score:.3f}, Pairwise Cos: {pwise_score:.3f})"

                if ctx_score <= median_context_cos and pwise_score > median_pairwise_cos: # Low Context, High Pairwise
                    scenarios["Novel & Focused"]["methods"].append(exp_type)
                    scenarios["Novel & Focused"]["details"].append(justification)
                elif ctx_score > median_context_cos and pwise_score > median_pairwise_cos: # High Context, High Pairwise
                    scenarios["Incremental & Coherent"]["methods"].append(exp_type)
                    scenarios["Incremental & Coherent"]["details"].append(justification)
                elif ctx_score <= median_context_cos and pwise_score <= median_pairwise_cos: # Low Context, Low Pairwise
                    scenarios["Innovative & Diverse"]["methods"].append(exp_type)
                    scenarios["Innovative & Diverse"]["details"].append(justification)
                elif ctx_score > median_context_cos and pwise_score <= median_pairwise_cos: # High Context, Low Pairwise
                    scenarios["Relevant & Diverse"]["methods"].append(exp_type)
                    scenarios["Relevant & Diverse"]["details"].append(justification)
            
            for scenario_name, data in scenarios.items():
                conclusions["scenario_recommendations"][scenario_name] = {
                    "description": data["desc"],
                    "methods": data["methods"],
                    "justification": "; ".join(data["details"]) if data["details"] else "N/A"
                }

        # Generic recommendations
        conclusions["recommendations"].append("Consider the specific strengths of each method for specialized tasks based on the scenario analysis.")
        conclusions["recommendations"].append("For the broadest range of hypotheses, consider using multiple prompt techniques in combination.")
        if conclusions["overall_best_method"]:
            conclusions["recommendations"].insert(0, f"Based on overall statistical wins, {conclusions['overall_best_method']} is generally recommended, but check scenario-specific advice.")

        return conclusions
    
    def generate_research_dashboard(
        self, 
        cross_analyzer: CrossExperimentAnalyzer,
        conclusions: Dict[str, Any],
        significant_differences: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Generate a research-grade dashboard with comprehensive statistical analysis.
        
        Args:
            cross_analyzer: CrossExperimentAnalyzer instance
            conclusions: Dictionary of conclusions
            significant_differences: Dictionary of significant differences
            
        Returns:
            Path to the generated dashboard
        """
        # First, generate the standard cross-experiment dashboard
        dashboard_path = cross_analyzer.generate_comparison_dashboard()
        
        # Load the generated HTML
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            dashboard_html = f.read()
        
        # Generate the overall conclusion section
        overall_conclusion_html = self._generate_overall_conclusion_html(conclusions)
        
        # Add statistical analysis sections
        evidence_section = self._generate_evidence_section(conclusions, significant_differences)
        
        # Insert the sections: Overall Conclusion first, then the rest of comparison dashboard, then evidence & stats.
        # We need a placeholder in the comparison_dashboard.html or a robust way to insert at the top.
        # For now, let's assume comparison_dashboard.html has a <div class="container">...</div> structure.
        # We will insert the overall conclusion right after the main H1 and before the existing summary.
        
        # A more robust way is to insert after a specific, unique comment/tag if we control comparison_dashboard.html more directly.
        # Or, modify comparison_dashboard.html generation to leave a placeholder.
        # Current approach: replace the body tag to prepend, then add the other sections at the end.

        # Prepend Overall Conclusion to the body of the base dashboard
        # This is a bit of a hack; ideally, the base dashboard has a placeholder.
        if "<body>" in dashboard_html:
            dashboard_html = dashboard_html.replace("<body>", f"<body>\n{overall_conclusion_html}", 1)
        else:
            # Fallback if no body tag (shouldn't happen for full HTML)
            dashboard_html = overall_conclusion_html + dashboard_html

        # Append the statistical sections to the end of the modified dashboard body
        dashboard_html = dashboard_html.replace('</body>', f'{evidence_section}</body>', 1)
        
        # Save the enhanced dashboard
        enhanced_path = os.path.join(self.output_dir, "statistical_analysis_dashboard.html")
        with open(enhanced_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        return enhanced_path
    
    def _generate_overall_conclusion_html(self, conclusions: Dict[str, Any]) -> str:
        """Generates an HTML section for the overall conclusions."""
        if not conclusions or not conclusions.get("key_findings"):
            return "<div class='overall-conclusions'><h2>Overall Conclusions</h2><p>No specific overall conclusions or key findings were generated based on the current data and significance criteria. Please review the detailed statistical tests and metric comparisons for more granular insights.</p></div>"

        html = "<div class='overall-conclusions'>"
        html += "<h2>Overall Conclusions & Key Findings</h2>"
        
        overall_best = conclusions.get("overall_best_method")
        if overall_best:
            html += f"<p><strong>Overall Most Effective Prompting Technique:</strong> {overall_best}</p>"
        else:
            html += "<p><strong>Overall Most Effective Prompting Technique:</strong> Not definitively identified based on current criteria.</p>"
        
        key_findings = conclusions.get("key_findings", [])
        if key_findings:
            html += "<h3>Key Findings:</h3><ul>"
            for finding in key_findings:
                html += f"<li>{finding}</li>"
            html += "</ul>"
        else:
            html += "<p>No specific key findings were highlighted.</p>"
            
        html += "</div>"
        return html

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
            <h2>Statistical Evidence and Method Comparison</h2>
            <div class="summary">
        """
        
        # Key Findings (from overall scoring, if kept)
        if conclusions.get("key_findings"):
            html += "<h3>Key Findings (Based on Statistical Wins):</h3><ul>"
            for finding in conclusions["key_findings"]:
                html += f"<li>{finding}</li>\n"
            html += "</ul>"

        # Scenario-based recommendations
        if conclusions.get("scenario_recommendations"):
            html += "<h3>Scenario-Based Recommendations:</h3><ul>"
            for rec_category, rec_details in conclusions["scenario_recommendations"].items():
                html += f"<li><strong>For {rec_category}:</strong> {rec_details['description']}"
                if rec_details['methods']:
                    html += " Consider using: " + ", ".join(rec_details['methods']) + "."
                else:
                    html += " No specific methods stood out for this profile based on current analysis."
                if rec_details.get('justification'):
                    html += f" (Justification: {rec_details['justification']})"
                html += "</li>"
            html += "</ul>"

        # Generic recommendations (if any are still desired here)
        if conclusions.get("recommendations"):
             html += "<h3>General Recommendations:</h3><ul>"
             for rec in conclusions["recommendations"]:
                 html += f"<li>{rec}</li>\n"
             html += "</ul>"

        html += """
            </div>
            
            <h3>Statistical Significance Tests (Pairwise Comparisons)</h3>
            <p class="section-description">Kolmogorov-Smirnov (KS) test compares distributions between pairs of experiment types for each metric. A small p-value (<0.05) indicates a significant difference. Effect size (Cohen's d) measures the magnitude.</p>
            <div style="max-height: 600px; overflow-y: auto;">
            <div class="tabset" id="evidence-tabs">
        """
        metrics_to_display = sorted([k for k in significant_differences.keys() if significant_differences[k]]) # Only display metrics with differences
        for i, metric in enumerate(metrics_to_display):
            active_class = "active" if i == 0 else ""
            # Make titles more readable
            title_metric = metric.replace("context_", "Context ").replace("pairwise_", "Pairwise ").replace("_", " ").title()
            html += f'<div class="tab-label {active_class}" onclick="openTab(\'evidence-tab-{i}\', \'evidence-tabs\')">{title_metric}</div>\n'
        html += "</div>\n"
        for i, metric in enumerate(metrics_to_display):
            active_class = "active" if i == 0 else ""
            diffs = significant_differences[metric]
            title_metric = metric.replace("context_", "Context ").replace("pairwise_", "Pairwise ").replace("_", " ").title()
            html += f'<div id="evidence-tab-{i}" class="tab-content {active_class}"><h4>Significant Differences for {title_metric}</h4>'
            if diffs:
                html += '<table class="stat-table"><tr><th>Comparison</th><th>Better Method</th><th>Effect Size</th><th>Interpretation</th><th>KS p-value</th></tr>'
                for diff in diffs:
                    html += f'<tr><td>{diff["comparison"]}</td><td><strong>{diff["better_method"]}</strong></td><td>{diff["effect_size"]:.4f}</td><td>{diff["effect_interpretation"]}</td><td>{diff["p_value_ks"]:.4f}</td></tr>'
                html += "</table>"
            else:
                html += "<p>No statistically significant differences found for this metric.</p>"
            html += "</div>"
        html += "</div></div>"
        # Method scores section (can be kept or removed based on preference)
        # ... (existing code for method_scores table) ... 
        return html 