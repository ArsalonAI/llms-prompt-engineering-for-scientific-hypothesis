#!/usr/bin/env python3
"""
Test script to verify that our statistical analyzer can handle non-numeric data.
"""
import os
import sys
import json
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple

# Create a simplified version of StatisticalAnalyzer for testing
class TestStatisticalAnalyzer:
    """Simplified analyzer for testing the fix with mixed data types."""
    
    def __init__(self, current_results={}, experiment_dir="."):
        self.current_results = current_results
        self.experiment_dir = experiment_dir
        
    def calculate_extended_statistics(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate extended statistical measures with improved error handling.
        """
        extended_stats = {}
        
        for metric_name, metric_data in metrics.items():
            extended_stats[metric_name] = {}
            
            for exp_type, values in metric_data.items():
                if not values:
                    continue
                
                # Ensure values are numeric - filter out any non-numeric values
                numeric_values = []
                for val in values:
                    try:
                        # Try to convert to float
                        numeric_val = float(val)
                        numeric_values.append(numeric_val)
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        print(f"[WARNING] Skipping non-numeric value in {metric_name} for {exp_type}: {val}")
                        continue
                
                if not numeric_values:
                    print(f"[WARNING] No numeric values found in {metric_name} for {exp_type}. Skipping statistics calculation.")
                    continue
                
                # Convert to numpy array of numeric values
                data = np.array(numeric_values)
                
                # Calculate extended statistics
                stats_dict = {
                    "mean": np.mean(data),
                    "median": np.median(data),
                    "std": np.std(data),
                    "min": np.min(data),
                    "max": np.max(data),
                    "shapiro_test": {
                        "statistic": stats.shapiro(data)[0] if len(data) >= 3 else None,
                        "p_value": stats.shapiro(data)[1] if len(data) >= 3 else None,
                        "is_normal": stats.shapiro(data)[1] > 0.05 if len(data) >= 3 else None
                    }
                }
                
                extended_stats[metric_name][exp_type] = stats_dict
                
        return extended_stats
    
    def identify_significant_differences(self, comparison_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify statistically significant differences with improved error handling.
        """
        significant_diffs = {}
        
        if "statistical_tests" not in comparison_results:
            print("[WARNING] No statistical tests found in comparison results")
            return {}
        
        for metric, tests in comparison_results["statistical_tests"].items():
            significant_diffs[metric] = []
            
            for comparison, result in tests.items():
                try:
                    # Check if difference is statistically significant
                    is_significant = (
                        result["mann_whitney"]["significant"] or 
                        result["t_test"]["significant"]
                    )
                    
                    # Convert effect size to float if it's a string
                    effect_size_raw = result["effect_size"]["cohen_d"]
                    try:
                        effect_size = float(effect_size_raw)
                    except (ValueError, TypeError):
                        print(f"[WARNING] Invalid effect size in {metric} for {comparison}: {effect_size_raw}")
                        continue
                        
                    meaningful_effect = abs(effect_size) > 0.5
                    
                    if is_significant and meaningful_effect:
                        # Parse experiment types from comparison string (e.g., "Type1 vs Type2")
                        exp_types = comparison.split(" vs ")
                        if len(exp_types) != 2:
                            continue
                        
                        # Determine which method is better
                        better_method = exp_types[0]
                        
                        # Ensure p-values are numeric
                        try:
                            p_value_mw = float(result["mann_whitney"]["p_value"])
                            p_value_t = float(result["t_test"]["p_value"])
                        except (ValueError, TypeError):
                            print(f"[WARNING] Invalid p-values in {metric} for {comparison}")
                            p_value_mw = 1.0  # Default to non-significant
                            p_value_t = 1.0   # Default to non-significant
                        
                        significant_diffs[metric].append({
                            "comparison": comparison,
                            "effect_size": effect_size,
                            "effect_interpretation": result["effect_size"]["interpretation"],
                            "p_value_mw": p_value_mw,
                            "p_value_t": p_value_t,
                            "better_method": better_method
                        })
                except Exception as e:
                    print(f"[WARNING] Error processing {metric} comparison {comparison}: {str(e)}")
                    continue
        
        return significant_diffs

# Create a test metrics dictionary with mixed data types
test_metrics = {
    "test_metric": {
        "experiment_type_1": [0.5, 0.6, 0.7, "non_numeric", 0.9],
        "experiment_type_2": ["0.4", "0.5", "0.6"]  # Strings that can be converted to floats
    },
    "another_metric": {
        "experiment_type_1": ["completely", "non", "numeric", "data"],
        "experiment_type_2": []  # Empty list
    }
}

# Create a test comparison results dictionary with non-numeric data
test_comparison_results = {
    "statistical_tests": {
        "cosine": {
            "Type1 vs Type2": {
                "mann_whitney": {"significant": True, "p_value": "0.03"},
                "t_test": {"significant": False, "p_value": 0.08},
                "effect_size": {"cohen_d": "0.8", "interpretation": "large"}
            }
        }
    }
}

def test_statistical_analyzer():
    """Test that the statistical analyzer can handle non-numeric data."""
    print("\n=== Testing StatisticalAnalyzer with Mixed Data Types ===")
    
    # Set up analyzer with dummy data
    analyzer = TestStatisticalAnalyzer(
        current_results={},
        experiment_dir="."
    )
    
    # Test calculate_extended_statistics
    print("\nTesting calculate_extended_statistics...")
    try:
        extended_stats = analyzer.calculate_extended_statistics(test_metrics)
        print("✅ calculate_extended_statistics completed successfully")
        
        # Verify we got stats for numeric data but not for non-numeric
        if "test_metric" in extended_stats and "experiment_type_1" in extended_stats["test_metric"]:
            print("✅ Numeric data was processed correctly")
        else:
            print("❌ Failed to process numeric data")
            
        if "another_metric" in extended_stats and "experiment_type_1" in extended_stats["another_metric"]:
            print("❌ Non-numeric data was incorrectly processed")
        else:
            print("✅ Non-numeric data was correctly skipped")
    except Exception as e:
        print(f"❌ Error in calculate_extended_statistics: {str(e)}")
    
    # Test identify_significant_differences
    print("\nTesting identify_significant_differences...")
    try:
        significant_diffs = analyzer.identify_significant_differences(test_comparison_results)
        print("✅ identify_significant_differences completed successfully")
        
        # Verify we got significant differences from the string-based numeric data
        if "cosine" in significant_diffs and len(significant_diffs["cosine"]) > 0:
            diff = significant_diffs["cosine"][0]
            if isinstance(diff["effect_size"], float) and isinstance(diff["p_value_mw"], float):
                print("✅ String numeric values were correctly converted to floats")
            else:
                print("❌ String numeric values were not correctly converted to floats")
        else:
            print("❌ Failed to identify significant differences")
    except Exception as e:
        print(f"❌ Error in identify_significant_differences: {str(e)}")
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    test_statistical_analyzer() 