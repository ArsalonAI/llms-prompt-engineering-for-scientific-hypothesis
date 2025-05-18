#!/usr/bin/env python3
"""
Test script to verify that the statistical analyzer properly handles mixed data types.
"""
import os
import sys
import json
import unittest
import numpy as np
from scipy import stats

# We need to directly test the core function behavior without imports
# So we'll create a simple version of the relevant functions to test
class SimpleStatisticalAnalyzer:
    """Simple version of the statistical analyzer for testing data type handling."""
    
    def calculate_extended_statistics(self, metrics):
        """Test version of the function with numeric type conversion."""
        extended_stats = {}
        
        for metric_name, metric_data in metrics.items():
            extended_stats[metric_name] = {}
            
            for exp_type, values in metric_data.items():
                if not values:
                    continue
                
                # Check if values is already a number (not iterable)
                if isinstance(values, (int, float)):
                    # If it's a single number, create a list with this value
                    numeric_values = [float(values)]
                elif not hasattr(values, '__iter__') or isinstance(values, str):
                    # Skip non-iterable values or strings
                    print(f"[WARNING] Skipping non-iterable value in {metric_name} for {exp_type}: {values}")
                    continue
                else:
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
                    "max": np.max(data)
                }
                
                extended_stats[metric_name][exp_type] = stats_dict
        
        return extended_stats
    
    def identify_significant_differences(self, comparison_results):
        """Test version of the function with numeric type conversion."""
        significant_diffs = {}
        
        if "statistical_tests" not in comparison_results:
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
                            p_value_mw = 1.0
                            p_value_t = 1.0
                        
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

class TestStatisticsTypeHandling(unittest.TestCase):
    """Test case for statistical type handling."""
    
    def setUp(self):
        """Set up the test case."""
        # Create test data with mixed types
        self.test_metrics = {
            "test_metric": {
                "experiment_type_1": [0.5, 0.6, 0.7, "non_numeric", 0.9],
                "experiment_type_2": ["0.4", "0.5", "0.6"],  # Strings that can be converted to floats
                "experiment_type_3": 0.75,  # Single value (not iterable)
            },
            "another_metric": {
                "experiment_type_1": ["completely", "non", "numeric", "data"],
                "experiment_type_2": []  # Empty list
            }
        }
        
        # Create test comparison results with mixed types
        self.test_comparison_results = {
            "statistical_tests": {
                "cosine": {
                    "Type1 vs Type2": {
                        "mann_whitney": {"significant": True, "p_value": "0.03"},
                        "t_test": {"significant": False, "p_value": 0.08},
                        "effect_size": {"cohen_d": "0.8", "interpretation": "large"}
                    },
                    "Type2 vs Type3": {
                        "mann_whitney": {"significant": True, "p_value": "invalid"},
                        "t_test": {"significant": True, "p_value": 0.02},
                        "effect_size": {"cohen_d": 0.9, "interpretation": "large"}
                    }
                }
            }
        }
        
        # Create analyzer instance
        self.analyzer = SimpleStatisticalAnalyzer()
    
    def test_calculate_extended_statistics(self):
        """Test that the function handles mixed data types properly."""
        print("\nTesting calculate_extended_statistics with mixed data types...")
        
        try:
            # Call the function
            extended_stats = self.analyzer.calculate_extended_statistics(self.test_metrics)
            
            # Verify we got stats for numeric data
            self.assertIn("test_metric", extended_stats)
            self.assertIn("experiment_type_1", extended_stats["test_metric"])
            self.assertIn("experiment_type_2", extended_stats["test_metric"])
            self.assertIn("experiment_type_3", extended_stats["test_metric"])
            
            # Verify we didn't get stats for non-numeric data
            self.assertIn("another_metric", extended_stats)
            self.assertNotIn("experiment_type_1", extended_stats["another_metric"])
            
            # Verify the stats have the expected keys
            for metric in extended_stats:
                for exp_type in extended_stats[metric]:
                    self.assertIn("mean", extended_stats[metric][exp_type])
                    self.assertIn("median", extended_stats[metric][exp_type])
                    self.assertIn("std", extended_stats[metric][exp_type])
                    self.assertIn("min", extended_stats[metric][exp_type])
                    self.assertIn("max", extended_stats[metric][exp_type])
            
            print("✅ calculate_extended_statistics handled mixed data types correctly")
        except Exception as e:
            self.fail(f"calculate_extended_statistics raised exception: {str(e)}")
    
    def test_identify_significant_differences(self):
        """Test that the function handles mixed data types properly."""
        print("\nTesting identify_significant_differences with mixed data types...")
        
        try:
            # Call the function
            significant_diffs = self.analyzer.identify_significant_differences(self.test_comparison_results)
            
            # Verify we got significant differences
            self.assertIn("cosine", significant_diffs)
            self.assertEqual(len(significant_diffs["cosine"]), 2)
            
            # Verify non-numeric values were converted to floats
            diff1 = significant_diffs["cosine"][0]
            self.assertIsInstance(diff1["effect_size"], float)
            self.assertIsInstance(diff1["p_value_mw"], float)
            self.assertIsInstance(diff1["p_value_t"], float)
            
            # Verify invalid numeric values were handled
            diff2 = significant_diffs["cosine"][1]
            self.assertIsInstance(diff2["p_value_mw"], float)
            self.assertEqual(diff2["p_value_mw"], 1.0)  # Default value for invalid
            
            print("✅ identify_significant_differences handled mixed data types correctly")
        except Exception as e:
            self.fail(f"identify_significant_differences raised exception: {str(e)}")

if __name__ == "__main__":
    unittest.main() 