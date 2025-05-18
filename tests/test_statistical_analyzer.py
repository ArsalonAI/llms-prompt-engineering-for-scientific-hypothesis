"""
Tests for the statistical analysis module.
"""
import os
import tempfile
import unittest
import json
import shutil
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import pandas as pd

from src.statistical_analysis import StatisticalAnalyzer
from src.cross_experiment_analysis.analyzer import CrossExperimentAnalyzer


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test cases for the statistical analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample experiment results
        self.experiment_results = {
            "Scientific_Hypothesis": {
                "ideas": ["Idea 1", "Idea 2", "Idea 3"],
                "quality_scores": [
                    {"evaluation": "SCORE: 0.8", "evaluation_full": "Full eval", "score": 0.8},
                    {"evaluation": "SCORE: 0.9", "evaluation_full": "Full eval", "score": 0.9},
                    {"evaluation": "SCORE: 0.7", "evaluation_full": "Full eval", "score": 0.7}
                ],
                "cosine_similarities": [0.6, 0.65, 0.7],
                "self_bleu_scores": [0.5, 0.55, 0.6],
                "bertscore_scores": [0.8, 0.85, 0.9]
            },
            "Role_Based_Hypothesis": {
                "ideas": ["Idea 4", "Idea 5", "Idea 6"],
                "quality_scores": [
                    {"evaluation": "SCORE: 0.7", "evaluation_full": "Full eval", "score": 0.7},
                    {"evaluation": "SCORE: 0.8", "evaluation_full": "Full eval", "score": 0.8},
                    {"evaluation": "SCORE: 0.6", "evaluation_full": "Full eval", "score": 0.6}
                ],
                "cosine_similarities": [0.5, 0.55, 0.6],
                "self_bleu_scores": [0.4, 0.45, 0.5],
                "bertscore_scores": [0.7, 0.75, 0.8]
            },
            "Few_Shot_Hypothesis": {
                "ideas": ["Idea 7", "Idea 8", "Idea 9"],
                "quality_scores": [
                    {"evaluation": "SCORE: 0.9", "evaluation_full": "Full eval", "score": 0.9},
                    {"evaluation": "SCORE: 0.8", "evaluation_full": "Full eval", "score": 0.8},
                    {"evaluation": "SCORE: 0.7", "evaluation_full": "Full eval", "score": 0.7}
                ],
                "cosine_similarities": [0.7, 0.75, 0.8],
                "self_bleu_scores": [0.6, 0.65, 0.7],
                "bertscore_scores": [0.9, 0.95, 1.0]
            }
        }
        
        # Initialize the analyzer
        self.analyzer = StatisticalAnalyzer(
            current_results=self.experiment_results,
            experiment_dir=self.temp_dir
        )
        
        # Create mock cross experiment analyzer
        self.mock_cross_analyzer = MagicMock(spec=CrossExperimentAnalyzer)
        self.mock_cross_analyzer.extract_metrics.return_value = {
            "cosine": {
                "Scientific_Hypothesis": [0.6, 0.65, 0.7],
                "Role_Based_Hypothesis": [0.5, 0.55, 0.6],
                "Few_Shot_Hypothesis": [0.7, 0.75, 0.8]
            },
            "self_bleu": {
                "Scientific_Hypothesis": [0.5, 0.55, 0.6],
                "Role_Based_Hypothesis": [0.4, 0.45, 0.5],
                "Few_Shot_Hypothesis": [0.6, 0.65, 0.7]
            },
            "bertscore": {
                "Scientific_Hypothesis": [0.8, 0.85, 0.9],
                "Role_Based_Hypothesis": [0.7, 0.75, 0.8],
                "Few_Shot_Hypothesis": [0.9, 0.95, 1.0]
            }
        }
        
        self.mock_cross_analyzer.compare_metrics.return_value = {
            "statistical_tests": {
                "cosine": {
                    "Scientific_Hypothesis vs Role_Based_Hypothesis": {
                        "mann_whitney": {"p_value": 0.03, "significant": True},
                        "t_test": {"p_value": 0.04, "significant": True},
                        "effect_size": {"cohen_d": 0.8, "interpretation": "large"}
                    },
                    "Scientific_Hypothesis vs Few_Shot_Hypothesis": {
                        "mann_whitney": {"p_value": 0.01, "significant": True},
                        "t_test": {"p_value": 0.02, "significant": True},
                        "effect_size": {"cohen_d": -0.7, "interpretation": "large"}
                    },
                    "Role_Based_Hypothesis vs Few_Shot_Hypothesis": {
                        "mann_whitney": {"p_value": 0.01, "significant": True},
                        "t_test": {"p_value": 0.02, "significant": True},
                        "effect_size": {"cohen_d": -1.2, "interpretation": "very large"}
                    }
                },
                "self_bleu": {
                    "Scientific_Hypothesis vs Role_Based_Hypothesis": {
                        "mann_whitney": {"p_value": 0.03, "significant": True},
                        "t_test": {"p_value": 0.04, "significant": True},
                        "effect_size": {"cohen_d": 0.8, "interpretation": "large"}
                    },
                    "Scientific_Hypothesis vs Few_Shot_Hypothesis": {
                        "mann_whitney": {"p_value": 0.01, "significant": True},
                        "t_test": {"p_value": 0.02, "significant": True},
                        "effect_size": {"cohen_d": -0.7, "interpretation": "large"}
                    },
                    "Role_Based_Hypothesis vs Few_Shot_Hypothesis": {
                        "mann_whitney": {"p_value": 0.01, "significant": True},
                        "t_test": {"p_value": 0.02, "significant": True},
                        "effect_size": {"cohen_d": -1.2, "interpretation": "very large"}
                    }
                },
                "bertscore": {
                    "Scientific_Hypothesis vs Role_Based_Hypothesis": {
                        "mann_whitney": {"p_value": 0.03, "significant": True},
                        "t_test": {"p_value": 0.04, "significant": True},
                        "effect_size": {"cohen_d": 0.8, "interpretation": "large"}
                    },
                    "Scientific_Hypothesis vs Few_Shot_Hypothesis": {
                        "mann_whitney": {"p_value": 0.01, "significant": True},
                        "t_test": {"p_value": 0.02, "significant": True},
                        "effect_size": {"cohen_d": -0.7, "interpretation": "large"}
                    },
                    "Role_Based_Hypothesis vs Few_Shot_Hypothesis": {
                        "mann_whitney": {"p_value": 0.01, "significant": True},
                        "t_test": {"p_value": 0.02, "significant": True},
                        "effect_size": {"cohen_d": -1.2, "interpretation": "very large"}
                    }
                }
            }
        }
        
        # Sample HTML for the dashboard
        self.mock_html = """<!DOCTYPE html>
        <html>
        <head>
            <title>Comparison Dashboard</title>
        </head>
        <body>
            <h1>Experiment Comparison Dashboard</h1>
            <div id="content">
                <!-- Content will be here -->
            </div>
        </body>
        </html>"""
        
        dashboard_path = os.path.join(self.temp_dir, "comparison_dashboard.html")
        with open(dashboard_path, "w") as f:
            f.write(self.mock_html)
            
        self.mock_cross_analyzer.generate_comparison_dashboard.return_value = dashboard_path
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch("src.statistical_analysis.analyzer.CrossExperimentAnalyzer")
    def test_perform_analysis(self, mock_cross_analyzer_class):
        """Test the perform_analysis method."""
        # Setup mock
        mock_cross_analyzer_class.return_value = self.mock_cross_analyzer
        
        # Run the analysis
        results = self.analyzer.perform_analysis()
        
        # Assert cross analyzer was used correctly
        mock_cross_analyzer_class.assert_called_once()
        self.mock_cross_analyzer.load_experiment_results.assert_called_once()
        self.mock_cross_analyzer.extract_metrics.assert_called_once()
        self.mock_cross_analyzer.compare_metrics.assert_called_once()
        
        # Check results structure
        self.assertIn("metrics", results)
        self.assertIn("comparison_results", results)
        self.assertIn("extended_statistics", results)
        self.assertIn("significant_differences", results)
        self.assertIn("conclusions", results)
        self.assertIn("dashboard_path", results)
        
        # Check dashboard was created
        dashboard_path = results["dashboard_path"]
        self.assertTrue(os.path.exists(dashboard_path))
        
    def test_calculate_extended_statistics(self):
        """Test the calculate_extended_statistics method."""
        # Sample metrics
        metrics = {
            "cosine": {
                "Scientific_Hypothesis": [0.6, 0.65, 0.7],
                "Role_Based_Hypothesis": [0.5, 0.55, 0.6],
                "Few_Shot_Hypothesis": [0.7, 0.75, 0.8]
            }
        }
        
        # Calculate extended statistics
        extended_stats = self.analyzer.calculate_extended_statistics(metrics)
        
        # Check structure
        self.assertIn("cosine", extended_stats)
        for exp_type in ["Scientific_Hypothesis", "Role_Based_Hypothesis", "Few_Shot_Hypothesis"]:
            self.assertIn(exp_type, extended_stats["cosine"])
            
            stats_dict = extended_stats["cosine"][exp_type]
            # Check all expected statistics are present
            expected_stats = ["mean", "median", "std", "min", "max", "q1", "q3", "iqr", 
                             "skewness", "kurtosis", "shapiro_test", "confidence_interval_95"]
            for stat in expected_stats:
                self.assertIn(stat, stats_dict)
            
            # Check specific values for Scientific_Hypothesis
            if exp_type == "Scientific_Hypothesis":
                self.assertAlmostEqual(stats_dict["mean"], 0.65, places=2)
                self.assertAlmostEqual(stats_dict["median"], 0.65, places=2)
                
    def test_identify_significant_differences(self):
        """Test the identify_significant_differences method."""
        # Sample comparison results
        comparison_results = {
            "statistical_tests": {
                "cosine": {
                    "Scientific_Hypothesis vs Role_Based_Hypothesis": {
                        "mann_whitney": {"p_value": 0.03, "significant": True},
                        "t_test": {"p_value": 0.04, "significant": True},
                        "effect_size": {"cohen_d": 0.8, "interpretation": "large"}
                    }
                }
            }
        }
        
        # Identify significant differences
        significant_diffs = self.analyzer.identify_significant_differences(comparison_results)
        
        # Check structure
        self.assertIn("cosine", significant_diffs)
        self.assertEqual(len(significant_diffs["cosine"]), 1)
        
        # Check specific values
        diff = significant_diffs["cosine"][0]
        self.assertEqual(diff["comparison"], "Scientific_Hypothesis vs Role_Based_Hypothesis")
        self.assertEqual(diff["effect_size"], 0.8)
        self.assertEqual(diff["effect_interpretation"], "large")
        self.assertEqual(diff["p_value_mw"], 0.03)
        self.assertEqual(diff["p_value_t"], 0.04)
        # For cosine, lower is better, so positive effect size means second method is better
        self.assertEqual(diff["better_method"], "Role_Based_Hypothesis")
        
    def test_draw_evidence_based_conclusions(self):
        """Test the draw_evidence_based_conclusions method."""
        # Sample metrics
        metrics = {
            "cosine": {
                "Scientific_Hypothesis": [0.6, 0.65, 0.7],
                "Role_Based_Hypothesis": [0.5, 0.55, 0.6],
                "Few_Shot_Hypothesis": [0.7, 0.75, 0.8]
            }
        }
        
        # Sample significant differences
        significant_differences = {
            "cosine": [
                {
                    "comparison": "Scientific_Hypothesis vs Role_Based_Hypothesis",
                    "effect_size": 0.8,
                    "effect_interpretation": "large",
                    "p_value_mw": 0.03,
                    "p_value_t": 0.04,
                    "better_method": "Role_Based_Hypothesis"
                }
            ]
        }
        
        # Draw conclusions
        conclusions = self.analyzer.draw_evidence_based_conclusions(metrics, significant_differences)
        
        # Check structure
        expected_keys = ["overall_best_method", "method_scores", "metric_conclusions", 
                        "key_findings", "detailed_comparisons", "recommendations"]
        for key in expected_keys:
            self.assertIn(key, conclusions)
        
        # Check specific values
        self.assertEqual(conclusions["overall_best_method"], "Role_Based_Hypothesis")
        self.assertEqual(conclusions["method_scores"]["Role_Based_Hypothesis"], 1)
        
        # Check key findings
        self.assertGreater(len(conclusions["key_findings"]), 0)
        
        # Check recommendations
        self.assertGreater(len(conclusions["recommendations"]), 0)
        
    def test_generate_research_dashboard(self):
        """Test the generate_research_dashboard method."""
        # Sample conclusions
        conclusions = {
            "overall_best_method": "Role_Based_Hypothesis",
            "method_scores": {"Role_Based_Hypothesis": 1},
            "metric_conclusions": {"cosine": {"best_method": "Role_Based_Hypothesis", "significant_findings": []}},
            "key_findings": ["Role_Based_Hypothesis is the best method"],
            "detailed_comparisons": {},
            "recommendations": ["Use Role_Based_Hypothesis"]
        }
        
        # Sample significant differences
        significant_differences = {
            "cosine": [
                {
                    "comparison": "Scientific_Hypothesis vs Role_Based_Hypothesis",
                    "effect_size": 0.8,
                    "effect_interpretation": "large",
                    "p_value_mw": 0.03,
                    "p_value_t": 0.04,
                    "better_method": "Role_Based_Hypothesis"
                }
            ]
        }
        
        # Sample extended statistics
        extended_stats = {
            "cosine": {
                "Scientific_Hypothesis": {
                    "mean": 0.65,
                    "median": 0.65,
                    "std": 0.05,
                    "min": 0.6,
                    "max": 0.7,
                    "q1": 0.6,
                    "q3": 0.7,
                    "iqr": 0.1,
                    "skewness": 0.0,
                    "kurtosis": -1.5,
                    "shapiro_test": {
                        "statistic": 0.99,
                        "p_value": 0.8,
                        "is_normal": True
                    },
                    "confidence_interval_95": (0.55, 0.75)
                }
            }
        }
        
        # Generate dashboard
        dashboard_path = self.analyzer.generate_research_dashboard(
            self.mock_cross_analyzer, conclusions, significant_differences, extended_stats
        )
        
        # Check dashboard was created
        self.assertTrue(os.path.exists(dashboard_path))
        
        # Read the dashboard content
        with open(dashboard_path, "r") as f:
            content = f.read()
        
        # Check dashboard content
        self.assertIn("Statistical Evidence and Conclusions", content)
        self.assertIn("Extended Statistical Measures", content)
        self.assertIn("Role_Based_Hypothesis is the best method", content)
        
    def test_generate_evidence_section(self):
        """Test the _generate_evidence_section private method."""
        # Sample conclusions
        conclusions = {
            "overall_best_method": "Role_Based_Hypothesis",
            "method_scores": {"Role_Based_Hypothesis": 1},
            "metric_conclusions": {"cosine": {"best_method": "Role_Based_Hypothesis", "significant_findings": []}},
            "key_findings": ["Role_Based_Hypothesis is the best method"],
            "detailed_comparisons": {},
            "recommendations": ["Use Role_Based_Hypothesis"]
        }
        
        # Sample significant differences
        significant_differences = {
            "cosine": [
                {
                    "comparison": "Scientific_Hypothesis vs Role_Based_Hypothesis",
                    "effect_size": 0.8,
                    "effect_interpretation": "large",
                    "p_value_mw": 0.03,
                    "p_value_t": 0.04,
                    "better_method": "Role_Based_Hypothesis"
                }
            ]
        }
        
        # Generate evidence section
        html = self.analyzer._generate_evidence_section(conclusions, significant_differences)
        
        # Check HTML content
        self.assertIsInstance(html, str)
        self.assertIn("Statistical Evidence and Conclusions", html)
        self.assertIn("Role_Based_Hypothesis", html)
        self.assertIn("Scientific_Hypothesis vs Role_Based_Hypothesis", html)
        
    def test_generate_extended_statistics_section(self):
        """Test the _generate_extended_statistics_section private method."""
        # Sample extended statistics
        extended_stats = {
            "cosine": {
                "Scientific_Hypothesis": {
                    "mean": 0.65,
                    "median": 0.65,
                    "std": 0.05,
                    "min": 0.6,
                    "max": 0.7,
                    "q1": 0.6,
                    "q3": 0.7,
                    "iqr": 0.1,
                    "skewness": 0.0,
                    "kurtosis": -1.5,
                    "shapiro_test": {
                        "statistic": 0.99,
                        "p_value": 0.8,
                        "is_normal": True
                    },
                    "confidence_interval_95": (0.55, 0.75)
                }
            }
        }
        
        # Generate extended statistics section
        html = self.analyzer._generate_extended_statistics_section(extended_stats)
        
        # Check HTML content
        self.assertIsInstance(html, str)
        self.assertIn("Extended Statistical Measures", html)
        self.assertIn("Scientific_Hypothesis", html)
        self.assertIn("95% CI", html)
        self.assertIn("Shapiro-Wilk", html)
        

if __name__ == "__main__":
    unittest.main() 