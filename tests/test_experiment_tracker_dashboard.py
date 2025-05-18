import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import plotly.graph_objects as go

from src.experiment_tracker import ExperimentTracker


class TestExperimentTrackerDashboard(unittest.TestCase):
    """Test cases for ExperimentTracker dashboard creation and results table."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_output_dir = "test_results_temp_dashboard"
        self.tracker = ExperimentTracker(output_dir=self.test_output_dir)
        
        # Create a mock DataFrame that will be used by _create_results_table
        self.tracker._results_df = pd.DataFrame({
            'step': [1],
            'run_id': ['test_run'],
            'ideas': [[
                "Idea 1", 
                "Idea 2", 
                "Idea 3"
            ]],
            'context_cosine': [[0.75, 0.65, 0.85]],
            'context_self_bleu': [[0.30, 0.25, 0.35]],
            'context_bertscore': [[0.80, 0.70, 0.90]]
        })
    
    def test_create_results_table_with_dictionary_quality_scores(self):
        """Test _create_results_table with quality_scores as a list of dictionaries."""
        # Set quality_scores as dictionaries (the correct format after our fix)
        self.tracker._results_df['quality_scores'] = [[
            {'evaluation': 'ACCEPT', 'evaluation_full': 'Good idea 1'},
            {'evaluation': 'PRUNE', 'evaluation_full': 'Bad idea 2'},
            {'evaluation': 'ACCEPT', 'evaluation_full': 'Great idea 3'}
        ]]
        
        # Call the method
        fig = self.tracker._create_results_table()
        
        # Check that a figure was returned
        self.assertIsInstance(fig, go.Figure)
        
        # Verify figure data structure
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Table)
        
        # Check headers
        self.assertEqual(fig.data[0].header.values[0], 'Idea #')
        self.assertEqual(fig.data[0].header.values[2], 'Evaluation')
        
        # Check cells
        cells = fig.data[0].cells.values
        # Check idea indices
        self.assertEqual(cells[0], [1, 2, 3])
        # Check ideas text
        self.assertEqual(cells[1], ["Idea 1", "Idea 2", "Idea 3"])
        # Check evaluation summaries
        self.assertEqual(cells[2], ['ACCEPT', 'PRUNE', 'ACCEPT'])
    
    def test_create_results_table_with_float_quality_scores(self):
        """Test _create_results_table with quality_scores as a list of floats (the bug case)."""
        # Set quality_scores as floats (which would have caused an error before our fix)
        self.tracker._results_df['quality_scores'] = [[0.7, 0.4, 0.9]]
        
        # Call the method (should not raise AttributeError after our fix)
        fig = self.tracker._create_results_table()
        
        # Check that a figure was returned
        self.assertIsInstance(fig, go.Figure)
        
        # Verify figure data structure
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Table)
        
        # Check headers
        self.assertEqual(fig.data[0].header.values[0], 'Idea #')
        self.assertEqual(fig.data[0].header.values[2], 'Evaluation')
        
        # Check cells
        cells = fig.data[0].cells.values
        # These should be the string representations of the float values after our fix
        self.assertEqual(cells[2], ['0.7', '0.4', '0.9'])
    
    def test_create_results_table_with_mixed_quality_scores(self):
        """Test _create_results_table with a mix of quality score types."""
        # Set quality_scores as a mix of dicts and floats
        self.tracker._results_df['quality_scores'] = [[
            {'evaluation': 'ACCEPT', 'evaluation_full': 'Good idea 1'},
            0.4,  # This would have caused an error before our fix
            {'evaluation': 'ACCEPT', 'evaluation_full': 'Great idea 3'}
        ]]
        
        # Call the method (should not raise AttributeError after our fix)
        fig = self.tracker._create_results_table()
        
        # Check that a figure was returned
        self.assertIsInstance(fig, go.Figure)
        
        # Check cells
        cells = fig.data[0].cells.values
        # The first and third should be from dicts, the second should be converted from float
        self.assertEqual(cells[2], ['ACCEPT', '0.4', 'ACCEPT'])
    
    def test_create_results_table_with_none_quality_scores(self):
        """Test _create_results_table with None in quality_scores."""
        # Set quality_scores with None values
        self.tracker._results_df['quality_scores'] = [[
            {'evaluation': 'ACCEPT', 'evaluation_full': 'Good idea 1'},
            None,  # This should be handled gracefully
            {'evaluation': 'ACCEPT', 'evaluation_full': 'Great idea 3'}
        ]]
        
        # Call the method (should not raise AttributeError)
        fig = self.tracker._create_results_table()
        
        # Check that a figure was returned
        self.assertIsInstance(fig, go.Figure)
        
        # Check cells
        cells = fig.data[0].cells.values
        # The None should be displayed as 'N/A'
        self.assertEqual(cells[2], ['ACCEPT', 'N/A', 'ACCEPT'])
    
    def test_create_results_table_with_no_quality_scores(self):
        """Test _create_results_table with no quality_scores column."""
        # Remove quality_scores column
        if 'quality_scores' in self.tracker._results_df.columns:
            self.tracker._results_df = self.tracker._results_df.drop('quality_scores', axis=1)
        
        # Call the method (should still work)
        fig = self.tracker._create_results_table()
        
        # Check that a figure was returned
        self.assertIsInstance(fig, go.Figure)
        
        # Check cells
        cells = fig.data[0].cells.values
        # All evaluations should be 'N/A'
        self.assertEqual(cells[2], ['N/A', 'N/A', 'N/A'])
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        import os
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)


if __name__ == '__main__':
    unittest.main() 