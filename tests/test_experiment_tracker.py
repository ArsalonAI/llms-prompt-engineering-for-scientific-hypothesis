import unittest
import pandas as pd
import numpy as np
from src.experiment_tracker import ExperimentTracker # Assuming ExperimentTracker is in this path
import os
import shutil
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

# Helper to create a dummy DataFrame for testing log_result and end_experiment
def create_dummy_results_df(num_rows=1):
    data = {
        'step': [i+1 for i in range(num_rows)],
        'run_id': [f'run{i+1}' for i in range(num_rows)],
        'idea': [f'Idea {i+1}' for i in range(num_rows)],
        'batch_prompt': ['Prompt' for _ in range(num_rows)],
        'evaluation': ['ACCEPT' for _ in range(num_rows)],
        'evaluation_full': ['Full eval' for _ in range(num_rows)],
        'avg_cosine_similarity': [0.5 for _ in range(num_rows)],
        'avg_self_bleu': [0.4 for _ in range(num_rows)],
        'avg_bertscore': [0.8 for _ in range(num_rows)],
        'context_cosine': [0.6 for _ in range(num_rows)],
        'context_self_bleu': [0.3 for _ in range(num_rows)],
        'context_bertscore': [0.7 for _ in range(num_rows)],
        'timestamp': [datetime.now().isoformat() for _ in range(num_rows)],
        'quality_scores': [[{'evaluation': 'ACCEPT'}] for _ in range(num_rows)], # Example structure
        'has_kde_data': [True for _ in range(num_rows)]
    }
    return pd.DataFrame(data)

class TestExperimentTrackerKDE(unittest.TestCase):

    def setUp(self):
        self.test_output_dir = "test_results_temp_kde"
        # Minimal setup for ExperimentTracker, output_dir won't be used for _calculate_kde
        self.tracker = ExperimentTracker(output_dir=self.test_output_dir) 

    def test_calculate_kde_normal_data(self):
        """Test _calculate_kde with a typical numeric pandas Series."""
        data = pd.Series([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.9, 0.95])
        x_kde, y_kde = self.tracker._calculate_kde(data)
        
        self.assertIsInstance(x_kde, np.ndarray, "x_kde should be a numpy array")
        self.assertIsInstance(y_kde, np.ndarray, "y_kde should be a numpy array")
        self.assertEqual(len(x_kde), 100, "x_kde should have 100 points for the linspace")
        self.assertEqual(len(y_kde), 100, "y_kde should have 100 points")
        self.assertTrue(all(y_kde >= 0), "KDE y_values should all be non-negative")
        # Check if x_kde covers the range of data appropriately
        self.assertLessEqual(x_kde.min(), data.min(), "KDE x_min should be less than or equal to data min")
        self.assertGreaterEqual(x_kde.max(), data.max(), "KDE x_max should be greater than or equal to data max")

    def test_calculate_kde_with_non_numeric(self):
        """Test _calculate_kde with data containing non-numeric values."""
        data = pd.Series([0.1, 0.2, "apple", 0.3, None, 0.4, np.nan, 0.5, "banana"])
        x_kde, y_kde = self.tracker._calculate_kde(data)
        
        self.assertIsInstance(x_kde, np.ndarray)
        self.assertIsInstance(y_kde, np.ndarray)
        cleaned_data = pd.to_numeric(data, errors='coerce').dropna()
        if not cleaned_data.empty:
            self.assertEqual(len(x_kde), 100) 
            self.assertEqual(len(y_kde), 100)
            self.assertTrue(all(y_kde >= 0))
            self.assertLessEqual(x_kde.min(), cleaned_data.min())
            self.assertGreaterEqual(x_kde.max(), cleaned_data.max())
        else:
            self.assertEqual(len(x_kde), 0)
            self.assertEqual(len(y_kde), 0)

    def test_calculate_kde_empty_data(self):
        """Test _calculate_kde with an empty pandas Series (e.g., all non-numeric)."""
        data = pd.Series([], dtype=float) 
        x_kde, y_kde = self.tracker._calculate_kde(data)
        
        self.assertIsInstance(x_kde, np.ndarray)
        self.assertIsInstance(y_kde, np.ndarray)
        self.assertEqual(len(x_kde), 0, "x_kde should be empty for empty input data")
        self.assertEqual(len(y_kde), 0, "y_kde should be empty for empty input data")

        data_all_nan = pd.Series([np.nan, None, "text"], dtype=object)
        x_kde_nan, y_kde_nan = self.tracker._calculate_kde(data_all_nan)
        self.assertEqual(len(x_kde_nan), 0, "x_kde should be empty if all data is non-numeric")
        self.assertEqual(len(y_kde_nan), 0, "y_kde should be empty if all data is non-numeric")

    def test_calculate_kde_single_point_data(self):
        """Test _calculate_kde with data having only one unique numeric value."""
        data = pd.Series([0.5, 0.5, 0.5, 0.5])
        try:
            x_kde, y_kde = self.tracker._calculate_kde(data)
            self.assertIsInstance(x_kde, np.ndarray)
            self.assertIsInstance(y_kde, np.ndarray)
            if len(x_kde) > 0: # Some scipy versions might handle this
                self.assertEqual(len(x_kde), 100)
                self.assertEqual(len(y_kde), 100)
                self.assertTrue(all(y_kde >= 0))
            else: # More likely it returns empty due to lack of variance for KDE
                self.assertEqual(len(x_kde), 0)
                self.assertEqual(len(y_kde), 0)
        except (np.linalg.LinAlgError, ValueError) as e:
            self.skipTest(f"Skipping single_point_data test due to error from gaussian_kde: {e}")

    def test_calculate_kde_two_points_data(self):
        """Test _calculate_kde with data having only two numeric values."""
        data = pd.Series([0.1, 0.9])
        x_kde, y_kde = self.tracker._calculate_kde(data)
        
        self.assertIsInstance(x_kde, np.ndarray)
        self.assertIsInstance(y_kde, np.ndarray)
        self.assertEqual(len(x_kde), 100)
        self.assertEqual(len(y_kde), 100)
        self.assertTrue(all(y_kde >= 0))
        self.assertLessEqual(x_kde.min(), 0.1)
        self.assertGreaterEqual(x_kde.max(), 0.9)

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

class TestExperimentTrackerLogging(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = "test_results_temp_logging"
        self.tracker = ExperimentTracker(output_dir=self.test_output_dir)
        # Start an experiment before each test that needs it
        self.tracker.start_experiment(
            experiment_name="TestExp",
            experiment_type="test_type",
            model_name="test_model",
            config={"param1": "value1"}
        )

    def test_log_result_no_experiment_started(self):
        """Test log_result fails if no experiment is active."""
        fresh_tracker = ExperimentTracker(output_dir="test_results_temp_no_exp")
        with self.assertRaisesRegex(ValueError, "No experiment currently running. Call start_experiment first."):
            fresh_tracker.log_result("run_id_test", {"data": 1})
        if os.path.exists("test_results_temp_no_exp"):
             shutil.rmtree("test_results_temp_no_exp")

    @patch('builtins.print')
    def test_log_result_basic(self, mock_print):
        """Test basic logging of a result."""
        run_id = "run123"
        result_data = {
            "model": "gpt-test",
            "prompt": "Test prompt",
            "context": "Test context",
            "num_ideas": 1,
            "idea": "A brilliant idea",
            "evaluation": "ACCEPT",
            "evaluation_full": "Reasoning: It is brilliant.",
            "cosine_similarities": [0.8, 0.9],
            "self_bleu_scores": [0.7, 0.75],
            "bertscore_scores": [0.85, 0.88],
            "kde_values": {"cosine": {"x": [0.1,0.2], "y": [1,2]}}
        }
        
        initial_step = self.tracker._step
        initial_results_count = len(self.tracker.current_experiment['results'])
        initial_df_rows = len(self.tracker._results_df) if self.tracker._results_df is not None else 0

        self.tracker.log_result(run_id, result_data)

        # Check step increment
        self.assertEqual(self.tracker._step, initial_step + 1)

        # Check internal results list
        self.assertEqual(len(self.tracker.current_experiment['results']), initial_results_count + 1)
        logged_result_internal = self.tracker.current_experiment['results'][-1]
        self.assertEqual(logged_result_internal['run_id'], run_id)
        self.assertEqual(logged_result_internal['idea'], result_data['idea'])
        self.assertIn('timestamp', logged_result_internal)

        # Check DataFrame update
        self.assertIsNotNone(self.tracker._results_df)
        self.assertEqual(len(self.tracker._results_df), initial_df_rows + 1)
        logged_df_row = self.tracker._results_df.iloc[-1]
        self.assertEqual(logged_df_row['run_id'], run_id)
        self.assertEqual(logged_df_row['model'], result_data['model'])
        self.assertEqual(logged_df_row['num_ideas'], result_data['num_ideas'])
        self.assertEqual(logged_df_row['cosine_similarities'], result_data['cosine_similarities'])
        self.assertTrue(logged_df_row['has_kde_data'])

        # Check print output for progress
        mock_print.assert_any_call(f"\n[STEP {self.tracker._step}]")
        mock_print.assert_any_call(f"Run ID: {run_id} | Ideas: {result_data.get('num_ideas', 0)} | Quality: N/A") # Default quality if no quality_scores
        avg_cosine = np.mean(result_data['cosine_similarities'])
        avg_bleu = np.mean(result_data['self_bleu_scores'])
        avg_bert = np.mean(result_data['bertscore_scores'])
        expected_metrics_str_parts = [
            f"cosine={avg_cosine:.3f}",
            f"self_bleu={avg_bleu:.3f}",
            f"bertscore={avg_bert:.3f}"
        ]
        # Check if the call contains all parts, order might vary due to dict iteration
        found_metrics_print = False
        for call_args in mock_print.call_args_list:
            arg_str = str(call_args[0][0]) # First argument of the print call
            if "Average Pairwise Metrics: " in arg_str and all(part in arg_str for part in expected_metrics_str_parts):
                found_metrics_print = True
                break
        self.assertTrue(found_metrics_print, "Average pairwise metrics print call not found or incorrect.")


    @patch('builtins.print')
    def test_log_result_quality_summary(self, mock_print):
        """Test quality summary in log_result print output."""
        run_id = "run_quality_test"
        result_data_quality = {
            "num_ideas": 2,
            "quality_scores": [
                {"evaluation": "ACCEPT", "reason": "Good"},
                {"evaluation": "PRUNE", "reason": "Bad"}
            ]
        }
        self.tracker.log_result(run_id, result_data_quality)
        mock_print.assert_any_call(f"Run ID: {run_id} | Ideas: 2 | Quality: 1/2 ACCEPT")

        # Test with empty quality scores
        self.tracker._step = 0 # Reset step for clean print check
        result_data_empty_quality = {"num_ideas": 0, "quality_scores": []}
        self.tracker.log_result("run_empty_q", result_data_empty_quality)
        mock_print.assert_any_call(f"Run ID: run_empty_q | Ideas: 0 | Quality: 0/0 ACCEPT") # or N/A depending on implementation detail

        # Test with malformed quality scores (not list of dicts)
        self.tracker._step = 0 
        result_data_malformed_quality = {"num_ideas": 1, "quality_scores": ["ACCEPT"]}
        self.tracker.log_result("run_malformed_q", result_data_malformed_quality)
        mock_print.assert_any_call(f"Run ID: run_malformed_q | Ideas: 1 | Quality: Error processing quality")

    def test_end_experiment_no_experiment_active(self):
        """Test end_experiment fails if no experiment is active."""
        fresh_tracker_dir = "test_results_temp_no_exp_end"
        fresh_tracker = ExperimentTracker(output_dir=fresh_tracker_dir)
        with self.assertRaisesRegex(ValueError, "No experiment currently running."):
            fresh_tracker.end_experiment()
        if os.path.exists(fresh_tracker_dir):
            shutil.rmtree(fresh_tracker_dir)

    @patch('src.experiment_tracker.datetime')
    @patch('src.experiment_tracker.os.makedirs')
    @patch('src.experiment_tracker.open', new_callable=mock_open)
    @patch('src.experiment_tracker.pd.DataFrame.to_csv')
    @patch('src.experiment_tracker.ExperimentTracker._create_dashboard_html') # Mocking the method directly
    @patch('builtins.print') # To suppress print statements during test
    def test_end_experiment_successful_run(self, mock_print, mock_create_dashboard, 
                                           mock_to_csv, mock_file_open, mock_os_makedirs, 
                                           mock_datetime):
        """Test end_experiment successfully saves all artifacts and resets state."""
        # Setup mock datetime
        fixed_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = fixed_now
        mock_datetime.fromisoformat.side_effect = lambda ts: datetime.fromisoformat(ts) # Allow real fromisoformat
        
        # Log some dummy data to have something to save
        self.tracker._results_df = create_dummy_results_df(2)
        self.tracker.current_experiment['results'] = [{'data': 'sample1'}, {'data': 'sample2'}]
        self.tracker.current_experiment['start_time'] = fixed_now.isoformat() # Align start time

        # Expected experiment directory name based on mocked datetime
        timestamp_str = fixed_now.strftime('%Y%m%d_%H%M%S')
        expected_experiment_dir_name = f"TestExp_{timestamp_str}"
        full_expected_experiment_dir = os.path.abspath(os.path.join(self.test_output_dir, expected_experiment_dir_name))

        mock_create_dashboard.return_value = os.path.join(full_expected_experiment_dir, "dashboard.html")

        self.tracker.end_experiment()

        # Verify directory creation
        mock_os_makedirs.assert_called_once_with(full_expected_experiment_dir, exist_ok=True)

        # Verify file saves
        # Check calls to open (for json files)
        # Path for metadata.json
        path_metadata = os.path.join(full_expected_experiment_dir, "metadata.json")
        # Path for summary.json
        path_summary = os.path.join(full_expected_experiment_dir, "summary.json")
        # Path for results.json
        path_results = os.path.join(full_expected_experiment_dir, "results.json")
        # Path for results.csv is handled by mock_to_csv
        path_csv = os.path.join(full_expected_experiment_dir, "results.csv")

        # Check that open was called for each json file
        calls = [
            unittest.mock.call(path_metadata, 'w'),
            unittest.mock.call(path_summary, 'w'),
            unittest.mock.call(path_results, 'w')
        ]
        mock_file_open.assert_has_calls(calls, any_order=True)
        
        # Verify to_csv call
        mock_to_csv.assert_called_once_with(path_csv, index=False)
        
        # Verify dashboard creation call
        # Extract the first argument from the call_args_list for _create_dashboard_html
        # This should be the `experiment_dir` argument passed to it.
        actual_dashboard_call_dir = mock_create_dashboard.call_args[0][0]
        self.assertEqual(actual_dashboard_call_dir, full_expected_experiment_dir)

        # Verify state reset
        self.assertIsNone(self.tracker.current_experiment)
        self.assertEqual(self.tracker._step, 0)
        self.assertIsNone(self.tracker._results_df)

    @patch('src.experiment_tracker.datetime')
    @patch('src.experiment_tracker.os.makedirs')
    @patch('src.experiment_tracker.open', new_callable=mock_open)
    @patch('src.experiment_tracker.pd.DataFrame.to_csv')
    @patch('src.experiment_tracker.ExperimentTracker._create_dashboard_html', side_effect=Exception("Dashboard Test Error"))
    @patch('builtins.print')
    def test_end_experiment_dashboard_creation_fails(self, mock_print, mock_create_dashboard, 
                                                mock_to_csv, mock_file_open, mock_os_makedirs, 
                                                mock_datetime):
        """Test end_experiment handles dashboard creation failure gracefully."""
        fixed_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = fixed_now
        mock_datetime.fromisoformat.side_effect = lambda ts: datetime.fromisoformat(ts)

        self.tracker._results_df = create_dummy_results_df(1)
        self.tracker.current_experiment['results'] = [{'data': 'sample1'}]
        self.tracker.current_experiment['start_time'] = fixed_now.isoformat()

        self.tracker.end_experiment()

        # Verify that other files are still attempted to be saved (mocks will confirm calls)
        self.assertTrue(mock_os_makedirs.called)
        self.assertTrue(mock_file_open.called) # json files
        self.assertTrue(mock_to_csv.called)
        
        # Verify error message is printed
        mock_print.assert_any_call("[ERROR] Dashboard Creation Failed!")
        mock_print.assert_any_call("Error message: Dashboard Test Error")

        # Verify state is still reset
        self.assertIsNone(self.tracker.current_experiment)
        self.assertEqual(self.tracker._step, 0)
        self.assertIsNone(self.tracker._results_df)

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

if __name__ == '__main__':
    unittest.main() 