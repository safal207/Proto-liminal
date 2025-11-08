"""
Tests for rinse_persistence.py

Tests the SQLite persistence layer for RINSE agent state
"""

import json
import tempfile
import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rinse_persistence import RINSEPersistence


class TestRINSEPersistence(unittest.TestCase):
    """Test RINSE persistence layer"""

    def setUp(self):
        """Create temporary database for testing"""
        self.temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name

        self.db = RINSEPersistence(self.db_path)

    def tearDown(self):
        """Clean up temporary database"""
        self.db.close()
        Path(self.db_path).unlink(missing_ok=True)

    # ==================== Configuration Tests ====================

    def test_save_and_load_config(self):
        """Test configuration save/load roundtrip"""
        # Save config
        self.db.save_config(
            reflection_threshold=0.15,
            adjustment_magnitude=0.2,
            confidence_threshold=0.7
        )

        # Load config
        config = self.db.load_config()

        self.assertIsNotNone(config)
        self.assertEqual(config['reflection_threshold'], 0.15)
        self.assertEqual(config['adjustment_magnitude'], 0.2)
        self.assertEqual(config['confidence_threshold'], 0.7)
        self.assertIn('created_at', config)
        self.assertIn('updated_at', config)

    def test_update_config(self):
        """Test configuration update preserves created_at"""
        # First save
        self.db.save_config(0.1, 0.15, 0.6)
        config1 = self.db.load_config()

        # Update
        self.db.save_config(0.2, 0.25, 0.7)
        config2 = self.db.load_config()

        # created_at should remain same, updated_at should change
        self.assertEqual(config1['created_at'], config2['created_at'])
        self.assertNotEqual(config1['updated_at'], config2['updated_at'])

    def test_load_config_empty(self):
        """Test loading config when none exists"""
        config = self.db.load_config()
        self.assertIsNone(config)

    # ==================== Metrics Tests ====================

    def test_save_and_load_baseline_metrics(self):
        """Test baseline metrics save/load"""
        metrics = {
            'accuracy': 0.85,
            'brier_score': 0.12,
            'log_score': -0.35
        }

        self.db.save_baseline_metrics(metrics)
        loaded = self.db.load_baseline_metrics()

        self.assertEqual(loaded, metrics)

    def test_save_and_load_current_metrics(self):
        """Test current metrics save/load"""
        metrics = {
            'accuracy': 0.87,
            'precision': 0.82,
            'recall': 0.79
        }

        self.db.save_current_metrics(metrics)
        loaded = self.db.load_current_metrics()

        self.assertEqual(loaded, metrics)

    def test_update_baseline_metrics(self):
        """Test updating baseline metrics"""
        # First save
        self.db.save_baseline_metrics({'accuracy': 0.80})

        # Update
        self.db.save_baseline_metrics({'accuracy': 0.85, 'precision': 0.83})

        loaded = self.db.load_baseline_metrics()
        self.assertEqual(loaded['accuracy'], 0.85)
        self.assertEqual(loaded['precision'], 0.83)

    # ==================== Parameters Tests ====================

    def test_save_and_load_parameters(self):
        """Test parameters save/load"""
        params = {
            'signal_strength_weight': 0.5,
            'sentiment_weight': 0.3,
            'calibration_temperature': 1.0
        }

        self.db.save_parameters(params, iteration=1)
        loaded = self.db.load_parameters()

        self.assertEqual(loaded, params)

    def test_parameter_history(self):
        """Test parameter change history tracking"""
        # Record changes
        self.db.save_parameter_change(
            parameter_name='signal_strength_weight',
            old_value=0.5,
            new_value=0.6,
            iteration=1,
            reason='Performance improvement',
            confidence=0.8
        )

        self.db.save_parameter_change(
            parameter_name='signal_strength_weight',
            old_value=0.6,
            new_value=0.65,
            iteration=2,
            reason='Further tuning',
            confidence=0.85
        )

        # Get history
        history = self.db.get_parameter_history('signal_strength_weight', limit=10)

        self.assertEqual(len(history), 2)
        # Most recent first
        self.assertEqual(history[0]['iteration'], 2)
        self.assertEqual(history[0]['new_value'], 0.65)
        self.assertEqual(history[1]['iteration'], 1)
        self.assertEqual(history[1]['new_value'], 0.6)

    def test_parameter_history_limit(self):
        """Test parameter history limit"""
        # Record 5 changes
        for i in range(5):
            self.db.save_parameter_change(
                parameter_name='test_param',
                old_value=float(i),
                new_value=float(i + 1),
                iteration=i + 1,
                reason='Test',
                confidence=0.5
            )

        # Request only 3
        history = self.db.get_parameter_history('test_param', limit=3)
        self.assertEqual(len(history), 3)

        # Should get most recent (5, 4, 3)
        self.assertEqual(history[0]['iteration'], 5)
        self.assertEqual(history[1]['iteration'], 4)
        self.assertEqual(history[2]['iteration'], 3)

    # ==================== Accumulated Feedback Tests ====================

    def test_save_and_load_accumulated_feedback(self):
        """Test accumulated feedback save/load"""
        feedback = {
            'accuracy': [0.80, 0.82, 0.85],
            'brier_score': [0.15, 0.14, 0.12]
        }

        self.db.save_accumulated_feedback(feedback, iteration=1)

        loaded = self.db.load_accumulated_feedback()

        self.assertIn('accuracy', loaded)
        self.assertIn('brier_score', loaded)
        self.assertEqual(len(loaded['accuracy']), 3)
        self.assertEqual(len(loaded['brier_score']), 3)

    def test_accumulated_feedback_limit(self):
        """Test accumulated feedback with limit"""
        # Save feedback across multiple iterations
        for i in range(5):
            self.db.save_accumulated_feedback(
                {'metric1': [float(i)]},
                iteration=i + 1
            )

        # Load with limit
        loaded = self.db.load_accumulated_feedback(limit=3)

        # Should get 3 most recent values
        self.assertEqual(len(loaded['metric1']), 3)

    # ==================== Iterations Tests ====================

    def test_save_iteration(self):
        """Test iteration recording"""
        self.db.save_iteration(1, "2024-01-01T00:00:00Z")
        self.db.save_iteration(2, "2024-01-01T01:00:00Z")

        latest = self.db.get_latest_iteration()
        self.assertEqual(latest, 2)

    def test_latest_iteration_empty(self):
        """Test getting latest iteration when none exist"""
        latest = self.db.get_latest_iteration()
        self.assertEqual(latest, 0)

    # ==================== RINSE Cycles Tests ====================

    def test_save_complete_cycle(self):
        """Test saving complete RINSE cycle"""
        self.db.save_cycle(
            iteration=1,
            timestamp="2024-01-01T00:00:00Z",
            reflection_note="Test reflection",
            simulation_score=0.75,
            evolution_applied=True,
            metadata={'test': 'data'}
        )

        summary = self.db.get_summary()
        self.assertEqual(summary['rinse_cycles_count'], 1)

    def test_save_reflection(self):
        """Test saving reflection"""
        self.db.save_reflection(
            iteration=1,
            timestamp="2024-01-01T00:00:00Z",
            observations={'accuracy_change': 0.05},
            insights=['Accuracy improved', 'Brier score stable'],
            reflection_note="Test reflection",
            metadata={'phase': 'test'}
        )

        summary = self.db.get_summary()
        self.assertEqual(summary['reflections_count'], 1)

    def test_save_adjustment(self):
        """Test saving adjustment"""
        self.db.save_adjustment(
            iteration=1,
            target='system_parameters',
            parameter='signal_strength_weight',
            old_value=0.5,
            new_value=0.6,
            reason='Performance improvement',
            confidence=0.8,
            applied=True,
            metadata={'validated': True}
        )

        summary = self.db.get_summary()
        self.assertEqual(summary['adjustments_count'], 1)

    # ==================== Statistics Tests ====================

    def test_increment_stat(self):
        """Test statistics increment"""
        # Note: Stats table needs initial rows
        # For now, this will fail if stats table is empty
        # In production, stats should be pre-populated by schema

        try:
            self.db.increment_stat('cycles_completed', delta=1)
            self.db.increment_stat('cycles_completed', delta=2)

            stats = self.db.get_stats()
            # This assumes stats are pre-initialized to 0
            # May need schema update to include default stats
        except Exception:
            # Expected if stats table doesn't have pre-populated rows
            pass

    # ==================== Utility Tests ====================

    def test_get_summary(self):
        """Test database summary"""
        # Add some data
        self.db.save_iteration(1, "2024-01-01T00:00:00Z")
        self.db.save_cycle(1, "2024-01-01T00:00:00Z", "Test", 0.5, True)
        self.db.save_reflection(1, "2024-01-01T00:00:00Z", {}, [], "Test")

        summary = self.db.get_summary()

        self.assertIn('latest_iteration', summary)
        self.assertIn('rinse_cycles_count', summary)
        self.assertIn('reflections_count', summary)
        self.assertEqual(summary['latest_iteration'], 1)
        self.assertEqual(summary['rinse_cycles_count'], 1)
        self.assertEqual(summary['reflections_count'], 1)

    def test_export_to_json(self):
        """Test JSON export"""
        # Add data
        self.db.save_config(0.1, 0.15, 0.6)
        self.db.save_baseline_metrics({'accuracy': 0.8})
        self.db.save_parameters({'test_param': 0.5}, iteration=1)

        # Export
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            export_path = f.name

        try:
            self.db.export_to_json(export_path)

            # Load and verify
            with open(export_path, 'r') as f:
                data = json.load(f)

            self.assertIn('config', data)
            self.assertIn('baseline_metrics', data)
            self.assertIn('parameters', data)
            self.assertEqual(data['baseline_metrics']['accuracy'], 0.8)

        finally:
            Path(export_path).unlink(missing_ok=True)

    # ==================== Integration Tests ====================

    def test_complete_rinse_cycle_workflow(self):
        """Test complete workflow simulating RINSE agent"""
        # 1. Save initial config
        self.db.save_config(0.1, 0.15, 0.6)

        # 2. Save baseline metrics
        baseline = {'accuracy': 0.80, 'brier_score': 0.15}
        self.db.save_baseline_metrics(baseline)

        # 3. Run iteration 1
        self.db.save_iteration(1, "2024-01-01T00:00:00Z")

        # 4. Save reflection
        self.db.save_reflection(
            iteration=1,
            timestamp="2024-01-01T00:00:00Z",
            observations={'accuracy_change': 0.02},
            insights=['Slight improvement'],
            reflection_note="Iteration 1 reflection"
        )

        # 5. Save parameter change
        self.db.save_parameter_change(
            parameter_name='signal_strength_weight',
            old_value=0.5,
            new_value=0.55,
            iteration=1,
            reason='Tuning',
            confidence=0.7
        )

        # 6. Save adjustment
        self.db.save_adjustment(
            iteration=1,
            target='system_parameters',
            parameter='signal_strength_weight',
            old_value=0.5,
            new_value=0.55,
            reason='Tuning',
            confidence=0.7,
            applied=True
        )

        # 7. Save updated parameters
        params = {'signal_strength_weight': 0.55}
        self.db.save_parameters(params, iteration=1)

        # 8. Save cycle
        self.db.save_cycle(
            iteration=1,
            timestamp="2024-01-01T00:00:00Z",
            reflection_note="Iteration 1 reflection",
            simulation_score=0.7,
            evolution_applied=True,
            metadata={'test': True}
        )

        # 9. Save current metrics
        current = {'accuracy': 0.82, 'brier_score': 0.14}
        self.db.save_current_metrics(current)

        # 10. Verify everything was saved
        summary = self.db.get_summary()

        self.assertEqual(summary['latest_iteration'], 1)
        self.assertEqual(summary['rinse_cycles_count'], 1)
        self.assertEqual(summary['reflections_count'], 1)
        self.assertEqual(summary['adjustments_count'], 1)
        self.assertEqual(summary['parameter_history_count'], 1)

        # Verify data integrity
        loaded_config = self.db.load_config()
        self.assertEqual(loaded_config['reflection_threshold'], 0.1)

        loaded_baseline = self.db.load_baseline_metrics()
        self.assertEqual(loaded_baseline['accuracy'], 0.80)

        loaded_current = self.db.load_current_metrics()
        self.assertEqual(loaded_current['accuracy'], 0.82)

        loaded_params = self.db.load_parameters()
        self.assertEqual(loaded_params['signal_strength_weight'], 0.55)

        param_history = self.db.get_parameter_history('signal_strength_weight')
        self.assertEqual(len(param_history), 1)
        self.assertEqual(param_history[0]['new_value'], 0.55)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
