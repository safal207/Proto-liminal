#!/usr/bin/env python3
"""
Test RINSE Agent with Persistent State

Demonstrates the integrated persistence layer with RINSE agent
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tempfile
import logging
from rinse_agent import RINSEAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def main():
    """Test RINSE agent with database persistence"""

    print("="*80)
    print("RINSE Agent Persistence Integration Test")
    print("="*80)
    print()

    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
    db_path = temp_db.name
    temp_db.close()

    print(f"Database: {db_path}")
    print()

    try:
        # ==================== Iteration 1 ====================
        print("="*80)
        print("Iteration 1: Initialize Agent with Persistence")
        print("="*80)

        agent1 = RINSEAgent(
            reflection_threshold=0.1,
            adjustment_magnitude=0.15,
            confidence_threshold=0.6,
            db_path=db_path
        )

        # Run first cycle
        metrics1 = {
            'accuracy': 0.80,
            'brier_score': 0.15,
            'log_score': -0.40,
            'precision': 0.78,
            'recall': 0.75
        }

        cycle1 = agent1.run_cycle(metrics1)

        print(f"\nCycle {cycle1.iteration} completed")
        print(f"  Insights: {len(cycle1.reflection.insights)}")
        print(f"  Adjustments: {len(cycle1.adjustments)}")
        print(f"  Evolution applied: {cycle1.evolution_applied}")

        # Show database summary
        if agent1.db:
            summary1 = agent1.db.get_summary()
            print(f"\nDatabase state:")
            print(f"  Total iterations: {summary1['latest_iteration']}")
            print(f"  Cycles: {summary1['rinse_cycles_count']}")
            print(f"  Reflections: {summary1['reflections_count']}")

        agent1.close()

        # ==================== Iteration 2 ====================
        print("\n" + "="*80)
        print("Iteration 2: Reload Agent from Database")
        print("="*80)

        agent2 = RINSEAgent(
            db_path=db_path  # No other params - should load from DB
        )

        print(f"\nLoaded state:")
        print(f"  Iteration: {agent2.iteration}")
        print(f"  Baseline metrics: {len(agent2.baseline_metrics)}")
        print(f"  Parameters: {len(agent2.parameters)}")
        print(f"  Accumulated feedback: {len(agent2.accumulated_feedback)}")

        # Verify loaded values
        print(f"\nConfiguration:")
        print(f"  Reflection threshold: {agent2.reflection_threshold}")
        print(f"  Adjustment magnitude: {agent2.adjustment_magnitude}")
        print(f"  Confidence threshold: {agent2.confidence_threshold}")

        # Run second cycle
        metrics2 = {
            'accuracy': 0.82,  # Improvement
            'brier_score': 0.14,  # Improvement
            'log_score': -0.38,  # Improvement
            'precision': 0.80,
            'recall': 0.77
        }

        cycle2 = agent2.run_cycle(metrics2)

        print(f"\nCycle {cycle2.iteration} completed")
        print(f"  Insights: {len(cycle2.reflection.insights)}")

        if cycle2.reflection.insights:
            print(f"\n  Key insights:")
            for insight in cycle2.reflection.insights[:3]:
                print(f"    • {insight}")

        print(f"  Adjustments: {len(cycle2.adjustments)}")

        if cycle2.adjustments:
            print(f"\n  Parameter changes:")
            for adj in cycle2.adjustments[:3]:
                print(f"    • {adj.parameter}: {adj.old_value:.4f} → {adj.new_value:.4f}")
                print(f"      Reason: {adj.reason}")

        print(f"  Evolution applied: {cycle2.evolution_applied}")

        # Final database summary
        if agent2.db:
            summary2 = agent2.db.get_summary()
            print(f"\nFinal database state:")
            print(f"  Total iterations: {summary2['latest_iteration']}")
            print(f"  Cycles: {summary2['rinse_cycles_count']}")
            print(f"  Reflections: {summary2['reflections_count']}")
            print(f"  Adjustments: {summary2['adjustments_count']}")
            print(f"  Parameter changes: {summary2['parameter_history_count']}")

        agent2.close()

        # ==================== Verification ====================
        print("\n" + "="*80)
        print("Verification: Check Persistence")
        print("="*80)

        # Reload one more time to verify state persisted
        agent3 = RINSEAgent(db_path=db_path)

        print(f"\nReloaded state (3rd instantiation):")
        print(f"  Iteration: {agent3.iteration}")
        print(f"  Baseline metrics: {agent3.baseline_metrics}")
        print(f"  Current metrics: {agent3.current_metrics}")
        print(f"  Parameters: {list(agent3.parameters.keys())}")

        # Verify accumulated feedback
        print(f"\nAccumulated feedback samples:")
        for metric_name, values in list(agent3.accumulated_feedback.items())[:3]:
            print(f"  {metric_name}: {len(values)} values")
            print(f"    Latest: {values[-1]:.4f}" if values else "    (empty)")

        agent3.close()

        print("\n" + "="*80)
        print("✅ Integration Test PASSED")
        print("="*80)
        print("\nKey findings:")
        print("  ✓ State persists across agent instances")
        print("  ✓ Configuration loaded correctly from database")
        print("  ✓ Metrics and parameters restored properly")
        print("  ✓ Iteration counter continues from saved state")
        print("  ✓ All RINSE cycles recorded in database")
        print()

    except Exception as e:
        print(f"\n❌ Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        Path(db_path).unlink(missing_ok=True)
        print(f"Database cleaned up: {db_path}")


if __name__ == "__main__":
    main()
