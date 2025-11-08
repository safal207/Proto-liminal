"""
Module: rinse_persistence.py
Purpose: Persistent state management for RINSE agent using SQLite

Enables RINSE agent to preserve learning across sessions:
- Configuration
- Baseline metrics
- Parameter history
- Reflection logs
- Accumulated feedback

Philosophy: "Memory is the foundation of consciousness.
A system without memory cannot evolve."
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


class RINSEPersistence:
    """
    Persistent storage for RINSE agent state

    Uses SQLite for:
    - Configuration preservation
    - Metric baselines and history
    - Parameter evolution tracking
    - Reflection and adjustment logs
    """

    def __init__(self, db_path: str = "data/rinse_state.db"):
        """
        Initialize persistence layer

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = None
        self._init_database()

        LOGGER.info(f"RINSEPersistence initialized: {self.db_path}")

    def _init_database(self):
        """Initialize database with schema"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Access columns by name

        # Load schema
        schema_path = Path(__file__).parent / "db_schema.sql"

        if schema_path.exists():
            with open(schema_path) as f:
                schema_sql = f.read()
            self.conn.executescript(schema_sql)
            self.conn.commit()
            LOGGER.info("Database schema initialized")
        else:
            LOGGER.warning(f"Schema file not found: {schema_path}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            LOGGER.info("Database connection closed")

    # ==================== Configuration ====================

    def save_config(
        self,
        reflection_threshold: float,
        adjustment_magnitude: float,
        confidence_threshold: float
    ):
        """Save RINSE agent configuration"""
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute("""
            INSERT OR REPLACE INTO rinse_config (
                id, reflection_threshold, adjustment_magnitude,
                confidence_threshold, created_at, updated_at
            ) VALUES (1, ?, ?, ?, COALESCE((SELECT created_at FROM rinse_config WHERE id=1), ?), ?)
        """, (reflection_threshold, adjustment_magnitude, confidence_threshold, now, now))

        self.conn.commit()
        LOGGER.debug("Config saved")

    def load_config(self) -> Optional[Dict]:
        """Load RINSE agent configuration"""
        cursor = self.conn.execute("SELECT * FROM rinse_config WHERE id = 1")
        row = cursor.fetchone()

        if row:
            return {
                'reflection_threshold': row['reflection_threshold'],
                'adjustment_magnitude': row['adjustment_magnitude'],
                'confidence_threshold': row['confidence_threshold'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            }
        return None

    # ==================== Metrics ====================

    def save_baseline_metrics(self, metrics: Dict[str, float]):
        """Save baseline metrics"""
        now = datetime.now(timezone.utc).isoformat()

        for name, value in metrics.items():
            self.conn.execute("""
                INSERT OR REPLACE INTO baseline_metrics (metric_name, value, established_at, updated_at)
                VALUES (?, ?, COALESCE((SELECT established_at FROM baseline_metrics WHERE metric_name=?), ?), ?)
            """, (name, value, name, now, now))

        self.conn.commit()
        LOGGER.debug(f"Baseline metrics saved: {len(metrics)} metrics")

    def load_baseline_metrics(self) -> Dict[str, float]:
        """Load baseline metrics"""
        cursor = self.conn.execute("SELECT metric_name, value FROM baseline_metrics")
        return {row['metric_name']: row['value'] for row in cursor.fetchall()}

    def save_current_metrics(self, metrics: Dict[str, float]):
        """Save current metrics"""
        now = datetime.now(timezone.utc).isoformat()

        for name, value in metrics.items():
            self.conn.execute("""
                INSERT OR REPLACE INTO current_metrics (metric_name, value, updated_at)
                VALUES (?, ?, ?)
            """, (name, value, now))

        self.conn.commit()
        LOGGER.debug(f"Current metrics saved: {len(metrics)} metrics")

    def load_current_metrics(self) -> Dict[str, float]:
        """Load current metrics"""
        cursor = self.conn.execute("SELECT metric_name, value FROM current_metrics")
        return {row['metric_name']: row['value'] for row in cursor.fetchall()}

    # ==================== Accumulated Feedback ====================

    def save_accumulated_feedback(
        self,
        feedback: Dict[str, List[float]],
        iteration: int
    ):
        """Save accumulated feedback for an iteration"""
        now = datetime.now(timezone.utc).isoformat()

        for metric_name, values in feedback.items():
            for value in values:
                self.conn.execute("""
                    INSERT INTO accumulated_feedback (metric_name, value, iteration, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (metric_name, value, iteration, now))

        self.conn.commit()
        LOGGER.debug(f"Accumulated feedback saved for iteration {iteration}")

    def load_accumulated_feedback(
        self,
        limit: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Load accumulated feedback

        Args:
            limit: Max items per metric (most recent)

        Returns:
            Dict mapping metric_name to list of values
        """
        # Get all feedback grouped by metric
        cursor = self.conn.execute("""
            SELECT metric_name, value, iteration
            FROM accumulated_feedback
            ORDER BY metric_name, iteration DESC
        """)

        feedback = {}
        for row in cursor.fetchall():
            metric_name = row['metric_name']

            if metric_name not in feedback:
                feedback[metric_name] = []

            # Apply limit per metric
            if limit is None or len(feedback[metric_name]) < limit:
                feedback[metric_name].append(row['value'])

        # Reverse to get chronological order
        for metric_name in feedback:
            feedback[metric_name] = list(reversed(feedback[metric_name]))

        return feedback

    # ==================== Parameters ====================

    def save_parameters(self, parameters: Dict[str, float], iteration: Optional[int] = None):
        """Save current parameter values"""
        now = datetime.now(timezone.utc).isoformat()

        for name, value in parameters.items():
            self.conn.execute("""
                INSERT OR REPLACE INTO parameters (name, value, updated_at, updated_by_iteration)
                VALUES (?, ?, ?, ?)
            """, (name, value, now, iteration))

        self.conn.commit()
        LOGGER.debug(f"Parameters saved: {len(parameters)} params")

    def load_parameters(self) -> Dict[str, float]:
        """Load current parameter values"""
        cursor = self.conn.execute("SELECT name, value FROM parameters")
        return {row['name']: row['value'] for row in cursor.fetchall()}

    def save_parameter_change(
        self,
        parameter_name: str,
        old_value: float,
        new_value: float,
        iteration: int,
        reason: str,
        confidence: float
    ):
        """Record parameter evolution"""
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute("""
            INSERT INTO parameter_history (
                parameter_name, old_value, new_value, iteration, reason, confidence, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (parameter_name, old_value, new_value, iteration, reason, confidence, now))

        self.conn.commit()
        LOGGER.debug(f"Parameter change logged: {parameter_name} {old_value:.3f} → {new_value:.3f}")

    def get_parameter_history(self, parameter_name: str, limit: int = 10) -> List[Dict]:
        """Get parameter evolution history"""
        cursor = self.conn.execute("""
            SELECT * FROM parameter_history
            WHERE parameter_name = ?
            ORDER BY iteration DESC
            LIMIT ?
        """, (parameter_name, limit))

        return [dict(row) for row in cursor.fetchall()]

    # ==================== Iterations ====================

    def save_iteration(self, iteration: int, timestamp: str):
        """Record RINSE iteration"""
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute("""
            INSERT OR IGNORE INTO rinse_iterations (iteration, timestamp, created_at)
            VALUES (?, ?, ?)
        """, (iteration, timestamp, now))

        self.conn.commit()

    def get_latest_iteration(self) -> int:
        """Get latest iteration number"""
        cursor = self.conn.execute("SELECT MAX(iteration) as max_iter FROM rinse_iterations")
        row = cursor.fetchone()
        return row['max_iter'] if row['max_iter'] is not None else 0

    # ==================== RINSE Cycles ====================

    def save_cycle(
        self,
        iteration: int,
        timestamp: str,
        reflection_note: str,
        simulation_score: Optional[float],
        evolution_applied: bool,
        metadata: Optional[Dict] = None
    ):
        """Save complete RINSE cycle"""
        self.conn.execute("""
            INSERT OR REPLACE INTO rinse_cycles (
                iteration, timestamp, reflection_note, simulation_score,
                evolution_applied, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            iteration, timestamp, reflection_note, simulation_score,
            1 if evolution_applied else 0,
            json.dumps(metadata) if metadata else None
        ))

        self.conn.commit()
        LOGGER.debug(f"RINSE cycle {iteration} saved")

    def save_reflection(
        self,
        iteration: int,
        timestamp: str,
        observations: Dict,
        insights: List[str],
        reflection_note: str,
        metadata: Optional[Dict] = None
    ):
        """Save reflection"""
        self.conn.execute("""
            INSERT OR REPLACE INTO reflections (
                iteration, timestamp, observations, insights, reflection_note, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            iteration, timestamp,
            json.dumps(observations),
            json.dumps(insights),
            reflection_note,
            json.dumps(metadata) if metadata else None
        ))

        self.conn.commit()

    def save_adjustment(
        self,
        iteration: int,
        target: str,
        parameter: str,
        old_value: float,
        new_value: float,
        reason: str,
        confidence: float,
        applied: bool,
        metadata: Optional[Dict] = None
    ):
        """Save adjustment record"""
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute("""
            INSERT INTO adjustments (
                iteration, target, parameter, old_value, new_value,
                reason, confidence, applied, metadata, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            iteration, target, parameter, old_value, new_value,
            reason, confidence, 1 if applied else 0,
            json.dumps(metadata) if metadata else None,
            now
        ))

        self.conn.commit()

    # ==================== Statistics ====================

    def increment_stat(self, key: str, delta: int = 1):
        """Increment a statistic counter"""
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute("""
            UPDATE stats SET value = value + ?, updated_at = ? WHERE key = ?
        """, (delta, now, key))

        self.conn.commit()

    def get_stats(self) -> Dict[str, int]:
        """Get all statistics"""
        cursor = self.conn.execute("SELECT key, value FROM stats")
        return {row['key']: row['value'] for row in cursor.fetchall()}

    # ==================== Utility ====================

    def get_summary(self) -> Dict:
        """Get database summary statistics"""
        summary = {}

        # Count records
        tables = [
            'rinse_iterations',
            'rinse_cycles',
            'reflections',
            'adjustments',
            'parameter_history',
            'accumulated_feedback'
        ]

        for table in tables:
            cursor = self.conn.execute(f"SELECT COUNT(*) as cnt FROM {table}")
            summary[f"{table}_count"] = cursor.fetchone()['cnt']

        # Latest iteration
        summary['latest_iteration'] = self.get_latest_iteration()

        # Stats
        summary['stats'] = self.get_stats()

        return summary

    def export_to_json(self, output_path: str):
        """Export database to JSON for analysis"""
        data = {
            'config': self.load_config(),
            'baseline_metrics': self.load_baseline_metrics(),
            'current_metrics': self.load_current_metrics(),
            'parameters': self.load_parameters(),
            'accumulated_feedback': self.load_accumulated_feedback(),
            'stats': self.get_stats(),
            'summary': self.get_summary()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        LOGGER.info(f"Database exported to {output_path}")


# ==================== CLI ====================

def main():
    """CLI for database management"""
    import argparse

    parser = argparse.ArgumentParser(description="RINSE Persistence Manager")
    parser.add_argument('--db', default='data/rinse_state.db', help='Database path')
    parser.add_argument('--summary', action='store_true', help='Show summary')
    parser.add_argument('--export', help='Export to JSON file')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    db = RINSEPersistence(args.db)

    if args.summary:
        summary = db.get_summary()
        print("="*60)
        print("RINSE Database Summary")
        print("="*60)
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print("="*60)

    if args.export:
        db.export_to_json(args.export)
        print(f"Exported to: {args.export}")

    db.close()


if __name__ == "__main__":
    main()
