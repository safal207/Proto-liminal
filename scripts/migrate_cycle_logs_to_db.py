#!/usr/bin/env python3
"""
Migration Script: JSONL Cycle Logs → SQLite Database

Converts existing RINSE cycle logs from cycle_logs/*.jsonl
into the new SQLite persistent state database.

Usage:
    python scripts/migrate_cycle_logs_to_db.py --input cycle_logs/ --db data/rinse_state.db
    python scripts/migrate_cycle_logs_to_db.py --input cycle_logs/ --db data/rinse_state.db --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rinse_persistence import RINSEPersistence

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOGGER = logging.getLogger(__name__)


class CycleLogMigrator:
    """Migrate RINSE cycle logs from JSONL to SQLite"""

    def __init__(self, db_path: str, dry_run: bool = False):
        """
        Initialize migrator

        Args:
            db_path: Path to SQLite database
            dry_run: If True, parse but don't write to database
        """
        self.dry_run = dry_run
        self.db = None if dry_run else RINSEPersistence(db_path)

        # Statistics
        self.stats = {
            'files_processed': 0,
            'cycles_imported': 0,
            'reflections_imported': 0,
            'adjustments_imported': 0,
            'parameters_imported': 0,
            'errors': 0
        }

        LOGGER.info(f"Migrator initialized (dry_run={dry_run})")

    def load_cycle_from_jsonl(self, file_path: Path) -> Optional[Dict]:
        """
        Load RINSE cycle from JSONL file

        Args:
            file_path: Path to cycle JSONL file

        Returns:
            Cycle dictionary or None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                LOGGER.warning(f"Empty file: {file_path}")
                return None

            # Parse last line (most recent cycle)
            cycle_data = json.loads(lines[-1].strip())
            return cycle_data

        except json.JSONDecodeError as e:
            LOGGER.error(f"JSON decode error in {file_path}: {e}")
            self.stats['errors'] += 1
            return None

        except Exception as e:
            LOGGER.error(f"Error loading {file_path}: {e}")
            self.stats['errors'] += 1
            return None

    def migrate_cycle(self, cycle_data: Dict):
        """
        Migrate a single RINSE cycle to database

        Args:
            cycle_data: Cycle dictionary from JSONL
        """
        try:
            iteration = cycle_data.get('iteration')
            timestamp = cycle_data.get('timestamp')

            if iteration is None or not timestamp:
                LOGGER.warning("Missing iteration or timestamp, skipping")
                return

            # Extract reflection data
            reflection = cycle_data.get('reflection', {})
            observations = reflection.get('observations', {})
            insights = reflection.get('insights', [])
            reflection_note = reflection.get('reflection_note', '')

            # Extract metadata
            metadata = cycle_data.get('metadata', {})
            baseline_metrics = metadata.get('baseline_metrics', {})
            current_metrics = metadata.get('current_metrics', {})
            parameters = metadata.get('parameters', {})

            # Extract cycle info
            simulation_score = cycle_data.get('simulation_score', 0.0)
            evolution_applied = cycle_data.get('evolution_applied', False)
            adjustments = cycle_data.get('adjustments', [])

            if self.dry_run:
                LOGGER.info(f"[DRY RUN] Would import cycle {iteration}")
                self.stats['cycles_imported'] += 1
                self.stats['reflections_imported'] += 1
                self.stats['adjustments_imported'] += len(adjustments)
                self.stats['parameters_imported'] += len(parameters)
                return

            # Save to database
            # 1. Save iteration
            self.db.save_iteration(iteration, timestamp)

            # 2. Save reflection
            self.db.save_reflection(
                iteration=iteration,
                timestamp=timestamp,
                observations=observations,
                insights=insights,
                reflection_note=reflection_note,
                metadata=reflection.get('metadata')
            )

            # 3. Save complete cycle
            self.db.save_cycle(
                iteration=iteration,
                timestamp=timestamp,
                reflection_note=reflection_note,
                simulation_score=simulation_score,
                evolution_applied=evolution_applied,
                metadata=metadata
            )

            # 4. Save adjustments
            for adj in adjustments:
                self.db.save_adjustment(
                    iteration=iteration,
                    target=adj.get('target', 'unknown'),
                    parameter=adj.get('parameter', 'unknown'),
                    old_value=adj.get('old_value', 0.0),
                    new_value=adj.get('new_value', 0.0),
                    reason=adj.get('reason', ''),
                    confidence=adj.get('confidence', 0.0),
                    applied=evolution_applied,
                    metadata=adj.get('metadata')
                )

            # 5. Save metrics if this is the first or latest cycle
            if baseline_metrics:
                self.db.save_baseline_metrics(baseline_metrics)
            if current_metrics:
                self.db.save_current_metrics(current_metrics)

            # 6. Save parameters
            if parameters:
                self.db.save_parameters(parameters, iteration)

            # Update statistics
            self.stats['cycles_imported'] += 1
            self.stats['reflections_imported'] += 1
            self.stats['adjustments_imported'] += len(adjustments)
            self.stats['parameters_imported'] += len(parameters)

            LOGGER.info(f"✓ Migrated cycle {iteration} ({len(adjustments)} adjustments)")

        except Exception as e:
            LOGGER.error(f"Error migrating cycle: {e}")
            self.stats['errors'] += 1

    def migrate_directory(self, input_dir: Path):
        """
        Migrate all JSONL files in directory

        Args:
            input_dir: Directory containing cycle_logs/*.jsonl
        """
        if not input_dir.exists():
            LOGGER.error(f"Input directory not found: {input_dir}")
            return

        # Find all JSONL files
        jsonl_files = sorted(input_dir.glob("*.jsonl"))

        if not jsonl_files:
            LOGGER.warning(f"No JSONL files found in {input_dir}")
            return

        LOGGER.info(f"Found {len(jsonl_files)} JSONL files to process")

        # Process each file
        for file_path in jsonl_files:
            LOGGER.info(f"Processing {file_path.name}...")

            cycle_data = self.load_cycle_from_jsonl(file_path)

            if cycle_data:
                self.migrate_cycle(cycle_data)
                self.stats['files_processed'] += 1

    def print_summary(self):
        """Print migration summary"""
        print("\n" + "="*60)
        print("Migration Summary")
        print("="*60)
        print(f"Mode:                {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"Files processed:     {self.stats['files_processed']}")
        print(f"Cycles imported:     {self.stats['cycles_imported']}")
        print(f"Reflections:         {self.stats['reflections_imported']}")
        print(f"Adjustments:         {self.stats['adjustments_imported']}")
        print(f"Parameters:          {self.stats['parameters_imported']}")
        print(f"Errors:              {self.stats['errors']}")

        if not self.dry_run and self.db:
            print("\n" + "="*60)
            print("Database Summary")
            print("="*60)
            summary = self.db.get_summary()
            print(f"Total iterations:    {summary['latest_iteration']}")
            print(f"Cycles recorded:     {summary['rinse_cycles_count']}")
            print(f"Reflections:         {summary['reflections_count']}")
            print(f"Adjustments:         {summary['adjustments_count']}")

        print("="*60)

    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()


def main():
    """Main migration entry point"""
    parser = argparse.ArgumentParser(
        description="Migrate RINSE cycle logs from JSONL to SQLite"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input directory containing cycle_logs/*.jsonl'
    )
    parser.add_argument(
        '--db',
        required=True,
        help='Output SQLite database path'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Parse files but do not write to database'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = Path(args.input)

    print("="*60)
    print("RINSE Cycle Log Migration")
    print("="*60)
    print(f"Input:  {input_dir}")
    print(f"Output: {args.db}")
    print(f"Mode:   {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("="*60)
    print()

    if args.dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made to database")
        print()

    # Create migrator
    migrator = CycleLogMigrator(
        db_path=args.db,
        dry_run=args.dry_run
    )

    try:
        # Run migration
        migrator.migrate_directory(input_dir)

        # Print summary
        migrator.print_summary()

        print("\n✅ Migration complete!")

        if args.dry_run:
            print("\n💡 Run without --dry-run to actually import data")

        return 0

    except KeyboardInterrupt:
        print("\n⏹  Migration cancelled")
        return 1

    except Exception as e:
        LOGGER.error(f"Migration failed: {e}")
        return 1

    finally:
        migrator.close()


if __name__ == "__main__":
    sys.exit(main())
