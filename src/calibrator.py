"""
Module: calibrator.py
Purpose: Probability calibration stub with isotonic regression support
Part of LIMINAL ProtoConsciousness MVP — see docs/MVP_SPEC.md
"""
import logging
import pickle
from pathlib import Path
from typing import List, Optional

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    IsotonicRegression = None

LOGGER = logging.getLogger(__name__)


def load_calibrator(path: str) -> Optional[object]:
    """
    Load calibrator model from pickle file

    Args:
        path: Path to pickled calibrator model

    Returns:
        Calibrator object or None if not found/loadable
    """
    if not path:
        return None

    path_obj = Path(path)

    if not path_obj.exists():
        LOGGER.warning(f"Calibrator not found: {path}")
        return None

    try:
        with open(path, 'rb') as f:
            calibrator = pickle.load(f)

        LOGGER.info(f"Loaded calibrator from {path}")
        return calibrator

    except Exception as exc:
        LOGGER.error(f"Error loading calibrator: {exc}")
        return None


def apply_calibration(p_raw: float, calibrator: Optional[object] = None) -> float:
    """
    Apply calibration to raw probability

    Args:
        p_raw: Raw probability from predictor
        calibrator: Calibrator object (optional)

    Returns:
        Calibrated probability (or p_raw if no calibrator)
    """
    if calibrator is None:
        # No calibrator available - uncalibrated mode
        LOGGER.debug("Running in uncalibrated mode (no calibrator provided)")
        return p_raw

    try:
        # Apply calibrator
        if hasattr(calibrator, 'predict'):
            # sklearn-style calibrator
            p_cal = float(calibrator.predict([p_raw])[0])
        elif callable(calibrator):
            # Function-style calibrator
            p_cal = float(calibrator(p_raw))
        else:
            LOGGER.warning("Invalid calibrator object, returning raw probability")
            return p_raw

        # Ensure in valid range
        p_cal = max(0.0, min(1.0, p_cal))

        return p_cal

    except Exception as exc:
        LOGGER.error(f"Calibration error: {exc}, returning raw probability")
        return p_raw


def fit_isotonic(
    probabilities: List[float],
    outcomes: List[float],
    save_path: str
) -> Optional[object]:
    """
    Fit isotonic regression calibrator and save to disk

    Args:
        probabilities: List of predicted probabilities
        outcomes: List of actual outcomes (0 or 1)
        save_path: Path to save pickled calibrator

    Returns:
        Fitted calibrator object or None on error
    """
    if IsotonicRegression is None:
        LOGGER.error("sklearn not installed, cannot fit isotonic calibrator")
        return None

    if len(probabilities) != len(outcomes):
        LOGGER.error("Probabilities and outcomes must have same length")
        return None

    if len(probabilities) < 10:
        LOGGER.warning("Insufficient data for isotonic calibration (need ≥10 samples)")
        return None

    try:
        # Fit isotonic regression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(probabilities, outcomes)

        LOGGER.info(f"Fitted isotonic calibrator on {len(probabilities)} samples")

        # Save to disk
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(calibrator, f)

        LOGGER.info(f"Saved calibrator to {save_path}")

        return calibrator

    except Exception as exc:
        LOGGER.error(f"Error fitting isotonic calibrator: {exc}")
        return None


def main():
    """CLI interface for calibrator"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit isotonic calibrator from historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python src/calibrator.py \\
    --labels data/outcomes/labels_202510.csv \\
    --out models/isotonic.pkl
        """
    )

    parser.add_argument(
        '--labels',
        required=True,
        help='CSV file with columns: probability,outcome'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output pickle file for calibrator'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Load labels CSV
    try:
        import pandas as pd

        df = pd.read_csv(args.labels)

        if 'probability' not in df.columns or 'outcome' not in df.columns:
            LOGGER.error("CSV must have 'probability' and 'outcome' columns")
            return 1

        probabilities = df['probability'].tolist()
        outcomes = df['outcome'].tolist()

        LOGGER.info(f"Loaded {len(probabilities)} samples from {args.labels}")

    except ImportError:
        LOGGER.error("pandas is required. Install with: pip install pandas")
        return 1
    except FileNotFoundError:
        LOGGER.error(f"Labels file not found: {args.labels}")
        return 1
    except Exception as exc:
        LOGGER.error(f"Error loading labels: {exc}")
        return 1

    # Fit calibrator
    calibrator = fit_isotonic(probabilities, outcomes, args.out)

    if calibrator is None:
        LOGGER.error("Failed to fit calibrator")
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print("Calibrator Summary")
    print("=" * 60)
    print(f"Samples used:         {len(probabilities)}")
    print(f"Calibrator type:      Isotonic Regression")
    print(f"Output file:          {args.out}")

    return 0


if __name__ == "__main__":
    exit(main())
