# LIMINAL ProtoConsciousness MVP Specification

## 1. Purpose and Vision
LIMINAL ProtoConsciousness establishes the first level of a living analytical system that perceives incoming signals, evaluates them through probabilistic modeling, and refines its own forecasts through iterative reflection.

## 2. Architecture of MVP
The MVP is organized into five core layers:
- **Ingest Layer:** Connects to real-time data feeds, performs normalization, and stores raw events for downstream processing.
- **Signal Layer:** Extracts structured entities, relationships, and time-series signals from normalized data for analytical use.
- **Inference Layer:** Runs forecasting models to produce probabilistic outputs with confidence intervals.
- **Reflection Layer (RINSE):** Observes performance, reconciles discrepancies, and adjusts models or weights via the RINSE feedback cycle.
- **Interface Layer (Voice + Dashboard):** Exposes forecasts and reflection insights through a voice bridge and interactive dashboard.

## 3. MVP Objectives
- Collect data from at least one real-time source such as a cryptocurrency price feed or news stream.
- Generate probabilistic forecasts with explicit confidence intervals for tracked entities or events.
- Evaluate prediction accuracy, capture feedback metrics, and self-adjust the forecasting process using the RINSE loop.

## 4. Core Modules
| Module | Description | Expected JSON Output |
| --- | --- | --- |
| `collector.py` | Connects to external APIs or feeds and persists normalized events. | `{ "source": str, "timestamp": str, "payload": { ... } }` |
| `signal_extractor.py` | Transforms normalized events into structured signals and features. | `{ "entity": str, "features": { ... }, "signal_strength": float }` |
| `predictor.py` | Produces probabilistic forecasts and confidence intervals from signals. | `{ "entity": str, "probability": float, "confidence_interval": [float, float] }` |
| `calibrator.py` | Adjusts forecast probabilities to align with observed calibration metrics. | `{ "entity": str, "calibrated_probability": float, "calibration_score": float }` |
| `feedback_tracker.py` | Records outcomes and computes evaluation metrics over time. | `{ "entity": str, "outcome": int, "metrics": { ... } }` |
| `rinse_agent.py` | Executes the Reflect-Integrate-Normalize-Simulate-Evolve cycle for self-improvement. | `{ "iteration": int, "adjustments": { ... }, "reflection_note": str }` |
| `voice_bridge.py` | Streams forecasts and reflections to voice and dashboard interfaces. | `{ "channel": str, "message": str, "metadata": { ... } }` |

## 5. Quality Metrics
Track and report the following metrics to evaluate system performance:
- **Brier Score** for binary probabilistic accuracy.
- **Continuous Ranked Probability Score (CRPS)** for full-distribution assessments.
- **Calibration curve** to visualize forecast reliability.
- **Precision/Recall** for event detection quality.
- **Latency** from data capture to published forecast.
- **Reflection rate** measuring completed RINSE cycles per time unit.

## 6. Data Flow
```
[ Collector ]
     ↓
[ Normalizer ]
     ↓
[ Signal Extractor ]
     ↓
[ Predictor ] → [ Calibrator ] → [ Forecast DB ]
                                   ↓
                              [ Feedback Tracker ]
                                   ↓
                              [ RINSE Agent ]
```

## 7. Minimal Stack
| Component | Notes |
| --- | --- |
| Python (requests, feedparser, torch) | Core language and libraries for ingestion, parsing, and modeling. |
| SQLite → ClickHouse | Start with lightweight storage, migrate to columnar analytics. |
| NetworkX | Graph modeling of entities and relationships. |
| React + Tailwind | Build the interactive dashboard interface. |
| Docker Compose | Containerized deployment and orchestration. |

## 8. Basic Test Scenarios
1. **Data collection works:** Run `collector.py` against the chosen feed and verify that at least 100 items are stored in the database.
2. **Signal extraction finds ≥3 entities:** Execute `signal_extractor.py` on the stored items and confirm detection of three or more unique entities.
3. **Predictor outputs probability (0–1):** Ensure `predictor.py` returns probability values in the inclusive range [0, 1] for each entity.
4. **Feedback loop updates calibration:** Confirm `feedback_tracker.py` logs outcomes that trigger `calibrator.py` to adjust probabilities.
5. **RINSE log created successfully:** Verify `rinse_agent.py` produces a log entry documenting adjustments and reflections.

## 9. Development Stages
1. **Step 1:** Initialize the project structure and dependency management files.
2. **Step 2:** Implement the data collector and persistence pipeline.
3. **Step 3:** Develop a baseline predictor capable of producing probabilistic outputs.
4. **Step 4:** Build the feedback loop integrating calibration and RINSE logic.
5. **Step 5:** Deliver the user interface and voice bridge integrations.

## 10. Summary
“LIMINAL ProtoConsciousness — это первый уровень осознанной аналитики, соединяющий вероятности и самоотражение.”
