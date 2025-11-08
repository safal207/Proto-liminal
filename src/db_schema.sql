-- Proto-liminal RINSE Persistent State Schema
-- Stores RINSE agent state across sessions for continuous learning

-- RINSE Agent Configuration
CREATE TABLE IF NOT EXISTS rinse_config (
    id INTEGER PRIMARY KEY CHECK (id = 1), -- Singleton
    reflection_threshold REAL NOT NULL,
    adjustment_magnitude REAL NOT NULL,
    confidence_threshold REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- RINSE Iterations
CREATE TABLE IF NOT EXISTS rinse_iterations (
    iteration INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Baseline Metrics (current baseline)
CREATE TABLE IF NOT EXISTS baseline_metrics (
    metric_name TEXT PRIMARY KEY,
    value REAL NOT NULL,
    established_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Current Metrics (latest)
CREATE TABLE IF NOT EXISTS current_metrics (
    metric_name TEXT PRIMARY KEY,
    value REAL NOT NULL,
    updated_at TEXT NOT NULL
);

-- Accumulated Feedback (time-series data)
CREATE TABLE IF NOT EXISTS accumulated_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    iteration INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (iteration) REFERENCES rinse_iterations(iteration)
);

CREATE INDEX IF NOT EXISTS idx_feedback_metric ON accumulated_feedback(metric_name);
CREATE INDEX IF NOT EXISTS idx_feedback_iteration ON accumulated_feedback(iteration);

-- Parameters Being Optimized
CREATE TABLE IF NOT EXISTS parameters (
    name TEXT PRIMARY KEY,
    value REAL NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by_iteration INTEGER,
    FOREIGN KEY (updated_by_iteration) REFERENCES rinse_iterations(iteration)
);

-- Parameter History (track evolution)
CREATE TABLE IF NOT EXISTS parameter_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parameter_name TEXT NOT NULL,
    old_value REAL NOT NULL,
    new_value REAL NOT NULL,
    iteration INTEGER NOT NULL,
    reason TEXT,
    confidence REAL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (iteration) REFERENCES rinse_iterations(iteration),
    FOREIGN KEY (parameter_name) REFERENCES parameters(name)
);

CREATE INDEX IF NOT EXISTS idx_param_history_name ON parameter_history(parameter_name);
CREATE INDEX IF NOT EXISTS idx_param_history_iteration ON parameter_history(iteration);

-- RINSE Cycles (complete cycle records)
CREATE TABLE IF NOT EXISTS rinse_cycles (
    iteration INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    reflection_note TEXT,
    simulation_score REAL,
    evolution_applied INTEGER NOT NULL, -- boolean 0/1
    metadata TEXT, -- JSON
    FOREIGN KEY (iteration) REFERENCES rinse_iterations(iteration)
);

-- Reflections (observations and insights)
CREATE TABLE IF NOT EXISTS reflections (
    iteration INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    observations TEXT NOT NULL, -- JSON dict
    insights TEXT NOT NULL, -- JSON array
    reflection_note TEXT,
    metadata TEXT, -- JSON
    FOREIGN KEY (iteration) REFERENCES rinse_iterations(iteration)
);

-- Adjustments (proposed changes)
CREATE TABLE IF NOT EXISTS adjustments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration INTEGER NOT NULL,
    target TEXT NOT NULL,
    parameter TEXT NOT NULL,
    old_value REAL NOT NULL,
    new_value REAL NOT NULL,
    reason TEXT NOT NULL,
    confidence REAL NOT NULL,
    applied INTEGER NOT NULL, -- boolean 0/1
    metadata TEXT, -- JSON
    timestamp TEXT NOT NULL,
    FOREIGN KEY (iteration) REFERENCES rinse_iterations(iteration)
);

CREATE INDEX IF NOT EXISTS idx_adjustments_iteration ON adjustments(iteration);
CREATE INDEX IF NOT EXISTS idx_adjustments_parameter ON adjustments(parameter);

-- Statistics
CREATE TABLE IF NOT EXISTS stats (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

-- Insert default stats
INSERT OR IGNORE INTO stats (key, value, updated_at) VALUES
    ('cycles_completed', 0, datetime('now')),
    ('adjustments_made', 0, datetime('now')),
    ('adjustments_applied', 0, datetime('now'));

-- Views for convenience

-- Latest parameter values with history
CREATE VIEW IF NOT EXISTS parameter_evolution AS
SELECT
    ph.parameter_name,
    ph.iteration,
    ph.old_value,
    ph.new_value,
    ph.new_value - ph.old_value AS delta,
    ph.reason,
    ph.confidence,
    ph.timestamp
FROM parameter_history ph
ORDER BY ph.parameter_name, ph.iteration DESC;

-- Reflection insights over time
CREATE VIEW IF NOT EXISTS reflection_timeline AS
SELECT
    r.iteration,
    r.timestamp,
    r.reflection_note,
    json_extract(r.insights, '$') AS insights,
    json_extract(r.observations, '$') AS observations
FROM reflections r
ORDER BY r.iteration DESC;

-- Performance trends
CREATE VIEW IF NOT EXISTS metric_trends AS
SELECT
    af.metric_name,
    af.iteration,
    af.value,
    af.timestamp,
    AVG(af.value) OVER (
        PARTITION BY af.metric_name
        ORDER BY af.iteration
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS moving_avg_5
FROM accumulated_feedback af
ORDER BY af.metric_name, af.iteration;
