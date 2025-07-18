# Evaluation Checklist and Metrics (Unified Persistent Architecture)

This document is the reference for evaluating the unified crypto trading agent prototype (July 2025+). The system is now a single LangGraph graph with persistent memory in `unified_memory.sqlite` (SQLite, managed by LangGraph's SqliteSaver). All state (patterns, trades, meta-learning) is persistent and evolves over time.

## Core Evaluation Principles

- **Persistent State**: All agent state (pattern library, trade results, meta-learning, performance) must be stored in `unified_memory.sqlite` and survive restarts.
- **Emergence Detection**: Track novel patterns via signatures (e.g., unexpected correlations with >0.8 confidence).
- **Risk-Aware Metrics**: Backtest on historical data to simulate leverage trading risks.
- **Symbolic Residue**: Evaluate JSON outputs for parseability and reasoning depth.
- **Self-Improvement Loop**: Monitoring node must trigger adaptations if metrics fall below target.

## Checklist for Unified Prototype

### 1. Data Ingestion
- [ ] Fetches real market data (ApeX Pro testnet) and persists in state.
- [ ] Handles real-time and historical data (OHLCV, volume, etc.).
- [ ] Data is present in `unified_memory.sqlite` after each run.

### 2. Research Layer (o3-deep-research)
- [ ] Synthesizes external data (sentiment, news, on-chain) and persists in state.
- [ ] Outputs valid JSON per schema.
- [ ] Research insights are present in persistent state.

### 3. Intelligence & Pattern Discovery (o3-pro)
- [ ] Discovers emergent patterns from raw data and research insights.
- [ ] Validated patterns are added to `pattern_library` in persistent memory.
- [ ] Reasoning, confidence, and uniqueness are stored for each pattern.

### 4. Decision & Execution
- [ ] Generates valid trade signals and simulates execution.
- [ ] Trade results are appended to `trade_results` in persistent memory.
- [ ] Risk controls and position sizing are enforced.

### 5. Monitoring & Adaptation
- [ ] Computes performance metrics (confidence, win rate, Sharpe ratio, etc.).
- [ ] Triggers adaptation if metrics fall below target.
- [ ] All metrics and adaptations are persisted.

### 6. Persistence & Recovery
- [ ] All state is stored in `unified_memory.sqlite` (checkpoints table).
- [ ] After restart, agent resumes with full memory (patterns, trades, metrics).
- [ ] No data loss or reset between runs.

## Key Metrics (Persistent)

| Metric | Description | Target |
|--------|-------------|--------|
| **Pattern Library Size** | Number of unique patterns in persistent memory | >5 after 10 runs |
| **Trade Results** | Number of trades in persistent memory | >10 after 10 runs |
| **Confidence Score** | Average confidence of patterns | >0.8 |
| **Win Rate** | % of simulated trades that profit | >60% |
| **Sharpe Ratio** | Risk-adjusted return | >1.5 |
| **Latency** | End-to-end cycle time | <60s |
| **Persistence** | State survives restart | 100% |

## Inspecting Persistent Memory

To verify persistence and inspect state:

```python
import sqlite3, msgpack
conn = sqlite3.connect('unified_memory.sqlite')
row = conn.execute('SELECT checkpoint FROM checkpoints ORDER BY rowid DESC LIMIT 1').fetchone()
state = msgpack.unpackb(row[0], raw=False)
print(state['channel_values']['pattern_library'])  # See all patterns
print(state['channel_values']['trade_results'])    # See all trades
```

## Self-Improvement Protocol

1. Run the unified agent for several cycles.
2. Inspect `unified_memory.sqlite` to verify patterns, trades, and metrics are accumulating.
3. If any metric is below target, trigger adaptation via the monitoring node.
4. Repeat until all targets are met.

---

This checklist ensures the prototype is robust, persistent, and self-optimizing. For setup, see `guides/bootstrap.ipynb`. For architecture, see `architecture.md`. For usage, see `README.MD`.