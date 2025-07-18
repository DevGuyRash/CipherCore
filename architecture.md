# Unified Architecture Design (July 2025)

This document describes the **unified, persistent, self-improving crypto trading agent** architecture, as of July 2025. The system is built on:

- **LangGraph**: Orchestrates all agent logic as a stateful, multi-node graph. Each node is a function (e.g., ingestion, research, pattern discovery, validation, execution, monitoring).
- **Persistent Memory**: All agent state (patterns, trades, meta-learning, performance) is stored in a single SQLite file (`unified_memory.sqlite`) using LangGraph's `SqliteSaver`. This enables true learning and adaptation across runs.
- **OpenAI o3-pro & o3-deep-research**: o3-pro is used for emergent pattern discovery and reasoning; o3-deep-research augments with dynamic web synthesis and external context.
- **ApeX Pro SDK**: For real-time and historical market data ingestion.

## Node Flow (LangGraph)

1. **Ingestion Node**: Fetches real market data (ApeX Pro testnet) and stores it in persistent state.
2. **Research Node**: Uses o3-deep-research to synthesize external context (news, sentiment, on-chain data) and injects it into state.
3. **Autonomous Discovery Node**: Calls o3-pro to discover novel, emergent patterns from raw data and research insights. Patterns are validated and added to the persistent pattern library.
4. **Pattern Validation Node**: Scores and filters discovered patterns for integration.
5. **Intelligence Node**: Generates trading intelligence by combining validated patterns, research, and historical context.
6. **Execution Node**: Makes and simulates trade decisions based on ensemble confidence and pattern library.
7. **Monitoring Node**: Tracks performance, trade results, and triggers adaptation if needed. All results are persisted.

## Persistent Memory

- **All state is stored in `unified_memory.sqlite`** (SQLite, managed by LangGraph's `SqliteSaver`).
- The agent's memory includes:
  - `pattern_library`: All discovered and validated patterns (with confidence, uniqueness, reasoning, timeframes, etc.)
  - `trade_results`: All simulated trades and their outcomes
  - `performance_metrics`: Confidence, win rate, Sharpe ratio, etc.
  - `discovery_history`: All pattern discoveries (with timestamps)
  - `meta_learning_state`: Adaptation and learning parameters
- **Memory is loaded on startup and updated after every cycle.**
- **Inspecting memory:** Use any SQLite tool (e.g., DB Browser for SQLite, Python's `sqlite3` module) to view, backup, or export the agent's state.

## Example: Inspecting Memory

```python
import sqlite3, msgpack
conn = sqlite3.connect('unified_memory.sqlite')
row = conn.execute('SELECT checkpoint FROM checkpoints ORDER BY rowid DESC LIMIT 1').fetchone()
state = msgpack.unpackb(row[0], raw=False)
print(state['channel_values']['pattern_library'])  # See all patterns
```

## Monitoring and Usage

- **Run the agent:**
  ```bash
  python templates/langgraph_autonomous_unified.py
  ```
- **Monitor status:**
  ```bash
  python monitor_unified_system.py
  ```
- **All logs and discoveries are in `ai_debug_logs/` for audit.**

## Key Benefits
- **True persistent learning:** The agent improves over time, never forgetting patterns or trades.
- **Unified, maintainable codebase:** All logic is in `templates/langgraph_autonomous_unified.py` and `monitor_unified_system.py`.
- **Easy backup and migration:** Just copy `unified_memory.sqlite`.
- **No legacy scripts or in-memory hacks remain.**

---

For details on node logic, see `templates/langgraph_autonomous_unified.py`. For evaluation, see `reference/eval_checklist.md`. For setup, see `guides/bootstrap.ipynb` and `README.MD`.