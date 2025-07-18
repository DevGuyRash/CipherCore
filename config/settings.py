"""Central configuration for the CipherCore trading prototype.

This module centralizes default constants such as model names,
validation thresholds, and persistence paths. Future iterations
may extend this with environment variable parsing and dynamic
configuration handling.
"""

DEFAULT_ASSET = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"

OPENAI_MODEL_DISCOVERY = "o3-mini"
OPENAI_MODEL_RESEARCH = "o3-deep-research"
OPENAI_MODEL_REASONING = "o3-pro-2025-06-10"

MEMORY_DB_PATH = "unified_memory.sqlite"

DISCOVERY_CONFIDENCE_MIN = 0.3
DISCOVERY_UNIQUENESS_MIN = 0.3
VALIDATION_SCORE_MIN = 0.4

MAX_TRADE_RESULTS_HISTORY = 500
