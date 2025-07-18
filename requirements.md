# Requirements and Setup Guide

This document outlines the dependencies, installation steps, and environment configuration needed to set up the development environment for the crypto trading agent prototype. It follows Context-Engineering principles by providing atomic (basic) instructions building to molecular (few-shot) examples, ensuring minimal token overhead for LLM consumption during autonomous development.

## Python Environment

- **Version**: Python 3.12 (verified as stable for all libraries as of July 17, 2025).
- **Rationale**: Compatible with OpenAI's latest APIs (o3-pro and o3-deep-research), LangGraph, and crypto SDKs. Use a virtual environment (e.g., venv) for isolation.

**Setup Example (Few-Shot)**:
```bash
# Create and activate virtual environment
python -m venv trading-agent-env
source trading-agent-env/bin/activate  # On Unix/Mac
# Or on Windows: trading-agent-env\Scripts\activate
```

## Key Dependencies

Install via pip. These are selected for their relevance:
- `openai`: For accessing o3-pro (core pattern analysis model, launched June 10, 2025, with extended reasoning capabilities) and o3-deep-research (research tool for online gathering, introduced Feb 2, 2025, with July 17, 2025 update adding visual browser access).
- `langgraph`: For stateful graph orchestration with persistent memory.
- `hyperliquid-python-sdk`: Official SDK for Hyperliquid DEX (perp trading with leverage).
- `apexpro`: Python connector for ApeX Protocol (multi-chain perp DEX).
- `ccxt`: Unified crypto exchange library for supplemental data fetching.
- `pandas`: For data processing and analysis (e.g., handling OHLCV data).

**Installation Command**:
```bash
pip install openai langgraph hyperliquid-python-sdk apexpro ccxt pandas
```

**Version Notes (as of July 17, 2025)**:
- openai: >=1.35.0 (supports o3-pro-2025-06-10 and o3-deep-research models).
- langgraph: >=0.2.0 (includes checkpointing for neural field-like persistence).
- Others: Latest stable versions; no known conflicts.

## Environment Configuration

Create a `.env` file in the repo root for secure API keys. Use python-dotenv to load it (install if needed: `pip install python-dotenv`).

**Example .env File (Symbolic Schema from Context-Engineering `minimal_context.yaml`)**:
```
OPENAI_API_KEY=sk-your-openai-key-here  # Required for o3-pro and o3-deep-research APIs
HYPERLIQUID_API_KEY=your-hyperliquid-testnet-key  # Optional for testnet
APEX_API_KEY=your-apex-testnet-key  # Optional for testnet
```

**Loading in Code (Few-Shot Example)**:
```python
# In your main script or LangGraph nodes
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
# Usage: client = OpenAI(api_key=openai_api_key)
```

## Additional Setup for DEXes

- **Hyperliquid**: Sign up for testnet at hyperliquid.xyz/testnet. Generate API keys and set environment variables as above. SDK docs: https://github.com/hyperliquid-dex/hyperliquid-python-sdk.
- **ApeX**: Use testnet mode. API connector: https://github.com/ApeX-Protocol/apexpro-openapi. Requires wallet private key for signing (store securely, e.g., in .env).
- **Testnet Mode**: Always use testnet endpoints to simulate trades without real funds. Example in code:
  ```python
  # Hyperliquid example
  from hyperliquid import HyperLiquid
  exchange = HyperLiquid(testnet=True)
  ```

## Verification Steps

To confirm setup:
1. Run `python -c "import openai; print(openai.__version__)"` – should output >=1.35.0.
2. Test OpenAI API: Create a script `test_openai.py`:
   ```python
   from openai import OpenAI
   client = OpenAI()
   response = client.chat.completions.create(model="o3-pro-2025-06-10", messages=[{"role": "user", "content": "Test"}])
   print(response.choices[0].message.content)
   ```
3. If issues, check OpenAI status: Use o3-deep-research in the prototype for dynamic troubleshooting.

This setup enables emergent development in Cursor IDE, with persistence for iterative builds. Reference architecture.md for integration details.