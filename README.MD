# Crypto Trading Agent Prototype

"Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — Andrej Karpathy

This repository serves as the foundation for autonomously developing an end-to-end (e2e) functional prototype of an intelligent crypto leverage trading automation system. The system targets DEXes like Hyperliquid and ApeX, leveraging state-of-the-art LLMs for both intellectual labor (e.g., emergent pattern extrapolation) and execution. 

Key components:
- **Core LLM for Pattern Analysis**: OpenAI's o3-pro (launched June 10, 2025), a reasoning-focused model that uses extended compute for deliberate, high-quality analysis. It excels in benchmarks like AIME 2024/2025 and is ideal for discovering novel patterns in crypto data without relying on predefined indicators.
- **Research Layer**: Integrates OpenAI's o3-deep-research (part of the Deep Research API, introduced February 2, 2025, with recent updates including visual browser access as of July 17, 2025). This acts as a tool for online resource gathering, enabling reasoning, planning, and synthesis across real-world information (e.g., fetching news, sentiment, or on-chain data dynamically).
- **Orchestration Framework**: LangGraph for stateful, graph-based workflows with persistent memory and control flow.
- **Goal**: Enable the LLM (via Cursor IDE's agentic capabilities) to autonomously build and iterate on the prototype, fostering emergent behaviors through context engineering principles from https://github.com/davidkimai/Context-Engineering.

The prototype will operate in a closed-loop: Data ingestion → Pattern analysis (o3-pro) → Research augmentation (o3-deep-research) → Decision/Execution → Monitoring/Adaptation.

## Architecture Overview

Below is a Mermaid graph visualizing the high-level architecture, inspired by the biological metaphor and neural field theory from Context-Engineering (progressing from atoms to emergent fields).

```mermaid
graph TD
    classDef basic fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b
    classDef intermediate fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#2e7d32
    classDef advanced fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
    classDef meta fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#6a1b9a
    
    subgraph Ingestion["Data Ingestion Layer"]
        A[Raw Crypto Data<br>(Prices, Volume, News via APIs)]
    end
    
    subgraph Intelligence["LLM Intelligence Layer"]
        B[o3-pro Pattern Analysis<br>(Emergent Extrapolation)]
        C[o3-deep-research<br>(Online Resource Gathering)]
    end
    
    subgraph Execution["Decision & Execution Layer"]
        D[Trade Signals<br>(Long/Short with Leverage)]
        E[DEX SDK Execution<br>(Hyperliquid/ApeX)]
    end
    
    subgraph Monitoring["Monitoring & Adaptation Layer"]
        F[Feedback Loop<br>(Backtesting, Self-Repair)]
    end
    
    %% Connections
    A --> B
    A --> C
    B --> D
    C --> D
    D --> E
    E --> F
    F --> B
    F --> C
    
    %% Real-world parallels
    A1["Like: Few-shot data feeds"] -.-> A
    B1["Like: Symbolic abstraction for patterns"] -.-> B
    C1["Like: Retrieval-augmented research"] -.-> C
    D1["Like: Multi-agent decision flows"] -.-> D
    E1["Like: Control flow orchestration"] -.-> E
    F1["Like: Meta-recursive improvement"] -.-> F
    
    %% Apply classes
    class A,A1 basic
    class B,C,B1,C1 intermediate
    class D,E,D1,E1 advanced
    class F,F1 meta
```

## Cognitive Tool Integration

To scaffold reasoning during autonomous development, incorporate these modular cognitive tools (inspired by IBM Zurich's templates from Context-Engineering):

1. **[Understand Task]**: Build an autonomous trading agent using o3-pro for pattern analysis (e.g., "Analyze raw BTC data for emergent correlations without indicators; hypothesize trends."), o3-deep-research for research (e.g., "Gather and synthesize latest market sentiment from web sources."), and LangGraph for orchestration. Ensure emergent patterns via neural field dynamics (persistence, resonance, attractors).

2. **[Recall Related]**: Reference architecture.md for detailed layers, research_layer.md for o3-deep-research integration, and templates/langgraph_template.py as code base.

3. **[Reason Step-by-Step]**: Implement nodes iteratively in LangGraph (e.g., ingestion node fetches data, intelligence node calls o3-pro/o3-deep-research APIs).

4. **[Verify & Backtrack]**: Test in guides/bootstrap.ipynb; simulate trades on testnet; evaluate with metrics like pattern confidence >0.8.

5. **[Compose & Emerge]**: Enable self-repair and meta-recursion for iterative improvement.

## Dependencies

- Python 3.12
- Key libraries: `openai` (for o3-pro and o3-deep-research APIs), `langgraph`, `hyperliquid-python-sdk` (Hyperliquid), `apexpro` (ApeX), `ccxt` (supplemental crypto data), `pandas` (data handling).
- Environment setup: Create `.env` with `OPENAI_API_KEY` for API access.
- Install via: `pip install openai langgraph hyperliquid-python-sdk apexpro ccxt pandas`

Note: Use testnet endpoints for all DEX interactions to avoid real fund risks.

## Bootstrap Protocol

To initiate autonomous development in Cursor IDE:
1. Clone this repo.
2. Open in Cursor.
3. Feed the kick-off prompt (from the overall plan) into Cursor's chat/composer mode.
4. The LLM will edit/create files, run tests, and commit progress.

For manual exploration, start with `guides/bootstrap.ipynb` for a minimal working example.

## License

MIT License

## Acknowledgements

- Inspired by Context-Engineering repository (as of July 2025 updates).
- OpenAI for o3-pro and o3-deep-research models.