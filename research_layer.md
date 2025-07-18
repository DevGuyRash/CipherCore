# Research Layer Integration

This document provides a deep dive into the research layer of the crypto trading agent prototype, focusing on the integration of OpenAI's o3-deep-research model as a dedicated tool for online resource gathering and synthesis. It builds on Context-Engineering's "organs" level (multi-step control flows and system orchestration) and "field theory" (coordinating multiple fields with emergence protocols), enabling dynamic, resonant augmentation of the core pattern analysis performed by o3-pro.

As of July 17, 2025 (verified via up-to-date research):
- **o3-deep-research**: Part of OpenAI's Deep Research API (introduced February 2, 2025, with a July 17, 2025 update adding visual browser access). This model is powered by the o3 architecture, designed for automated, detailed analysis of knowledge sources. It excels in reasoning, planning, and synthesizing across real-world information (e.g., web searches, news aggregation, sentiment extraction). API supports models like o3-deep-research for in-depth synthesis and faster variants like o4-mini-deep-research. It's ideal for chaining research tasks without predefined queries, fostering emergent insights.
- **Integration Rationale**: While o3-pro (launched June 10, 2025, model: o3-pro-2025-06-10) handles core pattern extrapolation with extended compute for deliberate reasoning (top performer on AIME 2024/2025 benchmarks), o3-deep-research augments it by dynamically gathering external data (e.g., latest market sentiment or on-chain events), reducing hallucinations and enabling adaptive, context-aware decisions.

The research layer operates as a LangGraph node, injecting synthesized insights into the intelligence layer for resonant pattern discovery (e.g., via attractor dynamics: building on prior queries for deeper exploration).

## Key Functions of the Research Layer

- **Online Resource Gathering**: Use o3-deep-research's visual browser and synthesis capabilities to fetch and process real-time data (e.g., "Synthesize BTC sentiment from recent Twitter threads and CoinDesk articles").
- **Multi-Step Flows**: Chain queries for progressive depth (e.g., start broad, then drill down based on initial outputs).
- **Emergent Augmentation**: Outputs feed into o3-pro for novel pattern hypothesis (e.g., correlating web-sourced news spikes with price anomalies).
- **Symbolic Mechanisms**: Structure queries and responses with JSON schemas to trigger LLM-internal symbolic abstraction/induction/retrieval heads (per ICML Princeton research in Context-Engineering), enhancing reasoning over abstract variables.

## Integration in LangGraph

The research layer is a dedicated node in the LangGraph graph (see architecture.md). It shares state with the intelligence node for persistent context (e.g., using LangGraph's checkpointing to resonate prior research findings).

### API Setup and Calls

- **Dependencies**: `openai` library (version >=1.35.0 for Deep Research API support).
- **API Call Example**: Use the Deep Research endpoint for structured synthesis.
  ```python
  from openai import OpenAI
  import os

  client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

  def deep_research_query(query, model="o3-deep-research"):
      response = client.chat.completions.create(
          model=model,
          messages=[
              {"role": "system", "content": "You are a deep research assistant. Synthesize information from web sources using your visual browser."},
              {"role": "user", "content": query}
          ],
          # Optional: Use tools for browser access if enabled in API
          tools=[{"type": "browser"}]  # As per July 17, 2025 update
      )
      return response.choices[0].message.content
  ```

### Node Implementation

**Research Node Stub** (Adapt from templates/langgraph_template.py):
```python
def research_node(state):
    # Retrieve raw data from shared state
    raw_data = state.get('raw_data', {})
    
    # Formulate dynamic query based on state (for resonance)
    query = f"Synthesize latest market sentiment and external factors for {raw_data.get('asset', 'BTC-USD')} " \
            f"from web sources (news, social media, on-chain data). Focus on emergent correlations without predefined indicators. " \
            f"Output in JSON for symbolic processing."
    
    # Call o3-deep-research
    research_output = deep_research_query(query)
    
    # Parse and persist (symbolic residue)
    import json
    try:
        synthesized_data = json.loads(research_output)
    except json.JSONDecodeError:
        synthesized_data = {"error": "Parsing failed", "raw": research_output}
    
    state['research_insights'] = synthesized_data  # Persist for intelligence node
    return state
```

### Prompt Engineering for Emergence

Use open-ended, field-orchestrated prompts to enable quantum semantics (superposition of meanings) and attractor formation:
- **Base Prompt Template** (Inspired by Context-Engineering `cognitive-templates/reasoning.md`):
  "Analyze and synthesize from online sources: [query details]. Explore multiple angles (e.g., sentiment, events, correlations). Hypothesize novel insights with reasoning chains. Output as JSON: {'summary': str, 'key_findings': list, 'emergent_hypotheses': list, 'sources': list}."
- **Chaining for Depth**: If initial output lists follow-up URLs, feed them back via state for recursive emergence (e.g., "Drill into [URL] for deeper analysis").

### JSON Schema for Outputs (Adapted from Context-Engineering `symbolicResidue.v1.json`)

```json
{
  "type": "object",
  "properties": {
    "summary": {"type": "string", "description": "Concise synthesis of gathered resources"},
    "key_findings": {"type": "array", "items": {"type": "string"}, "description": "Bullet-point facts or correlations"},
    "emergent_hypotheses": {"type": "array", "items": {"type": "string"}, "description": "Novel pattern ideas for o3-pro to extrapolate"},
    "sources": {"type": "array", "items": {"type": "string"}, "description": "List of URLs or references used"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Overall synthesis reliability"}
  },
  "required": ["summary", "key_findings", "emergent_hypotheses", "sources"]
}
```

## Enhancements and Mitigations

- **Resonance & Attractor Dynamics**: Use state to build "attractors" (e.g., recurrent themes from past researches) for emergent protocol integration (per `11_emergence_and_attractor_dynamics.md`).
- **Token Budget Optimization**: Limit queries to focused instructions; prune outputs via symbolic pruning.
- **Error Handling & Self-Repair**: If API fails, fallback to cached state and trigger meta-recursion in monitoring node.
- **Testing**: In guides/bootstrap.ipynb, simulate with mock queries (e.g., "Gather test sentiment data").

This layer enables the prototype to dynamically incorporate real-world context, augmenting o3-pro's pattern analysis for truly intelligent, adaptive trading. Reference guides/bootstrap.ipynb for hands-on examples.