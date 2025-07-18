#!/usr/bin/env python3
"""
Unified LangGraph Autonomous Trading System
Integrates autonomous pattern discovery with LangGraph state management
"""

import os
import json
import time
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import sqlite3

from config.settings import (
    DEFAULT_ASSET,
    DEFAULT_TIMEFRAME,
    OPENAI_MODEL_DISCOVERY,
    OPENAI_MODEL_RESEARCH,
    OPENAI_MODEL_REASONING,
    MEMORY_DB_PATH,
    DISCOVERY_CONFIDENCE_MIN,
    DISCOVERY_UNIQUENESS_MIN,
    VALIDATION_SCORE_MIN,
    MAX_TRADE_RESULTS_HISTORY,
)

# Load environment
load_dotenv()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    print(f"Warning: OpenAI client initialization failed: {e}")
    client = None

# Try to import ApeX adapter
try:
    from apex_adapter_official import ApeXOfficialAdapter, ApeXConfig
    APEX_AVAILABLE = True
    print("✅ ApeX Official SDK adapter available")
except ImportError as e:
    APEX_AVAILABLE = False
    print(f"⚠️ ApeX adapter not available: {e}")

# Enhanced TradingState schema with autonomous discovery fields
class UnifiedTradingState(TypedDict):
    # Original trading fields
    asset: str
    timeframe: str
    raw_data: Dict[str, Any]
    market_summary: Dict[str, Any]
    research_insights: Dict[str, Any]
    patterns: Dict[str, Any]
    trade_signal: Dict[str, Any]
    execution_result: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    adaptations: List[str]
    needs_adaptation: bool
    errors: List[Dict[str, Any]]
    data_timestamp: str
    research_timestamp: str
    intelligence_timestamp: str
    execution_timestamp: str
    monitoring_timestamp: str
    start_time: str
    iteration_count: int
    
    # Autonomous discovery fields
    autonomous_mode_enabled: bool
    discovery_schedule: Dict[str, Any]
    pattern_library: Dict[str, Any]  # pattern_id -> pattern_data
    discovery_history: List[Dict[str, Any]]
    exploration_sessions: List[Dict[str, Any]]
    meta_learning_state: Dict[str, Any]
    autonomous_performance: Dict[str, Any]
    
    # Enhanced memory fields (migrated from ENHANCED_MEMORY)
    patterns_history: List[Dict[str, Any]]
    trade_results: List[Dict[str, Any]]
    emergence_signals: List[Dict[str, Any]]
    pattern_performance_map: Dict[str, List[float]]
    market_conditions_history: List[Dict[str, Any]]
    parameter_adjustments: List[Dict[str, Any]]
    ensemble_weights: Dict[str, float]
    risk_adjustment_history: List[Dict[str, Any]]
    
    # Discovery session tracking
    last_discovery_session: Optional[str]
    patterns_discovered_count: int
    patterns_integrated_count: int
    discovery_session_active: bool

# Discovery Pattern Data Structure
class DiscoveredPattern:
    def __init__(self, pattern_id: str, description: str, confidence_score: float,
                 uniqueness_score: float, reasoning: str, timeframes: List[str],
                 market_conditions: Dict[str, Any], discovered_at: str):
        self.pattern_id = pattern_id
        self.description = description
        self.confidence_score = confidence_score
        self.uniqueness_score = uniqueness_score
        self.reasoning = reasoning
        self.timeframes = timeframes
        self.market_conditions = market_conditions
        self.discovered_at = discovered_at
        self.validation_score = 0.0
        self.integration_score = 0.0
        self.trading_relevance = 0.0

    def to_dict(self):
        return {
            'pattern_id': self.pattern_id,
            'description': self.description,
            'confidence_score': self.confidence_score,
            'uniqueness_score': self.uniqueness_score,
            'reasoning': self.reasoning,
            'timeframes': self.timeframes,
            'market_conditions': self.market_conditions,
            'discovered_at': self.discovered_at,
            'validation_score': self.validation_score,
            'integration_score': self.integration_score,
            'trading_relevance': self.trading_relevance
        }

# Utility function to ensure serialization compatibility
def make_serializable(obj):
    """Convert numpy types and other non-serializable types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    else:
        return obj

# Error handling decorator
def safe_execute(func):
    """Decorator for safe node execution with error recovery and serialization safety."""
    def wrapper(state):
        try:
            result = func(state)
            # Ensure all values in state are serialization-safe
            return make_serializable(result)
        except Exception as e:
            error_info = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat(),
                'node': func.__name__
            }
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(error_info)
            print(f"Error in {func.__name__}: {e}")
            return make_serializable(state)
    return wrapper

@safe_execute
def ingestion_node(state: UnifiedTradingState) -> UnifiedTradingState:
    """Enhanced Data Ingestion with multi-timeframe support for autonomous discovery"""
    print("🔄 Data Ingestion Node: Fetching market data...")
    
    asset = state.get('asset', DEFAULT_ASSET)
    timeframe = state.get('timeframe', DEFAULT_TIMEFRAME)
    
    try:
        if APEX_AVAILABLE:
            config = ApeXConfig(api_key="", testnet=True)
            adapter = ApeXOfficialAdapter(config)
            
            # Convert asset format for ApeX (BTC/USDT -> BTCUSDT)
            apex_symbol = asset.replace('/', '')
            
            # Get comprehensive market data
            market_data = adapter.get_comprehensive_market_data(apex_symbol)
            
            state['raw_data'] = market_data
            state['market_summary'] = market_data.get('market_summary', {})
            state['data_timestamp'] = datetime.now().isoformat()
            
            print(f"✅ Real market data fetched for {asset}")
            
        else:
            # Fallback to synthetic data - convert numpy types to native Python types
            print("⚠️ Using synthetic market data")
            synthetic_price = float(45000 + np.random.normal(0, 1000))
            synthetic_volume = float(1000000 + np.random.normal(0, 100000))
            synthetic_change = float(np.random.uniform(-0.05, 0.05))
            
            state['raw_data'] = {
                'price': synthetic_price,
                'volume': synthetic_volume,
                'timestamp': datetime.now().isoformat()
            }
            state['market_summary'] = {
                'current_price': synthetic_price,
                'volume_24h': synthetic_volume,
                'price_change_24h': synthetic_change
            }
            state['data_timestamp'] = datetime.now().isoformat()
    
    except Exception as e:
        print(f"❌ Data ingestion failed: {e}")
        # Set empty data to continue
        state['raw_data'] = {}
        state['market_summary'] = {}
        state['data_timestamp'] = datetime.now().isoformat()
    
    return state

@safe_execute
def research_node(state: UnifiedTradingState) -> UnifiedTradingState:
    """Research and Analysis Node"""
    print("🔍 Research Node: Analyzing market conditions...")
    
    market_summary = state.get('market_summary', {})
    
    # Perform market analysis
    research_insights = {
        'market_sentiment': 'neutral',
        'volatility_assessment': 'medium',
        'trend_analysis': 'sideways',
        'risk_factors': ['market_uncertainty'],
        'opportunities': ['potential_breakout'],
        'confidence': 0.6,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add current price analysis if available
    if 'current_price' in market_summary:
        current_price = market_summary['current_price']
        research_insights['price_analysis'] = f"Current price: ${current_price:,.2f}"
    
    state['research_insights'] = research_insights
    state['research_timestamp'] = datetime.now().isoformat()
    
    print(f"✅ Research completed - Sentiment: {research_insights['market_sentiment']}")
    return state

@safe_execute
def autonomous_discovery_node(state: UnifiedTradingState) -> UnifiedTradingState:
    """Autonomous Pattern Discovery Node - Core AI discovery using o3-mini"""
    print("🤖 Autonomous Discovery Node: Discovering patterns...")
    
    # Check if autonomous mode is enabled
    if not state.get('autonomous_mode_enabled', False):
        print("ℹ️ Autonomous mode disabled - skipping discovery")
        return state
    
    # Check if discovery session is already active to avoid loops
    if state.get('discovery_session_active', False):
        print("ℹ️ Discovery session already active - skipping")
        return state
    
    # Mark discovery session as active
    state['discovery_session_active'] = True
    
    try:
        if not client:
            print("⚠️ OpenAI client not available")
            state['discovery_session_active'] = False
            return state
        
        # Prepare market data for AI analysis
        raw_data = state.get('raw_data', {})
        market_summary = state.get('market_summary', {})
        research_insights = state.get('research_insights', {})
        
        # Create AI prompt for pattern discovery
        discovery_prompt = f"""
AUTONOMOUS PATTERN DISCOVERY SESSION
TIMESTAMP: {datetime.now().isoformat()}

MARKET DATA ANALYSIS:
- Asset: {state.get('asset', DEFAULT_ASSET)}
- Current Price: {market_summary.get('current_price', 'N/A')}
- Volume 24h: {market_summary.get('volume_24h', 'N/A')}
- Price Change 24h: {market_summary.get('price_change_24h', 'N/A')}
- Market Sentiment: {research_insights.get('market_sentiment', 'neutral')}
- Volatility: {research_insights.get('volatility_assessment', 'medium')}

DISCOVERY OBJECTIVES:
1. Identify emergent patterns in price action, volume, and market behavior
2. Look for novel correlations and relationships not captured by traditional indicators
3. Focus on patterns that could provide trading edge or risk insights
4. Consider multi-timeframe implications and market structure changes

PATTERN DISCOVERY INSTRUCTIONS:
Analyze the provided market data and discover 1-3 novel patterns. For each pattern discovered, provide:

PATTERN_ID: unique_identifier
DESCRIPTION: detailed description of the pattern (minimum 50 characters)
REASONING: comprehensive explanation of why this pattern is significant (minimum 100 characters)
CONFIDENCE: confidence score between 0.0 and 1.0
UNIQUENESS: uniqueness score between 0.0 and 1.0 (how novel is this pattern)
TIMEFRAMES: applicable timeframes as comma-separated list
MARKET_CONDITIONS: relevant market conditions for this pattern

Focus on raw intelligence and emergent behavior rather than traditional technical analysis.
"""

        # Call AI for pattern discovery
        response = client.chat.completions.create(
            model=OPENAI_MODEL_DISCOVERY,
            messages=[
                {"role": "system", "content": "You are an expert autonomous pattern discovery AI specialized in identifying novel market patterns and emergent behaviors in cryptocurrency markets."},
                {"role": "user", "content": discovery_prompt}
            ],
            max_completion_tokens=1500
        )
        
        ai_response = response.choices[0].message.content
        
        # Parse AI response for patterns
        discovered_patterns = parse_discovery_response(ai_response)
        
        # Update state with discovered patterns
        current_patterns = state.get('pattern_library', {})
        patterns_added = 0
        
        for pattern in discovered_patterns:
            if (
                pattern.confidence_score >= DISCOVERY_CONFIDENCE_MIN
                and pattern.uniqueness_score >= DISCOVERY_UNIQUENESS_MIN
            ):
                current_patterns[pattern.pattern_id] = pattern.to_dict()
                patterns_added += 1
                
                # Add to discovery history
                if 'discovery_history' not in state:
                    state['discovery_history'] = []
                state['discovery_history'].append({
                    'pattern_id': pattern.pattern_id,
                    'discovered_at': pattern.discovered_at,
                    'confidence': pattern.confidence_score,
                    'uniqueness': pattern.uniqueness_score
                })
        
        state['pattern_library'] = current_patterns
        state['patterns_discovered_count'] = state.get('patterns_discovered_count', 0) + patterns_added
        state['last_discovery_session'] = datetime.now().isoformat()
        
        print(f"✅ Discovery completed - {patterns_added} patterns discovered")
        
        # Log for debugging
        debug_dir = "ai_debug_logs"
        os.makedirs(debug_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(f"{debug_dir}/unified_discovery_{ts}.txt", "w", encoding='utf-8') as f:
            f.write(f"DISCOVERY SESSION: {ts}\n")
            f.write(f"AI RESPONSE:\n{ai_response}\n")
            f.write(f"PATTERNS PARSED: {len(discovered_patterns)}\n")
            f.write(f"PATTERNS ADDED: {patterns_added}\n")
    
    except Exception as e:
        print(f"❌ Autonomous discovery failed: {e}")
        # Continue without discovery
    
    finally:
        state['discovery_session_active'] = False
    
    return state

def parse_discovery_response(ai_response: str) -> List[DiscoveredPattern]:
    """Parse AI response to extract discovered patterns with robust parsing"""
    patterns = []
    lines = ai_response.split('\n')
    current_pattern = {}
    
    def match_field(line, field):
        line = line.strip()
        return line.startswith(f'{field}:') or line.startswith(f'**{field}**:')
    
    def extract_value(line, field):
        line = line.strip()
        if line.startswith(f'{field}:'):
            return line.split(f'{field}:', 1)[1].strip()
        elif line.startswith(f'**{field}**:'):
            return line.split(f'**{field}**:', 1)[1].strip()
        return ""
    
    def extract_number(value_str):
        """Extract number from strings like '0.75' or 'seventy-five (0.75)'"""
        import re
        # Look for decimal numbers
        matches = re.findall(r'\d+\.?\d*', value_str)
        if matches:
            try:
                return float(matches[-1])  # Take the last number found
            except:
                pass
        return 0.0
    
    for line in lines:
        line = line.strip()
        
        if match_field(line, 'PATTERN_ID'):
            # Save previous pattern if exists
            if current_pattern and all(k in current_pattern for k in ['pattern_id', 'description', 'confidence', 'uniqueness']):
                pattern = DiscoveredPattern(
                    pattern_id=current_pattern['pattern_id'],
                    description=current_pattern.get('description', ''),
                    confidence_score=current_pattern.get('confidence', 0.0),
                    uniqueness_score=current_pattern.get('uniqueness', 0.0),
                    reasoning=current_pattern.get('reasoning', ''),
                    timeframes=current_pattern.get('timeframes', [DEFAULT_TIMEFRAME]),
                    market_conditions=current_pattern.get('market_conditions', {}),
                    discovered_at=datetime.now().isoformat()
                )
                patterns.append(pattern)
            
            # Start new pattern
            current_pattern = {'pattern_id': extract_value(line, 'PATTERN_ID')}
            
        elif match_field(line, 'DESCRIPTION'):
            current_pattern['description'] = extract_value(line, 'DESCRIPTION')
            
        elif match_field(line, 'REASONING'):
            current_pattern['reasoning'] = extract_value(line, 'REASONING')
            
        elif match_field(line, 'CONFIDENCE'):
            conf_str = extract_value(line, 'CONFIDENCE')
            current_pattern['confidence'] = extract_number(conf_str)
            
        elif match_field(line, 'UNIQUENESS'):
            uniq_str = extract_value(line, 'UNIQUENESS')
            current_pattern['uniqueness'] = extract_number(uniq_str)
            
        elif match_field(line, 'TIMEFRAMES'):
            tf_str = extract_value(line, 'TIMEFRAMES')
            current_pattern['timeframes'] = [tf.strip() for tf in tf_str.split(',')]
            
        elif match_field(line, 'MARKET_CONDITIONS'):
            mc_str = extract_value(line, 'MARKET_CONDITIONS')
            current_pattern['market_conditions'] = {'description': mc_str}
    
    # Don't forget the last pattern
    if current_pattern and all(k in current_pattern for k in ['pattern_id', 'description', 'confidence', 'uniqueness']):
        pattern = DiscoveredPattern(
            pattern_id=current_pattern['pattern_id'],
            description=current_pattern.get('description', ''),
            confidence_score=current_pattern.get('confidence', 0.0),
            uniqueness_score=current_pattern.get('uniqueness', 0.0),
            reasoning=current_pattern.get('reasoning', ''),
            timeframes=current_pattern.get('timeframes', [DEFAULT_TIMEFRAME]),
            market_conditions=current_pattern.get('market_conditions', {}),
            discovered_at=datetime.now().isoformat()
        )
        patterns.append(pattern)
    
    return patterns

@safe_execute
def pattern_validation_node(state: UnifiedTradingState) -> UnifiedTradingState:
    """Validate discovered patterns for trading integration"""
    print("🔍 Pattern Validation Node: Validating patterns...")
    
    pattern_library = state.get('pattern_library', {})
    if not pattern_library:
        print("ℹ️ No patterns to validate")
        return state
    
    validated_count = 0
    
    for pattern_id, pattern_data in pattern_library.items():
        # Simple validation logic
        confidence = pattern_data.get('confidence_score', 0.0)
        uniqueness = pattern_data.get('uniqueness_score', 0.0)
        description_len = len(pattern_data.get('description', ''))
        reasoning_len = len(pattern_data.get('reasoning', ''))
        
        # Calculate validation score
        validation_score = (
            confidence * 0.4 +
            uniqueness * 0.3 +
            (1.0 if description_len >= 50 else 0.5) * 0.15 +
            (1.0 if reasoning_len >= 100 else 0.5) * 0.15
        )
        
        pattern_data['validation_score'] = validation_score
        
        if validation_score >= VALIDATION_SCORE_MIN:
            validated_count += 1
            print(f"✅ Pattern validated: {pattern_id} (score: {validation_score:.3f})")
        else:
            print(f"⚠️ Pattern validation low: {pattern_id} (score: {validation_score:.3f})")
    
    state['pattern_library'] = pattern_library
    print(f"✅ Validation completed - {validated_count} patterns validated")
    
    return state

@safe_execute
def intelligence_node(state: UnifiedTradingState) -> UnifiedTradingState:
    """Enhanced Intelligence Node with autonomous pattern integration"""
    print("🧠 Intelligence Node: Generating trading intelligence...")
    
    research_insights = state.get('research_insights', {})
    pattern_library = state.get('pattern_library', {})
    market_summary = state.get('market_summary', {})
    
    # Base pattern analysis
    base_confidence = research_insights.get('confidence', 0.6)
    
    # Integrate autonomous patterns
    pattern_boost = 0.0
    relevant_patterns = []
    
    for pattern_id, pattern_data in pattern_library.items():
        validation_score = pattern_data.get('validation_score', 0.0)
        if validation_score >= VALIDATION_SCORE_MIN:
            relevant_patterns.append(pattern_data)
            pattern_boost += validation_score * 0.1  # Small boost per validated pattern
    
    # Calculate ensemble confidence
    ensemble_confidence = min(base_confidence + pattern_boost, 0.95)
    
    # Generate trading patterns
    patterns = {
        'detected_patterns': [p['pattern_id'] for p in relevant_patterns],
        'pattern_count': len(relevant_patterns),
        'confidence': ensemble_confidence,
        'base_analysis': research_insights,
        'autonomous_contribution': pattern_boost,
        'timestamp': datetime.now().isoformat()
    }
    
    state['patterns'] = patterns
    state['intelligence_timestamp'] = datetime.now().isoformat()
    
    print(f"✅ Intelligence analysis completed")
    print(f"   Base confidence: {base_confidence:.3f}")
    print(f"   Autonomous boost: {pattern_boost:.3f}")
    print(f"   Ensemble confidence: {ensemble_confidence:.3f}")
    print(f"   Relevant patterns: {len(relevant_patterns)}")
    
    return state

@safe_execute
def execution_node(state: UnifiedTradingState) -> UnifiedTradingState:
    """Trading Execution Node with pattern-informed decisions"""
    print("⚡ Execution Node: Making trading decisions...")
    
    patterns = state.get('patterns', {})
    confidence = patterns.get('confidence', 0.0)
    market_summary = state.get('market_summary', {})
    
    # Conservative decision making
    if confidence >= 0.75:
        action = 'buy'
        position_size = min(0.02, confidence * 0.03)
    elif confidence <= 0.25:
        action = 'sell'
        position_size = min(0.02, (1 - confidence) * 0.03)
    else:
        action = 'hold'
        position_size = 0.0
    
    trade_signal = {
        'action': action,
        'position_size': position_size,
        'confidence': confidence,
        'reasoning': f"Ensemble decision based on {patterns.get('pattern_count', 0)} patterns",
        'autonomous_patterns': patterns.get('detected_patterns', []),
        'timestamp': datetime.now().isoformat()
    }
    
    # Simulate execution result
    execution_result = {
        'executed': True,
        'action_taken': action,
        'size': position_size,
        'price': market_summary.get('current_price', 0),
        'status': 'completed',
        'timestamp': datetime.now().isoformat()
    }
    
    state['trade_signal'] = trade_signal
    state['execution_result'] = execution_result
    state['execution_timestamp'] = datetime.now().isoformat()
    
    print(f"✅ Execution completed - Action: {action}, Size: {position_size:.4f}")
    
    return state

@safe_execute
def monitoring_node(state: UnifiedTradingState) -> UnifiedTradingState:
    """Enhanced Monitoring with autonomous system tracking"""
    print("📊 Monitoring Node: Tracking performance...")
    
    execution_result = state.get('execution_result', {})
    patterns = state.get('patterns', {})
    pattern_library = state.get('pattern_library', {})
    
    # Calculate performance metrics
    performance_metrics = {
        'confidence_achieved': patterns.get('confidence', 0.0),
        'patterns_used': len(patterns.get('detected_patterns', [])),
        'autonomous_patterns_total': len(pattern_library),
        'execution_status': execution_result.get('status', 'unknown'),
        'discovery_session_count': len(state.get('discovery_history', [])),
        'patterns_discovered_total': state.get('patterns_discovered_count', 0),
        'last_discovery': state.get('last_discovery_session', 'never'),
        'timestamp': datetime.now().isoformat()
    }
    
    # Update trade results history
    if 'trade_results' not in state:
        state['trade_results'] = []
    
    trade_result = {
        'timestamp': datetime.now().isoformat(),
        'action': execution_result.get('action_taken', 'hold'),
        'confidence': patterns.get('confidence', 0.0),
        'patterns_used': patterns.get('detected_patterns', []),
        'autonomous_contribution': patterns.get('autonomous_contribution', 0.0)
    }
    
    state['trade_results'].append(trade_result)
    
    # Limit history size
    if len(state['trade_results']) > MAX_TRADE_RESULTS_HISTORY:
        state['trade_results'] = state['trade_results'][-MAX_TRADE_RESULTS_HISTORY:]
    
    state['performance_metrics'] = performance_metrics
    state['monitoring_timestamp'] = datetime.now().isoformat()
    
    # Determine if adaptation is needed
    state['needs_adaptation'] = False  # Simple logic for now
    
    print(f"✅ Monitoring completed")
    print(f"   Confidence: {performance_metrics['confidence_achieved']:.3f}")
    print(f"   Patterns used: {performance_metrics['patterns_used']}")
    print(f"   Total autonomous patterns: {performance_metrics['autonomous_patterns_total']}")
    
    return state

def should_continue(state: UnifiedTradingState) -> str:
    """Determine if the workflow should continue or end"""
    if state.get('needs_adaptation', False):
        return "ingestion"  # Continue with adaptations
    return END

def build_unified_trading_graph():
    """Build the unified LangGraph trading system with autonomous discovery"""
    
    # Create the graph
    workflow = StateGraph(UnifiedTradingState)
    
    # Add all nodes
    workflow.add_node("ingestion", ingestion_node)
    workflow.add_node("research", research_node)
    workflow.add_node("autonomous_discovery", autonomous_discovery_node)
    workflow.add_node("pattern_validation", pattern_validation_node)
    workflow.add_node("intelligence", intelligence_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("monitoring", monitoring_node)
    
    # Define the workflow
    workflow.set_entry_point("ingestion")
    workflow.add_edge("ingestion", "research")
    workflow.add_edge("research", "autonomous_discovery")
    workflow.add_edge("autonomous_discovery", "pattern_validation")
    workflow.add_edge("pattern_validation", "intelligence")
    workflow.add_edge("intelligence", "execution")
    workflow.add_edge("execution", "monitoring")
    
    # Add conditional ending
    workflow.add_conditional_edges(
        "monitoring",
        should_continue,
        {
            "ingestion": "ingestion",
            END: END
        }
    )
    
    # Compile with persistent SQLite memory
    conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
    memory = SqliteSaver(conn)
    app = workflow.compile(checkpointer=memory)
    
    print("✅ Unified LangGraph trading system compiled with autonomous discovery")
    return app

def run_unified_autonomous_system(asset: str = DEFAULT_ASSET,
                                 max_iterations: int = 3,
                                 autonomous_enabled: bool = True):
    """Run the unified autonomous trading system"""
    print(f"🚀 Starting Unified Autonomous Trading System for {asset}")
    print("=" * 70)
    
    app = build_unified_trading_graph()
    
    # Initialize state with autonomous discovery enabled
    initial_state: UnifiedTradingState = {
        # Trading fields
        'asset': asset,
        'timeframe': DEFAULT_TIMEFRAME,
        'start_time': datetime.now().isoformat(),
        'raw_data': {},
        'market_summary': {},
        'research_insights': {},
        'patterns': {},
        'trade_signal': {},
        'execution_result': {},
        'performance_metrics': {},
        'adaptations': [],
        'needs_adaptation': False,
        'errors': [],
        'data_timestamp': '',
        'research_timestamp': '',
        'intelligence_timestamp': '',
        'execution_timestamp': '',
        'monitoring_timestamp': '',
        'iteration_count': 0,
        
        # Autonomous fields
        'autonomous_mode_enabled': autonomous_enabled,
        'discovery_schedule': {
            'frequency_hours': 1,  # Frequent for demo
            'session_duration': 0.1,  # Short sessions
            'max_patterns_per_session': 3
        },
        'pattern_library': {},
        'discovery_history': [],
        'exploration_sessions': [],
        'meta_learning_state': {},
        'autonomous_performance': {},
        
        # Enhanced memory fields
        'patterns_history': [],
        'trade_results': [],
        'emergence_signals': [],
        'pattern_performance_map': {},
        'market_conditions_history': [],
        'parameter_adjustments': [],
        'ensemble_weights': {},
        'risk_adjustment_history': [],
        
        # Discovery tracking
        'last_discovery_session': None,
        'patterns_discovered_count': 0,
        'patterns_integrated_count': 0,
        'discovery_session_active': False
    }
    
    config = {
        "configurable": {"thread_id": f"unified_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"},
        "recursion_limit": 50
    }
    
    iteration = 0
    result = initial_state
    
    while iteration < max_iterations:
        try:
            print(f"\n🔄 Unified Cycle {iteration + 1}/{max_iterations}")
            result = app.invoke(initial_state, config)
            
            # Print comprehensive summary
            print(f"\n📈 Unified Cycle Summary:")
            print(f"   Trade Signal: {result.get('trade_signal', {}).get('action', 'N/A')}")
            print(f"   Confidence: {result.get('patterns', {}).get('confidence', 0):.3f}")
            print(f"   Autonomous Patterns: {len(result.get('pattern_library', {}))}")
            print(f"   Patterns Discovered: {result.get('patterns_discovered_count', 0)}")
            print(f"   Last Discovery: {result.get('last_discovery_session', 'Never')}")
            print(f"   Needs Adaptation: {result.get('needs_adaptation', False)}")
            
            # Check for completion
            if not result.get('needs_adaptation', False):
                print("✅ Unified cycle completed successfully")
                break
            
            # Prepare for next iteration
            initial_state = result
            iteration += 1
            
        except Exception as e:
            print(f"❌ Error in unified iteration {iteration + 1}: {e}")
            traceback.print_exc()
            break
    
    print("\n🏁 Unified Autonomous Trading System completed")
    
    # Print final autonomous summary
    final_patterns = len(result.get('pattern_library', {}))
    final_discoveries = result.get('patterns_discovered_count', 0)
    
    print(f"\n🤖 AUTONOMOUS DISCOVERY SUMMARY:")
    print(f"   Total Patterns in Library: {final_patterns}")
    print(f"   Total Patterns Discovered: {final_discoveries}")
    print(f"   Discovery History: {len(result.get('discovery_history', []))}")
    print(f"   Trade Results: {len(result.get('trade_results', []))}")
    
    return result

if __name__ == "__main__":
    # Test the unified system
    try:
        result = run_unified_autonomous_system(
            asset=DEFAULT_ASSET,
            max_iterations=2,
            autonomous_enabled=True
        )
        
        print("\n" + "=" * 70)
        print("🎯 FINAL UNIFIED SYSTEM SUMMARY:")
        print(json.dumps({
            'trade_signal': result.get('trade_signal', {}),
            'performance_metrics': result.get('performance_metrics', {}),
            'autonomous_patterns': len(result.get('pattern_library', {})),
            'patterns_discovered': result.get('patterns_discovered_count', 0),
            'adaptations_count': len(result.get('adaptations', []))
        }, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Unified system test failed: {e}")
        traceback.print_exc() 