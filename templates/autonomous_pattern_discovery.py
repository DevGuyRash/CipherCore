# templates/autonomous_pattern_discovery.py
# Autonomous Pattern Discovery Engine
# Uses raw AI intelligence to discover emergent patterns across multiple timeframes

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from openai import OpenAI
import asyncio
import itertools
from collections import defaultdict, deque
import hashlib
import pickle
from abc import ABC, abstractmethod
import re

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    print(f"Warning: OpenAI client initialization failed: {e}")
    client = None

@dataclass
class DiscoveredPattern:
    """Data structure for autonomously discovered patterns"""
    pattern_id: str
    discovery_timestamp: str
    raw_description: str
    ai_reasoning: str
    confidence_score: float
    timeframes_involved: List[str]
    data_dimensions: List[str]
    validation_results: Dict[str, Any]
    forward_test_results: Dict[str, Any]
    emergence_indicators: List[str]
    pattern_evolution: List[Dict[str, Any]]
    meta_features: Dict[str, Any]
    discovery_method: str
    uniqueness_score: float

@dataclass
class ExplorationSession:
    """Data structure for autonomous exploration sessions"""
    session_id: str
    start_time: str
    end_time: Optional[str]
    exploration_strategy: str
    datasets_explored: List[str]
    timeframes_analyzed: List[str]
    patterns_discovered: List[str]
    ai_insights: List[str]
    exploration_depth: int
    success_metrics: Dict[str, float]

class AutonomousDataExplorer:
    """
    Autonomous data exploration engine that discovers novel patterns
    using raw AI intelligence across multiple timeframes and datasets
    """
    
    def __init__(self, apex_adapter=None):
        self.apex_adapter = apex_adapter
        self.discovery_history = deque(maxlen=1000)
        self.exploration_sessions = deque(maxlen=100)
        self.pattern_library = {}
        self.meta_learning_state = {
            'successful_strategies': defaultdict(float),
            'exploration_preferences': defaultdict(float),
            'pattern_effectiveness': defaultdict(list),
            'ai_reasoning_quality': defaultdict(float)
        }
        
        # Autonomous exploration parameters
        self.exploration_strategies = [
            'temporal_decomposition',
            'cross_timeframe_resonance',
            'anomaly_emergence_tracking',
            'behavioral_regime_discovery',
            'micro_macro_pattern_bridging',
            'chaos_order_transition_detection',
            'market_microstructure_emergence',
            'collective_behavior_patterns',
            'information_flow_patterns',
            'adaptive_market_hypothesis_testing'
        ]
        
        self.timeframe_combinations = self._generate_timeframe_combinations()
        self.data_exploration_methods = self._initialize_exploration_methods()
    
    def _generate_timeframe_combinations(self) -> List[Tuple[str, ...]]:
        """Generate combinations of timeframes for multi-scale analysis"""
        base_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
        
        combinations = []
        # Single timeframes
        combinations.extend([(tf,) for tf in base_timeframes])
        
        # Pairs
        combinations.extend(list(itertools.combinations(base_timeframes, 2)))
        
        # Triplets (selected combinations)
        triplet_combinations = [
            ('1m', '15m', '1h'),
            ('5m', '1h', '4h'),
            ('15m', '4h', '1d'),
            ('1h', '1d', '1w'),
            ('1m', '1h', '1d'),
            ('5m', '4h', '1w')
        ]
        combinations.extend(triplet_combinations)
        
        return combinations
    
    def _initialize_exploration_methods(self) -> Dict[str, callable]:
        """Initialize different autonomous exploration methods"""
        return {
            'raw_pattern_emergence': self._explore_raw_pattern_emergence,
            'temporal_signature_analysis': self._explore_temporal_signatures,
            'cross_scale_resonance': self._explore_cross_scale_resonance,
            'behavioral_regime_mapping': self._explore_behavioral_regimes,
            'information_cascade_detection': self._explore_information_cascades,
            'emergent_structure_discovery': self._explore_emergent_structures,
            'adaptive_feature_evolution': self._explore_adaptive_features,
            'meta_pattern_synthesis': self._explore_meta_patterns,
            'chaos_emergence_tracking': self._explore_chaos_emergence,
            'collective_intelligence_patterns': self._explore_collective_patterns
        }
    
    async def autonomous_pattern_discovery_session(self, 
                                                 duration_hours: int = 8,
                                                 max_patterns: int = 50,
                                                 exploration_intensity: str = 'deep') -> ExplorationSession:
        """
        Run autonomous pattern discovery session that operates unattended
        
        Args:
            duration_hours: How long to run the discovery session
            max_patterns: Maximum patterns to discover
            exploration_intensity: 'light', 'medium', 'deep', 'exhaustive'
        """
        
        session_id = f"autonomous_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_start = datetime.now()
        
        print(f"🤖 Starting Autonomous Pattern Discovery Session: {session_id}")
        print(f"⏱️ Duration: {duration_hours} hours | Intensity: {exploration_intensity}")
        print(f"🎯 Target: {max_patterns} patterns | Strategy: AI-driven exploration")
        print("=" * 80)
        
        session = ExplorationSession(
            session_id=session_id,
            start_time=session_start.isoformat(),
            end_time=None,
            exploration_strategy=exploration_intensity,
            datasets_explored=[],
            timeframes_analyzed=[],
            patterns_discovered=[],
            ai_insights=[],
            exploration_depth=0,
            success_metrics={}
        )
        
        patterns_discovered = []
        exploration_cycle = 0
        
        end_time = session_start + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time and len(patterns_discovered) < max_patterns:
            exploration_cycle += 1
            
            print(f"\n🔍 Autonomous Exploration Cycle {exploration_cycle}")
            
            # Select exploration strategy based on meta-learning
            strategy = self._select_exploration_strategy()
            timeframe_combo = self._select_timeframe_combination()
            exploration_method = self._select_exploration_method()
            
            print(f"   Strategy: {strategy}")
            print(f"   Timeframes: {timeframe_combo}")
            print(f"   Method: {exploration_method}")
            
            try:
                # Fetch multi-timeframe data
                data_package = await self._fetch_multi_timeframe_data(timeframe_combo)
                session.datasets_explored.extend([f"{tf}_data" for tf in timeframe_combo])
                session.timeframes_analyzed.extend(timeframe_combo)
                
                # Apply exploration method
                exploration_results = await self.data_exploration_methods[exploration_method](
                    data_package, strategy, timeframe_combo
                )
                
                # Use AI to discover patterns in the exploration results
                discovered_patterns = await self._ai_pattern_discovery(
                    exploration_results, strategy, exploration_method
                )
                
                # Validate discovered patterns
                for pattern in discovered_patterns:
                    validation_result = await self._autonomous_pattern_validation(pattern)
                    pattern.validation_results = validation_result
                    
                    if validation_result.get('is_valid', False):
                        patterns_discovered.append(pattern)
                        session.patterns_discovered.append(pattern.pattern_id)
                        self.pattern_library[pattern.pattern_id] = pattern
                        
                        print(f"   ✅ Discovered valid pattern: {pattern.pattern_id}")
                        print(f"      Confidence: {pattern.confidence_score:.3f}")
                        print(f"      Uniqueness: {pattern.uniqueness_score:.3f}")
                
                # Update meta-learning
                self._update_meta_learning(strategy, exploration_method, len(discovered_patterns))
                
                # Brief pause to avoid overwhelming the system
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"   ❌ Error in exploration cycle {exploration_cycle}: {e}")
                continue
        
        session.end_time = datetime.now().isoformat()
        session.exploration_depth = exploration_cycle
        session.success_metrics = self._calculate_session_metrics(patterns_discovered)
        
        self.exploration_sessions.append(session)
        
        print(f"\n🏁 Autonomous Discovery Session Complete: {session_id}")
        print(f"📊 Patterns Discovered: {len(patterns_discovered)}")
        print(f"🔬 Exploration Cycles: {exploration_cycle}")
        print(f"⏱️ Duration: {(datetime.now() - session_start).total_seconds() / 3600:.2f} hours")
        
        return session
    
    async def _fetch_multi_timeframe_data(self, timeframes: Tuple[str, ...]) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes simultaneously"""
        
        data_package = {}
        
        for timeframe in timeframes:
            try:
                if self.apex_adapter:
                    # Fetch real data from ApeX (use reasonable limit)
                    # ApeX API works better with smaller limits (200 max according to docs)
                    limit = min(200, 100)  # Use 100 candles for good pattern analysis
                    
                    market_data = self.apex_adapter.get_comprehensive_market_data(
                        symbol='BTCUSDT',
                        timeframe=self._convert_timeframe_to_apex(timeframe),
                        limit=limit
                    )
                    
                    if 'ohlcv_data' in market_data:
                        df = pd.DataFrame(market_data['ohlcv_data'])
                        data_package[timeframe] = df
                else:
                    # Generate synthetic data for testing (use reasonable amount)
                    data_package[timeframe] = self._generate_synthetic_data(timeframe, 100)
                    
            except Exception as e:
                print(f"   ⚠️ Failed to fetch {timeframe} data: {e}")
                # Use synthetic data as fallback (use reasonable amount)
                data_package[timeframe] = self._generate_synthetic_data(timeframe, 100)
        
        return data_package
    
    def _convert_timeframe_to_apex(self, timeframe: str) -> str:
        """Convert timeframe to ApeX format"""
        conversion_map = {
            '1m': '1',
            '5m': '5',
            '15m': '15',
            '1h': '60',
            '4h': '240',
            '1d': '1440',
            '1w': '10080'
        }
        return conversion_map.get(timeframe, '60')
    
    def _generate_synthetic_data(self, timeframe: str, length: int) -> pd.DataFrame:
        """Generate synthetic market data for testing"""
        
        np.random.seed(42)  # For reproducible testing
        
        # Generate realistic OHLCV data with various patterns
        base_price = 120000
        prices = []
        volumes = []
        
        for i in range(length):
            # Add various patterns: trend, cycles, noise, regime changes
            trend = 0.001 * np.sin(i / 100)  # Long-term trend
            cycle = 0.005 * np.sin(i / 20)   # Medium cycle
            noise = 0.002 * np.random.randn() # Random noise
            regime = 0.01 if i > length * 0.7 else 0  # Regime change
            
            price_change = trend + cycle + noise + regime
            base_price *= (1 + price_change)
            
            # Generate OHLC from base price
            high = base_price * (1 + abs(np.random.randn()) * 0.002)
            low = base_price * (1 - abs(np.random.randn()) * 0.002)
            open_price = base_price + np.random.randn() * 0.001 * base_price
            close_price = base_price
            
            prices.append({
                'timestamp': datetime.now() - timedelta(minutes=i * self._timeframe_to_minutes(timeframe)),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': abs(np.random.randn() * 1000 + 500)
            })
        
        return pd.DataFrame(prices)
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe to minutes"""
        conversion = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60,
            '4h': 240, '1d': 1440, '1w': 10080
        }
        return conversion.get(timeframe, 60)
    
    async def _explore_raw_pattern_emergence(self, data_package: Dict[str, pd.DataFrame], 
                                           strategy: str, timeframes: Tuple[str, ...]) -> Dict[str, Any]:
        """Explore raw patterns without any traditional indicators"""
        
        exploration_results = {
            'method': 'raw_pattern_emergence',
            'timeframes': timeframes,
            'raw_features': {},
            'emergent_behaviors': [],
            'anomalies': [],
            'temporal_structures': {}
        }
        
        for timeframe, df in data_package.items():
            if len(df) < 50:
                continue
                
            # Extract raw features without traditional indicators
            raw_features = {
                'price_sequences': df['close'].values,
                'volume_sequences': df['volume'].values,
                'price_velocity': np.diff(df['close'].values),
                'price_acceleration': np.diff(np.diff(df['close'].values)),
                'volume_velocity': np.diff(df['volume'].values),
                'price_volume_correlation': np.corrcoef(df['close'], df['volume'])[0, 1],
                'price_range_sequences': (df['high'] - df['low']).values,
                'body_size_sequences': abs(df['close'] - df['open']).values,
                'wick_ratios': ((df['high'] - df['low']) - abs(df['close'] - df['open'])) / (df['high'] - df['low']),
                'temporal_gaps': np.diff(pd.to_datetime(df['timestamp']).astype(int) / 10**9),
            }
            
            # Detect emergent behaviors
            emergent_behaviors = self._detect_emergent_behaviors(raw_features)
            
            # Find anomalies in the raw data
            anomalies = self._detect_raw_anomalies(raw_features)
            
            exploration_results['raw_features'][timeframe] = raw_features
            exploration_results['emergent_behaviors'].extend(emergent_behaviors)
            exploration_results['anomalies'].extend(anomalies)
        
        # Cross-timeframe emergence analysis
        if len(data_package) > 1:
            cross_timeframe_emergence = self._analyze_cross_timeframe_emergence(exploration_results)
            exploration_results['cross_timeframe_emergence'] = cross_timeframe_emergence
        
        return exploration_results
    
    def _detect_emergent_behaviors(self, raw_features: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect emergent behaviors in raw market data"""
        
        behaviors = []
        
        # Analyze price sequences for emergent patterns
        price_seq = raw_features['price_sequences']
        if len(price_seq) > 100:
            
            # Detect self-similar structures
            autocorrelations = [np.corrcoef(price_seq[:-lag], price_seq[lag:])[0, 1] 
                              for lag in range(1, min(50, len(price_seq)//2))]
            
            significant_lags = [i+1 for i, corr in enumerate(autocorrelations) if abs(corr) > 0.3]
            
            if significant_lags:
                behaviors.append({
                    'type': 'temporal_self_similarity',
                    'significant_lags': significant_lags,
                    'max_correlation': max([abs(corr) for corr in autocorrelations]),
                    'description': f'Price shows self-similar behavior at lags: {significant_lags}'
                })
            
            # Detect regime-like structures
            price_changes = np.diff(price_seq)
            rolling_volatility = pd.Series(price_changes).rolling(20).std().values
            volatility_regimes = self._detect_volatility_regimes(rolling_volatility)
            
            if len(volatility_regimes) > 1:
                behaviors.append({
                    'type': 'volatility_regime_emergence',
                    'regimes': volatility_regimes,
                    'regime_count': len(volatility_regimes),
                    'description': f'Detected {len(volatility_regimes)} distinct volatility regimes'
                })
        
        # Analyze volume-price emergence
        if 'volume_sequences' in raw_features:
            volume_seq = raw_features['volume_sequences']
            price_volume_coupling = self._analyze_price_volume_coupling(price_seq, volume_seq)
            
            if price_volume_coupling['coupling_strength'] > 0.4:
                behaviors.append({
                    'type': 'price_volume_emergence',
                    'coupling_data': price_volume_coupling,
                    'description': f'Strong price-volume coupling detected: {price_volume_coupling["coupling_strength"]:.3f}'
                })
        
        return behaviors
    
    def _detect_volatility_regimes(self, volatility_series: np.ndarray) -> List[Dict[str, Any]]:
        """Detect volatility regimes using unsupervised methods"""
        
        if len(volatility_series) < 50:
            return []
        
        # Remove NaN values
        vol_clean = volatility_series[~np.isnan(volatility_series)]
        
        if len(vol_clean) < 20:
            return []
        
        # Simple regime detection using quantiles
        low_threshold = np.percentile(vol_clean, 33)
        high_threshold = np.percentile(vol_clean, 67)
        
        regimes = []
        current_regime = None
        regime_start = 0
        
        for i, vol in enumerate(vol_clean):
            if vol <= low_threshold:
                regime_type = 'low_volatility'
            elif vol >= high_threshold:
                regime_type = 'high_volatility'
            else:
                regime_type = 'medium_volatility'
            
            if current_regime != regime_type:
                if current_regime is not None:
                    regimes.append({
                        'type': current_regime,
                        'start': regime_start,
                        'end': i-1,
                        'duration': i - regime_start,
                        'avg_volatility': np.mean(vol_clean[regime_start:i])
                    })
                current_regime = regime_type
                regime_start = i
        
        # Add final regime
        if current_regime is not None:
            regimes.append({
                'type': current_regime,
                'start': regime_start,
                'end': len(vol_clean)-1,
                'duration': len(vol_clean) - regime_start,
                'avg_volatility': np.mean(vol_clean[regime_start:])
            })
        
        return regimes
    
    def _analyze_price_volume_coupling(self, price_seq: np.ndarray, volume_seq: np.ndarray) -> Dict[str, Any]:
        """Analyze coupling between price and volume"""
        
        if len(price_seq) != len(volume_seq) or len(price_seq) < 20:
            return {'coupling_strength': 0.0}
        
        # Calculate price changes
        price_changes = np.diff(price_seq)
        volume_changes = np.diff(volume_seq)
        
        # Basic correlation
        if len(price_changes) > 0 and len(volume_changes) > 0:
            basic_correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
            if np.isnan(basic_correlation):
                basic_correlation = 0.0
        else:
            basic_correlation = 0.0
        
        # Non-linear coupling analysis
        price_abs_changes = np.abs(price_changes)
        volume_norm = (volume_seq[1:] - np.mean(volume_seq)) / np.std(volume_seq)
        
        nonlinear_correlation = np.corrcoef(price_abs_changes, volume_norm)[0, 1]
        if np.isnan(nonlinear_correlation):
            nonlinear_correlation = 0.0
        
        coupling_strength = max(abs(basic_correlation), abs(nonlinear_correlation))
        
        return {
            'coupling_strength': coupling_strength,
            'linear_correlation': basic_correlation,
            'nonlinear_correlation': nonlinear_correlation,
            'dominant_coupling': 'linear' if abs(basic_correlation) > abs(nonlinear_correlation) else 'nonlinear'
        }
    
    def _detect_raw_anomalies(self, raw_features: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect anomalies in raw market features"""
        
        anomalies = []
        
        # Price anomalies
        price_seq = raw_features['price_sequences']
        if len(price_seq) > 50:
            price_changes = np.diff(price_seq) / price_seq[:-1]  # Relative changes
            
            # Statistical anomalies
            mean_change = np.mean(price_changes)
            std_change = np.std(price_changes)
            
            anomaly_threshold = 3 * std_change
            anomaly_indices = np.where(np.abs(price_changes - mean_change) > anomaly_threshold)[0]
            
            if len(anomaly_indices) > 0:
                anomalies.append({
                    'type': 'price_statistical_anomaly',
                    'indices': anomaly_indices.tolist(),
                    'severity': np.max(np.abs(price_changes[anomaly_indices] - mean_change) / std_change),
                    'count': len(anomaly_indices),
                    'description': f'Found {len(anomaly_indices)} statistical price anomalies'
                })
        
        # Volume anomalies
        if 'volume_sequences' in raw_features:
            volume_seq = raw_features['volume_sequences']
            if len(volume_seq) > 50:
                volume_changes = np.diff(volume_seq)
                
                # Detect volume spikes
                volume_mean = np.mean(volume_seq)
                volume_std = np.std(volume_seq)
                
                spike_threshold = volume_mean + 3 * volume_std
                spike_indices = np.where(volume_seq > spike_threshold)[0]
                
                if len(spike_indices) > 0:
                    anomalies.append({
                        'type': 'volume_spike_anomaly',
                        'indices': spike_indices.tolist(),
                        'max_spike': np.max(volume_seq[spike_indices]) / volume_mean,
                        'count': len(spike_indices),
                        'description': f'Found {len(spike_indices)} volume spike anomalies'
                    })
        
        return anomalies
    
    def _analyze_cross_timeframe_emergence(self, exploration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emergent behaviors across multiple timeframes"""
        
        cross_emergence = {
            'timeframe_correlations': {},
            'emergent_synchronizations': [],
            'scale_invariant_patterns': [],
            'temporal_cascades': []
        }
        
        # Analyze correlations between timeframes
        raw_features = exploration_results['raw_features']
        timeframes = list(raw_features.keys())
        
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i+1:]:
                if 'price_sequences' in raw_features[tf1] and 'price_sequences' in raw_features[tf2]:
                    
                    # Resample to common length for comparison
                    seq1 = raw_features[tf1]['price_sequences']
                    seq2 = raw_features[tf2]['price_sequences']
                    
                    min_len = min(len(seq1), len(seq2))
                    if min_len > 20:
                        # Sample every nth element to match lengths
                        step1 = len(seq1) // min_len
                        step2 = len(seq2) // min_len
                        
                        seq1_sampled = seq1[::step1][:min_len]
                        seq2_sampled = seq2[::step2][:min_len]
                        
                        correlation = np.corrcoef(seq1_sampled, seq2_sampled)[0, 1]
                        if not np.isnan(correlation):
                            cross_emergence['timeframe_correlations'][f'{tf1}_{tf2}'] = correlation
        
        return cross_emergence
    
    async def _explore_temporal_signatures(self, data_package: Dict[str, pd.DataFrame], 
                                         strategy: str, timeframes: Tuple[str, ...]) -> Dict[str, Any]:
        """Explore temporal signatures and patterns"""
        
        exploration_results = {
            'method': 'temporal_signature_analysis',
            'timeframes': timeframes,
            'temporal_patterns': {},
            'signature_emergence': [],
            'temporal_anomalies': []
        }
        
        for timeframe, df in data_package.items():
            if len(df) < 100:
                continue
            
            # Extract temporal signatures
            timestamps = pd.to_datetime(df['timestamp'])
            prices = df['close'].values
            
            # Time-of-day patterns
            hours = timestamps.dt.hour
            daily_patterns = {}
            for hour in range(24):
                hour_mask = hours == hour
                if hour_mask.sum() > 5:
                    hour_prices = prices[hour_mask]
                    daily_patterns[hour] = {
                        'avg_price': np.mean(hour_prices),
                        'volatility': np.std(hour_prices),
                        'sample_count': len(hour_prices)
                    }
            
            # Day-of-week patterns
            weekdays = timestamps.dt.dayofweek
            weekly_patterns = {}
            for day in range(7):
                day_mask = weekdays == day
                if day_mask.sum() > 5:
                    day_prices = prices[day_mask]
                    weekly_patterns[day] = {
                        'avg_price': np.mean(day_prices),
                        'volatility': np.std(day_prices),
                        'sample_count': len(day_prices)
                    }
            
            exploration_results['temporal_patterns'][timeframe] = {
                'daily_patterns': daily_patterns,
                'weekly_patterns': weekly_patterns
            }
        
        return exploration_results
    
    async def _explore_cross_scale_resonance(self, data_package: Dict[str, pd.DataFrame], 
                                           strategy: str, timeframes: Tuple[str, ...]) -> Dict[str, Any]:
        """Explore resonance patterns across different time scales"""
        
        exploration_results = {
            'method': 'cross_scale_resonance',
            'timeframes': timeframes,
            'resonance_patterns': [],
            'scale_coupling': {},
            'harmonic_analysis': {}
        }
        
        # This would involve complex cross-scale analysis
        # For now, provide a framework structure
        
        return exploration_results
    
    async def _explore_behavioral_regimes(self, data_package: Dict[str, pd.DataFrame], 
                                        strategy: str, timeframes: Tuple[str, ...]) -> Dict[str, Any]:
        """Explore behavioral regime patterns"""
        
        exploration_results = {
            'method': 'behavioral_regime_mapping',
            'timeframes': timeframes,
            'regimes_discovered': [],
            'regime_transitions': [],
            'behavioral_features': {}
        }
        
        # Framework for behavioral regime analysis
        
        return exploration_results
    
    async def _explore_information_cascades(self, data_package: Dict[str, pd.DataFrame], 
                                          strategy: str, timeframes: Tuple[str, ...]) -> Dict[str, Any]:
        """Explore information cascade patterns"""
        return {'method': 'information_cascade_detection', 'timeframes': timeframes}
    
    async def _explore_emergent_structures(self, data_package: Dict[str, pd.DataFrame], 
                                         strategy: str, timeframes: Tuple[str, ...]) -> Dict[str, Any]:
        """Explore emergent market structures"""
        return {'method': 'emergent_structure_discovery', 'timeframes': timeframes}
    
    async def _explore_adaptive_features(self, data_package: Dict[str, pd.DataFrame], 
                                       strategy: str, timeframes: Tuple[str, ...]) -> Dict[str, Any]:
        """Explore adaptive feature evolution"""
        return {'method': 'adaptive_feature_evolution', 'timeframes': timeframes}
    
    async def _explore_meta_patterns(self, data_package: Dict[str, pd.DataFrame], 
                                   strategy: str, timeframes: Tuple[str, ...]) -> Dict[str, Any]:
        """Explore meta-patterns and higher-order structures"""
        return {'method': 'meta_pattern_synthesis', 'timeframes': timeframes}
    
    async def _explore_chaos_emergence(self, data_package: Dict[str, pd.DataFrame], 
                                     strategy: str, timeframes: Tuple[str, ...]) -> Dict[str, Any]:
        """Explore chaos and emergence patterns"""
        return {'method': 'chaos_emergence_tracking', 'timeframes': timeframes}
    
    async def _explore_collective_patterns(self, data_package: Dict[str, pd.DataFrame], 
                                         strategy: str, timeframes: Tuple[str, ...]) -> Dict[str, Any]:
        """Explore collective intelligence patterns"""
        return {'method': 'collective_intelligence_patterns', 'timeframes': timeframes}
    
    async def _ai_pattern_discovery(self, exploration_results: Dict[str, Any], 
                                  strategy: str, exploration_method: str) -> List[DiscoveredPattern]:
        """Use AI to discover patterns in exploration results"""
        
        if not client:
            return []
        
        # Prepare data for AI analysis
        analysis_prompt = self._create_pattern_discovery_prompt(exploration_results, strategy, exploration_method)
        
        try:
            response = client.responses.create(
                model="o3-mini",
                input=analysis_prompt
            )
            
            # Extract AI reasoning
            ai_reasoning = ""
            if hasattr(response, 'reasoning_item') and response.reasoning_item:
                ai_reasoning = response.reasoning_item.summary
            elif hasattr(response, 'output_message') and response.output_message:
                ai_reasoning = response.output_message.content
            elif hasattr(response, 'content'):
                ai_reasoning = response.content
            else:
                ai_reasoning = str(response)
            
            # DEBUG: Save raw AI output to file
            debug_dir = "ai_debug_logs"
            os.makedirs(debug_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(os.path.join(debug_dir, f"ai_raw_output_{ts}.txt"), "w", encoding="utf-8") as f:
                f.write(ai_reasoning)
            print(f"   [DEBUG] Saved raw AI output to {debug_dir}/ai_raw_output_{ts}.txt")
            
            # Parse AI response to extract discovered patterns
            patterns, all_candidates = self._parse_ai_pattern_response(ai_reasoning, exploration_results, strategy, exploration_method, debug_dir, ts)
            
            # DEBUG: Save all candidate patterns to file
            with open(os.path.join(debug_dir, f"ai_candidates_{ts}.json"), "w", encoding="utf-8") as f:
                json.dump(all_candidates, f, indent=2, default=str)
            print(f"   [DEBUG] Saved all candidate patterns to {debug_dir}/ai_candidates_{ts}.json")
            
            return patterns
            
        except Exception as e:
            print(f"   ⚠️ AI pattern discovery failed: {e}")
            return []
    
    def _create_pattern_discovery_prompt(self, exploration_results: Dict[str, Any], 
                                       strategy: str, exploration_method: str) -> str:
        """Create prompt for AI pattern discovery"""
        
        prompt = f"""
# AUTONOMOUS PATTERN DISCOVERY TASK

You are an advanced AI pattern discovery engine analyzing raw market data to find emergent, novel patterns that have never been documented before. Your task is to use pure intelligence and reasoning to identify patterns without relying on traditional technical analysis.

## EXPLORATION DATA:
Method: {exploration_method}
Strategy: {strategy}
Timeframes: {exploration_results.get('timeframes', [])}

## RAW DATA ANALYSIS:
{json.dumps(exploration_results, indent=2, default=str)}

## YOUR MISSION:
1. **DISCOVER NOVEL PATTERNS**: Find patterns that emerge from the raw data without using traditional indicators
2. **USE RAW INTELLIGENCE**: Apply deep reasoning to understand underlying market behaviors
3. **IDENTIFY EMERGENCE**: Look for patterns that emerge from complex interactions
4. **AVOID TRADITIONAL FRAMEWORKS**: Do not use RSI, MACD, moving averages, or other standard indicators

## PATTERN DISCOVERY GUIDELINES:
- Focus on emergent behaviors that arise from complex market interactions
- Look for self-organizing patterns and regime changes
- Identify temporal signatures and cross-timeframe resonances
- Find anomalies that indicate structural changes
- Discover meta-patterns that govern other patterns

## REQUIRED OUTPUT FORMAT:
For each discovered pattern, provide:

**PATTERN_ID**: [Unique identifier]
**DESCRIPTION**: [Detailed description of the emergent pattern]
**CONFIDENCE**: [0.0-1.0 confidence in pattern validity]
**UNIQUENESS**: [0.0-1.0 how novel/unique this pattern is]
**EMERGENCE_INDICATORS**: [List of what indicates this pattern is emerging]
**TIMEFRAMES_INVOLVED**: [Which timeframes show this pattern]
**VALIDATION_SUGGESTIONS**: [How to validate this pattern]
**REASONING**: [Your detailed reasoning for why this is a valid pattern]

Discover as many novel, emergent patterns as possible. Think beyond traditional market analysis.
"""
        
        return prompt
    
    def _parse_ai_pattern_response(self, ai_reasoning: str, exploration_results: Dict[str, Any], 
                                 strategy: str, exploration_method: str, debug_dir=None, ts=None) -> (list, list):
        """Parse AI response to extract discovered patterns, robust to field formatting"""
        patterns = []
        all_candidates = []
        lines = ai_reasoning.split('\n')
        current_pattern = {}
        def match_field(line, field):
            # Accept both '**FIELD**:' and 'FIELD:'
            return line.startswith(f'**{field}**:') or line.startswith(f'{field}:')
        def extract_value(line, field):
            if line.startswith(f'**{field}**:'):
                return line.split(':', 1)[1].strip()
            elif line.startswith(f'{field}:'):
                return line.split(':', 1)[1].strip()
            return None
        for line in lines:
            line = line.strip()
            if match_field(line, 'PATTERN_ID'):
                if current_pattern:
                    all_candidates.append(current_pattern.copy())
                    pattern = self._create_discovered_pattern(current_pattern, exploration_results, strategy, exploration_method)
                    if pattern:
                        patterns.append(pattern)
                current_pattern = {'pattern_id': extract_value(line, 'PATTERN_ID')}
            elif match_field(line, 'DESCRIPTION') and current_pattern:
                current_pattern['description'] = extract_value(line, 'DESCRIPTION')
            elif match_field(line, 'CONFIDENCE') and current_pattern:
                try:
                    val = extract_value(line, 'CONFIDENCE')
                    # Handle cases like '0. seventy-five (0.75)'
                    m = re.search(r'([0-9]*\.?[0-9]+)', val)
                    current_pattern['confidence'] = float(m.group(1)) if m else 0.5
                except:
                    current_pattern['confidence'] = 0.5
            elif match_field(line, 'UNIQUENESS') and current_pattern:
                try:
                    val = extract_value(line, 'UNIQUENESS')
                    m = re.search(r'([0-9]*\.?[0-9]+)', val)
                    current_pattern['uniqueness'] = float(m.group(1)) if m else 0.5
                except:
                    current_pattern['uniqueness'] = 0.5
            elif match_field(line, 'REASONING') and current_pattern:
                current_pattern['reasoning'] = extract_value(line, 'REASONING')
        if current_pattern:
            all_candidates.append(current_pattern.copy())
            pattern = self._create_discovered_pattern(current_pattern, exploration_results, strategy, exploration_method)
            if pattern:
                patterns.append(pattern)
        return patterns, all_candidates
    
    def _create_discovered_pattern(self, pattern_data: Dict[str, Any], exploration_results: Dict[str, Any], 
                                 strategy: str, exploration_method: str) -> Optional[DiscoveredPattern]:
        """Create a DiscoveredPattern object from parsed data"""
        
        if 'pattern_id' not in pattern_data:
            return None
        
        pattern_id = f"{pattern_data['pattern_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return DiscoveredPattern(
            pattern_id=pattern_id,
            discovery_timestamp=datetime.now().isoformat(),
            raw_description=pattern_data.get('description', 'No description'),
            ai_reasoning=pattern_data.get('reasoning', 'No reasoning provided'),
            confidence_score=pattern_data.get('confidence', 0.5),
            timeframes_involved=exploration_results.get('timeframes', []),
            data_dimensions=list(exploration_results.keys()),
            validation_results={},
            forward_test_results={},
            emergence_indicators=pattern_data.get('emergence_indicators', []),
            pattern_evolution=[],
            meta_features={
                'discovery_method': exploration_method,
                'strategy': strategy,
                'exploration_session': datetime.now().isoformat()
            },
            discovery_method=exploration_method,
            uniqueness_score=pattern_data.get('uniqueness', 0.5)
        )
    
    async def _autonomous_pattern_validation(self, pattern: DiscoveredPattern) -> Dict[str, Any]:
        """Autonomously validate discovered patterns (lowered thresholds for debug)"""
        validation_result = {
            'is_valid': False,
            'validation_score': 0.0,
            'validation_methods': [],
            'confidence_adjustment': 1.0,
            'validation_notes': []
        }
        # Validation criteria (LOWERED for debug)
        criteria_checks = []
        # 1. Confidence threshold
        if pattern.confidence_score >= 0.3:
            criteria_checks.append(('confidence_threshold', True, 0.3))
            validation_result['validation_notes'].append('Confidence threshold met')
        else:
            criteria_checks.append(('confidence_threshold', False, 0.0))
            validation_result['validation_notes'].append('Low confidence score')
        # 2. Uniqueness threshold
        if pattern.uniqueness_score >= 0.3:
            criteria_checks.append(('uniqueness_threshold', True, 0.2))
            validation_result['validation_notes'].append('High uniqueness score')
        else:
            criteria_checks.append(('uniqueness_threshold', False, 0.0))
            validation_result['validation_notes'].append('Pattern may not be novel')
        # 3. Description quality
        if len(pattern.raw_description) > 20:
            criteria_checks.append(('description_quality', True, 0.2))
            validation_result['validation_notes'].append('Detailed description provided')
        else:
            criteria_checks.append(('description_quality', False, 0.0))
            validation_result['validation_notes'].append('Description too brief')
        # 4. AI reasoning quality
        if len(pattern.ai_reasoning) > 40:
            criteria_checks.append(('reasoning_quality', True, 0.3))
            validation_result['validation_notes'].append('Comprehensive AI reasoning')
        else:
            criteria_checks.append(('reasoning_quality', False, 0.0))
            validation_result['validation_notes'].append('Limited reasoning provided')
        # Calculate validation score
        total_score = sum(score for _, passed, score in criteria_checks if passed)
        validation_result['validation_score'] = total_score
        validation_result['is_valid'] = total_score >= 0.3
        validation_result['validation_methods'] = [method for method, passed, _ in criteria_checks if passed]
        return validation_result
    
    def _select_exploration_strategy(self) -> str:
        """Select exploration strategy based on meta-learning"""
        
        # Use meta-learning to prefer successful strategies
        if self.meta_learning_state['successful_strategies']:
            strategies = list(self.meta_learning_state['successful_strategies'].keys())
            weights = list(self.meta_learning_state['successful_strategies'].values())
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                probabilities = [w / total_weight for w in weights]
                return np.random.choice(strategies, p=probabilities)
        
        # Random selection if no history
        return np.random.choice(self.exploration_strategies)
    
    def _select_timeframe_combination(self) -> Tuple[str, ...]:
        """Select timeframe combination for exploration"""
        return self.timeframe_combinations[np.random.randint(len(self.timeframe_combinations))]
    
    def _select_exploration_method(self) -> str:
        """Select exploration method"""
        methods = list(self.data_exploration_methods.keys())
        return np.random.choice(methods)
    
    def _update_meta_learning(self, strategy: str, exploration_method: str, patterns_found: int):
        """Update meta-learning state based on exploration results"""
        
        # Update strategy success
        success_score = min(patterns_found / 5.0, 1.0)  # Normalize to 0-1
        self.meta_learning_state['successful_strategies'][strategy] = (
            self.meta_learning_state['successful_strategies'][strategy] * 0.9 + success_score * 0.1
        )
        
        # Update exploration preferences
        self.meta_learning_state['exploration_preferences'][exploration_method] = (
            self.meta_learning_state['exploration_preferences'][exploration_method] * 0.9 + success_score * 0.1
        )
    
    def _calculate_session_metrics(self, patterns_discovered: List[DiscoveredPattern]) -> Dict[str, float]:
        """Calculate metrics for the exploration session"""
        
        if not patterns_discovered:
            return {
                'patterns_per_hour': 0.0,
                'avg_confidence': 0.0,
                'avg_uniqueness': 0.0,
                'discovery_rate': 0.0
            }
        
        avg_confidence = np.mean([p.confidence_score for p in patterns_discovered])
        avg_uniqueness = np.mean([p.uniqueness_score for p in patterns_discovered])
        
        return {
            'patterns_discovered': len(patterns_discovered),
            'avg_confidence': avg_confidence,
            'avg_uniqueness': avg_uniqueness,
            'discovery_rate': len(patterns_discovered) / max(1, len(patterns_discovered))  # Placeholder
        }
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of all discovered patterns"""
        
        return {
            'total_patterns': len(self.pattern_library),
            'exploration_sessions': len(self.exploration_sessions),
            'pattern_library': {pid: asdict(pattern) for pid, pattern in self.pattern_library.items()},
            'meta_learning_state': dict(self.meta_learning_state)
        }

# Test function
async def test_autonomous_discovery():
    """Test the autonomous pattern discovery system"""
    
    print("🧪 Testing Autonomous Pattern Discovery System...")
    
    explorer = AutonomousDataExplorer()
    
    # Run a short discovery session
    session = await explorer.autonomous_pattern_discovery_session(
        duration_hours=0.1,  # 6 minutes for testing
        max_patterns=5,
        exploration_intensity='medium'
    )
    
    print(f"\n📊 Discovery Session Results:")
    print(f"   Session ID: {session.session_id}")
    print(f"   Patterns Discovered: {len(session.patterns_discovered)}")
    print(f"   Exploration Depth: {session.exploration_depth}")
    print(f"   Success Metrics: {session.success_metrics}")
    
    # Get discovery summary
    summary = explorer.get_discovery_summary()
    print(f"\n📋 Overall Discovery Summary:")
    print(f"   Total Patterns: {summary['total_patterns']}")
    print(f"   Sessions Completed: {summary['exploration_sessions']}")
    
    return session, summary

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_autonomous_discovery()) 