#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Continuous Data Capture System with ML Integration - FINAL VERSION
Sistema profesional con ML adaptativo para clasificación inteligente de patrones
Autor: biGGzeta
Fecha: 2025-10-05 15:32:31 UTC
Versión: 4.0.0 (ML Integrated)
"""

import asyncio
import datetime
import json
import os
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import websockets
from collections import deque
import statistics
from concurrent.futures import ThreadPoolExecutor
import pickle

# ML Imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

@dataclass
class ProfessionalDataPoint:
    """Data point profesional con todos los datos institucionales"""
    timestamp: datetime.datetime
    
    # Basic Price Data
    price: float
    open_price: float
    high_price: float
    low_price: float
    volume: float
    
    # Advanced Price Data
    weighted_avg_price: float
    price_change_24h: float
    price_change_pct_24h: float
    volume_change_pct: float
    trade_count: int
    
    # Real OrderBook Data
    best_bid: float
    best_ask: float
    real_spread: float
    bid_quantity: float
    ask_quantity: float
    orderbook_imbalance: float
    
    # OrderBook Depth (5 levels)
    bids_depth: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks_depth: List[Tuple[float, float]]
    total_bid_volume: float
    total_ask_volume: float
    depth_ratio: float
    
    # Whale Detection
    whale_bids: float
    whale_asks: float
    whale_activity_score: float
    large_orders_count: int
    
    # Market Microstructure
    taker_buy_volume: float = 0
    taker_sell_volume: float = 0
    buy_sell_ratio: float = 1.0
    market_momentum: float = 0

@dataclass
class MLEnhancedClassification:
    """Clasificación profesional mejorada con ML"""
    timestamp: datetime.datetime
    
    # Base Classification
    trend_1m: str
    trend_5m: str
    trend_15m: str
    trend_1h: str
    
    # ML Enhanced Predictions
    ml_breakout_probability: float
    ml_lateral_probability: float
    ml_whale_sentiment: str
    ml_setup_type: str
    ml_confidence: float
    
    # Advanced Volatility
    volatility_1m: float
    volatility_5m: float
    volatility_realized: float
    volatility_regime: str
    
    # Market Phases (ML Enhanced)
    market_phase: str
    regime_type: str
    
    # Volume Analysis (ML Enhanced)
    volume_profile: str
    volume_strength: float
    volume_anomaly: bool
    
    # OrderBook Intelligence (ML Enhanced)
    orderbook_pressure: str
    whale_sentiment: str
    liquidity_depth: str
    
    # ML Pattern Recognition
    coiling_breakout_score: float
    whale_accumulation_score: float
    lateral_compression_score: float
    false_breakout_risk: float
    
    # Enhanced Momentum & Strength
    momentum_1m: float
    momentum_5m: float
    momentum_strength: str
    
    # Support/Resistance (ML Enhanced)
    support_levels: List[float]
    resistance_levels: List[float]
    key_level_proximity: float
    
    # Risk Metrics (ML Enhanced)
    risk_score: float
    volatility_risk: str
    liquidity_risk: str
    ml_risk_adjustment: float

class ProfessionalMLEnhancement:
    """ML enhancement para clasificación profesional de patrones"""
    
    def __init__(self):
        self.feature_history = deque(maxlen=10000)  # 10k samples
        self.pattern_labels = deque(maxlen=10000)
        
        # ML Models especializados
        self.breakout_classifier = None
        self.lateral_classifier = None
        self.whale_behavior_model = None
        self.setup_predictor = None
        self.risk_adjuster = None
        
        # Feature scaling
        self.feature_scaler = StandardScaler() if ML_AVAILABLE else None
        
        # Pattern tracking
        self.pattern_history = {
            "coiling_breakout": [],
            "whale_accumulation": [],
            "false_breakout": [],
            "lateral_compression": [],
            "volume_divergence": []
        }
        
        # ML Configuration
        self.learning_enabled = ML_AVAILABLE
        self.confidence_threshold = 0.75
        self.retrain_interval = 100  # Retrain cada 100 samples
        self.model_performance = {}
        
        # Setup logging
        self.logger = logging.getLogger("ProfessionalML")
        
        if ML_AVAILABLE:
            self.logger.info("[ML] Professional ML Enhancement iniciado")
            self._initialize_models()
        else:
            self.logger.warning("[ML] scikit-learn no disponible - ML disabled")
    
    def _initialize_models(self):
        """Inicializar modelos ML"""
        
        try:
            # Inicializar con modelos básicos
            self.breakout_classifier = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            )
            
            self.lateral_classifier = RandomForestClassifier(
                n_estimators=80, 
                max_depth=8, 
                random_state=42,
                class_weight='balanced'
            )
            
            self.whale_behavior_model = GradientBoostingClassifier(
                n_estimators=50, 
                learning_rate=0.1, 
                random_state=42
            )
            
            self.setup_predictor = RandomForestClassifier(
                n_estimators=120, 
                max_depth=12, 
                random_state=42
            )
            
            self.risk_adjuster = GradientBoostingClassifier(
                n_estimators=60, 
                learning_rate=0.05, 
                random_state=42
            )
            
            self.logger.info("[ML] Modelos inicializados exitosamente")
            
        except Exception as e:
            self.logger.error(f"[ML] Error inicializando modelos: {str(e)}")
            self.learning_enabled = False
    
    async def enhance_classification(self, base_classification, recent_data, orderbook_data, klines_data):
        """Mejorar clasificación con ML"""
        
        if not self.learning_enabled:
            return self._convert_to_ml_enhanced(base_classification)
        
        try:
            # Extraer features profesionales
            features = self._extract_professional_features(recent_data, orderbook_data, klines_data)
            
            if not features:
                return self._convert_to_ml_enhanced(base_classification)
            
            # Predecir patrones con ML
            ml_predictions = await self._predict_patterns(features)
            
            # Combinar con clasificación base
            enhanced_classification = self._merge_classifications(
                base_classification, ml_predictions, features
            )
            
            # Learn from new data
            await self._update_learning_data(features, enhanced_classification)
            
            return enhanced_classification
            
        except Exception as e:
            self.logger.error(f"[ML] Error en enhancement: {str(e)}")
            return self._convert_to_ml_enhanced(base_classification)
    
    def _extract_professional_features(self, recent_data, orderbook_data, klines_data):
        """Extraer features profesionales para ML"""
        
        if len(recent_data) < 30:
            return None
        
        features = {}
        
        try:
            # 1. PRICE FEATURES AVANZADAS
            prices = [d.price for d in recent_data]
            features.update(self._extract_price_features(prices))
            
            # 2. ORDERBOOK FEATURES PROFESIONALES
            if orderbook_data:
                features.update(self._extract_orderbook_features(orderbook_data))
            
            # 3. VOLUME FEATURES MULTI-TIMEFRAME
            if klines_data:
                features.update(self._extract_volume_features(klines_data))
            
            # 4. WHALE FEATURES AVANZADAS
            features.update(self._extract_whale_features(recent_data, orderbook_data))
            
            # 5. TECHNICAL FEATURES
            features.update(self._extract_technical_features(recent_data))
            
            # 6. PATTERN FEATURES
            features.update(self._extract_pattern_features(recent_data, orderbook_data))
            
            return features
            
        except Exception as e:
            self.logger.error(f"[ML] Error extrayendo features: {str(e)}")
            return None
    
    def _extract_price_features(self, prices):
        """Extraer features avanzadas de precio"""
        
        features = {}
        
        # Basic stats
        features['price_mean'] = np.mean(prices)
        features['price_std'] = np.std(prices)
        features['price_range'] = (max(prices) - min(prices)) / np.mean(prices)
        
        # Momentum features
        features['momentum_5m'] = (prices[-1] - prices[0]) / prices[0]
        features['momentum_1m'] = (prices[-1] - prices[-12]) / prices[-12] if len(prices) >= 12 else 0
        
        # Acceleration
        if len(prices) >= 6:
            velocities = np.diff(prices)
            accelerations = np.diff(velocities)
            features['price_acceleration'] = np.mean(accelerations[-3:]) if len(accelerations) >= 3 else 0
        else:
            features['price_acceleration'] = 0
        
        # Compression ratio
        recent_range = max(prices[-20:]) - min(prices[-20:]) if len(prices) >= 20 else max(prices) - min(prices)
        features['compression_ratio'] = 1 - (recent_range / np.mean(prices))
        
        # Volatility clustering
        returns = np.diff(np.log(prices))
        if len(returns) > 10:
            features['volatility_clustering'] = np.std(returns[-10:]) / np.std(returns[:-10])
        else:
            features['volatility_clustering'] = 1.0
        
        return features
    
    def _extract_orderbook_features(self, orderbook_data):
        """Extraer features del orderbook"""
        
        features = {}
        
        # Imbalance features
        imbalances = [ob['orderbook_imbalance'] for ob in orderbook_data]
        features['imbalance_mean'] = np.mean(imbalances)
        features['imbalance_std'] = np.std(imbalances)
        features['imbalance_trend'] = np.polyfit(range(len(imbalances)), imbalances, 1)[0]
        
        # Spread features
        spreads = [ob['real_spread'] for ob in orderbook_data]
        features['spread_mean'] = np.mean(spreads)
        features['spread_volatility'] = np.std(spreads) / np.mean(spreads)
        features['spread_trend'] = np.polyfit(range(len(spreads)), spreads, 1)[0]
        
        # Depth features
        depth_ratios = [ob['depth_ratio'] for ob in orderbook_data]
        features['depth_ratio_mean'] = np.mean(depth_ratios)
        features['depth_stability'] = 1 - (np.std(depth_ratios) / np.mean(depth_ratios))
        
        # Liquidity features
        total_volumes = [ob['total_bid_volume'] + ob['total_ask_volume'] for ob in orderbook_data]
        features['liquidity_mean'] = np.mean(total_volumes)
        features['liquidity_trend'] = np.polyfit(range(len(total_volumes)), total_volumes, 1)[0]
        
        return features
    
    def _extract_volume_features(self, klines_data):
        """Extraer features de volumen"""
        
        features = {}
        
        if not klines_data:
            return features
        
        volumes = [k['volume'] for k in klines_data]
        buy_ratios = [k.get('buy_sell_ratio', 1.0) for k in klines_data]
        
        # Volume features
        features['volume_mean'] = np.mean(volumes)
        features['volume_trend'] = np.polyfit(range(len(volumes)), volumes, 1)[0]
        features['volume_volatility'] = np.std(volumes) / np.mean(volumes)
        
        # Buy/sell features
        features['buy_sell_mean'] = np.mean(buy_ratios)
        features['buy_sell_trend'] = np.polyfit(range(len(buy_ratios)), buy_ratios, 1)[0]
        features['buy_sell_imbalance'] = np.mean(buy_ratios) - 1.0
        
        # Volume profile
        if len(volumes) >= 10:
            recent_vol = np.mean(volumes[-5:])
            past_vol = np.mean(volumes[:-5])
            features['volume_acceleration'] = (recent_vol - past_vol) / past_vol
        else:
            features['volume_acceleration'] = 0
        
        return features
    
    def _extract_whale_features(self, recent_data, orderbook_data):
        """Extraer features de whale activity"""
        
        features = {}
        
        # From recent data
        whale_scores = [d.whale_activity_score for d in recent_data]
        large_orders = [d.large_orders_count for d in recent_data]
        
        features['whale_activity_mean'] = np.mean(whale_scores)
        features['whale_activity_trend'] = np.polyfit(range(len(whale_scores)), whale_scores, 1)[0]
        features['large_orders_frequency'] = np.mean(large_orders)
        
        # From orderbook data
        if orderbook_data:
            whale_bids = [ob['whale_bids'] for ob in orderbook_data]
            whale_asks = [ob['whale_asks'] for ob in orderbook_data]
            
            features['whale_bid_pressure'] = np.mean(whale_bids)
            features['whale_ask_pressure'] = np.mean(whale_asks)
            
            # Whale positioning
            total_whale = [wb + wa for wb, wa in zip(whale_bids, whale_asks)]
            if total_whale and max(total_whale) > 0:
                whale_bid_ratios = [wb / (wb + wa) if (wb + wa) > 0 else 0.5 
                                  for wb, wa in zip(whale_bids, whale_asks)]
                features['whale_positioning'] = np.mean(whale_bid_ratios)
                features['whale_positioning_consistency'] = 1 - np.std(whale_bid_ratios)
            else:
                features['whale_positioning'] = 0.5
                features['whale_positioning_consistency'] = 0
        
        return features
    
    def _extract_technical_features(self, recent_data):
        """Extraer features técnicas"""
        
        features = {}
        prices = [d.price for d in recent_data]
        
        # Support/Resistance strength
        features['sr_strength'] = self._calculate_sr_strength(prices)
        
        # Breakout potential
        features['breakout_potential'] = self._calculate_breakout_potential(recent_data)
        
        # Setup completion
        features['setup_completion'] = self._calculate_setup_completion(recent_data)
        
        # Range characteristics
        if len(prices) >= 20:
            # High-low ratio
            features['hl_ratio'] = (max(prices) - min(prices)) / np.mean(prices)
            
            # Price position in range
            current_price = prices[-1]
            price_min, price_max = min(prices), max(prices)
            if price_max > price_min:
                features['price_position'] = (current_price - price_min) / (price_max - price_min)
            else:
                features['price_position'] = 0.5
        
        return features
    
    def _extract_pattern_features(self, recent_data, orderbook_data):
        """Extraer features de patrones"""
        
        features = {}
        
        # Coiling pattern detection
        features['coiling_score'] = self._detect_coiling_pattern(recent_data)
        
        # Lateral compression
        features['lateral_compression'] = self._calculate_lateral_compression(recent_data)
        
        # Whale accumulation pattern
        features['whale_accumulation'] = self._detect_whale_accumulation(recent_data, orderbook_data)
        
        # Volume divergence
        features['volume_divergence'] = self._detect_volume_divergence(recent_data)
        
        # False breakout risk
        features['false_breakout_risk'] = self._calculate_false_breakout_risk(recent_data)
        
        return features
    
    def _detect_coiling_pattern(self, recent_data):
        """Detectar patrón de coiling"""
        
        if len(recent_data) < 30:
            return 0
        
        prices = [d.price for d in recent_data]
        volumes = [d.volume for d in recent_data]
        
        # Criteria para coiling
        score = 0
        
        # 1. Price compression
        price_range = (max(prices) - min(prices)) / np.mean(prices)
        if price_range < 0.015:  # <1.5% range
            score += 25
        
        # 2. Decreasing volume
        if len(volumes) >= 10:
            recent_vol = np.mean(volumes[-10:])
            earlier_vol = np.mean(volumes[:-10])
            if recent_vol < earlier_vol:
                score += 25
        
        # 3. Stability
        price_std = np.std(prices[-20:]) if len(prices) >= 20 else np.std(prices)
        if price_std / np.mean(prices) < 0.005:  # Very stable
            score += 25
        
        # 4. Time factor
        if len(recent_data) >= 60:  # At least 5 minutes
            score += 25
        
        return score
    
    def _calculate_lateral_compression(self, recent_data):
        """Calcular compresión lateral"""
        
        if len(recent_data) < 20:
            return 0
        
        prices = [d.price for d in recent_data]
        
        # Measure compression over time
        compressions = []
        window = 10
        
        for i in range(window, len(prices)):
            subset = prices[i-window:i]
            compression = 1 - ((max(subset) - min(subset)) / np.mean(subset))
            compressions.append(compression)
        
        return np.mean(compressions) if compressions else 0
    
    def _detect_whale_accumulation(self, recent_data, orderbook_data):
        """Detectar acumulación de whales"""
        
        score = 0
        
        # From recent data
        whale_scores = [d.whale_activity_score for d in recent_data]
        if whale_scores:
            if np.mean(whale_scores[-10:]) > np.mean(whale_scores[:-10]):
                score += 30
        
        # From orderbook
        if orderbook_data:
            whale_bids = [ob['whale_bids'] for ob in orderbook_data]
            whale_asks = [ob['whale_asks'] for ob in orderbook_data]
            
            # Increasing whale bids
            if len(whale_bids) >= 5:
                bid_trend = np.polyfit(range(len(whale_bids)), whale_bids, 1)[0]
                if bid_trend > 0:
                    score += 35
            
            # Whale bid dominance
            if whale_bids and whale_asks:
                avg_bid_dominance = np.mean([wb / (wb + wa) if (wb + wa) > 0 else 0.5 
                                           for wb, wa in zip(whale_bids, whale_asks)])
                if avg_bid_dominance > 0.6:
                    score += 35
        
        return score
    
    def _detect_volume_divergence(self, recent_data):
        """Detectar divergencia de volumen"""
        
        if len(recent_data) < 20:
            return 0
        
        prices = [d.price for d in recent_data]
        volumes = [d.volume for d in recent_data]
        
        # Price trend vs volume trend
        price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
        
        # Normalize trends
        price_trend_norm = price_trend / np.mean(prices)
        volume_trend_norm = volume_trend / np.mean(volumes)
        
        # Divergence score
        divergence = abs(price_trend_norm - volume_trend_norm)
        
        return min(100, divergence * 1000)  # Scale to 0-100
    
    def _calculate_false_breakout_risk(self, recent_data):
        """Calcular riesgo de false breakout"""
        
        if len(recent_data) < 30:
            return 50
        
        risk_factors = 0
        
        prices = [d.price for d in recent_data]
        volumes = [d.volume for d in recent_data]
        
        # 1. Low volume during move
        if len(volumes) >= 10:
            recent_vol = np.mean(volumes[-5:])
            avg_vol = np.mean(volumes)
            if recent_vol < avg_vol * 0.8:  # Volume 20% below average
                risk_factors += 25
        
        # 2. Rapid price movement
        if len(prices) >= 10:
            recent_change = abs(prices[-1] - prices[-10]) / prices[-10]
            if recent_change > 0.02:  # >2% rapid move
                risk_factors += 25
        
        # 3. Lack of whale support
        whale_scores = [d.whale_activity_score for d in recent_data[-10:]]
        if np.mean(whale_scores) < 0.05:
            risk_factors += 25
        
        # 4. High volatility
        if len(prices) >= 20:
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            if volatility > 0.01:  # >1% volatility
                risk_factors += 25
        
        return risk_factors
    
    async def _predict_patterns(self, features):
        """Predecir patrones con modelos ML"""
        
        if not features or not self.learning_enabled:
            return {}
        
        predictions = {}
        
        try:
            # Preparar features para modelos
            feature_array = np.array(list(features.values())).reshape(1, -1)
            
            # Scale features si hay scaler entrenado
            if hasattr(self.feature_scaler, 'mean_'):
                try:
                    feature_array = self.feature_scaler.transform(feature_array)
                except:
                    pass  # Use unscaled if scaling fails
            
            # 1. BREAKOUT CLASSIFIER
            if (self.breakout_classifier and 
                hasattr(self.breakout_classifier, 'classes_')):
                try:
                    breakout_proba = self.breakout_classifier.predict_proba(feature_array)[0]
                    predictions['breakout_probability'] = float(breakout_proba[1]) if len(breakout_proba) > 1 else 0.5
                    predictions['breakout_confidence'] = float(max(breakout_proba))
                except:
                    predictions['breakout_probability'] = 0.5
                    predictions['breakout_confidence'] = 0.5
            
            # 2. LATERAL CLASSIFIER
            if (self.lateral_classifier and 
                hasattr(self.lateral_classifier, 'classes_')):
                try:
                    lateral_proba = self.lateral_classifier.predict_proba(feature_array)[0]
                    predictions['lateral_probability'] = float(lateral_proba[1]) if len(lateral_proba) > 1 else 0.5
                    predictions['lateral_confidence'] = float(max(lateral_proba))
                except:
                    predictions['lateral_probability'] = 0.5
                    predictions['lateral_confidence'] = 0.5
            
            # 3. WHALE BEHAVIOR MODEL
            if (self.whale_behavior_model and 
                hasattr(self.whale_behavior_model, 'classes_')):
                try:
                    whale_pred = self.whale_behavior_model.predict(feature_array)[0]
                    whale_proba = self.whale_behavior_model.predict_proba(feature_array)[0]
                    predictions['whale_sentiment_ml'] = 'bullish' if whale_pred == 1 else 'bearish'
                    predictions['whale_confidence'] = float(max(whale_proba))
                except:
                    predictions['whale_sentiment_ml'] = 'neutral'
                    predictions['whale_confidence'] = 0.5
            
            # 4. SETUP SCORES (rule-based con features ML)
            predictions['coiling_breakout_score'] = features.get('coiling_score', 0)
            predictions['whale_accumulation_score'] = features.get('whale_accumulation', 0)
            predictions['lateral_compression_score'] = features.get('lateral_compression', 0) * 100
            predictions['false_breakout_risk'] = features.get('false_breakout_risk', 50)
            
            # 5. ML SETUP TYPE
            setup_scores = {
                'coiling': predictions['coiling_breakout_score'],
                'whale_accumulation': predictions['whale_accumulation_score'],
                'lateral': predictions['lateral_compression_score'],
                'breakout': predictions.get('breakout_probability', 0.5) * 100
            }
            
            predictions['ml_setup_type'] = max(setup_scores.items(), key=lambda x: x[1])[0]
            
            # 6. OVERALL ML CONFIDENCE
            confidences = [
                predictions.get('breakout_confidence', 0.5),
                predictions.get('lateral_confidence', 0.5),
                predictions.get('whale_confidence', 0.5)
            ]
            predictions['ml_confidence'] = float(np.mean(confidences))
            
        except Exception as e:
            self.logger.error(f"[ML] Error en predicciones: {str(e)}")
            # Default predictions
            predictions = {
                'breakout_probability': 0.5,
                'lateral_probability': 0.5,
                'whale_sentiment_ml': 'neutral',
                'ml_setup_type': 'lateral',
                'ml_confidence': 0.5,
                'coiling_breakout_score': 0,
                'whale_accumulation_score': 0,
                'lateral_compression_score': 0,
                'false_breakout_risk': 50
            }
        
        return predictions
    
    def _merge_classifications(self, base_classification, ml_predictions, features):
        """Combinar clasificación base con predicciones ML"""
        
        try:
            # Create ML enhanced classification
            enhanced = MLEnhancedClassification(
                timestamp=base_classification.timestamp,
                
                # Base trends
                trend_1m=base_classification.trend_1m,
                trend_5m=base_classification.trend_5m,
                trend_15m=base_classification.trend_15m,
                trend_1h=base_classification.trend_1h,
                
                # ML Enhanced predictions
                ml_breakout_probability=ml_predictions.get('breakout_probability', 0.5),
                ml_lateral_probability=ml_predictions.get('lateral_probability', 0.5),
                ml_whale_sentiment=ml_predictions.get('whale_sentiment_ml', 'neutral'),
                ml_setup_type=ml_predictions.get('ml_setup_type', 'lateral'),
                ml_confidence=ml_predictions.get('ml_confidence', 0.5),
                
                # Enhanced volatility
                volatility_1m=base_classification.volatility_1m,
                volatility_5m=base_classification.volatility_5m,
                volatility_realized=base_classification.volatility_realized,
                volatility_regime=base_classification.volatility_regime,
                
                # Enhanced market phases
                market_phase=self._enhance_market_phase(base_classification.market_phase, ml_predictions),
                regime_type=self._enhance_regime_type(base_classification.regime_type, ml_predictions),
                
                # Enhanced volume analysis
                volume_profile=self._enhance_volume_profile(base_classification.volume_profile, ml_predictions),
                volume_strength=base_classification.volume_strength,
                volume_anomaly=base_classification.volume_anomaly,
                
                # Enhanced orderbook intelligence
                orderbook_pressure=base_classification.orderbook_pressure,
                whale_sentiment=ml_predictions.get('whale_sentiment_ml', base_classification.whale_sentiment),
                liquidity_depth=base_classification.liquidity_depth,
                
                # ML Pattern scores
                coiling_breakout_score=ml_predictions.get('coiling_breakout_score', 0),
                whale_accumulation_score=ml_predictions.get('whale_accumulation_score', 0),
                lateral_compression_score=ml_predictions.get('lateral_compression_score', 0),
                false_breakout_risk=ml_predictions.get('false_breakout_risk', 50),
                
                # Enhanced momentum
                momentum_1m=base_classification.momentum_1m,
                momentum_5m=base_classification.momentum_5m,
                momentum_strength=base_classification.momentum_strength,
                
                # Enhanced support/resistance
                support_levels=base_classification.support_levels,
                resistance_levels=base_classification.resistance_levels,
                key_level_proximity=base_classification.key_level_proximity,
                
                # Enhanced risk
                risk_score=base_classification.risk_score,
                volatility_risk=base_classification.volatility_risk,
                liquidity_risk=base_classification.liquidity_risk,
                ml_risk_adjustment=self._calculate_ml_risk_adjustment(ml_predictions)
            )
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"[ML] Error merging classifications: {str(e)}")
            return self._convert_to_ml_enhanced(base_classification)
    
    def _enhance_market_phase(self, base_phase, ml_predictions):
        """Mejorar market phase con ML"""
        
        breakout_prob = ml_predictions.get('breakout_probability', 0.5)
        lateral_prob = ml_predictions.get('lateral_probability', 0.5)
        
        if breakout_prob > 0.7:
            return "pre_breakout"
        elif lateral_prob > 0.7:
            return "tight_consolidation"
        else:
            return base_phase
    
    def _enhance_regime_type(self, base_regime, ml_predictions):
        """Mejorar regime type con ML"""
        
        coiling_score = ml_predictions.get('coiling_breakout_score', 0)
        
        if coiling_score > 70:
            return "coiling"
        else:
            return base_regime
    
    def _enhance_volume_profile(self, base_profile, ml_predictions):
        """Mejorar volume profile con ML"""
        
        whale_acc_score = ml_predictions.get('whale_accumulation_score', 0)
        
        if whale_acc_score > 60:
            return "whale_accumulation"
        else:
            return base_profile
    
    def _calculate_ml_risk_adjustment(self, ml_predictions):
        """Calcular ajuste de riesgo basado en ML"""
        
        false_breakout_risk = ml_predictions.get('false_breakout_risk', 50)
        ml_confidence = ml_predictions.get('ml_confidence', 0.5)
        
        # Higher false breakout risk = higher risk adjustment
        # Lower ML confidence = higher risk adjustment
        risk_adjustment = (false_breakout_risk / 100) * 0.5 + (1 - ml_confidence) * 0.5
        
        return risk_adjustment
    
    def _convert_to_ml_enhanced(self, base_classification):
        """Convertir clasificación base a ML enhanced (fallback)"""
        
        return MLEnhancedClassification(
            timestamp=base_classification.timestamp,
            trend_1m=base_classification.trend_1m,
            trend_5m=base_classification.trend_5m,
            trend_15m=base_classification.trend_15m,
            trend_1h=base_classification.trend_1h,
            ml_breakout_probability=0.5,
            ml_lateral_probability=0.5,
            ml_whale_sentiment='neutral',
            ml_setup_type='lateral',
            ml_confidence=0.5,
            volatility_1m=base_classification.volatility_1m,
            volatility_5m=base_classification.volatility_5m,
            volatility_realized=base_classification.volatility_realized,
            volatility_regime=base_classification.volatility_regime,
            market_phase=base_classification.market_phase,
            regime_type=base_classification.regime_type,
            volume_profile=base_classification.volume_profile,
            volume_strength=base_classification.volume_strength,
            volume_anomaly=base_classification.volume_anomaly,
            orderbook_pressure=base_classification.orderbook_pressure,
            whale_sentiment=base_classification.whale_sentiment,
            liquidity_depth=base_classification.liquidity_depth,
            coiling_breakout_score=0,
            whale_accumulation_score=0,
            lateral_compression_score=0,
            false_breakout_risk=50,
            momentum_1m=base_classification.momentum_1m,
            momentum_5m=base_classification.momentum_5m,
            momentum_strength=base_classification.momentum_strength,
            support_levels=base_classification.support_levels,
            resistance_levels=base_classification.resistance_levels,
            key_level_proximity=base_classification.key_level_proximity,
            risk_score=base_classification.risk_score,
            volatility_risk=base_classification.volatility_risk,
            liquidity_risk=base_classification.liquidity_risk,
            ml_risk_adjustment=0.0
        )
    
    async def _update_learning_data(self, features, classification):
        """Actualizar datos de aprendizaje"""
        
        if not self.learning_enabled:
            return
        
        try:
            # Agregar a history
            self.feature_history.append(features)
            self.pattern_labels.append(classification)
            
            # Retrain periódicamente
            if len(self.feature_history) % self.retrain_interval == 0:
                await self._retrain_models()
            
        except Exception as e:
            self.logger.error(f"[ML] Error updating learning data: {str(e)}")
    
    async def _retrain_models(self):
        """Reentrenar modelos con datos acumulados"""
        
        if len(self.feature_history) < 50:
            return
        
        try:
            self.logger.info(f"[ML] Reentrenando modelos con {len(self.feature_history)} samples...")
            
            # Preparar datos
            recent_features = list(self.feature_history)[-1000:]  # Últimos 1000
            recent_labels = list(self.pattern_labels)[-1000:]
            
            # Crear feature matrix
            X = []
            for features in recent_features:
                X.append(list(features.values()))
            
            X = np.array(X)
            
            # Fit scaler
            self.feature_scaler.fit(X)
            X_scaled = self.feature_scaler.transform(X)
            
            # Create labels para breakout
            y_breakout = []
            for label in recent_labels:
                if hasattr(label, 'ml_breakout_probability'):
                    y_breakout.append(1 if label.ml_breakout_probability > 0.6 else 0)
                else:
                    # Fallback logic
                    y_breakout.append(1 if 'breakout' in str(label).lower() else 0)
            
            # Create labels para lateral
            y_lateral = []
            for label in recent_labels:
                if hasattr(label, 'ml_lateral_probability'):
                    y_lateral.append(1 if label.ml_lateral_probability > 0.6 else 0)
                else:
                    y_lateral.append(1 if 'lateral' in str(label).lower() else 0)
            
            # Create labels para whale
            y_whale = []
            for label in recent_labels:
                if hasattr(label, 'whale_sentiment'):
                    y_whale.append(1 if label.whale_sentiment == 'bullish' else 0)
                else:
                    y_whale.append(0)  # Default neutral
            
            # Retrain breakout classifier
            if len(set(y_breakout)) > 1:
                self.breakout_classifier.fit(X_scaled, y_breakout)
                self.model_performance['breakout'] = 'retrained'
            
            # Retrain lateral classifier
            if len(set(y_lateral)) > 1:
                self.lateral_classifier.fit(X_scaled, y_lateral)
                self.model_performance['lateral'] = 'retrained'
            
            # Retrain whale model
            if len(set(y_whale)) > 1:
                self.whale_behavior_model.fit(X_scaled, y_whale)
                self.model_performance['whale'] = 'retrained'
            
            self.logger.info(f"[ML] ✅ Modelos reentrenados exitosamente")
            
        except Exception as e:
            self.logger.error(f"[ML] Error reentrenando modelos: {str(e)}")
    
    # Helper methods para features
    def _calculate_sr_strength(self, prices):
        """Calcular fuerza de soporte/resistencia"""
        if len(prices) < 20:
            return 0.5
        
        # Simple implementation
        price_levels = {}
        tolerance = np.std(prices) * 0.5
        
        for price in prices:
            level_found = False
            for level in price_levels:
                if abs(price - level) <= tolerance:
                    price_levels[level] += 1
                    level_found = True
                    break
            
            if not level_found:
                price_levels[price] = 1
        
        max_touches = max(price_levels.values()) if price_levels else 1
        return min(1.0, max_touches / 5.0)
    
    def _calculate_breakout_potential(self, recent_data):
        """Calcular potencial de breakout"""
        if len(recent_data) < 20:
            return 0.5
        
        prices = [d.price for d in recent_data]
        volumes = [d.volume for d in recent_data]
        whale_activity = [d.whale_activity_score for d in recent_data]
        
        # Compression factor
        price_range = max(prices) - min(prices)
        compression = 1 - (price_range / np.mean(prices))
        
        # Volume factor
        volume_increase = np.mean(volumes[-5:]) / np.mean(volumes[:-5]) if len(volumes) > 5 else 1
        
        # Whale factor
        whale_factor = np.mean(whale_activity[-5:])
        
        potential = (compression * 0.4 + 
                    min(1.0, volume_increase - 1) * 0.4 +
                    whale_factor * 0.2)
        
        return min(1.0, potential)
    
    def _calculate_setup_completion(self, recent_data):
        """Calcular completitud del setup"""
        if len(recent_data) < 20:
            return 0.3
        
        criteria_met = 0
        total_criteria = 5
        
        prices = [d.price for d in recent_data]
        volumes = [d.volume for d in recent_data]
        
        # 1. Price compression
        price_range = (max(prices) - min(prices)) / np.mean(prices)
        if price_range < 0.02:
            criteria_met += 1
        
        # 2. Volume pattern
        if np.mean(volumes[-5:]) < np.mean(volumes[:-5]):
            criteria_met += 1
        
        # 3. Whale activity
        whale_scores = [d.whale_activity_score for d in recent_data]
        if np.mean(whale_scores) > 0.05:
            criteria_met += 1
        
        # 4. Time factor
        if len(recent_data) >= 30:
            criteria_met += 1
        
        # 5. Stability
        if np.std(prices[-10:]) / np.mean(prices[-10:]) < 0.005:
            criteria_met += 1
        
        return criteria_met / total_criteria

# Continuar con la clase principal ProfessionalDataCapture...
class ProfessionalDataCapture:
    """Sistema profesional de captura multi-stream con ML"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP"):
        self.symbol = symbol
        self.running = False
        
        # Professional data storage
        self.realtime_data = deque(maxlen=4320)  # 6 horas de data (5s intervals)
        self.klines_1m = deque(maxlen=1440)     # 24 horas de klines 1m
        self.klines_5m = deque(maxlen=288)      # 24 horas de klines 5m
        self.klines_15m = deque(maxlen=96)      # 24 horas de klines 15m
        self.orderbook_snapshots = deque(maxlen=720)  # 1 hora de orderbook
        self.trade_flow = deque(maxlen=10000)   # Trade flow analysis
        self.classifications = deque(maxlen=1440)  # 24 horas
        
        # ML Enhancement
        self.ml_enhancement = ProfessionalMLEnhancement()
        
        # WebSocket URLs for multiple streams
        self.websocket_urls = {
            "ticker": "wss://dstream.binance.com/ws/ethusd_perp@ticker",
            "orderbook": "wss://dstream.binance.com/ws/ethusd_perp@depth@100ms",
            "kline_1m": "wss://dstream.binance.com/ws/ethusd_perp@kline_1m",
            "kline_5m": "wss://dstream.binance.com/ws/ethusd_perp@kline_5m",
            "kline_15m": "wss://dstream.binance.com/ws/ethusd_perp@kline_15m",
            "trades": "wss://dstream.binance.com/ws/ethusd_perp@trade",
            "markPrice": "wss://dstream.binance.com/ws/ethusd_perp@markPrice"
        }
        
        # WebSocket connections
        self.connections = {}
        self.api_base = "https://dapi.binance.com/dapi/v1"
        
        # Configuration profesional
        self.config = {
            "capture_interval_seconds": 5,
            "classification_interval_seconds": 60,
            "save_interval_seconds": 60,
            "orderbook_levels": 10,
            "whale_threshold_usd": 100000,  # $100k orders
            "data_directory": "data/professional",
            "max_reconnect_attempts": 20,
            "heartbeat_interval": 15,
            "technical_indicators": True,
            "advanced_analytics": True,
            "ml_enabled": ML_AVAILABLE
        }
        
        # Professional state tracking
        self.data_source = "binance_professional_ml"
        self.capture_stats = {
            "ticker_points": 0,
            "orderbook_updates": 0,
            "klines_received": 0,
            "trades_processed": 0,
            "whale_orders_detected": 0,
            "classifications_made": 0,
            "ml_predictions_made": 0,
            "stream_errors": {},
            "connection_resets": 0,
            "data_quality_score": 100.0
        }
        
        # Threading for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup
        self.setup_logging()
        self.setup_directories()
        
        self.logger.info(f"[INIT] Professional Data Capture con ML iniciado para {symbol}")
        self.logger.info(f"[STREAMS] {len(self.websocket_urls)} streams configurados")
        self.logger.info(f"[CONFIG] Whale threshold: ${self.config['whale_threshold_usd']:,}")
        self.logger.info(f"[ML] ML Enhancement: {'✅ Enabled' if self.config['ml_enabled'] else '❌ Disabled'}")
    
    def setup_logging(self):
        """Setup logging profesional"""
        
        log_dir = "logs/professional"
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger("ProfessionalCapture")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        log_file = os.path.join(log_dir, f"professional_capture_ml_{self.symbol}_{datetime.datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def setup_directories(self):
        """Crear directorios profesionales con ML"""
        
        directories = [
            self.config["data_directory"],
            f"{self.config['data_directory']}/realtime",
            f"{self.config['data_directory']}/orderbook",
            f"{self.config['data_directory']}/klines",
            f"{self.config['data_directory']}/trades",
            f"{self.config['data_directory']}/whale_activity",
            f"{self.config['data_directory']}/classifications",
            f"{self.config['data_directory']}/ml_predictions",  # NUEVO
            f"{self.config['data_directory']}/ml_models",       # NUEVO
            f"{self.config['data_directory']}/analytics",
            f"{self.config['data_directory']}/risk_metrics"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    async def start_professional_capture(self):
        """Iniciar captura profesional multi-stream con ML"""
        
        self.logger.info("[START] Iniciando captura profesional multi-stream con ML")
        self.running = True
        
        # Saltamos validación como en la versión anterior
        self.logger.info("[START] Professional ML validation skipped - connecting directly to live streams")
        
        # Crear tasks para todos los streams + ML
        tasks = []
        
        # Stream tasks
        for stream_name, url in self.websocket_urls.items():
            task = asyncio.create_task(self._stream_handler(stream_name, url))
            tasks.append(task)
        
        # Analysis tasks con ML
        tasks.extend([
            asyncio.create_task(self._ml_enhanced_classification_loop()),  # NUEVO
            asyncio.create_task(self._whale_detection_loop()),
            asyncio.create_task(self._technical_analysis_loop()),
            asyncio.create_task(self._risk_analysis_loop()),
            asyncio.create_task(self._data_persistence_loop()),
            asyncio.create_task(self._connection_monitor_loop()),
            asyncio.create_task(self._stats_monitoring_loop()),
            asyncio.create_task(self._ml_monitoring_loop())  # NUEVO
        ])
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self._cleanup()
    
    # Continuar con los métodos de stream handling (mismos que la versión anterior)
    async def _stream_handler(self, stream_name: str, url: str):
        """Handler genérico para streams"""
        
        self.logger.info(f"[{stream_name.upper()}] Iniciando stream: {url}")
        
        while self.running:
            try:
                async with websockets.connect(url, ping_timeout=20) as websocket:
                    self.connections[stream_name] = websocket
                    self.logger.info(f"[{stream_name.upper()}] ✅ Conectado")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._process_stream_data(stream_name, data)
                        except Exception as e:
                            self.logger.debug(f"[{stream_name.upper()}] Error procesando: {str(e)}")
                            self._increment_stream_error(stream_name)
                        
            except Exception as e:
                self.logger.warning(f"[{stream_name.upper()}] ⚠️ Desconectado: {str(e)}")
                self.connections.pop(stream_name, None)
                
                if self.running:
                    await asyncio.sleep(5)  # Retry delay
    
    async def _process_stream_data(self, stream_name: str, data: Dict):
        """Procesar datos de diferentes streams"""
        
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        if stream_name == "ticker":
            await self._process_ticker_data(data, timestamp)
        elif stream_name == "orderbook":
            await self._process_orderbook_data(data, timestamp)
        elif stream_name.startswith("kline"):
            await self._process_kline_data(stream_name, data, timestamp)
        elif stream_name == "trades":
            await self._process_trade_data(data, timestamp)
        elif stream_name == "markPrice":
            await self._process_mark_price_data(data, timestamp)
    
    # Usar los mismos métodos de procesamiento de la versión anterior
    async def _process_ticker_data(self, data: Dict, timestamp: datetime.datetime):
        """Procesar datos del ticker"""
        
        try:
            if 'c' not in data:
                return
            
            # Crear data point básico
            basic_data = {
                'timestamp': timestamp,
                'price': float(data['c']),
                'open_price': float(data['o']),
                'high_price': float(data['h']),
                'low_price': float(data['l']),
                'volume': float(data['v']),
                'weighted_avg_price': float(data['w']),
                'price_change_24h': float(data['p']),
                'price_change_pct_24h': float(data['P']),
                'trade_count': int(data['n']),
                'volume_change_pct': 0
            }
            
            await self._enqueue_ticker_data(basic_data)
            self.capture_stats["ticker_points"] += 1
            
        except Exception as e:
            self.logger.error(f"[TICKER] Error: {str(e)}")
    
    async def _process_orderbook_data(self, data: Dict, timestamp: datetime.datetime):
        """Procesar datos del orderbook profesional"""
        
        try:
            if 'bids' not in data or 'asks' not in data:
                return
            
            bids = [[float(price), float(qty)] for price, qty in data['bids'][:self.config["orderbook_levels"]]]
            asks = [[float(price), float(qty)] for price, qty in data['asks'][:self.config["orderbook_levels"]]]
            
            if not bids or not asks:
                return
            
            # Calcular métricas profesionales del orderbook
            best_bid, bid_qty = bids[0]
            best_ask, ask_qty = asks[0]
            
            real_spread = best_ask - best_bid
            
            # Total volumes
            total_bid_volume = sum(qty for _, qty in bids)
            total_ask_volume = sum(qty for _, qty in asks)
            
            # Imbalance
            total_volume = total_bid_volume + total_ask_volume
            orderbook_imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
            
            # Whale detection
            whale_threshold = self.config["whale_threshold_usd"]
            whale_bids = sum(qty for price, qty in bids if qty * price > whale_threshold)
            whale_asks = sum(qty for price, qty in asks if qty * price > whale_threshold)
            
            # Crear orderbook snapshot profesional
            orderbook_snapshot = {
                'timestamp': timestamp,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'real_spread': real_spread,
                'bid_quantity': bid_qty,
                'ask_quantity': ask_qty,
                'orderbook_imbalance': orderbook_imbalance,
                'bids_depth': bids,
                'asks_depth': asks,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'depth_ratio': total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1.0,
                'whale_bids': whale_bids,
                'whale_asks': whale_asks,
                'whale_activity_score': (whale_bids + whale_asks) / total_volume if total_volume > 0 else 0,
                'large_orders_count': len([1 for p, q in bids + asks if q * p > whale_threshold])
            }
            
            self.orderbook_snapshots.append(orderbook_snapshot)
            self.capture_stats["orderbook_updates"] += 1
            
            # Detectar whales
            if whale_bids > 0 or whale_asks > 0:
                self.capture_stats["whale_orders_detected"] += 1
                await self._log_whale_activity(orderbook_snapshot)
            
        except Exception as e:
            self.logger.error(f"[ORDERBOOK] Error: {str(e)}")
    
    async def _process_kline_data(self, stream_name: str, data: Dict, timestamp: datetime.datetime):
        """Procesar datos de klines profesionales"""
        
        try:
            if 'k' not in data:
                return
            
            kline = data['k']
            if not kline.get('x', False):  # Solo klines cerradas
                return
            
            kline_data = {
                'timestamp': datetime.datetime.fromtimestamp(kline['t'] / 1000, tz=datetime.timezone.utc),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'trades': int(kline['n']),
                'taker_buy_volume': float(kline['V']),
                'taker_buy_quote_volume': float(kline['Q'])
            }
            
            # Calcular buy/sell ratio
            taker_sell_volume = kline_data['volume'] - kline_data['taker_buy_volume']
            kline_data['buy_sell_ratio'] = kline_data['taker_buy_volume'] / taker_sell_volume if taker_sell_volume > 0 else 1.0
            
            # Almacenar según timeframe
            if stream_name == "kline_1m":
                self.klines_1m.append(kline_data)
            elif stream_name == "kline_5m":
                self.klines_5m.append(kline_data)
            elif stream_name == "kline_15m":
                self.klines_15m.append(kline_data)
            
            self.capture_stats["klines_received"] += 1
            
        except Exception as e:
            self.logger.error(f"[{stream_name.upper()}] Error: {str(e)}")
    
    async def _process_trade_data(self, data: Dict, timestamp: datetime.datetime):
        """Procesar datos de trades individuales"""
        
        try:
            if 'p' not in data or 'q' not in data:
                return
            
            trade_data = {
                'timestamp': timestamp,
                'price': float(data['p']),
                'quantity': float(data['q']),
                'is_buyer_maker': data.get('m', False)
            }
            
            self.trade_flow.append(trade_data)
            self.capture_stats["trades_processed"] += 1
            
        except Exception as e:
            self.logger.error(f"[TRADES] Error: {str(e)}")
    
    async def _process_mark_price_data(self, data: Dict, timestamp: datetime.datetime):
        """Procesar datos de mark price"""
        
        try:
            if 'p' in data:
                mark_price = float(data['p'])
                # Agregar a data adicional si se necesita
                pass
            
        except Exception as e:
            self.logger.error(f"[MARKPRICE] Error: {str(e)}")
    
    async def _enqueue_ticker_data(self, basic_data: Dict):
        """Enqueue ticker data para procesamiento"""
        
        # Combine con último orderbook snapshot si existe
        latest_orderbook = self.orderbook_snapshots[-1] if self.orderbook_snapshots else {}
        
        # Crear data point profesional completo
        professional_point = ProfessionalDataPoint(
            timestamp=basic_data['timestamp'],
            price=basic_data['price'],
            open_price=basic_data['open_price'],
            high_price=basic_data['high_price'],
            low_price=basic_data['low_price'],
            volume=basic_data['volume'],
            weighted_avg_price=basic_data['weighted_avg_price'],
            price_change_24h=basic_data['price_change_24h'],
            price_change_pct_24h=basic_data['price_change_pct_24h'],
            trade_count=basic_data['trade_count'],
            volume_change_pct=basic_data['volume_change_pct'],
            
            # OrderBook data (si está disponible)
            best_bid=latest_orderbook.get('best_bid', basic_data['price'] * 0.9995),
            best_ask=latest_orderbook.get('best_ask', basic_data['price'] * 1.0005),
            real_spread=latest_orderbook.get('real_spread', basic_data['price'] * 0.001),
            bid_quantity=latest_orderbook.get('bid_quantity', 0),
            ask_quantity=latest_orderbook.get('ask_quantity', 0),
            orderbook_imbalance=latest_orderbook.get('orderbook_imbalance', 0),
            
            bids_depth=latest_orderbook.get('bids_depth', []),
            asks_depth=latest_orderbook.get('asks_depth', []),
            total_bid_volume=latest_orderbook.get('total_bid_volume', 0),
            total_ask_volume=latest_orderbook.get('total_ask_volume', 0),

            depth_ratio=latest_orderbook.get('depth_ratio', 1.0),
            
            # Whale data
            whale_bids=latest_orderbook.get('whale_bids', 0),
            whale_asks=latest_orderbook.get('whale_asks', 0),
            whale_activity_score=latest_orderbook.get('whale_activity_score', 0),
            large_orders_count=latest_orderbook.get('large_orders_count', 0),
            
            # Market microstructure (del último kline)
            taker_buy_volume=0,  # Se llenará del kline stream
            taker_sell_volume=0,
            buy_sell_ratio=1.0,
            market_momentum=0
        )
        
        self.realtime_data.append(professional_point)
        
        # Log cada 30 segundos con datos profesionales
        if self.capture_stats["ticker_points"] % 6 == 0:
            self.logger.info(f"[PROFESSIONAL] ${professional_point.price:.2f} "
                           f"Spread: ${professional_point.real_spread:.2f} "
                           f"Imbal: {professional_point.orderbook_imbalance:+.3f} "
                           f"Whales: {professional_point.large_orders_count} "
                           f"[{len(self.connections)}/{len(self.websocket_urls)} streams]")
    
    async def _log_whale_activity(self, orderbook_snapshot: Dict):
        """Log actividad de whales"""
        
        whale_bids = orderbook_snapshot['whale_bids']
        whale_asks = orderbook_snapshot['whale_asks']
        
        if whale_bids > whale_asks * 2:
            self.logger.warning(f"[WHALE] 🐋 Large bid pressure: ${whale_bids:,.0f} vs ${whale_asks:,.0f}")
        elif whale_asks > whale_bids * 2:
            self.logger.warning(f"[WHALE] 🐋 Large ask pressure: ${whale_asks:,.0f} vs ${whale_bids:,.0f}")
    
    def _increment_stream_error(self, stream_name: str):
        """Incrementar errores de stream"""
        
        if stream_name not in self.capture_stats["stream_errors"]:
            self.capture_stats["stream_errors"][stream_name] = 0
        self.capture_stats["stream_errors"][stream_name] += 1
    
    # ========== NUEVA FUNCIÓN ML ENHANCED CLASSIFICATION ==========
    
    async def _ml_enhanced_classification_loop(self):
        """Loop de clasificación profesional MEJORADO CON ML"""
        
        self.logger.info("[ML-CLASSIFY] Iniciando clasificación ML Enhanced cada 60s")
        
        while self.running:
            try:
                await asyncio.sleep(self.config["classification_interval_seconds"])
                
                if len(self.realtime_data) >= 60:  # Mínimo 5 minutos de data
                    # Crear clasificación base
                    base_classification = await self._create_base_classification()
                    
                    if base_classification and self.config["ml_enabled"]:
                        # MEJORAR CON ML
                        enhanced_classification = await self.ml_enhancement.enhance_classification(
                            base_classification,
                            list(self.realtime_data),
                            list(self.orderbook_snapshots),
                            list(self.klines_1m)
                        )
                        
                        self.classifications.append(enhanced_classification)
                        self.capture_stats["classifications_made"] += 1
                        self.capture_stats["ml_predictions_made"] += 1
                        
                        # Log clasificación ML ENHANCED
                        await self._log_ml_enhanced_classification(enhanced_classification)
                    
                    elif base_classification:
                        # Fallback sin ML
                        ml_enhanced = self.ml_enhancement._convert_to_ml_enhanced(base_classification)
                        self.classifications.append(ml_enhanced)
                        self.capture_stats["classifications_made"] += 1
                        
                        self.logger.info(f"[CLASSIFY] Base: 1m:{ml_enhanced.trend_1m} 5m:{ml_enhanced.trend_5m} "
                                       f"Phase: {ml_enhanced.market_phase} Risk: {ml_enhanced.risk_score:.0f}/100")
                
            except Exception as e:
                self.logger.error(f"[ML-CLASSIFY] Error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _log_ml_enhanced_classification(self, classification):
        """Log clasificación ML enhanced con detalles"""
        
        try:
            # Log básico
            self.logger.info(f"[ML-CLASSIFY] Trends: 1m:{classification.trend_1m} 5m:{classification.trend_5m} 1h:{classification.trend_1h}")
            self.logger.info(f"[ML-CLASSIFY] Phase: {classification.market_phase} | Regime: {classification.regime_type}")
            self.logger.info(f"[ML-CLASSIFY] ML Setup: {classification.ml_setup_type} (Conf: {classification.ml_confidence:.2f})")
            
            # Log ML predictions
            if classification.ml_breakout_probability > 0.7:
                self.logger.warning(f"[ML-PREDICT] 🚀 HIGH BREAKOUT PROBABILITY: {classification.ml_breakout_probability:.2f}")
            elif classification.ml_lateral_probability > 0.7:
                self.logger.info(f"[ML-PREDICT] ↔️ Strong lateral pattern: {classification.ml_lateral_probability:.2f}")
            
            # Log setup scores
            setup_scores = [
                f"Coiling: {classification.coiling_breakout_score:.0f}",
                f"Whale: {classification.whale_accumulation_score:.0f}",
                f"Lateral: {classification.lateral_compression_score:.0f}"
            ]
            self.logger.info(f"[ML-SETUPS] {' | '.join(setup_scores)}")
            
            # Log whale sentiment
            if classification.ml_whale_sentiment != 'neutral':
                self.logger.info(f"[ML-WHALE] 🐋 Sentiment: {classification.ml_whale_sentiment} | "
                               f"OB Sentiment: {classification.whale_sentiment}")
            
            # Log risk adjustment
            if classification.ml_risk_adjustment > 0.3:
                self.logger.warning(f"[ML-RISK] ⚠️ High risk adjustment: {classification.ml_risk_adjustment:.2f} | "
                                  f"False breakout risk: {classification.false_breakout_risk:.0f}/100")
            
            # Log volatility regime
            if classification.volatility_regime in ['high', 'extreme']:
                self.logger.warning(f"[ML-VOL] 📊 {classification.volatility_regime.upper()} volatility regime detected")
            
        except Exception as e:
            self.logger.error(f"[ML-CLASSIFY] Error logging: {str(e)}")
    
    async def _create_base_classification(self):
        """Crear clasificación base (sin ML)"""
        
        try:
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            
            # Extraer datos recientes
            recent_data = list(self.realtime_data)[-360:]  # Últimos 30 minutos
            prices = [d.price for d in recent_data]
            
            # Multi-timeframe trends básicos
            trend_1m = self._classify_basic_trend(prices[-12:], "1m")
            trend_5m = self._classify_basic_trend(prices[-60:] if len(prices) >= 60 else prices, "5m")
            trend_15m = self._classify_basic_trend(prices[-180:] if len(prices) >= 180 else prices, "15m")
            trend_1h = self._classify_basic_trend(prices, "1h")
            
            # Volatility básica
            volatility_1m = self._calculate_basic_volatility(prices[-12:])
            volatility_5m = self._calculate_basic_volatility(prices[-60:] if len(prices) >= 60 else prices)
            volatility_realized = self._calculate_basic_realized_volatility(prices)
            volatility_regime = self._classify_basic_volatility_regime(volatility_realized)
            
            # Market phase básica
            market_phase = self._determine_basic_market_phase(recent_data)
            regime_type = self._classify_basic_regime_type(recent_data)
            
            # Volume analysis básica
            volume_profile = self._analyze_basic_volume_profile(recent_data)
            volume_strength = self._calculate_basic_volume_strength(recent_data)
            volume_anomaly = self._detect_basic_volume_anomaly(recent_data)
            
            # OrderBook intelligence básica
            orderbook_pressure = self._analyze_basic_orderbook_pressure()
            whale_sentiment = self._analyze_basic_whale_sentiment()
            liquidity_depth = self._analyze_basic_liquidity_depth()
            
            # Momentum básico
            momentum_1m = self._calculate_basic_momentum(prices[-12:])
            momentum_5m = self._calculate_basic_momentum(prices[-60:] if len(prices) >= 60 else prices)
            momentum_strength = self._classify_basic_momentum_strength(momentum_1m, momentum_5m)
            
            # Support/Resistance básicos
            support_levels = self._calculate_basic_support_levels(prices)
            resistance_levels = self._calculate_basic_resistance_levels(prices)
            key_level_proximity = self._calculate_basic_key_level_proximity(prices[-1], support_levels + resistance_levels)
            
            # Risk metrics básicos
            risk_score = self._calculate_basic_risk_score(recent_data)
            volatility_risk = self._classify_basic_volatility_risk(volatility_realized)
            liquidity_risk = self._classify_basic_liquidity_risk()
            
            # Crear clasificación base
            from dataclasses import dataclass
            
            @dataclass
            class BaseClassification:
                timestamp: datetime.datetime
                trend_1m: str
                trend_5m: str
                trend_15m: str
                trend_1h: str
                volatility_1m: float
                volatility_5m: float
                volatility_realized: float
                volatility_regime: str
                market_phase: str
                regime_type: str
                volume_profile: str
                volume_strength: float
                volume_anomaly: bool
                orderbook_pressure: str
                whale_sentiment: str
                liquidity_depth: str
                momentum_1m: float
                momentum_5m: float
                momentum_strength: str
                support_levels: List[float]
                resistance_levels: List[float]
                key_level_proximity: float
                risk_score: float
                volatility_risk: str
                liquidity_risk: str
            
            return BaseClassification(
                timestamp=timestamp,
                trend_1m=trend_1m,
                trend_5m=trend_5m,
                trend_15m=trend_15m,
                trend_1h=trend_1h,
                volatility_1m=volatility_1m,
                volatility_5m=volatility_5m,
                volatility_realized=volatility_realized,
                volatility_regime=volatility_regime,
                market_phase=market_phase,
                regime_type=regime_type,
                volume_profile=volume_profile,
                volume_strength=volume_strength,
                volume_anomaly=volume_anomaly,
                orderbook_pressure=orderbook_pressure,
                whale_sentiment=whale_sentiment,
                liquidity_depth=liquidity_depth,
                momentum_1m=momentum_1m,
                momentum_5m=momentum_5m,
                momentum_strength=momentum_strength,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                key_level_proximity=key_level_proximity,
                risk_score=risk_score,
                volatility_risk=volatility_risk,
                liquidity_risk=liquidity_risk
            )
            
        except Exception as e:
            self.logger.error(f"[CLASSIFY] Error creating base classification: {str(e)}")
            return None
    
    # Métodos básicos simplificados
    def _classify_basic_trend(self, prices: List[float], timeframe: str) -> str:
        if len(prices) < 3:
            return "lateral"
        try:
            change_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
            threshold = 0.1 if timeframe == "1m" else 0.05
            if change_pct > threshold:
                return "bullish"
            elif change_pct < -threshold:
                return "bearish"
            else:
                return "lateral"
        except:
            return "lateral"
    
    def _calculate_basic_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        try:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = np.std(returns) * 100 * np.sqrt(len(returns))
            return round(volatility, 4)
        except:
            return 0.0
    
    def _calculate_basic_realized_volatility(self, prices: List[float]) -> float:
        if len(prices) < 10:
            return 0.0
        try:
            returns = np.diff(np.log(prices))
            realized_vol = np.sqrt(np.sum(returns**2)) * 100
            return round(realized_vol, 4)
        except:
            return 0.0
    
    def _classify_basic_volatility_regime(self, volatility: float) -> str:
        if volatility < 1.0:
            return "low"
        elif volatility < 2.5:
            return "normal"
        elif volatility < 5.0:
            return "high"
        else:
            return "extreme"
    
    def _determine_basic_market_phase(self, recent_data: List[ProfessionalDataPoint]) -> str:
        if len(recent_data) < 20:
            return "lateral"
        try:
            prices = [d.price for d in recent_data]
            volumes = [d.volume for d in recent_data]
            price_trend = self._classify_basic_trend(prices, "5m")
            
            if price_trend == "lateral":
                return "consolidation"
            elif np.mean(volumes[-10:]) > np.mean(volumes[:-10]):
                return "trending"
            else:
                return "distribution"
        except:
            return "lateral"
    
    def _classify_basic_regime_type(self, recent_data: List[ProfessionalDataPoint]) -> str:
        if len(recent_data) < 10:
            return "ranging"
        try:
            prices = [d.price for d in recent_data]
            volatility = self._calculate_basic_volatility(prices)
            if volatility > 3.0:
                return "breakout"
            elif volatility < 1.0:
                return "consolidation"
            else:
                return "trending"
        except:
            return "ranging"
    
    def _analyze_basic_volume_profile(self, recent_data: List[ProfessionalDataPoint]) -> str:
        if len(recent_data) < 20:
            return "neutral"
        try:
            volumes = [d.volume for d in recent_data]
            if np.mean(volumes[-10:]) > np.mean(volumes[:-10]):
                return "accumulation"
            else:
                return "distribution"
        except:
            return "neutral"
    
    def _calculate_basic_volume_strength(self, recent_data: List[ProfessionalDataPoint]) -> float:
        if len(recent_data) < 10:
            return 50.0
        try:
            volumes = [d.volume for d in recent_data]
            avg_volume = np.mean(volumes)
            recent_volume = np.mean(volumes[-5:])
            strength = (recent_volume / avg_volume) * 50
            return min(100.0, max(0.0, strength))
        except:
            return 50.0
    
    def _detect_basic_volume_anomaly(self, recent_data: List[ProfessionalDataPoint]) -> bool:
        if len(recent_data) < 20:
            return False
        try:
            volumes = [d.volume for d in recent_data]
            avg_volume = np.mean(volumes[:-5])
            std_volume = np.std(volumes[:-5])
            current_volume = recent_data[-1].volume
            return current_volume > avg_volume + 2 * std_volume
        except:
            return False
    
    def _analyze_basic_orderbook_pressure(self) -> str:
        if not self.orderbook_snapshots:
            return "neutral"
        try:
            recent_snapshots = list(self.orderbook_snapshots)[-10:]
            imbalances = [s['orderbook_imbalance'] for s in recent_snapshots]
            avg_imbalance = np.mean(imbalances)
            
            if avg_imbalance > 0.1:
                return "bullish"
            elif avg_imbalance < -0.1:
                return "bearish"
            else:
                return "neutral"
        except:
            return "neutral"
    
    def _analyze_basic_whale_sentiment(self) -> str:
        if not self.orderbook_snapshots:
            return "neutral"
        try:
            recent_snapshots = list(self.orderbook_snapshots)[-5:]
            whale_bid_ratios = []
            
            for snapshot in recent_snapshots:
                total_whale = snapshot['whale_bids'] + snapshot['whale_asks']
                if total_whale > 0:
                    whale_bid_ratio = snapshot['whale_bids'] / total_whale
                    whale_bid_ratios.append(whale_bid_ratio)
            
            if whale_bid_ratios:
                avg_whale_bid_ratio = np.mean(whale_bid_ratios)
                if avg_whale_bid_ratio > 0.6:
                    return "bullish"
                elif avg_whale_bid_ratio < 0.4:
                    return "bearish"
            
            return "neutral"
        except:
            return "neutral"
    
    def _analyze_basic_liquidity_depth(self) -> str:
        if not self.orderbook_snapshots:
            return "normal"
        try:
            latest_snapshot = self.orderbook_snapshots[-1]
            total_volume = latest_snapshot['total_bid_volume'] + latest_snapshot['total_ask_volume']
            
            if total_volume > 1000:
                return "thick"
            elif total_volume < 200:
                return "thin"
            else:
                return "normal"
        except:
            return "normal"
    
    def _calculate_basic_momentum(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        try:
            momentum = ((prices[-1] - prices[0]) / prices[0]) * 100
            return round(momentum, 4)
        except:
            return 0.0
    
    def _classify_basic_momentum_strength(self, momentum_1m: float, momentum_5m: float) -> str:
        try:
            combined_momentum = abs(momentum_1m) + abs(momentum_5m)
            if combined_momentum > 1.0:
                return "strong"
            elif combined_momentum > 0.3:
                return "moderate"
            else:
                return "weak"
        except:
            return "weak"
    
    def _calculate_basic_support_levels(self, prices: List[float]) -> List[float]:
        if len(prices) < 20:
            return []
        try:
            support_levels = [np.percentile(prices, 10), np.percentile(prices, 25)]
            return [round(level, 2) for level in support_levels]
        except:
            return []
    
    def _calculate_basic_resistance_levels(self, prices: List[float]) -> List[float]:
        if len(prices) < 20:
            return []
        try:
            resistance_levels = [np.percentile(prices, 75), np.percentile(prices, 90)]
            return [round(level, 2) for level in resistance_levels]
        except:
            return []
    
    def _calculate_basic_key_level_proximity(self, current_price: float, levels: List[float]) -> float:
        if not levels:
            return 50.0
        try:
            distances = [abs(current_price - level) / current_price * 100 for level in levels]
            min_distance = min(distances)
            proximity = max(0, 100 - min_distance * 50)
            return round(proximity, 1)
        except:
            return 50.0
    
    def _calculate_basic_risk_score(self, recent_data: List[ProfessionalDataPoint]) -> float:
        if len(recent_data) < 10:
            return 50.0
        try:
            prices = [d.price for d in recent_data]
            volatility = self._calculate_basic_volatility(prices)
            vol_risk = min(100, volatility * 20)
            return round(vol_risk, 1)
        except:
            return 50.0
    
    def _classify_basic_volatility_risk(self, volatility: float) -> str:
        if volatility > 5.0:
            return "extreme"
        elif volatility > 2.5:
            return "high"
        elif volatility > 1.0:
            return "moderate"
        else:
            return "low"
    
    def _classify_basic_liquidity_risk(self) -> str:
        liquidity_depth = self._analyze_basic_liquidity_depth()
        risk_mapping = {"thin": "high", "normal": "moderate", "thick": "low"}
        return risk_mapping.get(liquidity_depth, "moderate")
    
    # ========== ML MONITORING LOOP ==========
    
    async def _ml_monitoring_loop(self):
        """Loop de monitoreo ML específico"""
        
        self.logger.info("[ML-MONITOR] Iniciando monitoreo ML")
        
        while self.running:
            try:
                await asyncio.sleep(300)  # Cada 5 minutos
                
                if self.config["ml_enabled"]:
                    await self._log_ml_performance()
                    await self._save_ml_predictions()
                
            except Exception as e:
                self.logger.error(f"[ML-MONITOR] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _log_ml_performance(self):
        """Log performance del ML"""
        
        try:
            if hasattr(self.ml_enhancement, 'model_performance'):
                performance = self.ml_enhancement.model_performance
                if performance:
                    models_status = ", ".join([f"{model}:{status}" for model, status in performance.items()])
                    self.logger.info(f"[ML-PERF] Models: {models_status}")
            
            # Log training data size
            training_size = len(self.ml_enhancement.feature_history)
            if training_size > 0:
                self.logger.info(f"[ML-DATA] Training samples: {training_size}/10000")
                
                # Log when ready for retraining
                if training_size % self.ml_enhancement.retrain_interval == 0:
                    self.logger.info(f"[ML-TRAIN] Ready for retraining - {training_size} samples collected")
        
        except Exception as e:
            self.logger.error(f"[ML-PERF] Error logging performance: {str(e)}")
    
    async def _save_ml_predictions(self):
        """Guardar predicciones ML recientes"""
        
        try:
            if not self.classifications:
                return
            
            recent_classifications = list(self.classifications)[-12:]  # Última hora
            ml_data = []
            
            for classification in recent_classifications:
                if hasattr(classification, 'ml_confidence'):
                    ml_record = {
                        'timestamp': classification.timestamp.isoformat(),
                        'ml_setup_type': classification.ml_setup_type,
                        'ml_confidence': classification.ml_confidence,
                        'ml_breakout_probability': classification.ml_breakout_probability,
                        'ml_lateral_probability': classification.ml_lateral_probability,
                        'ml_whale_sentiment': classification.ml_whale_sentiment,
                        'coiling_breakout_score': classification.coiling_breakout_score,
                        'whale_accumulation_score': classification.whale_accumulation_score,
                        'lateral_compression_score': classification.lateral_compression_score,
                        'false_breakout_risk': classification.false_breakout_risk,
                        'ml_risk_adjustment': classification.ml_risk_adjustment
                    }
                    ml_data.append(ml_record)
            
            if ml_data:
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ml_file = f"{self.config['data_directory']}/ml_predictions/ml_predictions_{self.symbol}_{timestamp_str}.json"
                
                with open(ml_file, 'w', encoding='utf-8') as f:
                    json.dump(ml_data, f, indent=2, ensure_ascii=False)
                
                self.logger.debug(f"[ML-SAVE] ML predictions saved: {len(ml_data)} records")
        
        except Exception as e:
            self.logger.error(f"[ML-SAVE] Error saving ML predictions: {str(e)}")
    
    # ========== OTROS LOOPS (usar versiones anteriores) ==========
    
    async def _whale_detection_loop(self):
        """Loop de detección de whales"""
        self.logger.info("[WHALE] Iniciando detección de whales con ML")
        while self.running:
            try:
                await asyncio.sleep(30)
                if self.orderbook_snapshots and len(self.orderbook_snapshots) >= 3:
                    recent = list(self.orderbook_snapshots)[-3:]
                    if all(s['large_orders_count'] > 0 for s in recent):
                        self.logger.info("[WHALE] 🐋 Sustained whale activity detected")
                        
                        # ML enhancement for whale detection
                        if self.config["ml_enabled"] and self.classifications:
                            last_classification = self.classifications[-1]
                            if (hasattr(last_classification, 'whale_accumulation_score') and 
                                last_classification.whale_accumulation_score > 60):
                                self.logger.warning("[ML-WHALE] 🐋🧠 ML predicts whale accumulation pattern!")
                
            except Exception as e:
                self.logger.error(f"[WHALE] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _technical_analysis_loop(self):
        """Loop de análisis técnico con ML enhancement"""
        self.logger.info("[TECHNICAL] Iniciando análisis técnico con ML")
        while self.running:
            try:
                await asyncio.sleep(300)
                if len(self.klines_1m) >= 20:
                    klines = list(self.klines_1m)[-20:]
                    closes = [k['close'] for k in klines]
                    if len(closes) >= 14:
                        # Simple RSI calculation
                        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                        gains = [d if d > 0 else 0 for d in deltas]
                        losses = [-d if d < 0 else 0 for d in deltas]
                        avg_gain = np.mean(gains[-14:])
                        avg_loss = np.mean(losses[-14:])
                        if avg_loss > 0:
                            rs = avg_gain / avg_loss
                            rsi = 100 - (100 / (1 + rs))
                            
                            # ML Enhanced RSI interpretation
                            if self.config["ml_enabled"] and self.classifications:
                                last_classification = self.classifications[-1]
                                if hasattr(last_classification, 'ml_breakout_probability'):
                                    if rsi < 30 and last_classification.ml_breakout_probability > 0.7:
                                        self.logger.warning(f"[ML-TECHNICAL] 🚀 RSI oversold + ML breakout signal: RSI={rsi:.1f}, Breakout={last_classification.ml_breakout_probability:.2f}")
                                    elif rsi > 70 and last_classification.ml_lateral_probability > 0.7:
                                        self.logger.info(f"[ML-TECHNICAL] 📊 RSI overbought in lateral pattern: RSI={rsi:.1f}")
                            
                            if rsi < 30:
                                self.logger.info(f"[TECHNICAL] RSI oversold: {rsi:.1f}")
                            elif rsi > 70:
                                self.logger.info(f"[TECHNICAL] RSI overbought: {rsi:.1f}")
            except Exception as e:
                self.logger.error(f"[TECHNICAL] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _risk_analysis_loop(self):
        """Loop de análisis de riesgo con ML enhancement"""
        self.logger.info("[RISK] Iniciando análisis de riesgo con ML")
        while self.running:
            try:
                await asyncio.sleep(120)
                if len(self.realtime_data) >= 20:
                    recent_data = list(self.realtime_data)[-20:]
                    prices = [d.price for d in recent_data]
                    volatility = self._calculate_basic_volatility(prices)
                    
                    # ML Enhanced risk analysis
                    if self.config["ml_enabled"] and self.classifications:
                        last_classification = self.classifications[-1]
                        if hasattr(last_classification, 'ml_risk_adjustment'):
                            adjusted_risk = volatility * (1 + last_classification.ml_risk_adjustment)
                            
                            if last_classification.false_breakout_risk > 70:
                                self.logger.warning(f"[ML-RISK] ⚠️ High false breakout risk: {last_classification.false_breakout_risk:.0f}/100")
                            
                            if adjusted_risk > 5.0:
                                self.logger.warning(f"[ML-RISK] High volatility (ML adjusted): {adjusted_risk:.2f}%")
                    
                    elif volatility > 5.0:
                        self.logger.warning(f"[RISK] High volatility: {volatility:.2f}%")
                        
            except Exception as e:
                self.logger.error(f"[RISK] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _data_persistence_loop(self):
        """Loop de persistencia con datos ML"""
        self.logger.info("[PERSIST] Iniciando persistencia profesional con ML")
        while self.running:
            try:
                await asyncio.sleep(self.config["save_interval_seconds"])
                await self._save_professional_data_ml()
            except Exception as e:
                self.logger.error(f"[PERSIST] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _save_professional_data_ml(self):
        """Guardar datos profesionales con ML"""
        try:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []
            
            # 1. Raw professional data
            if self.realtime_data:
                data_file = f"{self.config['data_directory']}/realtime/professional_ml_data_{self.symbol}_{timestamp_str}.json"
                recent_data = list(self.realtime_data)[-60:]
                
                data_to_save = []
                for point in recent_data:
                    data_dict = asdict(point)
                    data_dict['timestamp'] = point.timestamp.isoformat()
                    data_to_save.append(data_dict)
                
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                
                saved_files.append(f"realtime: {len(data_to_save)} points")
            
            # 2. ML Enhanced Classifications
            if self.classifications:
                class_file = f"{self.config['data_directory']}/classifications/ml_enhanced_classifications_{self.symbol}_{timestamp_str}.json"
                recent_classifications = list(self.classifications)[-30:]
                
                class_data = []
                for classification in recent_classifications:
                    data_dict = asdict(classification)
                    data_dict['timestamp'] = classification.timestamp.isoformat()
                    class_data.append(data_dict)
                
                with open(class_file, 'w', encoding='utf-8') as f:
                    json.dump(class_data, f, indent=2, ensure_ascii=False)
                
                saved_files.append(f"ml_classifications: {len(class_data)} entries")
            
            # 3. OrderBook snapshots
            if self.orderbook_snapshots:
                ob_file = f"{self.config['data_directory']}/orderbook/orderbook_ml_{self.symbol}_{timestamp_str}.json"
                recent_obs = list(self.orderbook_snapshots)[-30:]
                
                ob_data = []
                for snapshot in recent_obs:
                    snapshot_copy = snapshot.copy()
                    snapshot_copy['timestamp'] = snapshot['timestamp'].isoformat()
                    ob_data.append(snapshot_copy)
                
                with open(ob_file, 'w', encoding='utf-8') as f:
                    json.dump(ob_data, f, indent=2, ensure_ascii=False)
                
                saved_files.append(f"orderbook: {len(ob_data)} snapshots")
            
            if saved_files:
                self.logger.info(f"[SAVE] Professional ML data saved: {', '.join(saved_files)}")
            
        except Exception as e:
            self.logger.error(f"[SAVE] Error: {str(e)}")
    
    async def _connection_monitor_loop(self):
        """Monitoreo de conexión"""
        while self.running:
            try:
                await asyncio.sleep(self.config["heartbeat_interval"])
                connected_ratio = len(self.connections) / len(self.websocket_urls)
                self.capture_stats["data_quality_score"] = connected_ratio * 100
            except Exception as e:
                self.logger.error(f"[MONITOR] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _stats_monitoring_loop(self):
        """Stats monitoring con ML metrics"""
        while self.running:
            try:
                await asyncio.sleep(600)
                
                # Basic stats
                self.logger.info(f"[STATS] Professional ML: Points: {self.capture_stats['ticker_points']}, "
                               f"OrderBook: {self.capture_stats['orderbook_updates']}, "
                               f"Klines: {self.capture_stats['klines_received']}, "
                               f"Whales: {self.capture_stats['whale_orders_detected']}")
                
                # ML specific stats
                if self.config["ml_enabled"]:
                    self.logger.info(f"[ML-STATS] ML Predictions: {self.capture_stats['ml_predictions_made']}, "
                                   f"Classifications: {self.capture_stats['classifications_made']}, "
                                   f"Training samples: {len(self.ml_enhancement.feature_history)}")
                
                self.logger.info(f"[STATS] Quality: {self.capture_stats['data_quality_score']:.1f}/100, "
                               f"Streams: {len(self.connections)}/{len(self.websocket_urls)}")
                
            except Exception as e:
                self.logger.error(f"[STATS] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup(self):
        """Cleanup profesional con ML"""
        try:
            self.logger.info("[CLEANUP] Iniciando cleanup profesional con ML...")
            
            # Guardar modelos ML antes de cerrar
            if self.config["ml_enabled"]:
                await self._save_ml_models()
            
            # Cerrar conexiones
            for stream_name, connection in self.connections.items():
                try:
                    await connection.close()
                except:
                    pass
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("[CLEANUP] ✅ Cleanup completado")
            
        except Exception as e:
            self.logger.error(f"[CLEANUP] Error: {str(e)}")
    
    async def _save_ml_models(self):
        """Guardar modelos ML entrenados"""
        
        try:
            if not self.ml_enhancement.learning_enabled:
                return
            
            models_dir = f"{self.config['data_directory']}/ml_models"
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Guardar modelos si están entrenados
            models_saved = []
            
            if (self.ml_enhancement.breakout_classifier and 
                hasattr(self.ml_enhancement.breakout_classifier, 'classes_')):
                
                breakout_file = f"{models_dir}/breakout_classifier_{timestamp_str}.pkl"
                with open(breakout_file, 'wb') as f:
                    pickle.dump(self.ml_enhancement.breakout_classifier, f)
                models_saved.append("breakout")
            
            if (self.ml_enhancement.lateral_classifier and 
                hasattr(self.ml_enhancement.lateral_classifier, 'classes_')):
                
                lateral_file = f"{models_dir}/lateral_classifier_{timestamp_str}.pkl"
                with open(lateral_file, 'wb') as f:
                    pickle.dump(self.ml_enhancement.lateral_classifier, f)
                models_saved.append("lateral")
            
            if (self.ml_enhancement.whale_behavior_model and 
                hasattr(self.ml_enhancement.whale_behavior_model, 'classes_')):
                
                whale_file = f"{models_dir}/whale_model_{timestamp_str}.pkl"
                with open(whale_file, 'wb') as f:
                    pickle.dump(self.ml_enhancement.whale_behavior_model, f)
                models_saved.append("whale")
            
            # Guardar scaler
            if hasattr(self.ml_enhancement.feature_scaler, 'mean_'):
                scaler_file = f"{models_dir}/feature_scaler_{timestamp_str}.pkl"
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.ml_enhancement.feature_scaler, f)
                models_saved.append("scaler")
            
            if models_saved:
                self.logger.info(f"[ML-SAVE] Modelos guardados: {', '.join(models_saved)}")
            
        except Exception as e:
            self.logger.error(f"[ML-SAVE] Error guardando modelos: {str(e)}")
    
    def get_professional_status(self) -> Dict:
        """Obtener estado profesional con ML"""
        
        # Estado básico
        last_data = None
        if self.realtime_data:
            last_point = self.realtime_data[-1]
            last_data = {
                "timestamp": last_point.timestamp.isoformat(),
                "price": last_point.price,
                "real_spread": last_point.real_spread,
                "orderbook_imbalance": last_point.orderbook_imbalance,
                "whale_activity_score": last_point.whale_activity_score,
                "large_orders_count": last_point.large_orders_count,
                "volume": last_point.volume
            }
        
        # ML Enhanced classification
        last_classification = None
        if self.classifications:
            last_class = self.classifications[-1]
            last_classification = {
                "timestamp": last_class.timestamp.isoformat(),
                "trend_1m": last_class.trend_1m,
                "trend_5m": last_class.trend_5m,
                "trend_1h": last_class.trend_1h,
                "market_phase": last_class.market_phase,
                "ml_setup_type": getattr(last_class, 'ml_setup_type', 'unknown'),
                "ml_confidence": getattr(last_class, 'ml_confidence', 0.5),
                "ml_breakout_probability": getattr(last_class, 'ml_breakout_probability', 0.5),
                "ml_whale_sentiment": getattr(last_class, 'ml_whale_sentiment', 'neutral'),
                "coiling_breakout_score": getattr(last_class, 'coiling_breakout_score', 0),
                "whale_accumulation_score": getattr(last_class, 'whale_accumulation_score', 0),
                "false_breakout_risk": getattr(last_class, 'false_breakout_risk', 50),
                "risk_score": last_class.risk_score,
                "volatility_regime": last_class.volatility_regime
            }
        
        # Connection status
        connection_status = {
            "active_streams": len(self.connections),
            "total_streams": len(self.websocket_urls),
            "connected_streams": list(self.connections.keys()),
            "disconnected_streams": [stream for stream in self.websocket_urls.keys() if stream not in self.connections],
            "data_quality_score": self.capture_stats["data_quality_score"]
        }
        
        # ML status
        ml_status = {
            "ml_enabled": self.config["ml_enabled"],
            "learning_enabled": self.ml_enhancement.learning_enabled if hasattr(self.ml_enhancement, 'learning_enabled') else False,
            "training_samples": len(self.ml_enhancement.feature_history) if hasattr(self.ml_enhancement, 'feature_history') else 0,
            "ml_predictions_made": self.capture_stats.get("ml_predictions_made", 0)
        }
        
        return {
            "running": self.running,
            "data_source": self.data_source,
            "capture_stats": self.capture_stats,
            "connection_status": connection_status,
            "ml_status": ml_status,
            "buffer_sizes": {
                "realtime_data": len(self.realtime_data),
                "orderbook_snapshots": len(self.orderbook_snapshots),
                "klines_1m": len(self.klines_1m),
                "klines_5m": len(self.klines_5m),
                "klines_15m": len(self.klines_15m),
                "classifications": len(self.classifications)
            },
            "last_data_point": last_data,
            "last_ml_classification": last_classification
        }
    
    def stop(self):
        """Detener captura profesional"""
        self.logger.info("[STOP] Deteniendo captura profesional con ML...")
        self.running = False

# ========== TEST FUNCTION ==========

async def test_professional_ml_capture():
    """Test del sistema profesional con ML"""
    
    print("🚀 TESTING PROFESSIONAL CONTINUOUS DATA CAPTURE - ML INTEGRATED")
    print(f"Current Time: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Current User: biGGzeta")
    print("=" * 80)
    
    capture = ProfessionalDataCapture("ETHUSD_PERP")
    
    # Ejecutar por 3 minutos como test
    try:
        await asyncio.wait_for(capture.start_professional_capture(), timeout=180)
    except asyncio.TimeoutError:
        print("✅ Test completado - sistema profesional con ML funcionando")
        
        # Mostrar estadísticas finales
        status = capture.get_professional_status()
        print(f"\n📊 RESULTADOS PROFESIONALES CON ML:")
        print(f"   Data source: {status['data_source']}")
        print(f"   ML enabled: {status['ml_status']['ml_enabled']}")
        print(f"   Active streams: {status['connection_status']['active_streams']}/{status['connection_status']['total_streams']}")
        print(f"   Data quality: {status['connection_status']['data_quality_score']:.1f}/100")
        print(f"   ML predictions made: {status['ml_status']['ml_predictions_made']}")
        print(f"   Training samples: {status['ml_status']['training_samples']}")
        print(f"   Connected streams: {status['connection_status']['connected_streams']}")
        print(f"   Realtime points: {status['buffer_sizes']['realtime_data']}")
        print(f"   OrderBook updates: {status['buffer_sizes']['orderbook_snapshots']}")
        print(f"   ML Classifications: {status['buffer_sizes']['classifications']}")
        print(f"   Whale orders detected: {status['capture_stats']['whale_orders_detected']}")
        
        if status['last_data_point']:
            last_data = status['last_data_point']
            print(f"   ETH profesional: ${last_data['price']:.2f}")
            print(f"   Real spread: ${last_data['real_spread']:.2f}")
            print(f"   OrderBook imbalance: {last_data['orderbook_imbalance']:+.3f}")
            print(f"   Whale activity: {last_data['whale_activity_score']:.3f}")
            print(f"   Large orders: {last_data['large_orders_count']}")
        
        if status['last_ml_classification']:
            last_class = status['last_ml_classification']
            print(f"   ML Setup Type: {last_class['ml_setup_type']}")
            print(f"   ML Confidence: {last_class['ml_confidence']:.2f}")
            print(f"   Breakout Probability: {last_class['ml_breakout_probability']:.2f}")
            print(f"   Whale Sentiment ML: {last_class['ml_whale_sentiment']}")
            print(f"   Coiling Score: {last_class['coiling_breakout_score']:.0f}")
            print(f"   Whale Accumulation: {last_class['whale_accumulation_score']:.0f}")
            print(f"   False Breakout Risk: {last_class['false_breakout_risk']:.0f}/100")
            print(f"   Multi-trend: 1m:{last_class['trend_1m']} 5m:{last_class['trend_5m']} 1h:{last_class['trend_1h']}")
            print(f"   Market phase: {last_class['market_phase']}")
            print(f"   Risk score: {last_class['risk_score']}/100")
        
        capture.stop()
        return True

if __name__ == "__main__":
    asyncio.run(test_professional_ml_capture())
                                                  
