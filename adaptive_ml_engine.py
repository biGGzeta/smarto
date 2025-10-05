#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive ML Engine for OrderBook Intelligence - PRODUCTION READY
Sistema ML que aprende y clasifica patrones adaptativamente
SAFE INTEGRATION: No rompe sistema existente
Autor: biGGzeta
Fecha: 2025-10-05 14:03:57 UTC
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import datetime
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MLFeatureSet:
    """Set de features para ML classification"""
    timestamp: datetime.datetime
    
    # OrderBook features
    spread_pct: float
    bid_ask_imbalance: float
    depth_ratio: float
    whale_activity_score: float
    
    # Volume features  
    volume_ratio: float
    trade_intensity: float
    avg_order_size_ratio: float
    
    # Price action features
    price_change_1m: float
    price_change_5m: float
    volatility_1m: float
    
    # Market condition features
    market_regime: str
    time_of_day: int
    
    # Target variables (for training)
    future_move_1m: Optional[float] = None
    future_move_5m: Optional[float] = None
    disruption_occurred: Optional[bool] = None
    signal_success: Optional[bool] = None

class AdaptiveMLEngine:
    """Motor ML adaptativo para OrderBook Intelligence - SAFE INTEGRATION"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP", enable_ml: bool = True):
        self.symbol = symbol
        self.enabled = enable_ml  # SAFETY SWITCH
        self.ml_dir = "ml_models/orderbook"
        os.makedirs(self.ml_dir, exist_ok=True)
        
        # SAFETY: Solo inicializar ML si está habilitado
        if not self.enabled:
            print(f"🤖 ML Engine creado pero DISABLED para {symbol}")
            return
        
        # ML Models (solo si enabled)
        try:
            self.disruption_classifier = RandomForestClassifier(
                n_estimators=50, max_depth=8, random_state=42, n_jobs=1
            )
            self.movement_predictor = RandomForestClassifier(
                n_estimators=50, max_depth=8, random_state=42, n_jobs=1
            )
            self.anomaly_detector = IsolationForest(
                contamination=0.1, random_state=42, n_jobs=1
            )
            self.feature_scaler = StandardScaler()
            
            # Training data storage
            self.feature_history = []
            self.prediction_accuracy = {}
            
            # Model state
            self.models_trained = False
            self.last_retrain = None
            self.initialization_error = None
            
            # Configuration
            self.config = {
                "retrain_interval_hours": 24,
                "min_samples_for_training": 50,  # Reducido para faster training
                "prediction_confidence_threshold": 0.6,
                "max_training_samples": 1000,  # Límite para memory
                "fallback_on_error": True
            }
            
            print(f"🤖 Adaptive ML Engine iniciado para {symbol}")
            self._load_existing_models()
            
        except Exception as e:
            self.initialization_error = str(e)
            print(f"⚠️ ML Engine initialization error: {str(e)}")
            print("🛡️ Continuing without ML (graceful fallback)")
            self.enabled = False
    
    def is_ready(self) -> bool:
        """Check si ML engine está listo para usar"""
        return self.enabled and self.initialization_error is None
    
    def _safe_execute(self, func, *args, **kwargs):
        """Execute ML function with error handling"""
        if not self.is_ready():
            return {"status": "ml_disabled"}
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ ML Error in {func.__name__}: {str(e)}")
            if self.config["fallback_on_error"]:
                return {"status": "ml_error", "message": str(e)}
            else:
                raise
    
    def _load_existing_models(self):
        """Cargar modelos existentes si existen - SAFE LOADING"""
        
        if not self.is_ready():
            return
        
        model_files = {
            "disruption_classifier": f"{self.ml_dir}/disruption_classifier_{self.symbol}.joblib",
            "movement_predictor": f"{self.ml_dir}/movement_predictor_{self.symbol}.joblib", 
            "anomaly_detector": f"{self.ml_dir}/anomaly_detector_{self.symbol}.joblib",
            "feature_scaler": f"{self.ml_dir}/feature_scaler_{self.symbol}.joblib"
        }
        
        loaded_count = 0
        for model_name, file_path in model_files.items():
            if os.path.exists(file_path):
                try:
                    model = joblib.load(file_path)
                    setattr(self, model_name, model)
                    loaded_count += 1
                except Exception as e:
                    print(f"⚠️ Error cargando {model_name}: {str(e)}")
        
        if loaded_count >= 3:  # Need at least 3 models loaded
            print(f"✅ {loaded_count} modelos ML cargados desde disco")
            self.models_trained = True
        else:
            print("🔄 No hay modelos suficientes - entrenamiento fresh")
    
    def extract_features_safe(self, orderbook_snapshot, market_context: Dict = None) -> Optional[MLFeatureSet]:
        """Extraer features de manera segura"""
        
        def _extract():
            return self._extract_features_internal(orderbook_snapshot, market_context or {})
        
        result = self._safe_execute(_extract)
        return result if isinstance(result, MLFeatureSet) else None
    
    def _extract_features_internal(self, orderbook_snapshot, market_context: Dict) -> MLFeatureSet:
        """Internal feature extraction"""
        
        # Calcular features de orderbook de manera segura
        bids = getattr(orderbook_snapshot, 'bids', [])
        asks = getattr(orderbook_snapshot, 'asks', [])
        
        if not bids or not asks:
            return self._create_fallback_features(orderbook_snapshot)
        
        try:
            # OrderBook features
            best_bid, best_ask = bids[0][0], asks[0][0]
            price = getattr(orderbook_snapshot, 'price', (best_bid + best_ask) / 2)
            
            spread_pct = ((best_ask - best_bid) / price) * 100
            
            # Safe volume calculations
            total_bid_vol = sum([qty for _, qty in bids[:5]])
            total_ask_vol = sum([qty for _, qty in asks[:5]])
            total_vol = total_bid_vol + total_ask_vol
            
            bid_ask_imbalance = (total_bid_vol - total_ask_vol) / total_vol if total_vol > 0 else 0
            depth_ratio = total_bid_vol / total_ask_vol if total_ask_vol > 0 else 1.0
            
            # Whale activity score (safe)
            whale_threshold = 50000
            whale_bids = sum([qty for price_level, qty in bids if qty * price_level > whale_threshold])
            whale_asks = sum([qty for price_level, qty in asks if qty * price_level > whale_threshold])
            whale_activity_score = (whale_bids + whale_asks) / total_vol if total_vol > 0 else 0
            
            # Volume features (safe access)
            volume_takers = getattr(orderbook_snapshot, 'volume_takers', 1000)
            trades_count = getattr(orderbook_snapshot, 'trades_count', 50)
            avg_order_size = getattr(orderbook_snapshot, 'avg_order_size', 20)
            
            volume_ratio = volume_takers / 1000
            trade_intensity = trades_count / 50
            avg_order_size_ratio = avg_order_size / 20
            
            # Price action features (from context)
            price_change_1m = market_context.get("price_change_1m", 0.0)
            price_change_5m = market_context.get("price_change_5m", 0.0)
            volatility_1m = market_context.get("volatility_1m", 0.0)
            
            # Market condition features
            market_regime = market_context.get("market_regime", "unknown")
            timestamp = getattr(orderbook_snapshot, 'timestamp', datetime.datetime.now(datetime.timezone.utc))
            time_of_day = timestamp.hour
            
            return MLFeatureSet(
                timestamp=timestamp,
                spread_pct=spread_pct,
                bid_ask_imbalance=bid_ask_imbalance,
                depth_ratio=depth_ratio,
                whale_activity_score=whale_activity_score,
                volume_ratio=volume_ratio,
                trade_intensity=trade_intensity,
                avg_order_size_ratio=avg_order_size_ratio,
                price_change_1m=price_change_1m,
                price_change_5m=price_change_5m,
                volatility_1m=volatility_1m,
                market_regime=market_regime,
                time_of_day=time_of_day
            )
            
        except Exception as e:
            print(f"⚠️ Error en feature extraction: {str(e)}")
            return self._create_fallback_features(orderbook_snapshot)
    
    def _create_fallback_features(self, snapshot) -> MLFeatureSet:
        """Crear features básicos como fallback"""
        
        timestamp = getattr(snapshot, 'timestamp', datetime.datetime.now(datetime.timezone.utc))
        volume_takers = getattr(snapshot, 'volume_takers', 1000)
        trades_count = getattr(snapshot, 'trades_count', 50)
        avg_order_size = getattr(snapshot, 'avg_order_size', 20)
        
        return MLFeatureSet(
            timestamp=timestamp,
            spread_pct=0.1,
            bid_ask_imbalance=0.0,
            depth_ratio=1.0,
            whale_activity_score=0.0,
            volume_ratio=volume_takers / 1000,
            trade_intensity=trades_count / 50,
            avg_order_size_ratio=avg_order_size / 20,
            price_change_1m=0.0,
            price_change_5m=0.0,
            volatility_1m=0.0,
            market_regime="unknown",
            time_of_day=timestamp.hour
        )
    
    def features_to_array(self, features: MLFeatureSet) -> np.ndarray:
        """Convertir features a array numpy para ML"""
        
        # Encode categorical variables safely
        regime_encoding = {"unknown": 0, "lateral": 1, "trending": 2, "disruptive": 3}
        regime_encoded = regime_encoding.get(features.market_regime, 0)
        
        # Safe array creation with bounds checking
        array_values = [
            max(-10, min(10, features.spread_pct)),  # Bound spread
            max(-1, min(1, features.bid_ask_imbalance)),  # Bound imbalance
            max(0.1, min(10, features.depth_ratio)),  # Bound depth ratio
            max(0, min(1, features.whale_activity_score)),  # Bound whale activity
            max(0, min(10, features.volume_ratio)),  # Bound volume
            max(0, min(5, features.trade_intensity)),  # Bound trade intensity
            max(0, min(5, features.avg_order_size_ratio)),  # Bound order size
            max(-5, min(5, features.price_change_1m)),  # Bound price changes
            max(-5, min(5, features.price_change_5m)),
            max(0, min(5, features.volatility_1m)),  # Bound volatility
            regime_encoded,
            features.time_of_day / 24.0  # Normalize hour
        ]
        
        return np.array(array_values, dtype=np.float32)
    
    def predict_disruption_safe(self, features: MLFeatureSet) -> Dict:
        """Predecir probabilidad de disrupción - SAFE VERSION"""
        
        def _predict():
            if not self.models_trained:
                return {"status": "not_trained", "probability": 0.5}
            
            # Preparar features
            X = self.features_to_array(features).reshape(1, -1)
            X_scaled = self.feature_scaler.transform(X)
            
            # Predicción con error handling
            proba_result = self.disruption_classifier.predict_proba(X_scaled)
            
            if len(proba_result[0]) < 2:
                return {"status": "insufficient_classes", "probability": 0.5}
            
            disruption_prob = proba_result[0][1]  # Prob de clase 1 (disrupción)
            confidence = max(disruption_prob, 1 - disruption_prob)
            
            # Clasificación
            if disruption_prob > 0.7:
                prediction = "high_disruption_risk"
            elif disruption_prob > 0.3:
                prediction = "moderate_disruption_risk"
            else:
                prediction = "low_disruption_risk"
            
            return {
                "status": "success",
                "prediction": prediction,
                "probability": float(disruption_prob),
                "confidence": float(confidence),
                "timestamp": features.timestamp.isoformat()
            }
        
        return self._safe_execute(_predict)
    
    def predict_price_movement_safe(self, features: MLFeatureSet) -> Dict:
        """Predecir dirección del movimiento de precio - SAFE VERSION"""
        
        def _predict():
            if not self.models_trained:
                return {"status": "not_trained", "direction": "neutral"}
            
            # Preparar features
            X = self.features_to_array(features).reshape(1, -1)
            X_scaled = self.feature_scaler.transform(X)
            
            # Predicción de dirección
            direction_prob = self.movement_predictor.predict_proba(X_scaled)[0]
            direction_class = self.movement_predictor.predict(X_scaled)[0]
            
            directions = ["bearish", "neutral", "bullish"]
            predicted_direction = directions[min(direction_class, len(directions) - 1)]
            confidence = float(max(direction_prob))
            
            # Safe probability extraction
            probs = {
                "bearish": float(direction_prob[0]) if len(direction_prob) > 0 else 0.33,
                "neutral": float(direction_prob[1]) if len(direction_prob) > 1 else 0.33,
                "bullish": float(direction_prob[2]) if len(direction_prob) > 2 else 0.33
            }
            
            return {
                "status": "success",
                "direction": predicted_direction,
                "probabilities": probs,
                "confidence": confidence
            }
        
        return self._safe_execute(_predict)
    
    def detect_anomalies_safe(self, features: MLFeatureSet) -> Dict:
        """Detectar anomalías - SAFE VERSION"""
        
        def _detect():
            if not self.models_trained:
                return {"status": "not_trained", "is_anomaly": False}
            
            # Preparar features
            X = self.features_to_array(features).reshape(1, -1)
            X_scaled = self.feature_scaler.transform(X)
            
            # Detección de anomalías
            anomaly_score = self.anomaly_detector.decision_function(X_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(X_scaled)[0] == -1
            
            # Normalizar anomaly score a 0-1 de manera segura
            anomaly_intensity = max(0, min(1, (0.5 - float(anomaly_score)) * 2))
            
            return {
                "status": "success",
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(anomaly_score),
                "anomaly_intensity": anomaly_intensity,
                "interpretation": "unusual_market_conditions" if is_anomaly else "normal_conditions"
            }
        
        return self._safe_execute(_detect)
    
    def generate_ml_insights_safe(self, features: MLFeatureSet) -> Dict:
        """Generar insights ML comprehensivos - SAFE VERSION"""
        
        if not self.is_ready():
            return {
                "status": "ml_disabled",
                "timestamp": features.timestamp.isoformat(),
                "message": "ML Engine not available"
            }
        
        insights = {
            "timestamp": features.timestamp.isoformat(),
            "disruption_prediction": self.predict_disruption_safe(features),
            "movement_prediction": self.predict_price_movement_safe(features),
            "anomaly_detection": self.detect_anomalies_safe(features),
            "model_status": {
                "enabled": self.enabled,
                "trained": self.models_trained,
                "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
                "training_samples": len(self.feature_history),
                "initialization_error": self.initialization_error
            }
        }
        
        return insights
    
    def add_training_sample_safe(self, features: MLFeatureSet, outcomes: Dict):
        """Agregar muestra para entrenamiento - SAFE VERSION"""
        
        if not self.is_ready():
            return
        
        try:
            # Agregar labels de outcome
            features.future_move_1m = outcomes.get("price_change_1m", 0.0)
            features.future_move_5m = outcomes.get("price_change_5m", 0.0)
            features.disruption_occurred = outcomes.get("disruption_occurred", False)
            features.signal_success = outcomes.get("signal_success", False)
            
            # Almacenar para entrenamiento (con límite de memoria)
            self.feature_history.append(features)
            
            # Mantener solo las últimas N muestras
            if len(self.feature_history) > self.config["max_training_samples"]:
                self.feature_history = self.feature_history[-self.config["max_training_samples"]:]
            
            # Trigger retrain si es necesario
            if (len(self.feature_history) >= self.config["min_samples_for_training"] and 
                self._should_retrain()):
                self._retrain_models_safe()
                
        except Exception as e:
            print(f"⚠️ Error adding training sample: {str(e)}")
    
    def _should_retrain(self) -> bool:
        """Determinar si es momento de reentrenar modelos"""
        
        if not self.last_retrain:
            return True
        
        hours_since_retrain = (datetime.datetime.now(datetime.timezone.utc) - self.last_retrain).total_seconds() / 3600
        return hours_since_retrain >= self.config["retrain_interval_hours"]
    
    def _retrain_models_safe(self):
        """Reentrenar modelos - SAFE VERSION"""
        
        if not self.is_ready():
            return
        
        print("🔄 Iniciando reentrenamiento adaptativo SAFE...")
        
        try:
            # Preparar datos de entrenamiento
            if len(self.feature_history) < self.config["min_samples_for_training"]:
                print("⚠️ Insuficientes samples para reentrenamiento")
                return
            
            # Preparar features array
            X = np.array([self.features_to_array(f) for f in self.feature_history])
            
            # Preparar labels para diferentes tareas
            y_disruption = []
            y_movement = []
            
            for f in self.feature_history:
                if f.disruption_occurred is not None:
                    y_disruption.append(f.disruption_occurred)
                
                if f.future_move_5m is not None:
                    y_movement.append(self._classify_movement(f.future_move_5m))
            
            # Verificar que tenemos suficientes samples con labels
            min_samples = 20
            if len(y_disruption) >= min_samples and len(y_movement) >= min_samples:
                
                # Escalar features
                X_scaled = self.feature_scaler.fit_transform(X)
                
                # Reentrenar disruption classifier
                X_disruption = X_scaled[:len(y_disruption)]
                self.disruption_classifier.fit(X_disruption, y_disruption)
                
                # Reentrenar movement predictor  
                X_movement = X_scaled[:len(y_movement)]
                self.movement_predictor.fit(X_movement, y_movement)
                
                # Reentrenar anomaly detector
                self.anomaly_detector.fit(X_scaled)
                
                # Update state
                self.models_trained = True
                self.last_retrain = datetime.datetime.now(datetime.timezone.utc)
                
                # Guardar modelos
                self._save_models_safe()
                
                print(f"✅ Modelos ML reentrenados: {len(y_disruption)} disruption samples, {len(y_movement)} movement samples")
                
            else:
                print(f"⚠️ Insuficientes labeled samples: disruption={len(y_disruption)}, movement={len(y_movement)}")
                
        except Exception as e:
            print(f"❌ Error en reentrenamiento SAFE: {str(e)}")
    
    def _classify_movement(self, price_change: float) -> int:
        """Clasificar movimiento de precio en categorías"""
        
        if price_change > 0.3:
            return 2  # Bullish
        elif price_change < -0.3:
            return 0  # Bearish
        else:
            return 1  # Neutral
    
    def _save_models_safe(self):
        """Guardar modelos entrenados - SAFE VERSION"""
        
        models_to_save = {
            "disruption_classifier": self.disruption_classifier,
            "movement_predictor": self.movement_predictor,
            "anomaly_detector": self.anomaly_detector,
            "feature_scaler": self.feature_scaler
        }
        
        saved_count = 0
        for model_name, model in models_to_save.items():
            try:
                file_path = f"{self.ml_dir}/{model_name}_{self.symbol}.joblib"
                joblib.dump(model, file_path)
                saved_count += 1
            except Exception as e:
                print(f"⚠️ Error guardando {model_name}: {str(e)}")
        
        print(f"💾 {saved_count}/{len(models_to_save)} modelos guardados")
    
    def get_status(self) -> Dict:
        """Obtener estado del ML Engine"""
        
        return {
            "enabled": self.enabled,
            "ready": self.is_ready(),
            "models_trained": self.models_trained,
            "training_samples": len(self.feature_history) if self.is_ready() else 0,
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "initialization_error": self.initialization_error
        }

# ========== SAFE INTEGRATION FUNCTIONS ==========

def integrate_ml_safely(orderbook_intel, scheduler, enable_ml: bool = True):
    """Integrar ML Engine de manera segura - NO ROMPE SISTEMA EXISTENTE"""
    
    try:
        # Crear ML Engine con safety switch
        ml_engine = AdaptiveMLEngine(orderbook_intel.symbol, enable_ml=enable_ml)
        
        # Agregar como attribute opcional
        orderbook_intel.ml_engine = ml_engine
        scheduler.ml_engine = ml_engine
        
        # SAFE OVERRIDE: Solo si ML está enabled y ready
        if ml_engine.is_ready():
            
            # Override seguro del método de procesamiento
            original_process = orderbook_intel._process_snapshot
            
            async def enhanced_process_with_ml(snapshot):
                """Procesamiento mejorado con ML - SAFE VERSION"""
                
                # SIEMPRE ejecutar procesamiento original primero
                await original_process(snapshot)
                
                # ML Analysis solo si está ready
                try:
                    if ml_engine.is_ready():
                        # Context básico
                        market_context = {
                            "market_regime": getattr(orderbook_intel, 'current_market_regime', 'unknown'),
                            "price_change_1m": 0.0,
                            "price_change_5m": 0.0,
                            "volatility_1m": 0.0
                        }
                        
                        # Extraer features
                        features = ml_engine.extract_features_safe(snapshot, market_context)
                        
                        if features:
                            # Generar insights ML
                            ml_insights = ml_engine.generate_ml_insights_safe(features)
                            
                            # Aplicar a señales si existen
                            if hasattr(scheduler, 'state') and scheduler.state.current_signals:
                                _apply_ml_insights_safely(scheduler, ml_insights)
                            
                            # Log solo insights importantes
                            disruption_data = ml_insights.get("disruption_prediction", {})
                            if disruption_data.get("status") == "success":
                                prob = disruption_data.get("probability", 0)
                                if prob > 0.75:
                                    orderbook_intel.logger.warning(f"[ML] HIGH disruption risk: {prob:.2f}")
                
                except Exception as e:
                    # Error en ML no debe romper el flujo principal
                    orderbook_intel.logger.debug(f"[ML] Non-critical ML error: {str(e)}")
            
            # Reemplazar método de manera segura
            orderbook_intel._process_snapshot = enhanced_process_with_ml
            
            print("🤖 ✅ ML Engine integrado SAFELY con OrderBook Intelligence")
        else:
            print("🤖 ⚠️ ML Engine created but not ready - continuing without ML")
        
        return ml_engine
        
    except Exception as e:
        print(f"🤖 ❌ Error en ML integration: {str(e)}")
        print("🛡️ Continuing without ML (sistema no se rompe)")
        # Crear ML engine dummy para mantener compatibilidad
        return type('DummyML', (), {
            'is_ready': lambda: False,
            'get_status': lambda: {"enabled": False, "error": str(e)}
        })()

def _apply_ml_insights_safely(scheduler, ml_insights):
    """Aplicar insights ML de manera segura"""
    
    try:
        # Solo procesar si insights son válidos
        if ml_insights.get("status") == "ml_disabled":
            return
        
        # Extraer predicciones de manera segura
        disruption_data = ml_insights.get("disruption_prediction", {})
        movement_data = ml_insights.get("movement_prediction", {})
        anomaly_data = ml_insights.get("anomaly_detection", {})
        
        disruption_risk = disruption_data.get("probability", 0.5) if disruption_data.get("status") == "success" else 0.5
        movement_direction = movement_data.get("direction", "neutral") if movement_data.get("status") == "success" else "neutral"
        is_anomaly = anomaly_data.get("is_anomaly", False) if anomaly_data.get("status") == "success" else False
        
        # Preparar ajustes ML
        ml_adjustments = {}
        
        # Ajuste conservador por riesgo de disrupción
        if disruption_risk > 0.8:
            ml_adjustments["disruption_risk"] = "very_high"
            ml_adjustments["position_size_adjustment"] = 0.5
        elif disruption_risk > 0.65:
            ml_adjustments["disruption_risk"] = "high" 
            ml_adjustments["position_size_adjustment"] = 0.75
        
        # Ajuste por predicción de dirección
        current_action = scheduler.state.current_signals.get("action", "WAIT")
        if current_action in ["BUY", "WEAK_BUY"] and movement_direction == "bearish":
            ml_adjustments["direction_conflict"] = True
            ml_adjustments["confidence_penalty"] = -10  # Conservador
        elif current_action in ["SELL", "WEAK_SELL"] and movement_direction == "bullish":
            ml_adjustments["direction_conflict"] = True
            ml_adjustments["confidence_penalty"] = -10
        
        # Ajuste por anomalía
        if is_anomaly:
            ml_adjustments["anomaly_detected"] = True
            ml_adjustments["caution_level"] = "elevated"
        
        # Aplicar solo si hay ajustes significativos
        if ml_adjustments:
            # Asegurarse de que realtime_adjustments existe
            if not hasattr(scheduler.state, 'realtime_adjustments'):
                scheduler.state.realtime_adjustments = {}
            
            scheduler.state.realtime_adjustments["ml_insights"] = ml_adjustments
            scheduler.logger.info(f"[ML] Applied conservative adjustments: {len(ml_adjustments)} factors")
        
    except Exception as e:
        # Errores en ML insights no deben romper sistema
        scheduler.logger.debug(f"[ML] Error applying insights (non-critical): {str(e)}")

# ========== INSTALLATION HELPER ==========

def install_ml_dependencies():
    """Helper para instalar dependencias ML si faltan"""
    
    required_packages = ['scikit-learn', 'joblib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ Missing ML packages: {missing_packages}")
        print("📦 Install with: pip install scikit-learn joblib")
        return False
    
    return True

# ========== TEST FUNCTION ==========

async def test_adaptive_ml_safe():
    """Test del sistema ML adaptativo - SAFE VERSION"""
    
    print("🧪 TESTING ADAPTIVE ML ENGINE - SAFE INTEGRATION")
    print("=" * 55)
    
    # Check dependencies
    if not install_ml_dependencies():
        print("❌ ML dependencies missing - skipping ML test")
        return False
    
    # Test con ML enabled
    print("\n1️⃣ Testing ML Engine ENABLED:")
    ml_engine_enabled = AdaptiveMLEngine("ETHUSD_PERP", enable_ml=True)
    print(f"   Status: {ml_engine_enabled.get_status()}")
    
    # Test con ML disabled  
    print("\n2️⃣ Testing ML Engine DISABLED:")
    ml_engine_disabled = AdaptiveMLEngine("ETHUSD_PERP", enable_ml=False)
    print(f"   Status: {ml_engine_disabled.get_status()}")
    
    # Test integration safety
    print("\n3️⃣ Testing SAFE INTEGRATION:")
    
    # Mock orderbook intel
    class MockOrderBookIntel:
        def __init__(self):
            self.symbol = "ETHUSD_PERP"
            self.current_market_regime = "lateral"
            self.logger = type('Logger', (), {
                'warning': lambda x: print(f"LOG: {x}"),
                'info': lambda x: print(f"LOG: {x}"),
                'debug': lambda x: None
            })()
        
        async def _process_snapshot(self, snapshot):
            print(f"   Original processing: {snapshot}")
    
    # Mock scheduler
    class MockScheduler:
        def __init__(self):
            self.state = type('State', (), {
                'current_signals': {"action": "WEAK_BUY", "confidence": 60},
                'realtime_adjustments': {}
            })()
            self.logger = type('Logger', (), {
                'info': lambda x: print(f"SCHEDULER: {x}"),
                'debug': lambda x: None
            })()
    
    mock_orderbook = MockOrderBookIntel()
    mock_scheduler = MockScheduler()
    
    # Test safe integration
    ml_engine = integrate_ml_safely(mock_orderbook, mock_scheduler, enable_ml=True)
    
    print(f"   Integration result: {ml_engine.get_status() if hasattr(ml_engine, 'get_status') else 'Fallback mode'}")
    
    print("\n✅ SAFE ML Integration test completado")
    return True

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_adaptive_ml_safe())