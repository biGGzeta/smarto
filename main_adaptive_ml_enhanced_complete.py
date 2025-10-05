#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Trading Adaptativo con ML Enhanced COMPLETO
Integra: Análisis Adaptativo + Triángulos + Volatilidad Live + ML Mejorado
Autor: biGGzeta
Fecha: 2025-10-04
Basado en tu arquitectura existente + mejoras enhanced
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import warnings
import math

# Imports de tu sistema existente
from data.csv_handler import BinanceDataDownloader
from analysis.adaptive_low_time_analyzer import AdaptiveLowTimeAnalyzer
from analysis.adaptive_high_time_analyzer import AdaptiveHighTimeAnalyzer
from analysis.adaptive_percentage_analyzer import AdaptivePercentageAnalyzer
from analysis.adaptive_panorama_analyzer import AdaptivePanoramaAnalyzer
from analysis.adaptive_weekly_analyzer import AdaptiveWeeklyAnalyzer
from analysis.max_min_analyzer import MaxMinAnalyzer
from utils.logger import QALogger
from config.settings import BinanceConfig

# Sistema ML existente
from ml.adaptive_ml_system import AdaptiveMLSystem
from ml.feature_engineering import MarketFeatures
from ml.decision_tree import TradingSignal, SignalType

# Logger de análisis profundo existente
from analysis.market_data_logger import MarketDataLogger

# Suprimir warnings
warnings.filterwarnings('ignore')

# =============================================================================
# NUEVAS CLASES ENHANCED PARA TRIÁNGULOS Y VOLATILIDAD
# =============================================================================

@dataclass
class TrianglePattern:
    """Clase para representar un patrón de triángulo"""
    type: str  # 'ascending', 'descending', 'symmetric'
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    convergence_point: float
    resistance_line: Tuple[float, float]  # (slope, intercept)
    support_line: Tuple[float, float]     # (slope, intercept)
    apex_time: pd.Timestamp
    apex_price: float
    breakout_probability: float
    height: float
    duration_hours: float
    confidence: float

class TriangleDetector:
    """Detector de triángulos integrado con tu sistema"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.min_touches = 2
        self.min_duration_hours = 1
        self.max_duration_hours = 48
        
    def detect_triangles_in_data(self, data: pd.DataFrame, timeframe_hours: float = 3) -> List[TrianglePattern]:
        """Detecta triángulos en tus datos existentes"""
        triangles = []
        
        try:
            if len(data) < 20:
                return triangles
            
            print(f"🔺 Analizando {len(data)} velas para detectar triángulos...")
            
            # Detectar extremos locales
            highs, lows = self._detect_local_extremes(data)
            
            print(f"   📈 Máximos detectados: {len(highs)}")
            print(f"   📉 Mínimos detectados: {len(lows)}")
            
            if len(highs) < 2 or len(lows) < 2:
                return triangles
            
            # Buscar patrones de triángulo
            triangle = self._analyze_triangle_pattern(data, highs, lows, timeframe_hours)
            
            if triangle and triangle.confidence > 40:
                triangles.append(triangle)
                print(f"   ✅ Triángulo {triangle.type} detectado (confianza: {triangle.confidence:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error detectando triángulos: {e}")
        
        return triangles
    
    def _detect_local_extremes(self, data: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
        """Detecta extremos locales adaptativos"""
        highs = []
        lows = []
        
        # Ventana adaptativa
        volatility = data['close'].pct_change().std()
        if volatility > 0.03:
            window = 3
        elif volatility > 0.015:
            window = 5
        else:
            window = 7
        
        for i in range(window, len(data) - window):
            current_high = data.iloc[i]['high']
            current_low = data.iloc[i]['low']
            
            # Verificar máximo local
            is_peak = all(data.iloc[j]['high'] <= current_high 
                         for j in range(i - window, i + window + 1) if j != i)
            
            if is_peak:
                highs.append({
                    'index': i,
                    'time': data.index[i] if hasattr(data.index[i], 'to_pydatetime') else pd.Timestamp(data.index[i]),
                    'price': current_high
                })
            
            # Verificar mínimo local
            is_trough = all(data.iloc[j]['low'] >= current_low 
                           for j in range(i - window, i + window + 1) if j != i)
            
            if is_trough:
                lows.append({
                    'index': i,
                    'time': data.index[i] if hasattr(data.index[i], 'to_pydatetime') else pd.Timestamp(data.index[i]),
                    'price': current_low
                })
        
        return highs[-4:], lows[-4:]  # Últimos 4 de cada uno
    
    def _analyze_triangle_pattern(self, data: pd.DataFrame, highs: List[dict], 
                                 lows: List[dict], timeframe_hours: float) -> Optional[TrianglePattern]:
        """Analiza patrón de triángulo"""
        
        try:
            if len(highs) < 2 or len(lows) < 2:
                return None
            
            # Calcular tendencias
            high_prices = [h['price'] for h in highs]
            low_prices = [l['price'] for l in lows]
            
            # Tendencia de máximos
            high_slope = (high_prices[-1] - high_prices[0]) / len(high_prices) if len(high_prices) > 1 else 0
            # Tendencia de mínimos  
            low_slope = (low_prices[-1] - low_prices[0]) / len(low_prices) if len(low_prices) > 1 else 0
            
            # Clasificar triángulo
            triangle_type = self._classify_triangle_type(high_slope, low_slope)
            
            # Calcular métricas
            start_time = min(highs[0]['time'], lows[0]['time'])
            end_time = max(highs[-1]['time'], lows[-1]['time'])
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            height = max(high_prices) - min(low_prices)
            
            # Calcular convergencia
            current_spread = abs(high_prices[-1] - low_prices[-1])
            initial_spread = abs(high_prices[0] - low_prices[0])
            convergence_ratio = current_spread / initial_spread if initial_spread > 0 else 1
            
            # Probabilidad de breakout
            breakout_probability = self._calculate_breakout_probability(
                triangle_type, convergence_ratio, duration_hours
            )
            
            # Confianza del patrón
            confidence = self._calculate_pattern_confidence(
                highs, lows, convergence_ratio, duration_hours
            )
            
            # Apex estimado
            apex_time = end_time + pd.Timedelta(hours=duration_hours * 0.3)
            apex_price = (high_prices[-1] + low_prices[-1]) / 2
            
            return TrianglePattern(
                type=triangle_type,
                start_time=start_time,
                end_time=end_time,
                convergence_point=convergence_ratio,
                resistance_line=(high_slope, high_prices[0]),
                support_line=(low_slope, low_prices[0]),
                apex_time=apex_time,
                apex_price=apex_price,
                breakout_probability=breakout_probability,
                height=height,
                duration_hours=duration_hours,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"❌ Error analizando patrón: {e}")
            return None
    
    def _classify_triangle_type(self, high_slope: float, low_slope: float) -> str:
        """Clasifica tipo de triángulo"""
        slope_threshold = 0.5
        
        if abs(high_slope) < slope_threshold and low_slope > slope_threshold:
            return "ascending"
        elif abs(low_slope) < slope_threshold and high_slope < -slope_threshold:
            return "descending"
        elif high_slope < 0 and low_slope > 0:
            return "symmetric"
        else:
            return "irregular"
    
    def _calculate_breakout_probability(self, triangle_type: str, convergence_ratio: float, 
                                      duration_hours: float) -> float:
        """Calcula probabilidad de breakout"""
        base_prob = 50.0
        
        # Tipo de triángulo
        if triangle_type == "ascending":
            base_prob += 25
        elif triangle_type == "descending":
            base_prob += 20
        elif triangle_type == "symmetric":
            base_prob += 15
        
        # Convergencia
        if convergence_ratio < 0.5:
            base_prob += 20
        elif convergence_ratio < 0.7:
            base_prob += 10
        
        # Duración
        if 2 <= duration_hours <= 12:
            base_prob += 15
        elif duration_hours > 24:
            base_prob -= 10
        
        return min(95, max(10, base_prob))
    
    def _calculate_pattern_confidence(self, highs: List[dict], lows: List[dict], 
                                    convergence_ratio: float, duration_hours: float) -> float:
        """Calcula confianza del patrón"""
        confidence = 50.0
        
        # Número de toques
        total_touches = len(highs) + len(lows)
        confidence += min(30, total_touches * 5)
        
        # Convergencia
        if convergence_ratio < 0.6:
            confidence += 20
        
        # Duración
        if 3 <= duration_hours <= 18:
            confidence += 15
        
        return min(100, max(0, confidence))
    
    def get_triangle_features_for_ml(self, triangles: List[TrianglePattern]) -> Dict[str, float]:
        """Convierte triángulos a features para ML"""
        features = {
            'triangle_count': len(triangles),
            'has_ascending_triangle': 0.0,
            'has_descending_triangle': 0.0,
            'has_symmetric_triangle': 0.0,
            'max_breakout_probability': 0.0,
            'avg_triangle_confidence': 0.0,
            'triangle_convergence_strength': 0.0,
            'triangle_time_to_apex_hours': 24.0
        }
        
        if not triangles:
            return features
        
        # Detectar tipos
        for triangle in triangles:
            if triangle.type == 'ascending':
                features['has_ascending_triangle'] = 1.0
            elif triangle.type == 'descending':
                features['has_descending_triangle'] = 1.0
            elif triangle.type == 'symmetric':
                features['has_symmetric_triangle'] = 1.0
        
        # Métricas del mejor triángulo
        best_triangle = max(triangles, key=lambda x: x.confidence)
        features['max_breakout_probability'] = best_triangle.breakout_probability / 100.0
        features['avg_triangle_confidence'] = np.mean([t.confidence for t in triangles]) / 100.0
        features['triangle_convergence_strength'] = 1.0 - best_triangle.convergence_point
        
        # Tiempo al apex
        now = pd.Timestamp.now()
        time_to_apex = (best_triangle.apex_time - now).total_seconds() / 3600
        features['triangle_time_to_apex_hours'] = max(0, min(48, time_to_apex))
        
        return features

class LiveVolatilityMonitor:
    """Monitor de volatilidad en tiempo real"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
    def analyze_live_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza volatilidad en tiempo real"""
        
        try:
            if len(data) < 20:
                return {'current_volatility': 0, 'trend': 'unknown', 'is_expanding': False}
            
            print(f"⚡ Analizando volatilidad en vivo...")
            
            # Calcular returns
            returns = data['close'].pct_change().dropna()
            
            # Volatilidad rolling
            vol_20 = returns.rolling(20).std() * 100
            vol_10 = returns.rolling(10).std() * 100
            vol_5 = returns.rolling(5).std() * 100
            
            current_vol = vol_5.iloc[-1] if not vol_5.empty else 0
            
            # Tendencia de volatilidad
            vol_trend = "neutral"
            if len(vol_20) >= 10:
                recent_vol = vol_20.iloc[-5:].mean()
                past_vol = vol_20.iloc[-15:-10].mean()
                
                if recent_vol > past_vol * 1.2:
                    vol_trend = "expanding"
                elif recent_vol < past_vol * 0.8:
                    vol_trend = "contracting"
            
            # Detección de expansión súbita
            is_expanding = False
            if len(vol_5) >= 5:
                recent_avg = vol_5.iloc[-3:].mean()
                past_avg = vol_5.iloc[-10:-7].mean() if len(vol_5) >= 10 else vol_5.iloc[-5:-2].mean()
                
                if recent_avg > past_avg * 1.5:
                    is_expanding = True
            
            # Alertas de volatilidad
            alerts = []
            if current_vol > 5:
                alerts.append("High volatility detected")
            if is_expanding:
                alerts.append("Volatility expansion detected")
            if vol_trend == "expanding":
                alerts.append("Sustained volatility increase")
            
            print(f"   📊 Volatilidad actual: {current_vol:.2f}%")
            print(f"   📈 Tendencia: {vol_trend}")
            print(f"   💥 Expansión: {'SÍ' if is_expanding else 'NO'}")
            print(f"   🚨 Alertas: {len(alerts)}")
            
            return {
                'current_volatility': current_vol,
                'volatility_20': vol_20.iloc[-1] if not vol_20.empty else 0,
                'volatility_10': vol_10.iloc[-1] if not vol_10.empty else 0,
                'trend': vol_trend,
                'is_expanding': is_expanding,
                'expansion_factor': recent_avg / past_avg if 'recent_avg' in locals() and past_avg > 0 else 1.0,
                'alerts': alerts,
                'alert_count': len(alerts)
            }
            
        except Exception as e:
            print(f"❌ Error analizando volatilidad: {e}")
            return {'current_volatility': 0, 'trend': 'unknown', 'is_expanding': False, 'alerts': []}
    
    def get_volatility_features_for_ml(self, vol_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Convierte análisis de volatilidad a features ML"""
        
        features = {
            'current_volatility_pct': vol_analysis.get('current_volatility', 0) / 100.0,
            'volatility_20_pct': vol_analysis.get('volatility_20', 0) / 100.0,
            'volatility_trend_expanding': 1.0 if vol_analysis.get('trend') == 'expanding' else 0.0,
            'volatility_trend_contracting': 1.0 if vol_analysis.get('trend') == 'contracting' else 0.0,
            'volatility_is_expanding': 1.0 if vol_analysis.get('is_expanding', False) else 0.0,
            'volatility_expansion_factor': min(3.0, vol_analysis.get('expansion_factor', 1.0)),
            'volatility_alert_count': min(5.0, vol_analysis.get('alert_count', 0)),
            'volatility_regime_high': 1.0 if vol_analysis.get('current_volatility', 0) > 4 else 0.0
        }
        
        return features

class EnhancedDecisionTree:
    """Árbol de decisiones enhanced con triángulos y volatilidad"""
    
    def __init__(self):
        pass
    
    def evaluate_enhanced_signal(self, base_features: MarketFeatures, 
                                triangle_features: Dict[str, float], 
                                volatility_features: Dict[str, float]) -> Dict[str, Any]:
        """Evalúa señal enhanced con todos los factores"""
        
        try:
            print(f"\n🌳 Evaluando señal enhanced...")
            print(f"   📊 Régimen: {base_features.regime}")
            print(f"   ⚡ Momentum 3d: {base_features.momentum_3d:.2f}%")
            print(f"   📍 Posición: {base_features.price_position_pct:.1f}%")
            print(f"   🔺 Triángulos: {triangle_features.get('triangle_count', 0)}")
            print(f"   💥 Vol expanding: {volatility_features.get('volatility_is_expanding', 0)}")
            
            # Lógica enhanced del árbol
            signal_type = "HOLD"
            confidence = 50
            reasoning = []
            
            # Factor 1: Régimen base
            if base_features.regime in ["low_volatility", "smooth_ranging"]:
                reasoning.append("Régimen calmo detectado")
                
                # Factor 2: Estructura de extremos
                if (base_features.maximos_trend == "crecientes" and 
                    base_features.minimos_trend == "crecientes"):
                    reasoning.append("Extremos alineados crecientes")
                    confidence += 15
                    
                    # Factor 3: Momentum (CORREGIDO el umbral)
                    if base_features.momentum_3d > 5:
                        reasoning.append(f"Momentum fuerte ({base_features.momentum_3d:.1f}%)")
                        confidence += 20
                        
                        # Factor 4: Triángulos (NUEVO)
                        if triangle_features.get('has_ascending_triangle', 0) == 1:
                            reasoning.append("Triángulo ascendente detectado")
                            confidence += 15
                            
                            breakout_prob = triangle_features.get('max_breakout_probability', 0)
                            if breakout_prob > 0.7:
                                reasoning.append(f"Alta probabilidad breakout ({breakout_prob*100:.0f}%)")
                                confidence += 10
                                signal_type = "BUY"
                        
                        # Factor 5: Volatilidad (NUEVO)
                        if volatility_features.get('volatility_is_expanding', 0) == 1:
                            reasoning.append("Expansión de volatilidad detectada")
                            confidence += 10
                            if signal_type == "HOLD":
                                signal_type = "BUY"
                        
                        # Factor 6: Posición (Umbral relajado)
                        if base_features.price_position_pct < 85:  # Era 90
                            reasoning.append("Posición favorable para compra")
                            confidence += 5
                            if signal_type == "HOLD" and base_features.momentum_3d > 7:
                                signal_type = "BUY"
                        
                        # Factor 7: Estructura fuerte (CORREGIDO)
                        avg_strength = (base_features.maximos_strength + base_features.minimos_strength) / 2
                        if avg_strength > 55:  # Era 60
                            reasoning.append(f"Estructura fuerte (promedio {avg_strength:.1f})")
                            confidence += 10
                            if signal_type == "HOLD":
                                signal_type = "BUY"
            
            # Casos especiales enhanced
            
            # Caso 1: Triple confirmación (momentum + triángulo + volatilidad)
            if (base_features.momentum_3d > 8 and 
                triangle_features.get('max_breakout_probability', 0) > 0.8 and 
                volatility_features.get('volatility_is_expanding', 0) == 1):
                signal_type = "STRONG_BUY"
                confidence = min(95, confidence + 20)
                reasoning.append("🚀 TRIPLE CONFIRMACIÓN: Momentum + Triángulo + Volatilidad")
            
            # Caso 2: Triángulo simétrico con momentum
            elif (triangle_features.get('has_symmetric_triangle', 0) == 1 and 
                  base_features.momentum_3d > 6):
                signal_type = "BUY"
                confidence = min(90, confidence + 15)
                reasoning.append("Triángulo simétrico + momentum alcista")
            
            # Caso 3: Volatilidad extrema con estructura
            elif (volatility_features.get('current_volatility_pct', 0) > 0.05 and
                  base_features.extremes_alignment and
                  base_features.momentum_3d > 4):
                signal_type = "BUY"
                confidence = min(85, confidence + 10)
                reasoning.append("Volatilidad alta + estructura alineada")
            
            # Limitaciones de riesgo
            if base_features.price_position_pct > 95:
                confidence = max(30, confidence - 20)
                reasoning.append("⚠️ Precio muy alto en rango semanal")
            
            if volatility_features.get('current_volatility_pct', 0) > 0.08:
                confidence = max(40, confidence - 10)
                reasoning.append("⚠️ Volatilidad extrema detectada")
            
            return {
                'signal_type': signal_type,
                'confidence': confidence,
                'reasoning': reasoning,
                'enhanced_factors': {
                    'base_momentum': base_features.momentum_3d,
                    'triangle_signal': triangle_features.get('has_ascending_triangle', 0),
                    'volatility_signal': volatility_features.get('volatility_is_expanding', 0),
                    'structure_strength': avg_strength if 'avg_strength' in locals() else 0
                }
            }
            
        except Exception as e:
            print(f"❌ Error en evaluación enhanced: {e}")
            return {
                'signal_type': 'HOLD',
                'confidence': 30,
                'reasoning': [f"Error en evaluación: {e}"],
                'enhanced_factors': {}
            }

# =============================================================================
# CLASE PRINCIPAL ENHANCED
# =============================================================================

class AdaptiveTradingMLSystemEnhanced:
    """Sistema principal ENHANCED que combina TODO"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP", enable_ml: bool = True, 
                 enable_logging: bool = True, enable_triangles: bool = True, 
                 enable_live_volatility: bool = True):
        
        self.symbol = symbol
        self.enable_ml = enable_ml
        self.enable_logging = enable_logging
        self.enable_triangles = enable_triangles
        self.enable_live_volatility = enable_live_volatility
        
        # Componentes base (tu sistema existente)
        self.downloader = BinanceDataDownloader(symbol)
        self.adaptive_low_analyzer = AdaptiveLowTimeAnalyzer(symbol)
        self.adaptive_high_analyzer = AdaptiveHighTimeAnalyzer(symbol)
        self.adaptive_percentage_analyzer = AdaptivePercentageAnalyzer(symbol)
        self.adaptive_panorama_analyzer = AdaptivePanoramaAnalyzer(symbol)
        self.adaptive_weekly_analyzer = AdaptiveWeeklyAnalyzer(symbol)
        self.original_max_min_analyzer = MaxMinAnalyzer(symbol)
        
        # Sistema ML existente
        self.ml_system = AdaptiveMLSystem(save_data=enable_ml) if enable_ml else None
        
        # Logger existente
        self.market_logger = MarketDataLogger(symbol) if enable_logging else None
        self.logger = QALogger(symbol)
        
        # NUEVOS: Componentes Enhanced
        if enable_triangles:
            self.triangle_detector = TriangleDetector(symbol)
        
        if enable_live_volatility:
            self.volatility_monitor = LiveVolatilityMonitor(symbol)
        
        self.enhanced_decision_tree = EnhancedDecisionTree()
        
        # Cache
        self.analysis_cache = {}
        
        print(f"🚀 Sistema Enhanced inicializado para {symbol}")
        print(f"   🤖 ML: {'✅' if enable_ml else '❌'}")
        print(f"   📝 Logging: {'✅' if enable_logging else '❌'}")
        print(f"   🔺 Triángulos: {'✅' if enable_triangles else '❌'}")
        print(f"   ⚡ Volatilidad Live: {'✅' if enable_live_volatility else '❌'}")
    
    def run_complete_enhanced_analysis(self, hours_3h: float = 3, hours_48h: float = 48) -> Dict[str, Any]:
        """Ejecutar análisis COMPLETO ENHANCED"""
        
        print("\n" + "=" * 100)
        print("🚀 INICIANDO SISTEMA COMPLETO ENHANCED: Adaptativo + Triángulos + Volatilidad + ML")
        print("=" * 100)
        print(f"📊 Símbolo: {self.symbol}")
        print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"👤 Usuario: biGGzeta")
        print(f"🤖 ML habilitado: {self.enable_ml}")
        print(f"🔺 Triángulos habilitados: {self.enable_triangles}")
        print(f"⚡ Volatilidad live habilitada: {self.enable_live_volatility}")
        print(f"📝 Logging profundo: {self.enable_logging}")
        print("=" * 100)
        
        try:
            # FASE 1: Análisis adaptativo base
            print(f"\n🔧 FASE 1: Ejecutando análisis adaptativo base...")
            adaptive_results = self._run_adaptive_analysis(hours_3h, hours_48h)
            
            # FASE 2: Análisis de triángulos (NUEVO)
            triangle_results = {}
            if self.enable_triangles:
                print(f"\n🔺 FASE 2: Ejecutando análisis de triángulos...")
                triangle_results = self._run_triangle_analysis()
            
            # FASE 3: Análisis de volatilidad live (NUEVO)
            volatility_results = {}
            if self.enable_live_volatility:
                print(f"\n⚡ FASE 3: Ejecutando análisis de volatilidad live...")
                volatility_results = self._run_volatility_analysis()
            
            # FASE 4: Pipeline ML Enhanced
            ml_results = {}
            if self.enable_ml:
                print(f"\n🤖 FASE 4: Ejecutando pipeline ML enhanced...")
                ml_results = self._run_enhanced_ml_pipeline(
                    adaptive_results, triangle_results, volatility_results
                )
            
            # FASE 5: Análisis integrado enhanced
            print(f"\n📊 FASE 5: Generando análisis integrado enhanced...")
            integrated_analysis = self._create_enhanced_integrated_analysis(
                adaptive_results, triangle_results, volatility_results, ml_results
            )
            
            # FASE 6: Logging enhanced
            if self.enable_logging and self.market_logger:
                print(f"\n📝 FASE 6: Guardando análisis enhanced...")
                try:
                    report_file = self.market_logger.log_complete_analysis(
                        adaptive_results, ml_results, integrated_analysis
                    )
                    integrated_analysis['analysis_report_file'] = report_file
                    print(f"✅ Logging enhanced completado")
                except Exception as e:
                    print(f"⚠️ Error en logging enhanced: {e}")
            
            # FASE 7: Debug enhanced
            print(f"\n🔍 FASE 7: Debug detallado enhanced...")
            self._debug_enhanced_decision_path(
                ml_results.get("features"), triangle_results, volatility_results
            )
            
            return integrated_analysis
            
        except Exception as e:
            print(f"❌ Error en análisis enhanced: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _run_adaptive_analysis(self, hours_3h: float, hours_48h: float) -> Dict[str, Any]:
        """Ejecutar análisis adaptativo completo"""
        
        adaptive_results = {}
        
        try:
            # Descargar datos
            print(f"📥 Descargando datos de {hours_3h}h y {hours_48h}h...")
            data_3h = self.downloader.get_klines("1m", hours_3h)
            data_48h = self.downloader.get_klines("1m", hours_48h)
            
            # Guardar en cache
            self.analysis_cache["data_3h"] = data_3h
            self.analysis_cache["data_48h"] = data_48h
            
            if data_3h.empty or data_48h.empty:
                return {"error": "No se pudieron obtener datos"}
            
            # Análisis adaptativo de extremos de poco tiempo (3h)
            self.adaptive_low_analyzer.load_data(data_3h)
            low_time_result = self.adaptive_low_analyzer.analyze_adaptive_low_time_minimums(hours_3h)
            adaptive_results['low_time_analysis'] = low_time_result
            
            # Análisis adaptativo de máximos de poco tiempo (3h)
            self.adaptive_high_analyzer.load_data(data_3h)
            high_time_result = self.adaptive_high_analyzer.analyze_adaptive_high_time_maximums(hours_3h)
            adaptive_results['high_time_analysis'] = high_time_result
            
            # Análisis adaptativo de porcentajes (3h)
            self.adaptive_percentage_analyzer.load_data(data_3h)
            percentage_result = self.adaptive_percentage_analyzer.analyze_adaptive_range_percentage(hours_3h)
            adaptive_results['percentage_analysis'] = percentage_result
            
            # Análisis panorama adaptativo (48h)
            self.adaptive_panorama_analyzer.load_data(data_48h)
            panorama_result = self.adaptive_panorama_analyzer.analyze_adaptive_48h_panorama(hours_48h)
            adaptive_results['panorama_analysis'] = panorama_result
            
            # Análisis semanal adaptativo (168h)
            data_weekly = self.downloader.get_klines("1m", 168)
            if not data_weekly.empty:
                self.adaptive_weekly_analyzer.load_data(data_weekly)
                weekly_result = self.adaptive_weekly_analyzer.analyze_adaptive_weekly_with_recent_extremes(3)
                adaptive_results['weekly_analysis'] = weekly_result
                self.analysis_cache["data_weekly"] = data_weekly
            
            print(f"✅ Análisis adaptativo completado")
            return adaptive_results
            
        except Exception as e:
            print(f"❌ Error en análisis adaptativo: {e}")
            return {"error": str(e)}
    
    def _run_triangle_analysis(self) -> Dict[str, Any]:
        """Ejecutar análisis de triángulos (NUEVO)"""
        
        triangle_results = {}
        
        try:
            # Analizar triángulos en datos de 3h
            if "data_3h" in self.analysis_cache:
                data_3h = self.analysis_cache["data_3h"]
                triangles_3h = self.triangle_detector.detect_triangles_in_data(data_3h, 3)
                triangle_results['triangles_3h'] = triangles_3h
                
                if triangles_3h:
                    best_triangle = triangles_3h[0]
                    print(f"   🎯 Mejor patrón 3h: {best_triangle.type.upper()}")
                    print(f"   📊 Confianza: {best_triangle.confidence:.1f}%")
                    print(f"   ⚡ Breakout prob: {best_triangle.breakout_probability:.1f}%")
            
            # Analizar triángulos en datos de 48h
            if "data_48h" in self.analysis_cache:
                data_48h = self.analysis_cache["data_48h"]
                triangles_48h = self.triangle_detector.detect_triangles_in_data(data_48h, 48)
                triangle_results['triangles_48h'] = triangles_48h
                
                print(f"   🔺 Triángulos 48h detectados: {len(triangles_48h)}")
            
            # Generar features para ML
            all_triangles = (triangle_results.get('triangles_3h', []) + 
                           triangle_results.get('triangles_48h', []))
            
            triangle_features = self.triangle_detector.get_triangle_features_for_ml(all_triangles)
            triangle_results['ml_features'] = triangle_features
            
            # Summary para debug
            triangle_results['summary'] = {
                'total_triangles': len(all_triangles),
                'has_ascending': any(t.type == 'ascending' for t in all_triangles),
                'has_descending': any(t.type == 'descending' for t in all_triangles),
                'has_symmetric': any(t.type == 'symmetric' for t in all_triangles),
                'max_confidence': max([t.confidence for t in all_triangles], default=0),
                'max_breakout_prob': max([t.breakout_probability for t in all_triangles], default=0)
            }
            
            return triangle_results
            
        except Exception as e:
            print(f"❌ Error en análisis de triángulos: {e}")
            return {'error': str(e)}
    
    def _run_volatility_analysis(self) -> Dict[str, Any]:
        """Ejecutar análisis de volatilidad live (NUEVO)"""
        
        volatility_results = {}
        
        try:
            # Analizar volatilidad en datos de 3h
            if "data_3h" in self.analysis_cache:
                data_3h = self.analysis_cache["data_3h"]
                vol_analysis_3h = self.volatility_monitor.analyze_live_volatility(data_3h)
                volatility_results['volatility_3h'] = vol_analysis_3h
            
            # Analizar volatilidad en datos de 48h  
            if "data_48h" in self.analysis_cache:
                data_48h = self.analysis_cache["data_48h"]
                vol_analysis_48h = self.volatility_monitor.analyze_live_volatility(data_48h)
                volatility_results['volatility_48h'] = vol_analysis_48h
            
            # Usar análisis de 3h como principal
            main_vol_analysis = volatility_results.get('volatility_3h', {})
            
            # Generar features para ML
            volatility_features = self.volatility_monitor.get_volatility_features_for_ml(main_vol_analysis)
            volatility_results['ml_features'] = volatility_features
            
            # Summary para debug
            volatility_results['summary'] = {
                'current_volatility': main_vol_analysis.get('current_volatility', 0),
                'trend': main_vol_analysis.get('trend', 'unknown'),
                'is_expanding': main_vol_analysis.get('is_expanding', False),
                'alert_count': main_vol_analysis.get('alert_count', 0),
                'expansion_factor': main_vol_analysis.get('expansion_factor', 1.0)
            }
            
            return volatility_results
            
        except Exception as e:
            print(f"❌ Error en análisis de volatilidad: {e}")
            return {'error': str(e)}
    
    def _run_enhanced_ml_pipeline(self, adaptive_results: Dict, triangle_results: Dict, 
                                 volatility_results: Dict) -> Dict[str, Any]:
        """Pipeline ML enhanced con todas las features (MEJORADO)"""
        
        try:
            print(f"📊 Extrayendo features base...")
            
            # Obtener features base del ML system existente
            base_ml_results = self.ml_system.analyze_and_signal(adaptive_results)
            
            if "error" in base_ml_results:
                print(f"❌ Error en ML base: {base_ml_results['error']}")
                return base_ml_results
            
            base_features = base_ml_results.get("features")
            if not base_features:
                print(f"❌ No se pudieron extraer features base")
                return {"error": "No features base disponibles"}
            
            print(f"🔺 Integrando features de triángulos...")
            # Integrar features de triángulos
            triangle_features = triangle_results.get('ml_features', {})
            
            print(f"⚡ Integrando features de volatilidad...")
            # Integrar features de volatilidad
            volatility_features = volatility_results.get('ml_features', {})
            
            print(f"🌳 Evaluando con árbol enhanced...")
            # Evaluar con árbol enhanced
            enhanced_signal = self.enhanced_decision_tree.evaluate_enhanced_signal(
                base_features, triangle_features, volatility_features
            )
            
            print(f"✅ Señal enhanced: {enhanced_signal['signal_type']} (confianza: {enhanced_signal['confidence']:.0f}%)")
            
            return {
                'base_ml_results': base_ml_results,
                'triangle_features': triangle_features,
                'volatility_features': volatility_features,
                'enhanced_signal': enhanced_signal,
                'features': base_features,  # Mantener compatibilidad
                'status': 'success'
            }
            
        except Exception as e:
            print(f"❌ Error en pipeline ML enhanced: {e}")
            return {'error': str(e)}
    
    def _create_enhanced_integrated_analysis(self, adaptive_results: Dict, triangle_results: Dict,
                                           volatility_results: Dict, ml_results: Dict) -> Dict[str, Any]:
        """Crear análisis integrado enhanced (NUEVO)"""
        
        if "error" in ml_results:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error_ml_enhanced",
                "adaptive_results": adaptive_results,
                "ml_error": ml_results["error"]
            }
        
        enhanced_signal = ml_results.get("enhanced_signal", {})
        base_features = ml_results.get("features")
        
        # Crear resumen ejecutivo enhanced
        executive_summary = self._create_enhanced_executive_summary(
            base_features, enhanced_signal, adaptive_results, triangle_results, volatility_results
        )
        
        # Crear recomendaciones enhanced
        trading_recommendations = self._create_enhanced_trading_recommendations(
            enhanced_signal, base_features, triangle_results, volatility_results
        )
        
        # Análisis de confianza enhanced
        confidence_analysis = self._analyze_enhanced_confidence_factors(
            enhanced_signal, base_features, triangle_results, volatility_results
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success_enhanced",
            "symbol": self.symbol,
            "user": "biGGzeta",
            
            # Resultados principales enhanced
            "executive_summary": executive_summary,
            "enhanced_trading_signal": enhanced_signal,
            "trading_recommendations": trading_recommendations,
            "confidence_analysis": confidence_analysis,
            
            # Análisis por componentes
            "triangle_analysis": triangle_results,
            "volatility_analysis": volatility_results,
            
            # Features detalladas
            "market_features": {
                "regime": base_features.regime if base_features else "unknown",
                "volatility": base_features.volatility if base_features else 0,
                "trend_strength": base_features.trend_strength if base_features else 0,
                "price_position_pct": base_features.price_position_pct if base_features else 50,
                "momentum_3d": base_features.momentum_3d if base_features else 0,
                "extremes_alignment": base_features.extremes_alignment if base_features else False,
                "maximos_trend": base_features.maximos_trend if base_features else "unknown",
                "minimos_trend": base_features.minimos_trend if base_features else "unknown",
                "maximos_strength": base_features.maximos_strength if base_features else 0,
                "minimos_strength": base_features.minimos_strength if base_features else 0,
            },
            
            # Triangle features
            "triangle_features": ml_results.get('triangle_features', {}),
            
            # Volatility features  
            "volatility_features": ml_results.get('volatility_features', {}),
            
            # Datos completos
            "adaptive_results": adaptive_results,
            "ml_results": ml_results,
            
            # Metadatos enhanced
            "pipeline_version": "2.0_enhanced",
            "enhanced_capabilities": {
                "triangles_enabled": self.enable_triangles,
                "volatility_live_enabled": self.enable_live_volatility,
                "ml_enhanced": True
            }
        }
    
    def _create_enhanced_executive_summary(self, features: MarketFeatures, enhanced_signal: Dict,
                                         adaptive_results: Dict, triangle_results: Dict, 
                                         volatility_results: Dict) -> str:
        """Crear resumen ejecutivo enhanced"""
        
        if not features:
            return "❌ Error: No se pudieron extraer features del mercado"
        
        summary_parts = []
        
        # 1. Situación base
        summary_parts.append(f"{features.regime.upper()} (vol {features.volatility:.1f}%)")
        
        # 2. Posición
        if features.price_position_pct > 80:
            position = "ZONA ALTA"
        elif features.price_position_pct < 20:
            position = "ZONA BAJA"
        else:
            position = "ZONA MEDIA"
        summary_parts.append(f"📍 {position} ({features.price_position_pct:.0f}% del rango)")
        
        # 3. Estructura con extremos
        if features.extremes_alignment:
            structure = f"🏗️ {features.maximos_trend.upper()}"
        else:
            structure = f"🏗️ DIVERGENTE"
        summary_parts.append(structure)
        
        # 4. Momentum
        if features.momentum_3d > 5:
            momentum = f"⚡ ALCISTA FUERTE (+{features.momentum_3d:.1f}%)"
        elif features.momentum_3d > 2:
            momentum = f"⚡ ALCISTA (+{features.momentum_3d:.1f}%)"
        elif features.momentum_3d < -5:
            momentum = f"⚡ BAJISTA FUERTE ({features.momentum_3d:.1f}%)"
        elif features.momentum_3d < -2:
            momentum = f"⚡ BAJISTA ({features.momentum_3d:.1f}%)"
        else:
            momentum = f"⚡ NEUTRAL ({features.momentum_3d:.1f}%)"
        summary_parts.append(momentum)
        
        # 5. Triángulos (NUEVO)
        triangle_summary = triangle_results.get('summary', {})
        if triangle_summary.get('total_triangles', 0) > 0:
            if triangle_summary.get('has_ascending'):
                triangle_info = "🔺 TRIÁNGULO ASCENDENTE"
            elif triangle_summary.get('has_descending'):
                triangle_info = "🔻 TRIÁNGULO DESCENDENTE"
            elif triangle_summary.get('has_symmetric'):
                triangle_info = "◊ TRIÁNGULO SIMÉTRICO"
            else:
                triangle_info = "🔺 TRIÁNGULO DETECTADO"
            
            triangle_info += f" ({triangle_summary.get('max_breakout_prob', 0):.0f}% breakout)"
            summary_parts.append(triangle_info)
        
        # 6. Volatilidad (NUEVO)
        vol_summary = volatility_results.get('summary', {})
        if vol_summary.get('is_expanding', False):
            vol_info = f"💥 VOL EXPANDIENDO ({vol_summary.get('expansion_factor', 1):.1f}x)"
            summary_parts.append(vol_info)
        
        # 7. Señal enhanced
        signal_type = enhanced_signal.get('signal_type', 'HOLD')
        confidence = enhanced_signal.get('confidence', 50)
        summary_parts.append(f"🎯 {signal_type} ({confidence:.0f}%)")
        
        return " | ".join(summary_parts)
    
    def _create_enhanced_trading_recommendations(self, enhanced_signal: Dict, features: MarketFeatures,
                                               triangle_results: Dict, volatility_results: Dict) -> Dict[str, Any]:
        """Crear recomendaciones enhanced"""
        
        signal_type = enhanced_signal.get('signal_type', 'HOLD')
        confidence = enhanced_signal.get('confidence', 50)
        
        recommendations = {
            "primary_action": signal_type,
            "confidence_level": self._get_confidence_level(confidence),
            "position_sizing": self._suggest_enhanced_position_sizing(enhanced_signal, features, triangle_results, volatility_results),
            "entry_strategy": self._suggest_enhanced_entry_strategy(enhanced_signal, triangle_results, volatility_results),
            "risk_management": self._suggest_enhanced_risk_management(enhanced_signal, features, volatility_results),
            "time_horizon": self._estimate_time_horizon(triangle_results, volatility_results),
            "market_context": self._get_enhanced_market_context(features, triangle_results, volatility_results)
        }
        
        return recommendations
    
    def _analyze_enhanced_confidence_factors(self, enhanced_signal: Dict, features: MarketFeatures,
                                           triangle_results: Dict, volatility_results: Dict) -> Dict[str, Any]:
        """Analizar factores de confianza enhanced"""
        
        factors = {
            "positive_factors": [],
            "negative_factors": [],
            "neutral_factors": [],
            "overall_confidence": enhanced_signal.get('confidence', 50),
            "enhancement_factors": []
        }
        
        # Factores base (adaptativos + ML)
        if features and features.extremes_alignment:
            factors["positive_factors"].append("Extremos alineados (estructura clara)")
        
        if features and abs(features.momentum_3d) > 5:
            factors["positive_factors"].append(f"Momentum fuerte ({features.momentum_3d:.1f}%)")
        
        # Factores de triángulos (NUEVO)
        triangle_summary = triangle_results.get('summary', {})
        if triangle_summary.get('total_triangles', 0) > 0:
            factors["enhancement_factors"].append(f"Triángulo detectado (confianza {triangle_summary.get('max_confidence', 0):.0f}%)")
            
            if triangle_summary.get('max_breakout_prob', 0) > 70:
                factors["positive_factors"].append(f"Alta probabilidad de breakout ({triangle_summary['max_breakout_prob']:.0f}%)")
        
        # Factores de volatilidad (NUEVO)
        vol_summary = volatility_results.get('summary', {})
        if vol_summary.get('is_expanding', False):
            factors["enhancement_factors"].append("Expansión de volatilidad detectada")
            factors["positive_factors"].append("Volatilidad creciente favorece movimientos grandes")
        
        if vol_summary.get('current_volatility', 0) > 6:
            factors["negative_factors"].append("Volatilidad muy alta puede generar ruido excesivo")
        
        # Factores de posición
        if features and features.price_position_pct > 90:
            factors["negative_factors"].append("Precio muy cerca del máximo semanal")
        elif features and features.price_position_pct < 10:
            factors["negative_factors"].append("Precio muy cerca del mínimo semanal")
        
        return factors
    
    def _debug_enhanced_decision_path(self, features: MarketFeatures, triangle_results: Dict, 
                                    volatility_results: Dict):
        """Debug enhanced del árbol de decisiones"""
        
        if not features:
            print("❌ No hay features para debug enhanced")
            return
        
        print("\n🔍 DEBUG DETALLADO DEL ÁRBOL ENHANCED:")
        print("=" * 80)
        
        # Debug base (tu árbol original)
        print("📊 ANÁLISIS BASE:")
        regime = features.regime
        print(f"1️⃣ Régimen: {regime}")
        print(f"2️⃣ Extremos: máx {features.maximos_trend} | mín {features.minimos_trend}")
        print(f"3️⃣ Momentum 3d: {features.momentum_3d:.2f}%")
        print(f"4️⃣ Posición: {features.price_position_pct:.1f}%")
        print(f"5️⃣ Fuerza estructura: máx {features.maximos_strength:.1f} | mín {features.minimos_strength:.1f}")
        
        # Debug triángulos (NUEVO)
        print(f"\n🔺 ANÁLISIS DE TRIÁNGULOS:")
        triangle_summary = triangle_results.get('summary', {})
        triangle_features = triangle_results.get('ml_features', {})
        
        print(f"6️⃣ Triángulos detectados: {triangle_summary.get('total_triangles', 0)}")
        print(f"7️⃣ Tipo ascendente: {'SÍ' if triangle_features.get('has_ascending_triangle', 0) else 'NO'}")
        print(f"8️⃣ Probabilidad breakout: {triangle_summary.get('max_breakout_prob', 0):.1f}%")
        print(f"9️⃣ Confianza patrón: {triangle_summary.get('max_confidence', 0):.1f}%")
        
        # Debug volatilidad (NUEVO)
        print(f"\n⚡ ANÁLISIS DE VOLATILIDAD:")
        vol_summary = volatility_results.get('summary', {})
        vol_features = volatility_results.get('ml_features', {})
        
        print(f"🔟 Volatilidad actual: {vol_summary.get('current_volatility', 0):.2f}%")
        print(f"1️⃣1️⃣ Tendencia vol: {vol_summary.get('trend', 'unknown')}")
        print(f"1️⃣2️⃣ Expandiendo: {'SÍ' if vol_summary.get('is_expanding', False) else 'NO'}")
        print(f"1️⃣3️⃣ Factor expansión: {vol_summary.get('expansion_factor', 1):.1f}x")
        
        # Evaluación combinada
        print(f"\n🌳 EVALUACIÓN COMBINADA:")
        
        # Condiciones para BUY enhanced
        momentum_ok = features.momentum_3d > 5
        structure_ok = features.extremes_alignment
        position_ok = features.price_position_pct < 85
        triangle_boost = triangle_features.get('has_ascending_triangle', 0) == 1
        volatility_boost = vol_summary.get('is_expanding', False)
        
        print(f"1️⃣4️⃣ Momentum fuerte (>5%): {'✅' if momentum_ok else '❌'}")
        print(f"1️⃣5️⃣ Estructura alineada: {'✅' if structure_ok else '❌'}")
        print(f"1️⃣6️⃣ Posición favorable (<85%): {'✅' if position_ok else '❌'}")
        print(f"1️⃣7️⃣ Triángulo ascendente: {'🔺✅' if triangle_boost else '❌'}")
        print(f"1️⃣8️⃣ Volatilidad expandiendo: {'⚡✅' if volatility_boost else '❌'}")
        
        # Resultado esperado
        total_conditions = sum([momentum_ok, structure_ok, position_ok, triangle_boost, volatility_boost])
        
        if total_conditions >= 4:
            expected_result = "STRONG_BUY (90-95% confianza)"
        elif total_conditions >= 3:
            expected_result = "BUY (75-85% confianza)"
        elif total_conditions >= 2:
            expected_result = "BUY (60-70% confianza)"
        else:
            expected_result = "HOLD (50% confianza)"
        
        print(f"\n🎯 RESULTADO ESPERADO ENHANCED: {expected_result}")
        print(f"   📊 Condiciones cumplidas: {total_conditions}/5")
        
        print("=" * 80)
    
    # Métodos auxiliares enhanced
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convertir confianza a descripción"""
        if confidence >= 85:
            return "MUY ALTA"
        elif confidence >= 75:
            return "ALTA"
        elif confidence >= 65:
            return "MEDIA-ALTA"
        elif confidence >= 55:
            return "MEDIA"
        elif confidence >= 45:
            return "MEDIA-BAJA"
        else:
            return "BAJA"
    
    def _suggest_enhanced_position_sizing(self, enhanced_signal: Dict, features: MarketFeatures,
                                        triangle_results: Dict, volatility_results: Dict) -> str:
        """Sugerir tamaño de posición enhanced"""
        
        base_size = 100
        confidence = enhanced_signal.get('confidence', 50)
        
        # Factor confianza
        if confidence >= 85:
            confidence_factor = 1.0
        elif confidence >= 75:
            confidence_factor = 0.85
        elif confidence >= 65:
            confidence_factor = 0.7
        elif confidence >= 55:
            confidence_factor = 0.55
        else:
            confidence_factor = 0.4
        
        # Factor volatilidad
        vol_summary = volatility_results.get('summary', {})
        current_vol = vol_summary.get('current_volatility', 2)
        
        if current_vol > 6:
            vol_factor = 0.6
        elif current_vol > 4:
            vol_factor = 0.8
        else:
            vol_factor = 1.0
        
        # Factor triángulo (boost si hay confirmación)
        triangle_factor = 1.0
        triangle_summary = triangle_results.get('summary', {})
        if triangle_summary.get('max_breakout_prob', 0) > 80:
            triangle_factor = 1.1  # Boost del 10%
        
        suggested_size = base_size * confidence_factor * vol_factor * triangle_factor
        
        return f"{suggested_size:.0f}% del tamaño normal de posición"
    
    def _suggest_enhanced_entry_strategy(self, enhanced_signal: Dict, triangle_results: Dict, 
                                       volatility_results: Dict) -> str:
        """Sugerir estrategia de entrada enhanced"""
        
        confidence = enhanced_signal.get('confidence', 50)
        
        # Factor triángulo
        triangle_summary = triangle_results.get('summary', {})
        breakout_prob = triangle_summary.get('max_breakout_prob', 0)
        
        # Factor volatilidad
        vol_summary = volatility_results.get('summary', {})
        is_expanding = vol_summary.get('is_expanding', False)
        
        if confidence >= 85 and breakout_prob > 80:
            return "🚀 Entrada inmediata - Triple confirmación (adaptativo + triángulo + volatilidad)"
        elif confidence >= 75 and (breakout_prob > 70 or is_expanding):
            return "⚡ Entrada en 2 partes: 70% inmediato, 30% en confirmación de breakout"
        elif confidence >= 65:
            return "📊 Entrada gradual: 40% inmediato, 60% en confirmación"
        elif breakout_prob > 75:
            return "🔺 Esperar breakout del triángulo para entrada completa"
        else:
            return "⏳ Esperar mayor confirmación antes de entrar"
    
    def _suggest_enhanced_risk_management(self, enhanced_signal: Dict, features: MarketFeatures,
                                        volatility_results: Dict) -> Dict[str, str]:
        """Sugerir gestión de riesgo enhanced"""
        
        risk_mgmt = {}
        
        # Stop loss adaptativos por volatilidad
        vol_summary = volatility_results.get('summary', {})
        current_vol = vol_summary.get('current_volatility', 2)
        
        if current_vol > 5:
            risk_mgmt["stop_loss"] = "Stop loss amplio debido a alta volatilidad (3-4%)"
            risk_mgmt["trailing_stop"] = "Recomendado con trailing amplio"
        elif current_vol > 3:
            risk_mgmt["stop_loss"] = "Stop loss medio (2-3%)"
            risk_mgmt["trailing_stop"] = "Opcional"
        else:
            risk_mgmt["stop_loss"] = "Stop loss ajustado (1.5-2%)"
        
        # Take profit por momentum
        if features and features.momentum_3d > 8:
            risk_mgmt["take_profit"] = "Take profit escalonado: 50% en +5%, 50% en +8%"
        elif features and features.momentum_3d > 5:
            risk_mgmt["take_profit"] = "Take profit: 75% en +4%, 25% trailing"
        else:
            risk_mgmt["take_profit"] = "Take profit conservador en +3%"
        
        # Revisión de posición
    
                # Revisión de posición
        if vol_summary.get('is_expanding', False):
            risk_mgmt["position_review"] = "Revisar cada 1-2 horas por expansión de volatilidad"
        else:
            risk_mgmt["position_review"] = "Revisar cada 3-4 horas"
        
        return risk_mgmt
    
    def _estimate_time_horizon(self, triangle_results: Dict, volatility_results: Dict) -> str:
        """Estimar horizonte temporal"""
        
        # Si hay triángulos, usar su timing
        triangle_summary = triangle_results.get('summary', {})
        if triangle_summary.get('total_triangles', 0) > 0:
            breakout_prob = triangle_summary.get('max_breakout_prob', 0)
            if breakout_prob > 80:
                return "1-6 horas (breakout inminente)"
            elif breakout_prob > 60:
                return "6-12 horas (breakout probable)"
            else:
                return "12-24 horas (patrón en desarrollo)"
        
        # Si hay expansión de volatilidad
        vol_summary = volatility_results.get('summary', {})
        if vol_summary.get('is_expanding', False):
            return "2-8 horas (movimiento por volatilidad)"
        
        return "1-2 días (análisis adaptativo)"
    
    def _get_enhanced_market_context(self, features: MarketFeatures, triangle_results: Dict,
                                   volatility_results: Dict) -> str:
        """Obtener contexto enhanced del mercado"""
        
        contexts = []
        
        # Contexto base
        if features:
            if features.regime == "high_volatility":
                contexts.append("mercado volátil")
            elif features.regime == "trending":
                contexts.append("mercado en tendencia")
            else:
                contexts.append("mercado lateral")
        
        # Contexto de triángulos
        triangle_summary = triangle_results.get('summary', {})
        if triangle_summary.get('has_ascending'):
            contexts.append("patrón alcista de consolidación")
        elif triangle_summary.get('has_descending'):
            contexts.append("patrón bajista de consolidación")
        elif triangle_summary.get('has_symmetric'):
            contexts.append("patrón neutro de consolidación")
        
        # Contexto de volatilidad
        vol_summary = volatility_results.get('summary', {})
        if vol_summary.get('is_expanding', False):
            contexts.append("volatilidad en expansión")
        elif vol_summary.get('trend') == 'contracting':
            contexts.append("volatilidad en contracción")
        
        return ", ".join(contexts) if contexts else "mercado sin características especiales"


# =============================================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =============================================================================

def test_enhanced_system():
    """Prueba completa del sistema enhanced"""
    print("🚀 Iniciando sistema ENHANCED completo")
    print("=" * 100)
    
    # Inicializar sistema enhanced
    system = AdaptiveTradingMLSystemEnhanced(
        symbol="ETHUSD_PERP", 
        enable_ml=True,
        enable_logging=True,
        enable_triangles=True,
        enable_live_volatility=True
    )
    
    # Ejecutar análisis completo enhanced
    try:
        results = system.run_complete_enhanced_analysis(hours_3h=3, hours_48h=48)
        
        if results.get("status") == "success_enhanced":
            print("\n" + "=" * 100)
            print("🎯 RESUMEN EJECUTIVO ENHANCED:")
            print("=" * 100)
            print(f"📊 {results['executive_summary']}")
            
            print("\n🎯 SEÑAL ENHANCED:")
            enhanced_signal = results.get('enhanced_trading_signal', {})
            signal_type = enhanced_signal.get('signal_type', 'UNKNOWN')
            confidence = enhanced_signal.get('confidence', 0)
            
            if signal_type == "STRONG_BUY":
                print(f"🚀 {signal_type} - Confianza: {confidence:.0f}%")
            elif signal_type == "BUY":
                print(f"📈 {signal_type} - Confianza: {confidence:.0f}%")
            else:
                print(f"⏸️ {signal_type} - Confianza: {confidence:.0f}%")
            
            # Mostrar reasoning
            reasoning = enhanced_signal.get('reasoning', [])
            if reasoning:
                print("\n🧠 RAZONAMIENTO:")
                for i, reason in enumerate(reasoning, 1):
                    print(f"  {i}. {reason}")
            
            print("\n📋 RECOMENDACIONES:")
            recommendations = results.get('trading_recommendations', {})
            print(f"  🎯 Acción: {recommendations.get('primary_action', 'N/A')}")
            print(f"  📊 Confianza: {recommendations.get('confidence_level', 'N/A')}")
            print(f"  💰 Tamaño: {recommendations.get('position_sizing', 'N/A')}")
            print(f"  🚀 Entrada: {recommendations.get('entry_strategy', 'N/A')}")
            print(f"  ⏰ Horizonte: {recommendations.get('time_horizon', 'N/A')}")
            
            # Análisis enhanced específico
            triangle_analysis = results.get('triangle_analysis', {})
            triangle_summary = triangle_analysis.get('summary', {})
            if triangle_summary.get('total_triangles', 0) > 0:
                print(f"\n🔺 TRIÁNGULOS DETECTADOS: {triangle_summary['total_triangles']}")
                print(f"  📊 Máxima confianza: {triangle_summary.get('max_confidence', 0):.1f}%")
                print(f"  ⚡ Máx probabilidad breakout: {triangle_summary.get('max_breakout_prob', 0):.1f}%")
            
            volatility_analysis = results.get('volatility_analysis', {})
            vol_summary = volatility_analysis.get('summary', {})
            print(f"\n⚡ VOLATILIDAD LIVE:")
            print(f"  📊 Actual: {vol_summary.get('current_volatility', 0):.2f}%")
            print(f"  📈 Tendencia: {vol_summary.get('trend', 'unknown')}")
            print(f"  💥 Expandiendo: {'SÍ' if vol_summary.get('is_expanding', False) else 'NO'}")
            
            print("\n" + "=" * 100)
            print("✅ SISTEMA ENHANCED COMPLETADO EXITOSAMENTE")
            print("=" * 100)
            
        else:
            print("❌ Error en el sistema enhanced:")
            print(results.get('error', 'Error desconocido'))
            
    except Exception as e:
        print(f"❌ Error ejecutando sistema enhanced: {e}")


if __name__ == "__main__":
    test_enhanced_system()