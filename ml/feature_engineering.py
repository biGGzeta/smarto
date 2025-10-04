import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime
import re

@dataclass
class MarketFeatures:
    """Estructura completa de features extraídas del análisis adaptativo"""
    
    # Meta features (contexto)
    timestamp: datetime
    symbol: str
    regime: str  # 'high_volatility', 'low_volatility', 'trending', etc.
    volatility: float
    trend_strength: float
    direction_density: float
    
    # Range features
    price_position_pct: float  # Posición en rango semanal (0-100)
    range_3h_pct: float       # Volatilidad 3h
    range_48h_pct: float      # Volatilidad 48h  
    range_weekly_pct: float   # Volatilidad semanal
    current_price: float      # Precio actual
    week_min: float           # Mínimo semanal
    week_max: float           # Máximo semanal
    
    # Extremes features (últimos 3 días)
    maximos_trend: str        # 'crecientes', 'decrecientes', 'estables'
    minimos_trend: str
    maximos_strength: float   # 0-100
    minimos_strength: float
    extremes_count_max: int
    extremes_count_min: int
    extremes_alignment: bool  # ¿Máximos y mínimos van en misma dirección?
    
    # Zone features (48h)
    zone_high_time_pct: float    # Tiempo en zona alta
    zone_low_time_pct: float     # Tiempo en zona baja
    zone_high_threshold: float   # Umbral adaptativo usado
    zone_low_threshold: float
    zone_imbalance: float        # Diferencia entre tiempo alto vs bajo
    
    # Momentum features
    momentum_1d: float
    momentum_3d: float
    momentum_7d: float
    momentum_strength: float     # Abs del momentum más fuerte
    momentum_direction: str      # 'alcista', 'bajista', 'neutral'
    
    # Cycle features
    cycles_count: int
    cycles_quality_avg: float
    dominant_cycle_direction: str  # 'alcista', 'bajista', 'lateral'
    cycle_consistency: float       # ¿Qué tan consistentes son los ciclos?
    
    # Adaptive parameters (qué parámetros eligió el sistema)
    window_sizes_used: List[int]
    wick_threshold_used: float
    stability_threshold_used: float
    max_time_used: int
    
    # Wick activity features
    wickdowns_count_3h: int
    wickups_count_3h: int
    total_wicks_3h: int
    strongest_wick_pct: float
    wick_activity_ratio: float   # wicks/total_candles
    
    # Volume features (cuando esté disponible)
    volume_trend: str
    volume_spike: bool
    
    # Derived features (calculadas)
    market_stress: float         # Combinación de volatilidad + wick activity
    trend_momentum_alignment: float  # ¿Trend y momentum van juntos?
    zone_pressure: str           # 'high', 'low', 'balanced'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para ML"""
        return asdict(self)
    
    def to_ml_features(self) -> Dict[str, float]:
        """Convertir a features numéricas para ML"""
        return {
            # Features numéricas directas
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
            "direction_density": self.direction_density,
            "price_position_pct": self.price_position_pct,
            "range_3h_pct": self.range_3h_pct,
            "range_48h_pct": self.range_48h_pct,
            "range_weekly_pct": self.range_weekly_pct,
            
            # Momentum features
            "momentum_1d": self.momentum_1d,
            "momentum_3d": self.momentum_3d,
            "momentum_7d": self.momentum_7d,
            "momentum_strength": self.momentum_strength,
            
            # Zone features
            "zone_high_time_pct": self.zone_high_time_pct,
            "zone_low_time_pct": self.zone_low_time_pct,
            "zone_imbalance": self.zone_imbalance,
            
            # Extremes strength
            "maximos_strength": self.maximos_strength,
            "minimos_strength": self.minimos_strength,
            "extremes_count_max": float(self.extremes_count_max),
            "extremes_count_min": float(self.extremes_count_min),
            
            # Wick activity
            "wickdowns_count_3h": float(self.wickdowns_count_3h),
            "wickups_count_3h": float(self.wickups_count_3h),
            "total_wicks_3h": float(self.total_wicks_3h),
            "strongest_wick_pct": self.strongest_wick_pct,
            "wick_activity_ratio": self.wick_activity_ratio,
            
            # Cycles
            "cycles_count": float(self.cycles_count),
            "cycles_quality_avg": self.cycles_quality_avg,
            "cycle_consistency": self.cycle_consistency,
            
            # Categorical encoded (one-hot)
            "regime_high_volatility": 1.0 if self.regime == "high_volatility" else 0.0,
            "regime_low_volatility": 1.0 if self.regime == "low_volatility" else 0.0,
            "regime_trending": 1.0 if self.regime == "trending" else 0.0,
            "regime_choppy_ranging": 1.0 if self.regime == "choppy_ranging" else 0.0,
            "regime_smooth_ranging": 1.0 if self.regime == "smooth_ranging" else 0.0,
            
            "maximos_crecientes": 1.0 if self.maximos_trend == "crecientes" else 0.0,
            "maximos_decrecientes": 1.0 if self.maximos_trend == "decrecientes" else 0.0,
            "maximos_estables": 1.0 if self.maximos_trend == "estables" else 0.0,
            
            "minimos_crecientes": 1.0 if self.minimos_trend == "crecientes" else 0.0,
            "minimos_decrecientes": 1.0 if self.minimos_trend == "decrecientes" else 0.0,
            "minimos_estables": 1.0 if self.minimos_trend == "estables" else 0.0,
            
            "momentum_alcista": 1.0 if self.momentum_direction == "alcista" else 0.0,
            "momentum_bajista": 1.0 if self.momentum_direction == "bajista" else 0.0,
            "momentum_neutral": 1.0 if self.momentum_direction == "neutral" else 0.0,
            
            "cycle_direction_alcista": 1.0 if self.dominant_cycle_direction == "alcista" else 0.0,
            "cycle_direction_bajista": 1.0 if self.dominant_cycle_direction == "bajista" else 0.0,
            "cycle_direction_lateral": 1.0 if self.dominant_cycle_direction == "lateral" else 0.0,
            
            "zone_pressure_high": 1.0 if self.zone_pressure == "high" else 0.0,
            "zone_pressure_low": 1.0 if self.zone_pressure == "low" else 0.0,
            "zone_pressure_balanced": 1.0 if self.zone_pressure == "balanced" else 0.0,
            
            # Derived features
            "extremes_alignment": 1.0 if self.extremes_alignment else 0.0,
            "market_stress": self.market_stress,
            "trend_momentum_alignment": self.trend_momentum_alignment,
            
            # Adaptive parameters
            "wick_threshold_used": self.wick_threshold_used,
            "stability_threshold_used": self.stability_threshold_used,
            "max_time_used": float(self.max_time_used),
            "avg_window_size": float(np.mean(self.window_sizes_used)) if self.window_sizes_used else 5.0,
        }

class FeatureExtractor:
    """Extrae features estructuradas del análisis adaptativo"""
    
    def __init__(self):
        self.feature_history = []
        
    def extract_features(self, adaptive_results: Dict[str, Any]) -> MarketFeatures:
        """Extraer features de los resultados del análisis adaptativo"""
        
        print("🔧 Extrayendo features del análisis adaptativo...")
        
        # Extraer datos de cada análisis
        max_min_data = self._extract_max_min_features(adaptive_results.get('max_min_basic', {}))
        percentage_data = self._extract_percentage_features(adaptive_results.get('range_percentage_adaptive', {}))
        minimums_data = self._extract_minimums_features(adaptive_results.get('low_minimums_adaptive', {}))
        maximums_data = self._extract_maximums_features(adaptive_results.get('high_maximums_adaptive', {}))
        panorama_data = self._extract_panorama_features(adaptive_results.get('panorama_48h_adaptive', {}))
        weekly_data = self._extract_weekly_features(adaptive_results.get('weekly_adaptive', {}))
        
        # Calcular features derivadas
        extremes_alignment = (weekly_data.get('maximos_trend', 'unknown') == 
                            weekly_data.get('minimos_trend', 'unknown'))
        
        zone_imbalance = abs(panorama_data.get('high_zone_time_pct', 0) - 
                           panorama_data.get('low_zone_time_pct', 0))
        
        total_wicks = minimums_data.get('wickdowns_count', 0) + maximums_data.get('wickups_count', 0)
        wick_activity_ratio = total_wicks / 180 if total_wicks > 0 else 0  # 180 = 3h en minutos
        
        momentum_1d = weekly_data.get('momentum_1d', 0)
        momentum_3d = weekly_data.get('momentum_3d', 0)
        momentum_7d = weekly_data.get('momentum_7d', 0)
        momentum_strength = max(abs(momentum_1d), abs(momentum_3d), abs(momentum_7d))
        
        # Determinar dirección del momentum (CORREGIDO)
        if momentum_3d > 1:
            momentum_direction = "alcista"
        elif momentum_3d < -1:
            momentum_direction = "bajista"
        else:
            momentum_direction = "neutral"
        
        print(f"📊 Momentum calculado: 1d={momentum_1d:.1f}%, 3d={momentum_3d:.1f}%, 7d={momentum_7d:.1f}% → {momentum_direction}")
        
        # Calcular market stress
        volatility = weekly_data.get('volatility', 0)
        market_stress = (volatility * 0.6) + (wick_activity_ratio * 100 * 0.4)
        
        # Trend-momentum alignment
        trend_strength = weekly_data.get('trend_strength', 0)
        trend_momentum_alignment = min(trend_strength, momentum_strength / 10) if momentum_strength > 0 else 0
        
        # Zone pressure
        high_time = panorama_data.get('high_zone_time_pct', 0)
        low_time = panorama_data.get('low_zone_time_pct', 0)
        if high_time > low_time * 2:
            zone_pressure = "high"
        elif low_time > high_time * 2:
            zone_pressure = "low"
        else:
            zone_pressure = "balanced"
        
        # Cycle consistency
        cycles_count = percentage_data.get('cycles_count', 0)
        cycles_quality = percentage_data.get('cycles_quality_avg', 0)
        cycle_consistency = cycles_quality / 100 if cycles_count > 0 else 0
        
        # Combinar en estructura unificada
        features = MarketFeatures(
            # Meta
            timestamp=datetime.utcnow(),
            symbol=weekly_data.get('symbol', 'UNKNOWN'),
            regime=weekly_data.get('regime', 'unknown'),
            volatility=volatility,
            trend_strength=trend_strength,
            direction_density=weekly_data.get('direction_density', 0),
            
            # Ranges
            price_position_pct=weekly_data.get('range_position_pct', 50.0),
            range_3h_pct=max_min_data.get('percentage_range', 0.0),
            range_48h_pct=panorama_data.get('percentage_range', 0.0),
            range_weekly_pct=weekly_data.get('week_range_pct', 0.0),
            current_price=weekly_data.get('current_price', 0.0),
            week_min=weekly_data.get('week_min', 0.0),
            week_max=weekly_data.get('week_max', 0.0),
            
            # Extremes
            maximos_trend=weekly_data.get('maximos_trend', 'unknown'),
            minimos_trend=weekly_data.get('minimos_trend', 'unknown'),
            maximos_strength=weekly_data.get('maximos_strength', 0.0),
            minimos_strength=weekly_data.get('minimos_strength', 0.0),
            extremes_count_max=weekly_data.get('extremes_count_max', 0),
            extremes_count_min=weekly_data.get('extremes_count_min', 0),
            extremes_alignment=extremes_alignment,
            
            # Zones
            zone_high_time_pct=panorama_data.get('high_zone_time_pct', 0.0),
            zone_low_time_pct=panorama_data.get('low_zone_time_pct', 0.0),
            zone_high_threshold=panorama_data.get('zone_high_threshold', 80.0),
            zone_low_threshold=panorama_data.get('zone_low_threshold', 20.0),
            zone_imbalance=zone_imbalance,
            
            # Momentum
            momentum_1d=momentum_1d,
            momentum_3d=momentum_3d,
            momentum_7d=momentum_7d,
            momentum_strength=momentum_strength,
            momentum_direction=momentum_direction,
            
            # Cycles
            cycles_count=cycles_count,
            cycles_quality_avg=cycles_quality,
            dominant_cycle_direction=percentage_data.get('dominant_direction', 'lateral'),
            cycle_consistency=cycle_consistency,
            
            # Adaptive params
            window_sizes_used=minimums_data.get('window_sizes', []),
            wick_threshold_used=minimums_data.get('wick_threshold', 0.05),
            stability_threshold_used=weekly_data.get('stability_threshold', 0.5),
            max_time_used=minimums_data.get('max_time', 15),
            
            # Wicks
            wickdowns_count_3h=minimums_data.get('wickdowns_count', 0),
            wickups_count_3h=maximums_data.get('wickups_count', 0),
            total_wicks_3h=total_wicks,
            strongest_wick_pct=max(minimums_data.get('strongest_wick', 0.0), 
                                 maximums_data.get('strongest_wick', 0.0)),
            wick_activity_ratio=wick_activity_ratio,
            
            # Volume
            volume_trend='unknown',
            volume_spike=False,
            
            # Derived
            market_stress=market_stress,
            trend_momentum_alignment=trend_momentum_alignment,
            zone_pressure=zone_pressure
        )
        
        # Guardar en historial
        self.feature_history.append(features)
        
        print(f"✅ Features extraídas: {len(features.to_ml_features())} características numéricas")
        
        return features
    
    def _extract_max_min_features(self, data: Tuple) -> Dict:
        """Extraer features del análisis básico de máximos y mínimos"""
        if not isinstance(data, tuple) or len(data) < 2:
            return {"percentage_range": 0.0, "max_price": 0.0, "min_price": 0.0}
        
        detailed_data = data[1]
        if not isinstance(detailed_data, dict):
            return {"percentage_range": 0.0, "max_price": 0.0, "min_price": 0.0}
            
        return {
            'percentage_range': detailed_data.get('percentage_range', 0.0),
            'max_price': detailed_data.get('max_price', 0.0),
            'min_price': detailed_data.get('min_price', 0.0)
        }
    
    def _extract_percentage_features(self, data: Tuple) -> Dict:
        """Extraer features del análisis de porcentajes"""
        if not isinstance(data, tuple) or len(data) < 2:
            return {"cycles_count": 0, "cycles_quality_avg": 0.0, "dominant_direction": "lateral"}
        
        detailed_data = data[1]
        if not isinstance(detailed_data, dict):
            return {"cycles_count": 0, "cycles_quality_avg": 0.0, "dominant_direction": "lateral"}
        
        cycles = detailed_data.get('cycles', [])
        cycles_count = len(cycles)
        
        if cycles_count > 0:
            quality_avg = np.mean([c.get('quality_metric', 0) for c in cycles])
            
            # Determinar dirección dominante
            directions = [c.get('trend_direction', 'lateral') for c in cycles]
            direction_counts = {d: directions.count(d) for d in set(directions)}
            dominant_direction = max(direction_counts, key=direction_counts.get, default='lateral')
        else:
            quality_avg = 0.0
            dominant_direction = 'lateral'
        
        return {
            "cycles_count": cycles_count,
            "cycles_quality_avg": quality_avg,
            "dominant_direction": dominant_direction
        }
    
    def _extract_minimums_features(self, data: Tuple) -> Dict:
        """Extraer features del análisis de mínimos"""
        if not isinstance(data, tuple) or len(data) < 2:
            return {"wickdowns_count": 0, "strongest_wick": 0.0, "window_sizes": [], "wick_threshold": 0.05, "max_time": 15}
        
        detailed_data = data[1]
        if not isinstance(detailed_data, dict):
            return {"wickdowns_count": 0, "strongest_wick": 0.0, "window_sizes": [], "wick_threshold": 0.05, "max_time": 15}
        
        ranked_minimums = detailed_data.get('ranked_minimums', [])
        wickdowns = [m for m in ranked_minimums if m.get('type') == 'wickdown']
        
        strongest_wick = 0.0
        if wickdowns:
            strongest_wick = max([w.get('wick_size_pct', 0) for w in wickdowns])
        
        adaptive_params = detailed_data.get('adaptive_parameters', {})
        
        return {
            "wickdowns_count": len(wickdowns),
            "strongest_wick": strongest_wick,
            "window_sizes": adaptive_params.get('window_sizes', []),
            "wick_threshold": adaptive_params.get('wick_min_pct', 0.05),
            "max_time": adaptive_params.get('max_time_minutes', 15)
        }
    
    def _extract_maximums_features(self, data: Tuple) -> Dict:
        """Extraer features del análisis de máximos"""
        if not isinstance(data, tuple) or len(data) < 2:
            return {"wickups_count": 0, "strongest_wick": 0.0}
        
        detailed_data = data[1]
        if not isinstance(detailed_data, dict):
            return {"wickups_count": 0, "strongest_wick": 0.0}
        
        ranked_maximums = detailed_data.get('ranked_maximums', [])
        wickups = [m for m in ranked_maximums if m.get('type') == 'wickup']
        
        strongest_wick = 0.0
        if wickups:
            strongest_wick = max([w.get('wick_size_pct', 0) for w in wickups])
        
        return {
            "wickups_count": len(wickups),
            "strongest_wick": strongest_wick
        }
    
    def _extract_panorama_features(self, data: Tuple) -> Dict:
        """Extraer features del análisis de panorama 48h"""
        if not isinstance(data, tuple) or len(data) < 2:
            return {"percentage_range": 0.0, "high_zone_time_pct": 0.0, "low_zone_time_pct": 0.0, 
                   "zone_high_threshold": 80.0, "zone_low_threshold": 20.0}
        
        detailed_data = data[1]
        if not isinstance(detailed_data, dict):
            return {"percentage_range": 0.0, "high_zone_time_pct": 0.0, "low_zone_time_pct": 0.0,
                   "zone_high_threshold": 80.0, "zone_low_threshold": 20.0}
        
        high_zone = detailed_data.get('high_zone_analysis', {})
        low_zone = detailed_data.get('low_zone_analysis', {})
        
        return {
            "percentage_range": detailed_data.get('percentage_range', 0.0),
            "high_zone_time_pct": high_zone.get('time_percentage', 0.0),
            "low_zone_time_pct": low_zone.get('time_percentage', 0.0),
            "zone_high_threshold": detailed_data.get('zone_high_threshold', 80.0),
            "zone_low_threshold": detailed_data.get('zone_low_threshold', 20.0)
        }
    
    def _extract_weekly_features(self, data: Tuple) -> Dict:
        """Extraer features del análisis semanal - CORREGIDO MOMENTUM"""
        if not isinstance(data, tuple) or len(data) < 2:
            return {
                'symbol': 'UNKNOWN', 'regime': 'unknown', 'volatility': 0.0, 'trend_strength': 0.0,
                'direction_density': 0.0, 'range_position_pct': 50.0, 'week_range_pct': 0.0,
                'current_price': 0.0, 'week_min': 0.0, 'week_max': 0.0,
                'maximos_trend': 'unknown', 'minimos_trend': 'unknown',
                'maximos_strength': 0.0, 'minimos_strength': 0.0,
                'extremes_count_max': 0, 'extremes_count_min': 0,
                'momentum_1d': 0.0, 'momentum_3d': 0.0, 'momentum_7d': 0.0,
                'stability_threshold': 0.5
            }
        
        detailed_data = data[1]
        if not isinstance(detailed_data, dict):
            return {
                'symbol': 'UNKNOWN', 'regime': 'unknown', 'volatility': 0.0, 'trend_strength': 0.0,
                'direction_density': 0.0, 'range_position_pct': 50.0, 'week_range_pct': 0.0,
                'current_price': 0.0, 'week_min': 0.0, 'week_max': 0.0,
                'maximos_trend': 'unknown', 'minimos_trend': 'unknown',
                'maximos_strength': 0.0, 'minimos_strength': 0.0,
                'extremes_count_max': 0, 'extremes_count_min': 0,
                'momentum_1d': 0.0, 'momentum_3d': 0.0, 'momentum_7d': 0.0,
                'stability_threshold': 0.5
            }
        
        market_conditions = detailed_data.get('market_conditions', {})
        maximos_trend = detailed_data.get('maximos_trend', {})
        minimos_trend = detailed_data.get('minimos_trend', {})
        momentum = detailed_data.get('weekly_momentum', {})
        adaptive_params = detailed_data.get('adaptive_parameters', {})
        
        momentum_by_period = momentum.get('momentum_by_period', {})
        
        # 🔧 CORREGIDO: Extraer momentum con claves flexibles
        momentum_1d = 0.0
        momentum_3d = 0.0
        momentum_7d = 0.0
        
        print(f"🔍 Momentum disponible en datos: {list(momentum_by_period.keys())}")
        
        # Buscar claves posibles para cada período
        for key, value in momentum_by_period.items():
            if key in ['1d', '24h', '1day']:
                momentum_1d = value
                print(f"📊 Momentum 1d encontrado: {value}% (key: {key})")
            elif key in ['3d', '72h', '3day', '3days']:
                momentum_3d = value
                print(f"📊 Momentum 3d encontrado: {value}% (key: {key})")
            elif key in ['7d', '168h', '1w', 'week', '7day', '7days']:
                momentum_7d = value
                print(f"📊 Momentum 7d encontrado: {value}% (key: {key})")
        
        # 🔧 FALLBACK: Si no hay momentum específico, calcularlo desde extremos
        if momentum_3d == 0.0:
            print("🔧 Momentum 3d es 0, intentando extraer desde análisis de extremos...")
            
            # Usar datos de extremos para calcular momentum aproximado
            if (maximos_trend.get('trend') == 'crecientes' and 
                minimos_trend.get('trend') == 'crecientes'):
                
                print("✅ Extremos alineados crecientes detectados")
                
                # Extraer cambio porcentual de los extremos
                maximos_analysis = maximos_trend.get('analysis', '')
                print(f"🔍 Análisis de máximos: {maximos_analysis}")
                
                if '(+' in maximos_analysis:
                    # Buscar el porcentaje en el string: "+8.1%"
                    match = re.search(r'\(\+(\d+\.?\d*)%\)', maximos_analysis)
                    if match:
                        momentum_3d = float(match.group(1))
                        print(f"🔧 Momentum 3d extraído de análisis de máximos: +{momentum_3d}%")
                
                # Si no encontramos en máximos, buscar en mínimos
                if momentum_3d == 0.0:
                    minimos_analysis = minimos_trend.get('analysis', '')
                    print(f"🔍 Análisis de mínimos: {minimos_analysis}")
                    
                    if '(+' in minimos_analysis:
                        match = re.search(r'\(\+(\d+\.?\d*)%\)', minimos_analysis)
                        if match:
                            momentum_3d = float(match.group(1))
                            print(f"🔧 Momentum 3d extraído de análisis de mínimos: +{momentum_3d}%")
            
            elif (maximos_trend.get('trend') == 'decrecientes' and 
                  minimos_trend.get('trend') == 'decrecientes'):
                
                print("⬇️ Extremos alineados decrecientes detectados")
                
                # Para extremos decrecientes, buscar porcentaje negativo
                maximos_analysis = maximos_trend.get('analysis', '')
                if '(-' in maximos_analysis:
                    match = re.search(r'\(-(\d+\.?\d*)%\)', maximos_analysis)
                    if match:
                        momentum_3d = -float(match.group(1))  # Negativo
                        print(f"🔧 Momentum 3d extraído (bajista): {momentum_3d}%")
        
        print(f"📊 Momentum final extraído: 1d={momentum_1d}%, 3d={momentum_3d}%, 7d={momentum_7d}%")
        
        return {
            'symbol': market_conditions.get('symbol', 'UNKNOWN'),
            'regime': market_conditions.get('market_regime', 'unknown'),
            'volatility': market_conditions.get('volatility', 0.0),
            'trend_strength': market_conditions.get('trend_strength', 0.0),
            'direction_density': market_conditions.get('direction_density', 0.0),
            'range_position_pct': detailed_data.get('range_position_pct', 50.0),
            'week_range_pct': detailed_data.get('week_range_pct', 0.0),
            'current_price': detailed_data.get('current_price', 0.0),
            'week_min': detailed_data.get('week_min', 0.0),
            'week_max': detailed_data.get('week_max', 0.0),
            'maximos_trend': maximos_trend.get('trend', 'unknown'),
            'minimos_trend': minimos_trend.get('trend', 'unknown'),
            'maximos_strength': maximos_trend.get('trend_strength', 0.0),
            'minimos_strength': minimos_trend.get('trend_strength', 0.0),
            'extremes_count_max': maximos_trend.get('extremes_count', 0),
            'extremes_count_min': minimos_trend.get('extremes_count', 0),
            'momentum_1d': momentum_1d,  # 🔧 CORREGIDO
            'momentum_3d': momentum_3d,  # 🔧 CORREGIDO  
            'momentum_7d': momentum_7d,  # 🔧 CORREGIDO
            'stability_threshold': adaptive_params.get('stability_threshold', 0.5)
        }
    
    def save_features_to_csv(self, filepath: str = "features_history.csv"):
        """Guardar historial de features en CSV para análisis ML"""
        if not self.feature_history:
            print("❌ No hay features para guardar")
            return
        
        # Convertir a DataFrame
        features_data = []
        for features in self.feature_history:
            row = features.to_ml_features()
            row['timestamp'] = features.timestamp
            row['symbol'] = features.symbol
            features_data.append(row)
        
        df = pd.DataFrame(features_data)
        df.to_csv(filepath, index=False)
        
        print(f"💾 Features guardadas en {filepath}: {len(df)} filas, {len(df.columns)} columnas")