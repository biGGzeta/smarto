import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime
import json

class AdaptiveParameters:
    """Sistema que ajusta parámetros basado en condiciones del mercado en tiempo real"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.market_conditions = {}
        self.parameter_history = []
        self.calibration_data = {}
        
    def analyze_market_conditions(self, data: pd.DataFrame, timeframe: str = "3h") -> Dict:
        """Analizar condiciones actuales del mercado para ajustar parámetros"""
        
        if data.empty:
            return self._get_default_conditions()
        
        # 1. Volatilidad (desviación estándar de retornos)
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * 100 * np.sqrt(1440)  # Anualizada
        
        # 2. Rango del período (cuán amplio es el movimiento)
        price_range_pct = ((data['high'].max() - data['low'].min()) / data['low'].min()) * 100
        
        # 3. Densidad de precio (cuántos cambios de dirección)
        price_changes = np.diff(data['close'])
        direction_changes = np.sum(np.diff(np.sign(price_changes)) != 0)
        density = direction_changes / len(data) * 100
        
        # 4. Tendencia (correlación temporal)
        x = np.arange(len(data))
        y = data['close'].values
        if len(x) > 1:
            correlation = np.corrcoef(x, y)[0, 1]
            trend_strength = abs(correlation)
        else:
            trend_strength = 0
        
        # 5. Momento (aceleración del precio)
        price_momentum = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100
        
        # 6. Distribución de extremos
        highs_std = data['high'].rolling(window=20).std().iloc[-1] if len(data) > 20 else data['high'].std()
        lows_std = data['low'].rolling(window=20).std().iloc[-1] if len(data) > 20 else data['low'].std()
        extremes_volatility = (highs_std + lows_std) / 2
        
        # 7. Clasificación de régimen
        regime = self._classify_market_regime(volatility, trend_strength, density)
        
        conditions = {
            "timestamp": datetime.utcnow(),
            "timeframe": timeframe,
            "volatility": round(volatility, 3),
            "price_range_pct": round(price_range_pct, 3),
            "direction_density": round(density, 3),
            "trend_strength": round(trend_strength, 3),
            "price_momentum": round(price_momentum, 3),
            "extremes_volatility": round(extremes_volatility, 3),
            "market_regime": regime,
            "data_points": len(data),
            "current_price": round(data['close'].iloc[-1], 2)
        }
        
        self.market_conditions = conditions
        print(f"📊 Condiciones del mercado analizadas:")
        print(f"   🔥 Volatilidad: {volatility:.2f}%")
        print(f"   📈 Rango: {price_range_pct:.2f}%")
        print(f"   🌊 Densidad: {density:.1f} cambios/100 velas")
        print(f"   📊 Tendencia: {trend_strength:.2f} (0=lateral, 1=fuerte)")
        print(f"   🎯 Régimen: {regime}")
        
        return conditions
    
    def get_adaptive_parameters(self, analysis_type: str = "general") -> Dict:
        """Obtener parámetros optimizados para las condiciones actuales"""
        
        if not self.market_conditions:
            print("⚠️ Sin análisis de condiciones, usando parámetros por defecto")
            return self._get_default_parameters()
        
        volatility = self.market_conditions["volatility"]
        trend_strength = self.market_conditions["trend_strength"]
        density = self.market_conditions["direction_density"]
        regime = self.market_conditions["market_regime"]
        price_range = self.market_conditions["price_range_pct"]
        
        print(f"🔧 Adaptando parámetros para régimen: {regime}")
        
        # Parámetros base
        params = {}
        
        # 1. WINDOW SIZES (para detección de extremos)
        if regime == "high_volatility":
            # Mercado muy volátil: ventanas pequeñas para captar movimientos rápidos
            params["window_sizes"] = [2, 3, 4]
            params["min_separation_minutes"] = 5
        elif regime == "low_volatility": 
            # Mercado calmo: ventanas grandes para evitar ruido
            params["window_sizes"] = [7, 10, 15]
            params["min_separation_minutes"] = 30
        elif regime == "trending":
            # Mercado en tendencia: ventanas medianas, enfoque en breakouts
            params["window_sizes"] = [4, 6, 8]
            params["min_separation_minutes"] = 15
        else:  # ranging
            # Mercado lateral: ventanas estándar
            params["window_sizes"] = [3, 5, 7]
            params["min_separation_minutes"] = 20
        
        # 2. WICK DETECTION (sensibilidad a wicks)
        if volatility < 5:
            params["wick_min_pct"] = 0.02  # Muy sensible para mercados calmos
        elif volatility > 15:
            params["wick_min_pct"] = 0.10  # Menos sensible para mercados locos
        else:
            params["wick_min_pct"] = 0.03 + (volatility - 5) * 0.007  # Escalado
        
        # 3. STABILITY THRESHOLD (cuándo considerar extremos "estables")
        base_threshold = max(0.3, volatility * 0.4)
        if regime == "trending":
            params["stability_threshold"] = base_threshold * 1.5  # Más permisivo
        else:
            params["stability_threshold"] = base_threshold
        
        # 4. TIME THRESHOLDS (qué se considera "poco tiempo")
        if density > 15:  # Mercado muy activo
            params["max_time_minutes"] = 8
            params["short_time_minutes"] = 5
        elif density < 5:   # Mercado poco activo
            params["max_time_minutes"] = 25
            params["short_time_minutes"] = 15
        else:
            params["max_time_minutes"] = 12 + int(density * 0.5)
            params["short_time_minutes"] = 8 + int(density * 0.3)
        
        # 5. ZONE THRESHOLDS (zonas altas/bajas adaptativos)
        if price_range < 1.5:  # Rango muy estrecho
            params["zone_high"] = 70
            params["zone_low"] = 30
        elif price_range > 10:  # Rango muy amplio
            params["zone_high"] = 90
            params["zone_low"] = 10
        else:
            # Interpolación lineal
            range_factor = (price_range - 1.5) / 8.5  # Normalizar 1.5-10 a 0-1
            params["zone_high"] = 70 + (range_factor * 20)  # 70-90
            params["zone_low"] = 30 - (range_factor * 20)   # 30-10
        
        # 6. SIGNIFICANCE WEIGHTS (cómo priorizar extremos)
        if regime == "trending":
            params["depth_weight"] = 0.5      # Más importancia a profundidad
            params["time_weight"] = 0.2       # Menos a tiempo
            params["context_weight"] = 0.3    # Contexto importante
        elif regime == "high_volatility":
            params["depth_weight"] = 0.3      # Menos a profundidad (mucho ruido)
            params["time_weight"] = 0.4       # Más a velocidad
            params["context_weight"] = 0.3
        else:  # ranging o low volatility
            params["depth_weight"] = 0.3
            params["time_weight"] = 0.3
            params["context_weight"] = 0.4    # Más a contexto
        
        # 7. CYCLE DETECTION (para análisis de porcentajes)
        if trend_strength > 0.7:
            params["cycle_overlap_tolerance"] = 0.3  # Más overlap OK en tendencias
            params["min_cycle_duration"] = 15       # Ciclos más cortos
        else:
            params["cycle_overlap_tolerance"] = 0.1  # Menos overlap en ranging
            params["min_cycle_duration"] = 30       # Ciclos más largos
        
        # 8. CONFIRMATION REQUIREMENTS
        if regime == "high_volatility":
            params["confirmation_touches"] = 3  # Más confirmaciones en volatilidad
        else:
            params["confirmation_touches"] = 2  # Menos en mercados estables
        
        # Redondear valores numéricos
        for key, value in params.items():
            if isinstance(value, float):
                params[key] = round(value, 3)
        
        # Guardar en historial
        self._save_parameters_to_history(params)
        
        print(f"✅ Parámetros adaptativos generados:")
        print(f"   🔍 Windows: {params['window_sizes']}")
        print(f"   ⚡ Wick threshold: {params['wick_min_pct']:.3f}%")
        print(f"   ⏱️ Max time: {params['max_time_minutes']}min")
        print(f"   🎯 Zonas: {params['zone_low']:.0f}%-{params['zone_high']:.0f}%")
        print(f"   📊 Estabilidad: {params['stability_threshold']:.3f}%")
        
        return params
    
    def _classify_market_regime(self, volatility: float, trend_strength: float, density: float) -> str:
        """Clasificar el régimen actual del mercado"""
        
        if volatility > 15:
            return "high_volatility"
        elif volatility < 3:
            return "low_volatility"
        elif trend_strength > 0.6:
            return "trending"
        elif density > 12:
            return "choppy_ranging"
        else:
            return "smooth_ranging"
    
    def _get_default_conditions(self) -> Dict:
        """Condiciones por defecto cuando no se puede analizar"""
        return {
            "timestamp": datetime.utcnow(),
            "volatility": 7.0,
            "price_range_pct": 3.0,
            "direction_density": 8.0,
            "trend_strength": 0.3,
            "market_regime": "smooth_ranging",
            "data_points": 0
        }
    
    def _get_default_parameters(self) -> Dict:
        """Parámetros por defecto cuando no hay contexto"""
        return {
            "window_sizes": [3, 5, 7],
            "wick_min_pct": 0.05,
            "stability_threshold": 0.5,
            "max_time_minutes": 15,
            "short_time_minutes": 10,
            "zone_high": 80,
            "zone_low": 20,
            "depth_weight": 0.33,
            "time_weight": 0.33,
            "context_weight": 0.34,
            "min_separation_minutes": 20,
            "cycle_overlap_tolerance": 0.2,
            "min_cycle_duration": 25,
            "confirmation_touches": 2
        }
    
    def _save_parameters_to_history(self, params: Dict):
        """Guardar parámetros en historial para análisis futuro"""
        entry = {
            "timestamp": datetime.utcnow(),
            "market_conditions": self.market_conditions.copy(),
            "parameters": params.copy()
        }
        
        self.parameter_history.append(entry)
        
        # Mantener solo últimos 100 registros
        if len(self.parameter_history) > 100:
            self.parameter_history = self.parameter_history[-100:]
    
    def get_parameters_performance(self) -> Dict:
        """Analizar rendimiento de parámetros históricos (para futuro ML)"""
        if not self.parameter_history:
            return {"status": "no_data"}
        
        # Por ahora, retornar estadísticas básicas
        volatilities = [entry["market_conditions"]["volatility"] for entry in self.parameter_history]
        regimes = [entry["market_conditions"]["market_regime"] for entry in self.parameter_history]
        
        return {
            "entries_count": len(self.parameter_history),
            "avg_volatility": np.mean(volatilities),
            "regime_distribution": {regime: regimes.count(regime) for regime in set(regimes)},
            "last_update": self.parameter_history[-1]["timestamp"]
        }