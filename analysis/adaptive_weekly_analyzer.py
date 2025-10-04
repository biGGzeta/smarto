from .adaptive_base_analyzer import AdaptiveBaseAnalyzer
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

class AdaptiveWeeklyAnalyzer(AdaptiveBaseAnalyzer):
    """Analizador semanal con extremos adaptativos de 3 días"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Implementación del método abstracto requerido"""
        recent_days = question_params.get('recent_days', 3)
        simple_answer, _ = self.analyze_weekly_with_recent_extremes_adaptive(recent_days)
        return simple_answer
    
    def analyze_weekly_with_recent_extremes_adaptive(self, recent_days: float = 3) -> Tuple[str, Dict[str, Any]]:
        """
        Análisis semanal completo con tendencia de extremos adaptativos de últimos 3 días
        """
        if self.data is None or self.data.empty:
            return "Sin datos", {}
        
        # Obtener datos semanales (168 horas)
        weekly_data = self.get_time_range_data(168)
        
        if weekly_data.empty:
            return "Sin datos para análisis semanal", {}
        
        print(f"🔄 Iniciando análisis semanal adaptativo con extremos de {recent_days} días...")
        
        # Analizar condiciones del mercado para toda la semana
        self.load_data_and_analyze_conditions(weekly_data, "7d")
        
        # Análisis básico semanal
        week_max = weekly_data['high'].max()
        week_min = weekly_data['low'].min()
        week_range = week_max - week_min
        week_range_pct = (week_range / week_min) * 100
        
        # Precio actual y posición en el rango
        current_price = weekly_data['close'].iloc[-1]
        range_position = ((current_price - week_min) / week_range) * 100
        
        # Análisis de extremos recientes (3 días) con parámetros adaptativos
        recent_hours = recent_days * 24
        recent_data = self.get_time_range_data(recent_hours)
        
        if not recent_data.empty:
            # Usar análisis adaptativo para los extremos recientes
            maximos_trend = self._analyze_adaptive_extremes_trend(recent_data, 'maximos', recent_days)
            minimos_trend = self._analyze_adaptive_extremes_trend(recent_data, 'minimos', recent_days)
        else:
            maximos_trend = {"trend": "sin_datos", "analysis": "sin datos suficientes"}
            minimos_trend = {"trend": "sin_datos", "analysis": "sin datos suficientes"}
        
        # Contexto de posición adaptativo
        position_context = self._get_adaptive_position_context(range_position, week_range_pct)
        
        # Análisis de momentum semanal adaptativo
        weekly_momentum = self._analyze_adaptive_weekly_momentum(weekly_data)
        
        # Formatear respuesta
        simple_answer = self._format_adaptive_weekly_response(
            week_min, week_max, week_range_pct, current_price, 
            range_position, position_context, maximos_trend, minimos_trend, weekly_momentum
        )
        
        detailed_data = {
            "analysis_type": "adaptive_weekly_with_recent_extremes",
            "week_min": week_min,
            "week_max": week_max,
            "week_range": week_range,
            "week_range_pct": week_range_pct,
            "current_price": current_price,
            "range_position_pct": range_position,
            "position_context": position_context,
            "recent_days_analyzed": recent_days,
            "maximos_trend": maximos_trend,
            "minimos_trend": minimos_trend,
            "weekly_momentum": weekly_momentum,
            "market_conditions": self.market_conditions,
            "adaptive_parameters": self.current_parameters,
            "weekly_data_points": len(weekly_data),
            "recent_data_points": len(recent_data) if not recent_data.empty else 0
        }
        
        return simple_answer, detailed_data
    
    def _analyze_adaptive_extremes_trend(self, data: pd.DataFrame, extreme_type: str, days: float) -> Dict:
        """Analizar tendencia de extremos con parámetros adaptativos"""
        
        if extreme_type == 'maximos':
            extremes = self._detect_adaptive_local_maxima(data)
            price_key = 'high'
        else:
            extremes = self._detect_adaptive_local_minima(data)
            price_key = 'low'
        
        if len(extremes) < 2:
            return {
                "trend": "insuficiente",
                "extremes_count": len(extremes),
                "analysis": f"solo {len(extremes)} {extreme_type} detectados en {days} días",
                "trend_strength": 0,
                "avg_change_pct": 0,
                "adaptive_analysis": True
            }
        
        # Calcular cambios con pesos adaptativos
        price_changes = []
        weighted_changes = []
        
        regime = self.market_conditions.get("market_regime", "smooth_ranging")
        
        for i in range(1, len(extremes)):
            prev_price = extremes[i-1]['price']
            curr_price = extremes[i]['price']
            change_pct = ((curr_price - prev_price) / prev_price) * 100
            
            # Peso basado en significancia adaptativa
            time_weight = self._calculate_time_weight(extremes[i-1], extremes[i], regime)
            
            price_changes.append(change_pct)
            weighted_changes.append(change_pct * time_weight)
        
        # Determinar tendencia con umbrales adaptativos
        avg_change = np.mean(price_changes)
        weighted_avg_change = np.mean(weighted_changes)
        std_change = np.std(price_changes) if len(price_changes) > 1 else 0
        
        # Umbrales adaptativos según volatilidad y período
        volatility = self.market_conditions.get("volatility", 5.0)
        base_threshold = max(0.5, volatility * 0.15)  # Umbral base adaptativo
        
        if regime == "high_volatility":
            stability_threshold = base_threshold * 1.5  # Más permisivo en volatilidad
        elif regime == "trending":
            stability_threshold = base_threshold * 0.8   # Más estricto en tendencias
        else:
            stability_threshold = base_threshold
        
        # Clasificar tendencia
        if abs(weighted_avg_change) <= stability_threshold and std_change <= stability_threshold * 1.2:
            trend = "estables"
            trend_strength = 100 - (std_change / stability_threshold * 50)  # Más estable = mayor strength
        elif weighted_avg_change > stability_threshold:
            trend = "crecientes"
            trend_strength = min(100, abs(weighted_avg_change) / stability_threshold * 50)
        else:
            trend = "decrecientes"
            trend_strength = min(100, abs(weighted_avg_change) / stability_threshold * 50)
        
        return {
            "trend": trend,
            "extremes_count": len(extremes),
            "analysis": self._format_adaptive_extremes_analysis(extreme_type, trend, extremes, price_changes, days),
            "trend_strength": round(trend_strength, 1),
            "avg_change_pct": round(avg_change, 2),
            "weighted_avg_change_pct": round(weighted_avg_change, 2),
            "std_change_pct": round(std_change, 2),
            "stability_threshold_used": round(stability_threshold, 3),
            "regime": regime,
            "extremes_data": extremes,
            "adaptive_analysis": True
        }
    
    def _detect_adaptive_local_maxima(self, data: pd.DataFrame) -> List[Dict]:
        """Detectar máximos locales con ventanas adaptativas"""
        maxima = []
        
        # Usar ventanas adaptativas
        window_sizes = self.get_adaptive_window_sizes()
        regime = self.market_conditions.get("market_regime", "smooth_ranging")
        
        # Seleccionar ventana según régimen para análisis de 3 días
        if regime == "high_volatility":
            primary_window = max(window_sizes)  # Ventana más grande para filtrar ruido
        elif regime == "trending":
            primary_window = min(window_sizes)  # Ventana más pequeña para captar cambios
        else:
            primary_window = window_sizes[1] if len(window_sizes) > 1 else window_sizes[0]
        
        print(f"🔍 Detectando máximos con ventana adaptativa: {primary_window} (régimen: {regime})")
        
        for i in range(primary_window, len(data) - primary_window):
            current_high = data['high'].iloc[i]
            current_time = data.index[i]
            
            # Verificar si es máximo local
            window_highs = data['high'].iloc[i-primary_window:i+primary_window+1]
            
            if current_high == window_highs.max():
                # Calcular significancia adaptativa
                significance = self._calculate_adaptive_extreme_significance(
                    current_high, i, data, 'maximum', regime
                )
                
                maxima.append({
                    'timestamp': current_time,
                    'time_str': current_time.strftime('%m/%d %H:%M'),
                    'price': current_high,
                    'index': i,
                    'significance_score': significance,
                    'window_used': primary_window,
                    'regime': regime
                })
        
        # Filtrar máximos cercanos con criterios adaptativos
        filtered_maxima = self._filter_adaptive_close_extremes(maxima, 'maxima')
        
        print(f"✅ Detectados {len(filtered_maxima)} máximos adaptativos")
        return filtered_maxima
    
    def _detect_adaptive_local_minima(self, data: pd.DataFrame) -> List[Dict]:
        """Detectar mínimos locales con ventanas adaptativas"""
        minima = []
        
        window_sizes = self.get_adaptive_window_sizes()
        regime = self.market_conditions.get("market_regime", "smooth_ranging")
        
        if regime == "high_volatility":
            primary_window = max(window_sizes)
        elif regime == "trending":
            primary_window = min(window_sizes)
        else:
            primary_window = window_sizes[1] if len(window_sizes) > 1 else window_sizes[0]
        
        print(f"🔍 Detectando mínimos con ventana adaptativa: {primary_window} (régimen: {regime})")
        
        for i in range(primary_window, len(data) - primary_window):
            current_low = data['low'].iloc[i]
            current_time = data.index[i]
            
            window_lows = data['low'].iloc[i-primary_window:i+primary_window+1]
            
            if current_low == window_lows.min():
                significance = self._calculate_adaptive_extreme_significance(
                    current_low, i, data, 'minimum', regime
                )
                
                minima.append({
                    'timestamp': current_time,
                    'time_str': current_time.strftime('%m/%d %H:%M'),
                    'price': current_low,
                    'index': i,
                    'significance_score': significance,
                    'window_used': primary_window,
                    'regime': regime
                })
        
        filtered_minima = self._filter_adaptive_close_extremes(minima, 'minima')
        
        print(f"✅ Detectados {len(filtered_minima)} mínimos adaptativos")
        return filtered_minima
    
    def _calculate_adaptive_extreme_significance(self, price: float, index: int, 
                                               data: pd.DataFrame, extreme_type: str, regime: str) -> float:
        """Calcular significancia de extremo con criterios adaptativos"""
        
        # Obtener pesos adaptativos
        weights = self.get_significance_weights()
        
        # Factor 1: Posición en el rango (adaptativo)
        data_range = data['high'].max() - data['low'].min()
        if extreme_type == 'maximum':
            position_factor = ((price - data['low'].min()) / data_range) * 100
        else:  # minimum
            position_factor = ((data['high'].max() - price) / data_range) * 100
        
        position_score = position_factor * weights['depth']
        
        # Factor 2: Aislamiento temporal (adaptativo según régimen)
        if regime == "high_volatility":
            isolation_window = 30  # Ventana más grande en volatilidad
        elif regime == "trending":
            isolation_window = 15  # Ventana más pequeña en tendencias
        else:
            isolation_window = 20  # Ventana estándar
        
        isolation_score = self._calculate_isolation_score(price, index, data, extreme_type, isolation_window)
        isolation_score *= weights['time']
        
        # Factor 3: Contexto del régimen (adaptativo)
        if regime == "trending":
            if extreme_type == 'maximum':
                regime_bonus = 20  # Máximos en tendencia son muy significativos
            else:
                regime_bonus = 15  # Mínimos en tendencia también importantes
        elif regime == "high_volatility":
            regime_bonus = 10  # Menos bonus en volatilidad (más comunes)
        else:
            regime_bonus = 15  # Bonus estándar
        
        context_score = regime_bonus * weights['context']
        
        # Factor 4: Volumen relativo (si disponible)
        volume_bonus = 0
        if 'volume' in data.columns:
            avg_volume = data['volume'].mean()
            current_volume = data['volume'].iloc[index]
            if current_volume > avg_volume * 1.5:
                volume_bonus = 10  # Bonus por alto volumen
        
        total_significance = position_score + isolation_score + context_score + volume_bonus
        
        return min(total_significance, 100)
    
    def _calculate_isolation_score(self, price: float, index: int, data: pd.DataFrame, 
                                 extreme_type: str, window: int) -> float:
        """Calcular score de aislamiento del extremo"""
        
        start_idx = max(0, index - window)
        end_idx = min(len(data), index + window + 1)
        
        window_data = data.iloc[start_idx:end_idx]
        
        if extreme_type == 'maximum':
            other_highs = window_data['high'][window_data['high'] != price]
            if len(other_highs) > 0:
                distance_pct = ((price - other_highs.max()) / price) * 100
            else:
                distance_pct = 5  # Valor por defecto
        else:  # minimum
            other_lows = window_data['low'][window_data['low'] != price]
            if len(other_lows) > 0:
                distance_pct = ((other_lows.min() - price) / price) * 100
            else:
                distance_pct = 5
        
        # Convertir distancia a score (más distancia = más aislado = mayor score)
        isolation_score = min(100, distance_pct * 20)
        
        return max(0, isolation_score)
    
    def _filter_adaptive_close_extremes(self, extremes: List[Dict], extreme_type: str) -> List[Dict]:
        """Filtrar extremos cercanos con criterios adaptativos"""
        
        if len(extremes) <= 1:
            return extremes
        
        # Separación mínima adaptativa (en horas)
        regime = self.market_conditions.get("market_regime", "smooth_ranging")
        if regime == "high_volatility":
            min_separation_hours = 4  # Más separación en volatilidad
        elif regime == "trending":
            min_separation_hours = 2  # Menos separación en tendencias (cambios rápidos)
        else:
            min_separation_hours = 3  # Separación estándar
        
        extremes.sort(key=lambda x: x['timestamp'])
        filtered = [extremes[0]]
        
        for extreme in extremes[1:]:
            last_kept = filtered[-1]
            time_diff_hours = abs((extreme['timestamp'] - last_kept['timestamp']).total_seconds()) / 3600
            
            if time_diff_hours >= min_separation_hours:
                filtered.append(extreme)
            else:
                # Mantener el más significativo
                if extreme['significance_score'] > last_kept['significance_score']:
                    filtered[-1] = extreme
        
        return filtered
    
    def _calculate_time_weight(self, prev_extreme: Dict, curr_extreme: Dict, regime: str) -> float:
        """Calcular peso temporal para extremos según régimen"""
        
        time_diff_hours = abs((curr_extreme['timestamp'] - prev_extreme['timestamp']).total_seconds()) / 3600
        
        # Peso basado en tiempo y régimen
        if regime == "trending":
            # En tendencias, extremos más recientes tienen más peso
            if time_diff_hours < 12:
                return 1.2  # Más peso a cambios recientes
            elif time_diff_hours < 24:
                return 1.0  # Peso normal
            else:
                return 0.8  # Menos peso a cambios antiguos
        elif regime == "high_volatility":
            # En volatilidad, peso más uniforme
            return 1.0
        else:
            # En ranging, extremos consistentes tienen más peso
            if 6 < time_diff_hours < 18:
                return 1.1  # Peso ligeramente mayor para spacing normal
            else:
                return 0.9
    
    def _get_adaptive_position_context(self, range_position: float, week_range_pct: float) -> str:
        """Determinar contexto de posición adaptativo"""
        
        # Ajustar umbrales según la volatilidad semanal
        if week_range_pct > 12:  # Semana muy volátil
            low_threshold, high_threshold = 15, 85  # Umbrales más amplios
        elif week_range_pct < 4:  # Semana muy estable
            low_threshold, high_threshold = 25, 75  # Umbrales más estrictos
        else:  # Semana normal
            low_threshold, high_threshold = 20, 80  # Umbrales estándar
        
        if range_position <= low_threshold:
            return "zona_baja_adaptativa"
        elif range_position <= 40:
            return "zona_baja_media"
        elif range_position <= 60:
            return "zona_media"
        elif range_position <= high_threshold:
            return "zona_media_alta"
        else:
            return "zona_alta_adaptativa"
    
    def _analyze_adaptive_weekly_momentum(self, data: pd.DataFrame) -> Dict:
        """Analizar momentum semanal con criterios adaptativos"""
        
        regime = self.market_conditions.get("market_regime", "smooth_ranging")
        
        # Calcular momentum en diferentes ventanas según régimen
        if regime == "trending":
            momentum_windows = [24, 72, 168]  # 1d, 3d, 7d para tendencias
        else:
            momentum_windows = [12, 48, 168]  # 12h, 2d, 7d para ranging
        
        momentum_analysis = {}
        
        for window in momentum_windows:
            if len(data) >= window:
                start_price = data['close'].iloc[-window]
                end_price = data['close'].iloc[-1]
                momentum_pct = ((end_price - start_price) / start_price) * 100
                
                window_name = f"{window//24}d" if window >= 24 else f"{window}h"
                momentum_analysis[window_name] = round(momentum_pct, 2)
        
        # Determinar dirección del momentum
        if len(momentum_analysis) > 0:
            recent_momentum = list(momentum_analysis.values())[0]  # Más reciente
            
            if recent_momentum > 1:
                momentum_direction = "alcista"
            elif recent_momentum < -1:
                momentum_direction = "bajista"
            else:
                momentum_direction = "neutral"
        else:
            momentum_direction = "unknown"
        
        return {
            "direction": momentum_direction,
            "momentum_by_period": momentum_analysis,
            "regime": regime,
            "adaptive_analysis": True
        }
    
    def _format_adaptive_extremes_analysis(self, extreme_type: str, trend: str, 
                                         extremes: List[Dict], changes: List[float], days: float) -> str:
        """Formatear análisis de extremos adaptativo"""
        
        if not extremes:
            return f"sin {extreme_type}"
        
        if trend == "estables":
            avg_price = np.mean([e['price'] for e in extremes])
            regime = extremes[0].get('regime', 'unknown')
            return f"{len(extremes)} {extreme_type} estables ~${avg_price:,.0f} en {days:.0f}d ({regime})"
        elif trend == "crecientes":
            first_price = extremes[0]['price']
            last_price = extremes[-1]['price']
            change_pct = ((last_price - first_price) / first_price) * 100
            return f"{len(extremes)} {extreme_type} crecientes ${first_price:,.0f}→${last_price:,.0f} (+{change_pct:.1f}%)"
        else:  # decrecientes
            first_price = extremes[0]['price']
            last_price = extremes[-1]['price']
            change_pct = ((first_price - last_price) / first_price) * 100
            return f"{len(extremes)} {extreme_type} decrecientes ${first_price:,.0f}→${last_price:,.0f} (-{change_pct:.1f}%)"
    
    def _format_adaptive_weekly_response(self, week_min: float, week_max: float, week_range_pct: float,
                                       current_price: float, range_position: float, position_context: str,
                                       maximos_trend: Dict, minimos_trend: Dict, weekly_momentum: Dict) -> str:
        """Formatear respuesta semanal adaptativa"""
        
        # Información básica semanal
        basic_info = f"semana adaptativa: ${week_min:,.0f}-${week_max:,.0f} ({week_range_pct:.2f}%)"
        
        # Posición actual con contexto adaptativo
        position_info = f"actual ${current_price:,.0f} ({range_position:.1f}% del rango, {position_context.replace('_', ' ')})"
        
        # Tendencias de extremos adaptativos
        maximos_info = f"máximos 3d: {maximos_trend['trend']}"
        if maximos_trend['trend'] not in ['insuficiente', 'sin_datos']:
            maximos_info += f" ({maximos_trend['analysis']})"
        
        minimos_info = f"mínimos 3d: {minimos_trend['trend']}"
        if minimos_trend['trend'] not in ['insuficiente', 'sin_datos']:
            minimos_info += f" ({minimos_trend['analysis']})"
        
        # Momentum semanal
        momentum_direction = weekly_momentum.get('direction', 'unknown')
        regime = self.market_conditions.get("market_regime", "unknown")
        volatility = self.market_conditions.get("volatility", 0)
        
        momentum_info = f"momentum: {momentum_direction} ({regime}, vol {volatility:.1f}%)"
        
        return f"{basic_info}; {position_info}; {maximos_info}; {minimos_info}; {momentum_info}"