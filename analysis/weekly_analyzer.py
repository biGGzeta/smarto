from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

class WeeklyAnalyzer(BaseAnalyzer):
    """Analizador para panorama semanal con análisis de tendencia de extremos de últimos 3 días"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Implementación del método abstracto requerido"""
        return "Análisis genérico no implementado aún"
    
    def analyze_weekly_with_recent_extremes(self, recent_days: float = 3) -> Tuple[str, Dict[str, Any]]:
        """
        Pregunta específica: Análisis semanal + tendencia de extremos recientes
        
        Analiza: min/max semana, % rango, posición actual, tendencia máximos/mínimos últimos 3 días
        """
        if self.data is None or self.data.empty:
            return "Sin datos", {}
        
        # Obtener datos de la semana completa
        weekly_data = self.get_time_range_data(168)  # 7 días = 168 horas
        
        if weekly_data.empty:
            return "Sin datos para análisis semanal", {}
        
        # Análisis básico semanal
        week_max = weekly_data['high'].max()
        week_min = weekly_data['low'].min()
        week_range = week_max - week_min
        week_range_pct = (week_range / week_min) * 100
        
        # Precio actual y posición en el rango
        current_price = weekly_data['close'].iloc[-1]
        range_position = ((current_price - week_min) / week_range) * 100
        
        # Análisis de extremos recientes (últimos 3 días)
        recent_hours = recent_days * 24  # Convertir días a horas
        recent_data = self.get_time_range_data(recent_hours)
        
        if not recent_data.empty:
            maximos_trend = self._analyze_extremes_trend(recent_data, 'maximos', recent_days)
            minimos_trend = self._analyze_extremes_trend(recent_data, 'minimos', recent_days)
        else:
            maximos_trend = {"trend": "sin_datos", "analysis": "sin datos suficientes"}
            minimos_trend = {"trend": "sin_datos", "analysis": "sin datos suficientes"}
        
        # Contexto de posición
        position_context = self._get_position_context(range_position)
        
        # Formatear respuesta
        simple_answer = self._format_weekly_response(
            week_min, week_max, week_range_pct, current_price, 
            range_position, position_context, maximos_trend, minimos_trend
        )
        
        detailed_data = {
            "analysis_type": "weekly_with_recent_extremes",
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
            "weekly_data_points": len(weekly_data),
            "recent_data_points": len(recent_data) if not recent_data.empty else 0
        }
        
        return simple_answer, detailed_data
    
    def _analyze_extremes_trend(self, data: pd.DataFrame, extreme_type: str, days: float) -> Dict:
        """Analiza la tendencia de máximos o mínimos en el período reciente (3 días)"""
        
        if extreme_type == 'maximos':
            # Detectar máximos locales con ventana más amplia para 3 días
            extremes = self._detect_local_maxima(data, window=5)
            price_key = 'high'
        else:
            # Detectar mínimos locales con ventana más amplia para 3 días
            extremes = self._detect_local_minima(data, window=5)
            price_key = 'low'
        
        if len(extremes) < 2:
            return {
                "trend": "insuficiente",
                "extremes_count": len(extremes),
                "analysis": f"solo {len(extremes)} {extreme_type} detectados en {days} días",
                "trend_strength": 0,
                "avg_change_pct": 0
            }
        
        # Calcular cambios porcentuales entre extremos consecutivos
        price_changes = []
        for i in range(1, len(extremes)):
            prev_price = extremes[i-1]['price']
            curr_price = extremes[i]['price']
            change_pct = ((curr_price - prev_price) / prev_price) * 100
            price_changes.append(change_pct)
        
        # Determinar tendencia
        avg_change = np.mean(price_changes)
        std_change = np.std(price_changes) if len(price_changes) > 1 else 0
        
        # Clasificar tendencia (umbrales ajustados para 3 días)
        stability_threshold = 0.8  # 0.8% para 3 días (más que 0.5% para 2h)
        
        if abs(avg_change) <= stability_threshold and std_change <= 1.0:
            trend = "estables"
            trend_strength = std_change  # Menor std = más estable
        elif avg_change > stability_threshold:
            trend = "crecientes"
            trend_strength = avg_change
        else:
            trend = "decrecientes"
            trend_strength = abs(avg_change)
        
        return {
            "trend": trend,
            "extremes_count": len(extremes),
            "analysis": self._format_extremes_analysis(extreme_type, trend, extremes, price_changes, days),
            "trend_strength": trend_strength,
            "avg_change_pct": avg_change,
            "std_change_pct": std_change,
            "extremes_data": extremes
        }
    
    def _detect_local_maxima(self, data: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Detecta máximos locales con ventana ajustada para 3 días"""
        maxima = []
        
        for i in range(window, len(data) - window):
            current_high = data['high'].iloc[i]
            current_time = data.index[i]
            
            # Verificar si es máximo local
            window_highs = data['high'].iloc[i-window:i+window+1]
            
            if current_high == window_highs.max():
                maxima.append({
                    'timestamp': current_time,
                    'time_str': current_time.strftime('%m/%d %H:%M'),  # Formato para días
                    'price': current_high,
                    'index': i
                })
        
        # Filtrar máximos muy cercanos (dentro de 2 horas para 3 días)
        filtered_maxima = self._filter_close_extremes(maxima, min_separation_hours=2)
        
        return filtered_maxima
    
    def _detect_local_minima(self, data: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Detecta mínimos locales con ventana ajustada para 3 días"""
        minima = []
        
        for i in range(window, len(data) - window):
            current_low = data['low'].iloc[i]
            current_time = data.index[i]
            
            # Verificar si es mínimo local
            window_lows = data['low'].iloc[i-window:i+window+1]
            
            if current_low == window_lows.min():
                minima.append({
                    'timestamp': current_time,
                    'time_str': current_time.strftime('%m/%d %H:%M'),  # Formato para días
                    'price': current_low,
                    'index': i
                })
        
        # Filtrar mínimos muy cercanos (dentro de 2 horas para 3 días)
        filtered_minima = self._filter_close_extremes(minima, min_separation_hours=2)
        
        return filtered_minima
    
    def _filter_close_extremes(self, extremes: List[Dict], min_separation_hours: int = 2) -> List[Dict]:
        """Filtra extremos muy cercanos en tiempo"""
        if len(extremes) <= 1:
            return extremes
        
        filtered = [extremes[0]]
        
        for i in range(1, len(extremes)):
            current = extremes[i]
            last_kept = filtered[-1]
            
            time_diff_hours = abs((current['timestamp'] - last_kept['timestamp']).total_seconds()) / 3600
            
            if time_diff_hours >= min_separation_hours:
                filtered.append(current)
            else:
                # Si están muy cerca, mantener el más extremo
                if 'high' in str(current.get('price', 0)):  # Para máximos
                    if current['price'] > last_kept['price']:
                        filtered[-1] = current
                else:  # Para mínimos
                    if current['price'] < last_kept['price']:
                        filtered[-1] = current
        
        return filtered
    
    def _get_position_context(self, range_position: float) -> str:
        """Determina el contexto de la posición en el rango"""
        if range_position <= 20:
            return "zona_baja"
        elif range_position <= 40:
            return "zona_baja_media"
        elif range_position <= 60:
            return "zona_media"
        elif range_position <= 80:
            return "zona_media_alta"
        else:
            return "zona_alta"
    
    def _format_extremes_analysis(self, extreme_type: str, trend: str, extremes: List[Dict], changes: List[float], days: float) -> str:
        """Formatea el análisis de extremos para 3 días"""
        if not extremes:
            return f"sin {extreme_type}"
        
        if trend == "estables":
            avg_price = np.mean([e['price'] for e in extremes])
            return f"{len(extremes)} {extreme_type} estables ~${avg_price:,.0f} en {days:.0f}d"
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
    
    def _format_weekly_response(self, week_min: float, week_max: float, week_range_pct: float,
                               current_price: float, range_position: float, position_context: str,
                               maximos_trend: Dict, minimos_trend: Dict) -> str:
        """Formatea la respuesta del análisis semanal con tendencias de 3 días"""
        
        # Información básica semanal
        basic_info = f"semana: ${week_min:,.0f}-${week_max:,.0f} ({week_range_pct:.2f}%)"
        
        # Posición actual
        position_info = f"actual ${current_price:,.0f} ({range_position:.1f}% del rango, {position_context.replace('_', ' ')})"
        
        # Tendencias recientes de 3 días
        maximos_info = f"máximos 3d: {maximos_trend['trend']}"
        if maximos_trend['trend'] not in ['insuficiente', 'sin_datos']:
            maximos_info += f" ({maximos_trend['analysis']})"
        
        minimos_info = f"mínimos 3d: {minimos_trend['trend']}"
        if minimos_trend['trend'] not in ['insuficiente', 'sin_datos']:
            minimos_info += f" ({minimos_trend['analysis']})"
        
        return f"{basic_info}; {position_info}; {maximos_info}; {minimos_info}"