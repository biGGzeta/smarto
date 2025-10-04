from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

class LowTimeAnalyzer(BaseAnalyzer):
    """Analizador específico para mínimos con poco tiempo de permanencia - versión con ranking por significancia"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Implementación del método abstracto requerido"""
        return "Análisis genérico no implementado aún"
    
    def analyze_low_time_minimums(self, hours: float, max_time_minutes: int = 15) -> Tuple[str, Dict[str, Any]]:
        """
        Pregunta específica: ¿Cuáles fueron los mínimos más bajos en los que el precio estuvo poco tiempo?
        
        Versión mejorada que prioriza los más marcados/significativos
        """
        if self.data is None or self.data.empty:
            return "Sin datos", {}
            
        range_data = self.get_time_range_data(hours)
        
        if range_data.empty:
            return f"Sin datos para {hours}h", {}
        
        # Calcular contexto del rango para evaluar significancia
        price_context = self._calculate_price_context(range_data)
        
        # Método 1: Detectar wickdowns (wicks largos hacia abajo)
        wickdowns = self._detect_wickdowns(range_data, price_context)
        
        # Método 2: Detectar mínimos locales con permanencia corta
        short_minimums = self._detect_short_time_minimums(range_data, max_time_minutes, price_context)
        
        # Combinar y rankear por significancia
        ranked_minimums = self._rank_by_significance(wickdowns, short_minimums, price_context)
        
        # Formatear respuesta priorizando los más marcados
        simple_answer = self._format_ranked_response(ranked_minimums, max_time_minutes)
        
        detailed_data = {
            "analysis_type": "ranked_low_time_minimums",
            "time_range_hours": hours,
            "max_time_threshold_minutes": max_time_minutes,
            "price_context": price_context,
            "wickdowns_detected": len(wickdowns),
            "short_minimums_detected": len(short_minimums),
            "total_minimums": len(ranked_minimums),
            "ranked_minimums": ranked_minimums
        }
        
        return simple_answer, detailed_data
    
    def _calculate_price_context(self, data: pd.DataFrame) -> Dict:
        """Calcula el contexto de precios para evaluar significancia"""
        return {
            "min_price": data['low'].min(),
            "max_price": data['high'].max(),
            "avg_price": data['close'].mean(),
            "price_range": data['high'].max() - data['low'].min(),
            "price_range_pct": ((data['high'].max() - data['low'].min()) / data['low'].min()) * 100,
            "std_deviation": data['close'].std(),
            "volatility": data['close'].pct_change().std() * 100
        }
    
    def _detect_wickdowns(self, data: pd.DataFrame, context: Dict, min_wick_pct: float = 0.05) -> List[Dict]:
        """Detecta wickdowns con scoring de significancia"""
        wickdowns = []
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # Calcular el cuerpo de la vela
            body_top = max(open_price, close_price)
            body_bottom = min(open_price, close_price)
            
            # Calcular la mecha inferior
            lower_wick = body_bottom - low_price
            lower_wick_pct = (lower_wick / low_price) * 100
            
            # Calcular el tamaño del cuerpo
            body_size = abs(close_price - open_price)
            
            # Es un wickdown si la mecha inferior es significativa
            if lower_wick_pct >= min_wick_pct and lower_wick > body_size * 0.5:
                
                # Calcular scoring de significancia
                significance_score = self._calculate_wickdown_significance(
                    low_price, lower_wick_pct, context, timestamp, data
                )
                
                wickdowns.append({
                    'timestamp': timestamp,
                    'time_str': timestamp.strftime('%H:%M'),
                    'price': low_price,
                    'wick_size_pct': lower_wick_pct,
                    'wick_size_points': lower_wick,
                    'type': 'wickdown',
                    'duration_estimate': 1,
                    'significance_score': significance_score,
                    'rank_factors': {
                        'price_depth': ((context['max_price'] - low_price) / context['price_range']) * 100,
                        'wick_strength': lower_wick_pct,
                        'context_significance': significance_score
                    }
                })
        
        return wickdowns
    
    def _calculate_wickdown_significance(self, low_price: float, wick_pct: float, context: Dict, timestamp, data: pd.DataFrame) -> float:
        """Calcula la significancia de un wickdown"""
        # Factor 1: Qué tan bajo es vs el rango total (0-100)
        depth_factor = ((context['max_price'] - low_price) / context['price_range']) * 100
        
        # Factor 2: Tamaño de la mecha (0-100)
        wick_factor = min(wick_pct * 10, 100)  # Escalar wick_pct
        
        # Factor 3: Qué tan cerca está del mínimo absoluto (0-100)
        min_proximity_factor = 100 - (((low_price - context['min_price']) / context['price_range']) * 100)
        
        # Factor 4: Contexto temporal - si está en momento de alta volatilidad
        window_start = max(0, data.index.get_loc(timestamp) - 10)
        window_end = min(len(data), data.index.get_loc(timestamp) + 10)
        local_volatility = data.iloc[window_start:window_end]['close'].pct_change().std() * 100
        volatility_factor = min(local_volatility * 20, 100)
        
        # Combinar factores con pesos
        significance = (
            depth_factor * 0.3 +      # 30% peso a profundidad
            wick_factor * 0.3 +       # 30% peso a tamaño de mecha
            min_proximity_factor * 0.3 + # 30% peso a cercanía al mínimo
            volatility_factor * 0.1    # 10% peso a volatilidad local
        )
        
        return significance
    
    def _detect_short_time_minimums(self, data: pd.DataFrame, max_time_minutes: int, context: Dict) -> List[Dict]:
        """Detecta mínimos con permanencia corta y scoring de significancia"""
        short_minimums = []
        
        # Detectar mínimos locales con ventana más agresiva
        for window_size in [3, 5, 7]:
            for i in range(window_size, len(data) - window_size):
                current_low = data.iloc[i]['low']
                current_timestamp = data.index[i]
                
                # Verificar si es mínimo local en esta ventana
                window_lows = data.iloc[i-window_size:i+window_size+1]['low']
                
                if current_low == window_lows.min():
                    # Calcular cuánto tiempo estuvo cerca de este nivel
                    tolerance = current_low * 0.002  # 0.2% tolerancia
                    near_level_count = 0
                    
                    # Contar velas en rango extendido
                    for j in range(max(0, i-15), min(len(data), i+15)):
                        if abs(data.iloc[j]['low'] - current_low) <= tolerance:
                            near_level_count += 1
                    
                    if near_level_count <= max_time_minutes:
                        # Calcular significancia
                        significance_score = self._calculate_minimum_significance(
                            current_low, near_level_count, context, current_timestamp, data
                        )
                        
                        short_minimums.append({
                            'timestamp': current_timestamp,
                            'time_str': current_timestamp.strftime('%H:%M'),
                            'price': current_low,
                            'duration_estimate': near_level_count,
                            'type': 'short_minimum',
                            'significance_score': significance_score,
                            'rank_factors': {
                                'price_depth': ((context['max_price'] - current_low) / context['price_range']) * 100,
                                'time_brevity': max(0, (max_time_minutes - near_level_count) / max_time_minutes * 100),
                                'context_significance': significance_score
                            }
                        })
        
        # Eliminar duplicados
        unique_minimums = self._remove_duplicate_minimums(short_minimums)
        return unique_minimums
    
    def _calculate_minimum_significance(self, price: float, duration: int, context: Dict, timestamp, data: pd.DataFrame) -> float:
        """Calcula la significancia de un mínimo con tiempo corto"""
        # Factor 1: Profundidad vs rango total
        depth_factor = ((context['max_price'] - price) / context['price_range']) * 100
        
        # Factor 2: Brevedad del tiempo (menos tiempo = más significativo)
        brevity_factor = max(0, (15 - duration) / 15 * 100)
        
        # Factor 3: Cercanía al mínimo absoluto
        min_proximity = 100 - (((price - context['min_price']) / context['price_range']) * 100)
        
        # Combinar con pesos
        significance = (
            depth_factor * 0.4 +      # 40% peso a profundidad
            brevity_factor * 0.3 +    # 30% peso a brevedad
            min_proximity * 0.3       # 30% peso a cercanía al mínimo
        )
        
        return significance
    
    def _remove_duplicate_minimums(self, minimums: List[Dict]) -> List[Dict]:
        """Elimina mínimos duplicados manteniendo el más significativo"""
        if not minimums:
            return []
        
        # Ordenar por timestamp
        minimums.sort(key=lambda x: x['timestamp'])
        
        unique = [minimums[0]]
        
        for minimum in minimums[1:]:
            time_diff = abs((minimum['timestamp'] - unique[-1]['timestamp']).total_seconds())
            price_diff = abs(minimum['price'] - unique[-1]['price'])
            
            # Si están muy cerca en tiempo y precio
            if time_diff < 300 and price_diff < unique[-1]['price'] * 0.003:
                # Mantener el de mayor significancia
                if minimum['significance_score'] > unique[-1]['significance_score']:
                    unique[-1] = minimum
            else:
                unique.append(minimum)
        
        return unique
    
    def _rank_by_significance(self, wickdowns: List[Dict], short_minimums: List[Dict], context: Dict) -> List[Dict]:
        """Rankea todos los mínimos por significancia (más marcados primero)"""
        all_minimums = wickdowns + short_minimums
        
        # Eliminar duplicados manteniendo el más significativo
        all_minimums = self._remove_duplicate_minimums(all_minimums)
        
        # Ordenar por significance_score descendente (más significativo primero)
        all_minimums.sort(key=lambda x: x['significance_score'], reverse=True)
        
        # Agregar ranking
        for i, minimum in enumerate(all_minimums):
            minimum['rank'] = i + 1
        
        return all_minimums
    
    def _format_ranked_response(self, minimums: List[Dict], max_time_minutes: int) -> str:
        """Formatea la respuesta priorizando los más marcados"""
        if not minimums:
            return f"no hay mínimos rápidos detectados (≤{max_time_minutes}min)"
        
        # Tomar los 4 más significativos (no los últimos)
        top_significant = minimums[:4]
        
        descriptions = []
        for minimum in top_significant:
            if minimum['type'] == 'wickdown':
                desc = f"${minimum['price']:,.0f} wick {minimum['time_str']} ({minimum['wick_size_pct']:.2f}%, rank #{minimum['rank']})"
            else:
                desc = f"${minimum['price']:,.0f} {minimum['duration_estimate']}min {minimum['time_str']} (rank #{minimum['rank']})"
            
            descriptions.append(desc)
        
        return "mínimos más marcados: " + ", ".join(descriptions)