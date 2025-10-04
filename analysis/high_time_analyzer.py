from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

class HighTimeAnalyzer(BaseAnalyzer):
    """Analizador específico para máximos con poco tiempo de permanencia"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Implementación del método abstracto requerido"""
        return "Análisis genérico no implementado aún"
    
    def analyze_high_time_maximums(self, hours: float, max_time_minutes: int = 15) -> Tuple[str, Dict[str, Any]]:
        """
        Pregunta específica: ¿Cuáles fueron los máximos más altos en los que el precio estuvo poco tiempo?
        
        Detecta wickups y máximos con permanencia corta, rankeados por significancia
        """
        if self.data is None or self.data.empty:
            return "Sin datos", {}
            
        range_data = self.get_time_range_data(hours)
        
        if range_data.empty:
            return f"Sin datos para {hours}h", {}
        
        # Calcular contexto del rango para evaluar significancia
        price_context = self._calculate_price_context(range_data)
        
        # Método 1: Detectar wickups (wicks largos hacia arriba)
        wickups = self._detect_wickups(range_data, price_context)
        
        # Método 2: Detectar máximos locales con permanencia corta
        short_maximums = self._detect_short_time_maximums(range_data, max_time_minutes, price_context)
        
        # Combinar y rankear por significancia
        ranked_maximums = self._rank_by_significance(wickups, short_maximums, price_context)
        
        # Formatear respuesta priorizando los más marcados
        simple_answer = self._format_ranked_response(ranked_maximums, max_time_minutes)
        
        detailed_data = {
            "analysis_type": "ranked_high_time_maximums",
            "time_range_hours": hours,
            "max_time_threshold_minutes": max_time_minutes,
            "price_context": price_context,
            "wickups_detected": len(wickups),
            "short_maximums_detected": len(short_maximums),
            "total_maximums": len(ranked_maximums),
            "ranked_maximums": ranked_maximums
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
    
    def _detect_wickups(self, data: pd.DataFrame, context: Dict, min_wick_pct: float = 0.05) -> List[Dict]:
        """Detecta wickups (mechas largas hacia arriba) con scoring de significancia"""
        wickups = []
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # Calcular el cuerpo de la vela
            body_top = max(open_price, close_price)
            body_bottom = min(open_price, close_price)
            
            # Calcular la mecha superior
            upper_wick = high_price - body_top
            upper_wick_pct = (upper_wick / high_price) * 100
            
            # Calcular el tamaño del cuerpo
            body_size = abs(close_price - open_price)
            
            # Es un wickup si la mecha superior es significativa
            if upper_wick_pct >= min_wick_pct and upper_wick > body_size * 0.5:
                
                # Calcular scoring de significancia
                significance_score = self._calculate_wickup_significance(
                    high_price, upper_wick_pct, context, timestamp, data
                )
                
                wickups.append({
                    'timestamp': timestamp,
                    'time_str': timestamp.strftime('%H:%M'),
                    'price': high_price,
                    'wick_size_pct': upper_wick_pct,
                    'wick_size_points': upper_wick,
                    'type': 'wickup',
                    'duration_estimate': 1,
                    'significance_score': significance_score,
                    'rank_factors': {
                        'price_height': ((high_price - context['min_price']) / context['price_range']) * 100,
                        'wick_strength': upper_wick_pct,
                        'context_significance': significance_score
                    }
                })
        
        return wickups
    
    def _calculate_wickup_significance(self, high_price: float, wick_pct: float, context: Dict, timestamp, data: pd.DataFrame) -> float:
        """Calcula la significancia de un wickup"""
        # Factor 1: Qué tan alto es vs el rango total (0-100)
        height_factor = ((high_price - context['min_price']) / context['price_range']) * 100
        
        # Factor 2: Tamaño de la mecha (0-100)
        wick_factor = min(wick_pct * 10, 100)  # Escalar wick_pct
        
        # Factor 3: Qué tan cerca está del máximo absoluto (0-100)
        max_proximity_factor = 100 - (((context['max_price'] - high_price) / context['price_range']) * 100)
        
        # Factor 4: Contexto temporal - volatilidad local
        window_start = max(0, data.index.get_loc(timestamp) - 10)
        window_end = min(len(data), data.index.get_loc(timestamp) + 10)
        local_volatility = data.iloc[window_start:window_end]['close'].pct_change().std() * 100
        volatility_factor = min(local_volatility * 20, 100)
        
        # Combinar factores con pesos
        significance = (
            height_factor * 0.3 +        # 30% peso a altura
            wick_factor * 0.3 +          # 30% peso a tamaño de mecha
            max_proximity_factor * 0.3 + # 30% peso a cercanía al máximo
            volatility_factor * 0.1      # 10% peso a volatilidad local
        )
        
        return significance
    
    def _detect_short_time_maximums(self, data: pd.DataFrame, max_time_minutes: int, context: Dict) -> List[Dict]:
        """Detecta máximos con permanencia corta y scoring de significancia"""
        short_maximums = []
        
        # Detectar máximos locales con ventana más agresiva
        for window_size in [3, 5, 7]:
            for i in range(window_size, len(data) - window_size):
                current_high = data.iloc[i]['high']
                current_timestamp = data.index[i]
                
                # Verificar si es máximo local en esta ventana
                window_highs = data.iloc[i-window_size:i+window_size+1]['high']
                
                if current_high == window_highs.max():
                    # Calcular cuánto tiempo estuvo cerca de este nivel
                    tolerance = current_high * 0.002  # 0.2% tolerancia
                    near_level_count = 0
                    
                    # Contar velas en rango extendido
                    for j in range(max(0, i-15), min(len(data), i+15)):
                        if abs(data.iloc[j]['high'] - current_high) <= tolerance:
                            near_level_count += 1
                    
                    if near_level_count <= max_time_minutes:
                        # Calcular significancia
                        significance_score = self._calculate_maximum_significance(
                            current_high, near_level_count, context, current_timestamp, data
                        )
                        
                        short_maximums.append({
                            'timestamp': current_timestamp,
                            'time_str': current_timestamp.strftime('%H:%M'),
                            'price': current_high,
                            'duration_estimate': near_level_count,
                            'type': 'short_maximum',
                            'significance_score': significance_score,
                            'rank_factors': {
                                'price_height': ((current_high - context['min_price']) / context['price_range']) * 100,
                                'time_brevity': max(0, (max_time_minutes - near_level_count) / max_time_minutes * 100),
                                'context_significance': significance_score
                            }
                        })
        
        # Eliminar duplicados
        unique_maximums = self._remove_duplicate_maximums(short_maximums)
        return unique_maximums
    
    def _calculate_maximum_significance(self, price: float, duration: int, context: Dict, timestamp, data: pd.DataFrame) -> float:
        """Calcula la significancia de un máximo con tiempo corto"""
        # Factor 1: Altura vs rango total
        height_factor = ((price - context['min_price']) / context['price_range']) * 100
        
        # Factor 2: Brevedad del tiempo (menos tiempo = más significativo)
        brevity_factor = max(0, (15 - duration) / 15 * 100)
        
        # Factor 3: Cercanía al máximo absoluto
        max_proximity = 100 - (((context['max_price'] - price) / context['price_range']) * 100)
        
        # Combinar con pesos
        significance = (
            height_factor * 0.4 +      # 40% peso a altura
            brevity_factor * 0.3 +     # 30% peso a brevedad
            max_proximity * 0.3        # 30% peso a cercanía al máximo
        )
        
        return significance
    
    def _remove_duplicate_maximums(self, maximums: List[Dict]) -> List[Dict]:
        """Elimina máximos duplicados manteniendo el más significativo"""
        if not maximums:
            return []
        
        # Ordenar por timestamp
        maximums.sort(key=lambda x: x['timestamp'])
        
        unique = [maximums[0]]
        
        for maximum in maximums[1:]:
            time_diff = abs((maximum['timestamp'] - unique[-1]['timestamp']).total_seconds())
            price_diff = abs(maximum['price'] - unique[-1]['price'])
            
            # Si están muy cerca en tiempo y precio
            if time_diff < 300 and price_diff < unique[-1]['price'] * 0.003:
                # Mantener el de mayor significancia
                if maximum['significance_score'] > unique[-1]['significance_score']:
                    unique[-1] = maximum
            else:
                unique.append(maximum)
        
        return unique
    
    def _rank_by_significance(self, wickups: List[Dict], short_maximums: List[Dict], context: Dict) -> List[Dict]:
        """Rankea todos los máximos por significancia (más marcados primero)"""
        all_maximums = wickups + short_maximums
        
        # Eliminar duplicados manteniendo el más significativo
        all_maximums = self._remove_duplicate_maximums(all_maximums)
        
        # Ordenar por significance_score descendente (más significativo primero)
        all_maximums.sort(key=lambda x: x['significance_score'], reverse=True)
        
        # Agregar ranking
        for i, maximum in enumerate(all_maximums):
            maximum['rank'] = i + 1
        
        return all_maximums
    
    def _format_ranked_response(self, maximums: List[Dict], max_time_minutes: int) -> str:
        """Formatea la respuesta priorizando los más marcados"""
        if not maximums:
            return f"no hay máximos rápidos detectados (≤{max_time_minutes}min)"
        
        # Tomar los 4 más significativos
        top_significant = maximums[:4]
        
        descriptions = []
        for maximum in top_significant:
            if maximum['type'] == 'wickup':
                desc = f"${maximum['price']:,.0f} wick {maximum['time_str']} ({maximum['wick_size_pct']:.2f}%, rank #{maximum['rank']})"
            else:
                desc = f"${maximum['price']:,.0f} {maximum['duration_estimate']}min {maximum['time_str']} (rank #{maximum['rank']})"
            
            descriptions.append(desc)
        
        return "máximos más marcados: " + ", ".join(descriptions)