from .adaptive_base_analyzer import AdaptiveBaseAnalyzer
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

class AdaptiveHighTimeAnalyzer(AdaptiveBaseAnalyzer):
    """Analizador de máximos con parámetros auto-adaptativos"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Implementación del método abstracto requerido"""
        hours = question_params.get('hours', 3)
        simple_answer, _ = self.analyze_high_time_maximums_adaptive(hours)
        return simple_answer
    
    def analyze_high_time_maximums_adaptive(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """
        Análisis de máximos con parámetros que se ajustan automáticamente
        """
        if self.data is None or self.data.empty:
            return "Sin datos", {}
            
        range_data = self.get_time_range_data(hours)
        
        if range_data.empty:
            return f"Sin datos para {hours}h", {}
        
        print(f"🔄 Iniciando análisis adaptativo de máximos para {hours}h...")
        
        # Analizar condiciones y obtener parámetros adaptativos
        self.load_data_and_analyze_conditions(range_data, f"{hours}h")
        
        # Calcular contexto adaptativo
        price_context = self._calculate_adaptive_price_context(range_data)
        
        # Detectar wickups con parámetros dinámicos
        wickups = self._detect_adaptive_wickups(range_data, price_context)
        
        # Detectar máximos con tiempo corto
        short_maximums = self._detect_adaptive_short_maximums(range_data, price_context)
        
        # Combinar y rankear
        ranked_maximums = self._rank_with_adaptive_significance(wickups, short_maximums, price_context)
        
        # Formatear respuesta
        simple_answer = self._format_adaptive_response(ranked_maximums)
        
        detailed_data = {
            "analysis_type": "adaptive_high_time_maximums",
            "time_range_hours": hours,
            "market_conditions": self.market_conditions,
            "adaptive_parameters": self.current_parameters,
            "price_context": price_context,
            "wickups_detected": len(wickups),
            "short_maximums_detected": len(short_maximums),
            "total_maximums": len(ranked_maximums),
            "ranked_maximums": ranked_maximums
        }
        
        return simple_answer, detailed_data
    
    # ... resto de métodos como antes ...
    def _calculate_adaptive_price_context(self, data: pd.DataFrame) -> Dict:
        """Calcular contexto de precios para máximos con métricas adaptativas"""
        
        base_context = {
            "min_price": data['low'].min(),
            "max_price": data['high'].max(),
            "avg_price": data['close'].mean(),
            "current_price": data['close'].iloc[-1],
            "volatility": self.market_conditions.get("volatility", 5.0),
            "regime": self.market_conditions.get("market_regime", "smooth_ranging")
        }
        
        base_context["price_range"] = base_context["max_price"] - base_context["min_price"]
        base_context["price_range_pct"] = (base_context["price_range"] / base_context["min_price"]) * 100
        
        return base_context
    
    def _detect_adaptive_wickups(self, data: pd.DataFrame, context: Dict) -> List[Dict]:
        """Detectar wickups usando umbrales adaptativos"""
        wickups = []
        wick_threshold = self.get_adaptive_wick_threshold()
        
        print(f"🔍 Detectando wickups con umbral adaptativo: {wick_threshold:.3f}%")
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            
            body_top = max(open_price, close_price)
            upper_wick = high_price - body_top
            upper_wick_pct = (upper_wick / high_price) * 100
            body_size = abs(close_price - open_price)
            
            is_significant_wick = upper_wick_pct >= wick_threshold
            regime = context.get("regime", "smooth_ranging")
            
            if regime == "high_volatility":
                wick_vs_body_ratio = 0.3
            else:
                wick_vs_body_ratio = 0.5
            
            has_long_wick = upper_wick > body_size * wick_vs_body_ratio
            
            if is_significant_wick and has_long_wick:
                significance = self._calculate_adaptive_wickup_significance(
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
                    'significance_score': significance,
                    'adaptive_threshold_used': wick_threshold,
                    'regime': regime
                })
        
        print(f"✅ Detectados {len(wickups)} wickups con criterios adaptativos")
        return wickups
    
    def _detect_adaptive_short_maximums(self, data: pd.DataFrame, context: Dict) -> List[Dict]:
        """Detectar máximos con permanencia corta usando parámetros adaptativos"""
        short_maximums = []
        window_sizes = self.get_adaptive_window_sizes()
        max_time_threshold = self.get_adaptive_time_threshold()
        
        for window_size in window_sizes:
            for i in range(window_size, len(data) - window_size):
                current_high = data.iloc[i]['high']
                current_timestamp = data.index[i]
                
                window_highs = data.iloc[i-window_size:i+window_size+1]['high']
                
                if current_high == window_highs.max():
                    volatility = context.get("volatility", 5.0)
                    if volatility > 10:
                        tolerance = current_high * 0.004
                    elif volatility < 3:
                        tolerance = current_high * 0.001
                    else:
                        tolerance = current_high * 0.002
                    
                    near_level_count = 0
                    search_range = min(30, len(data) - i - 1)
                    
                    for j in range(max(0, i-15), min(len(data), i+search_range)):
                        if abs(data.iloc[j]['high'] - current_high) <= tolerance:
                            near_level_count += 1
                    
                    if near_level_count <= max_time_threshold:
                        significance = self._calculate_adaptive_maximum_significance(
                            current_high, near_level_count, context, current_timestamp, data
                        )
                        
                        short_maximums.append({
                            'timestamp': current_timestamp,
                            'time_str': current_timestamp.strftime('%H:%M'),
                            'price': current_high,
                            'duration_estimate': near_level_count,
                            'type': 'short_maximum',
                            'significance_score': significance,
                            'window_size_used': window_size,
                            'tolerance_used': tolerance,
                            'max_time_used': max_time_threshold
                        })
        
        return self._remove_duplicates_adaptive(short_maximums)
    
    def _calculate_adaptive_wickup_significance(self, price: float, wick_pct: float, 
                                               context: Dict, timestamp, data: pd.DataFrame) -> float:
        """Calcular significancia de wickup con pesos adaptativos"""
        weights = self.get_significance_weights()
        
        height_factor = ((price - context['min_price']) / context['price_range']) * 100
        height_score = height_factor * weights['depth']
        
        wick_factor = min(wick_pct * 15, 100)
        wick_score = wick_factor * weights['time']
        
        regime = context.get('regime', 'smooth_ranging')
        if regime == 'trending':
            context_bonus = 20
        elif regime == 'high_volatility':
            context_bonus = 8
        else:
            context_bonus = 15
        
        context_score = context_bonus * weights['context']
        
        max_proximity = (price / context['max_price']) * 100
        if max_proximity > 95:
            proximity_bonus = 15
        elif max_proximity > 85:
            proximity_bonus = 10
        else:
            proximity_bonus = 5
        
        total_significance = height_score + wick_score + context_score + proximity_bonus
        return min(total_significance, 100)
    
    def _calculate_adaptive_maximum_significance(self, price: float, duration: int, 
                                               context: Dict, timestamp, data: pd.DataFrame) -> float:
        """Calcular significancia de máximo con criterios adaptativos"""
        weights = self.get_significance_weights()
        max_time = self.get_adaptive_time_threshold()
        
        height_factor = ((price - context['min_price']) / context['price_range']) * 100
        height_score = height_factor * weights['depth']
        
        brevity_factor = max(0, (max_time - duration) / max_time * 100)
        brevity_score = brevity_factor * weights['time']
        
        regime = context.get('regime', 'smooth_ranging')
        momentum = context.get('momentum', 0)
        
        if regime == 'trending' and momentum > 2:
            context_bonus = 25
        elif regime == 'high_volatility':
            context_bonus = 10
        else:
            context_bonus = 15
        
        context_score = context_bonus * weights['context']
        total_significance = height_score + brevity_score + context_score
        return min(total_significance, 100)
    
    def _remove_duplicates_adaptive(self, maximums: List[Dict]) -> List[Dict]:
        """Eliminar duplicados con criterios adaptativos para máximos"""
        if not maximums:
            return []
        
        min_separation = self.current_parameters.get("min_separation_minutes", 20)
        maximums.sort(key=lambda x: x['timestamp'])
        unique = [maximums[0]]
        
        for maximum in maximums[1:]:
            time_diff = abs((maximum['timestamp'] - unique[-1]['timestamp']).total_seconds()) / 60
            price_diff_pct = abs(maximum['price'] - unique[-1]['price']) / unique[-1]['price'] * 100
            
            volatility = self.market_conditions.get("volatility", 5.0)
            price_tolerance = 0.5 if volatility > 10 else 0.2
            
            if time_diff >= min_separation or price_diff_pct >= price_tolerance:
                unique.append(maximum)
            else:
                if maximum['significance_score'] > unique[-1]['significance_score']:
                    unique[-1] = maximum
        
        return unique
    
    def _rank_with_adaptive_significance(self, wickups: List[Dict], 
                                       short_maximums: List[Dict], context: Dict) -> List[Dict]:
        """Rankear con significancia adaptativa"""
        all_maximums = wickups + short_maximums
        all_maximums = self._remove_duplicates_adaptive(all_maximums)
        all_maximums.sort(key=lambda x: x['significance_score'], reverse=True)
        
        for i, maximum in enumerate(all_maximums):
            maximum['rank'] = i + 1
            maximum['adaptive_analysis'] = True
        
        return all_maximums
    
    def _format_adaptive_response(self, maximums: List[Dict]) -> str:
        """Formatear respuesta con información adaptativa"""
        if not maximums:
            regime = self.market_conditions.get("market_regime", "unknown")
            return f"no hay máximos rápidos detectados (régimen: {regime})"
        
        top_maximums = maximums[:4]
        descriptions = []
        
        for maximum in top_maximums:
            if maximum['type'] == 'wickup':
                desc = f"${maximum['price']:,.0f} wick {maximum['time_str']} ({maximum['wick_size_pct']:.2f}%, rank #{maximum['rank']})"
            else:
                desc = f"${maximum['price']:,.0f} {maximum['duration_estimate']}min {maximum['time_str']} (rank #{maximum['rank']})"
            descriptions.append(desc)
        
        regime = self.market_conditions.get("market_regime", "unknown")
        volatility = self.market_conditions.get("volatility", 0)
        
        return f"máximos adaptativos ({regime}, vol {volatility:.1f}%): " + ", ".join(descriptions)