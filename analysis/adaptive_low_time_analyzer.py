from .adaptive_base_analyzer import AdaptiveBaseAnalyzer
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

class AdaptiveLowTimeAnalyzer(AdaptiveBaseAnalyzer):
    """Analizador de mínimos con parámetros auto-adaptativos"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Implementación del método abstracto requerido"""
        hours = question_params.get('hours', 3)
        simple_answer, _ = self.analyze_low_time_minimums_adaptive(hours)
        return simple_answer
    
    def analyze_low_time_minimums_adaptive(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """
        Análisis de mínimos con parámetros que se ajustan automáticamente
        según las condiciones del mercado detectadas
        """
        if self.data is None or self.data.empty:
            return "Sin datos", {}
            
        range_data = self.get_time_range_data(hours)
        
        if range_data.empty:
            return f"Sin datos para {hours}h", {}
        
        print(f"🔄 Iniciando análisis adaptativo de mínimos para {hours}h...")
        
        # Analizar condiciones y obtener parámetros adaptativos
        self.load_data_and_analyze_conditions(range_data, f"{hours}h")
        
        # Calcular contexto del precio con parámetros adaptativos
        price_context = self._calculate_adaptive_price_context(range_data)
        
        # Detectar wickdowns con parámetros dinámicos
        wickdowns = self._detect_adaptive_wickdowns(range_data, price_context)
        
        # Detectar mínimos con tiempo corto usando umbrales adaptativos
        short_minimums = self._detect_adaptive_short_minimums(range_data, price_context)
        
        # Combinar y rankear con pesos adaptativos
        ranked_minimums = self._rank_with_adaptive_significance(wickdowns, short_minimums, price_context)
        
        # Formatear respuesta con contexto adaptativo
        simple_answer = self._format_adaptive_response(ranked_minimums)
        
        detailed_data = {
            "analysis_type": "adaptive_low_time_minimums",
            "time_range_hours": hours,
            "market_conditions": self.market_conditions,
            "adaptive_parameters": self.current_parameters,
            "price_context": price_context,
            "wickdowns_detected": len(wickdowns),
            "short_minimums_detected": len(short_minimums),
            "total_minimums": len(ranked_minimums),
            "ranked_minimums": ranked_minimums
        }
        
        return simple_answer, detailed_data
    
    # ... resto de los métodos igual que antes ...
    
    def _calculate_adaptive_price_context(self, data: pd.DataFrame) -> Dict:
        """Calcular contexto de precios con métricas adaptativas"""
        
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
        
        # Métricas adaptativas según régimen
        if base_context["regime"] == "high_volatility":
            # En alta volatilidad, usar ventanas más cortas para contexto
            base_context["local_volatility"] = data['close'].rolling(10).std().iloc[-1]
            base_context["momentum"] = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10] * 100
        else:
            # En mercados estables, usar ventanas más amplias
            base_context["local_volatility"] = data['close'].rolling(30).std().iloc[-1] 
            base_context["momentum"] = (data['close'].iloc[-1] - data['close'].iloc[-30]) / data['close'].iloc[-30] * 100
        
        return base_context
    
    def _detect_adaptive_wickdowns(self, data: pd.DataFrame, context: Dict) -> List[Dict]:
        """Detectar wickdowns usando umbrales adaptativos"""
        wickdowns = []
        
        # Usar umbral de wick adaptativo
        wick_threshold = self.get_adaptive_wick_threshold()
        
        print(f"🔍 Detectando wickdowns con umbral adaptativo: {wick_threshold:.3f}%")
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # Calcular cuerpo y mecha inferior
            body_bottom = min(open_price, close_price)
            lower_wick = body_bottom - low_price
            lower_wick_pct = (lower_wick / low_price) * 100
            body_size = abs(close_price - open_price)
            
            # Usar umbral adaptativo
            is_significant_wick = lower_wick_pct >= wick_threshold
            
            # Criterio adaptativo para body vs wick
            regime = context.get("regime", "smooth_ranging")
            if regime == "high_volatility":
                wick_vs_body_ratio = 0.3  # Más permisivo en volatilidad
            else:
                wick_vs_body_ratio = 0.5  # Más estricto en mercados calmos
            
            has_long_wick = lower_wick > body_size * wick_vs_body_ratio
            
            if is_significant_wick and has_long_wick:
                # Calcular significancia con pesos adaptativos
                significance = self._calculate_adaptive_wickdown_significance(
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
                    'significance_score': significance,
                    'adaptive_threshold_used': wick_threshold,
                    'regime': regime
                })
        
        print(f"✅ Detectados {len(wickdowns)} wickdowns con criterios adaptativos")
        return wickdowns
    
    def _detect_adaptive_short_minimums(self, data: pd.DataFrame, context: Dict) -> List[Dict]:
        """Detectar mínimos con permanencia corta usando parámetros adaptativos"""
        short_minimums = []
        
        # Usar ventanas adaptativas
        window_sizes = self.get_adaptive_window_sizes()
        max_time_threshold = self.get_adaptive_time_threshold()
        
        print(f"🔍 Detectando mínimos cortos con ventanas {window_sizes} y tiempo máx {max_time_threshold}min")
        
        for window_size in window_sizes:
            for i in range(window_size, len(data) - window_size):
                current_low = data.iloc[i]['low']
                current_timestamp = data.index[i]
                
                # Verificar si es mínimo local en ventana adaptativa
                window_lows = data.iloc[i-window_size:i+window_size+1]['low']
                
                if current_low == window_lows.min():
                    # Calcular permanencia con tolerancia adaptativa según volatilidad
                    volatility = context.get("volatility", 5.0)
                    if volatility > 10:
                        tolerance = current_low * 0.004  # 0.4% en alta volatilidad
                    elif volatility < 3:
                        tolerance = current_low * 0.001  # 0.1% en baja volatilidad
                    else:
                        tolerance = current_low * 0.002  # 0.2% en volatilidad normal
                    
                    # Contar tiempo cerca del nivel con tolerancia adaptativa
                    near_level_count = 0
                    search_range = min(30, len(data) - i - 1)  # Buscar hacia adelante
                    
                    for j in range(max(0, i-15), min(len(data), i+search_range)):
                        if abs(data.iloc[j]['low'] - current_low) <= tolerance:
                            near_level_count += 1
                    
                    if near_level_count <= max_time_threshold:
                        # Calcular significancia con pesos adaptativos
                        significance = self._calculate_adaptive_minimum_significance(
                            current_low, near_level_count, context, current_timestamp, data
                        )
                        
                        short_minimums.append({
                            'timestamp': current_timestamp,
                            'time_str': current_timestamp.strftime('%H:%M'),
                            'price': current_low,
                            'duration_estimate': near_level_count,
                            'type': 'short_minimum',
                            'significance_score': significance,
                            'window_size_used': window_size,
                            'tolerance_used': tolerance,
                            'max_time_used': max_time_threshold
                        })
        
        # Eliminar duplicados con criterio adaptativo
        unique_minimums = self._remove_duplicates_adaptive(short_minimums)
        
        print(f"✅ Detectados {len(unique_minimums)} mínimos cortos únicos")
        return unique_minimums
    
    def _calculate_adaptive_wickdown_significance(self, price: float, wick_pct: float, 
                                                context: Dict, timestamp, data: pd.DataFrame) -> float:
        """Calcular significancia de wickdown con pesos adaptativos"""
        
        # Obtener pesos adaptativos
        weights = self.get_significance_weights()
        
        # Factor 1: Profundidad del precio (peso adaptativo)
        depth_factor = ((price - context['min_price']) / context['price_range']) * 100
        depth_score = (100 - depth_factor) * weights['depth']  # Invertir: más bajo = más significativo
        
        # Factor 2: Tamaño del wick (peso adaptativo)
        wick_factor = min(wick_pct * 15, 100)  # Escalar y limitar
        wick_score = wick_factor * weights['time']
        
        # Factor 3: Contexto del mercado (peso adaptativo)
        regime = context.get('regime', 'smooth_ranging')
        if regime == 'high_volatility':
            context_bonus = 10  # Menos bonus en volatilidad (es más común)
        elif regime == 'low_volatility':
            context_bonus = 25  # Más bonus en calma (es más raro)
        else:
            context_bonus = 15
        
        context_score = context_bonus * weights['context']
        
        # Factor 4: Momento del mercado
        momentum = context.get('momentum', 0)
        if momentum < -2:  # Caída fuerte
            momentum_bonus = 10  # Wickdown en caída es significativo
        else:
            momentum_bonus = 5
        
        total_significance = depth_score + wick_score + context_score + momentum_bonus
        
        return min(total_significance, 100)  # Limitar a 100
    
    def _calculate_adaptive_minimum_significance(self, price: float, duration: int, 
                                               context: Dict, timestamp, data: pd.DataFrame) -> float:
        """Calcular significancia de mínimo con criterios adaptativos"""
        
        weights = self.get_significance_weights()
        max_time = self.get_adaptive_time_threshold()
        
        # Factor 1: Profundidad (peso adaptativo)
        depth_factor = ((price - context['min_price']) / context['price_range']) * 100
        depth_score = (100 - depth_factor) * weights['depth']
        
        # Factor 2: Brevedad (peso adaptativo)
        brevity_factor = max(0, (max_time - duration) / max_time * 100)
        brevity_score = brevity_factor * weights['time']
        
        # Factor 3: Contexto adaptativo
        regime = context.get('regime', 'smooth_ranging')
        volatility = context.get('volatility', 5.0)
        
        if regime == 'trending':
            context_bonus = 20  # Mínimos en tendencia son más significativos
        elif volatility > 10:
            context_bonus = 8   # Menos significativo en alta volatilidad
        else:
            context_bonus = 15
        
        context_score = context_bonus * weights['context']
        
        total_significance = depth_score + brevity_score + context_score
        
        return min(total_significance, 100)
    
    def _remove_duplicates_adaptive(self, minimums: List[Dict]) -> List[Dict]:
        """Eliminar duplicados con criterios adaptativos"""
        if not minimums:
            return []
        
        # Usar separación mínima adaptativa
        min_separation = self.current_parameters.get("min_separation_minutes", 20)
        
        minimums.sort(key=lambda x: x['timestamp'])
        unique = [minimums[0]]
        
        for minimum in minimums[1:]:
            time_diff = abs((minimum['timestamp'] - unique[-1]['timestamp']).total_seconds()) / 60
            price_diff_pct = abs(minimum['price'] - unique[-1]['price']) / unique[-1]['price'] * 100
            
            # Criterios adaptativos para considerar duplicados
            volatility = self.market_conditions.get("volatility", 5.0)
            if volatility > 10:
                price_tolerance = 0.5  # Más tolerancia en alta volatilidad
            else:
                price_tolerance = 0.2  # Menos tolerancia en baja volatilidad
            
            if time_diff >= min_separation or price_diff_pct >= price_tolerance:
                unique.append(minimum)
            else:
                # Mantener el más significativo
                if minimum['significance_score'] > unique[-1]['significance_score']:
                    unique[-1] = minimum
        
        return unique
    
    def _rank_with_adaptive_significance(self, wickdowns: List[Dict], 
                                       short_minimums: List[Dict], context: Dict) -> List[Dict]:
        """Rankear con significancia adaptativa"""
        
        all_minimums = wickdowns + short_minimums
        
        # Eliminar duplicados con criterios adaptativos
        all_minimums = self._remove_duplicates_adaptive(all_minimums)
        
        # Ordenar por significancia (más alto primero)
        all_minimums.sort(key=lambda x: x['significance_score'], reverse=True)
        
        # Agregar ranking
        for i, minimum in enumerate(all_minimums):
            minimum['rank'] = i + 1
            minimum['adaptive_analysis'] = True
        
        return all_minimums
    
    def _format_adaptive_response(self, minimums: List[Dict]) -> str:
        """Formatear respuesta con información adaptativa"""
        if not minimums:
            regime = self.market_conditions.get("market_regime", "unknown")
            return f"no hay mínimos rápidos detectados (régimen: {regime})"
        
        # Tomar los 4 más significativos
        top_minimums = minimums[:4]
        
        descriptions = []
        for minimum in top_minimums:
            if minimum['type'] == 'wickdown':
                desc = f"${minimum['price']:,.0f} wick {minimum['time_str']} ({minimum['wick_size_pct']:.2f}%, rank #{minimum['rank']})"
            else:
                desc = f"${minimum['price']:,.0f} {minimum['duration_estimate']}min {minimum['time_str']} (rank #{minimum['rank']})"
            
            descriptions.append(desc)
        
        regime = self.market_conditions.get("market_regime", "unknown")
        volatility = self.market_conditions.get("volatility", 0)
        
        return f"mínimos adaptativos ({regime}, vol {volatility:.1f}%): " + ", ".join(descriptions)