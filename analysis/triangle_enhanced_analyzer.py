import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

@dataclass
class EnhancedSignal:
    """Señal mejorada que combina extremos adaptativos + triángulos"""
    
    # Señal base (del sistema existente)
    base_signal_type: str
    base_confidence: float
    
    # Información de triángulos
    triangle_detected: bool
    triangle_type: str  # 'ascending', 'descending', 'symmetrical', 'none'
    triangle_confidence: float
    
    # Timing refinado
    entry_timing: str  # 'immediate', 'wait_for_breakout', 'wait_for_pullback', 'avoid'
    time_to_optimal_entry: float  # Horas hasta entrada óptima
    
    # Targets refinados
    primary_target: float
    secondary_target: float
    stop_loss: float
    risk_reward_ratio: float
    
    # Contexto completo
    market_structure: str  # Del análisis adaptativo
    triangle_position_pct: float  # Posición dentro del triángulo (0-100%)
    volatility_squeeze: float
    breakout_probability: float
    
    # Señal final
    final_signal_type: str
    final_confidence: float
    reasoning: List[str]

class TriangleEnhancedAnalyzer:
    """Analizador que combina extremos adaptativos con detección de triángulos"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP"):
        self.symbol = symbol
        
    def analyze_with_triangle_enhancement(self, 
                                        adaptive_results: Dict[str, Any],
                                        ml_results: Dict[str, Any],
                                        market_data: pd.DataFrame) -> EnhancedSignal:
        """Análisis completo combinando sistemas"""
        
        print(f"🔺 Iniciando análisis híbrido: Extremos + Triángulos")
        print(f"📅 Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"👤 Usuario: biGGzeta")
        
        # 1. Extraer señal base del ML
        base_signal = ml_results.get('signal')
        base_signal_type = base_signal.signal_type.value if base_signal else 'HOLD'
        base_confidence = base_signal.confidence if base_signal else 50
        
        # 2. Extraer estructura del mercado de extremos adaptativos
        market_structure = self._extract_market_structure(adaptive_results)
        
        # 3. Detectar triángulos en los datos
        triangle_analysis = self._detect_triangles_from_extremes(adaptive_results, market_data)
        
        # 4. Análizar timing óptimo
        timing_analysis = self._analyze_optimal_timing(
            market_structure, triangle_analysis, market_data
        )
        
        # 5. Refinar targets y stops
        refined_targets = self._refine_targets_with_triangles(
            base_signal, triangle_analysis, market_data
        )
        
        # 6. Generar señal final mejorada
        enhanced_signal = self._generate_enhanced_signal(
            base_signal_type, base_confidence, market_structure, 
            triangle_analysis, timing_analysis, refined_targets
        )
        
        print(f"✅ Análisis híbrido completado:")
        print(f"   📊 Señal base: {base_signal_type} ({base_confidence}%)")
        print(f"   🔺 Triángulo: {triangle_analysis['type']} ({triangle_analysis['confidence']}%)")
        print(f"   🎯 Señal final: {enhanced_signal.final_signal_type} ({enhanced_signal.final_confidence}%)")
        print(f"   ⏰ Timing: {enhanced_signal.entry_timing}")
        
        return enhanced_signal
    
    def _extract_market_structure(self, adaptive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer estructura del mercado desde extremos adaptativos"""
        
        structure = {
            'trend_direction': 'neutral',
            'strength': 0,
            'extremes_alignment': False,
            'momentum_3d': 0,
            'position_in_range': 50,
            'regime': 'unknown'
        }
        
        # Extraer datos del análisis semanal
        weekly_data = adaptive_results.get('weekly_adaptive', ('', {}))[1] if isinstance(adaptive_results.get('weekly_adaptive'), tuple) else {}
        
        if isinstance(weekly_data, dict):
            maximos_trend = weekly_data.get('maximos_trend', {})
            minimos_trend = weekly_data.get('minimos_trend', {})
            market_conditions = weekly_data.get('market_conditions', {})
            
            # Determinar dirección de tendencia
            if (maximos_trend.get('trend') == 'crecientes' and 
                minimos_trend.get('trend') == 'crecientes'):
                structure['trend_direction'] = 'bullish'
                structure['extremes_alignment'] = True
                structure['strength'] = min(maximos_trend.get('trend_strength', 0),
                                          minimos_trend.get('trend_strength', 0))
            elif (maximos_trend.get('trend') == 'decrecientes' and 
                  minimos_trend.get('trend') == 'decrecientes'):
                structure['trend_direction'] = 'bearish' 
                structure['extremes_alignment'] = True
                structure['strength'] = min(maximos_trend.get('trend_strength', 0),
                                          minimos_trend.get('trend_strength', 0))
            else:
                structure['trend_direction'] = 'mixed'
                structure['extremes_alignment'] = False
            
            structure['regime'] = market_conditions.get('market_regime', 'unknown')
            structure['position_in_range'] = weekly_data.get('range_position_pct', 50)
        
        return structure
    
    def _detect_triangles_from_extremes(self, adaptive_results: Dict[str, Any], 
                                      market_data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar triángulos usando los extremos adaptativos ya calculados"""
        
        # Extraer extremos de los análisis adaptativos
        highs = self._extract_adaptive_highs(adaptive_results)
        lows = self._extract_adaptive_lows(adaptive_results)
        
        if len(highs) < 3 or len(lows) < 3:
            return {
                'detected': False,
                'type': 'none',
                'confidence': 0,
                'formation_time_hours': 0,
                'position_pct': 0,
                'volatility_squeeze': 0,
                'breakout_probability': 0
            }
        
        print(f"🔍 Analizando {len(highs)} máximos y {len(lows)} mínimos adaptativos")
        
        # Detectar patrones de triángulo
        triangle_patterns = []
        
        # 1. Triángulo ascendente: máximos horizontales + mínimos ascendentes
        ascending = self._check_ascending_triangle(highs, lows)
        if ascending['confidence'] > 60:
            triangle_patterns.append(ascending)
        
        # 2. Triángulo descendente: mínimos horizontales + máximos descendentes  
        descending = self._check_descending_triangle(highs, lows)
        if descending['confidence'] > 60:
            triangle_patterns.append(descending)
        
        # 3. Triángulo simétrico: máximos descendentes + mínimos ascendentes
        symmetrical = self._check_symmetrical_triangle(highs, lows)
        if symmetrical['confidence'] > 60:
            triangle_patterns.append(symmetrical)
        
        # Seleccionar el mejor patrón
        if triangle_patterns:
            best_triangle = max(triangle_patterns, key=lambda x: x['confidence'])
            best_triangle['detected'] = True
            
            # Calcular volatilidad squeeze
            best_triangle['volatility_squeeze'] = self._calculate_volatility_squeeze_simple(market_data)
            
            # Calcular probabilidad de breakout basada en posición y squeeze
            best_triangle['breakout_probability'] = self._calculate_breakout_probability(best_triangle)
            
            return best_triangle
        
        return {
            'detected': False,
            'type': 'none', 
            'confidence': 0,
            'formation_time_hours': 0,
            'position_pct': 0,
            'volatility_squeeze': 0,
            'breakout_probability': 0
        }
    
    def _extract_adaptive_highs(self, adaptive_results: Dict[str, Any]) -> List[Dict]:
        """Extraer máximos del análisis de máximos adaptativos"""
        
        highs = []
        
        # Del análisis de máximos adaptativos (Pregunta 4)
        maximums_data = adaptive_results.get('high_maximums_adaptive', ('', {}))[1] if isinstance(adaptive_results.get('high_maximums_adaptive'), tuple) else {}
        
        if isinstance(maximums_data, dict):
            ranked_maximums = maximums_data.get('ranked_maximums', [])
            
            for i, maximum in enumerate(ranked_maximums[:6]):  # Top 6 máximos
                if isinstance(maximum, dict):
                    highs.append({
                        'price': maximum.get('price', 0),
                        'time_str': maximum.get('time', ''),
                        'rank': maximum.get('rank', i+1),
                        'significance': maximum.get('wick_size_pct', 0) * 100,
                        'type': maximum.get('type', 'unknown')
                    })
        
        # Ordenar por tiempo (más reciente primero)
        highs.sort(key=lambda x: x['rank'])
        
        return highs
    
    def _extract_adaptive_lows(self, adaptive_results: Dict[str, Any]) -> List[Dict]:
        """Extraer mínimos del análisis de mínimos adaptativos"""
        
        lows = []
        
        # Del análisis de mínimos adaptativos (Pregunta 3)
        minimums_data = adaptive_results.get('low_minimums_adaptive', ('', {}))[1] if isinstance(adaptive_results.get('low_minimums_adaptive'), tuple) else {}
        
        if isinstance(minimums_data, dict):
            ranked_minimums = minimums_data.get('ranked_minimums', [])
            
            for i, minimum in enumerate(ranked_minimums[:6]):  # Top 6 mínimos
                if isinstance(minimum, dict):
                    lows.append({
                        'price': minimum.get('price', 0),
                        'time_str': minimum.get('time', ''),
                        'rank': minimum.get('rank', i+1),
                        'significance': minimum.get('wick_size_pct', 0) * 100,
                        'type': minimum.get('type', 'unknown')
                    })
        
        # Ordenar por tiempo
        lows.sort(key=lambda x: x['rank'])
        
        return lows
    
    def _check_ascending_triangle(self, highs: List[Dict], lows: List[Dict]) -> Dict[str, Any]:
        """Verificar patrón de triángulo ascendente"""
        
        if len(highs) < 3 or len(lows) < 3:
            return {'type': 'ascending', 'confidence': 0}
        
        # Verificar máximos horizontales (resistencia)
        recent_highs = [h['price'] for h in highs[:4]]  # Últimos 4 máximos
        high_prices = np.array(recent_highs)
        high_mean = np.mean(high_prices)
        high_std = np.std(high_prices)
        
        # Resistencia horizontal si la desviación es pequeña
        horizontal_resistance = (high_std / high_mean * 100) < 1.0  # Menos del 1% de variación
        
        # Verificar mínimos ascendentes (soporte)
        recent_lows = [l['price'] for l in lows[:4]]  # Últimos 4 mínimos
        
        if len(recent_lows) < 3:
            return {'type': 'ascending', 'confidence': 0}
        
        # Calcular tendencia de los mínimos (regresión lineal simple)
        x = np.arange(len(recent_lows))
        y = np.array(recent_lows)
        slope = np.polyfit(x, y, 1)[0]
        
        ascending_support = slope > 0  # Pendiente positiva
        
        # Calcular confianza
        confidence = 0
        if horizontal_resistance:
            confidence += 40
        if ascending_support:
            confidence += 40
        
        # Bonus por significancia de extremos
        avg_significance = np.mean([h.get('significance', 0) for h in highs[:3]])
        confidence += min(20, avg_significance / 5)
        
        # Calcular posición en el triángulo
        current_range = high_mean - recent_lows[-1]
        initial_range = high_mean - recent_lows[0]
        position_pct = ((initial_range - current_range) / initial_range * 100) if initial_range > 0 else 0
        
        return {
            'type': 'ascending',
            'confidence': min(100, confidence),
            'resistance_level': high_mean,
            'support_slope': slope,
            'formation_time_hours': 3,  # Estimado basado en análisis 3h
            'position_pct': max(0, min(100, position_pct)),
            'expected_breakout_direction': 'bullish'
        }
    
    def _check_descending_triangle(self, highs: List[Dict], lows: List[Dict]) -> Dict[str, Any]:
        """Verificar patrón de triángulo descendente"""
        
        if len(highs) < 3 or len(lows) < 3:
            return {'type': 'descending', 'confidence': 0}
        
        # Verificar mínimos horizontales (soporte)
        recent_lows = [l['price'] for l in lows[:4]]
        low_prices = np.array(recent_lows)
        low_mean = np.mean(low_prices)
        low_std = np.std(low_prices)
        
        horizontal_support = (low_std / low_mean * 100) < 1.0
        
        # Verificar máximos descendentes (resistencia)
        recent_highs = [h['price'] for h in highs[:4]]
        
        if len(recent_highs) < 3:
            return {'type': 'descending', 'confidence': 0}
        
        x = np.arange(len(recent_highs))
        y = np.array(recent_highs)
        slope = np.polyfit(x, y, 1)[0]
        
        descending_resistance = slope < 0  # Pendiente negativa
        
        # Calcular confianza
        confidence = 0
        if horizontal_support:
            confidence += 40
        if descending_resistance:
            confidence += 40
        
        avg_significance = np.mean([l.get('significance', 0) for l in lows[:3]])
        confidence += min(20, avg_significance / 5)
        
        # Posición en triángulo
        current_range = recent_highs[-1] - low_mean
        initial_range = recent_highs[0] - low_mean
        position_pct = ((initial_range - current_range) / initial_range * 100) if initial_range > 0 else 0
        
        return {
            'type': 'descending',
            'confidence': min(100, confidence),
            'support_level': low_mean,
            'resistance_slope': slope,
            'formation_time_hours': 3,
            'position_pct': max(0, min(100, position_pct)),
            'expected_breakout_direction': 'bearish'
        }
    
    def _check_symmetrical_triangle(self, highs: List[Dict], lows: List[Dict]) -> Dict[str, Any]:
        """Verificar patrón de triángulo simétrico"""
        
        if len(highs) < 3 or len(lows) < 3:
            return {'type': 'symmetrical', 'confidence': 0}
        
        # Máximos descendentes
        recent_highs = [h['price'] for h in highs[:4]]
        if len(recent_highs) < 3:
            return {'type': 'symmetrical', 'confidence': 0}
        
        x_highs = np.arange(len(recent_highs))
        high_slope = np.polyfit(x_highs, recent_highs, 1)[0]
        
        # Mínimos ascendentes
        recent_lows = [l['price'] for l in lows[:4]]
        if len(recent_lows) < 3:
            return {'type': 'symmetrical', 'confidence': 0}
        
        x_lows = np.arange(len(recent_lows))
        low_slope = np.polyfit(x_lows, recent_lows, 1)[0]
        
        # Verificar convergencia (máximos bajando, mínimos subiendo)
        converging = high_slope < 0 and low_slope > 0
        
        # Verificar simetría (pendientes similares en magnitud)
        slope_ratio = abs(high_slope / low_slope) if low_slope != 0 else 0
        symmetric = 0.5 <= slope_ratio <= 2.0
        
        confidence = 0
        if converging:
            confidence += 50
        if symmetric:
            confidence += 30
        
        # Bonus por balance de significancia
        high_significance = np.mean([h.get('significance', 0) for h in highs[:3]])
        low_significance = np.mean([l.get('significance', 0) for l in lows[:3]])
        balance = min(high_significance, low_significance)
        confidence += min(20, balance / 5)
        
        # Posición en triángulo (convergencia)
        current_range = recent_highs[-1] - recent_lows[-1]
        initial_range = recent_highs[0] - recent_lows[0]
        position_pct = ((initial_range - current_range) / initial_range * 100) if initial_range > 0 else 0
        
        return {
            'type': 'symmetrical',
            'confidence': min(100, confidence),
            'resistance_slope': high_slope,
            'support_slope': low_slope,
            'formation_time_hours': 3,
            'position_pct': max(0, min(100, position_pct)),
            'expected_breakout_direction': 'neutral'  # Puede ir en cualquier dirección
        }
    
    def _calculate_volatility_squeeze_simple(self, market_data: pd.DataFrame) -> float:
        """Calcular compresión de volatilidad de forma simple"""
        
        if len(market_data) < 20:
            return 0
        
        # Volatilidad reciente (últimas 20 velas)
        recent_data = market_data.tail(20)
        recent_vol = recent_data['high'].pct_change().std() * 100
        
        # Volatilidad de referencia (últimas 60 velas)
        reference_data = market_data.tail(60)
        reference_vol = reference_data['high'].pct_change().std() * 100
        
        if reference_vol == 0:
            return 0
        
        # Squeeze = reducción de volatilidad
        squeeze = ((reference_vol - recent_vol) / reference_vol) * 100
        return max(0, min(100, squeeze))
    
    def _calculate_breakout_probability(self, triangle: Dict[str, Any]) -> float:
        """Calcular probabilidad de breakout basada en varios factores"""
        
        probability = 0
        
        # Factor 1: Posición en el triángulo (más cerca del apex = mayor probabilidad)
        position_factor = triangle['position_pct'] / 100 * 40
        probability += position_factor
        
        # Factor 2: Confianza del patrón
        confidence_factor = triangle['confidence'] / 100 * 30
        probability += confidence_factor
        
        # Factor 3: Volatilidad squeeze
        squeeze_factor = triangle['volatility_squeeze'] / 100 * 20
        probability += squeeze_factor
        
        # Factor 4: Tiempo de formación (optimal 4-12 horas)
        time_hours = triangle['formation_time_hours']
        if 4 <= time_hours <= 12:
            probability += 10
        elif 2 <= time_hours <= 20:
            probability += 5
        
        return min(100, probability)
    
    def _analyze_optimal_timing(self, market_structure: Dict, triangle_analysis: Dict, 
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analizar timing óptimo para entrada"""
        
        timing = {
            'recommendation': 'wait',
            'time_to_optimal_hours': 0,
            'entry_conditions': [],
            'avoid_conditions': []
        }
        
        if not triangle_analysis['detected']:
            # Sin triángulo, usar lógica tradicional
            if market_structure['extremes_alignment']:
                timing['recommendation'] = 'immediate'
                timing['entry_conditions'].append('Extremos alineados - estructura clara')
            else:
                timing['recommendation'] = 'wait_for_clarity'
                timing['entry_conditions'].append('Esperar alineación de extremos')
            return timing
        
        # Con triángulo detectado
        triangle_position = triangle_analysis['position_pct']
        breakout_probability = triangle_analysis['breakout_probability']
        
        if triangle_position > 85 and breakout_probability > 70:
            timing['recommendation'] = 'immediate'
            timing['entry_conditions'].append('Breakout inminente - alta probabilidad')
        elif triangle_position > 70:
            timing['recommendation'] = 'wait_for_breakout'
            timing['time_to_optimal_hours'] = (100 - triangle_position) / 10  # Estimación
            timing['entry_conditions'].append('Esperar confirmación de breakout')
        elif triangle_position < 40:
            timing['recommendation'] = 'wait_for_development'
            timing['time_to_optimal_hours'] = triangle_position / 20
            timing['entry_conditions'].append('Triángulo en desarrollo - esperar más formación')
        else:
            timing['recommendation'] = 'monitor'
            timing['entry_conditions'].append('Posición intermedia en triángulo - monitorear')
        
        return timing
    
    def _refine_targets_with_triangles(self, base_signal, triangle_analysis: Dict, 
                                     market_data: pd.DataFrame) -> Dict[str, float]:
        """Refinar targets usando geometría del triángulo"""
        
        current_price = market_data['close'].iloc[-1]
        
        # Targets base del signal original
        base_target = base_signal.price_target if base_signal and base_signal.price_target else current_price * 1.02
        base_stop = base_signal.stop_loss if base_signal and base_signal.stop_loss else current_price * 0.98
        
        if not triangle_analysis['detected']:
            return {
                'primary_target': base_target,
                'secondary_target': base_target * 1.01,
                'stop_loss': base_stop,
                'risk_reward_ratio': abs((base_target - current_price) / (current_price - base_stop)) if base_stop != current_price else 1
            }
        
        # Calcular targets basados en triángulo
        triangle_type = triangle_analysis['type']
        
        if triangle_type == 'ascending':
            # Target = altura del triángulo proyectada desde breakout
            resistance_level = triangle_analysis.get('resistance_level', current_price * 1.01)
            triangle_height = resistance_level - current_price  # Estimación simplificada
            
            primary_target = resistance_level + triangle_height
            secondary_target = resistance_level + (triangle_height * 1.5)
            stop_loss = current_price * 0.985  # Tight stop para triángulo
            
        elif triangle_type == 'descending':
            support_level = triangle_analysis.get('support_level', current_price * 0.99)
            triangle_height = current_price - support_level
            
            primary_target = support_level - triangle_height
            secondary_target = support_level - (triangle_height * 1.5)
            stop_loss = current_price * 1.015
            
        else:  # symmetrical
            # Para simétrico, usar altura actual del triángulo
            triangle_height = current_price * 0.02  # Estimación 2%
            
            if triangle_analysis.get('expected_breakout_direction') == 'bullish':
                primary_target = current_price + triangle_height
                secondary_target = current_price + (triangle_height * 1.5)
                stop_loss = current_price - (triangle_height * 0.5)
            else:
                primary_target = current_price - triangle_height
                secondary_target = current_price - (triangle_height * 1.5)
                stop_loss = current_price + (triangle_height * 0.5)
        
        # Calcular risk/reward
        risk = abs(current_price - stop_loss)
        reward = abs(primary_target - current_price)
        risk_reward_ratio = reward / risk if risk > 0 else 1
        
        return {
            'primary_target': primary_target,
            'secondary_target': secondary_target,
            'stop_loss': stop_loss,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def _generate_enhanced_signal(self, base_signal_type: str, base_confidence: float,
                                market_structure: Dict, triangle_analysis: Dict,
                                timing_analysis: Dict, refined_targets: Dict) -> EnhancedSignal:
        """Generar señal final mejorada"""
        
        # Combinar confianzas
        triangle_bonus = 0
        if triangle_analysis['detected']:
            triangle_bonus = triangle_analysis['confidence'] * 0.3  # 30% de peso
        
        structure_bonus = 0
        if market_structure['extremes_alignment']:
            structure_bonus = market_structure['strength'] * 0.2  # 20% de peso
        
        final_confidence = min(100, base_confidence + triangle_bonus + structure_bonus)
        
        # Ajustar señal según timing
        final_signal_type = base_signal_type
        
        if timing_analysis['recommendation'] == 'immediate' and triangle_analysis['detected']:
            # Upgrade señal si hay confirmación de triángulo
            if base_signal_type in ['BUY', 'WEAK_BUY']:
                final_signal_type = 'STRONG_BUY'
            elif base_signal_type in ['SELL', 'WEAK_SELL']:
                final_signal_type = 'STRONG_SELL'
        elif timing_analysis['recommendation'] in ['wait_for_breakout', 'wait_for_development']:
            # Downgrade si timing no es óptimo
            if base_signal_type in ['STRONG_BUY', 'STRONG_SELL']:
                final_signal_type = 'HOLD'  # Esperar mejor momento
        
        # Generar reasoning
        reasoning = [
            f"Señal base: {base_signal_type} ({base_confidence}%)",
            f"Estructura de mercado: {market_structure['trend_direction']} (alineación: {market_structure['extremes_alignment']})"
        ]
        
        if triangle_analysis['detected']:
            reasoning.append(f"Triángulo {triangle_analysis['type']} detectado ({triangle_analysis['confidence']:.0f}% confianza)")
            reasoning.append(f"Posición en triángulo: {triangle_analysis['position_pct']:.0f}%")
            reasoning.append(f"Probabilidad breakout: {triangle_analysis['breakout_probability']:.0f}%")
        
        reasoning.append(f"Timing recomendado: {timing_analysis['recommendation']}")
        
        if timing_analysis['time_to_optimal_hours'] > 0:
            reasoning.append(f"Tiempo hasta entrada óptima: {timing_analysis['time_to_optimal_hours']:.1f}h")
        
        return EnhancedSignal(
            base_signal_type=base_signal_type,
            base_confidence=base_confidence,
            triangle_detected=triangle_analysis['detected'],
            triangle_type=triangle_analysis['type'],
            triangle_confidence=triangle_analysis['confidence'],
            entry_timing=timing_analysis['recommendation'],
            time_to_optimal_entry=timing_analysis['time_to_optimal_hours'],
            primary_target=refined_targets['primary_target'],
            secondary_target=refined_targets['secondary_target'],
            stop_loss=refined_targets['stop_loss'],
            risk_reward_ratio=refined_targets['risk_reward_ratio'],
            market_structure=market_structure['trend_direction'],
            triangle_position_pct=triangle_analysis['position_pct'],
            volatility_squeeze=triangle_analysis['volatility_squeeze'],
            breakout_probability=triangle_analysis['breakout_probability'],
            final_signal_type=final_signal_type,
            final_confidence=final_confidence,
            reasoning=reasoning
        )

def test_triangle_enhanced_analysis():
    """Test del sistema híbrido"""
    
    print(f"🔺 Testing Triangle-Enhanced Analysis System")
    print(f"📅 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"👤 Usuario: biGGzeta")
    
    # Este sería llamado después del análisis adaptativo normal
    # analyzer = TriangleEnhancedAnalyzer("ETHUSD_PERP")
    # enhanced_signal = analyzer.analyze_with_triangle_enhancement(adaptive_results, ml_results, market_data)
    
    print("Sistema híbrido listo para integración! 🚀")

if __name__ == "__main__":
    test_triangle_enhanced_analysis()