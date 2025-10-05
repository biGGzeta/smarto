#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor de Interpretación Probabilística
Convierte señales quirúrgicas en probabilidades claras de precio
Autor: biGGzeta
Fecha: 2025-10-04
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os

class ProbabilisticInterpreter:
    """Motor que convierte señales quirúrgicas en probabilidades claras"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP"):
        self.symbol = symbol
        self.timestamp = datetime.utcnow()
        
        # Crear directorio de logs
        self.logs_dir = "logs/probabilities"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        print(f"🎯 Motor de Interpretación Probabilística iniciado para {symbol}")
    
    def interpret_complete_analysis(self, analysis_results: Dict) -> Dict:
        """Interpretar análisis completo y generar probabilidades"""
        
        print(f"\n🧠 INICIANDO INTERPRETACIÓN PROBABILÍSTICA:")
        print("=" * 60)
        
        # Extraer las 6 señales quirúrgicas
        qa_signals = self._extract_qa_signals(analysis_results)
        
        # Interpretar cada señal
        interpretations = {
            'range_signal': self._interpret_range_dynamics(qa_signals.get('q1', {})),
            'trend_signal': self._interpret_trend_probabilities(qa_signals.get('q2', {})),
            'support_signal': self._interpret_support_strength(qa_signals.get('q3', {})),
            'resistance_signal': self._interpret_resistance_pressure(qa_signals.get('q4', {})),
            'context_signal': self._interpret_market_context(qa_signals.get('q5', {})),
            'structure_signal': self._interpret_structural_bias(qa_signals.get('q6', {}))
        }
        
        # Sintetizar en escenarios probabilísticos
        scenarios = self._create_scenarios(interpretations)
        
        # Resultado final
        result = {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'interpretations': interpretations,
            'scenarios': scenarios,
            'primary_recommendation': scenarios[0] if scenarios else {},
            'signal_for_external': self._generate_external_signal(scenarios)
        }
        
        # Generar logs
        self._save_logs(result)
        
        return result
    
    def _extract_qa_signals(self, analysis_results: Dict) -> Dict:
        """Extraer las 6 señales quirúrgicas del análisis"""
        
        adaptive_results = analysis_results.get('adaptive_results', {})
        
        return {
            'q1': adaptive_results.get('max_min_basic', ('', {}))[1] if isinstance(adaptive_results.get('max_min_basic'), tuple) else {},
            'q2': adaptive_results.get('percentage_adaptive', ('', {}))[1] if isinstance(adaptive_results.get('percentage_adaptive'), tuple) else {},
            'q3': adaptive_results.get('low_minimums_adaptive', ('', {}))[1] if isinstance(adaptive_results.get('low_minimums_adaptive'), tuple) else {},
            'q4': adaptive_results.get('high_maximums_adaptive', ('', {}))[1] if isinstance(adaptive_results.get('high_maximums_adaptive'), tuple) else {},
            'q5': adaptive_results.get('panorama_48h_adaptive', ('', {}))[1] if isinstance(adaptive_results.get('panorama_48h_adaptive'), tuple) else {},
            'q6': adaptive_results.get('weekly_adaptive', ('', {}))[1] if isinstance(adaptive_results.get('weekly_adaptive'), tuple) else {}
        }
    
    def _interpret_range_dynamics(self, max_min_data: Dict) -> Dict:
        """
        Interpretar dinámicas de rango (Pregunta 1)
        De: 'ETH 4,495-4,451 (Volatilidad: 1.00%)'
        A: Probabilidades de breakout con magnitudes específicas
        """
        
        range_pct = max_min_data.get('percentage_range', 0)
        max_price = max_min_data.get('max_price', 0)
        min_price = max_min_data.get('min_price', 0)
        range_abs = max_price - min_price if max_price and min_price else 0
        
        interpretation = {}
        
        if range_pct < 1.5:  # Rango compacto
            interpretation = {
                'pattern': 'tight_consolidation',
                'breakout_probability': 78,
                'expected_magnitude_pct': round(range_pct * 2.8, 2),
                'time_horizon_hours': 4,
                'coiling_strength': 'high',
                'range_absolute': round(range_abs, 2),
                'interpretation': f'Rango compacto {range_pct:.1f}% (${range_abs:.0f}) indica coiling fuerte. Breakout probable en 4h con magnitud ~{range_pct * 2.8:.1f}%'
            }
        elif range_pct > 3.0:  # Rango amplio
            interpretation = {
                'pattern': 'wide_ranging_consolidation',
                'breakout_probability': 45,
                'expected_magnitude_pct': round(range_pct * 1.2, 2),
                'time_horizon_hours': 8,
                'coiling_strength': 'low',
                'range_absolute': round(range_abs, 2),
                'interpretation': f'Rango amplio {range_pct:.1f}% indica consolidación extendida. Breakout menos probable.'
            }
        else:  # Rango moderado
            interpretation = {
                'pattern': 'moderate_consolidation',
                'breakout_probability': 60,
                'expected_magnitude_pct': round(range_pct * 2.0, 2),
                'time_horizon_hours': 6,
                'coiling_strength': 'medium',
                'range_absolute': round(range_abs, 2),
                'interpretation': f'Rango moderado {range_pct:.1f}% indica consolidación normal.'
            }
        
        print(f"📊 Rango: {interpretation['interpretation']}")
        return interpretation
    
    def _interpret_trend_probabilities(self, percentage_data: Dict) -> Dict:
        """
        Interpretar probabilidades de tendencia (Pregunta 2)
        De: 'tendencia mixta 1.00% lateral → lateral → lateral'
        A: Probabilidades direccionales con sesgos
        """
        
        # Buscar información de tendencia en los datos
        trend_info = percentage_data.get('trend_analysis', {})
        cycles = percentage_data.get('cycles', [])
        
        # Analizar ciclos para detectar bias
        bullish_bias = 0
        bearish_bias = 0
        lateral_count = 0
        
        for cycle in cycles[:3]:  # Últimos 3 ciclos
            if isinstance(cycle, dict):
                trend = cycle.get('trend', 'lateral')
                if 'lateral' in trend.lower():
                    lateral_count += 1
                elif any(word in trend.lower() for word in ['alcista', 'bullish', 'up']):
                    bullish_bias += 1
                elif any(word in trend.lower() for word in ['bajista', 'bearish', 'down']):
                    bearish_bias += 1
        
        # Determinar probabilidades
        if lateral_count >= 2:  # Mayoría lateral
            probabilities = {
                'continuation_lateral': 58,
                'bullish_breakout': 32,
                'bearish_breakdown': 10
            }
            classification = 'indecision_with_slight_bullish_bias'
            reasoning = 'Múltiples períodos laterales con ligero sesgo alcista'
        elif bullish_bias > bearish_bias:
            probabilities = {
                'bullish_breakout': 65,
                'continuation_lateral': 25,
                'bearish_breakdown': 10
            }
            classification = 'bullish_bias_detected'
            reasoning = 'Sesgo alcista detectado en análisis de ciclos'
        else:
            probabilities = {
                'bearish_breakdown': 55,
                'continuation_lateral': 30,
                'bullish_breakout': 15
            }
            classification = 'bearish_bias_detected'
            reasoning = 'Sesgo bajista detectado en análisis de ciclos'
        
        interpretation = {
            'trend_classification': classification,
            'probabilities': probabilities,
            'bias_reasoning': reasoning,
            'confidence_level': 'medium_high',
            'cycles_analyzed': len(cycles),
            'interpretation': f'{classification.replace("_", " ").title()}: {reasoning}'
        }
        
        print(f"📈 Tendencia: {interpretation['interpretation']}")
        return interpretation
    
    def _interpret_support_strength(self, minimums_data: Dict) -> Dict:
        """
        Interpretar fuerza del soporte (Pregunta 3)
        De: '$4,453 → $4,481 → $4,484 (mínimos ascendentes)'
        A: Strength del soporte y probabilidades de hold/break
        """
        
        ranked_minimums = minimums_data.get('ranked_minimums', [])
        
        if len(ranked_minimums) < 2:
            return {
                'support_type': 'insufficient_data',
                'strength_rating': 5.0,
                'hold_probability': 50,
                'interpretation': 'Datos insuficientes para análisis de soporte'
            }
        
        # Extraer precios de los mínimos
        prices = []
        for minimum in ranked_minimums[:3]:
            if isinstance(minimum, dict):
                prices.append(minimum.get('price', 0))
        
        if len(prices) < 2:
            return {
                'support_type': 'insufficient_data',
                'strength_rating': 5.0,
                'hold_probability': 50,
                'interpretation': 'Datos insuficientes para análisis de soporte'
            }
        
        # Analizar progresión de precios
        price_progression = self._analyze_price_progression(prices)
        
        if price_progression == 'ascending':
            strongest_support = max(prices)
            interpretation = {
                'support_type': 'ascending_dynamic',
                'strength_rating': 8.5,
                'hold_probability': 82,
                'dynamic_level': round(strongest_support + 3, 2),
                'prices_sequence': prices,
                'break_consequences': {
                    'probability': 18,
                    'target': round(min(prices) - 25, 2),
                    'severity': 'moderate'
                },
                'interpretation': f'Soporte ascendente robusto ${min(prices):.0f}→${max(prices):.0f}. Próximo test esperado en ${strongest_support + 3:.0f}'
            }
        elif price_progression == 'stable':
            avg_support = np.mean(prices)
            interpretation = {
                'support_type': 'horizontal_stable',
                'strength_rating': 7.8,
                'hold_probability': 75,
                'static_level': round(avg_support, 2),
                'prices_sequence': prices,
                'break_consequences': {
                    'probability': 25,
                    'target': round(avg_support - 30, 2),
                    'severity': 'moderate_to_high'
                },
                'interpretation': f'Soporte horizontal estable ~${avg_support:.0f}. Nivel clave bien definido'
            }
        else:  # descending
            weakest_support = min(prices)
            interpretation = {
                'support_type': 'descending_weak',
                'strength_rating': 4.2,
                'hold_probability': 35,
                'failing_level': round(weakest_support, 2),
                'prices_sequence': prices,
                'break_consequences': {
                    'probability': 65,
                    'target': round(weakest_support - 40, 2),
                    'severity': 'high'
                },
                'interpretation': f'Soporte descendente débil ${max(prices):.0f}→${min(prices):.0f}. Ruptura probable'
            }
        
        print(f"🛡️ Soporte: {interpretation['interpretation']}")
        return interpretation
    
    def _interpret_resistance_pressure(self, maximums_data: Dict) -> Dict:
        """
        Interpretar presión de resistencia (Pregunta 4)
        De: '$4,495 → $4,483 → $4,454 (máximos descendentes)'
        A: Presión bajista y probabilidades de break
        """
        
        ranked_maximums = maximums_data.get('ranked_maximums', [])
        
        if len(ranked_maximums) < 2:
            return {
                'resistance_pattern': 'insufficient_data',
                'pressure_intensity': 5.0,
                'breakdown_probability': 50,
                'interpretation': 'Datos insuficientes para análisis de resistencia'
            }
        
        # Extraer precios de los máximos
        prices = []
        for maximum in ranked_maximums[:3]:
            if isinstance(maximum, dict):
                prices.append(maximum.get('price', 0))
        
        if len(prices) < 2:
            return {
                'resistance_pattern': 'insufficient_data',
                'pressure_intensity': 5.0,
                'breakdown_probability': 50,
                'interpretation': 'Datos insuficientes para análisis de resistencia'
            }
        
        # Analizar progresión de máximos
        price_progression = self._analyze_price_progression(prices)
        
        if price_progression == 'descending':
            highest_resistance = max(prices)
            interpretation = {
                'resistance_pattern': 'weakening_descending',
                'pressure_intensity': 7.8,
                'breakdown_probability': 74,
                'failed_attempts': len(prices),
                'ceiling_level': round(highest_resistance, 2),
                'prices_sequence': prices,
                'breakdown_targets': {
                    'immediate': round(min(prices) - 20, 2),
                    'extended': round(min(prices) - 45, 2)
                },
                'interpretation': f'Resistencia debilitándose ${max(prices):.0f}→${min(prices):.0f}. Breakdown probable'
            }
        elif price_progression == 'stable':
            avg_resistance = np.mean(prices)
            interpretation = {
                'resistance_pattern': 'horizontal_strong',
                'pressure_intensity': 8.5,
                'breakdown_probability': 25,
                'static_ceiling': round(avg_resistance, 2),
                'prices_sequence': prices,
                'breakdown_targets': {
                    'immediate': round(avg_resistance - 35, 2),
                    'extended': round(avg_resistance - 60, 2)
                },
                'interpretation': f'Resistencia horizontal fuerte ~${avg_resistance:.0f}. Difícil de romper'
            }
        else:  # ascending
            strongest_resistance = max(prices)
            interpretation = {
                'resistance_pattern': 'ascending_strengthening',
                'pressure_intensity': 9.2,
                'breakdown_probability': 15,
                'dynamic_ceiling': round(strongest_resistance, 2),
                'prices_sequence': prices,
                'breakdown_targets': {
                    'immediate': round(min(prices) - 15, 2),
                    'extended': round(min(prices) - 30, 2)
                },
                'interpretation': f'Resistencia ascendente ${min(prices):.0f}→${max(prices):.0f}. Muy fuerte'
            }
        
        print(f"⚡ Resistencia: {interpretation['interpretation']}")
        return interpretation
    
    def _interpret_market_context(self, panorama_data: Dict) -> Dict:
        """
        Interpretar contexto de mercado (Pregunta 5)
        De: 'zona alta 0.7% vs zona baja 18.4%'
        A: Bias de liquidez y probabilidades contextuales
        """
        
        high_zone_time = panorama_data.get('high_zone_analysis', {}).get('time_percentage', 0)
        low_zone_time = panorama_data.get('low_zone_analysis', {}).get('time_percentage', 0)
        range_pct = panorama_data.get('percentage_range', 0)
        
        if high_zone_time == 0:
            high_zone_time = 0.1  # Evitar división por cero
        
        asymmetry_ratio = low_zone_time / high_zone_time
        
        if asymmetry_ratio > 10:  # Fuerte asimetría bajista
            interpretation = {
                'liquidity_profile': 'heavily_skewed_downside',
                'asymmetry_factor': round(asymmetry_ratio, 1),
                'downside_magnetism': min(90, round(50 + asymmetry_ratio * 1.5)),
                'context_bias': 'strong_bearish',
                'high_zone_time_pct': high_zone_time,
                'low_zone_time_pct': low_zone_time,
                'range_48h_pct': range_pct,
                'probability_implications': {
                    'downside_moves_favored': min(90, round(50 + asymmetry_ratio * 1.5)),
                    'upside_resistance_strong': min(85, round(60 + asymmetry_ratio * 1.2))
                },
                'interpretation': f'Mercado pasa {asymmetry_ratio:.1f}x más tiempo en zona baja. Fuerte imán bajista'
            }
        elif asymmetry_ratio < 0.5:  # Asimetría alcista
            interpretation = {
                'liquidity_profile': 'skewed_upside',
                'asymmetry_factor': round(asymmetry_ratio, 1),
                'upside_magnetism': 75,
                'context_bias': 'bullish',
                'high_zone_time_pct': high_zone_time,
                'low_zone_time_pct': low_zone_time,
                'range_48h_pct': range_pct,
                'probability_implications': {
                    'upside_moves_favored': 75,
                    'downside_support_strong': 70
                },
                'interpretation': f'Mercado favorece zona alta. Sesgo alcista contextual'
            }
        else:  # Distribución balanceada
            interpretation = {
                'liquidity_profile': 'balanced_distribution',
                'asymmetry_factor': round(asymmetry_ratio, 1),
                'context_bias': 'neutral',
                'high_zone_time_pct': high_zone_time,
                'low_zone_time_pct': low_zone_time,
                'range_48h_pct': range_pct,
                'probability_implications': {
                    'directional_bias': 'minimal',
                    'range_trading_favored': 70
                },
                'interpretation': f'Distribución balanceada entre zonas. Contexto neutral'
            }
        
        print(f"🌊 Contexto: {interpretation['interpretation']}")
        return interpretation
    
    def _interpret_structural_bias(self, weekly_data: Dict) -> Dict:
        """
        Interpretar bias estructural (Pregunta 6)
        De: '83.3% del rango + extremos decrecientes'
        A: Bias estructural y probabilidades de reversión
        """
        
        position_pct = weekly_data.get('range_position_pct', 50)
        week_range_pct = weekly_data.get('week_range_pct', 0)
        
        # Buscar información de extremos
        extremes_info = ""
        maximos_trend = weekly_data.get('maximos_trend', {})
        minimos_trend = weekly_data.get('minimos_trend', {})
        
        if isinstance(maximos_trend, dict) and isinstance(minimos_trend, dict):
            max_trend = maximos_trend.get('trend', 'unknown')
            min_trend = minimos_trend.get('trend', 'unknown')
            extremes_info = f"máximos {max_trend}, mínimos {min_trend}"
        
        if position_pct > 80:  # Zona alta
            if 'decrecientes' in extremes_info or 'decreasing' in extremes_info:
                interpretation = {
                    'structural_classification': 'high_zone_distribution',
                    'mean_reversion_pressure': 85,
                    'distribution_evidence': 'extremes_both_declining',
                    'position_percentile': position_pct,
                    'weekly_range_pct': week_range_pct,
                    'extremes_analysis': extremes_info,
                    'target_zones': {
                        'primary': {'range_pct': '45-55%', 'probability': 68},
                        'extended': {'range_pct': '35-45%', 'probability': 25}
                    },
                    'resistance_ceiling_pct': 90,
                    'interpretation': f'Posición {position_pct:.1f}% con extremos decrecientes = distribución activa. Reversión probable'
                }
            else:
                interpretation = {
                    'structural_classification': 'high_zone_resistance',
                    'mean_reversion_pressure': 65,
                    'position_percentile': position_pct,
                    'weekly_range_pct': week_range_pct,
                    'target_zones': {
                        'primary': {'range_pct': '60-75%', 'probability': 55}
                    },
                    'interpretation': f'Zona alta {position_pct:.1f}% con resistencia. Corrección moderada esperada'
                }
        elif position_pct < 20:  # Zona baja
            interpretation = {
                'structural_classification': 'low_zone_accumulation',
                'mean_reversion_pressure': 75,
                'position_percentile': position_pct,
                'weekly_range_pct': week_range_pct,
                'target_zones': {
                    'primary': {'range_pct': '40-60%', 'probability': 70}
                },
                'interpretation': f'Zona baja {position_pct:.1f}%. Rebote hacia zona media probable'
            }
        else:  # Zona media
            interpretation = {
                'structural_classification': 'middle_zone_neutral',
                'mean_reversion_pressure': 35,
                'position_percentile': position_pct,
                'weekly_range_pct': week_range_pct,
                'interpretation': f'Zona media {position_pct:.1f}%. Bias estructural mínimo'
            }
        
        print(f"🏗️ Estructura: {interpretation['interpretation']}")
        return interpretation
    
    def _analyze_price_progression(self, prices: List[float]) -> str:
        """Analizar si los precios son ascendentes, descendentes o estables"""
        if len(prices) < 2:
            return 'insufficient_data'
        
        # Calcular diferencias
        diffs = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
        
        # Determinar tendencia
        positive_diffs = sum(1 for d in diffs if d > 0)
        negative_diffs = sum(1 for d in diffs if d < 0)
        
        if positive_diffs > negative_diffs:
            return 'ascending'
        elif negative_diffs > positive_diffs:
            return 'descending'
        else:
            return 'stable'
    
    def _create_scenarios(self, interpretations: Dict) -> List[Dict]:
        """Sintetizar interpretaciones en escenarios probabilísticos claros"""
        
        print(f"\n🔮 Generando escenarios probabilísticos...")
        
        # Contar señales bajistas y alcistas
        bearish_signals = self._count_bearish_signals(interpretations)
        bullish_signals = self._count_bullish_signals(interpretations)
        neutral_signals = 6 - bearish_signals - bullish_signals
        
        print(f"📊 Análisis de señales: {bearish_signals} bajistas, {bullish_signals} alcistas, {neutral_signals} neutrales")
        
        scenarios = []
        
        # ESCENARIO PRIMARIO - Basado en mayoría de señales
        if bearish_signals >= 4:
            primary_scenario = {
                'name': 'Corrección Bajista Estructural',
                'probability': min(85, 60 + bearish_signals * 5),
                'target_range': self._calculate_bearish_targets(interpretations),
                'timeframe': '18-48h',
                'risk_reward': '3.4:1',
                'entry_strategy': 'Venta en rebotes hacia resistencia',
                'stop_loss_level': self._calculate_stop_loss(interpretations, 'bearish'),
                'reasoning': self._get_bearish_reasoning(interpretations),
                'confidence_factors': bearish_signals
            }
            scenarios.append(primary_scenario)
            
        elif bullish_signals >= 4:
            primary_scenario = {
                'name': 'Impulso Alcista Confirmado',
                'probability': min(80, 55 + bullish_signals * 5),
                'target_range': self._calculate_bullish_targets(interpretations),
                'timeframe': '12-36h',
                'risk_reward': '2.8:1',
                'entry_strategy': 'Compra en retrocesos hacia soporte',
                'stop_loss_level': self._calculate_stop_loss(interpretations, 'bullish'),
                'reasoning': self._get_bullish_reasoning(interpretations),
                'confidence_factors': bullish_signals
            }
            scenarios.append(primary_scenario)
            
        else:
            primary_scenario = {
                'name': 'Consolidación Lateral Dominante',
                'probability': 65,
                'target_range': self._calculate_lateral_range(interpretations),
                'timeframe': '24-72h',
                'risk_reward': '1.8:1',
                'entry_strategy': 'Trading de rango',
                'reasoning': ['Señales mixtas', 'Indecisión del mercado', 'Ausencia de consenso direccional'],
                'confidence_factors': neutral_signals
            }
            scenarios.append(primary_scenario)
        
        # ESCENARIOS SECUNDARIOS
        if bearish_signals >= 4:
            # Escenario lateral
            scenarios.append({
                'name': 'Consolidación Lateral Extendida',
                'probability': max(10, 25 - bearish_signals * 2),
                'target_range': self._calculate_lateral_range(interpretations),
                'timeframe': '12-24h',
                'reasoning': ['Soporte inesperadamente fuerte']
            })
            
            # Escenario alcista (baja probabilidad)
            scenarios.append({
                'name': 'Reversión Alcista con Volumen',
                'probability': max(5, 15 - bearish_signals * 2),
                'target_range': self._calculate_bullish_targets(interpretations),
                'timeframe': '6-12h',
                'catalyst_needed': 'Volumen excepcional + catalizador externo',
                'reasoning': ['Posible liquidación de cortos', 'Oversold bounce']
            })
            
        elif bullish_signals >= 4:
            # Escenario bajista
            scenarios.append({
                'name': 'Corrección Técnica Temporal',
                'probability': max(10, 25 - bullish_signals * 2),
                'target_range': self._calculate_bearish_targets(interpretations),
                'timeframe': '8-16h',
                'reasoning': ['Resistencia inesperadamente fuerte']
            })
            
            # Escenario lateral
            scenarios.append({
                'name': 'Pausa Antes de Continuación',
                'probability': max(10, 20 - bullish_signals),
                'target_range': self._calculate_lateral_range(interpretations),
                'timeframe': '6-12h',
                'reasoning': ['Consolidación antes de impulso']
            })
        
        # Asegurar que las probabilidades sumen ~100%
        total_prob = sum(s.get('probability', 0) for s in scenarios)
        if total_prob > 100:
            factor = 100 / total_prob
            for scenario in scenarios:
                scenario['probability'] = round(scenario['probability'] * factor, 1)
        
        return scenarios[:3]  # Máximo 3 escenarios
    
    def _count_bearish_signals(self, interpretations: Dict) -> int:
        """Contar señales bajistas"""
        count = 0
        
        # Rango compacto = mayor probabilidad de movimento (neutral)
        range_signal = interpretations.get('range_signal', {})
        if range_signal.get('breakout_probability', 0) > 70:
            # No cuenta como bajista por sí solo
            pass
        
        # Tendencia bajista
        trend_signal = interpretations.get('trend_signal', {})
        probs = trend_signal.get('probabilities', {})
        if probs.get('bearish_breakdown', 0) > probs.get('bullish_breakout', 0):
            count += 1
        
        # Soporte débil
        support_signal = interpretations.get('support_signal', {})
        if support_signal.get('support_type') == 'descending_weak':
            count += 1
        elif support_signal.get('hold_probability', 50) < 60:
            count += 0.5
        
        # Resistencia débil (bajista)
        resistance_signal = interpretations.get('resistance_signal', {})
        if resistance_signal.get('breakdown_probability', 0) > 60:
            count += 1
        
        # Contexto bajista
        context_signal = interpretations.get('context_signal', {})
        if context_signal.get('context_bias') == 'strong_bearish':
            count += 1
        elif context_signal.get('downside_magnetism', 0) > 70:
            count += 0.5
        
        # Estructura bajista
        structure_signal = interpretations.get('structure_signal', {})
        if structure_signal.get('structural_classification') == 'high_zone_distribution':
            count += 1
        elif structure_signal.get('mean_reversion_pressure', 0) > 75:
            count += 0.5
        
        return int(count)
    
    def _count_bullish_signals(self, interpretations: Dict) -> int:
        """Contar señales alcistas"""
        count = 0
        
        # Tendencia alcista
        trend_signal = interpretations.get('trend_signal', {})
        probs = trend_signal.get('probabilities', {})
        if probs.get('bullish_breakout', 0) > probs.get('bearish_breakdown', 0):
            count += 1
        
        # Soporte fuerte
        support_signal = interpretations.get('support_signal', {})
        if support_signal.get('support_type') == 'ascending_dynamic':
            count += 1
        elif support_signal.get('hold_probability', 50) > 75:
            count += 0.5
        
        # Resistencia fuerte (alcista si rompe)
        resistance_signal = interpretations.get('resistance_signal', {})
        if resistance_signal.get('resistance_pattern') == 'ascending_strengthening':
            count += 0.5  # Neutral a alcista
        
        # Contexto alcista
        context_signal = interpretations.get('context_signal', {})
        if context_signal.get('context_bias') == 'bullish':
            count += 1
        elif context_signal.get('upside_magnetism', 0) > 70:
            count += 0.5
        
        # Estructura alcista (zona baja)
        structure_signal = interpretations.get('structure_signal', {})
        if structure_signal.get('structural_classification') == 'low_zone_accumulation':
            count += 1
        elif structure_signal.get('position_percentile', 50) < 30:
            count += 0.5
        
        return int(count)
    
    def _calculate_bearish_targets(self, interpretations: Dict) -> str:
        """Calcular targets bajistas basados en interpretaciones"""
        
        # Obtener nivel actual estimado (usar máximo de resistencia como proxy)
        current_level = 4490  # Default, se puede mejorar
        resistance_signal = interpretations.get('resistance_signal', {})
        if resistance_signal.get('ceiling_level'):
            current_level = resistance_signal['ceiling_level']
        
        # Target primario: -2% a -4%
        primary_target_low = round(current_level * 0.96, 0)
        primary_target_high = round(current_level * 0.98, 0)
        
        # Target extendido: -5% a -8%
        extended_target_low = round(current_level * 0.92, 0)
        extended_target_high = round(current_level * 0.95, 0)
        
        return f"${int(extended_target_low)}-{int(primary_target_high)}"
    
    def _calculate_bullish_targets(self, interpretations: Dict) -> str:
        """Calcular targets alcistas basados en interpretaciones"""
        
        current_level = 4490  # Default
        support_signal = interpretations.get('support_signal', {})
        if support_signal.get('dynamic_level'):
            current_level = support_signal['dynamic_level']
        
        # Target primario: +2% a +4%
        primary_target_low = round(current_level * 1.02, 0)
        primary_target_high = round(current_level * 1.04, 0)
        
        # Target extendido: +5% a +7%
        extended_target_high = round(current_level * 1.07, 0)
        
        return f"${int(primary_target_low)}-{int(extended_target_high)}"
    
    def _calculate_lateral_range(self, interpretations: Dict) -> str:
        """Calcular rango lateral basado en interpretaciones"""
        
        current_level = 4490  # Default
        
        # Usar soporte y resistencia para definir rango
        support_signal = interpretations.get('support_signal', {})
        resistance_signal = interpretations.get('resistance_signal', {})
        
        support_level = support_signal.get('dynamic_level', current_level * 0.985)
        resistance_level = resistance_signal.get('ceiling_level', current_level * 1.015)
        
        return f"${int(support_level)}-{int(resistance_level)}"
    
    def _calculate_stop_loss(self, interpretations: Dict, direction: str) -> str:
        """Calcular nivel de stop loss"""
        
        current_level = 4490  # Default
        
        if direction == 'bearish':
            resistance_signal = interpretations.get('resistance_signal', {})
            stop_level = resistance_signal.get('ceiling_level', current_level * 1.012)
            return f"${int(stop_level)}"
        else:
            support_signal = interpretations.get('support_signal', {})
            stop_level = support_signal.get('dynamic_level', current_level * 0.988)
            return f"${int(stop_level)}"
    
    def _get_bearish_reasoning(self, interpretations: Dict) -> List[str]:
        """Obtener reasoning para escenario bajista"""
        reasons = []
        
        structure_signal = interpretations.get('structure_signal', {})
        if structure_signal.get('position_percentile', 50) > 80:
            reasons.append(f"Zona alta del rango ({structure_signal.get('position_percentile', 0):.1f}%)")
        
        resistance_signal = interpretations.get('resistance_signal', {})
        if resistance_signal.get('breakdown_probability', 0) > 60:
            reasons.append("Resistencia debilitándose progresivamente")
        
        context_signal = interpretations.get('context_signal', {})
        if context_signal.get('asymmetry_factor', 1) > 5:
            reasons.append(f"Bias de liquidez bajista ({context_signal.get('asymmetry_factor', 0):.1f}x)")
        
        if structure_signal.get('distribution_evidence'):
            reasons.append("Estructura de extremos decrecientes")
        
        return reasons[:4]  # Máximo 4 razones
    
    def _get_bullish_reasoning(self, interpretations: Dict) -> List[str]:
        """Obtener reasoning para escenario alcista"""
        reasons = []
        
        support_signal = interpretations.get('support_signal', {})
        if support_signal.get('support_type') == 'ascending_dynamic':
            reasons.append("Soporte ascendente robusto")
        
        structure_signal = interpretations.get('structure_signal', {})
        if structure_signal.get('position_percentile', 50) < 30:
            reasons.append(f"Zona baja del rango ({structure_signal.get('position_percentile', 0):.1f}%)")
        
        trend_signal = interpretations.get('trend_signal', {})
        probs = trend_signal.get('probabilities', {})
        if probs.get('bullish_breakout', 0) > 50:
            reasons.append("Sesgo alcista en análisis de tendencia")
        
        range_signal = interpretations.get('range_signal', {})
        if range_signal.get('coiling_strength') == 'high':
            reasons.append("Coiling fuerte detectado")
        
        return reasons[:4]
    
    def _generate_external_signal(self, scenarios: List[Dict]) -> Dict:
        """Generar señal simple para códigos externos"""
        
        if not scenarios:
            return {
                'action': 'MONITOR',
                'confidence': 0,
                'reasoning': 'Sin escenarios generados'
            }
        
        primary = scenarios[0]
        
        # Determinar acción
        if primary['probability'] > 70:
            if 'bajista' in primary['name'].lower() or 'bearish' in primary['name'].lower():
                action = 'SELL'
            elif 'alcista' in primary['name'].lower() or 'bullish' in primary['name'].lower():
                action = 'BUY'
            else:
                action = 'MONITOR'
        elif primary['probability'] > 50:
            action = 'MONITOR_CLOSE'
        else:
            action = 'MONITOR'
        
        return {
            'symbol': self.symbol,
            'action': action,
            'confidence': primary['probability'],
            'primary_scenario': primary['name'],
            'target_range': primary['target_range'],
            'timeframe': primary['timeframe'],
            'reasoning': primary.get('reasoning', [])[:3],  # Top 3 razones
            'timestamp': self.timestamp.isoformat()
        }
    
    def _save_logs(self, result: Dict):
        """Guardar logs de interpretación probabilística"""
        
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Log detallado
        detailed_file = os.path.join(self.logs_dir, f"interpretation_{self.symbol}_{timestamp_str}.json")
        with open(detailed_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Señal para externos
        signal_file = os.path.join(self.logs_dir, f"signal_external_{self.symbol}.json")
        with open(signal_file, 'w') as f:
            json.dump(result['signal_for_external'], f, indent=2)
        
        # Log simple para revisión
        simple_file = os.path.join(self.logs_dir, f"scenarios_{timestamp_str}.log")
        with open(simple_file, 'w') as f:
            f.write(f"INTERPRETACIÓN PROBABILÍSTICA - {self.symbol}\n")
            f.write(f"Timestamp: {result['timestamp']}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, scenario in enumerate(result['scenarios'], 1):
                f.write(f"ESCENARIO {i}: {scenario['name']}\n")
                f.write(f"Probabilidad: {scenario['probability']}%\n")
                f.write(f"Target: {scenario['target_range']}\n")
                f.write(f"Timeframe: {scenario['timeframe']}\n")
                if scenario.get('reasoning'):
                    f.write("Reasoning:\n")
                    for reason in scenario['reasoning']:
                        f.write(f"  - {reason}\n")
                f.write("\n" + "-" * 30 + "\n\n")
        
        print(f"💾 Logs guardados en {self.logs_dir}/")
        print(f"   📊 Detallado: interpretation_{self.symbol}_{timestamp_str}.json")
        print(f"   🚨 Señal externa: signal_external_{self.symbol}.json")
        print(f"   📝 Escenarios: scenarios_{timestamp_str}.log")