#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Trading Adaptativo con ML Enhanced
Integra análisis adaptativo + triángulos + volatilidad en vivo + ML
Autor: biGGzeta
Fecha: 2025-10-04
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, List, Any, Tuple, Optional

# Suprimir warnings
warnings.filterwarnings('ignore')

# Agregar path para imports locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports locales
from api.binance_data import BinanceDataProvider
from analysis.adaptive_analyzer import AdaptiveMarketAnalyzer
from analysis.triangle_enhanced_analyzer import TriangleEnhancedAnalyzer
from analysis.live_volatility_monitor import LiveVolatilityMonitor
from ml.feature_extractor import FeatureExtractor
from ml.decision_tree import AdaptiveDecisionTree
from utils.logger import setup_logger
from utils.file_manager import FileManager
from reports.session_reporter import SessionReporter

class EnhancedAdaptiveTradingMLSystem:
    """Sistema de Trading Adaptativo con ML y Detección de Triángulos"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP", enable_ml: bool = True, 
                 enable_deep_logging: bool = True, enable_triangles: bool = True,
                 enable_live_volatility: bool = True):
        self.symbol = symbol
        self.original_symbol = symbol
        self.enable_ml = enable_ml
        self.enable_deep_logging = enable_deep_logging
        self.enable_triangles = enable_triangles
        self.enable_live_volatility = enable_live_volatility
        
        # Configuración de usuario
        self.user = "biGGzeta"
        
        # Componentes principales
        self.data_provider = BinanceDataProvider()
        self.adaptive_analyzer = AdaptiveMarketAnalyzer(self.data_provider)
        
        # Componentes enhanced
        if self.enable_triangles:
            self.triangle_analyzer = TriangleEnhancedAnalyzer(self.data_provider)
        
        if self.enable_live_volatility:
            self.volatility_monitor = LiveVolatilityMonitor(self.data_provider)
        
        # ML Components
        if self.enable_ml:
            self.feature_extractor = FeatureExtractor()
            self.decision_tree = AdaptiveDecisionTree()
        
        # Logging y reporting
        if self.enable_deep_logging:
            self.logger = setup_logger("adaptive_analysis")
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_reporter = SessionReporter(
                user=self.user,
                symbol=self.symbol,
                session_id=session_id
            )
        
        # File manager
        self.file_manager = FileManager()
        
        # Cache de datos
        self.cached_data = {}
        self.analysis_results = {}
        
        print(f"🚀 Iniciando test completo del sistema - Usuario: {self.user}")
        print(f"🕐 Fecha actual: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"🔗 Usando API: {self.data_provider.base_url}")
        print(f"📊 Símbolo original: {self.original_symbol}")
        print(f"📊 Símbolo API: {self.symbol}")
        
        if self.enable_ml:
            print(f"🌳 Construyendo árbol de decisiones adaptativo...")
            self.decision_tree.build_tree()
            print(f"✅ Árbol de decisiones construido")
        
        if self.enable_deep_logging:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"📝 Iniciando logging de análisis profundo - Sesión: {session_id}")
        
        print(f"🎯 Sistema inicializado para {self.symbol}")
        print(f"   🤖 ML: {'✅ Habilitado' if self.enable_ml else '❌ Deshabilitado'}")
        print(f"   🔺 Triángulos: {'✅ Habilitado' if self.enable_triangles else '❌ Deshabilitado'}")
        print(f"   ⚡ Volatilidad Live: {'✅ Habilitado' if self.enable_live_volatility else '❌ Deshabilitado'}")
        print(f"   📝 Logging: {'✅ Habilitado' if self.enable_deep_logging else '❌ Deshabilitado'}")

    async def run_complete_analysis(self) -> Dict[str, Any]:
        """Ejecuta análisis completo con todas las capacidades enhanced"""
        
        print(f"\n🚀 Iniciando Sistema Completo Enhanced: Análisis Adaptativo + Triángulos + ML + Volatilidad Live")
        print("=" * 100)
        print(f"📊 Símbolo: {self.symbol}")
        print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"👤 Usuario: {self.user}")
        print(f"🤖 ML habilitado: {self.enable_ml}")
        print(f"🔺 Triángulos habilitados: {self.enable_triangles}")
        print(f"⚡ Volatilidad live: {self.enable_live_volatility}")
        print(f"📝 Logging profundo: {self.enable_deep_logging}")
        print("=" * 100)
        
        try:
            # FASE 1: Análisis adaptativo base
            print(f"\n🔧 FASE 1: Ejecutando análisis adaptativo base...")
            adaptive_results = await self._run_adaptive_analysis()
            
            # FASE 2: Análisis de triángulos (si está habilitado)
            triangle_results = {}
            if self.enable_triangles:
                print(f"\n🔺 FASE 2: Ejecutando análisis de triángulos...")
                triangle_results = await self._run_triangle_analysis()
            
            # FASE 3: Monitor de volatilidad en vivo (si está habilitado)
            volatility_results = {}
            if self.enable_live_volatility:
                print(f"\n⚡ FASE 3: Monitoreando volatilidad en vivo...")
                volatility_results = await self._run_live_volatility_analysis()
            
            # FASE 4: Pipeline ML enhanced
            ml_results = {}
            if self.enable_ml:
                print(f"\n🤖 FASE 4: Ejecutando pipeline ML enhanced...")
                ml_results = await self._run_enhanced_ml_pipeline(
                    adaptive_results, triangle_results, volatility_results
                )
            
            # FASE 5: Integración de resultados
            print(f"\n📊 FASE 5: Integrando todos los análisis...")
            integrated_results = await self._integrate_all_results(
                adaptive_results, triangle_results, volatility_results, ml_results
            )
            
            # FASE 6: Logging profundo
            if self.enable_deep_logging:
                print(f"\n📝 FASE 6: Guardando análisis profundo enhanced...")
                await self._save_enhanced_analysis(integrated_results)
            
            # FASE 7: Debug detallado
            if self.enable_ml:
                print(f"\n🔍 FASE 7: Debug detallado del árbol enhanced...")
                await self._debug_enhanced_tree()
            
            # FASE 8: Reporte final
            print(f"\n📋 FASE 8: Generando reporte final enhanced...")
            await self._generate_enhanced_report(integrated_results)
            
            return integrated_results
            
        except Exception as e:
            print(f"❌ Error en análisis completo enhanced: {e}")
            if self.enable_deep_logging:
                self.logger.error(f"Error en análisis completo: {e}")
            return {}

    async def _run_adaptive_analysis(self) -> Dict[str, Any]:
        """Ejecuta el análisis adaptativo base"""
        
        questions = [
            ("1", "¿Cuáles son los máximos y mínimos de las últimas 3 horas?", "basic_extremes", "3h"),
            ("2", "¿Cuál es el análisis adaptativo de porcentajes de las últimas 3 horas?", "adaptive_percentages", "3h"),
            ("3", "¿Cuáles son los mínimos adaptativos de las últimas 3 horas?", "adaptive_minimos", "3h"),
            ("4", "¿Cuáles son los máximos adaptativos de las últimas 3 horas?", "adaptive_maximos", "3h"),
            ("5", "¿Cuál es el panorama adaptativo de las últimas 48 horas?", "adaptive_panorama", "48h"),
            ("6", "¿Cuál es el análisis semanal adaptativo completo?", "weekly_adaptive", "7d")
        ]
        
        results = {}
        
        for q_num, question, analysis_type, timeframe in questions:
            print(f"\n📊 PREGUNTA {q_num}: {analysis_type.replace('_', ' ').title()}")
            
            try:
                if analysis_type == "basic_extremes":
                    answer = await self.adaptive_analyzer.get_basic_extremes_analysis()
                elif analysis_type == "adaptive_percentages":
                    answer = await self.adaptive_analyzer.get_adaptive_percentages_analysis()
                elif analysis_type == "adaptive_minimos":
                    answer = await self.adaptive_analyzer.get_adaptive_minimos_analysis()
                elif analysis_type == "adaptive_maximos":
                    answer = await self.adaptive_analyzer.get_adaptive_maximos_analysis()
                elif analysis_type == "adaptive_panorama":
                    answer = await self.adaptive_analyzer.get_adaptive_panorama_analysis()
                elif analysis_type == "weekly_adaptive":
                    answer = await self.adaptive_analyzer.get_weekly_adaptive_analysis()
                
                # Log detallado
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] {self.symbol}")
                print(f"Q: {question}")
                print(f"A: {answer}")
                print("-" * 40)
                
                results[analysis_type] = {
                    'question': question,
                    'answer': answer,
                    'timeframe': timeframe,
                    'timestamp': timestamp
                }
                
                print(f"✅ Respuesta {q_num}: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                
            except Exception as e:
                print(f"❌ Error en pregunta {q_num}: {e}")
                results[analysis_type] = {'error': str(e)}
        
        # Extraer información clave para el análisis enhanced
        results['extremos_trend'] = self._extract_extremos_trend(results)
        results['market_regime'] = self._extract_market_regime(results)
        results['momentum_data'] = self._extract_momentum_data(results)
        
        return results

    async def _run_triangle_analysis(self) -> Dict[str, Any]:
        """Ejecuta análisis de triángulos enhanced"""
        
        try:
            # Obtener datos para análisis de triángulos
            data_3h = await self.data_provider.get_klines(self.symbol, "1m", 180)
            data_6h = await self.data_provider.get_klines(self.symbol, "1m", 360)
            
            # Análisis de triángulos en múltiples timeframes
            triangles_3h = await self.triangle_analyzer.detect_triangles(data_3h, timeframe="3h")
            triangles_6h = await self.triangle_analyzer.detect_triangles(data_6h, timeframe="6h")
            
            # Análisis de convergencia
            convergence_analysis = await self.triangle_analyzer.analyze_convergence(data_3h)
            
            # Predicción de breakout
            breakout_prediction = await self.triangle_analyzer.predict_breakout_timing(data_3h)
            
            # Targets geométricos
            geometric_targets = await self.triangle_analyzer.calculate_geometric_targets(triangles_3h)
            
            print(f"🔺 Triángulos detectados (3h): {len(triangles_3h)}")
            print(f"🔺 Triángulos detectados (6h): {len(triangles_6h)}")
            print(f"⚡ Convergencia detectada: {convergence_analysis.get('is_converging', False)}")
            print(f"🎯 Probabilidad de breakout: {breakout_prediction.get('probability', 0):.1f}%")
            
            return {
                'triangles_3h': triangles_3h,
                'triangles_6h': triangles_6h,
                'convergence': convergence_analysis,
                'breakout_prediction': breakout_prediction,
                'geometric_targets': geometric_targets,
                'summary': {
                    'total_triangles': len(triangles_3h) + len(triangles_6h),
                    'is_converging': convergence_analysis.get('is_converging', False),
                    'breakout_probability': breakout_prediction.get('probability', 0),
                    'primary_pattern': triangles_3h[0]['type'] if triangles_3h else None
                }
            }
            
        except Exception as e:
            print(f"❌ Error en análisis de triángulos: {e}")
            return {'error': str(e)}

    async def _run_live_volatility_analysis(self) -> Dict[str, Any]:
        """Ejecuta análisis de volatilidad en vivo"""
        
        try:
            # Monitor de volatilidad instantánea
            live_volatility = await self.volatility_monitor.get_live_volatility(self.symbol)
            
            # Tendencias de volatilidad
            volatility_trend = await self.volatility_monitor.analyze_volatility_trend(self.symbol)
            
            # Detección de expansión/contracción
            expansion_analysis = await self.volatility_monitor.detect_volatility_expansion(self.symbol)
            
            # Alertas de volatilidad
            volatility_alerts = await self.volatility_monitor.check_volatility_alerts(self.symbol)
            
            print(f"⚡ Volatilidad actual: {live_volatility.get('current', 0):.2f}%")
            print(f"📈 Tendencia vol: {volatility_trend.get('direction', 'neutral')}")
            print(f"💥 Expansión detectada: {expansion_analysis.get('is_expanding', False)}")
            print(f"🚨 Alertas activas: {len(volatility_alerts)}")
            
            return {
                'live_volatility': live_volatility,
                'volatility_trend': volatility_trend,
                'expansion_analysis': expansion_analysis,
                'volatility_alerts': volatility_alerts,
                'summary': {
                    'current_volatility': live_volatility.get('current', 0),
                    'trend_direction': volatility_trend.get('direction', 'neutral'),
                    'is_expanding': expansion_analysis.get('is_expanding', False),
                    'alert_count': len(volatility_alerts)
                }
            }
            
        except Exception as e:
            print(f"❌ Error en análisis de volatilidad live: {e}")
            return {'error': str(e)}

    async def _run_enhanced_ml_pipeline(self, adaptive_results: Dict, 
                                      triangle_results: Dict, 
                                      volatility_results: Dict) -> Dict[str, Any]:
        """Pipeline ML enhanced con todas las features"""
        
        try:
            print(f"🚀 Iniciando pipeline ML enhanced...")
            
            # Paso 1: Extraer features base
            print(f"📊 Paso 1: Extrayendo features base...")
            base_features = self.feature_extractor.extract_features(adaptive_results)
            
            # Paso 2: Extraer features de triángulos
            triangle_features = {}
            if self.enable_triangles and triangle_results:
                print(f"🔺 Paso 2: Extrayendo features de triángulos...")
                triangle_features = self._extract_triangle_features(triangle_results)
            
            # Paso 3: Extraer features de volatilidad
            volatility_features = {}
            if self.enable_live_volatility and volatility_results:
                print(f"⚡ Paso 3: Extrayendo features de volatilidad...")
                volatility_features = self._extract_volatility_features(volatility_results)
            
            # Paso 4: Combinar todas las features
            print(f"🔗 Paso 4: Combinando features enhanced...")
            enhanced_features = {**base_features, **triangle_features, **volatility_features}
            
            # Paso 5: Evaluar árbol enhanced
            print(f"🌳 Paso 5: Evaluando árbol enhanced...")
            enhanced_signal = self.decision_tree.evaluate_enhanced(enhanced_features)
            
            # Paso 6: Calcular confianza enhanced
            print(f"🎯 Paso 6: Calculando confianza enhanced...")
            enhanced_confidence = self._calculate_enhanced_confidence(
                enhanced_features, triangle_results, volatility_results
            )
            
            # Paso 7: Preparar datos ML enhanced
            print(f"💾 Paso 7: Preparando datos ML enhanced...")
            ml_data = {
                'features': enhanced_features,
                'signal': enhanced_signal,
                'confidence': enhanced_confidence,
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'user': self.user
            }
            
            # Guardar datos enhanced
            await self.file_manager.save_json(ml_data, "ml_data_enhanced", "ml_enhanced_data")
            print(f"✅ Pipeline ML enhanced completado: {enhanced_signal} (confianza: {enhanced_confidence:.0f}%)")
            
            return {
                'enhanced_features': enhanced_features,
                'enhanced_signal': enhanced_signal,
                'enhanced_confidence': enhanced_confidence,
                'ml_data': ml_data
            }
            
        except Exception as e:
            print(f"❌ Error en pipeline ML enhanced: {e}")
            return {'error': str(e)}

    async def _integrate_all_results(self, adaptive_results: Dict, 
                                   triangle_results: Dict,
                                   volatility_results: Dict, 
                                   ml_results: Dict) -> Dict[str, Any]:
        """Integra todos los resultados en un análisis unificado"""
        
        # Análisis integrado
        integrated_analysis = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'user': self.user,
            
            # Resultados base
            'adaptive_analysis': adaptive_results,
            'triangle_analysis': triangle_results,
            'volatility_analysis': volatility_results,
            'ml_analysis': ml_results,
            
            # Síntesis integrada
            'integrated_signal': self._synthesize_signal(adaptive_results, triangle_results, volatility_results, ml_results),
            'integrated_confidence': self._synthesize_confidence(adaptive_results, triangle_results, volatility_results, ml_results),
            'integrated_targets': self._synthesize_targets(adaptive_results, triangle_results, volatility_results),
            'integrated_timing': self._synthesize_timing(triangle_results, volatility_results),
            
            # Factores de decisión
            'decision_factors': self._analyze_decision_factors(adaptive_results, triangle_results, volatility_results, ml_results),
            
            # Recomendaciones finales
            'recommendations': self._generate_final_recommendations(adaptive_results, triangle_results, volatility_results, ml_results)
        }
        
        return integrated_analysis

    async def _save_enhanced_analysis(self, integrated_results: Dict[str, Any]):
        """Guarda análisis enhanced profundo"""
        
        try:
            # Preparar datos para logging
            analysis_data = {
                'session_info': {
                    'timestamp': integrated_results['timestamp'],
                    'symbol': self.symbol,
                    'user': self.user,
                    'enhanced_features': {
                        'triangles_enabled': self.enable_triangles,
                        'volatility_live_enabled': self.enable_live_volatility,
                        'ml_enabled': self.enable_ml
                    }
                },
                'results': integrated_results
            }
            
            # Guardar con file manager
            await self.file_manager.save_json(analysis_data, "enhanced_analysis", "enhanced_analysis_data")
            
            # Log con session reporter si está habilitado
            if self.enable_deep_logging:
                self.session_reporter.add_analysis_data(analysis_data)
            
            print(f"✅ Análisis enhanced guardado exitosamente")
            
        except Exception as e:
            print(f"⚠️ Error en logging enhanced: {e}")

    async def _debug_enhanced_tree(self):
        """Debug detallado del árbol enhanced"""
        
        try:
            print(f"\n🔍 DEBUG DETALLADO DEL ÁRBOL ENHANCED:")
            print("=" * 80)
            
            # Obtener features actuales
            current_features = getattr(self, '_current_features', {})
            
            if current_features:
                # Debug tradicional
                debug_info = self.decision_tree.debug_decision_path(current_features)
                
                for i, step in enumerate(debug_info, 1):
                    print(f"{i}️⃣ {step['description']}: {step['result']}")
                    print(f"   📊 {step['details']}")
                    if step.get('branch'):
                        print(f"   ➡️ Tomando rama: {step['branch']}")
                    print()
                
                # Debug enhanced específico
                if self.enable_triangles:
                    triangle_debug = self._debug_triangle_features(current_features)
                    print(f"🔺 FEATURES DE TRIÁNGULOS:")
                    for key, value in triangle_debug.items():
                        print(f"   {key}: {value}")
                    print()
                
                if self.enable_live_volatility:
                    volatility_debug = self._debug_volatility_features(current_features)
                    print(f"⚡ FEATURES DE VOLATILIDAD:")
                    for key, value in volatility_debug.items():
                        print(f"   {key}: {value}")
                    print()
            
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ Error en debug enhanced: {e}")

    async def _generate_enhanced_report(self, integrated_results: Dict[str, Any]):
        """Genera reporte final enhanced"""
        
        try:
            # Extraer información clave
            signal = integrated_results.get('integrated_signal', 'UNKNOWN')
            confidence = integrated_results.get('integrated_confidence', 0)
            
            # Generar resumen ejecutivo enhanced
            print(f"\n" + "=" * 100)
            print(f"🎯 RESUMEN EJECUTIVO ENHANCED:")
            print("=" * 100)
            
            # Información del mercado
            market_info = self._extract_market_summary(integrated_results)
            print(f"📊 SITUACIÓN: {market_info}")
            
            # Señal principal
            print(f"\n🚨 SEÑAL DE TRADING ENHANCED:")
            print(f"   Tipo: {signal}")
            print(f"   Confianza: {confidence:.0f}%")
            
            # Análisis por componentes
            if self.enable_triangles:
                triangle_summary = self._extract_triangle_summary(integrated_results)
                print(f"\n🔺 ANÁLISIS DE TRIÁNGULOS:")
                print(f"   {triangle_summary}")
            
            if self.enable_live_volatility:
                volatility_summary = self._extract_volatility_summary(integrated_results)
                print(f"\n⚡ ANÁLISIS DE VOLATILIDAD:")
                print(f"   {volatility_summary}")
            
            # Recomendaciones finales
            recommendations = integrated_results.get('recommendations', {})
            print(f"\n📊 RECOMENDACIONES ENHANCED:")
            for key, value in recommendations.items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
            
            # Factores de decisión
            factors = integrated_results.get('decision_factors', {})
            print(f"\n🔍 ANÁLISIS DE CONFIANZA ENHANCED:")
            positive_factors = factors.get('positive_factors', [])
            negative_factors = factors.get('negative_factors', [])
            
            if positive_factors:
                print(f"   ✅ Factores positivos: {len(positive_factors)}")
                for factor in positive_factors[:3]:  # Top 3
                    print(f"      • {factor}")
            
            if negative_factors:
                print(f"   ⚠️ Factores de riesgo: {len(negative_factors)}")
                for factor in negative_factors[:3]:  # Top 3
                    print(f"      • {factor}")
            
            # Guardar reporte de sesión enhanced
            if self.enable_deep_logging:
                report_path = self.session_reporter.generate_enhanced_report(integrated_results)
                print(f"📋 Reporte enhanced generado: {report_path}")
            
            print(f"\n✅ Test enhanced completado - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
        except Exception as e:
            print(f"❌ Error generando reporte enhanced: {e}")

    # Métodos auxiliares para procesamiento enhanced
    
    def _extract_triangle_features(self, triangle_results: Dict) -> Dict[str, float]:
        """Extrae features numéricas de triángulos"""
        features = {}
        
        try:
            summary = triangle_results.get('summary', {})
            
            features['triangle_count'] = summary.get('total_triangles', 0)
            features['is_converging'] = 1.0 if summary.get('is_converging', False) else 0.0
            features['breakout_probability'] = summary.get('breakout_probability', 0) / 100.0
            
            # Tipo de patrón principal
            primary_pattern = summary.get('primary_pattern', 'none')
            features['triangle_ascending'] = 1.0 if primary_pattern == 'ascending' else 0.0
            features['triangle_descending'] = 1.0 if primary_pattern == 'descending' else 0.0
            features['triangle_symmetric'] = 1.0 if primary_pattern == 'symmetric' else 0.0
            
        except Exception as e:
            print(f"⚠️ Error extrayendo features de triángulos: {e}")
        
        return features
    
    def _extract_volatility_features(self, volatility_results: Dict) -> Dict[str, float]:
        """Extrae features numéricas de volatilidad"""
        features = {}
        
        try:
            summary = volatility_results.get('summary', {})
            
            features['current_volatility'] = summary.get('current_volatility', 0) / 100.0
            features['volatility_expanding'] = 1.0 if summary.get('is_expanding', False) else 0.0
            features['volatility_alert_count'] = min(summary.get('alert_count', 0), 5)  # Cap a 5
            
            # Dirección de tendencia de volatilidad
            trend_direction = summary.get('trend_direction', 'neutral')
            features['volatility_trend_up'] = 1.0 if trend_direction == 'up' else 0.0
            features['volatility_trend_down'] = 1.0 if trend_direction == 'down' else 0.0
            
        except Exception as e:
            print(f"⚠️ Error extrayendo features de volatilidad: {e}")
        
        return features
    
    def _calculate_enhanced_confidence(self, features: Dict, triangle_results: Dict, volatility_results: Dict) -> float:
        """Calcula confianza enhanced basada en múltiples factores"""
        
        base_confidence = 50.0  # Confianza base
        
        try:
            # Factor de triángulos
            if triangle_results and not triangle_results.get('error'):
                triangle_factor = triangle_results.get('summary', {}).get('breakout_probability', 0) * 0.3
                base_confidence += triangle_factor
            
            # Factor de volatilidad
            if volatility_results and not volatility_results.get('error'):
                if volatility_results.get('summary', {}).get('is_expanding', False):
                    base_confidence += 15  # Expansión de volatilidad es bullish
            
            # Factor de momentum (desde features base)
            momentum_3d = features.get('momentum_3d', 0)
            if momentum_3d > 0.05:  # >5%
                base_confidence += 20
            elif momentum_3d > 0.03:  # >3%
                base_confidence += 10
            
            # Factor de extremos alineados
            if features.get('extremos_alineados', 0) == 1:
                base_confidence += 15
            
            # Cap a 95% máximo
            base_confidence = min(base_confidence, 95.0)
            
        except Exception as e:
            print(f"⚠️ Error calculando confianza enhanced: {e}")
        
        return base_confidence
    
    def _synthesize_signal(self, adaptive_results: Dict, triangle_results: Dict, 
                          volatility_results: Dict, ml_results: Dict) -> str:
        """Sintetiza señal final de todos los componentes"""
        
        signals = []
        
        # Señal ML base
        if ml_results and not ml_results.get('error'):
            ml_signal = ml_results.get('enhanced_signal', 'HOLD')
            signals.append(('ML', ml_signal))
        
        # Señal de triángulos
        if triangle_results and not triangle_results.get('error'):
            triangle_summary = triangle_results.get('summary', {})
            if triangle_summary.get('breakout_probability', 0) > 70:
                triangle_signal = 'BUY' if triangle_summary.get('primary_pattern') == 'ascending' else 'SELL'
                signals.append(('TRIANGLE', triangle_signal))
        
        # Señal de volatilidad
        if volatility_results and not volatility_results.get('error'):
            vol_summary = volatility_results.get('summary', {})
            if vol_summary.get('is_expanding', False):
                signals.append(('VOLATILITY', 'BUY'))  # Expansión suele ser alcista
        
        # Lógica de consenso
        buy_votes = sum(1 for _, signal in signals if signal == 'BUY')
        sell_votes = sum(1 for _, signal in signals if signal == 'SELL')
        
        if buy_votes > sell_votes:
            return 'BUY'
        elif sell_votes > buy_votes:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _synthesize_confidence(self, adaptive_results: Dict, triangle_results: Dict, 
                             volatility_results: Dict, ml_results: Dict) -> float:
        """Sintetiza confianza final"""
        
        if ml_results and not ml_results.get('error'):
            return ml_results.get('enhanced_confidence', 50.0)
        
        return 50.0  # Confianza por defecto
    
    def _synthesize_targets(self, adaptive_results: Dict, triangle_results: Dict, 
                           volatility_results: Dict) -> Dict[str, float]:
        """Sintetiza targets de precio"""
        
        targets = {}
        
        try:
            # Targets geométricos de triángulos
            if triangle_results and not triangle_results.get('error'):
                geometric_targets = triangle_results.get('geometric_targets', {})
                if geometric_targets:
                    targets.update(geometric_targets)
            
            # Targets adaptativos (de extremos)
            if adaptive_results:
                # Extraer de análisis semanal
                weekly = adaptive_results.get('weekly_adaptive', {})
                if weekly and not weekly.get('error'):
                    # Lógica para extraer targets de extremos adaptativos
                    pass
        
        except Exception as e:
            print(f"⚠️ Error sintetizando targets: {e}")
        
        return targets
    
    def _synthesize_timing(self, triangle_results: Dict, volatility_results: Dict) -> Dict[str, Any]:
        """Sintetiza timing óptimo de entrada"""
        
        timing = {'immediate': False, 'wait_for_breakout': False, 'wait_for_pullback': False}
        
        try:
            # Timing de triángulos
            if triangle_results and not triangle_results.get('error'):
                breakout_pred = triangle_results.get('breakout_prediction', {})
                if breakout_pred.get('probability', 0) > 80:
                    timing['wait_for_breakout'] = True
                elif breakout_pred.get('probability', 0) > 60:
                    timing['immediate'] = True
            
            # Timing de volatilidad
            if volatility_results and not volatility_results.get('error'):
                if volatility_results.get('summary', {}).get('is_expanding', False):
                    timing['immediate'] = True
        
        except Exception as e:
            print(f"⚠️ Error sintetizando timing: {e}")
        
        return timing
    
    # Métodos auxiliares adicionales (extractores de info)
    
    def _extract_extremos_trend(self, results: Dict) -> str:
        """Extrae tendencia de extremos del análisis adaptativo"""
        try:
            weekly = results.get('weekly_adaptive', {})
            if weekly and 'answer' in weekly:
                answer = weekly['answer']
                if 'máximos 3d: crecientes' in answer and 'mínimos 3d: crecientes' in answer:
                    return 'both_rising'
                elif 'máximos 3d: crecientes' in answer:
                    return 'maximos_rising'
                elif 'mínimos 3d: crecientes' in answer:
                    return 'minimos_rising'
        except Exception:
            pass
        return 'neutral'
    
    def _extract_market_regime(self, results: Dict) -> str:
        """Extrae régimen de mercado"""
        try:
            for result in results.values():
                if isinstance(result, dict) and 'answer' in result:
                    answer = result['answer']
                    if 'low_volatility' in answer:
                        return 'low_volatility'
                    elif 'high_volatility' in answer:
                        return 'high_volatility'
                    elif 'trending' in answer:
                        return 'trending'
        except Exception:
            pass
        return 'unknown'
    
    def _extract_momentum_data(self, results: Dict) -> Dict[str, float]:
        """Extrae datos de momentum"""
        momentum = {'1d': 0.0, '3d': 0.0, '7d': 0.0}
        
        try:
            weekly = results.get('weekly_adaptive', {})
            if weekly and 'answer' in weekly:
                answer = weekly['answer']
                # Buscar patrones de momentum
                import re
                momentum_pattern = r'\+(\d+\.?\d*)%'
                matches = re.findall(momentum_pattern, answer)
                if matches:
                    momentum['3d'] = float(matches[0])
        except Exception:
            pass
        
        return momentum
    
    def _analyze_decision_factors(self, adaptive_results: Dict, triangle_results: Dict, 
                                volatility_results: Dict, ml_results: Dict) -> Dict[str, List[str]]:
        """Analiza factores de decisión positivos y negativos"""
        
        positive_factors = []
        negative_factors = []
        
        # Factores adaptativos
        extremos_trend = self._extract_extremos_trend(adaptive_results)
        if extremos_trend == 'both_rising':
            positive_factors.append("Extremos alineados crecientes (estructura clara)")
        
        momentum_data = self._extract_momentum_data(adaptive_results)
        if momentum_data['3d'] > 5:
            positive_factors.append(f"Momentum fuerte ({momentum_data['3d']:.1f}%)")
        elif momentum_data['3d'] < -5:
            negative_factors.append(f"Momentum negativo ({momentum_data['3d']:.1f}%)")
        
        # Factores de triángulos
        if triangle_results and not triangle_results.get('error'):
            triangle_summary = triangle_results.get('summary', {})
            if triangle_summary.get('is_converging', False):
                positive_factors.append("Patrón de triángulo en convergencia")
            if triangle_summary.get('breakout_probability', 0) > 70:
                positive_factors.append(f"Alta probabilidad de breakout ({triangle_summary['breakout_probability']:.0f}%)")
        
        # Factores de volatilidad
        if volatility_results and not volatility_results.get('error'):
            vol_summary = volatility_results.get('summary', {})
            if vol_summary.get('is_expanding', False):
                positive_factors.append("Expansión de volatilidad detectada")
            if vol_summary.get('alert_count', 0) > 2:
                negative_factors.append("Múltiples alertas de volatilidad")
        
        return {
            'positive_factors': positive_factors,
            'negative_factors': negative_factors
        }
    
    def _generate_final_recommendations(self, adaptive_results: Dict, triangle_results: Dict, 
                                      volatility_results: Dict, ml_results: Dict) -> Dict[str, str]:
        """Genera recomendaciones finales basadas en todos los análisis"""
        
        recommendations = {}
        
        # Acción principal
        signal = self._synthesize_signal(adaptive_results, triangle_results, volatility_results, ml_results)
        confidence = self._synthesize_confidence(adaptive_results, triangle_results, volatility_results, ml_results)
        
        recommendations['Acción'] = signal
        recommendations['Confianza'] = f"{confidence:.0f}%"
        
        # Tamaño de posición
        if confidence > 80:
            recommendations['Posición'] = "100% del tamaño normal de posición"
        elif confidence > 60:
            recommendations['Posición'] = "75% del tamaño normal de posición"
        elif confidence > 40:
            recommendations['Posición'] = "50% del tamaño normal de posición"
        else:
            recommendations['Posición'] = "25% del tamaño normal de posición"
        
        # Timing
        timing = self._synthesize_timing(triangle_results, volatility_results)
        if timing.get('immediate', False):
            recommendations['Entrada'] = "Entrada inmediata"
        elif timing.get('wait_for_breakout', False):
            recommendations['Entrada'] = "Esperar confirmación de breakout"
        else:
            recommendations['Entrada'] = "Esperar confirmación adicional antes de entrar"
        
        return recommendations
    
    # Métodos de debug
    
    def _debug_triangle_features(self, features: Dict) -> Dict[str, Any]:
        """Debug específico de features de triángulos"""
        triangle_debug = {}
        
        for key, value in features.items():
            if 'triangle' in key.lower():
                triangle_debug[key] = value
        
        return triangle_debug
    
    def _debug_volatility_features(self, features: Dict) -> Dict[str, Any]:
        """Debug específico de features de volatilidad"""
        volatility_debug = {}
        
        for key, value in features.items():
            if 'volatility' in key.lower() or 'vol' in key.lower():
                volatility_debug[key] = value
        
        return volatility_debug
    
    # Métodos de extracción para reporte
    
    def _extract_market_summary(self, integrated_results: Dict) -> str:
        """Extrae resumen del mercado para el reporte final"""
        
        try:
            adaptive = integrated_results.get('adaptive_analysis', {})
            regime = self._extract_market_regime(adaptive)
            momentum = self._extract_momentum_data(adaptive)
            extremos = self._extract_extremos_trend(adaptive)
            
            # Extraer posición y volatilidad
            weekly = adaptive.get('weekly_adaptive', {})
            if weekly and 'answer' in weekly:
                answer = weekly['answer']
                # Extraer posición del rango
                import re
                pos_match = re.search(r'(\d+\.?\d*)% del rango', answer)
                position = pos_match.group(1) if pos_match else "??"
                
                # Extraer volatilidad
                vol_match = re.search(r'vol (\d+\.?\d*)%', answer)
                volatility = vol_match.group(1) if vol_match else "??"
                
                return f"{regime.upper()} (vol {volatility}%) | 📍 POSICIÓN: ZONA {'ALTA' if float(position) > 70 else 'BAJA' if float(position) < 30 else 'MEDIA'} ({position}% del rango semanal) | 🏗️ ESTRUCTURA: {extremos.replace('_', ' ').title()} | ⚡ MOMENTUM 3D: {'ALCISTA FUERTE' if momentum['3d'] > 5 else 'BAJISTA FUERTE' if momentum['3d'] < -5 else 'NEUTRAL'} ({momentum['3d']:+.1f}%)"
        
        except Exception as e:
            return f"ERROR EXTRAYENDO RESUMEN: {e}"
    
    def _extract_triangle_summary(self, integrated_results: Dict) -> str:
        """Extrae resumen de triángulos"""
        
        try:
            triangle_analysis = integrated_results.get('triangle_analysis', {})
            if triangle_analysis and not triangle_analysis.get('error'):
                summary = triangle_analysis.get('summary', {})
                total = summary.get('total_triangles', 0)
                converging = summary.get('is_converging', False)
                probability = summary.get('breakout_probability', 0)
                pattern = summary.get('primary_pattern', 'none')
                
                return f"{total} triángulos detectados | Convergencia: {'SÍ' if converging else 'NO'} | Patrón: {pattern.upper()} | Breakout: {probability:.0f}%"
            else:
                return "No disponible"
        except Exception as e:
            return f"Error: {e}"
    
    def _extract_volatility_summary(self, integrated_results: Dict) -> str:
        """Extrae resumen de volatilidad"""
        
        try:
            volatility_analysis = integrated_results.get('volatility_analysis', {})
            if volatility_analysis and not volatility_analysis.get('error'):
                summary = volatility_analysis.get('summary', {})
                current = summary.get('current_volatility', 0)
                expanding = summary.get('is_expanding', False)
                trend = summary.get('trend_direction', 'neutral')
                alerts = summary.get('alert_count', 0)
                
                return f"Actual: {current:.2f}% | Tendencia: {trend.upper()} | Expansión: {'SÍ' if expanding else 'NO'} | Alertas: {alerts}"
            else:
                return "No disponible"
        except Exception as e:
            return f"Error: {e}"


# Función main
async def main():
    """Función principal"""
    
    try:
        # Crear sistema enhanced
        system = EnhancedAdaptiveTradingMLSystem(
            symbol="ETHUSD_PERP",
            enable_ml=True,
            enable_deep_logging=True,
            enable_triangles=True,
            enable_live_volatility=True
        )
        
        # Ejecutar análisis completo enhanced
        results = await system.run_complete_analysis()
        
        return results
        
    except Exception as e:
        print(f"❌ Error en main enhanced: {e}")
        return None


# Punto de entrada
if __name__ == "__main__":
    # Ejecutar análisis
    results = asyncio.run(main())
    
    if results:
        print(f"\n🎉 Análisis enhanced completado exitosamente")
    else:
        print(f"\n❌ Error en análisis enhanced")