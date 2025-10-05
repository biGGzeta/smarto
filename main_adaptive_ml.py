#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Trading Adaptativo con ML + Interpretación Probabilística
Integra análisis adaptativo + ML + interpretación probabilística de señales
Autor: biGGzeta
Fecha: 2025-10-04
Actualizado: 2025-10-04 22:56:45 UTC
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

# Imports locales CORREGIDOS - Usando tus módulos existentes
from data.csv_handler import BinanceDataDownloader
from analysis.max_min_analyzer import MaxMinAnalyzer
from analysis.percentage_analyzer import PercentageAnalyzer
from analysis.low_time_analyzer import LowTimeAnalyzer
from analysis.high_time_analyzer import HighTimeAnalyzer
from analysis.panorama_analyzer import PanoramaAnalyzer
from analysis.weekly_analyzer import WeeklyAnalyzer
from ml.adaptive_ml_system import AdaptiveMLSystem
from utils.logger import QALogger
from config.settings import BinanceConfig

class AdaptiveTradingMLSystem:
    """Sistema completo: Análisis Adaptativo + ML + Interpretación Probabilística"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP", enable_ml: bool = True, enable_logging: bool = True):
        self.symbol = symbol
        self.enable_ml = enable_ml
        self.enable_logging = enable_logging
        
        # Inicializar componentes principales (usando tus módulos existentes)
        self.downloader = BinanceDataDownloader(symbol)
        self.max_min_analyzer = MaxMinAnalyzer(symbol)
        self.percentage_analyzer = PercentageAnalyzer(symbol)
        self.low_time_analyzer = LowTimeAnalyzer(symbol)
        self.high_time_analyzer = HighTimeAnalyzer(symbol)
        self.panorama_analyzer = PanoramaAnalyzer(symbol)
        self.weekly_analyzer = WeeklyAnalyzer(symbol)
        
        # Logger QA
        if enable_logging:
            self.logger = QALogger(symbol)
        
        # Componentes ML
        if enable_ml:
            self.ml_system = AdaptiveMLSystem(symbol)
        
        # Variables para reutilizar datos
        self.last_data = None
        self.last_48h_data = None
        self.last_weekly_data = None
        
        print(f"🎯 Sistema inicializado para {symbol}")
        if enable_ml:
            print(f"   🤖 ML: ✅ Habilitado")
        if enable_logging:
            print(f"   📝 Logging: ✅ Habilitado")
    
    def run_complete_analysis_with_ml(self) -> Dict[str, Any]:
        """Ejecutar análisis completo: Adaptativo + ML + Interpretación"""
        
        timestamp = datetime.utcnow()
        
        print(f"\n🚀 Iniciando Sistema Completo: Análisis Adaptativo + ML + Logging")
        print("=" * 75)
        print(f"📊 Símbolo: {self.symbol}")
        print(f"🕐 Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"👤 Usuario: biGGzeta")
        print(f"🤖 ML habilitado: {self.enable_ml}")
        print(f"📝 Logging profundo: {self.enable_logging}")
        print("=" * 75)
        
        try:
            # FASE 1: Análisis Adaptativo Completo (usando tu sistema existente)
            print(f"\n🔧 FASE 1: Ejecutando análisis adaptativo...")
            adaptive_results = self._run_adaptive_analysis()
            
            if not adaptive_results:
                return {"status": "error", "message": "Error en análisis adaptativo"}
            
            # FASE 2: Pipeline ML (si está habilitado)
            ml_results = {}
            if self.enable_ml:
                print(f"\n🤖 FASE 2: Ejecutando pipeline ML...")
                try:
                    ml_results = self.ml_system.analyze_and_signal(adaptive_results)
                    print(f"✅ Pipeline ML completado: {ml_results['signal'].signal_type.value} (confianza: {ml_results['signal'].confidence}%)")
                except Exception as e:
                    print(f"❌ Error en pipeline ML: {str(e)}")
                    return {"status": "error_ml", "ml_error": str(e), "adaptive_results": adaptive_results}
            
            # FASE 3: Análisis Integrado
            print(f"\n📊 FASE 3: Generando análisis integrado...")
            integrated_analysis = self._create_integrated_analysis(adaptive_results, ml_results)
            
            # FASE 4: Debug detallado del árbol de decisiones
            if self.enable_ml and ml_results:
                print(f"\n🔍 FASE 4: Debug detallado del árbol de decisiones...")
                self._debug_decision_tree(ml_results)
            
            # Resultado final
            result = {
                "status": "success",
                "timestamp": timestamp.isoformat(),
                "symbol": self.symbol,
                "adaptive_results": adaptive_results,
                "ml_results": ml_results,
                "integrated_analysis": integrated_analysis,
                "executive_summary": self._generate_executive_summary(adaptive_results, ml_results),
                "trading_signal": self._extract_trading_signal(ml_results),
                "trading_recommendations": self._generate_trading_recommendations(ml_results),
                "confidence_analysis": self._analyze_confidence_factors(adaptive_results, ml_results)
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error en análisis completo: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _run_adaptive_analysis(self) -> Dict[str, Any]:
        """Ejecutar las 6 preguntas adaptativas usando tu sistema existente"""
        
        results = {}
        
        try:
            # Pregunta 1: Máximos y mínimos básicos
            print(f"\n📊 PREGUNTA 1: Máximos y mínimos básicos")
            simple_answer1, detailed_data1 = self.ask_max_min_last_hours(3)
            results['max_min_basic'] = (simple_answer1, detailed_data1)
            
            # Pregunta 2: Análisis de porcentajes
            print(f"\n📊 PREGUNTA 2: Análisis de porcentajes")
            simple_answer2, detailed_data2 = self.ask_range_percentage(3)
            results['percentage_adaptive'] = (simple_answer2, detailed_data2)
            
            # Pregunta 3: Mínimos adaptativos
            print(f"\n📊 PREGUNTA 3: Mínimos adaptativos")
            simple_answer3, detailed_data3 = self.ask_low_time_minimums(3, 15)
            results['low_minimums_adaptive'] = (simple_answer3, detailed_data3)
            
            # Pregunta 4: Máximos adaptativos
            print(f"\n📊 PREGUNTA 4: Máximos adaptativos")
            simple_answer4, detailed_data4 = self.ask_high_time_maximums(3, 15)
            results['high_maximums_adaptive'] = (simple_answer4, detailed_data4)
            
            # Pregunta 5: Panorama 48h
            print(f"\n📊 PREGUNTA 5: Panorama 48h")
            simple_answer5, detailed_data5 = self.ask_48h_panorama()
            results['panorama_48h_adaptive'] = (simple_answer5, detailed_data5)
            
            # Pregunta 6: Análisis semanal
            print(f"\n📊 PREGUNTA 6: Análisis semanal")
            simple_answer6, detailed_data6 = self.ask_weekly_analysis()
            results['weekly_adaptive'] = (simple_answer6, detailed_data6)
            
            return results
            
        except Exception as e:
            print(f"❌ Error en análisis adaptativo: {str(e)}")
            return {}
    
    # MÉTODOS COPIADOS DE TU main.py ORIGINAL
    def ask_max_min_last_hours(self, hours: float):
        """Pregunta 1: ¿Cuáles son los máximos y mínimos de las últimas X horas?"""
        question = f"¿Cuáles son los máximos y mínimos de las últimas {hours} horas?"
        
        print(f"🔄 Descargando datos de {self.symbol} (1m) para las últimas {hours} horas...")
        
        # Descargar datos en timeframe 1m
        data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            if self.enable_logging:
                self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        # Guardar datos para reutilizar
        self.last_data = data.copy()
        
        # Cargar datos y analizar
        self.max_min_analyzer.load_data(data)
        simple_answer, detailed_data = self.max_min_analyzer.analyze_max_min_last_hours(hours)
        
        # Registrar en logs duales
        if self.enable_logging:
            self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data
    
    def ask_range_percentage(self, hours: float):
        """Pregunta 2: ¿De cuánto porcentaje fue el rango?"""
        question = f"¿De cuánto porcentaje fue el rango de las últimas {hours} horas?"
        
        # Reutilizar datos si ya los tenemos
        if self.last_data is not None and not self.last_data.empty:
            print(f"🔄 Reutilizando datos existentes para análisis de porcentaje...")
            data = self.last_data
        else:
            print(f"🔄 Descargando datos de {self.symbol} (1m) para análisis de porcentaje...")
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            if self.enable_logging:
                self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        # Cargar datos y analizar
        self.percentage_analyzer.load_data(data)
        simple_answer, detailed_data = self.percentage_analyzer.analyze_range_percentage(hours)
        
        # Registrar en logs duales
        if self.enable_logging:
            self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data
    
    def ask_low_time_minimums(self, hours: float, max_time_minutes: int = 15):
        """Pregunta 3: ¿Cuáles fueron los mínimos más bajos en los que el precio estuvo poco tiempo?"""
        question = f"¿Cuáles fueron los mínimos más bajos en los que el precio estuvo poco tiempo (≤{max_time_minutes}min)?"
        
        # Reutilizar datos si ya los tenemos
        if self.last_data is not None and not self.last_data.empty:
            print(f"🔄 Reutilizando datos existentes para análisis de mínimos rápidos...")
            data = self.last_data
        else:
            print(f"🔄 Descargando datos de {self.symbol} (1m) para análisis de mínimos...")
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            if self.enable_logging:
                self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        # Cargar datos y analizar
        self.low_time_analyzer.load_data(data)
        simple_answer, detailed_data = self.low_time_analyzer.analyze_low_time_minimums(hours, max_time_minutes)
        
        # Registrar en logs duales
        if self.enable_logging:
            self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data
    
    def ask_high_time_maximums(self, hours: float, max_time_minutes: int = 15):
        """Pregunta 4: ¿Cuáles fueron los máximos más altos en los que el precio estuvo poco tiempo?"""
        question = f"¿Cuáles fueron los máximos más altos en los que el precio estuvo poco tiempo (≤{max_time_minutes}min)?"
        
        # Reutilizar datos si ya los tenemos
        if self.last_data is not None and not self.last_data.empty:
            print(f"🔄 Reutilizando datos existentes para análisis de máximos rápidos...")
            data = self.last_data
        else:
            print(f"🔄 Descargando datos de {self.symbol} (1m) para análisis de máximos...")
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            if self.enable_logging:
                self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        # Cargar datos y analizar
        self.high_time_analyzer.load_data(data)
        simple_answer, detailed_data = self.high_time_analyzer.analyze_high_time_maximums(hours, max_time_minutes)
        
        # Registrar en logs duales
        if self.enable_logging:
            self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data
    
    def ask_48h_panorama(self):
        """Pregunta 5: Panorama de las últimas 48 horas"""
        question = "¿Cuál es el panorama de las últimas 48 horas?"
        
        print(f"🔄 Descargando datos de {self.symbol} (1m) para panorama de 48 horas...")
        
        # Descargar datos de 48 horas
        data = self.downloader.get_klines("1m", 48)
        
        if data.empty:
            simple_answer = "Sin datos disponibles para 48h"
            if self.enable_logging:
                self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        # Guardar datos de 48h para reutilizar
        self.last_48h_data = data.copy()
        
        # Cargar datos y analizar
        self.panorama_analyzer.load_data(data)
        simple_answer, detailed_data = self.panorama_analyzer.analyze_48h_panorama(48)
        
        # Registrar en logs duales
        if self.enable_logging:
            self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data
    
    def ask_weekly_analysis(self):
        """Pregunta 6: Análisis semanal completo con tendencia de extremos recientes"""
        question = "¿Cuál es el análisis semanal completo con tendencia de extremos recientes?"
        
        print(f"🔄 Descargando datos de {self.symbol} (1m) para análisis semanal...")
        
        # Descargar datos de 1 semana (168 horas)
        data = self.downloader.get_klines("1m", 168)
        
        if data.empty:
            simple_answer = "Sin datos disponibles para análisis semanal"
            if self.enable_logging:
                self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        # Guardar datos semanales para reutilizar
        self.last_weekly_data = data.copy()
        
        # Cargar datos y analizar
        self.weekly_analyzer.load_data(data)
        simple_answer, detailed_data = self.weekly_analyzer.analyze_weekly_with_recent_extremes(3)  # 3 días
        
        # Registrar en logs duales
        if self.enable_logging:
            self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data
    
    def _create_integrated_analysis(self, adaptive_results: Dict, ml_results: Dict) -> Dict[str, Any]:
        """Crear análisis integrado combinando todos los componentes"""
        
        integrated = {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "components_analyzed": []
        }
        
        # Análisis adaptativo
        if adaptive_results:
            integrated["adaptive_summary"] = self._summarize_adaptive_results(adaptive_results)
            integrated["components_analyzed"].append("adaptive_analysis")
        
        # Análisis ML
        if ml_results:
            integrated["ml_summary"] = self._summarize_ml_results(ml_results)
            integrated["components_analyzed"].append("ml_analysis")
        
        return integrated
    
    def _summarize_adaptive_results(self, adaptive_results: Dict) -> Dict[str, Any]:
        """Resumir resultados del análisis adaptativo"""
        
        summary = {
            "questions_completed": len(adaptive_results),
            "analysis_timeframes": ["3h", "3h", "3h", "3h", "48h", "1w"],
            "key_findings": []
        }
        
        # Extraer hallazgos clave
        for question, result in adaptive_results.items():
            if isinstance(result, tuple) and len(result) >= 2:
                simple_answer, detailed_data = result
                if isinstance(detailed_data, dict):
                    summary["key_findings"].append({
                        "question": question,
                        "finding": simple_answer,
                        "confidence": detailed_data.get("confidence", "medium")
                    })
        
        return summary
    
    def _summarize_ml_results(self, ml_results: Dict) -> Dict[str, Any]:
        """Resumir resultados del análisis ML"""
        
        if not ml_results or 'signal' not in ml_results:
            return {"status": "no_ml_signal"}
        
        signal = ml_results['signal']
        
        return {
            "signal_type": signal.signal_type.value,
            "confidence": signal.confidence,
            "expected_move": signal.expected_move_pct,
            "risk_reward_ratio": signal.risk_reward_ratio,
            "features_used_count": len(signal.features_used.__dict__) if signal.features_used else 0,
            "decision_path": ml_results.get("decision_path", []),
            "quality_metrics": ml_results.get("quality_metrics", {})
        }
    
    def _generate_executive_summary(self, adaptive_results: Dict, ml_results: Dict) -> str:
        """Generar resumen ejecutivo del análisis"""
        
        # Obtener información clave
        ml_signal = ml_results.get('signal') if ml_results else None
        features = ml_results.get('features') if ml_results else None
        
        # Construir resumen
        summary_parts = []
        
        # Situación del mercado
        if features:
            regime = features.regime.upper()
            volatility = features.volatility
            position_pct = features.price_position_pct
            momentum_3d = features.momentum_3d
            
            # Determinar zona
            if position_pct > 75:
                zone = "ZONA ALTA"
            elif position_pct < 25:
                zone = "ZONA BAJA"
            else:
                zone = "ZONA MEDIA"
            
            # Determinar estructura
            extremes_alignment = getattr(features, 'extremes_alignment', False)
            if extremes_alignment:
                structure = "ALINEADOS"
            else:
                structure = "DECRECIENTES"  # Basado en análisis semanal típico
            
            # Determinar momentum
            if abs(momentum_3d) < 1:
                momentum_status = "NEUTRAL"
            elif momentum_3d > 1:
                momentum_status = "ALCISTA"
            else:
                momentum_status = "BAJISTA"
            
            summary_parts.append(f"📊 SITUACIÓN: {regime} (vol {volatility:.1f}%)")
            summary_parts.append(f"📍 POSICIÓN: {zone} ({position_pct:.0f}% del rango semanal)")
            summary_parts.append(f"🏗️ ESTRUCTURA: {structure}")
            summary_parts.append(f"⚡ MOMENTUM 3D: {momentum_status} ({momentum_3d:.1f}%)")
        
        # Señal ML
        if ml_signal:
            signal_type = ml_signal.signal_type.value
            confidence = ml_signal.confidence
            summary_parts.append(f"🎯 SEÑAL: {signal_type} (confianza {confidence:.0f}%)")
        
        return " | ".join(summary_parts)
    
    def _extract_trading_signal(self, ml_results: Dict) -> Dict[str, Any]:
        """Extraer señal de trading del análisis ML"""
        
        if not ml_results or 'signal' not in ml_results:
            return {
                "type": "UNKNOWN",
                "confidence": 0,
                "message": "Sin señal ML disponible"
            }
        
        signal = ml_results['signal']
        
        return {
            "type": signal.signal_type.value,
            "confidence": signal.confidence,
            "price_target": signal.price_target,
            "stop_loss": signal.stop_loss,
            "expected_move_pct": signal.expected_move_pct,
            "risk_reward_ratio": signal.risk_reward_ratio,
            "timeframe": signal.timeframe,
            "reasoning": signal.reasoning[:3] if signal.reasoning else []  # Top 3 razones
        }
    
    def _generate_trading_recommendations(self, ml_results: Dict) -> Dict[str, Any]:
        """Generar recomendaciones de trading"""
        
        if not ml_results or 'signal' not in ml_results:
            return {
                "primary_action": "MONITOR",
                "confidence_level": "LOW",
                "position_sizing": "No recomendado",
                "entry_strategy": "Esperar señal clara"
            }
        
        signal = ml_results['signal']
        confidence = signal.confidence
        
        # Determinar acción primaria
        signal_type = signal.signal_type.value
        if signal_type in ["BUY", "STRONG_BUY"]:
            primary_action = "BUY"
        elif signal_type in ["SELL", "STRONG_SELL"]:
            primary_action = "SELL"
        else:
            primary_action = "MONITOR"
        
        # Determinar nivel de confianza
        if confidence >= 80:
            confidence_level = "MUY ALTA"
            position_sizing = "100% del tamaño normal de posición"
        elif confidence >= 70:
            confidence_level = "ALTA"
            position_sizing = "80% del tamaño normal de posición"
        elif confidence >= 60:
            confidence_level = "MEDIA"
            position_sizing = "60% del tamaño normal de posición"
        else:
            confidence_level = "BAJA"
            position_sizing = "40% del tamaño normal de posición"
        
        # Estrategia de entrada
        if primary_action == "BUY":
            if confidence >= 75:
                entry_strategy = "Entrada inmediata con confirmación"
            else:
                entry_strategy = "Entrada en 2 partes: 60% inmediato, 40% en retroceso"
        elif primary_action == "SELL":
            if confidence >= 75:
                entry_strategy = "Entrada inmediata en fortaleza"
            else:
                entry_strategy = "Entrada en 2 partes: 60% inmediato, 40% en retroceso"
        else:
            entry_strategy = "Esperar mayor claridad del mercado"
        
        return {
            "primary_action": primary_action,
            "confidence_level": confidence_level,
            "position_sizing": position_sizing,
            "entry_strategy": entry_strategy,
            "risk_management": f"Stop loss: ${signal.stop_loss:.2f}" if signal.stop_loss else "Definir stop según volatilidad"
        }
    
    def _analyze_confidence_factors(self, adaptive_results: Dict, ml_results: Dict) -> Dict[str, Any]:
        """Analizar factores que afectan la confianza"""
        
        factors = {
            "positive_factors": [],
            "negative_factors": [],
            "neutral_factors": []
        }
        
        # Factores del análisis ML
        if ml_results and 'signal' in ml_results:
            signal = ml_results['signal']
            features = ml_results.get('features')
            
            if features:
                # Factores positivos
                if signal.risk_reward_ratio and signal.risk_reward_ratio >= 2:
                    factors["positive_factors"].append(f"Excelente R:R ({signal.risk_reward_ratio:.1f}:1)")
                
                if getattr(features, 'extremes_alignment', False):
                    factors["positive_factors"].append("Extremos alineados (estructura clara)")
                
                if getattr(features, 'trend_strength', 0) > 0.8:
                    factors["positive_factors"].append("Tendencia fuerte y definida")
                
                if abs(getattr(features, 'momentum_3d', 0)) > 5:
                    factors["positive_factors"].append(f"Momentum fuerte ({features.momentum_3d:.1f}%)")
                
                # Factores negativos
                if getattr(features, 'volatility', 0) > 5:
                    factors["negative_factors"].append("Alta volatilidad puede generar ruido")
                
                position_pct = getattr(features, 'price_position_pct', 50)
                if position_pct > 90 and signal.signal_type.value in ["BUY", "STRONG_BUY"]:
                    factors["negative_factors"].append("Precio muy cerca del máximo semanal")
                
                if position_pct < 10 and signal.signal_type.value in ["SELL", "STRONG_SELL"]:
                    factors["negative_factors"].append("Precio muy cerca del mínimo semanal")
                
                # Detectar desalineación entre trend y momentum
                trend_strength = getattr(features, 'trend_strength', 0)
                momentum_3d = getattr(features, 'momentum_3d', 0)
                if trend_strength > 0.7 and abs(momentum_3d) < 2:
                    factors["negative_factors"].append("Desalineación entre trend y momentum")
        
        # Factores del análisis adaptativo
        total_questions = len(adaptive_results)
        if total_questions >= 6:
            factors["positive_factors"].append(f"Análisis completo de {total_questions} dimensiones")
        
        return factors
    
    def _debug_decision_tree(self, ml_results: Dict):
        """Debug detallado del árbol de decisiones"""
        
        if not ml_results or 'decision_path' not in ml_results:
            print("⚠️ No hay información de decision path disponible")
            return
        
        print(f"\n🔍 DEBUG DETALLADO DEL ÁRBOL DE DECISIONES:")
        print("=" * 60)
        
        decision_path = ml_results.get('decision_path', [])
        features = ml_results.get('features')
        signal = ml_results.get('signal')
        
        # Mostrar features clave
        if features:
            print(f"📊 Features de entrada:")
            print(f"   Régimen: {getattr(features, 'regime', 'unknown')}")
            print(f"   Volatilidad: {getattr(features, 'volatility', 0):.2f}%")
            print(f"   Momentum 3d: {getattr(features, 'momentum_3d', 0):.1f}%")
            print(f"   Posición: {getattr(features, 'price_position_pct', 0):.1f}%")
            print(f"   Extremos alineados: {getattr(features, 'extremes_alignment', False)}")
        
        # Mostrar path de decisiones
        if decision_path:
            print(f"\n🌳 Camino de decisiones:")
            for i, decision in enumerate(decision_path, 1):
                print(f"{i}️⃣ {decision}")
        
        # Mostrar resultado final
        if signal:
            print(f"\n🎯 Resultado final:")
            print(f"   Señal: {signal.signal_type.value}")
            print(f"   Confianza: {signal.confidence}%")
            if hasattr(signal, 'reasoning'):
                print(f"   Reasoning: {signal.reasoning}")
        
        print("=" * 60)

def test_complete_system_with_logging():
    """Probar el sistema completo con logging, debug detallado + Interpretación Probabilística"""
    
    print(f"🚀 Iniciando test completo del sistema - Usuario: biGGzeta")
    print(f"🕐 Fecha actual: 2025-10-04 22:56:45 UTC")
    
    # Inicializar sistema completo con logging
    system = AdaptiveTradingMLSystem("ETHUSD_PERP", enable_ml=True, enable_logging=True)
    
    # Ejecutar análisis completo
    results = system.run_complete_analysis_with_ml()
    
    # Mostrar resumen ejecutivo
    if results.get("status") == "success":
        print("\n" + "=" * 75)
        print("🎯 RESUMEN EJECUTIVO:")
        print("=" * 75)
        print(results["executive_summary"])
        
        signal = results["trading_signal"]
        print(f"\n🚨 SEÑAL DE TRADING:")
        print(f"   Tipo: {signal['type']}")
        print(f"   Confianza: {signal['confidence']:.0f}%")
        if signal.get('price_target'):
            print(f"   Target: ${signal['price_target']:.2f}")
        if signal.get('stop_loss'):
            print(f"   Stop Loss: ${signal['stop_loss']:.2f}")
        if signal.get('expected_move_pct'):
            print(f"   Movimiento esperado: {signal['expected_move_pct']:.1f}%")
        if signal.get('risk_reward_ratio'):
            print(f"   Risk/Reward: {signal['risk_reward_ratio']:.1f}:1")
        
        print(f"\n📊 RECOMENDACIONES:")
        recs = results["trading_recommendations"]
        print(f"   Acción: {recs['primary_action']}")
        print(f"   Confianza: {recs['confidence_level']}")
        print(f"   Posición: {recs['position_sizing']}")
        print(f"   Entrada: {recs['entry_strategy']}")
        
        # Mostrar factores de confianza
        confidence = results["confidence_analysis"]
        print(f"\n🔍 ANÁLISIS DE CONFIANZA:")
        if confidence.get('positive_factors'):
            print(f"   ✅ Factores positivos: {len(confidence['positive_factors'])}")
            for factor in confidence['positive_factors'][:3]:  # Top 3
                print(f"      • {factor}")
        
        if confidence.get('negative_factors'):
            print(f"   ⚠️ Factores negativos: {len(confidence['negative_factors'])}")
            for factor in confidence['negative_factors'][:3]:  # Top 3
                print(f"      • {factor}")
        
        # **NUEVA SECCIÓN: INTERPRETACIÓN PROBABILÍSTICA**
        print("\n" + "=" * 75)
        print("🧠 INICIANDO INTERPRETACIÓN PROBABILÍSTICA:")
        print("=" * 75)
        
        try:
            from analysis.probabilistic_interpreter import ProbabilisticInterpreter
            
            interpreter = ProbabilisticInterpreter("ETHUSD_PERP")
            probability_results = interpreter.interpret_complete_analysis(results)
            
            # Mostrar escenarios probabilísticos
            print(f"\n🎯 ESCENARIOS PROBABILÍSTICOS GENERADOS:")
            print("=" * 60)
            
            for i, scenario in enumerate(probability_results["scenarios"], 1):
                print(f"\n📊 ESCENARIO {i}: {scenario['name']}")
                print(f"   🎯 Probabilidad: {scenario['probability']}%")
                print(f"   📈 Target: {scenario['target_range']}")
                print(f"   ⏰ Timeframe: {scenario['timeframe']}")
                if scenario.get('risk_reward'):
                    print(f"   📊 R:R: {scenario['risk_reward']}")
                if scenario.get('reasoning'):
                    print(f"   💡 Reasoning:")
                    for reason in scenario['reasoning'][:3]:
                        print(f"      • {reason}")
            
            # Mostrar recomendación final
            external_signal = probability_results["signal_for_external"]
            print(f"\n🚨 RECOMENDACIÓN PROBABILÍSTICA FINAL:")
            print(f"   🎯 Acción: {external_signal['action']}")
            print(f"   📊 Confianza: {external_signal['confidence']}%")
            print(f"   🎪 Escenario principal: {external_signal['primary_scenario']}")
            print(f"   📍 Target: {external_signal['target_range']}")
            
            # Mostrar archivo de señal externa generado
            print(f"\n📁 ARCHIVOS DE SEÑALES GENERADOS:")
            print(f"   🚨 Señal externa: logs/probabilities/signal_external_{system.symbol}.json")
            print(f"   📊 Análisis completo: logs/probabilities/interpretation_{system.symbol}_*.json")
            
        except ImportError:
            print("⚠️ Módulo de interpretación probabilística no encontrado")
            print("   Asegúrate de que analysis/probabilistic_interpreter.py esté en el directorio correcto")
            print("   Ejecuta: mkdir -p analysis && # coloca el archivo ahí")
        except Exception as e:
            print(f"❌ Error en interpretación probabilística: {str(e)}")
            print(f"   Detalles: {type(e).__name__}")
    
    elif results.get("status") == "error_ml":
        print("\n❌ Error en pipeline ML:")
        print(f"   {results['ml_error']}")
    
    print(f"\n✅ Test completado - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"👤 Usuario: biGGzeta")
    return results

if __name__ == "__main__":
    test_complete_system_with_logging()