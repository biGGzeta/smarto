import asyncio
import pandas as pd
from typing import Dict, Any, Tuple
from data.csv_handler import BinanceDataDownloader
from analysis.adaptive_low_time_analyzer import AdaptiveLowTimeAnalyzer
from analysis.adaptive_high_time_analyzer import AdaptiveHighTimeAnalyzer
from analysis.adaptive_percentage_analyzer import AdaptivePercentageAnalyzer
from analysis.adaptive_panorama_analyzer import AdaptivePanoramaAnalyzer
from analysis.adaptive_weekly_analyzer import AdaptiveWeeklyAnalyzer
from analysis.max_min_analyzer import MaxMinAnalyzer  # Mantener el original para comparación
from utils.logger import QALogger
from config.settings import BinanceConfig

# Importar sistema ML
from ml.adaptive_ml_system import AdaptiveMLSystem
from ml.feature_engineering import MarketFeatures
from ml.decision_tree import TradingSignal, SignalType

# Importar logger de análisis profundo
from analysis.market_data_logger import MarketDataLogger

class AdaptiveTradingMLSystem:
    """Sistema principal que combina análisis adaptativo + ML + señales automáticas + logging completo"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP", enable_ml: bool = True, enable_logging: bool = True):
        self.symbol = symbol
        self.downloader = BinanceDataDownloader(symbol)
        
        # Analizadores adaptativos (existentes)
        self.adaptive_low_analyzer = AdaptiveLowTimeAnalyzer(symbol)
        self.adaptive_high_analyzer = AdaptiveHighTimeAnalyzer(symbol)
        self.adaptive_percentage_analyzer = AdaptivePercentageAnalyzer(symbol)
        self.adaptive_panorama_analyzer = AdaptivePanoramaAnalyzer(symbol)
        self.adaptive_weekly_analyzer = AdaptiveWeeklyAnalyzer(symbol)
        self.original_max_min_analyzer = MaxMinAnalyzer(symbol)
        
        # Sistema ML (NUEVO)
        self.ml_system = AdaptiveMLSystem(save_data=enable_ml) if enable_ml else None
        self.enable_ml = enable_ml
        
        # Logger de análisis profundo (NUEVO)
        self.market_logger = MarketDataLogger(symbol) if enable_logging else None
        self.enable_logging = enable_logging
        
        self.logger = QALogger(symbol)
        self.analysis_cache = {}
        
        print(f"🎯 Sistema inicializado para {symbol}")
        print(f"   🤖 ML: {'✅ Habilitado' if enable_ml else '❌ Deshabilitado'}")
        print(f"   📝 Logging: {'✅ Habilitado' if enable_logging else '❌ Deshabilitado'}")
        
    def run_complete_analysis_with_ml(self, hours_3h: float = 3, hours_48h: float = 48) -> Dict[str, Any]:
        """Ejecutar análisis completo: Adaptativo + ML + Señales + Logging"""
        
        print("🚀 Iniciando Sistema Completo: Análisis Adaptativo + ML + Logging")
        print("=" * 75)
        print(f"📊 Símbolo: {self.symbol}")
        print(f"🕐 Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"👤 Usuario: biGGzeta")
        print(f"🤖 ML habilitado: {self.enable_ml}")
        print(f"📝 Logging profundo: {self.enable_logging}")
        print("=" * 75)
        
        # FASE 1: Análisis adaptativo (como antes)
        print("\n🔧 FASE 1: Ejecutando análisis adaptativo...")
        adaptive_results = self._run_adaptive_analysis(hours_3h, hours_48h)
        
        if not self.enable_ml:
            print("ℹ️  ML deshabilitado, retornando solo análisis adaptativo")
            return {"adaptive_results": adaptive_results, "status": "adaptive_only"}
        
        # FASE 2: ML Pipeline (NUEVO)
        print("\n🤖 FASE 2: Ejecutando pipeline ML...")
        ml_results = self.ml_system.analyze_and_signal(adaptive_results)
        
        # FASE 3: Análisis integrado
        print("\n📊 FASE 3: Generando análisis integrado...")
        integrated_analysis = self._create_integrated_analysis(adaptive_results, ml_results)
        
        # FASE 4: Logging completo (NUEVO)
        if self.enable_logging and self.market_logger:
            print("\n📝 FASE 4: Guardando análisis profundo...")
            try:
                report_file = self.market_logger.log_complete_analysis(
                    adaptive_results, ml_results, integrated_analysis
                )
                integrated_analysis['analysis_report_file'] = report_file
                
                # Crear dashboard data
                dashboard_file = self.market_logger.create_analysis_dashboard_data()
                integrated_analysis['dashboard_data_file'] = dashboard_file
                
                print(f"✅ Logging completado - Archivos generados en market_analysis/")
                
            except Exception as e:
                print(f"⚠️ Error en logging: {str(e)}")
                integrated_analysis['logging_error'] = str(e)
        
        # FASE 5: Debug detallado de decisiones (NUEVO)
        if integrated_analysis.get("status") == "success":
            print("\n🔍 FASE 5: Debug detallado del árbol de decisiones...")
            self._debug_decision_tree_path(ml_results.get("features"))
        
        return integrated_analysis
    
    def _run_adaptive_analysis(self, hours_3h: float, hours_48h: float) -> Dict[str, Any]:
        """Ejecutar análisis adaptativo (código existente)"""
        
        results = {}
        
        # Pregunta 1: Máximos y mínimos básicos
        print("\n📊 PREGUNTA 1: Máximos y mínimos básicos")
        result1 = self._ask_max_min_last_hours(hours_3h)
        results["max_min_basic"] = result1
        
        # Pregunta 2: Análisis de porcentajes adaptativo
        print("\n📊 PREGUNTA 2: Análisis de porcentajes adaptativo")
        result2 = self._ask_adaptive_range_percentage(hours_3h)
        results["range_percentage_adaptive"] = result2
        
        # Pregunta 3: Mínimos adaptativos
        print("\n📊 PREGUNTA 3: Mínimos adaptativos")
        result3 = self._ask_adaptive_low_time_minimums(hours_3h)
        results["low_minimums_adaptive"] = result3
        
        # Pregunta 4: Máximos adaptativos
        print("\n📊 PREGUNTA 4: Máximos adaptativos")
        result4 = self._ask_adaptive_high_time_maximums(hours_3h)
        results["high_maximums_adaptive"] = result4
        
        # Pregunta 5: Panorama 48h adaptativo
        print("\n📊 PREGUNTA 5: Panorama 48h adaptativo")
        result5 = self._ask_adaptive_48h_panorama()
        results["panorama_48h_adaptive"] = result5
        
        # Pregunta 6: Análisis semanal adaptativo
        print("\n📊 PREGUNTA 6: Análisis semanal adaptativo")
        result6 = self._ask_adaptive_weekly_analysis()
        results["weekly_adaptive"] = result6
        
        return results
    
    def _create_integrated_analysis(self, adaptive_results: Dict, ml_results: Dict) -> Dict[str, Any]:
        """Crear análisis integrado combinando adaptativo + ML"""
        
        if "error" in ml_results:
            return {
                "timestamp": pd.Timestamp.now().isoformat(),
                "status": "error_ml",
                "adaptive_results": adaptive_results,
                "ml_error": ml_results["error"]
            }
        
        features = ml_results.get("features")
        signal = ml_results.get("signal")
        ml_data = ml_results.get("ml_data")
        
        # Crear resumen ejecutivo
        executive_summary = self._create_executive_summary(features, signal, adaptive_results)
        
        # Crear recomendaciones de trading
        trading_recommendations = self._create_trading_recommendations(signal, features)
        
        # Análisis de confianza
        confidence_analysis = self._analyze_confidence_factors(signal, features)
        
        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "status": "success",
            "symbol": self.symbol,
            "user": "biGGzeta",
            
            # Resultados principales
            "executive_summary": executive_summary,
            "trading_signal": {
                "type": signal.signal_type.value,
                "confidence": signal.confidence,
                "price_target": signal.price_target,
                "stop_loss": signal.stop_loss,
                "expected_move_pct": signal.expected_move_pct,
                "risk_reward_ratio": signal.risk_reward_ratio,
                "timeframe": signal.timeframe
            },
            "trading_recommendations": trading_recommendations,
            "confidence_analysis": confidence_analysis,
            
            # Datos detallados
            "market_features": {
                "regime": features.regime,
                "volatility": features.volatility,
                "trend_strength": features.trend_strength,
                "price_position_pct": features.price_position_pct,
                "momentum_3d": features.momentum_3d,
                "extremes_alignment": features.extremes_alignment,
                "maximos_trend": features.maximos_trend,
                "minimos_trend": features.minimos_trend,
                "maximos_strength": features.maximos_strength,
                "minimos_strength": features.minimos_strength,
                "trend_momentum_alignment": features.trend_momentum_alignment
            },
            "decision_reasoning": signal.reasoning,
            
            # Datos completos para análisis
            "adaptive_results": adaptive_results,
            "ml_results": ml_results,
            
            # Metadatos
            "pipeline_version": "1.0",
            "analysis_quality": ml_results.get("quality_metrics", {})
        }
    
    def _debug_decision_tree_path(self, features: MarketFeatures):
        """Debug manual del path del árbol de decisiones"""
        
        if not features:
            print("❌ No hay features para debug")
            return
        
        print("\n🔍 DEBUG DETALLADO DEL ÁRBOL DE DECISIONES:")
        print("=" * 60)
        
        # Nodo 1: Régimen
        is_volatile = features.regime in ["high_volatility", "trending"]
        print(f"1️⃣ Régimen volátil/trending: {is_volatile}")
        print(f"   📊 Régimen actual: {features.regime}")
        
        if not is_volatile:  # Va por rama calma
            print("   ➡️ Tomando rama: CALMA/RANGING")
            
            # Nodo 2: Estructura extremos
            both_growing = features.maximos_trend == "crecientes" and features.minimos_trend == "crecientes"
            print(f"\n2️⃣ Ambos extremos crecientes: {both_growing}")
            print(f"   📈 Máximos: {features.maximos_trend}")
            print(f"   📉 Mínimos: {features.minimos_trend}")
            
            if both_growing:  # Va por rama alcista
                print("   ➡️ Tomando rama: ESTRUCTURA ALCISTA")
                
                # Nodo 3: Estructura fuerte
                strong_structure = features.maximos_strength > 60 and features.minimos_strength > 60
                print(f"\n3️⃣ Estructura fuerte (>60): {strong_structure}")
                print(f"   💪 Fuerza máximos: {features.maximos_strength:.1f}")
                print(f"   💪 Fuerza mínimos: {features.minimos_strength:.1f}")
                
                if strong_structure:
                    print("   ➡️ Estructura FUERTE detectada")
                    
                    # Nodo 4: Momentum alignment
                    momentum_ok = features.momentum_3d > 1 and features.trend_momentum_alignment > 0.5
                    print(f"\n4️⃣ Momentum + alignment OK: {momentum_ok}")
                    print(f"   ⚡ Momentum 3d: {features.momentum_3d:.2f}%")
                    print(f"   🎯 Trend-momentum alignment: {features.trend_momentum_alignment:.3f}")
                    print(f"   ✅ Momentum > 1: {features.momentum_3d > 1}")
                    print(f"   ✅ Alignment > 0.5: {features.trend_momentum_alignment > 0.5}")
                    
                    if momentum_ok:
                        print("   ➡️ Momentum y alignment VÁLIDOS")
                        
                        # Nodo 5: Momentum súper fuerte
                        super_momentum = features.momentum_3d > 7
                        print(f"\n5️⃣ Momentum súper fuerte (>7): {super_momentum}")
                        
                        if super_momentum:
                            print("   ➡️ MOMENTUM SÚPER FUERTE detectado")
                            
                            # Nodo 6: Posición final
                            position_ok = features.price_position_pct < 90
                            print(f"\n6️⃣ Posición OK (<90): {position_ok}")
                            print(f"   📍 Posición actual: {features.price_position_pct:.1f}%")
                            
                            if position_ok:
                                print("   🎯 RESULTADO ESPERADO: BULLISH_SUPER_MOMENTUM (90% confianza)")
                            else:
                                print("   ⚠️ RESULTADO ESPERADO: BULLISH_MOMENTUM_TOP (75% confianza)")
                        else:
                            print("   ➡️ Momentum fuerte pero no súper fuerte")
                            # Verificar siguiente nivel
                            strong_momentum = features.momentum_3d > 5
                            print(f"\n5️⃣.1 Momentum fuerte (>5): {strong_momentum}")
                            
                            if strong_momentum:
                                position_strong = features.price_position_pct < 85
                                print(f"6️⃣.1 Posición OK (<85): {position_strong}")
                                print(f"   📍 Posición: {features.price_position_pct:.1f}%")
                                
                                if position_strong:
                                    print("   🎯 RESULTADO ESPERADO: BULLISH_STRONG_MOMENTUM (85% confianza)")
                                else:
                                    print("   ⚠️ RESULTADO ESPERADO: BULLISH_MOMENTUM_HIGH (70% confianza)")
                    else:
                        print("   ❌ Momentum o alignment insuficientes")
                        print("   🎯 RESULTADO ESPERADO: BULLISH_STRUCTURE_WEAK_MOMENTUM (65% confianza)")
                else:
                    print("   ❌ Estructura débil")
                    print("   🎯 RESULTADO ESPERADO: BULLISH_STRUCTURE_WAIT o similar")
            else:
                print("   ❌ Extremos no alineados crecientes")
        else:
            print("   ➡️ Tomando rama: VOLATIL/TRENDING")
        
        print("=" * 60)
    
    def _create_executive_summary(self, features: MarketFeatures, signal: TradingSignal, adaptive_results: Dict) -> str:
        """Crear resumen ejecutivo para traders"""
        
        # Extraer datos clave del análisis adaptativo
        weekly_result = adaptive_results.get("weekly_adaptive", ("", {}))
        if isinstance(weekly_result, tuple) and len(weekly_result) > 1:
            weekly_simple = weekly_result[0]
        else:
            weekly_simple = "Análisis semanal no disponible"
        
        summary_parts = []
        
        # 1. Situación del mercado
        summary_parts.append(f"📊 SITUACIÓN: {features.regime.upper()} (vol {features.volatility:.1f}%)")
        
        # 2. Posición en rango
        if features.price_position_pct > 80:
            position_desc = "ZONA ALTA"
        elif features.price_position_pct < 20:
            position_desc = "ZONA BAJA"
        else:
            position_desc = "ZONA MEDIA"
        summary_parts.append(f"📍 POSICIÓN: {position_desc} ({features.price_position_pct:.0f}% del rango semanal)")
        
        # 3. Estructura de mercado
        if features.extremes_alignment:
            structure = f"Estructura {features.maximos_trend.upper()}"
        else:
            structure = f"Divergencia: máx {features.maximos_trend} vs mín {features.minimos_trend}"
        summary_parts.append(f"🏗️ ESTRUCTURA: {structure}")
        
        # 4. Momentum
        if features.momentum_3d > 2:
            momentum_desc = f"ALCISTA FUERTE (+{features.momentum_3d:.1f}%)"
        elif features.momentum_3d > 0.5:
            momentum_desc = f"ALCISTA (+{features.momentum_3d:.1f}%)"
        elif features.momentum_3d < -2:
            momentum_desc = f"BAJISTA FUERTE ({features.momentum_3d:.1f}%)"
        elif features.momentum_3d < -0.5:
            momentum_desc = f"BAJISTA ({features.momentum_3d:.1f}%)"
        else:
            momentum_desc = f"NEUTRAL ({features.momentum_3d:.1f}%)"
        summary_parts.append(f"⚡ MOMENTUM 3D: {momentum_desc}")
        
        # 5. Señal principal
        summary_parts.append(f"🎯 SEÑAL: {signal.signal_type.value} (confianza {signal.confidence:.0f}%)")
        
        return " | ".join(summary_parts)
    
    def _create_trading_recommendations(self, signal: TradingSignal, features: MarketFeatures) -> Dict[str, Any]:
        """Crear recomendaciones específicas de trading"""
        
        recommendations = {
            "primary_action": signal.signal_type.value,
            "confidence_level": self._get_confidence_level(signal.confidence),
            "position_sizing": self._suggest_position_sizing(signal, features),
            "entry_strategy": self._suggest_entry_strategy(signal, features),
            "risk_management": self._suggest_risk_management(signal, features),
            "time_horizon": signal.timeframe,
            "market_context": self._get_market_context(features)
        }
        
        return recommendations
    
    def _analyze_confidence_factors(self, signal: TradingSignal, features: MarketFeatures) -> Dict[str, Any]:
        """Analizar factores que afectan la confianza de la señal"""
        
        factors = {
            "positive_factors": [],
            "negative_factors": [],
            "neutral_factors": [],
            "overall_confidence": signal.confidence
        }
        
        # Factores positivos
        if features.extremes_alignment:
            factors["positive_factors"].append("Extremos alineados (estructura clara)")
        
        if signal.risk_reward_ratio and signal.risk_reward_ratio > 2:
            factors["positive_factors"].append(f"Excelente R:R ({signal.risk_reward_ratio:.1f}:1)")
        
        if features.trend_strength > 0.7:
            factors["positive_factors"].append("Tendencia fuerte y definida")
        
        if abs(features.momentum_3d) > 5:
            factors["positive_factors"].append(f"Momentum fuerte ({features.momentum_3d:.1f}%)")
        
        # Factores negativos
        if features.volatility > 5:
            factors["negative_factors"].append("Alta volatilidad puede generar ruido")
        
        if features.price_position_pct > 90 and signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            factors["negative_factors"].append("Precio muy cerca del máximo semanal")
        
        if features.price_position_pct < 10 and signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            factors["negative_factors"].append("Precio muy cerca del mínimo semanal")
        
        if features.trend_momentum_alignment < 0.5:
            factors["negative_factors"].append("Desalineación entre trend y momentum")
        
        # Factores neutrales
        if features.regime == "smooth_ranging":
            factors["neutral_factors"].append("Mercado lateral - movimientos limitados")
        
        if features.maximos_strength < 60 or features.minimos_strength < 60:
            factors["neutral_factors"].append("Estructura de extremos débil")
        
        return factors
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convertir confianza numérica a descripción"""
        if confidence >= 80:
            return "MUY ALTA"
        elif confidence >= 70:
            return "ALTA"
        elif confidence >= 60:
            return "MEDIA"
        elif confidence >= 50:
            return "BAJA"
        else:
            return "MUY BAJA"
    
    def _suggest_position_sizing(self, signal: TradingSignal, features: MarketFeatures) -> str:
        """Sugerir tamaño de posición basado en confianza y volatilidad"""
        
        base_size = 100  # Tamaño base
        
        # Ajustar por confianza
        if signal.confidence >= 80:
            confidence_factor = 1.0
        elif signal.confidence >= 70:
            confidence_factor = 0.8
        elif signal.confidence >= 60:
            confidence_factor = 0.6
        else:
            confidence_factor = 0.4
        
        # Ajustar por volatilidad
        if features.volatility > 5:
            volatility_factor = 0.7
        elif features.volatility > 3:
            volatility_factor = 0.8
        else:
            volatility_factor = 1.0
        
        suggested_size = base_size * confidence_factor * volatility_factor
        
        return f"{suggested_size:.0f}% del tamaño normal de posición"
    
    def _suggest_entry_strategy(self, signal: TradingSignal, features: MarketFeatures) -> str:
        """Sugerir estrategia de entrada"""
        
        if signal.confidence >= 80:
            return "Entrada inmediata - señal muy fuerte"
        elif signal.confidence >= 70:
            return "Entrada en 2 partes: 60% inmediato, 40% en retroceso"
        elif signal.confidence >= 60:
            return "Esperar retroceso para mejor entrada"
        else:
            return "Esperar confirmación adicional antes de entrar"
    
    def _suggest_risk_management(self, signal: TradingSignal, features: MarketFeatures) -> Dict[str, str]:
        """Sugerir gestión de riesgo"""
        
        risk_management = {}
        
        if signal.stop_loss:
            risk_management["stop_loss"] = f"${signal.stop_loss:.2f}"
        
        if signal.price_target:
            risk_management["take_profit"] = f"${signal.price_target:.2f}"
        
        if features.volatility > 5:
            risk_management["trailing_stop"] = "Recomendado debido a alta volatilidad"
        
        risk_management["position_review"] = "Revisar en 3-6 horas"
        
        return risk_management
    
    def _get_market_context(self, features: MarketFeatures) -> str:
        """Obtener contexto del mercado"""
        
        contexts = []
        
        if features.regime == "high_volatility":
            contexts.append("mercado volátil")
        elif features.regime == "trending":
            contexts.append("mercado en tendencia")
        else:
            contexts.append("mercado lateral")
        
        if features.price_position_pct > 80:
            contexts.append("cerca de resistencia semanal")
        elif features.price_position_pct < 20:
            contexts.append("cerca de soporte semanal")
        
        return ", ".join(contexts)
    
    # Métodos existentes (copiados del main_adaptive.py)
    def _ask_max_min_last_hours(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 1: Máximos y mínimos básicos"""
        question = f"¿Cuáles son los máximos y mínimos de las últimas {hours} horas?"
        
        print(f"🔄 Descargando datos de {self.symbol} (1m) para las últimas {hours} horas...")
        
        data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        self.analysis_cache["data_3h"] = data.copy()
        
        self.original_max_min_analyzer.load_data(data)
        simple_answer, detailed_data = self.original_max_min_analyzer.analyze_max_min_last_hours(hours)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        if isinstance(detailed_data, dict):
            print(f"✅ Respuesta 1: {simple_answer}")
            print(f"📈 Volatilidad: {detailed_data.get('percentage_range', 0):.2f}%")
        
        return simple_answer, detailed_data
    
    def _ask_adaptive_range_percentage(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 2: Análisis de porcentajes con parámetros adaptativos"""
        question = f"¿Cuál es el análisis adaptativo de porcentajes de las últimas {hours} horas?"
        
        if "data_3h" in self.analysis_cache:
            print(f"🔄 Reutilizando datos existentes para análisis adaptativo...")
            data = self.analysis_cache["data_3h"]
        else:
            print(f"🔄 Descargando datos de {self.symbol} para análisis adaptativo...")
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        self.adaptive_percentage_analyzer.load_data(data)
        simple_answer, detailed_data = self.adaptive_percentage_analyzer.analyze_range_percentage_adaptive(hours)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        print(f"✅ Respuesta 2: {simple_answer}")
        
        return simple_answer, detailed_data
    
    def _ask_adaptive_low_time_minimums(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 3: Mínimos adaptativos"""
        question = f"¿Cuáles son los mínimos adaptativos de las últimas {hours} horas?"
        
        if "data_3h" in self.analysis_cache:
            print(f"🔄 Reutilizando datos para análisis adaptativo de mínimos...")
            data = self.analysis_cache["data_3h"]
        else:
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        self.adaptive_low_analyzer.load_data(data)
        simple_answer, detailed_data = self.adaptive_low_analyzer.analyze_low_time_minimums_adaptive(hours)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        print(f"✅ Respuesta 3: {simple_answer}")
        
        return simple_answer, detailed_data
    
    def _ask_adaptive_high_time_maximums(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 4: Máximos adaptativos"""
        question = f"¿Cuáles son los máximos adaptativos de las últimas {hours} horas?"
        
        if "data_3h" in self.analysis_cache:
            print(f"🔄 Reutilizando datos para análisis adaptativo de máximos...")
            data = self.analysis_cache["data_3h"]
        else:
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        self.adaptive_high_analyzer.load_data(data)
        simple_answer, detailed_data = self.adaptive_high_analyzer.analyze_high_time_maximums_adaptive(hours)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        print(f"✅ Respuesta 4: {simple_answer}")
        
        return simple_answer, detailed_data
    
    def _ask_adaptive_48h_panorama(self) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 5: Panorama 48h adaptativo"""
        question = "¿Cuál es el panorama adaptativo de las últimas 48 horas?"
        
        print(f"🔄 Descargando datos de {self.symbol} para panorama adaptativo 48h...")
        
        data = self.downloader.get_klines("1m", 48)
        
        if data.empty:
            simple_answer = "Sin datos disponibles para 48h"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        self.analysis_cache["data_48h"] = data.copy()
        
        self.adaptive_panorama_analyzer.load_data(data)
        simple_answer, detailed_data = self.adaptive_panorama_analyzer.analyze_48h_panorama_adaptive(48)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        print(f"✅ Respuesta 5: {simple_answer}")
        
        return simple_answer, detailed_data
    
    def _ask_adaptive_weekly_analysis(self) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 6: Análisis semanal adaptativo"""
        question = "¿Cuál es el análisis semanal adaptativo completo?"
        
        print(f"🔄 Descargando datos de {self.symbol} para análisis semanal adaptativo...")
        
        data = self.downloader.get_klines("1m", 168)
        
        if data.empty:
            simple_answer = "Sin datos disponibles para análisis semanal"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        self.analysis_cache["data_weekly"] = data.copy()
        
        self.adaptive_weekly_analyzer.load_data(data)
        simple_answer, detailed_data = self.adaptive_weekly_analyzer.analyze_weekly_with_recent_extremes_adaptive(3)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        print(f"✅ Respuesta 6: {simple_answer}")
        
        if isinstance(detailed_data, dict):
            maximos_trend = detailed_data.get('maximos_trend', {})
            minimos_trend = detailed_data.get('minimos_trend', {})
            print(f"🎯 Máximos 3d: {maximos_trend.get('trend', 'unknown')}")
            print(f"🎯 Mínimos 3d: {minimos_trend.get('trend', 'unknown')}")
        
        return simple_answer, detailed_data

def test_complete_system_with_logging():
    """Probar el sistema completo con logging y debug detallado"""
    
    print(f"🚀 Iniciando test completo del sistema - Usuario: biGGzeta")
    print(f"🕐 Fecha actual: 2025-10-04 04:47:48 UTC")
    
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
        
        # Mostrar archivos generados
        if 'analysis_report_file' in results:
            print(f"\n📝 ARCHIVOS GENERADOS:")
            print(f"   📊 Reporte completo: {results['analysis_report_file']}")
            print(f"   📈 Dashboard data: {results.get('dashboard_data_file', 'N/A')}")
        
        # Generar reporte final de sesión
        if hasattr(system, 'market_logger') and system.market_logger:
            session_report = system.market_logger.generate_session_report()
            print(f"   📋 Reporte de sesión: {session_report}")
    
    elif results.get("status") == "error_ml":
        print("\n❌ Error en pipeline ML:")
        print(f"   {results['ml_error']}")
    
    print(f"\n✅ Test completado - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    return results

if __name__ == "__main__":
    test_complete_system_with_logging()