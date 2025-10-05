#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Trading System - VERSIÓN INTEGRADA COMPLETA
Sistema Base + Volume Layer + Setup Detection
Autor: biGGzeta
Fecha: 2025-10-05 12:31:27 UTC
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import datetime
import json

# Imports del sistema base
from main_adaptive_ml import AdaptiveTradingMLSystem

# Import corregido para Volume Layer
sys.path.append(os.path.join('layers', 'volume'))
from volume_analyzer import VolumeAnalyzer

# Import del Setup Detector
from setup_detector import SetupDetector

class EnhancedTradingSystem:
    """Sistema de trading mejorado completo: Base + Volume + Setup Detection"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP", capital: float = 1000.0):
        self.symbol = symbol
        self.capital = capital
        
        # Sistemas componentes
        self.base_system = AdaptiveTradingMLSystem(symbol, enable_ml=True, enable_logging=True)
        self.volume_analyzer = VolumeAnalyzer(symbol)
        self.setup_detector = SetupDetector(symbol)
        
        # Configuración de fusion
        self.fusion_weights = {
            "base_system": 0.7,      # 70% peso al sistema base
            "volume_layer": 0.3      # 30% peso al volumen
        }
        
        # Logs
        self.enhanced_logs_dir = "logs/enhanced"
        os.makedirs(self.enhanced_logs_dir, exist_ok=True)
        
        print(f"🚀 Enhanced Trading System COMPLETO iniciado para {symbol}")
        print(f"💰 Capital: ${capital:,.2f}")
        print(f"⚖️ Pesos: Base {self.fusion_weights['base_system']*100:.0f}% + Volume {self.fusion_weights['volume_layer']*100:.0f}%")
        print(f"🎯 Módulos: Base + Volume + Setup Detection")
    
    def run_complete_enhanced_analysis(self) -> Dict:
        """Ejecutar análisis completo mejorado con setup detection"""
        
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        print(f"\n🚀 INICIANDO ANÁLISIS COMPLETO MEJORADO - {timestamp.strftime('%H:%M:%S')} UTC")
        print("=" * 75)
        
        try:
            # FASE 1: Ejecutar sistema base
            print("📊 FASE 1: Ejecutando sistema base...")
            base_results = self._run_base_analysis()
            
            # FASE 2: Extraer datos OHLCV del sistema base
            print("📊 FASE 2: Extrayendo datos OHLCV...")
            ohlcv_data = self._extract_ohlcv_from_base_system()
            
            # FASE 3: Ejecutar análisis de volumen
            print("📊 FASE 3: Ejecutando análisis de volumen...")
            volume_results = self._run_volume_analysis(ohlcv_data)
            
            # FASE 4: Fusionar señales
            print("📊 FASE 4: Fusionando señales...")
            enhanced_results = self._fuse_signals(base_results, volume_results)
            
            # FASE 5: Generar interpretación mejorada
            print("📊 FASE 5: Generando interpretación mejorada...")
            enhanced_interpretation = self._generate_enhanced_interpretation(enhanced_results)
            
            # FASE 6: NUEVA - Detectar setup específico
            print("📊 FASE 6: Detectando tipo de setup...")
            setup_detection_result = self._run_setup_detection(base_results, volume_results, enhanced_results)
            
            # FASE 7: NUEVA - Generar recomendación final integrada
            print("📊 FASE 7: Generando recomendación final...")
            final_recommendation = self._generate_final_recommendation(enhanced_results, setup_detection_result)
            
            # Resultado final completo
            complete_results = {
                "timestamp": timestamp.isoformat(),
                "symbol": self.symbol,
                "analysis_type": "complete_enhanced_base_plus_volume_plus_setup",
                
                # Componentes
                "base_analysis": base_results,
                "volume_analysis": volume_results,
                
                # Resultado mejorado
                "enhanced_signal": enhanced_results,
                "enhanced_interpretation": enhanced_interpretation,
                
                # NUEVO: Setup detection
                "setup_detection": setup_detection_result,
                "final_recommendation": final_recommendation,
                
                # Metadatos
                "fusion_weights": self.fusion_weights,
                "performance_metrics": self._calculate_performance_metrics(base_results, volume_results, enhanced_results),
                "complete_pipeline_version": "1.0.0"
            }
            
            # Guardar logs
            self._save_complete_enhanced_log(complete_results)
            
            # Mostrar resumen completo
            self._display_complete_enhanced_summary(complete_results)
            
            return complete_results
            
        except Exception as e:
            print(f"❌ Error en análisis completo mejorado: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e), "timestamp": timestamp.isoformat()}
    
    def _run_setup_detection(self, base_results: Dict, volume_results: Dict, enhanced_results: Dict) -> Dict:
        """Ejecutar detección de setup"""
        
        try:
            # Preparar datos para setup detector
            setup_input = {
                "base_analysis": base_results,
                "volume_analysis": volume_results,
                "enhanced_signal": enhanced_results
            }
            
            # Extraer precio actual
            current_price = self._extract_current_price_from_results(base_results, volume_results)
            
            # Ejecutar detección
            setup_result = self.setup_detector.detect_setup(setup_input, current_price=current_price)
            
            if setup_result.get("status") != "error":
                detected_setup = setup_result.get("detected_setup", "NO_CLEAR_SETUP")
                confidence = setup_result.get("setup_confidence", 0)
                
                print(f"✅ Setup detection completado: {detected_setup} ({confidence}/100)")
                return setup_result
            else:
                print(f"❌ Error en setup detection: {setup_result.get('message', 'Unknown')}")
                return setup_result
                
        except Exception as e:
            print(f"❌ Error ejecutando setup detection: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _extract_current_price_from_results(self, base_results: Dict, volume_results: Dict) -> float:
        """Extraer precio actual de los resultados del análisis"""
        
        try:
            # Método 1: Desde respuestas del bot (más reciente)
            bot_responses = base_results.get("bot_responses", {})
            for question, response in bot_responses.items():
                if "precio actual" in response.get("response", "").lower():
                    price = self._parse_price_from_text(response.get("response", ""))
                    if price:
                        return price
            
            # Método 2: Desde cualquier respuesta del bot
            for question, response in bot_responses.items():
                price = self._parse_price_from_text(response.get("response", ""))
                if price:
                    return price
            
            # Método 3: Fallback precio ETH típico actual
            return 4538.79  # Precio aproximado del contexto actual
            
        except Exception as e:
            print(f"⚠️ Error extrayendo precio actual: {str(e)}")
            return 4538.79
    
    def _parse_price_from_text(self, text: str) -> Optional[float]:
        """Parsear precio del texto"""
        
        import re
        
        # Buscar patrones como "$4,540.50" o "precio actual: $4,540"
        patterns = [
            r'actual[^$]*\$([0-9,]+\.?[0-9]*)',
            r'\$([0-9,]+\.?[0-9]*)',
            r'precio[^$]*\$([0-9,]+\.?[0-9]*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Tomar el último precio encontrado (más reciente)
                    price_str = matches[-1].replace(',', '')
                    price = float(price_str)
                    # Validar que sea un precio razonable para ETH
                    if 3000 <= price <= 6000:
                        return price
                except ValueError:
                    continue
        
        return None
    
    def _generate_final_recommendation(self, enhanced_results: Dict, setup_detection: Dict) -> Dict:
        """Generar recomendación final integrando enhanced signal + setup detection"""
        
        if (enhanced_results.get("status") == "error" or 
            setup_detection.get("status") == "error"):
            return {
                "status": "error",
                "message": "Cannot generate final recommendation due to component errors"
            }
        
        # Extraer señales
        enhanced_action = enhanced_results.get("enhanced_action", "MONITOR")
        enhanced_score = enhanced_results.get("enhanced_score", 0)
        
        setup_viable = setup_detection.get("setup_viable", False)
        setup_type = setup_detection.get("detected_setup", "NO_CLEAR_SETUP")
        setup_confidence = setup_detection.get("setup_confidence", 0)
        
        # Lógica de recomendación final
        if setup_viable and setup_confidence >= 70:
            # Setup detector tiene alta confianza - usar su recomendación
            trading_params = setup_detection.get("trading_parameters", {})
            setup_action = trading_params.get("action", "MONITOR")
            
            final_action = setup_action
            confidence_source = "setup_detection"
            reasoning = f"High confidence {setup_type} setup detected ({setup_confidence}/100)"
            
            # Usar parámetros del setup detector
            final_params = trading_params
            
        elif enhanced_score >= 65:
            # Enhanced signal tiene buena confianza - usar enhanced
            final_action = enhanced_action
            confidence_source = "enhanced_signal"
            reasoning = f"Enhanced signal strong ({enhanced_score}/100)"
            
            # Usar parámetros básicos
            final_params = self._generate_basic_trading_params(enhanced_results)
            
        else:
            # Ambos débiles - no trade
            final_action = "WAIT"
            confidence_source = "conservative"
            reasoning = f"Both enhanced ({enhanced_score}/100) and setup ({setup_confidence}/100) below threshold"
            
            final_params = {"action": "WAIT", "reason": "No strong signal"}
        
        # Determinar confianza final
        if confidence_source == "setup_detection":
            final_confidence = setup_confidence
        elif confidence_source == "enhanced_signal":
            final_confidence = enhanced_score
        else:
            final_confidence = max(enhanced_score, setup_confidence) if setup_confidence > 0 else enhanced_score
        
        return {
            "status": "success",
            "final_action": final_action,
            "final_confidence": final_confidence,
            "confidence_source": confidence_source,
            "reasoning": reasoning,
            "trading_parameters": final_params,
            "signal_agreement": self._check_signal_agreement(enhanced_action, setup_detection),
            "risk_assessment": self._assess_overall_risk(enhanced_results, setup_detection)
        }
    
    def _generate_basic_trading_params(self, enhanced_results: Dict) -> Dict:
        """Generar parámetros básicos de trading desde enhanced signal"""
        
        enhanced_targets = enhanced_results.get("enhanced_targets", {})
        enhanced_action = enhanced_results.get("enhanced_action", "MONITOR")
        
        return {
            "action": enhanced_action,
            "source": "enhanced_signal",
            "entry_price": 0,  # Se llenaría en ejecución real
            "target_price": enhanced_targets.get("target", 0),
            "stop_loss": enhanced_targets.get("stop_loss", 0),
            "risk_reward": enhanced_targets.get("risk_reward", 0),
            "position_size_pct": 75,  # Default conservador
            "timeframe_hours": 12  # Default
        }
    
    def _check_signal_agreement(self, enhanced_action: str, setup_detection: Dict) -> Dict:
        """Verificar acuerdo entre señales"""
        
        if not setup_detection.get("setup_viable", False):
            return {
                "agreement": "no_setup",
                "description": "No viable setup detected for comparison"
            }
        
        trading_params = setup_detection.get("trading_parameters", {})
        setup_action = trading_params.get("action", "MONITOR")
        
        # Normalizar acciones para comparación
        enhanced_direction = "BUY" if "BUY" in enhanced_action else ("SELL" if "SELL" in enhanced_action else "NEUTRAL")
        setup_direction = "BUY" if "BUY" in setup_action else ("SELL" if "SELL" in setup_action else "NEUTRAL")
        
        if enhanced_direction == setup_direction:
            agreement = "strong_agreement"
            description = f"Both signals agree: {enhanced_direction}"
        elif enhanced_direction == "NEUTRAL" or setup_direction == "NEUTRAL":
            agreement = "partial_agreement"
            description = f"One signal neutral: Enhanced={enhanced_direction}, Setup={setup_direction}"
        else:
            agreement = "contradiction"
            description = f"Signals contradict: Enhanced={enhanced_direction}, Setup={setup_direction}"
        
        return {
            "agreement": agreement,
            "description": description,
            "enhanced_direction": enhanced_direction,
            "setup_direction": setup_direction
        }
    
    def _assess_overall_risk(self, enhanced_results: Dict, setup_detection: Dict) -> Dict:
        """Evaluar riesgo general del trade"""
        
        risk_factors = []
        risk_score = 0  # 0 = low risk, 100 = high risk
        
        # Factor 1: Enhanced score
        enhanced_score = enhanced_results.get("enhanced_score", 0)
        if enhanced_score < 50:
            risk_score += 30
            risk_factors.append("Low enhanced confidence")
        elif enhanced_score < 70:
            risk_score += 15
            risk_factors.append("Moderate enhanced confidence")
        
        # Factor 2: Setup viability
        if not setup_detection.get("setup_viable", False):
            risk_score += 25
            risk_factors.append("No clear setup detected")
        
        # Factor 3: Signal agreement
        agreement = self._check_signal_agreement(
            enhanced_results.get("enhanced_action", ""), setup_detection
        )
        if agreement["agreement"] == "contradiction":
            risk_score += 40
            risk_factors.append("Signal contradiction")
        elif agreement["agreement"] == "partial_agreement":
            risk_score += 20
            risk_factors.append("Partial signal agreement")
        
        # Factor 4: Volume contradiction
        fusion_breakdown = enhanced_results.get("fusion_breakdown", {})
        if fusion_breakdown.get("volume_boost", 0) < -10:
            risk_score += 15
            risk_factors.append("Volume contradicts base signal")
        
        # Clasificar riesgo
        if risk_score <= 25:
            risk_level = "LOW"
        elif risk_score <= 50:
            risk_level = "MODERATE"
        elif risk_score <= 75:
            risk_level = "HIGH"
        else:
            risk_level = "VERY_HIGH"
        
        return {
            "risk_score": min(100, risk_score),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendation": self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Obtener recomendación basada en nivel de riesgo"""
        
        recommendations = {
            "LOW": "Normal position size acceptable",
            "MODERATE": "Reduce position size by 25%",
            "HIGH": "Reduce position size by 50% or wait for better setup",
            "VERY_HIGH": "Avoid trade - wait for clearer signals"
        }
        
        return recommendations.get(risk_level, "Exercise caution")
    
    # Mantener todos los métodos existentes...
    def _run_base_analysis(self) -> Dict:
        """Ejecutar análisis del sistema base"""
        
        try:
            base_results = self.base_system.run_complete_analysis_with_ml()
            
            if base_results.get("status") == "success":
                base_score = base_results["trading_signal"]["confidence"]
                base_action = base_results["trading_signal"]["type"]
                
                print(f"✅ Sistema base completado: {base_action} (confianza: {base_score}%)")
                return base_results
            else:
                print(f"⚠️ Sistema base tuvo issues: {base_results.get('message', 'Unknown')}")
                return base_results
                
        except Exception as e:
            print(f"❌ Error en sistema base: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _extract_ohlcv_from_base_system(self) -> pd.DataFrame:
        """Extraer datos OHLCV del sistema base para análisis de volumen"""
        
        try:
            print("📊 Descargando datos frescos para análisis de volumen...")
            
            # Usar el data downloader del sistema base
            from data.data_downloader import DataDownloader
            
            downloader = DataDownloader()
            
            # Descargar últimas 3 horas de datos 1m para volume analysis
            symbol_api = self.symbol  # Ya está en formato correcto
            ohlcv_data = downloader.download_data(symbol_api, "1m", hours=3)
            
            if ohlcv_data is not None and not ohlcv_data.empty:
                print(f"✅ Datos OHLCV obtenidos: {len(ohlcv_data)} velas")
                print(f"📈 Rango temporal: {ohlcv_data.index[0]} a {ohlcv_data.index[-1]}")
                return ohlcv_data
            else:
                print("⚠️ No se pudieron obtener datos OHLCV, usando datos simulados...")
                return self._generate_fallback_data()
                
        except Exception as e:
            print(f"⚠️ Error obteniendo datos OHLCV: {str(e)}")
            print("🔄 Usando datos simulados como fallback...")
            return self._generate_fallback_data()
    
    def _generate_fallback_data(self) -> pd.DataFrame:
        """Generar datos simulados como fallback"""
        
        dates = pd.date_range(start='2025-10-05 09:31', periods=180, freq='1min')
        
        # Simular datos realistas basados en ETH
        np.random.seed(int(datetime.datetime.now().timestamp()) % 1000)  # Seed variable
        
        base_price = 4538  # Precio base realista para ETH actual
        data = []
        
        for i, date in enumerate(dates):
            # Precio con micro-tendencia y ruido realista
            trend = i * 0.03  # Tendencia muy sutil
            noise = np.random.normal(0, 6)  # Ruido realista
            price = base_price + trend + noise
            
            # OHLCV simulado realista
            open_price = price + np.random.normal(0, 1.2)
            high_price = price + abs(np.random.normal(1.5, 2.5))
            low_price = price - abs(np.random.normal(1.5, 2.5))
            close_price = price + np.random.normal(0, 1.2)
            
            # Volumen con patrones realistas (simulando volume drying up actual)
            base_volume = 1100  # Volumen base ETH típico
            
            # Simular diferentes fases de volumen
            if i > 150:  # Últimas 30 velas: volume drying up (como en realidad)
                volume = base_volume * (0.3 + np.random.random() * 0.25)
            elif i > 120 and i < 140:  # Spike de volumen en el medio
                volume = base_volume * (1.6 + np.random.random() * 1.0)
            else:  # Volumen normal con variación
                volume = base_volume * (0.6 + np.random.random() * 0.5)
            
            data.append({
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 0)
            })
        
        df = pd.DataFrame(data, index=dates)
        
        print(f"📊 Datos simulados generados: {len(df)} velas")
        print(f"💰 Rango de precios: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"📈 Rango de volumen: {df['volume'].min():.0f} - {df['volume'].max():.0f}")
        
        return df
    
    def _run_volume_analysis(self, ohlcv_data: pd.DataFrame) -> Dict:
        """Ejecutar análisis de volumen"""
        
        try:
            volume_results = self.volume_analyzer.analyze_volume_context(ohlcv_data)
            
            if volume_results.get("status") != "error":
                volume_score = volume_results["confirmation_score"]["score"]
                volume_level = volume_results["confirmation_score"]["level"]
                
                print(f"✅ Volume analysis completado: {volume_level} ({volume_score}/100)")
                return volume_results
            else:
                print(f"❌ Error en volume analysis: {volume_results.get('message', 'Unknown')}")
                return volume_results
                
        except Exception as e:
            print(f"❌ Error en volume analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _fuse_signals(self, base_results: Dict, volume_results: Dict) -> Dict:
        """Fusionar señales del sistema base y volume analysis"""
        
        if base_results.get("status") == "error" or volume_results.get("status") == "error":
            return {
                "status": "error",
                "message": "Cannot fuse signals due to component errors"
            }
        
        # Extraer scores base
        base_score = base_results["trading_signal"]["confidence"]
        base_action = base_results["trading_signal"]["type"]
        
        # Extraer scores de volumen
        volume_score = volume_results["confirmation_score"]["score"]
        volume_level = volume_results["confirmation_score"]["level"]
        
        # Algoritmo de fusion
        enhanced_score = self._calculate_enhanced_score(
            base_score, volume_score, base_action, volume_results
        )
        
        # Determinar acción mejorada
        enhanced_action = self._determine_enhanced_action(
            enhanced_score, base_action, volume_results
        )
        
        # Calcular target y stop mejorados
        enhanced_targets = self._calculate_enhanced_targets(
            base_results, volume_results, enhanced_score
        )
        
        return {
            "status": "success",
            "enhanced_score": enhanced_score,
            "enhanced_action": enhanced_action,
            "enhanced_targets": enhanced_targets,
            
            # Breakdown de la fusion
            "fusion_breakdown": {
                "base_score": base_score,
                "volume_score": volume_score,
                "volume_boost": enhanced_score - base_score,
                "fusion_weights": self.fusion_weights
            },
            
            # Factores de confirmación
            "confirmation_factors": self._extract_confirmation_factors(volume_results),
            
            # Interpretación
            "fusion_reasoning": self._generate_fusion_reasoning(
                base_score, volume_score, enhanced_score, base_action, enhanced_action
            )
        }
    
    def _calculate_enhanced_score(self, base_score: int, volume_score: int, 
                                base_action: str, volume_results: Dict) -> int:
        """Calcular score mejorado usando fusion algorithm"""
        
        # Score base ponderado
        weighted_base = base_score * self.fusion_weights["base_system"]
        weighted_volume = volume_score * self.fusion_weights["volume_layer"]
        
        # Score fusion básico
        fusion_score = weighted_base + weighted_volume
        
        # Bonificaciones por confirmación
        confirmation_bonus = 0
        
        # Volume confirmation bonus
        if volume_score >= 70:
            confirmation_bonus += 8  # Strong volume confirmation
        elif volume_score >= 50:
            confirmation_bonus += 4  # Moderate volume confirmation
        elif volume_score < 30:
            confirmation_bonus -= 5  # Volume contradicts
        
        # Breakout potential bonus
        breakout_potential = volume_results["breakout_potential"]["potential_score"]
        if breakout_potential >= 60:
            confirmation_bonus += 6
        elif breakout_potential >= 40:
            confirmation_bonus += 3
        
        # Volume trend alignment bonus
        volume_trend = volume_results["volume_trend"]["direction"]
        if base_action in ["BUY", "STRONG_BUY", "WEAK_BUY"]:
            if volume_trend in ["increasing_strong", "increasing_moderate"]:
                confirmation_bonus += 5  # Volume supports bullish signal
            elif volume_trend == "decreasing_strong":
                confirmation_bonus -= 8  # Volume contradicts bullish signal
        elif base_action in ["SELL", "STRONG_SELL", "WEAK_SELL"]:
            if volume_trend in ["decreasing_strong", "decreasing_moderate"]:
                confirmation_bonus += 5  # Volume supports bearish signal
            elif volume_trend == "increasing_strong":
                confirmation_bonus -= 8  # Volume contradicts bearish signal
        
        # Volume-price relationship bonus
        vp_relationship = volume_results["volume_price_relationship"]["relationship"]
        if base_action in ["BUY", "STRONG_BUY", "WEAK_BUY"] and vp_relationship == "bullish_confirmation":
            confirmation_bonus += 4
        elif base_action in ["SELL", "STRONG_SELL", "WEAK_SELL"] and vp_relationship == "bearish_confirmation":
            confirmation_bonus += 4
        
        # Score final
        enhanced_score = fusion_score + confirmation_bonus
        
        # Normalizar (0-100)
        enhanced_score = max(0, min(100, int(enhanced_score)))
        
        return enhanced_score
    
    def _determine_enhanced_action(self, enhanced_score: int, base_action: str, 
                                 volume_results: Dict) -> str:
        """Determinar acción mejorada basada en score y análisis"""
        
        # Mapeo base
        if enhanced_score >= 85:
            action_prefix = "STRONG_"
        elif enhanced_score >= 65:
            action_prefix = ""
        elif enhanced_score >= 45:
            action_prefix = "WEAK_"
        else:
            action_prefix = "VERY_WEAK_"
        
        # Extraer dirección base
        if "BUY" in base_action:
            base_direction = "BUY"
        elif "SELL" in base_action:
            base_direction = "SELL"
        else:
            base_direction = "MONITOR"
        
        # Ajustes por volume analysis
        volume_score = volume_results["confirmation_score"]["score"]
        
        # Downgrade si volume no confirma
        if volume_score < 25 and action_prefix == "STRONG_":
            action_prefix = ""
        elif volume_score < 15 and action_prefix == "":
            action_prefix = "WEAK_"
        
        # Upgrade si volume confirma fuertemente
        breakout_potential = volume_results["breakout_potential"]["potential_score"]
        if volume_score >= 80 and breakout_potential >= 70 and action_prefix == "":
            action_prefix = "STRONG_"
        
        enhanced_action = action_prefix + base_direction
        
        return enhanced_action
    
    def _calculate_enhanced_targets(self, base_results: Dict, volume_results: Dict, 
                                  enhanced_score: int) -> Dict:
        """Calcular targets mejorados"""
        
        # Targets base
        base_signal = base_results["trading_signal"]
        base_target = base_signal.get("target", 0)
        base_stop = base_signal.get("stop_loss", 0)
        
        # Ajustes por volume analysis
        volume_score = volume_results["confirmation_score"]["score"]
        breakout_potential = volume_results["breakout_potential"]["potential_score"]
        
        # Multiplicador de target basado en volume
        if volume_score >= 80 and breakout_potential >= 70:
            target_multiplier = 1.3  # Volume muy confirmativo
        elif volume_score >= 60 and breakout_potential >= 50:
            target_multiplier = 1.15  # Volume moderadamente confirmativo
        elif volume_score < 30:
            target_multiplier = 0.8  # Volume no confirmativo
        else:
            target_multiplier = 1.0  # Sin ajuste
        
        # Targets ajustados
        enhanced_target = base_target * target_multiplier if base_target > 0 else base_target
        enhanced_stop = base_stop  # Stop loss se mantiene por seguridad
        
        # R:R ratio
        if enhanced_target > 0 and enhanced_stop > 0:
            risk = abs(enhanced_target - enhanced_stop) * 0.5  # Estimado
            reward = abs(enhanced_target - enhanced_stop) * 0.5  # Estimado
            rr_ratio = reward / risk if risk > 0 else 0
        else:
            rr_ratio = base_signal.get("risk_reward", 0)
        
        return {
            "target": round(enhanced_target, 2),
            "stop_loss": round(enhanced_stop, 2),
            "risk_reward": round(rr_ratio, 2),
            "target_multiplier": target_multiplier,
            "volume_influence": {
                "volume_score": volume_score,
                "breakout_potential": breakout_potential,
                "adjustment_reason": self._get_target_adjustment_reason(target_multiplier)
            }
        }
    
    def _extract_confirmation_factors(self, volume_results: Dict) -> List[str]:
        """Extraer factores de confirmación del análisis de volumen"""
        
        factors = []
        
        # Volume trend
        vol_trend = volume_results["volume_trend"]
        factors.append(f"Volume trend: {vol_trend['direction']} ({vol_trend['change_pct']:+.1f}%)")
        
        # Relative volume
        rel_vol = volume_results["relative_volume"]
        factors.append(f"Relative volume: {rel_vol['average_ratio']:.1f}x ({rel_vol['general_status']})")
        
        # Breakout potential
        breakout = volume_results["breakout_potential"]
        factors.append(f"Breakout potential: {breakout['potential_level']} ({breakout['potential_score']}/100)")
        
        # A/D Line
        ad = volume_results["accumulation_distribution"]
        factors.append(f"A/D trend: {ad['trend']} (slope: {ad['slope']:+.1f})")
        
        # Patterns
        patterns = volume_results["patterns"]
        detected_patterns = [name for name, data in patterns.items() if data.get("detected", False)]
        if detected_patterns:
            factors.append(f"Patterns: {', '.join(detected_patterns)}")
        
        return factors
    
    def _generate_fusion_reasoning(self, base_score: int, volume_score: int, 
                                 enhanced_score: int, base_action: str, enhanced_action: str) -> List[str]:
        """Generar razonamiento de la fusion"""
        
        reasoning = []
        
        # Score fusion
        score_change = enhanced_score - base_score
        if score_change > 5:
            reasoning.append(f"Volume analysis boosted signal by +{score_change} points")
        elif score_change < -5:
            reasoning.append(f"Volume analysis weakened signal by {score_change} points")
        else:
            reasoning.append(f"Volume analysis provided neutral confirmation ({score_change:+} points)")
        
        # Action change
        if enhanced_action != base_action:
            reasoning.append(f"Action upgraded from {base_action} to {enhanced_action}")
        else:
            reasoning.append(f"Action maintained as {enhanced_action}")
        
        # Volume quality
        if volume_score >= 70:
            reasoning.append("Strong volume confirmation supports the signal")
        elif volume_score >= 50:
            reasoning.append("Moderate volume confirmation provides some support")
        elif volume_score < 30:
            reasoning.append("Weak volume raises concerns about signal strength")
        
        return reasoning
    
    def _get_target_adjustment_reason(self, multiplier: float) -> str:
        """Obtener razón del ajuste de target"""
        
        if multiplier > 1.2:
            return "Strong volume confirmation allows for extended targets"
        elif multiplier > 1.05:
            return "Moderate volume confirmation supports slightly higher targets"
        elif multiplier < 0.9:
            return "Weak volume suggests conservative targets"
        else:
            return "Volume analysis suggests maintaining original targets"
    
    def _generate_enhanced_interpretation(self, enhanced_results: Dict) -> str:
        """Generar interpretación mejorada"""
        
        if enhanced_results.get("status") == "error":
            return "Enhanced analysis failed due to component errors."
        
        enhanced_score = enhanced_results["enhanced_score"]
        enhanced_action = enhanced_results["enhanced_action"]
        fusion_breakdown = enhanced_results["fusion_breakdown"]
        
        # Interpretación base
        interpretation = f"Enhanced signal: {enhanced_action} (confidence: {enhanced_score}/100). "
        
        # Breakdown
        volume_boost = fusion_breakdown["volume_boost"]
        if volume_boost > 0:
            interpretation += f"Volume analysis improved signal by +{volume_boost} points. "
        elif volume_boost < 0:
            interpretation += f"Volume analysis reduced signal by {volume_boost} points. "
        else:
            interpretation += "Volume analysis provided neutral confirmation. "
        
        # Factores de confirmación
        confirmation_factors = enhanced_results["confirmation_factors"]
        if len(confirmation_factors) >= 2:
            interpretation += f"Key factors: {confirmation_factors[0]}, {confirmation_factors[1]}."
        
        return interpretation
    
    def _calculate_performance_metrics(self, base_results: Dict, volume_results: Dict, 
                                     enhanced_results: Dict) -> Dict:
        """Calcular métricas de performance del sistema mejorado"""
        
        if (base_results.get("status") == "error" or 
            volume_results.get("status") == "error" or 
            enhanced_results.get("status") == "error"):
            return {"status": "error", "message": "Cannot calculate metrics due to component errors"}
        
        base_score = base_results["trading_signal"]["confidence"]
        volume_score = volume_results["confirmation_score"]["score"]
        enhanced_score = enhanced_results["enhanced_score"]
        
        return {
            "score_improvement": enhanced_score - base_score,
            "score_improvement_pct": ((enhanced_score - base_score) / base_score * 100) if base_score > 0 else 0,
            "volume_contribution": volume_score * self.fusion_weights["volume_layer"],
            "base_contribution": base_score * self.fusion_weights["base_system"],
            "fusion_effectiveness": min(100, abs(enhanced_score - base_score) * 10),  # 0-100 scale
            "component_scores": {
                "base": base_score,
                "volume": volume_score,
                "enhanced": enhanced_score
            }
        }
    
    def _save_complete_enhanced_log(self, results: Dict):
        """Guardar log del análisis completo mejorado"""
        
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.enhanced_logs_dir, f"complete_enhanced_analysis_{self.symbol}_{timestamp_str}.json")
        
        try:
            def json_serializer(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif pd.api.types.is_scalar(obj) and pd.isna(obj):
                    return None
                elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
                    return str(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                else:
                    return str(obj)
            
            with open(log_file, 'w') as f:
                json.dump(results, f, indent=2, default=json_serializer)
            print(f"💾 Log completo mejorado guardado: {log_file}")
            
        except Exception as e:
            print(f"⚠️ Error guardando log completo mejorado: {str(e)}")
    
    def _display_complete_enhanced_summary(self, results: Dict):
        """Mostrar resumen del análisis completo mejorado"""
        
        print(f"\n🚀 RESUMEN ANÁLISIS COMPLETO MEJORADO:")
        print("=" * 75)
        
        if results.get("enhanced_signal", {}).get("status") == "error":
            print("❌ Error en análisis mejorado")
            return
        
        # Componentes principales
        enhanced_signal = results["enhanced_signal"]
        setup_detection = results.get("setup_detection", {})
        final_recommendation = results.get("final_recommendation", {})
        performance = results["performance_metrics"]
        
        # SECCIÓN 1: Enhanced Signal
        print(f"📊 ENHANCED SIGNAL:")
        print(f"   🎯 Señal: {enhanced_signal['enhanced_action']}")
        print(f"   📈 Score: {enhanced_signal['enhanced_score']}/100")
        
        fusion_breakdown = enhanced_signal["fusion_breakdown"]
        print(f"   📋 Breakdown: Base {fusion_breakdown['base_score']} + Volume {fusion_breakdown['volume_score']} = {enhanced_signal['enhanced_score']} ({fusion_breakdown['volume_boost']:+})")
        
        # SECCIÓN 2: Setup Detection
        print(f"\n🎯 SETUP DETECTION:")
        if setup_detection.get("setup_viable", False):
            detected_setup = setup_detection["detected_setup"]
            setup_confidence = setup_detection["setup_confidence"]
            trading_params = setup_detection.get("trading_parameters", {})
            
            print(f"   ✅ Setup: {detected_setup} ({setup_confidence}/100)")
            print(f"   🎯 Acción: {trading_params.get('action', 'N/A')}")
            print(f"   💰 Entry: ${trading_params.get('entry_price', 0):.2f}")
            print(f"   🎯 Target: ${trading_params.get('target_price', 0):.2f} ({trading_params.get('target_return_pct', 0):+.1f}%)")
            print(f"   🛡️ Stop: ${trading_params.get('stop_loss', 0):.2f} ({trading_params.get('risk_pct', 0):-.1f}%)")
            print(f"   ⚖️ R:R: {trading_params.get('risk_reward', 0):.1f}:1")
            print(f"   📊 Size: {trading_params.get('position_size_pct', 0)}%")
        else:
            print(f"   ❌ No viable setup detected")
            best_setup = setup_detection.get("detected_setup", "N/A")
            best_confidence = setup_detection.get("setup_confidence", 0)
            print(f"   📊 Best attempt: {best_setup} ({best_confidence}/100)")
        
        # SECCIÓN 3: Final Recommendation
        print(f"\n💡 RECOMENDACIÓN FINAL:")
        if final_recommendation.get("status") == "success":
            final_action = final_recommendation["final_action"]
            final_confidence = final_recommendation["final_confidence"]
            confidence_source = final_recommendation["confidence_source"]
            
            print(f"   🎯 Acción Final: {final_action}")
            print(f"   📊 Confianza: {final_confidence}/100 (fuente: {confidence_source})")
            print(f"   🧠 Razonamiento: {final_recommendation['reasoning']}")
            
            # Signal agreement
            signal_agreement = final_recommendation.get("signal_agreement", {})
            agreement_status = signal_agreement.get("agreement", "unknown")
            if agreement_status == "strong_agreement":
                print(f"   ✅ Acuerdo de señales: {signal_agreement['description']}")
            elif agreement_status == "contradiction":
                print(f"   ⚠️ Contradicción de señales: {signal_agreement['description']}")
            else:
                print(f"   📊 Señales: {signal_agreement.get('description', 'N/A')}")
            
            # Risk assessment
            risk_assessment = final_recommendation.get("risk_assessment", {})
            risk_level = risk_assessment.get("risk_level", "UNKNOWN")
            risk_score = risk_assessment.get("risk_score", 0)
            
            print(f"   🛡️ Riesgo: {risk_level} ({risk_score}/100)")
            print(f"   📋 Recomendación de riesgo: {risk_assessment.get('recommendation', 'N/A')}")
            
            if risk_assessment.get("risk_factors"):
                print(f"   ⚠️ Factores de riesgo: {', '.join(risk_assessment['risk_factors'][:3])}")
        else:
            print(f"   ❌ Error generando recomendación final")
        
        # SECCIÓN 4: Performance Metrics
        if performance.get("status") != "error":
            print(f"\n⚡ PERFORMANCE:")
            print(f"   📈 Mejora de score: {performance['score_improvement']:+.1f} puntos ({performance['score_improvement_pct']:+.1f}%)")
            print(f"   🔧 Efectividad de fusion: {performance['fusion_effectiveness']:.1f}/100")
            print(f"   📊 Contribuciones: Base {performance['base_contribution']:.1f} + Volume {performance['volume_contribution']:.1f}")
        
        # SECCIÓN 5: Factores de Confirmación
        confirmation_factors = enhanced_signal.get("confirmation_factors", [])
        if confirmation_factors:
            print(f"\n🔍 FACTORES CLAVE:")
            for i, factor in enumerate(confirmation_factors[:3], 1):
                print(f"   {i}. {factor}")
        
        print("=" * 75)

# Métodos de ejecución
def run_complete_enhanced_system():
    """Ejecutar sistema completo mejorado"""
    
    system = EnhancedTradingSystem("ETHUSD_PERP", capital=1000.0)
    results = system.run_complete_enhanced_analysis()
    return results

def test_complete_enhanced_system():
    """Test del sistema completo mejorado"""
    
    print("🧪 TESTING COMPLETE ENHANCED TRADING SYSTEM")
    print("=" * 75)
    
    result = run_complete_enhanced_system()
    
    status = result.get('final_recommendation', {}).get('status', 'unknown')
    print(f"\n✅ Test completado - Status: {status}")
    return result

if __name__ == "__main__":
    test_complete_enhanced_system()