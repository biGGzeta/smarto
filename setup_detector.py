#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup Detector - Sistema de Detección Automática de Setups
Detecta automáticamente tipos de setups basado en análisis mejorado
Autor: biGGzeta
Fecha: 2025-10-05 12:20:34 UTC
"""

import datetime
import json
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

class SetupDetector:
    """Detector automático de setups de trading basado en análisis mejorado"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP"):
        self.symbol = symbol
        self.setup_logs_dir = "logs/setups"
        os.makedirs(self.setup_logs_dir, exist_ok=True)
        
        # Configuración de setups
        self.setup_configs = {
            "COILING_BREAKOUT": {
                "min_enhanced_score": 80,
                "min_volume_score": 60,
                "min_breakout_potential": 60,
                "max_price_range_pct": 2.0,
                "target_multiplier": 2.0,
                "stop_multiplier": 1.0,
                "timeframe_hours": 12,
                "risk_reward_min": 2.0,
                "position_size_pct": 100
            },
            "ZONE_REVERSAL": {
                "min_enhanced_score": 50,
                "zone_threshold_high": 85,
                "zone_threshold_low": 15,
                "volume_contradiction_threshold": 20,
                "target_multiplier": 3.0,
                "stop_multiplier": 2.0,
                "timeframe_hours": 48,
                "risk_reward_min": 2.5,
                "position_size_pct": 75
            },
            "VOLUME_CONFIRMATION_SCALP": {
                "min_base_score": 70,
                "min_volume_score": 70,
                "min_enhanced_score": 75,
                "max_timeframe_hours": 6,
                "target_multiplier": 1.5,
                "stop_multiplier": 0.5,
                "timeframe_hours": 4,
                "risk_reward_min": 1.5,
                "position_size_pct": 75
            },
            "PROTECTED_FADE": {
                "min_base_score": 50,
                "max_enhanced_score": 50,
                "min_zone_position": 80,
                "volume_contradiction_required": True,
                "target_multiplier": 2.5,
                "stop_multiplier": 1.5,
                "timeframe_hours": 36,
                "risk_reward_min": 2.0,
                "position_size_pct": 50
            }
        }
        
        print(f"🎯 Setup Detector iniciado para {symbol}")
        print(f"📊 Configurados {len(self.setup_configs)} tipos de setups")
        print(f"💾 Logs en: {self.setup_logs_dir}")
    
    def detect_setup(self, enhanced_results: Dict, current_price: float = None) -> Dict:
        """Detectar tipo de setup basado en análisis mejorado"""
        
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        print(f"\n🔍 DETECTANDO SETUP - {timestamp.strftime('%H:%M:%S')} UTC")
        print("=" * 60)
        
        try:
            # Extraer métricas clave
            metrics = self._extract_metrics(enhanced_results)
            
            # Detectar cada tipo de setup
            setup_scores = {}
            setup_details = {}
            
            for setup_type in self.setup_configs.keys():
                score, details = self._evaluate_setup_type(setup_type, metrics, enhanced_results)
                setup_scores[setup_type] = score
                setup_details[setup_type] = details
            
            # Determinar mejor setup
            best_setup, best_score = self._select_best_setup(setup_scores)
            
            # Calcular parámetros de trading
            trading_params = self._calculate_trading_parameters(
                best_setup, enhanced_results, current_price
            )
            
            # Resultado final
            detection_result = {
                "timestamp": timestamp.isoformat(),
                "symbol": self.symbol,
                "current_price": current_price,
                
                # Detección principal
                "detected_setup": best_setup,
                "setup_confidence": best_score,
                "setup_viable": best_score >= 60,
                
                # Todos los scores
                "all_setup_scores": setup_scores,
                "setup_details": setup_details,
                
                # Parámetros de trading
                "trading_parameters": trading_params,
                
                # Métricas de entrada
                "input_metrics": metrics,
                
                # Recomendación final
                "recommendation": self._generate_recommendation(best_setup, best_score, trading_params)
            }
            
            # Guardar y mostrar
            self._save_setup_log(detection_result)
            self._display_setup_summary(detection_result)
            
            return detection_result
            
        except Exception as e:
            print(f"❌ Error detectando setup: {str(e)}")
            return {"status": "error", "message": str(e), "timestamp": timestamp.isoformat()}
    
    def _extract_metrics(self, enhanced_results: Dict) -> Dict:
        """Extraer métricas clave del análisis mejorado"""
        
        # Scores principales
        enhanced_signal = enhanced_results.get("enhanced_signal", {})
        base_analysis = enhanced_results.get("base_analysis", {})
        volume_analysis = enhanced_results.get("volume_analysis", {})
        
        # Extraer con fallbacks seguros
        base_score = base_analysis.get("trading_signal", {}).get("confidence", 0)
        volume_score = volume_analysis.get("confirmation_score", {}).get("score", 0)
        enhanced_score = enhanced_signal.get("enhanced_score", 0)
        
        # Posición en rango semanal (extraer del análisis base)
        weekly_analysis = self._extract_weekly_position(base_analysis)
        position_pct = weekly_analysis.get("position_percentile", 50)
        
        # Volume metrics
        volume_trend = volume_analysis.get("volume_trend", {})
        breakout_potential = volume_analysis.get("breakout_potential", {})
        relative_volume = volume_analysis.get("relative_volume", {})
        
        # Price action metrics
        price_range_pct = breakout_potential.get("price_range_pct", 5.0)
        
        return {
            "base_score": base_score,
            "volume_score": volume_score,
            "enhanced_score": enhanced_score,
            "position_percentile": position_pct,
            "volume_trend_direction": volume_trend.get("direction", "unknown"),
            "volume_change_pct": volume_trend.get("change_pct", 0),
            "breakout_potential_score": breakout_potential.get("potential_score", 0),
            "relative_volume_ratio": relative_volume.get("average_ratio", 1.0),
            "price_range_pct": price_range_pct,
            "volume_contradicts_base": self._check_volume_contradiction(enhanced_results)
        }
    
    def _extract_weekly_position(self, base_analysis: Dict) -> Dict:
        """Extraer posición en rango semanal del análisis base"""
        
        try:
            # Buscar en diferentes partes del análisis base
            # El análisis semanal suele estar en la pregunta 6
            
            # Método 1: Buscar en respuestas del bot
            bot_responses = base_analysis.get("bot_responses", {})
            for question, response in bot_responses.items():
                if "semanal" in question.lower() or "semana" in question.lower():
                    # Extraer porcentaje del texto de respuesta
                    response_text = response.get("response", "")
                    position_pct = self._parse_position_percentage(response_text)
                    if position_pct is not None:
                        return {"position_percentile": position_pct, "source": "bot_response"}
            
            # Método 2: Buscar en features ML
            ml_data = base_analysis.get("ml_pipeline", {})
            features = ml_data.get("features", {})
            if "weekly_position" in features:
                return {"position_percentile": features["weekly_position"], "source": "ml_features"}
            
            # Método 3: Fallback - asumir posición media
            return {"position_percentile": 50, "source": "fallback"}
            
        except Exception as e:
            print(f"⚠️ Error extrayendo posición semanal: {str(e)}")
            return {"position_percentile": 50, "source": "error_fallback"}
    
    def _parse_position_percentage(self, text: str) -> Optional[float]:
        """Parsear porcentaje de posición del texto"""
        
        import re
        
        # Buscar patrones como "87.4% del rango" o "actual $4,539 (87.4%"
        patterns = [
            r'(\d+\.?\d*)%\s+del\s+rango',
            r'\((\d+\.?\d*)%[^)]*\)',
            r'actual.*\((\d+\.?\d*)%',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _check_volume_contradiction(self, enhanced_results: Dict) -> bool:
        """Verificar si volume contradice la señal base"""
        
        enhanced_signal = enhanced_results.get("enhanced_signal", {})
        fusion_breakdown = enhanced_signal.get("fusion_breakdown", {})
        
        # Si volume boost es negativo, hay contradicción
        volume_boost = fusion_breakdown.get("volume_boost", 0)
        return volume_boost < -5
    
    def _evaluate_setup_type(self, setup_type: str, metrics: Dict, enhanced_results: Dict) -> Tuple[int, Dict]:
        """Evaluar un tipo específico de setup"""
        
        config = self.setup_configs[setup_type]
        score = 0
        details = {"criteria_met": [], "criteria_failed": []}
        
        if setup_type == "COILING_BREAKOUT":
            score, details = self._evaluate_coiling_breakout(config, metrics, enhanced_results)
        elif setup_type == "ZONE_REVERSAL":
            score, details = self._evaluate_zone_reversal(config, metrics, enhanced_results)
        elif setup_type == "VOLUME_CONFIRMATION_SCALP":
            score, details = self._evaluate_volume_scalp(config, metrics, enhanced_results)
        elif setup_type == "PROTECTED_FADE":
            score, details = self._evaluate_protected_fade(config, metrics, enhanced_results)
        
        return score, details
    
    def _evaluate_coiling_breakout(self, config: Dict, metrics: Dict, enhanced_results: Dict) -> Tuple[int, Dict]:
        """Evaluar setup de Coiling Breakout"""
        
        score = 0
        details = {"criteria_met": [], "criteria_failed": []}
        
        # Criterio 1: Enhanced score alto
        if metrics["enhanced_score"] >= config["min_enhanced_score"]:
            score += 30
            details["criteria_met"].append(f"Enhanced score {metrics['enhanced_score']} >= {config['min_enhanced_score']}")
        else:
            details["criteria_failed"].append(f"Enhanced score {metrics['enhanced_score']} < {config['min_enhanced_score']}")
        
        # Criterio 2: Volume score confirmatorio
        if metrics["volume_score"] >= config["min_volume_score"]:
            score += 25
            details["criteria_met"].append(f"Volume score {metrics['volume_score']} >= {config['min_volume_score']}")
        else:
            details["criteria_failed"].append(f"Volume score {metrics['volume_score']} < {config['min_volume_score']}")
        
        # Criterio 3: Breakout potential alto
        if metrics["breakout_potential_score"] >= config["min_breakout_potential"]:
            score += 25
            details["criteria_met"].append(f"Breakout potential {metrics['breakout_potential_score']} >= {config['min_breakout_potential']}")
        else:
            details["criteria_failed"].append(f"Breakout potential {metrics['breakout_potential_score']} < {config['min_breakout_potential']}")
        
        # Criterio 4: Rango compacto
        if metrics["price_range_pct"] <= config["max_price_range_pct"]:
            score += 20
            details["criteria_met"].append(f"Price range {metrics['price_range_pct']:.1f}% <= {config['max_price_range_pct']}%")
        else:
            details["criteria_failed"].append(f"Price range {metrics['price_range_pct']:.1f}% > {config['max_price_range_pct']}%")
        
        return score, details
    
    def _evaluate_zone_reversal(self, config: Dict, metrics: Dict, enhanced_results: Dict) -> Tuple[int, Dict]:
        """Evaluar setup de Zone Reversal"""
        
        score = 0
        details = {"criteria_met": [], "criteria_failed": []}
        
        # Criterio 1: Enhanced score mínimo
        if metrics["enhanced_score"] >= config["min_enhanced_score"]:
            score += 20
            details["criteria_met"].append(f"Enhanced score {metrics['enhanced_score']} >= {config['min_enhanced_score']}")
        else:
            details["criteria_failed"].append(f"Enhanced score {metrics['enhanced_score']} < {config['min_enhanced_score']}")
        
        # Criterio 2: Zona extrema
        in_extreme_zone = (metrics["position_percentile"] >= config["zone_threshold_high"] or 
                          metrics["position_percentile"] <= config["zone_threshold_low"])
        if in_extreme_zone:
            score += 40
            zone_type = "alta" if metrics["position_percentile"] >= config["zone_threshold_high"] else "baja"
            details["criteria_met"].append(f"En zona {zone_type} ({metrics['position_percentile']:.1f}%)")
        else:
            details["criteria_failed"].append(f"No en zona extrema ({metrics['position_percentile']:.1f}%)")
        
        # Criterio 3: Volume contradictorio (opcional pero bueno)
        if metrics["volume_contradicts_base"]:
            score += 25
            details["criteria_met"].append("Volume contradice señal base (reversión probable)")
        
        # Criterio 4: Volume bajo (para reversión)
        if metrics["volume_score"] <= config["volume_contradiction_threshold"]:
            score += 15
            details["criteria_met"].append(f"Volume bajo {metrics['volume_score']} <= {config['volume_contradiction_threshold']}")
        
        return score, details
    
    def _evaluate_volume_scalp(self, config: Dict, metrics: Dict, enhanced_results: Dict) -> Tuple[int, Dict]:
        """Evaluar setup de Volume Confirmation Scalp"""
        
        score = 0
        details = {"criteria_met": [], "criteria_failed": []}
        
        # Criterio 1: Base score alto
        if metrics["base_score"] >= config["min_base_score"]:
            score += 30
            details["criteria_met"].append(f"Base score {metrics['base_score']} >= {config['min_base_score']}")
        else:
            details["criteria_failed"].append(f"Base score {metrics['base_score']} < {config['min_base_score']}")
        
        # Criterio 2: Volume score alto
        if metrics["volume_score"] >= config["min_volume_score"]:
            score += 35
            details["criteria_met"].append(f"Volume score {metrics['volume_score']} >= {config['min_volume_score']}")
        else:
            details["criteria_failed"].append(f"Volume score {metrics['volume_score']} < {config['min_volume_score']}")
        
        # Criterio 3: Enhanced score alto
        if metrics["enhanced_score"] >= config["min_enhanced_score"]:
            score += 25
            details["criteria_met"].append(f"Enhanced score {metrics['enhanced_score']} >= {config['min_enhanced_score']}")
        else:
            details["criteria_failed"].append(f"Enhanced score {metrics['enhanced_score']} < {config['min_enhanced_score']}")
        
        # Criterio 4: No contradicción de volume
        if not metrics["volume_contradicts_base"]:
            score += 10
            details["criteria_met"].append("Volume confirma señal base")
        else:
            details["criteria_failed"].append("Volume contradice señal base")
        
        return score, details
    
    def _evaluate_protected_fade(self, config: Dict, metrics: Dict, enhanced_results: Dict) -> Tuple[int, Dict]:
        """Evaluar setup de Protected Fade"""
        
        score = 0
        details = {"criteria_met": [], "criteria_failed": []}
        
        # Criterio 1: Base score decente pero enhanced score bajo
        base_ok = metrics["base_score"] >= config["min_base_score"]
        enhanced_low = metrics["enhanced_score"] <= config["max_enhanced_score"]
        
        if base_ok and enhanced_low:
            score += 40
            details["criteria_met"].append(f"Base score {metrics['base_score']} >= {config['min_base_score']} but enhanced {metrics['enhanced_score']} <= {config['max_enhanced_score']}")
        else:
            details["criteria_failed"].append("Score differential not suitable for fade")
        
        # Criterio 2: Zona alta (fade típicamente contra zona alta)
        if metrics["position_percentile"] >= config["min_zone_position"]:
            score += 35
            details["criteria_met"].append(f"Zona alta {metrics['position_percentile']:.1f}% >= {config['min_zone_position']}%")
        else:
            details["criteria_failed"].append(f"No en zona alta suficiente ({metrics['position_percentile']:.1f}%)")
        
        # Criterio 3: Volume contradictorio (requerido)
        if metrics["volume_contradicts_base"]:
            score += 25
            details["criteria_met"].append("Volume contradice señal base (setup para fade)")
        else:
            details["criteria_failed"].append("Volume NO contradice (no setup para fade)")
        
        return score, details
    
    def _select_best_setup(self, setup_scores: Dict) -> Tuple[str, int]:
        """Seleccionar el mejor setup basado en scores"""
        
        # Filtrar setups viables (score >= 60)
        viable_setups = {k: v for k, v in setup_scores.items() if v >= 60}
        
        if not viable_setups:
            # No hay setups viables, retornar el mejor de todos
            best_setup = max(setup_scores.items(), key=lambda x: x[1])
            return best_setup[0], best_setup[1]
        
        # Retornar el mejor viable
        best_setup = max(viable_setups.items(), key=lambda x: x[1])
        return best_setup[0], best_setup[1]
    
    def _calculate_trading_parameters(self, setup_type: str, enhanced_results: Dict, 
                                    current_price: float = None) -> Dict:
        """Calcular parámetros de trading para el setup detectado"""
        
        if setup_type == "NO_CLEAR_SETUP":
            return {
                "action": "WAIT",
                "entry_price": 0,
                "target_price": 0,
                "stop_loss": 0,
                "position_size_pct": 0,
                "risk_reward": 0,
                "timeframe_hours": 0
            }
        
        config = self.setup_configs[setup_type]
        
        # Price base (usar precio actual o extraer del análisis)
        if current_price is None:
            current_price = self._extract_current_price(enhanced_results)
        
        # Determinar dirección del trade
        enhanced_signal = enhanced_results.get("enhanced_signal", {})
        base_action = enhanced_signal.get("enhanced_action", "MONITOR")
        
        # Para PROTECTED_FADE, invertir la dirección
        if setup_type == "PROTECTED_FADE":
            if "BUY" in base_action:
                trade_direction = "SELL"
            elif "SELL" in base_action:
                trade_direction = "BUY"
            else:
                trade_direction = "MONITOR"
        else:
            trade_direction = "BUY" if "BUY" in base_action else ("SELL" if "SELL" in base_action else "MONITOR")
        
        # Calcular niveles
        entry_price = current_price
        
        # Estimar rango para targets/stops
        price_range_pct = enhanced_results.get("volume_analysis", {}).get("breakout_potential", {}).get("price_range_pct", 2.0)
        base_range = current_price * (price_range_pct / 100)
        
        if trade_direction == "BUY":
            target_price = entry_price + (base_range * config["target_multiplier"])
            stop_loss = entry_price - (base_range * config["stop_multiplier"])
        elif trade_direction == "SELL":
            target_price = entry_price - (base_range * config["target_multiplier"])
            stop_loss = entry_price + (base_range * config["stop_multiplier"])
        else:
            target_price = entry_price
            stop_loss = entry_price
        
        # Risk/Reward
        if trade_direction != "MONITOR":
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 0
        
        return {
            "action": trade_direction,
            "setup_type": setup_type,
            "entry_price": round(entry_price, 2),
            "target_price": round(target_price, 2),
            "stop_loss": round(stop_loss, 2),
            "position_size_pct": config["position_size_pct"],
            "risk_reward": round(risk_reward, 2),
            "timeframe_hours": config["timeframe_hours"],
            "target_return_pct": round(abs(target_price - entry_price) / entry_price * 100, 2),
            "risk_pct": round(abs(entry_price - stop_loss) / entry_price * 100, 2)
        }
    
    def _extract_current_price(self, enhanced_results: Dict) -> float:
        """Extraer precio actual del análisis"""
        
        # Buscar precio en diferentes partes del análisis
        try:
            # Método 1: Desde volume analysis
            volume_data = enhanced_results.get("volume_analysis", {})
            if "current_price" in volume_data:
                return float(volume_data["current_price"])
            
            # Método 2: Desde base analysis (respuestas del bot)
            base_analysis = enhanced_results.get("base_analysis", {})
            bot_responses = base_analysis.get("bot_responses", {})
            
            for response in bot_responses.values():
                response_text = response.get("response", "")
                price = self._parse_price_from_text(response_text)
                if price:
                    return price
            
            # Método 3: Fallback precio ETH típico
            return 4540.0
            
        except Exception as e:
            print(f"⚠️ Error extrayendo precio actual: {str(e)}")
            return 4540.0
    
    def _parse_price_from_text(self, text: str) -> Optional[float]:
        """Parsear precio del texto"""
        
        import re
        
        # Buscar patrones como "$4,540.50" o "precio actual: $4,540"
        patterns = [
            r'\$([0-9,]+\.?[0-9]*)',
            r'precio.*\$([0-9,]+\.?[0-9]*)',
            r'actual.*\$([0-9,]+\.?[0-9]*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    # Tomar el último precio encontrado (más reciente)
                    price_str = matches[-1].replace(',', '')
                    return float(price_str)
                except ValueError:
                    continue
        
        return None
    
    def _generate_recommendation(self, setup_type: str, setup_confidence: int, 
                               trading_params: Dict) -> Dict:
        """Generar recomendación final"""
        
        if setup_confidence < 60:
            return {
                "action": "WAIT",
                "confidence": "LOW",
                "reason": f"No viable setup detected. Best: {setup_type} ({setup_confidence}/100)",
                "recommendation": "Monitor market for better setup development"
            }
        
        action = trading_params["action"]
        confidence_level = "HIGH" if setup_confidence >= 85 else ("MEDIUM" if setup_confidence >= 70 else "LOW")
        
        return {
            "action": action,
            "confidence": confidence_level,
            "setup_type": setup_type,
            "setup_confidence": setup_confidence,
            "reason": f"{setup_type} setup detected with {setup_confidence}/100 confidence",
            "recommendation": self._get_setup_recommendation(setup_type, trading_params)
        }
    
    def _get_setup_recommendation(self, setup_type: str, trading_params: Dict) -> str:
        """Obtener recomendación específica del setup"""
        
        recommendations = {
            "COILING_BREAKOUT": f"Execute {trading_params['action']} on breakout confirmation. Target: {trading_params['target_return_pct']:.1f}%, Risk: {trading_params['risk_pct']:.1f}%, R:R: {trading_params['risk_reward']:.1f}:1",
            "ZONE_REVERSAL": f"Execute {trading_params['action']} fade from extreme zone. Conservative size due to contrarian nature. Target: {trading_params['target_return_pct']:.1f}%",
            "VOLUME_CONFIRMATION_SCALP": f"Quick {trading_params['action']} scalp with strong volume confirmation. Short timeframe: {trading_params['timeframe_hours']}h",
            "PROTECTED_FADE": f"Protected fade setup - {trading_params['action']} against base signal. Volume contradicts, providing protection."
        }
        
        return recommendations.get(setup_type, f"Execute {trading_params['action']} according to {setup_type} parameters")
    
    def _save_setup_log(self, detection_result: Dict):
        """Guardar log de detección de setup"""
        
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.setup_logs_dir, f"setup_detection_{self.symbol}_{timestamp_str}.json")
        
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
                json.dump(detection_result, f, indent=2, default=json_serializer)
            print(f"💾 Log de setup guardado: {log_file}")
            
        except Exception as e:
            print(f"⚠️ Error guardando log de setup: {str(e)}")
    
    def _display_setup_summary(self, detection_result: Dict):
        """Mostrar resumen de detección de setup"""
        
        print(f"\n🎯 RESUMEN DETECCIÓN DE SETUP:")
        print("=" * 60)
        
        # Setup principal
        detected = detection_result["detected_setup"]
        confidence = detection_result["setup_confidence"]
        viable = detection_result["setup_viable"]
        
        print(f"🔍 Setup Detectado: {detected}")
        print(f"📊 Confianza: {confidence}/100 ({'VIABLE' if viable else 'NO VIABLE'})")
        
        # Scores de todos los setups
        print(f"\n📈 Scores de Todos los Setups:")
        for setup_type, score in detection_result["all_setup_scores"].items():
            status = "✅" if score >= 60 else ("⚠️" if score >= 40 else "❌")
            print(f"   {status} {setup_type}: {score}/100")
        
        # Parámetros de trading
        if viable:
            params = detection_result["trading_parameters"]
            print(f"\n🎯 Parámetros de Trading:")
            print(f"   Acción: {params['action']}")
            print(f"   Entry: ${params['entry_price']:.2f}")
            print(f"   Target: ${params['target_price']:.2f} (+{params['target_return_pct']:.1f}%)")
            print(f"   Stop: ${params['stop_loss']:.2f} (-{params['risk_pct']:.1f}%)")
            print(f"   R:R: {params['risk_reward']:.1f}:1")
            print(f"   Size: {params['position_size_pct']}%")
            print(f"   Timeframe: {params['timeframe_hours']}h")
        
        # Recomendación
        recommendation = detection_result["recommendation"]
        print(f"\n💡 Recomendación: {recommendation['action']} ({recommendation['confidence']})")
        print(f"📋 {recommendation['recommendation']}")
        
        print("=" * 60)

def test_setup_detector():
    """Test del detector de setups"""
    
    print("🧪 TESTING SETUP DETECTOR")
    print("=" * 60)
    
    # Crear datos de prueba simulando enhanced_results
    mock_enhanced_results = {
        "enhanced_signal": {
            "enhanced_score": 46,
            "enhanced_action": "WEAK_BUY",
            "fusion_breakdown": {
                "base_score": 60,
                "volume_score": 32,
                "volume_boost": -14
            }
        },
        "base_analysis": {
            "trading_signal": {"confidence": 60},
            "bot_responses": {
                "semanal": {"response": "semana: $3,966-$4,621 (16.54%); actual $4,539 (87.4% del rango, zona alta)"}
            }
        },
        "volume_analysis": {
            "confirmation_score": {"score": 32},
            "volume_trend": {"direction": "decreasing_strong", "change_pct": -28.4},
            "breakout_potential": {"potential_score": 50, "price_range_pct": 1.5},
            "relative_volume": {"average_ratio": 0.5}
        }
    }
    
    # Inicializar detector
    detector = SetupDetector("ETHUSD_PERP")
    
    # Detectar setup
    result = detector.detect_setup(mock_enhanced_results, current_price=4538.79)
    
    print(f"\n✅ Test completado")
    return result

if __name__ == "__main__":
    test_setup_detector()