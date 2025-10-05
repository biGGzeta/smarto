#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Integrado: Bot Smarto + Scalping Engine
Ejecuta análisis + trading automático
Autor: biGGzeta
Fecha: 2025-10-04
"""

import sys
import os
import time
import schedule
from datetime import datetime
import threading

# Agregar path para imports locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading'))

# Imports del sistema existente
from main_adaptive_ml import AdaptiveTradingMLSystem

class IntegratedTradingSystem:
    """Sistema integrado: Análisis + Scalping automático"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP", capital: float = 1000.0):
        self.symbol = symbol
        self.capital = capital
        
        # Sistema principal de análisis
        self.analysis_system = AdaptiveTradingMLSystem(symbol, enable_ml=True, enable_logging=True)
        
        # Configuración de timing
        self.analysis_interval_minutes = 30  # Ejecutar bot cada 30 min
        self.last_analysis_time = None
        
        print(f"🚀 Sistema Integrado Iniciado")
        print(f"📊 Símbolo: {symbol}")
        print(f"💰 Capital: ${capital:,.2f}")
        print(f"🔄 Análisis cada: {self.analysis_interval_minutes} min")
    
    def run_integrated_system(self):
        """Ejecutar sistema integrado completo"""
        
        print(f"\n🚀 INICIANDO SISTEMA INTEGRADO COMPLETO")
        print(f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 70)
        
        # Programar tareas
        schedule.every(self.analysis_interval_minutes).minutes.do(self._run_analysis)
        
        # Ejecutar análisis inicial
        self._run_analysis()
        
        # Loop principal para análisis continuo
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check cada minuto
                
        except KeyboardInterrupt:
            print("\n🛑 Sistema interrumpido por usuario")
            self._shutdown()
    
    def _run_analysis(self):
        """Ejecutar ciclo de análisis del bot"""
        
        timestamp = datetime.utcnow()
        print(f"\n📊 EJECUTANDO ANÁLISIS - {timestamp.strftime('%H:%M:%S')} UTC")
        print("-" * 50)
        
        try:
            # Ejecutar análisis completo del bot
            results = self.analysis_system.run_complete_analysis_with_ml()
            
            if results.get("status") == "success":
                print("✅ Análisis completado exitosamente")
                
                # Mostrar resumen ejecutivo
                print(f"📊 {results['executive_summary']}")
                
                # Mostrar señal de trading
                signal = results["trading_signal"]
                print(f"🚨 Señal: {signal['type']} - {signal['confidence']:.0f}%")
                
                # Ejecutar interpretación probabilística
                try:
                    from analysis.probabilistic_interpreter import ProbabilisticInterpreter
                    
                    interpreter = ProbabilisticInterpreter(self.symbol)
                    probability_results = interpreter.interpret_complete_analysis(results)
                    
                    # Mostrar escenario principal
                    scenarios = probability_results.get("scenarios", [])
                    if scenarios:
                        main_scenario = scenarios[0]
                        print(f"🎯 Escenario principal: {main_scenario['name']} ({main_scenario['probability']}%)")
                    
                    external_signal = probability_results["signal_for_external"]
                    print(f"📡 Señal para trading: {external_signal['action']} - {external_signal['confidence']}%")
                    
                    # Evaluar para scalping
                    self._evaluate_scalping_opportunity(external_signal, probability_results)
                    
                except Exception as e:
                    print(f"⚠️ Error en interpretación probabilística: {str(e)}")
            
            else:
                print(f"❌ Error en análisis: {results.get('message', 'Unknown error')}")
            
            self.last_analysis_time = timestamp
            
        except Exception as e:
            print(f"❌ Error ejecutando análisis: {str(e)}")
    
    def _evaluate_scalping_opportunity(self, external_signal: dict, probability_results: dict):
        """Evaluar oportunidad de scalping"""
        
        print(f"\n⚡ EVALUANDO OPORTUNIDAD DE SCALPING:")
        print("-" * 40)
        
        # Obtener interpretación del rango
        interpretations = probability_results.get('interpretations', {})
        range_signal = interpretations.get('range_signal', {})
        
        # Calcular score de scalping
        score = 0
        reasons = []
        
        # Factor 1: Coiling strength
        coiling_strength = range_signal.get('coiling_strength', 'low')
        if coiling_strength == 'high':
            score += 30
            reasons.append("✅ Coiling fuerte detectado (+30)")
        elif coiling_strength == 'medium':
            score += 15
            reasons.append("⚠️ Coiling moderado (+15)")
        
        # Factor 2: Breakout probability
        breakout_prob = range_signal.get('breakout_probability', 0)
        if breakout_prob > 75:
            score += 25
            reasons.append(f"✅ Alta probabilidad breakout ({breakout_prob}%) (+25)")
        elif breakout_prob > 60:
            score += 15
            reasons.append(f"⚠️ Probabilidad moderada ({breakout_prob}%) (+15)")
        
        # Factor 3: Timeframe
        timeframe = range_signal.get('time_horizon_hours', 24)
        if timeframe <= 4:
            score += 20
            reasons.append(f"✅ Timeframe ideal ({timeframe}h) (+20)")
        elif timeframe <= 6:
            score += 10
            reasons.append(f"⚠️ Timeframe aceptable ({timeframe}h) (+10)")
        
        # Factor 4: Acción del bot
        action = external_signal.get('action', 'MONITOR')
        confidence = external_signal.get('confidence', 0)
        
        if action in ['BUY', 'SELL'] and confidence > 70:
            score += 25
            reasons.append(f"✅ Señal direccional fuerte: {action} ({confidence}%) (+25)")
        elif action == 'MONITOR_CLOSE' and confidence > 60:
            score += 10
            reasons.append(f"⚠️ Monitoreo cercano recomendado ({confidence}%) (+10)")
        
        # Mostrar evaluación
        print(f"📊 Score de Scalping: {score}/100")
        for reason in reasons:
            print(f"   {reason}")
        
        # Decisión
        if score >= 70:
            print(f"🟢 OPORTUNIDAD DE SCALPING DETECTADA")
            self._log_scalping_opportunity(external_signal, range_signal, score, "ENTRY")
        elif score >= 50:
            print(f"🟡 OPORTUNIDAD MARGINAL - MONITOREAR")
            self._log_scalping_opportunity(external_signal, range_signal, score, "MONITOR")
        else:
            print(f"🔴 SIN OPORTUNIDAD DE SCALPING")
            self._log_scalping_opportunity(external_signal, range_signal, score, "PASS")
    
    def _log_scalping_opportunity(self, external_signal: dict, range_signal: dict, score: int, decision: str):
        """Guardar oportunidad de scalping"""
        
        timestamp = datetime.utcnow()
        
        opportunity = {
            "timestamp": timestamp.isoformat(),
            "symbol": self.symbol,
            "decision": decision,
            "score": score,
            "external_signal": external_signal,
            "range_signal": range_signal,
            "evaluation": {
                "coiling_strength": range_signal.get('coiling_strength', 'unknown'),
                "breakout_probability": range_signal.get('breakout_probability', 0),
                "time_horizon_hours": range_signal.get('time_horizon_hours', 0),
                "bot_action": external_signal.get('action', 'UNKNOWN'),
                "bot_confidence": external_signal.get('confidence', 0)
            }
        }
        
        # Guardar en archivo
        logs_dir = "logs/trading"
        os.makedirs(logs_dir, exist_ok=True)
        
        log_file = os.path.join(logs_dir, f"scalping_opportunities_{timestamp.strftime('%Y%m%d')}.json")
        
        try:
            # Leer oportunidades existentes
            opportunities = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    import json
                    existing_data = json.load(f)
                    opportunities = existing_data.get('opportunities', [])
            
            # Agregar nueva oportunidad
            opportunities.append(opportunity)
            
            # Guardar
            with open(log_file, 'w') as f:
                import json
                json.dump({
                    "date": timestamp.strftime('%Y-%m-%d'),
                    "symbol": self.symbol,
                    "total_opportunities": len(opportunities),
                    "opportunities": opportunities
                }, f, indent=2)
            
            print(f"💾 Oportunidad guardada en: {log_file}")
            
        except Exception as e:
            print(f"⚠️ Error guardando oportunidad: {str(e)}")
    
    def _shutdown(self):
        """Cerrar sistema integrado"""
        
        print(f"\n🛑 CERRANDO SISTEMA INTEGRADO")
        print("=" * 50)
        print("✅ Sistema cerrado correctamente")
    
    def run_single_cycle(self):
        """Ejecutar un solo ciclo (para testing)"""
        
        print("🧪 TEST CICLO ÚNICO INTEGRADO")
        print("=" * 50)
        
        # Ejecutar análisis
        self._run_analysis()
        
        print("\n✅ Ciclo único completado")

def run_production_system():
    """Ejecutar sistema en producción"""
    
    system = IntegratedTradingSystem("ETHUSD_PERP", capital=1000.0)
    system.run_integrated_system()

def test_integrated_system():
    """Test del sistema integrado"""
    
    system = IntegratedTradingSystem("ETHUSD_PERP", capital=1000.0)
    system.run_single_cycle()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema Integrado de Trading')
    parser.add_argument('--mode', choices=['test', 'production'], default='test', 
                       help='Modo de ejecución')
    parser.add_argument('--capital', type=float, default=1000.0, 
                       help='Capital inicial')
    
    args = parser.parse_args()
    
    if args.mode == 'production':
        print("🚀 INICIANDO EN MODO PRODUCCIÓN")
        print(f"💰 Capital: ${args.capital:,.2f}")
        run_production_system()
    else:
        print("🧪 INICIANDO EN MODO TEST")
        test_integrated_system()