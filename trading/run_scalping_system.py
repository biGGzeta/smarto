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

# Imports del sistema de scalping
from trading.scalping_monitor import ScalpingMonitor

class IntegratedTradingSystem:
    """Sistema integrado: Análisis + Scalping automático"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP", capital: float = 1000.0):
        self.symbol = symbol
        self.capital = capital
        
        # Sistemas principales
        self.analysis_system = AdaptiveTradingMLSystem(symbol, enable_ml=True, enable_logging=True)
        self.scalping_monitor = ScalpingMonitor(symbol, capital)
        
        # Configuración de timing
        self.analysis_interval_minutes = 30  # Ejecutar bot cada 30 min
        self.last_analysis_time = None
        
        print(f"🚀 Sistema Integrado Iniciado")
        print(f"📊 Símbolo: {symbol}")
        print(f"💰 Capital: ${capital:,.2f}")
        print(f"🔄 Análisis cada: {self.analysis_interval_minutes} min")
        print(f"⚡ Scalping cada: 15 min")
    
    def run_integrated_system(self):
        """Ejecutar sistema integrado completo"""
        
        print(f"\n🚀 INICIANDO SISTEMA INTEGRADO COMPLETO")
        print(f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 70)
        
        # Programar tareas
        schedule.every(self.analysis_interval_minutes).minutes.do(self._run_analysis)
        
        # Ejecutar análisis inicial
        self._run_analysis()
        
        # Esperar un momento para que se generen las señales
        print("⏳ Esperando 30 segundos para que se generen las señales...")
        time.sleep(30)
        
        # Iniciar monitor de scalping en hilo separado
        scalping_thread = threading.Thread(target=self.scalping_monitor.start_monitoring)
        scalping_thread.daemon = True
        scalping_thread.start()
        
        # Loop principal para análisis
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
                    print(f"📡 Señal para scalping: {external_signal['action']} - {external_signal['confidence']}%")
                    
                except Exception as e:
                    print(f"⚠️ Error en interpretación probabilística: {str(e)}")
            
            else:
                print(f"❌ Error en análisis: {results.get('message', 'Unknown error')}")
            
            self.last_analysis_time = timestamp
            
        except Exception as e:
            print(f"❌ Error ejecutando análisis: {str(e)}")
    
    def _shutdown(self):
        """Cerrar sistema integrado"""
        
        print(f"\n🛑 CERRANDO SISTEMA INTEGRADO")
        print("=" * 50)
        
        # El monitor de scalping se cerrará automáticamente
        print("✅ Sistema cerrado correctamente")
    
    def run_single_analysis_cycle(self):
        """Ejecutar un solo ciclo de análisis + scalping (para testing)"""
        
        print("🧪 TEST CICLO ÚNICO INTEGRADO")
        print("=" * 50)
        
        # 1. Ejecutar análisis
        print("📊 Ejecutando análisis...")
        self._run_analysis()
        
        # 2. Esperar y ejecutar scalping
        print("⏳ Esperando 10 segundos...")
        time.sleep(10)
        
        print("⚡ Ejecutando evaluación de scalping...")
        scalping_result = self.scalping_monitor.run_single_cycle()
        
        print("\n✅ Ciclo único completado")
        return scalping_result

def run_production_system():
    """Ejecutar sistema en producción"""
    
    system = IntegratedTradingSystem("ETHUSD_PERP", capital=1000.0)
    system.run_integrated_system()

def test_integrated_system():
    """Test del sistema integrado"""
    
    system = IntegratedTradingSystem("ETHUSD_PERP", capital=1000.0)
    result = system.run_single_analysis_cycle()
    return result

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
        run_production_system()
    else:
        print("🧪 INICIANDO EN MODO TEST")
        test_integrated_system()