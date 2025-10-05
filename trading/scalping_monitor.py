#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de Scalping en Tiempo Real
Ejecuta ciclos automáticos y monitorea performance
Autor: biGGzeta
Fecha: 2025-10-04
"""

import time
import schedule
from datetime import datetime, timedelta
import signal
import sys
from scalping_engine import ScalpingEngine
import json
import os

class ScalpingMonitor:
    """Monitor que ejecuta el motor de scalping automáticamente"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP", capital: float = 1000.0):
        self.symbol = symbol
        self.capital = capital
        self.engine = ScalpingEngine(symbol, capital)
        self.running = False
        self.cycle_count = 0
        
        # Configuración de monitoreo
        self.cycle_interval_minutes = 15  # Cada 15 minutos
        self.daily_reset_hour = 0  # Medianoche UTC
        
        # Logs
        self.monitor_logs_dir = "logs/monitor"
        os.makedirs(self.monitor_logs_dir, exist_ok=True)
        
        print(f"🔄 Monitor de Scalping iniciado")
        print(f"📊 Símbolo: {symbol}")
        print(f"💰 Capital: ${capital:,.2f}")
        print(f"⏰ Intervalo: {self.cycle_interval_minutes} minutos")
    
    def start_monitoring(self):
        """Iniciar monitoreo automático"""
        
        print(f"\n🚀 INICIANDO MONITOREO AUTOMÁTICO")
        print(f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 60)
        
        # Configurar tareas programadas
        schedule.every(self.cycle_interval_minutes).minutes.do(self._run_cycle)
        schedule.every().day.at(f"{self.daily_reset_hour:02d}:00").do(self._daily_reset)
        
        # Configurar manejo de señales para parada limpia
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        
        # Ejecutar primer ciclo inmediatamente
        self._run_cycle()
        
        # Loop principal
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(30)  # Check cada 30 segundos
                
        except KeyboardInterrupt:
            self._shutdown()
    
    def _run_cycle(self):
        """Ejecutar un ciclo de scalping"""
        
        self.cycle_count += 1
        timestamp = datetime.utcnow()
        
        print(f"\n🔄 CICLO #{self.cycle_count} - {timestamp.strftime('%H:%M:%S')} UTC")
        print("-" * 50)
        
        try:
            # Ejecutar ciclo del motor
            result = self.engine.run_scalping_cycle()
            
            # Procesar resultado
            self._process_cycle_result(result)
            
            # Mostrar resumen cada 4 ciclos (1 hora)
            if self.cycle_count % 4 == 0:
                self._show_hourly_summary()
            
        except Exception as e:
            print(f"❌ Error en ciclo #{self.cycle_count}: {str(e)}")
            self._save_error_log(e)
    
    def _process_cycle_result(self, result: Dict):
        """Procesar resultado del ciclo"""
        
        status = result.get('status', 'unknown')
        
        if status == 'success':
            # Mostrar información relevante
            if result.get('bot_signal_processed'):
                print("📊 Señal del bot procesada ✅")
            
            entry_eval = result.get('entry_evaluation')
            if entry_eval:
                action = entry_eval.get('action', 'UNKNOWN')
                score = entry_eval.get('score', 0)
                
                if action == 'SCALP_ENTRY':
                    print(f"⚡ ENTRADA EJECUTADA - Score: {score}/100")
                    trade_details = entry_eval.get('trade_details', {})
                    if trade_details:
                        print(f"   ID: {trade_details.get('id')}")
                        print(f"   R:R: {trade_details.get('rr_ratio', 0):.1f}:1")
                elif action == 'MONITOR_CLOSE':
                    print(f"👀 MONITOREO CERCANO - Score: {score}/100")
                else:
                    print(f"⏸️ SIN ACCIÓN - Score: {score}/100")
            
            # Mostrar trades activos
            active_trades = result.get('active_trades', 0)
            if active_trades > 0:
                print(f"🔄 Trades activos: {active_trades}")
            
            # Mostrar stats diarias
            daily_stats = result.get('daily_stats', {})
            if daily_stats.get('trades_taken', 0) > 0:
                print(f"📈 Trades hoy: {daily_stats['trades_taken']} | "
                      f"P&L: {daily_stats['total_pnl_pct']:+.2f}%")
        
        elif status == 'no_signal':
            print("⚠️ Sin señal del bot disponible")
        
        else:
            print(f"❌ Error en ciclo: {result.get('message', 'Unknown error')}")
    
    def _show_hourly_summary(self):
        """Mostrar resumen cada hora"""
        
        print(f"\n📊 RESUMEN HORARIO (Ciclos #{self.cycle_count-3} a #{self.cycle_count})")
        print("=" * 60)
        
        # Obtener reporte diario del motor
        daily_report = self.engine.get_daily_report()
        
        # Extraer líneas relevantes
        lines = daily_report.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['Total:', 'Win Rate:', 'Total %:', 'R:R promedio:']):
                print(line)
        
        print("=" * 60)
    
    def _daily_reset(self):
        """Reset diario a medianoche"""
        
        print(f"\n🌅 RESET DIARIO - {datetime.utcnow().strftime('%Y-%m-%d')} UTC")
        print("=" * 60)
        
        # Guardar reporte del día anterior
        yesterday_report = self.engine.get_daily_report()
        self._save_daily_report(yesterday_report)
        
        # Reset estadísticas
        self.engine.reset_daily_stats()
        self.cycle_count = 0
        
        print("✅ Reset diario completado")
    
    def _save_daily_report(self, report: str):
        """Guardar reporte diario"""
        
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")
        report_file = os.path.join(self.monitor_logs_dir, f"daily_report_{yesterday}.txt")
        
        try:
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"💾 Reporte diario guardado: {report_file}")
        except Exception as e:
            print(f"⚠️ Error guardando reporte: {str(e)}")
    
    def _save_error_log(self, error: Exception):
        """Guardar log de error"""
        
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        error_file = os.path.join(self.monitor_logs_dir, f"error_{timestamp_str}.json")
        
        error_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "cycle_number": self.cycle_count,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        try:
            with open(error_file, 'w') as f:
                json.dump(error_data, f, indent=2)
        except Exception:
            pass  # No fallar si no se puede guardar el log
    
    def _signal_handler(self, signum, frame):
        """Manejar señales de sistema para parada limpia"""
        print(f"\n🛑 Señal recibida: {signum}. Cerrando monitor...")
        self._shutdown()
    
    def _shutdown(self):
        """Cerrar monitor limpiamente"""
        
        print(f"\n🛑 CERRANDO MONITOR DE SCALPING")
        print("=" * 50)
        
        # Mostrar reporte final
        final_report = self.engine.get_daily_report()
        print(final_report)
        
        # Guardar estado final
        self._save_daily_report(final_report)
        
        self.running = False
        print("✅ Monitor cerrado correctamente")
        sys.exit(0)
    
    def run_single_cycle(self):
        """Ejecutar un solo ciclo (para testing)"""
        
        print("🔄 EJECUTANDO CICLO ÚNICO")
        result = self.engine.run_scalping_cycle()
        self._process_cycle_result(result)
        return result

def start_monitor():
    """Iniciar monitor de scalping"""
    
    monitor = ScalpingMonitor("ETHUSD_PERP", capital=1000.0)
    monitor.start_monitoring()

def test_single_cycle():
    """Test de un solo ciclo"""
    
    print("🧪 TEST CICLO ÚNICO")
    monitor = ScalpingMonitor("ETHUSD_PERP", capital=1000.0)
    result = monitor.run_single_cycle()
    return result

if __name__ == "__main__":
    # Para producción: start_monitor()
    # Para testing: test_single_cycle()
    test_single_cycle()