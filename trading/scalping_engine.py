#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor de Scalping Adaptativo
Integra señales del bot smarto para scalping de alta frecuencia
Autor: biGGzeta
Fecha: 2025-10-04
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
import time

class ScalpingEngine:
    """Motor de scalping que consume señales del bot smarto"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP", capital: float = 1000.0):
        self.symbol = symbol
        self.capital = capital
        self.position_size_pct = 1.5  # 1.5% del capital por trade
        self.max_daily_trades = 4
        self.max_time_in_trade = 6  # horas
        
        # Estado del sistema
        self.active_trades = []
        self.daily_stats = self._init_daily_stats()
        self.trade_history = []
        
        # Configuración de validación
        self.min_score_entry = 70
        self.min_coiling_strength = "medium"
        self.max_range_pct = 1.5
        self.min_breakout_prob = 75
        self.min_rr_ratio = 2.0
        
        # Directorios
        self.signals_dir = "logs/probabilities"
        self.trading_logs_dir = "logs/trading"
        os.makedirs(self.trading_logs_dir, exist_ok=True)
        
        print(f"⚡ Motor de Scalping iniciado para {symbol}")
        print(f"💰 Capital: ${capital:,.2f}")
        print(f"📊 Tamaño posición: {self.position_size_pct}%")
        print(f"🎯 Máximo trades diarios: {self.max_daily_trades}")
    
    def _init_daily_stats(self) -> Dict:
        """Inicializar estadísticas diarias"""
        return {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "trades_taken": 0,
            "trades_won": 0,
            "trades_lost": 0,
            "total_pnl_pct": 0.0,
            "total_pnl_usd": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "avg_time_in_trade": 0.0,
            "coiling_accuracy": 0.0,
            "avg_rr_realized": 0.0,
            "signals_processed": 0,
            "signals_acted": 0
        }
    
    def run_scalping_cycle(self) -> Dict[str, Any]:
        """Ejecutar un ciclo completo de scalping"""
        
        timestamp = datetime.utcnow()
        print(f"\n⚡ CICLO SCALPING - {timestamp.strftime('%H:%M:%S')} UTC")
        print("=" * 60)
        
        try:
            # 1. Leer señal más reciente del bot
            bot_signal = self._read_bot_signal()
            if not bot_signal:
                return {"status": "no_signal", "message": "Sin señal del bot disponible"}
            
            # 2. Validar posiciones activas
            self._update_active_trades()
            
            # 3. Evaluar nueva entrada si hay capacidad
            entry_result = None
            if self._can_take_new_trade():
                entry_result = self._evaluate_scalping_entry(bot_signal)
            
            # 4. Actualizar estadísticas
            self._update_daily_stats()
            
            # 5. Generar reporte
            cycle_result = {
                "status": "success",
                "timestamp": timestamp.isoformat(),
                "bot_signal_processed": bool(bot_signal),
                "active_trades": len(self.active_trades),
                "entry_evaluation": entry_result,
                "daily_stats": self.daily_stats.copy(),
                "can_trade": self._can_take_new_trade()
            }
            
            # 6. Guardar logs
            self._save_cycle_log(cycle_result)
            
            return cycle_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "timestamp": timestamp.isoformat(),
                "error": str(e),
                "active_trades": len(self.active_trades)
            }
            print(f"❌ Error en ciclo scalping: {str(e)}")
            return error_result
    
    def _read_bot_signal(self) -> Optional[Dict]:
        """Leer señal más reciente del bot smarto"""
        
        signal_file = os.path.join(self.signals_dir, f"signal_external_{self.symbol}.json")
        
        try:
            if not os.path.exists(signal_file):
                print(f"⚠️ Archivo de señal no encontrado: {signal_file}")
                return None
            
            with open(signal_file, 'r') as f:
                signal = json.load(f)
            
            # Verificar que la señal sea reciente (últimos 30 minutos)
            signal_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
            if datetime.utcnow().replace(tzinfo=signal_time.tzinfo) - signal_time > timedelta(minutes=30):
                print(f"⚠️ Señal muy antigua: {signal_time}")
                return None
            
            print(f"📊 Señal leída: {signal['action']} - {signal['confidence']}%")
            return signal
            
        except Exception as e:
            print(f"❌ Error leyendo señal: {str(e)}")
            return None
    
    def _can_take_new_trade(self) -> bool:
        """Verificar si se puede tomar un nuevo trade"""
        
        # Verificar límite diario
        if self.daily_stats["trades_taken"] >= self.max_daily_trades:
            return False
        
        # Verificar trades activos (máximo 1 para scalping)
        if len(self.active_trades) >= 1:
            return False
        
        # Verificar horario (evitar fines de semana)
        current_time = datetime.utcnow()
        weekday = current_time.weekday()
        if weekday >= 5:  # Sábado o domingo
            return False
        
        return True
    
    def _evaluate_scalping_entry(self, bot_signal: Dict) -> Dict[str, Any]:
        """Evaluar si entrar en scalping basado en señal del bot"""
        
        print(f"\n🔍 EVALUANDO ENTRADA SCALPING:")
        print("-" * 40)
        
        # Incrementar contador de señales procesadas
        self.daily_stats["signals_processed"] += 1
        
        # Leer detalles de la interpretación probabilística
        interpretation = self._read_detailed_interpretation()
        
        if not interpretation:
            return {
                "action": "PASS",
                "reason": "Sin interpretación detallada disponible",
                "score": 0
            }
        
        # Calcular score de entrada
        score, reasons = self._calculate_scalping_score(bot_signal, interpretation)
        
        print(f"📊 Score calculado: {score}/100")
        for reason in reasons:
            print(f"   {reason}")
        
        # Decisión de entrada
        if score >= self.min_score_entry:
            trade = self._create_scalping_trade(bot_signal, interpretation, score)
            if trade:
                self.active_trades.append(trade)
                self.daily_stats["trades_taken"] += 1
                self.daily_stats["signals_acted"] += 1
                
                print(f"✅ ENTRADA EJECUTADA:")
                print(f"   Precio: ${trade['entry_price']:.2f}")
                print(f"   Stop: ${trade['stop_loss']:.2f}")
                print(f"   Target: ${trade['take_profit']:.2f}")
                print(f"   R:R: {trade['rr_ratio']:.1f}:1")
                
                return {
                    "action": "SCALP_ENTRY",
                    "trade_id": trade['id'],
                    "score": score,
                    "reasons": reasons,
                    "trade_details": trade
                }
        
        elif score >= 50:
            return {
                "action": "MONITOR_CLOSE",
                "score": score,
                "reasons": reasons,
                "message": "Señal interesante pero no cumple criterios de entrada"
            }
        
        else:
            return {
                "action": "PASS",
                "score": score,
                "reasons": reasons,
                "message": "Señal no apta para scalping"
            }
    
    def _read_detailed_interpretation(self) -> Optional[Dict]:
        """Leer interpretación detallada más reciente"""
        
        try:
            # Buscar archivo más reciente de interpretación
            files = [f for f in os.listdir(self.signals_dir) 
                    if f.startswith(f"interpretation_{self.symbol}_") and f.endswith(".json")]
            
            if not files:
                return None
            
            # Ordenar por timestamp en el nombre del archivo
            latest_file = sorted(files)[-1]
            filepath = os.path.join(self.signals_dir, latest_file)
            
            with open(filepath, 'r') as f:
                interpretation = json.load(f)
            
            return interpretation
            
        except Exception as e:
            print(f"⚠️ Error leyendo interpretación detallada: {str(e)}")
            return None
    
    def _calculate_scalping_score(self, bot_signal: Dict, interpretation: Dict) -> Tuple[int, List[str]]:
        """Calcular score para entrada de scalping"""
        
        score = 0
        reasons = []
        
        # FACTOR 1: Coiling Detection (30 puntos)
        range_signal = interpretation.get('interpretations', {}).get('range_signal', {})
        coiling_strength = range_signal.get('coiling_strength', 'low')
        
        if coiling_strength == 'high':
            score += 30
            reasons.append("✅ Coiling fuerte detectado (+30)")
        elif coiling_strength == 'medium':
            score += 15
            reasons.append("⚠️ Coiling moderado (+15)")
        else:
            reasons.append("❌ Sin coiling significativo (0)")
        
        # FACTOR 2: Rango Compacto (25 puntos)
        range_pct = range_signal.get('expected_magnitude_pct', 0) / 2.8  # Estimado inverso
        
        if range_pct < 0.6:  # <0.6% muy compacto
            score += 25
            reasons.append(f"✅ Rango muy compacto (~{range_pct:.1f}%) (+25)")
        elif range_pct < 1.0:
            score += 15
            reasons.append(f"✅ Rango compacto (~{range_pct:.1f}%) (+15)")
        elif range_pct < 1.5:
            score += 8
            reasons.append(f"⚠️ Rango moderado (~{range_pct:.1f}%) (+8)")
        else:
            reasons.append(f"❌ Rango muy amplio (~{range_pct:.1f}%) (0)")
        
        # FACTOR 3: Probabilidad Breakout (20 puntos)
        breakout_prob = range_signal.get('breakout_probability', 0)
        
        if breakout_prob > 80:
            score += 20
            reasons.append(f"✅ Alta probabilidad breakout ({breakout_prob}%) (+20)")
        elif breakout_prob > 70:
            score += 12
            reasons.append(f"✅ Buena probabilidad breakout ({breakout_prob}%) (+12)")
        elif breakout_prob > 60:
            score += 6
            reasons.append(f"⚠️ Probabilidad moderada ({breakout_prob}%) (+6)")
        else:
            reasons.append(f"❌ Baja probabilidad breakout ({breakout_prob}%) (0)")
        
        # FACTOR 4: Timeframe (15 puntos)
        timeframe = range_signal.get('time_horizon_hours', 24)
        
        if timeframe <= 4:
            score += 15
            reasons.append(f"✅ Timeframe ideal ({timeframe}h) (+15)")
        elif timeframe <= 6:
            score += 10
            reasons.append(f"✅ Timeframe bueno ({timeframe}h) (+10)")
        elif timeframe <= 8:
            score += 5
            reasons.append(f"⚠️ Timeframe aceptable ({timeframe}h) (+5)")
        else:
            reasons.append(f"❌ Timeframe muy largo ({timeframe}h) (0)")
        
        # FACTOR 5: R:R Potencial (10 puntos)
        expected_magnitude = range_signal.get('expected_magnitude_pct', 0)
        if range_pct > 0:
            potential_rr = expected_magnitude / range_pct
            
            if potential_rr >= 2.5:
                score += 10
                reasons.append(f"✅ Excelente R:R potencial ({potential_rr:.1f}:1) (+10)")
            elif potential_rr >= 2.0:
                score += 6
                reasons.append(f"✅ Buen R:R potencial ({potential_rr:.1f}:1) (+6)")
            elif potential_rr >= 1.5:
                score += 3
                reasons.append(f"⚠️ R:R moderado ({potential_rr:.1f}:1) (+3)")
            else:
                reasons.append(f"❌ R:R insuficiente ({potential_rr:.1f}:1) (0)")
        
        return score, reasons
    
    def _create_scalping_trade(self, bot_signal: Dict, interpretation: Dict, score: int) -> Optional[Dict]:
        """Crear trade de scalping"""
        
        try:
            # Obtener precio actual (simulado por ahora)
            current_price = self._get_current_price()
            
            # Calcular parámetros del trade
            range_signal = interpretation.get('interpretations', {}).get('range_signal', {})
            expected_magnitude = range_signal.get('expected_magnitude_pct', 1.4)
            range_abs = range_signal.get('range_absolute', 20)
            
            # Determinar dirección (por ahora, asumimos alcista si coiling)
            direction = "LONG"  # Simplificado para validación
            
            # Calcular precios
            if direction == "LONG":
                entry_price = current_price
                stop_loss = current_price - range_abs
                take_profit = current_price + (expected_magnitude / 100 * current_price)
            else:
                entry_price = current_price
                stop_loss = current_price + range_abs
                take_profit = current_price - (expected_magnitude / 100 * current_price)
            
            # Calcular R:R
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Calcular tamaño de posición
            position_size_usd = self.capital * (self.position_size_pct / 100)
            
            trade = {
                "id": f"SCALP_{int(time.time())}",
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": self.symbol,
                "direction": direction,
                "entry_price": round(entry_price, 2),
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(take_profit, 2),
                "position_size_usd": round(position_size_usd, 2),
                "rr_ratio": round(rr_ratio, 2),
                "score": score,
                "status": "ACTIVE",
                "bot_signal": bot_signal,
                "expected_magnitude_pct": expected_magnitude,
                "max_time_hours": self.max_time_in_trade
            }
            
            return trade
            
        except Exception as e:
            print(f"❌ Error creando trade: {str(e)}")
            return None
    
    def _get_current_price(self) -> float:
        """Obtener precio actual (simulado por ahora)"""
        # TODO: Integrar con API real de Binance
        # Por ahora, simulamos basado en interpretaciones
        return 4487.0  # Precio base de ejemplo
    
    def _update_active_trades(self):
        """Actualizar estado de trades activos"""
        
        current_time = datetime.utcnow()
        updated_trades = []
        
        for trade in self.active_trades:
            # Verificar tiempo límite
            trade_time = datetime.fromisoformat(trade['timestamp'])
            hours_elapsed = (current_time - trade_time).total_seconds() / 3600
            
            if hours_elapsed > trade['max_time_hours']:
                # Cerrar por tiempo
                trade['status'] = 'CLOSED_TIME'
                trade['exit_reason'] = 'Max time reached'
                trade['exit_time'] = current_time.isoformat()
                self._process_trade_exit(trade, 'TIME_EXIT')
            else:
                # Simular verificación de TP/SL
                # TODO: Integrar con precio real
                updated_trades.append(trade)
        
        self.active_trades = updated_trades
    
    def _process_trade_exit(self, trade: Dict, exit_type: str):
        """Procesar salida de trade"""
        
        # Calcular PnL simulado
        if exit_type == 'TAKE_PROFIT':
            pnl_pct = trade['expected_magnitude_pct']
            trade['result'] = 'WIN'
        elif exit_type == 'STOP_LOSS':
            pnl_pct = -(trade['expected_magnitude_pct'] / trade['rr_ratio'])
            trade['result'] = 'LOSS'
        else:  # TIME_EXIT
            pnl_pct = 0  # Breakeven
            trade['result'] = 'BREAKEVEN'
        
        trade['pnl_pct'] = pnl_pct
        trade['pnl_usd'] = trade['position_size_usd'] * (pnl_pct / 100)
        
        # Actualizar estadísticas
        if trade['result'] == 'WIN':
            self.daily_stats['trades_won'] += 1
        elif trade['result'] == 'LOSS':
            self.daily_stats['trades_lost'] += 1
        
        self.daily_stats['total_pnl_pct'] += pnl_pct
        self.daily_stats['total_pnl_usd'] += trade['pnl_usd']
        
        # Actualizar mejores/peores trades
        if pnl_pct > self.daily_stats['best_trade_pct']:
            self.daily_stats['best_trade_pct'] = pnl_pct
        if pnl_pct < self.daily_stats['worst_trade_pct']:
            self.daily_stats['worst_trade_pct'] = pnl_pct
        
        # Guardar en historial
        self.trade_history.append(trade)
        
        print(f"🔚 Trade cerrado: {trade['id']} - {trade['result']} - {pnl_pct:+.2f}%")
    
    def _update_daily_stats(self):
        """Actualizar estadísticas diarias calculadas"""
        
        # Win rate
        total_closed = self.daily_stats['trades_won'] + self.daily_stats['trades_lost']
        if total_closed > 0:
            self.daily_stats['win_rate'] = (self.daily_stats['trades_won'] / total_closed) * 100
        
        # R:R promedio realizado
        if len(self.trade_history) > 0:
            rr_ratios = [t.get('rr_ratio', 0) for t in self.trade_history if t.get('result') == 'WIN']
            if rr_ratios:
                self.daily_stats['avg_rr_realized'] = np.mean(rr_ratios)
        
        # Coiling accuracy
        if self.daily_stats['signals_processed'] > 0:
            self.daily_stats['coiling_accuracy'] = (self.daily_stats['signals_acted'] / 
                                                   self.daily_stats['signals_processed']) * 100
    
    def _save_cycle_log(self, cycle_result: Dict):
        """Guardar log del ciclo"""
        
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.trading_logs_dir, f"scalping_cycle_{timestamp_str}.json")
        
        try:
            with open(log_file, 'w') as f:
                json.dump(cycle_result, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error guardando log: {str(e)}")
    
    def get_daily_report(self) -> str:
        """Generar reporte diario"""
        
        stats = self.daily_stats
        
        report = f"""
📊 REPORTE DIARIO SCALPING - {stats['date']}
{'=' * 50}

📈 TRADES:
   Total: {stats['trades_taken']}
   Ganados: {stats['trades_won']}
   Perdidos: {stats['trades_lost']}
   Win Rate: {stats.get('win_rate', 0):.1f}%

💰 P&L:
   Total %: {stats['total_pnl_pct']:+.2f}%
   Total USD: ${stats['total_pnl_usd']:+.2f}
   Mejor trade: {stats['best_trade_pct']:+.2f}%
   Peor trade: {stats['worst_trade_pct']:+.2f}%

🎯 PERFORMANCE:
   R:R promedio: {stats['avg_rr_realized']:.1f}:1
   Señales procesadas: {stats['signals_processed']}
   Señales actuadas: {stats['signals_acted']}
   Accuracy coiling: {stats['coiling_accuracy']:.1f}%

🔄 ESTADO ACTUAL:
   Trades activos: {len(self.active_trades)}
   Puede operar: {'✅' if self._can_take_new_trade() else '❌'}
"""
        
        return report
    
    def reset_daily_stats(self):
        """Reset estadísticas diarias (llamar a medianoche)"""
        self.daily_stats = self._init_daily_stats()
        print("🔄 Estadísticas diarias reseteadas")

def test_scalping_cycle():
    """Test del motor de scalping"""
    
    print("⚡ INICIANDO TEST MOTOR SCALPING")
    print("=" * 50)
    
    # Inicializar motor
    engine = ScalpingEngine("ETHUSD_PERP", capital=1000.0)
    
    # Ejecutar ciclo
    result = engine.run_scalping_cycle()
    
    # Mostrar resultado
    print(f"\n📊 RESULTADO DEL CICLO:")
    print(f"   Status: {result['status']}")
    print(f"   Trades activos: {result['active_trades']}")
    if result.get('entry_evaluation'):
        eval_result = result['entry_evaluation']
        print(f"   Evaluación: {eval_result['action']}")
        if 'score' in eval_result:
            print(f"   Score: {eval_result['score']}/100")
    
    # Mostrar reporte diario
    print(engine.get_daily_report())
    
    return result

if __name__ == "__main__":
    test_scalping_cycle()