#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrderBook Intelligence System
Sistema avanzado de análisis de microestructura del mercado
Autor: biGGzeta
Fecha: 2025-10-05 13:45:50 UTC
"""

import asyncio
import datetime
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import websockets
from collections import deque
import statistics

@dataclass
class OrderBookSnapshot:
    """Snapshot del order book en un momento específico"""
    timestamp: datetime.datetime
    price: float
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]  # (price, quantity)
    trades_count: int
    volume_takers: float
    avg_order_size: float
    
@dataclass
class MarketCondition:
    """Condición del mercado en un período"""
    start_time: datetime.datetime
    end_time: datetime.datetime
    condition_type: str  # "lateral", "disruptive", "trending"
    price_range_pct: float
    volatility: float
    volume_profile: Dict
    order_book_stats: Dict
    
@dataclass
class MovementContext:
    """Contexto completo de un movimiento"""
    movement_start: datetime.datetime
    movement_end: datetime.datetime
    price_change_pct: float
    movement_type: str  # "spike", "gradual", "reversal"
    pre_movement: MarketCondition
    during_movement: MarketCondition
    post_movement: MarketCondition
    disruption_signals: List[str]

class OrderBookIntelligence:
    """Sistema de inteligencia de order book"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP"):
        self.symbol = symbol
        self.setup_logging()
        
        # Data storage
        self.orderbook_snapshots = deque(maxlen=1440)  # 24h de snapshots (1/min)
        self.trade_history = deque(maxlen=10080)  # 1 semana de trades
        self.market_conditions = deque(maxlen=288)  # 24h de condiciones (5min intervals)
        
        # Configuration
        self.config = {
            "snapshot_interval_seconds": 60,  # Snapshot cada 1 min
            "analysis_interval_seconds": 300,  # Análisis cada 5 min
            "movement_threshold_pct": 0.5,  # Movimiento >0.5% = disruptivo
            "lateral_volatility_threshold": 0.2,  # <0.2% = lateral
            "whale_order_threshold": 50000,  # >$50k = whale order
            "orderbook_depth_levels": 10,  # Analizar 10 niveles
        }
        
        # State
        self.current_market_regime = "unknown"
        self.last_signal_context = None
        self.disruption_alerts = []
        
        self.logger.info(f"[INIT] OrderBook Intelligence iniciado para {symbol}")
    
    def setup_logging(self):
        """Configurar logging específico para orderbook"""
        
        log_dir = "logs/orderbook"
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger("OrderBookIntel")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        log_file = os.path.join(log_dir, f"orderbook_intel_{self.symbol}_{datetime.datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    async def start_monitoring(self):
        """Iniciar monitoreo completo del order book"""
        
        self.logger.info("[START] Iniciando OrderBook Intelligence monitoring")
        
        # Crear tasks concurrentes
        tasks = [
            asyncio.create_task(self._orderbook_snapshot_loop()),
            asyncio.create_task(self._market_analysis_loop()),
            asyncio.create_task(self._disruption_detection_loop())
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _orderbook_snapshot_loop(self):
        """Loop para capturar snapshots del order book"""
        
        self.logger.info("[SNAPSHOT] OrderBook snapshot loop iniciado")
        
        while True:
            try:
                # Simular captura de order book (en real sería WebSocket)
                snapshot = await self._capture_orderbook_snapshot()
                
                if snapshot:
                    self.orderbook_snapshots.append(snapshot)
                    await self._process_snapshot(snapshot)
                
                await asyncio.sleep(self.config["snapshot_interval_seconds"])
                
            except Exception as e:
                self.logger.error(f"[ERROR] Error en snapshot loop: {str(e)}")
                await asyncio.sleep(10)
    
    async def _capture_orderbook_snapshot(self) -> Optional[OrderBookSnapshot]:
        """Capturar snapshot actual del order book"""
        
        try:
            # TODO: Integrar con WebSocket real de order book
            # Por ahora simulamos datos realistas
            
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            current_price = 4539.0 + np.random.normal(0, 2)  # Precio base con ruido
            
            # Simular order book depth
            bids = []
            asks = []
            
            # Generar niveles del order book
            for i in range(self.config["orderbook_depth_levels"]):
                bid_price = current_price - (i + 1) * 0.5
                ask_price = current_price + (i + 1) * 0.5
                
                # Cantidad con distribución realista
                base_qty = np.random.exponential(10) + 1
                
                # Añadir whale orders ocasionalmente
                if np.random.random() < 0.05:  # 5% chance de whale
                    base_qty *= np.random.uniform(5, 20)
                
                bids.append((bid_price, base_qty))
                asks.append((ask_price, base_qty))
            
            # Simular estadísticas de trading
            trades_count = np.random.poisson(50)  # ~50 trades por minuto
            volume_takers = np.random.exponential(1000)
            avg_order_size = volume_takers / max(trades_count, 1)
            
            return OrderBookSnapshot(
                timestamp=timestamp,
                price=current_price,
                bids=bids,
                asks=asks,
                trades_count=trades_count,
                volume_takers=volume_takers,
                avg_order_size=avg_order_size
            )
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error capturando snapshot: {str(e)}")
            return None
    
    async def _process_snapshot(self, snapshot: OrderBookSnapshot):
        """Procesar snapshot recién capturado"""
        
        # Detectar cambios significativos
        if len(self.orderbook_snapshots) >= 2:
            prev_snapshot = self.orderbook_snapshots[-2]
            price_change_pct = ((snapshot.price - prev_snapshot.price) / prev_snapshot.price) * 100
            
            # Detectar movimientos disruptivos
            if abs(price_change_pct) >= self.config["movement_threshold_pct"]:
                await self._detect_disruption(prev_snapshot, snapshot, price_change_pct)
        
        # Analizar microestructura
        microstructure = self._analyze_microstructure(snapshot)
        
        # Log datos importantes
        if len(self.orderbook_snapshots) % 10 == 0:  # Log cada 10 snapshots
            self.logger.info(f"[MICRO] Price: ${snapshot.price:.2f}, Spread: {microstructure['spread']:.3f}, Imbalance: {microstructure['imbalance']:.2f}")
    
    def _analyze_microstructure(self, snapshot: OrderBookSnapshot) -> Dict:
        """Analizar microestructura del order book"""
        
        # Calcular spread
        best_bid = snapshot.bids[0][0] if snapshot.bids else snapshot.price - 1
        best_ask = snapshot.asks[0][0] if snapshot.asks else snapshot.price + 1
        spread = best_ask - best_bid
        spread_pct = (spread / snapshot.price) * 100
        
        # Calcular imbalance bid/ask
        total_bid_volume = sum([qty for _, qty in snapshot.bids[:5]])
        total_ask_volume = sum([qty for _, qty in snapshot.asks[:5]])
        total_volume = total_bid_volume + total_ask_volume
        
        imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
        
        # Detectar whale walls
        whale_bids = [qty for _, qty in snapshot.bids if qty * snapshot.price > self.config["whale_order_threshold"]]
        whale_asks = [qty for _, qty in snapshot.asks if qty * snapshot.price > self.config["whale_order_threshold"]]
        
        # Calcular depth
        bid_depth = sum([qty for _, qty in snapshot.bids])
        ask_depth = sum([qty for _, qty in snapshot.asks])
        
        return {
            "spread": spread,
            "spread_pct": spread_pct,
            "imbalance": imbalance,
            "whale_bids_count": len(whale_bids),
            "whale_asks_count": len(whale_asks),
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
            "trades_per_minute": snapshot.trades_count,
            "avg_order_size": snapshot.avg_order_size
        }
    
    async def _detect_disruption(self, prev_snapshot: OrderBookSnapshot, 
                               current_snapshot: OrderBookSnapshot, price_change_pct: float):
        """Detectar y analizar disrupciones del mercado"""
        
        disruption_type = "bullish" if price_change_pct > 0 else "bearish"
        magnitude = abs(price_change_pct)
        
        # Analizar contexto pre-movimiento
        pre_context = self._analyze_pre_movement_context()
        
        # Detectar señales de disrupción
        disruption_signals = self._identify_disruption_signals(prev_snapshot, current_snapshot)
        
        disruption_event = {
            "timestamp": current_snapshot.timestamp.isoformat(),
            "type": disruption_type,
            "magnitude_pct": magnitude,
            "price_from": prev_snapshot.price,
            "price_to": current_snapshot.price,
            "pre_context": pre_context,
            "disruption_signals": disruption_signals
        }
        
        self.disruption_alerts.append(disruption_event)
        
        self.logger.warning(f"[DISRUPTION] {disruption_type.upper()} {magnitude:.2f}% - Signals: {', '.join(disruption_signals)}")
        
        # Guardar evento para análisis posterior
        await self._save_disruption_event(disruption_event)
    
    def _analyze_pre_movement_context(self) -> Dict:
        """Analizar contexto previo al movimiento"""
        
        if len(self.orderbook_snapshots) < 5:
            return {"status": "insufficient_data"}
        
        # Analizar últimos 5 snapshots (5 minutos)
        recent_snapshots = list(self.orderbook_snapshots)[-5:]
        
        # Calcular volatilidad previa
        prices = [s.price for s in recent_snapshots]
        volatility = np.std(prices) / np.mean(prices) * 100
        
        # Analizar volumen previo
        avg_volume = np.mean([s.volume_takers for s in recent_snapshots])
        avg_trades = np.mean([s.trades_count for s in recent_snapshots])
        
        # Detectar patrón de acumulación/distribución
        volume_trend = self._detect_volume_trend(recent_snapshots)
        
        return {
            "volatility_pct": volatility,
            "avg_volume": avg_volume,
            "avg_trades_per_min": avg_trades,
            "volume_trend": volume_trend,
            "market_condition": "lateral" if volatility < self.config["lateral_volatility_threshold"] else "active"
        }
    
    def _detect_volume_trend(self, snapshots: List[OrderBookSnapshot]) -> str:
        """Detectar tendencia de volumen"""
        
        volumes = [s.volume_takers for s in snapshots]
        
        if len(volumes) < 3:
            return "unknown"
        
        # Calcular tendencia simple
        recent_avg = np.mean(volumes[-2:])
        earlier_avg = np.mean(volumes[:-2])
        
        if recent_avg > earlier_avg * 1.2:
            return "increasing"
        elif recent_avg < earlier_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _identify_disruption_signals(self, prev_snapshot: OrderBookSnapshot, 
                                   current_snapshot: OrderBookSnapshot) -> List[str]:
        """Identificar señales que causaron la disrupción"""
        
        signals = []
        
        # Analizar cambios en order book
        prev_micro = self._analyze_microstructure(prev_snapshot)
        current_micro = self._analyze_microstructure(current_snapshot)
        
        # Cambio en imbalance
        imbalance_change = current_micro["imbalance"] - prev_micro["imbalance"]
        if abs(imbalance_change) > 0.3:
            signals.append(f"imbalance_shift_{imbalance_change:+.2f}")
        
        # Whale activity
        if current_micro["whale_bids_count"] > prev_micro["whale_bids_count"]:
            signals.append("whale_bids_appeared")
        if current_micro["whale_asks_count"] > prev_micro["whale_asks_count"]:
            signals.append("whale_asks_appeared")
        
        # Volume spike
        volume_ratio = current_snapshot.volume_takers / max(prev_snapshot.volume_takers, 1)
        if volume_ratio > 2.0:
            signals.append(f"volume_spike_{volume_ratio:.1f}x")
        
        # Spread changes
        spread_change = (current_micro["spread"] - prev_micro["spread"]) / prev_micro["spread"]
        if abs(spread_change) > 0.5:
            signals.append(f"spread_change_{spread_change:+.1f}%")
        
        # Trade intensity
        trades_ratio = current_snapshot.trades_count / max(prev_snapshot.trades_count, 1)
        if trades_ratio > 1.5:
            signals.append(f"trade_intensity_{trades_ratio:.1f}x")
        
        return signals if signals else ["unknown_catalyst"]
    
    async def _market_analysis_loop(self):
        """Loop para análisis de condiciones de mercado cada 5 min"""
        
        self.logger.info("[ANALYSIS] Market analysis loop iniciado")
        
        while True:
            try:
                await asyncio.sleep(self.config["analysis_interval_seconds"])
                
                if len(self.orderbook_snapshots) >= 5:
                    market_condition = self._analyze_market_condition()
                    
                    if market_condition:
                        self.market_conditions.append(market_condition)
                        await self._process_market_condition(market_condition)
                
            except Exception as e:
                self.logger.error(f"[ERROR] Error en market analysis: {str(e)}")
                await asyncio.sleep(30)
    
    def _analyze_market_condition(self) -> Optional[MarketCondition]:
        """Analizar condición actual del mercado"""
        
        if len(self.orderbook_snapshots) < 5:
            return None
        
        # Analizar últimos 5 minutos
        recent_snapshots = list(self.orderbook_snapshots)[-5:]
        
        start_time = recent_snapshots[0].timestamp
        end_time = recent_snapshots[-1].timestamp
        
        # Calcular métricas del período
        prices = [s.price for s in recent_snapshots]
        price_range_pct = ((max(prices) - min(prices)) / np.mean(prices)) * 100
        volatility = np.std(prices) / np.mean(prices) * 100
        
        # Determinar tipo de condición
        if volatility < self.config["lateral_volatility_threshold"]:
            condition_type = "lateral"
        elif price_range_pct > self.config["movement_threshold_pct"]:
            condition_type = "disruptive"
        else:
            condition_type = "trending"
        
        # Analizar volumen profile
        volume_profile = {
            "total_volume": sum([s.volume_takers for s in recent_snapshots]),
            "avg_trades_per_min": np.mean([s.trades_count for s in recent_snapshots]),
            "avg_order_size": np.mean([s.avg_order_size for s in recent_snapshots])
        }
        
        # Estadísticas de order book
        order_book_stats = {
            "avg_spread_pct": np.mean([self._analyze_microstructure(s)["spread_pct"] for s in recent_snapshots]),
            "avg_imbalance": np.mean([self._analyze_microstructure(s)["imbalance"] for s in recent_snapshots]),
            "whale_activity": sum([self._analyze_microstructure(s)["whale_bids_count"] + 
                                 self._analyze_microstructure(s)["whale_asks_count"] for s in recent_snapshots])
        }
        
        return MarketCondition(
            start_time=start_time,
            end_time=end_time,
            condition_type=condition_type,
            price_range_pct=price_range_pct,
            volatility=volatility,
            volume_profile=volume_profile,
            order_book_stats=order_book_stats
        )
    
    async def _process_market_condition(self, condition: MarketCondition):
        """Procesar nueva condición de mercado"""
        
        # Detectar cambios de régimen
        if len(self.market_conditions) >= 2:
            prev_condition = self.market_conditions[-2]
            
            if prev_condition.condition_type != condition.condition_type:
                self.logger.info(f"[REGIME] Market regime change: {prev_condition.condition_type} → {condition.condition_type}")
                self.current_market_regime = condition.condition_type
        
        # Log condición actual
        self.logger.info(f"[CONDITION] {condition.condition_type} - Vol: {condition.volatility:.2f}% - Range: {condition.price_range_pct:.2f}%")
        
        # Guardar para análisis posterior
        await self._save_market_condition(condition)
    
    async def _disruption_detection_loop(self):
        """Loop para detección continua de disrupciones"""
        
        self.logger.info("[DISRUPTION] Disruption detection loop iniciado")
        
        while True:
            try:
                await asyncio.sleep(30)  # Check cada 30 segundos
                
                # Analizar patrones de disrupción recientes
                await self._analyze_disruption_patterns()
                
            except Exception as e:
                self.logger.error(f"[ERROR] Error en disruption detection: {str(e)}")
                await asyncio.sleep(60)
    
    async def _analyze_disruption_patterns(self):
        """Analizar patrones en disrupciones recientes"""
        
        if len(self.disruption_alerts) < 2:
            return
        
        recent_disruptions = self.disruption_alerts[-5:]  # Últimas 5 disrupciones
        
        # Analizar frecuencia
        time_diff = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(recent_disruptions[0]["timestamp"])
        disruption_frequency = len(recent_disruptions) / (time_diff.total_seconds() / 3600)  # Por hora
        
        # Detectar clustering de disrupciones
        if disruption_frequency > 2.0:  # Más de 2 disrupciones por hora
            self.logger.warning(f"[PATTERN] High disruption frequency detected: {disruption_frequency:.1f}/hour")
    
    async def _save_disruption_event(self, event: Dict):
        """Guardar evento de disrupción"""
        
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/orderbook/disruption_{self.symbol}_{timestamp_str}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(event, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"[ERROR] Error saving disruption event: {str(e)}")
    
    async def _save_market_condition(self, condition: MarketCondition):
        """Guardar condición de mercado"""
        
        timestamp_str = condition.end_time.strftime("%Y%m%d_%H")
        filename = f"logs/orderbook/market_conditions_{self.symbol}_{timestamp_str}.json"
        
        try:
            # Cargar condiciones existentes o crear nuevo
            conditions = []
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    conditions = json.load(f)
            
            # Añadir nueva condición
            conditions.append(asdict(condition))
            
            # Guardar actualizado
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conditions, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error saving market condition: {str(e)}")
    
    # ========== INTEGRATION WITH TRADING SIGNALS ==========
    
    def evaluate_signal_context(self, trading_signal: Dict) -> Dict:
        """Evaluar contexto de order book para una señal de trading"""
        
        if not self.market_conditions:
            return {"status": "insufficient_data"}
        
        current_condition = self.market_conditions[-1]
        
        # Determinar si las condiciones del order book apoyan la señal
        signal_support = self._calculate_signal_support(trading_signal, current_condition)
        
        # Identificar riesgos específicos
        risks = self._identify_signal_risks(trading_signal, current_condition)
        
        # Sugerir ajustes
        adjustments = self._suggest_signal_adjustments(trading_signal, current_condition)
        
        return {
            "status": "success",
            "current_market_condition": current_condition.condition_type,
            "signal_support_score": signal_support,
            "identified_risks": risks,
            "suggested_adjustments": adjustments,
            "orderbook_context": {
                "volatility": current_condition.volatility,
                "whale_activity": current_condition.order_book_stats["whale_activity"],
                "avg_imbalance": current_condition.order_book_stats["avg_imbalance"]
            }
        }
    
    def _calculate_signal_support(self, signal: Dict, condition: MarketCondition) -> int:
        """Calcular score de soporte del order book para la señal"""
        
        support_score = 50  # Base neutral
        
        signal_action = signal.get("action", "WAIT")
        
        # Ajustar por tipo de mercado
        if condition.condition_type == "lateral":
            if signal_action in ["WEAK_BUY", "WEAK_SELL"]:
                support_score += 10  # Favorecer signals débiles en lateral
            elif signal_action in ["STRONG_BUY", "STRONG_SELL"]:
                support_score -= 15  # Penalizar signals fuertes en lateral
        
        elif condition.condition_type == "disruptive":
            if signal_action in ["STRONG_BUY", "STRONG_SELL"]:
                support_score += 15  # Favorecer signals fuertes en disruptivo
            elif signal_action == "WAIT":
                support_score -= 10  # Penalizar espera en mercado activo
        
        # Ajustar por imbalance
        imbalance = condition.order_book_stats["avg_imbalance"]
        if signal_action in ["BUY", "STRONG_BUY"] and imbalance > 0.2:
            support_score += 10  # Imbalance alcista apoya BUY
        elif signal_action in ["SELL", "STRONG_SELL"] and imbalance < -0.2:
            support_score += 10  # Imbalance bajista apoya SELL
        
        # Ajustar por actividad whale
        whale_activity = condition.order_book_stats["whale_activity"]
        if whale_activity > 5:  # Alta actividad whale
            support_score -= 5  # Más cautela con whales activos
        
        return max(0, min(100, support_score))
    
    def _identify_signal_risks(self, signal: Dict, condition: MarketCondition) -> List[str]:
        """Identificar riesgos específicos para la señal"""
        
        risks = []
        
        # Riesgo por volatilidad
        if condition.volatility > 1.0:
            risks.append("high_volatility")
        
        # Riesgo por whale activity
        if condition.order_book_stats["whale_activity"] > 8:
            risks.append("high_whale_activity")
        
        # Riesgo por spread
        if condition.order_book_stats["avg_spread_pct"] > 0.1:
            risks.append("wide_spread")
        
        # Riesgo por disrupciones recientes
        recent_disruptions = len([d for d in self.disruption_alerts[-5:] 
                                if datetime.datetime.now(datetime.timezone.utc) - 
                                datetime.datetime.fromisoformat(d["timestamp"]) < datetime.timedelta(minutes=15)])
        
        if recent_disruptions >= 2:
            risks.append("recent_disruptions")
        
        return risks
    
    def _suggest_signal_adjustments(self, signal: Dict, condition: MarketCondition) -> Dict:
        """Sugerir ajustes a la señal basado en order book"""
        
        adjustments = {}
        
        # Ajuste de posición size
        base_size = signal.get("position_size_pct", 100)
        
        if condition.condition_type == "lateral":
            adjustments["position_size_pct"] = min(base_size, 75)  # Reducir en lateral
        elif condition.volatility > 1.0:
            adjustments["position_size_pct"] = min(base_size, 50)  # Reducir en alta volatilidad
        
        # Ajuste de timeframe
        if condition.condition_type == "disruptive":
            adjustments["suggested_timeframe"] = "short_term"  # Movimientos rápidos
        elif condition.condition_type == "lateral":
            adjustments["suggested_timeframe"] = "extended"  # Esperar breakout
        
        # Ajuste de stop loss
        if condition.volatility > 0.5:
            adjustments["stop_loss_multiplier"] = 1.5  # Stops más amplios
        
        return adjustments

# ========== INTEGRATION FUNCTION ==========

async def integrate_orderbook_with_scheduler(scheduler, orderbook_intel):
    """Integrar OrderBook Intelligence con el Hybrid Scheduler"""
    
    # Inicializar OrderBook Intelligence
    orderbook_task = asyncio.create_task(orderbook_intel.start_monitoring())
    
    # Modificar el scheduler para usar orderbook context
    original_evaluate = scheduler._evaluate_signal_adjustments
    
    async def enhanced_evaluate(tick_data):
        """Evaluación mejorada con contexto de orderbook"""
        
        # Evaluación original
        await original_evaluate(tick_data)
        
        # Evaluación adicional con orderbook
        if scheduler.state.current_signals:
            orderbook_context = orderbook_intel.evaluate_signal_context(scheduler.state.current_signals)
            
            if orderbook_context.get("status") == "success":
                # Aplicar ajustes sugeridos
                suggested_adjustments = orderbook_context["suggested_adjustments"]
                
                if suggested_adjustments:
                    scheduler.state.realtime_adjustments["orderbook_adjustments"] = suggested_adjustments
                    scheduler.logger.info(f"[ORDERBOOK] Adjustments applied: {suggested_adjustments}")
    
    # Reemplazar método
    scheduler._evaluate_signal_adjustments = enhanced_evaluate
    
    return orderbook_task

# ========== TEST FUNCTION ==========

async def test_orderbook_intelligence():
    """Test del sistema de OrderBook Intelligence"""
    
    print("🧪 TESTING ORDERBOOK INTELLIGENCE SYSTEM")
    print("=" * 60)
    
    # Crear sistema
    orderbook_intel = OrderBookIntelligence("ETHUSD_PERP")
    
    # Ejecutar por 5 minutos como test
    try:
        await asyncio.wait_for(orderbook_intel.start_monitoring(), timeout=300)
    except asyncio.TimeoutError:
        print("✅ Test completado - sistema funcionando correctamente")

if __name__ == "__main__":
    asyncio.run(test_orderbook_intelligence())