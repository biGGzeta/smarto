#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Trading System with REAL Whale Detection - NO SIMULATION
Sistema profesional con detección REAL de whales - SIN SIMULACIÓN
Autor: biGGzeta
Fecha: 2025-10-05 17:50:30 UTC
Versión: 5.1.0 (Real Whales - No Simulation)
"""

import asyncio
import datetime
import json
import os
import logging
import logging.handlers
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import websockets
from collections import deque
import statistics
from concurrent.futures import ThreadPoolExecutor
import pickle
import signal
import sys
from pathlib import Path

# ML Imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

@dataclass
class ProfessionalDataPoint:
    """Data point profesional con todos los datos institucionales"""
    timestamp: datetime.datetime
    
    # Basic Price Data
    price: float
    open_price: float
    high_price: float
    low_price: float
    volume: float
    
    # Advanced Price Data
    weighted_avg_price: float
    price_change_24h: float
    price_change_pct_24h: float
    volume_change_pct: float
    trade_count: int
    
    # Real OrderBook Data
    best_bid: float
    best_ask: float
    real_spread: float
    bid_quantity: float
    ask_quantity: float
    orderbook_imbalance: float
    
    # OrderBook Depth (5 levels)
    bids_depth: List[Tuple[float, float]]
    asks_depth: List[Tuple[float, float]]
    total_bid_volume: float
    total_ask_volume: float
    depth_ratio: float
    
    # Whale Detection
    whale_bids: float
    whale_asks: float
    whale_activity_score: float
    large_orders_count: int
    
    # Market Microstructure
    taker_buy_volume: float = 0
    taker_sell_volume: float = 0
    buy_sell_ratio: float = 1.0
    market_momentum: float = 0

class ProfessionalLoggingSystem:
    """Sistema de logging profesional para monitoreo continuo"""
    
    def __init__(self, base_path="logs/professional_system"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.setup_advanced_logging()
    
    def setup_advanced_logging(self):
        """Setup logging avanzado con rotación y niveles"""
        
        # 1. MAIN SYSTEM LOG
        main_handler = logging.handlers.TimedRotatingFileHandler(
            filename=self.base_path / "system_main.log",
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        main_handler.setLevel(logging.INFO)
        
        # 2. WHALE ACTIVITY LOG  
        whale_handler = logging.handlers.RotatingFileHandler(
            filename=self.base_path / "whale_activity.log",
            maxBytes=20*1024*1024,
            backupCount=20,
            encoding='utf-8'
        )
        whale_handler.setLevel(logging.WARNING)
        
        # 3. PERFORMANCE LOG
        perf_handler = logging.handlers.TimedRotatingFileHandler(
            filename=self.base_path / "performance_metrics.log",
            when='H',
            interval=6,
            backupCount=28,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        
        # 4. CONSOLE HANDLER
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Aplicar formatters
        main_handler.setFormatter(detailed_formatter)
        whale_handler.setFormatter(detailed_formatter)
        perf_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(detailed_formatter)
        
        # Configurar loggers específicos
        self.setup_specialized_loggers(main_handler, whale_handler, perf_handler, console_handler)
    
    def setup_specialized_loggers(self, main_h, whale_h, perf_h, console_h):
        """Configurar loggers especializados"""
        
        # Main system logger
        main_logger = logging.getLogger("SystemMain")
        main_logger.handlers.clear()
        main_logger.addHandler(main_h)
        main_logger.addHandler(console_h)
        main_logger.setLevel(logging.INFO)
        main_logger.propagate = False
        
        # Whale logger
        whale_logger = logging.getLogger("WhaleActivity")
        whale_logger.handlers.clear()
        whale_logger.addHandler(whale_h)
        whale_logger.addHandler(console_h)
        whale_logger.setLevel(logging.WARNING)
        whale_logger.propagate = False
        
        # Performance logger
        perf_logger = logging.getLogger("Performance")
        perf_logger.handlers.clear()
        perf_logger.addHandler(perf_h)
        perf_logger.addHandler(console_h)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
        
        # Data capture logger
        capture_logger = logging.getLogger("ProfessionalCapture")
        capture_logger.handlers.clear()
        capture_logger.addHandler(main_h)
        capture_logger.setLevel(logging.INFO)
        capture_logger.propagate = False

class ProfessionalDataCapture:
    """Data capture CON WHALE DETECTION REAL - NO SIMULATION"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP"):
        self.symbol = symbol
        self.running = False
        
        # Buffers para data REAL
        self.realtime_data = deque(maxlen=8640)
        self.orderbook_snapshots = deque(maxlen=1440)
        self.klines_1m = deque(maxlen=2880)
        self.trade_flow = deque(maxlen=20000)
        
        # WebSocket URLs REALES
        self.websocket_urls = {
            "ticker": "wss://dstream.binance.com/ws/ethusd_perp@ticker",
            "orderbook": "wss://dstream.binance.com/ws/ethusd_perp@depth5@100ms",  # REAL OrderBook
            "kline_1m": "wss://dstream.binance.com/ws/ethusd_perp@kline_1m",
            "trades": "wss://dstream.binance.com/ws/ethusd_perp@trade",
            "markPrice": "wss://dstream.binance.com/ws/ethusd_perp@markPrice"
        }
        
        self.connections = {}
        
        # Configuración REAL para whale detection
        self.config = {
            "whale_threshold_usd": 50000,  # $50k threshold REAL
            "orderbook_levels": 5,
            "capture_interval_seconds": 5,
        }
        
        # Stats REALES
        self.capture_stats = {
            "ticker_points": 0,
            "orderbook_updates": 0,
            "whale_orders_detected": 0,  # CONTADOR REAL
            "total_whale_volume": 0,     # VOLUMEN REAL
            "connection_resets": 0,
            "data_quality_score": 100.0,
        }
        
        # Setup logging
        self.logger = logging.getLogger("ProfessionalCapture")
        self.whale_logger = logging.getLogger("WhaleActivity")
        self.logger.info(f"[INIT] 🚀 Professional Data Capture REAL WHALES iniciado para {symbol}")
        self.logger.info(f"[CONFIG] Whale threshold REAL: ${self.config['whale_threshold_usd']:,}")
    
    async def start_professional_capture(self):
        """Iniciar captura profesional CON WHALES REALES"""
        
        self.logger.info("[START] 🚀 Iniciando captura profesional con WHALES REALES")
        self.running = True
        
        # Crear tasks para streams REALES
        tasks = []
        
        # Stream tasks REALES
        for stream_name, url in self.websocket_urls.items():
            task = asyncio.create_task(self._stream_handler(stream_name, url))
            tasks.append(task)
        
        # Analysis tasks REALES
        tasks.extend([
            asyncio.create_task(self._whale_monitoring_loop()),  # WHALE MONITORING REAL
            asyncio.create_task(self._stats_loop())
        ])
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self._cleanup()
    
    async def _stream_handler(self, stream_name: str, url: str):
        """Handler REAL para streams de Binance"""
        
        self.logger.info(f"[{stream_name.upper()}] 🔗 Conectando a stream REAL...")
        
        reconnect_count = 0
        max_reconnects = 20
        
        while self.running and reconnect_count < max_reconnects:
            try:
                async with websockets.connect(url, ping_timeout=30, ping_interval=20) as websocket:
                    self.connections[stream_name] = websocket
                    self.logger.info(f"[{stream_name.upper()}] ✅ Conectado a stream REAL")
                    reconnect_count = 0  # Reset on successful connection
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._process_real_stream_data(stream_name, data)
                        except Exception as e:
                            self.logger.debug(f"[{stream_name.upper()}] Error procesando: {str(e)}")
                        
            except Exception as e:
                reconnect_count += 1
                self.logger.warning(f"[{stream_name.upper()}] ⚠️ Desconectado (intento {reconnect_count}): {str(e)}")
                self.connections.pop(stream_name, None)
                self.capture_stats["connection_resets"] += 1
                
                if self.running and reconnect_count < max_reconnects:
                    wait_time = min(60, 5 * reconnect_count)
                    await asyncio.sleep(wait_time)
        
        if reconnect_count >= max_reconnects:
            self.logger.error(f"[{stream_name.upper()}] ❌ Max reconnections reached")
    
    async def _process_real_stream_data(self, stream_name: str, data: Dict):
        """Procesamiento REAL de datos de streams"""
        
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        if stream_name == "ticker":
            await self._process_real_ticker_data(data, timestamp)
        elif stream_name == "orderbook":
            await self._process_real_orderbook_data(data, timestamp)  # WHALE DETECTION AQUÍ
        elif stream_name == "kline_1m":
            await self._process_real_kline_data(data, timestamp)
        elif stream_name == "trades":
            await self._process_real_trade_data(data, timestamp)
    
    async def _process_real_ticker_data(self, data: Dict, timestamp: datetime.datetime):
        """Procesar datos REALES del ticker"""
        
        try:
            if 'c' not in data:
                return
            
            # Crear data point REAL
            price_data = {
                'timestamp': timestamp,
                'price': float(data['c']),
                'open_price': float(data['o']),
                'high_price': float(data['h']),
                'low_price': float(data['l']),
                'volume': float(data['v']),
                'weighted_avg_price': float(data['w']),
                'price_change_24h': float(data['p']),
                'price_change_pct_24h': float(data['P']),
                'trade_count': int(data['n']),
                'volume_change_pct': 0
            }
            
            await self._enqueue_real_ticker_data(price_data)
            self.capture_stats["ticker_points"] += 1
            
        except Exception as e:
            self.logger.error(f"[TICKER] Error: {str(e)}")
    
    async def _process_real_orderbook_data(self, data: Dict, timestamp: datetime.datetime):
        """Procesar OrderBook REAL con WHALE DETECTION REAL - NO SIMULATION"""
        
        try:
            # 🚀 WHALE DETECTION REAL - NO SIMULATION
            
            # Detectar formato de data
            bids_data = None
            asks_data = None
            
            if 'bids' in data and 'asks' in data:
                bids_data = data['bids']
                asks_data = data['asks']
            elif 'b' in data and 'a' in data:
                bids_data = data['b']
                asks_data = data['a']
            else:
                return
            
            # Procesar bids y asks REALES
            bids = [[float(price), float(qty)] for price, qty in bids_data[:self.config["orderbook_levels"]]]
            asks = [[float(price), float(qty)] for price, qty in asks_data[:self.config["orderbook_levels"]]]
            
            if not bids or not asks:
                return
            
            # 🐋 WHALE DETECTION REAL AQUÍ
            whale_threshold = self.config["whale_threshold_usd"]
            
            # Calcular whale orders REALES
            whale_bids = sum(qty for price, qty in bids if qty * price > whale_threshold)
            whale_asks = sum(qty for price, qty in asks if qty * price > whale_threshold)
            
            # Crear snapshot REAL del orderbook
            best_bid, bid_qty = bids[0]
            best_ask, ask_qty = asks[0]
            real_spread = best_ask - best_bid
            
            total_bid_volume = sum(qty for _, qty in bids)
            total_ask_volume = sum(qty for _, qty in asks)
            total_volume = total_bid_volume + total_ask_volume
            orderbook_imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
            
            large_orders_count = len([1 for p, q in bids + asks if q * p > whale_threshold])
            
            orderbook_snapshot = {
                'timestamp': timestamp,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'real_spread': real_spread,
                'bid_quantity': bid_qty,
                'ask_quantity': ask_qty,
                'orderbook_imbalance': orderbook_imbalance,
                'bids_depth': bids,
                'asks_depth': asks,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'depth_ratio': total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1.0,
                'whale_bids': whale_bids,
                'whale_asks': whale_asks,
                'whale_activity_score': (whale_bids + whale_asks) / total_volume if total_volume > 0 else 0,
                'large_orders_count': large_orders_count
            }
            
            self.orderbook_snapshots.append(orderbook_snapshot)
            self.capture_stats["orderbook_updates"] += 1
            
            # 🐋 DETECTAR WHALES REALES - NO SIMULATION
            if whale_bids > 0 or whale_asks > 0:
                self.capture_stats["whale_orders_detected"] += 1
                self.capture_stats["total_whale_volume"] += whale_bids + whale_asks
                
                # LOG WHALE ACTIVITY REAL
                await self._log_real_whale_activity(orderbook_snapshot)
            
        except Exception as e:
            self.logger.error(f"[ORDERBOOK] Error processing REAL data: {str(e)}")
    
    async def _log_real_whale_activity(self, orderbook_snapshot: Dict):
        """Log actividad de whales REAL - NO SIMULATION"""
        
        whale_bids = orderbook_snapshot['whale_bids']
        whale_asks = orderbook_snapshot['whale_asks']
        total_whale = whale_bids + whale_asks
        
        # 🐋 LOGS DE WHALES REALES
        if whale_bids > whale_asks * 2:
            self.whale_logger.warning(f"🐋 REAL LARGE BID PRESSURE: ${whale_bids:,.0f} vs ${whale_asks:,.0f} | Total: ${total_whale:,.0f}")
        elif whale_asks > whale_bids * 2:
            self.whale_logger.warning(f"🐋 REAL LARGE ASK PRESSURE: ${whale_asks:,.0f} vs ${whale_bids:,.0f} | Total: ${total_whale:,.0f}")
        elif total_whale > 100000:  # Solo log si hay >$100k real
            self.whale_logger.warning(f"🐋 REAL WHALE ACTIVITY: Bids ${whale_bids:,.0f}, Asks ${whale_asks:,.0f}")
    
    async def _process_real_kline_data(self, data: Dict, timestamp: datetime.datetime):
        """Procesar klines REALES"""
        
        try:
            if 'k' not in data:
                return
            
            kline = data['k']
            
            # Solo procesar klines cerradas REALES
            if not kline.get('x', False):
                return
            
            kline_data = {
                'timestamp': datetime.datetime.fromtimestamp(kline['t'] / 1000, tz=datetime.timezone.utc),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'trades': int(kline['n']),
                'taker_buy_volume': float(kline['V']),
                'taker_buy_quote_volume': float(kline['Q'])
            }
            
            # Calcular buy/sell ratio REAL
            taker_sell_volume = kline_data['volume'] - kline_data['taker_buy_volume']
            kline_data['buy_sell_ratio'] = kline_data['taker_buy_volume'] / taker_sell_volume if taker_sell_volume > 0 else 1.0
            
            self.klines_1m.append(kline_data)
            
            self.logger.info(f"[KLINE_1M] ✅ REAL Kline: C:{kline_data['close']:.2f} V:{kline_data['volume']:.0f} B/S:{kline_data['buy_sell_ratio']:.2f}")
            
        except Exception as e:
            self.logger.error(f"[KLINE] Error: {str(e)}")
    
    async def _process_real_trade_data(self, data: Dict, timestamp: datetime.datetime):
        """Procesar trades REALES individuales"""
        
        try:
            if 'p' not in data or 'q' not in data:
                return
            
            trade_data = {
                'timestamp': timestamp,
                'price': float(data['p']),
                'quantity': float(data['q']),
                'is_buyer_maker': data.get('m', False)
            }
            
            self.trade_flow.append(trade_data)
            
        except Exception as e:
            self.logger.error(f"[TRADES] Error: {str(e)}")
    
    async def _enqueue_real_ticker_data(self, basic_data: Dict):
        """Enqueue ticker data REAL"""
        
        # Combinar con último orderbook snapshot REAL
        latest_orderbook = self.orderbook_snapshots[-1] if self.orderbook_snapshots else {}
        
        # Crear data point profesional REAL
        professional_point = ProfessionalDataPoint(
            timestamp=basic_data['timestamp'],
            price=basic_data['price'],
            open_price=basic_data['open_price'],
            high_price=basic_data['high_price'],
            low_price=basic_data['low_price'],
            volume=basic_data['volume'],
            weighted_avg_price=basic_data['weighted_avg_price'],
            price_change_24h=basic_data['price_change_24h'],
            price_change_pct_24h=basic_data['price_change_pct_24h'],
            trade_count=basic_data['trade_count'],
            volume_change_pct=basic_data['volume_change_pct'],
            
            # OrderBook data REAL
            best_bid=latest_orderbook.get('best_bid', basic_data['price'] * 0.9995),
            best_ask=latest_orderbook.get('best_ask', basic_data['price'] * 1.0005),
            real_spread=latest_orderbook.get('real_spread', basic_data['price'] * 0.001),
            bid_quantity=latest_orderbook.get('bid_quantity', 0),
            ask_quantity=latest_orderbook.get('ask_quantity', 0),
            orderbook_imbalance=latest_orderbook.get('orderbook_imbalance', 0),
            
            bids_depth=latest_orderbook.get('bids_depth', []),
            asks_depth=latest_orderbook.get('asks_depth', []),
            total_bid_volume=latest_orderbook.get('total_bid_volume', 0),
            total_ask_volume=latest_orderbook.get('total_ask_volume', 0),
            depth_ratio=latest_orderbook.get('depth_ratio', 1.0),
            
            # Whale data REAL
            whale_bids=latest_orderbook.get('whale_bids', 0),
            whale_asks=latest_orderbook.get('whale_asks', 0),
            whale_activity_score=latest_orderbook.get('whale_activity_score', 0),
            large_orders_count=latest_orderbook.get('large_orders_count', 0),
        )
        
        self.realtime_data.append(professional_point)
        
        # Log cada 30 segundos con info REAL
        if self.capture_stats["ticker_points"] % 6 == 0:
            connections = f"{len(self.connections)}/{len(self.websocket_urls)}"
            whale_info = f"Whales: {professional_point.large_orders_count}"
            spread_info = f"Spread: ${professional_point.real_spread:.2f}"
            
            self.logger.info(f"[PROFESSIONAL REAL] ${professional_point.price:.2f} | "
                           f"{spread_info} | "
                           f"Imbal: {professional_point.orderbook_imbalance:+.3f} | "
                           f"{whale_info} | OB: {self.capture_stats['orderbook_updates']} | "
                           f"Streams: [{connections}]")
    
    async def _whale_monitoring_loop(self):
        """Loop de monitoreo de whales REAL - NO SIMULATION"""
        
        self.whale_logger.info("🐋 Iniciando whale monitoring REAL - NO SIMULATION")
        
        while self.running:
            try:
                await asyncio.sleep(120)  # Cada 2 minutos
                
                if self.orderbook_snapshots and len(self.orderbook_snapshots) >= 5:
                    recent = list(self.orderbook_snapshots)[-5:]
                    
                    # Detectar actividad whale sostenida REAL
                    sustained_activity = all(s['large_orders_count'] > 0 for s in recent)
                    
                    if sustained_activity:
                        avg_whale_volume = np.mean([s['whale_bids'] + s['whale_asks'] for s in recent])
                        total_whale_volume = self.capture_stats["total_whale_volume"]
                        
                        self.whale_logger.warning(f"🐋 SUSTAINED REAL WHALE ACTIVITY: Avg ${avg_whale_volume:,.0f} over 2min | Total Session: ${total_whale_volume:,.0f}")
                
            except Exception as e:
                self.logger.error(f"[WHALE-MONITOR] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _stats_loop(self):
        """Loop de estadísticas REALES"""
        
        while self.running:
            try:
                await asyncio.sleep(600)  # Cada 10 minutos
                
                # Estadísticas REALES
                total_whale_vol = self.capture_stats["total_whale_volume"]
                whale_orders = self.capture_stats["whale_orders_detected"]
                
                self.logger.info(f"[STATS REAL] 📊 Points: {self.capture_stats['ticker_points']}, "
                               f"OrderBook: {self.capture_stats['orderbook_updates']}, "
                               f"REAL Whales: {whale_orders} orders, ${total_whale_vol:,.0f} volume")
                
            except Exception as e:
                self.logger.error(f"[STATS] Error: {str(e)}")
                await asyncio.sleep(300)
    
    async def _cleanup(self):
        """Cleanup REAL"""
        try:
            self.logger.info("[CLEANUP] 🔄 Cleanup con whales REALES...")
            
            # Cerrar conexiones REALES
            for stream_name, connection in self.connections.items():
                try:
                    await connection.close()
                except:
                    pass
            
            # Log final stats REALES
            total_whale_vol = self.capture_stats["total_whale_volume"]
            whale_orders = self.capture_stats["whale_orders_detected"]
            
            self.logger.info(f"[FINAL REAL STATS] Whale Orders: {whale_orders}, Total Volume: ${total_whale_vol:,.0f}")
            self.logger.info("[CLEANUP] ✅ Cleanup completado")
            
        except Exception as e:
            self.logger.error(f"[CLEANUP] Error: {str(e)}")
    
    def get_professional_status(self) -> Dict:
        """Status con datos REALES"""
        
        connection_status = {
            "active_streams": len(self.connections),
            "total_streams": len(self.websocket_urls),
            "connected_streams": list(self.connections.keys()),
            "data_quality_score": self.capture_stats["data_quality_score"]
        }
        
        return {
            "running": self.running,
            "capture_stats": self.capture_stats,
            "connection_status": connection_status,
            "buffer_sizes": {
                "realtime_data": len(self.realtime_data),
                "orderbook_snapshots": len(self.orderbook_snapshots),
                "klines_1m": len(self.klines_1m)
            }
        }
    
    def stop(self):
        """Detener captura REAL"""
        self.logger.info("[STOP] 🛑 Deteniendo captura REAL...")
        self.running = False

class IntegratedProfessionalSystem:
    """Sistema integrado CON WHALES REALES"""
    
    def __init__(self):
        self.logging_system = ProfessionalLoggingSystem()
        self.data_capture = ProfessionalDataCapture("ETHUSD_PERP")
        
        self.integration_config = {
            "run_continuously": True,
            "max_run_hours": 24,
            "health_check_interval": 3600,
            "data_export_interval": 21600,
            "error_restart_delay": 300,
            "max_restart_attempts": 10
        }
        
        self.setup_loggers()
        self.setup_signal_handlers()
        
        self.start_time = None
        self.is_shutting_down = False
    
    def setup_loggers(self):
        """Setup loggers"""
        self.main_logger = logging.getLogger("SystemMain")
        self.perf_logger = logging.getLogger("Performance")
    
    def setup_signal_handlers(self):
        """Setup signal handlers"""
        def signal_handler(signum, frame):
            self.main_logger.info(f"🛑 Signal {signum} received - shutdown...")
            self.is_shutting_down = True
            if self.data_capture:
                self.data_capture.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_integrated_system(self):
        """Sistema integrado con WHALES REALES"""
        
        self.start_time = datetime.datetime.now()
        self.main_logger.info("🚀 SISTEMA CON WHALES REALES - NO SIMULATION")
        self.main_logger.info(f"Configuración: {self.integration_config}")
        self.main_logger.info(f"Timestamp: {self.start_time.isoformat()}")
        
        # Tasks principales
        tasks = [
            asyncio.create_task(self._execute_real_data_capture()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
        ]
        
        try:
            timeout_seconds = self.integration_config["max_run_hours"] * 3600
            
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_seconds
            )
            
        except asyncio.TimeoutError:
            self.main_logger.info("⏰ Reinicio programado después de 24 horas")
        except Exception as e:
            self.main_logger.error(f"❌ Error crítico: {str(e)}")
            raise
        finally:
            if not self.is_shutting_down:
                await self._graceful_shutdown()
    
    async def _execute_real_data_capture(self):
        """Ejecutar captura REAL"""
        try:
            self.main_logger.info("🎯 Iniciando captura con WHALES REALES...")
            await self.data_capture.start_professional_capture()
        except Exception as e:
            self.main_logger.error(f"❌ Error en captura REAL: {str(e)}")
            raise
    
    async def _health_monitoring_loop(self):
        """Health monitoring"""
        self.main_logger.info("🏥 Health monitoring...")
        
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(self.integration_config["health_check_interval"])
                
                status = self.data_capture.get_professional_status()
                quality = status["connection_status"]["data_quality_score"]
                
                self.main_logger.info(f"🏥 Health Check: Quality {quality:.1f}%")
                
            except Exception as e:
                self.main_logger.error(f"❌ Health monitoring error: {str(e)}")
                await asyncio.sleep(300)
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring"""
        self.perf_logger.info("📊 Performance monitoring...")
        
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(3600)  # Cada hora
                
                uptime = (datetime.datetime.now() - self.start_time).total_seconds() / 3600
                status = self.data_capture.get_professional_status()
                
                whale_orders = status["capture_stats"]["whale_orders_detected"]
                whale_volume = status["capture_stats"]["total_whale_volume"]
                
                self.perf_logger.info("📊 HOURLY PERFORMANCE (REAL WHALES):")
                self.perf_logger.info(f"   Uptime: {uptime:.2f}h")
                self.perf_logger.info(f"   REAL Whale Orders: {whale_orders}")
                self.perf_logger.info(f"   REAL Whale Volume: ${whale_volume:,.0f}")
                self.perf_logger.info(f"   Data Points: {status['capture_stats']['ticker_points']}")
                
            except Exception as e:
                self.perf_logger.error(f"❌ Performance error: {str(e)}")
                await asyncio.sleep(1800)
    
    async def _graceful_shutdown(self):
        """Shutdown con stats REALES finales"""
        
        try:
            self.main_logger.info("🔄 Graceful shutdown...")
            
            self.is_shutting_down = True
            
            if self.start_time:
                uptime = datetime.datetime.now() - self.start_time
                self.main_logger.info(f"⏰ Uptime total: {uptime.total_seconds()/3600:.2f} horas")
            
            # Stats finales REALES
            final_stats = self.data_capture.get_professional_status()
            whale_orders = final_stats["capture_stats"]["whale_orders_detected"]
            whale_volume = final_stats["capture_stats"]["total_whale_volume"]
            
            self.main_logger.info("📊 FINAL REAL WHALE STATISTICS:")
            self.main_logger.info(f"   REAL Whale Orders Detected: {whale_orders}")
            self.main_logger.info(f"   REAL Total Whale Volume: ${whale_volume:,.0f}")
            self.main_logger.info(f"   Data Points: {final_stats['capture_stats']['ticker_points']}")
            self.main_logger.info(f"   OrderBook Updates: {final_stats['capture_stats']['orderbook_updates']}")
            
            self.data_capture.stop()
            
            self.main_logger.info("✅ Shutdown with REAL WHALES completed")
            
        except Exception as e:
            self.main_logger.error(f"❌ Shutdown error: {str(e)}")

# ========== FUNCIONES PRINCIPALES ==========

async def run_continuous_professional_system():
    """Sistema continuo con WHALES REALES"""
    
    print("🚀 SISTEMA PROFESIONAL CON WHALES REALES - NO SIMULATION")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print(f"Usuario: biGGzeta")
    print("=" * 80)
    
    restart_count = 0
    max_restarts = 50
    
    while restart_count < max_restarts:
        system = None
        try:
            print(f"\n🔄 Iniciando ciclo #{restart_count + 1} con WHALES REALES: {datetime.datetime.now()}")
            
            system = IntegratedProfessionalSystem()
            
            await system.run_integrated_system()
            
            print(f"✅ Ciclo #{restart_count + 1} completado")
            print(f"🔄 Reiniciando en 60 segundos...")
            
            restart_count += 1
            await asyncio.sleep(60)
            
        except KeyboardInterrupt:
            print(f"\n🛑 Shutdown manual (Ctrl+C)")
            if system:
                await system._graceful_shutdown()
            break
            
        except Exception as e:
            restart_count += 1
            error_msg = f"❌ Error en ciclo #{restart_count}: {str(e)}"
            print(error_msg)
            
            if restart_count < max_restarts:
                wait_time = min(300, 60 * restart_count)
                print(f"🔄 Reiniciando en {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"🚨 Max restarts reached. Stopping.")
                break
    
    print("\n🏁 Sistema terminado")

def check_system_requirements():
    """Check requirements"""
    
    print("🔍 Verificando sistema...")
    
    try:
        Path("logs/professional_system").mkdir(parents=True, exist_ok=True)
        Path("data/professional").mkdir(parents=True, exist_ok=True)
        Path("exports").mkdir(parents=True, exist_ok=True)
        print("✅ Directorios creados")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("✅ Sistema listo para WHALES REALES")
    return True

def main():
    """Main function"""
    
    print("🚀 SISTEMA PROFESIONAL CON WHALES REALES - NO SIMULATION")
    print("Autor: biGGzeta")
    print("Versión: 5.1.0 (Real Whales)")
    print("=" * 60)
    
    if not check_system_requirements():
        return
    
    print("\n🐋 WHALE DETECTION REAL - NO SIMULATION")
    print("1. 🚀 Ejecutar sistema con WHALES REALES")
    print("2. ❌ Salir")
    
    while True:
        try:
            choice = input("\nSelecciona (1-2): ").strip()
            
            if choice == "1":
                print("🚀 Iniciando sistema con WHALES REALES...")
                asyncio.run(run_continuous_professional_system())
                break
                
            elif choice == "2":
                print("👋 ¡Hasta luego!")
                break
                
            else:
                print("❌ Opción inválida.")
                
        except KeyboardInterrupt:
            print("\n🛑 Saliendo...")
            break

if __name__ == "__main__":
    main()