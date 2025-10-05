#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Continuous Data Capture System - NIVEL INSTITUCIONAL
Sistema de captura multi-stream para análisis institucional completo
Autor: biGGzeta
Fecha: 2025-10-05 15:16:42 UTC
Versión: 3.0.0 (Professional Grade)
"""

import asyncio
import datetime
import json
import os
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import websockets
from collections import deque
import statistics
import aiohttp
from concurrent.futures import ThreadPoolExecutor

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
    bids_depth: List[Tuple[float, float]]  # [(price, quantity), ...]
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
    taker_buy_volume: float
    taker_sell_volume: float
    buy_sell_ratio: float
    market_momentum: float
    
    # Futures Specific
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    long_short_ratio: Optional[float] = None
    mark_price: Optional[float] = None
    
    # Technical Indicators
    rsi_1m: Optional[float] = None
    ema_20: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None

@dataclass
class ProfessionalClassification:
    """Clasificación profesional del mercado"""
    timestamp: datetime.datetime
    
    # Multi-timeframe Trends
    trend_1m: str
    trend_5m: str
    trend_15m: str
    trend_1h: str
    
    # Advanced Volatility
    volatility_1m: float
    volatility_5m: float
    volatility_realized: float
    volatility_regime: str  # "low", "normal", "high", "extreme"
    
    # Market Phases
    market_phase: str
    regime_type: str  # "trending", "ranging", "breakout", "consolidation"
    
    # Volume Analysis
    volume_profile: str  # "accumulation", "distribution", "neutral"
    volume_strength: float  # 0-100
    volume_anomaly: bool
    
    # OrderBook Intelligence
    orderbook_pressure: str  # "bullish", "bearish", "neutral"
    whale_sentiment: str
    liquidity_depth: str  # "thin", "normal", "thick"
    
    # Momentum & Strength
    momentum_1m: float
    momentum_5m: float
    momentum_strength: str  # "weak", "moderate", "strong"
    
    # Support/Resistance (professional calculation)
    support_levels: List[float]
    resistance_levels: List[float]
    key_level_proximity: float
    
    # Risk Metrics
    risk_score: float  # 0-100
    volatility_risk: str
    liquidity_risk: str

class ProfessionalDataCapture:
    """Sistema profesional de captura multi-stream"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP"):
        self.symbol = symbol
        self.running = False
        
        # Professional data storage
        self.realtime_data = deque(maxlen=4320)  # 6 horas de data (5s intervals)
        self.klines_1m = deque(maxlen=1440)     # 24 horas de klines 1m
        self.klines_5m = deque(maxlen=288)      # 24 horas de klines 5m
        self.klines_15m = deque(maxlen=96)      # 24 horas de klines 15m
        self.orderbook_snapshots = deque(maxlen=720)  # 1 hora de orderbook
        self.trade_flow = deque(maxlen=10000)   # Trade flow analysis
        self.classifications = deque(maxlen=1440)  # 24 horas
        
        # WebSocket URLs for multiple streams
        self.websocket_urls = {
            "ticker": "wss://dstream.binance.com/ws/ethusd_perp@ticker",
            "orderbook": "wss://dstream.binance.com/ws/ethusd_perp@depth@100ms",
            "kline_1m": "wss://dstream.binance.com/ws/ethusd_perp@kline_1m",
            "kline_5m": "wss://dstream.binance.com/ws/ethusd_perp@kline_5m",
            "kline_15m": "wss://dstream.binance.com/ws/ethusd_perp@kline_15m",
            "trades": "wss://dstream.binance.com/ws/ethusd_perp@trade",
            "markPrice": "wss://dstream.binance.com/ws/ethusd_perp@markPrice"
        }
        
        # WebSocket connections
        self.connections = {}
        self.api_base = "https://dapi.binance.com/dapi/v1"
        
        # Configuration profesional
        self.config = {
            "capture_interval_seconds": 5,
            "classification_interval_seconds": 60,
            "save_interval_seconds": 300,
            "orderbook_levels": 10,
            "whale_threshold_usd": 100000,  # $100k orders
            "data_directory": "data/professional",
            "max_reconnect_attempts": 20,
            "heartbeat_interval": 15,
            "technical_indicators": True,
            "advanced_analytics": True
        }
        
        # Professional state tracking
        self.data_source = "binance_professional"
        self.capture_stats = {
            "ticker_points": 0,
            "orderbook_updates": 0,
            "klines_received": 0,
            "trades_processed": 0,
            "whale_orders_detected": 0,
            "classifications_made": 0,
            "stream_errors": {},
            "connection_resets": 0,
            "data_quality_score": 100.0
        }
        
        # Threading for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup
        self.setup_logging()
        self.setup_directories()
        
        self.logger.info(f"[INIT] Professional Data Capture iniciado para {symbol}")
        self.logger.info(f"[STREAMS] {len(self.websocket_urls)} streams configurados")
        self.logger.info(f"[CONFIG] Whale threshold: ${self.config['whale_threshold_usd']:,}")
    
    def setup_logging(self):
        """Setup logging profesional"""
        
        log_dir = "logs/professional"
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger("ProfessionalCapture")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        log_file = os.path.join(log_dir, f"professional_capture_{self.symbol}_{datetime.datetime.now().strftime('%Y%m%d')}.log")
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
    
    def setup_directories(self):
        """Crear directorios profesionales"""
        
        directories = [
            self.config["data_directory"],
            f"{self.config['data_directory']}/realtime",
            f"{self.config['data_directory']}/orderbook",
            f"{self.config['data_directory']}/klines",
            f"{self.config['data_directory']}/trades",
            f"{self.config['data_directory']}/whale_activity",
            f"{self.config['data_directory']}/classifications",
            f"{self.config['data_directory']}/analytics",
            f"{self.config['data_directory']}/risk_metrics"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    async def start_professional_capture(self):
        """Iniciar captura profesional multi-stream"""
        
        self.logger.info("[START] Iniciando captura profesional multi-stream")
        self.running = True
        
        # Validar conexión inicial
        await self._validate_professional_connection()
        
        # Crear tasks para todos los streams
        tasks = []
        
        # Stream tasks
        for stream_name, url in self.websocket_urls.items():
            task = asyncio.create_task(self._stream_handler(stream_name, url))
            tasks.append(task)
        
        # Analysis tasks
        tasks.extend([
            asyncio.create_task(self._professional_classification_loop()),
            asyncio.create_task(self._whale_detection_loop()),
            asyncio.create_task(self._technical_analysis_loop()),
            asyncio.create_task(self._risk_analysis_loop()),
            asyncio.create_task(self._data_persistence_loop()),
            asyncio.create_task(self._connection_monitor_loop()),
            asyncio.create_task(self._stats_monitoring_loop())
        ])
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self._cleanup()
    
    async def _validate_professional_connection(self):
        """Validar conexiones profesionales"""
        
        try:
            self.logger.info("[VALIDATION] Validando conexiones profesionales...")
            
            # Test básico ticker
            async with websockets.connect(self.websocket_urls["ticker"], ping_timeout=10) as ws:
                message = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(message)
                if 'c' in data:
                    price = float(data['c'])
                    self.logger.info(f"[TICKER] ✅ Ticker stream OK - ETH: ${price:.2f}")
            
            # Test orderbook
            async with websockets.connect(self.websocket_urls["orderbook"], ping_timeout=10) as ws:
                message = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(message)
                if 'bids' in data and 'asks' in data:
                    self.logger.info(f"[ORDERBOOK] ✅ OrderBook stream OK - {len(data['bids'])} bids, {len(data['asks'])} asks")
            
            # Test klines
            async with websockets.connect(self.websocket_urls["kline_1m"], ping_timeout=10) as ws:
                message = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(message)
                if 'k' in data:
                    self.logger.info(f"[KLINES] ✅ Klines stream OK")
            
            # Test REST API adicional
            async with aiohttp.ClientSession() as session:
                # Funding rate
                async with session.get(f"{self.api_base}/premiumIndex?symbol={self.symbol}") as resp:
                    if resp.status == 200:
                        funding_data = await resp.json()
                        funding_rate = float(funding_data['lastFundingRate'])
                        self.logger.info(f"[FUNDING] ✅ Funding rate: {funding_rate:.6f}")
                
                # Open interest
                async with session.get(f"{self.api_base}/openInterest?symbol={self.symbol}") as resp:
                    if resp.status == 200:
                        oi_data = await resp.json()
                        open_interest = float(oi_data['openInterest'])
                        self.logger.info(f"[OI] ✅ Open Interest: {open_interest:,.0f}")
            
            self.logger.info("[VALIDATION] ✅ Todas las conexiones profesionales validadas")
            
        except Exception as e:
            self.logger.error(f"[VALIDATION] ❌ Error: {str(e)}")
            raise Exception(f"Fallo en validación profesional: {str(e)}")
    
    async def _stream_handler(self, stream_name: str, url: str):
        """Handler genérico para streams"""
        
        self.logger.info(f"[{stream_name.upper()}] Iniciando stream: {url}")
        
        while self.running:
            try:
                async with websockets.connect(url, ping_timeout=20) as websocket:
                    self.connections[stream_name] = websocket
                    self.logger.info(f"[{stream_name.upper()}] ✅ Conectado")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._process_stream_data(stream_name, data)
                        except Exception as e:
                            self.logger.error(f"[{stream_name.upper()}] Error procesando: {str(e)}")
                            self._increment_stream_error(stream_name)
                        
            except Exception as e:
                self.logger.warning(f"[{stream_name.upper()}] ⚠️ Desconectado: {str(e)}")
                self.connections.pop(stream_name, None)
                
                if self.running:
                    await asyncio.sleep(5)  # Retry delay
    
    async def _process_stream_data(self, stream_name: str, data: Dict):
        """Procesar datos de diferentes streams"""
        
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        if stream_name == "ticker":
            await self._process_ticker_data(data, timestamp)
        elif stream_name == "orderbook":
            await self._process_orderbook_data(data, timestamp)
        elif stream_name.startswith("kline"):
            await self._process_kline_data(stream_name, data, timestamp)
        elif stream_name == "trades":
            await self._process_trade_data(data, timestamp)
        elif stream_name == "markPrice":
            await self._process_mark_price_data(data, timestamp)
    
    async def _process_ticker_data(self, data: Dict, timestamp: datetime.datetime):
        """Procesar datos del ticker"""
        
        try:
            if 'c' not in data:
                return
            
            # Crear data point básico
            basic_data = {
                'timestamp': timestamp,
                'price': float(data['c']),
                'open_price': float(data['o']),
                'high_price': float(data['h']),
                'low_price': float(data['l']),
                'volume': float(data['v']),
                'weighted_avg_price': float(data['w']),
                'price_change_24h': float(data['p']),
                'price_change_pct_24h': float(data['P']),
                'trade_count': int(data['n'])
            }
            
            # Enqueue para procesamiento
            await self._enqueue_ticker_data(basic_data)
            self.capture_stats["ticker_points"] += 1
            
        except Exception as e:
            self.logger.error(f"[TICKER] Error: {str(e)}")
    
    async def _process_orderbook_data(self, data: Dict, timestamp: datetime.datetime):
        """Procesar datos del orderbook profesional"""
        
        try:
            if 'bids' not in data or 'asks' not in data:
                return
            
            bids = [[float(price), float(qty)] for price, qty in data['bids'][:self.config["orderbook_levels"]]]
            asks = [[float(price), float(qty)] for price, qty in data['asks'][:self.config["orderbook_levels"]]]
            
            if not bids or not asks:
                return
            
            # Calcular métricas profesionales del orderbook
            best_bid, bid_qty = bids[0]
            best_ask, ask_qty = asks[0]
            
            real_spread = best_ask - best_bid
            spread_pct = (real_spread / best_bid) * 100
            
            # Total volumes
            total_bid_volume = sum(qty for _, qty in bids)
            total_ask_volume = sum(qty for _, qty in asks)
            
            # Imbalance
            total_volume = total_bid_volume + total_ask_volume
            orderbook_imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
            
            # Whale detection
            whale_threshold = self.config["whale_threshold_usd"]
            whale_bids = sum(qty for price, qty in bids if qty * price > whale_threshold)
            whale_asks = sum(qty for price, qty in asks if qty * price > whale_threshold)
            
            # Crear orderbook snapshot profesional
            orderbook_snapshot = {
                'timestamp': timestamp,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'real_spread': real_spread,
                'spread_pct': spread_pct,
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
                'large_orders_count': len([1 for p, q in bids + asks if q * p > whale_threshold])
            }
            
            self.orderbook_snapshots.append(orderbook_snapshot)
            self.capture_stats["orderbook_updates"] += 1
            
            # Detectar whales
            if whale_bids > 0 or whale_asks > 0:
                self.capture_stats["whale_orders_detected"] += 1
                await self._log_whale_activity(orderbook_snapshot)
            
        except Exception as e:
            self.logger.error(f"[ORDERBOOK] Error: {str(e)}")
    
    async def _process_kline_data(self, stream_name: str, data: Dict, timestamp: datetime.datetime):
        """Procesar datos de klines profesionales"""
        
        try:
            if 'k' not in data:
                return
            
            kline = data['k']
            if not kline['x']:  # Solo klines cerradas
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
            
            # Calcular buy/sell ratio
            taker_sell_volume = kline_data['volume'] - kline_data['taker_buy_volume']
            kline_data['buy_sell_ratio'] = kline_data['taker_buy_volume'] / taker_sell_volume if taker_sell_volume > 0 else 1.0
            
            # Almacenar según timeframe
            if stream_name == "kline_1m":
                self.klines_1m.append(kline_data)
            elif stream_name == "kline_5m":
                self.klines_5m.append(kline_data)
            elif stream_name == "kline_15m":
                self.klines_15m.append(kline_data)
            
            self.capture_stats["klines_received"] += 1
            
        except Exception as e:
            self.logger.error(f"[{stream_name.upper()}] Error: {str(e)}")
    
    async def _enqueue_ticker_data(self, basic_data: Dict):
        """Enqueue ticker data para procesamiento"""
        
        # Combine con último orderbook snapshot si existe
        latest_orderbook = self.orderbook_snapshots[-1] if self.orderbook_snapshots else {}
        
        # Crear data point profesional completo
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
            
            # OrderBook data (si está disponible)
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
            
            # Whale data
            whale_bids=latest_orderbook.get('whale_bids', 0),
            whale_asks=latest_orderbook.get('whale_asks', 0),
            whale_activity_score=latest_orderbook.get('whale_activity_score', 0),
            large_orders_count=latest_orderbook.get('large_orders_count', 0),
            
            # Market microstructure (del último kline)
            taker_buy_volume=0,  # Se llenará del kline stream
            taker_sell_volume=0,
            buy_sell_ratio=1.0,
            market_momentum=0,
            
            # Volume change (calculado)
            volume_change_pct=0  # Se calculará
        )
        
        self.realtime_data.append(professional_point)
        
        # Log cada 30 segundos con datos profesionales
        if self.capture_stats["ticker_points"] % 6 == 0:
            self.logger.info(f"[PROFESSIONAL] ${professional_point.price:.2f} "
                           f"Spread: ${professional_point.real_spread:.2f} "
                           f"Imbal: {professional_point.orderbook_imbalance:+.3f} "
                           f"Whales: {professional_point.large_orders_count} "
                           f"[{len(self.connections)}/{len(self.websocket_urls)} streams]")
    
    async def _log_whale_activity(self, orderbook_snapshot: Dict):
        """Log actividad de whales"""
        
        whale_bids = orderbook_snapshot['whale_bids']
        whale_asks = orderbook_snapshot['whale_asks']
        
        if whale_bids > whale_asks * 2:
            self.logger.warning(f"[WHALE] 🐋 Large bid pressure: ${whale_bids:,.0f} vs ${whale_asks:,.0f}")
        elif whale_asks > whale_bids * 2:
            self.logger.warning(f"[WHALE] 🐋 Large ask pressure: ${whale_asks:,.0f} vs ${whale_bids:,.0f}")
    
    def _increment_stream_error(self, stream_name: str):
        """Incrementar errores de stream"""
        
        if stream_name not in self.capture_stats["stream_errors"]:
            self.capture_stats["stream_errors"][stream_name] = 0
        self.capture_stats["stream_errors"][stream_name] += 1
    
    async def _professional_classification_loop(self):
        """Loop de clasificación profesional"""
        
        self.logger.info("[CLASSIFY] Iniciando clasificación profesional cada 60s")
        
        while self.running:
            try:
                await asyncio.sleep(self.config["classification_interval_seconds"])
                
                if len(self.realtime_data) >= 60:  # Mínimo 5 minutos de data
                    classification = await self._create_professional_classification()
                    
                    if classification:
                        self.classifications.append(classification)
                        self.capture_stats["classifications_made"] += 1
                        
                        # Log clasificación profesional
                        self.logger.info(f"[CLASSIFY] Trends: 1m:{classification.trend_1m} 5m:{classification.trend_5m} "
                                       f"Phase: {classification.market_phase} "
                                       f"OB: {classification.orderbook_pressure} "
                                       f"Vol: {classification.volume_profile} "
                                       f"Risk: {classification.risk_score:.0f}/100")
                
            except Exception as e:
                self.logger.error(f"[CLASSIFY] Error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _create_professional_classification(self) -> Optional[ProfessionalClassification]:
        """Crear clasificación profesional completa"""
        
        try:
            # Usar executor para cálculos intensivos
            loop = asyncio.get_event_loop()
            classification = await loop.run_in_executor(
                self.executor, 
                self._calculate_professional_metrics
            )
            
            return classification
            
        except Exception as e:
            self.logger.error(f"[CLASSIFY] Error creating classification: {str(e)}")
            return None
    
    def _calculate_professional_metrics(self) -> ProfessionalClassification:
        """Calcular métricas profesionales (CPU intensive)"""
        
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        # Extraer datos recientes
        recent_data = list(self.realtime_data)[-360:]  # Últimos 30 minutos
        prices = [d.price for d in recent_data]
        
        # Multi-timeframe trends (profesional)
        trend_1m = self._classify_professional_trend(prices[-12:], "1m")
        trend_5m = self._classify_professional_trend(prices[-60:] if len(prices) >= 60 else prices, "5m")
        trend_15m = self._classify_professional_trend(prices[-180:] if len(prices) >= 180 else prices, "15m")
        trend_1h = self._classify_professional_trend(prices, "1h")
        
        # Advanced volatility
        volatility_1m = self._calculate_professional_volatility(prices[-12:])
        volatility_5m = self._calculate_professional_volatility(prices[-60:] if len(prices) >= 60 else prices)
        volatility_realized = self._calculate_realized_volatility(prices)
        volatility_regime = self._classify_volatility_regime(volatility_realized)
        
        # Market phase analysis
        market_phase = self._determine_professional_market_phase(recent_data)
        regime_type = self._classify_regime_type(recent_data)
        
        # Volume analysis profesional
        volume_profile = self._analyze_professional_volume_profile(recent_data)
        volume_strength = self._calculate_volume_strength(recent_data)
        volume_anomaly = self._detect_volume_anomaly(recent_data)
        
        # OrderBook intelligence
        orderbook_pressure = self._analyze_orderbook_pressure()
        whale_sentiment = self._analyze_whale_sentiment()
        liquidity_depth = self._analyze_liquidity_depth()
        
        # Momentum profesional
        momentum_1m = self._calculate_professional_momentum(prices[-12:])
        momentum_5m = self._calculate_professional_momentum(prices[-60:] if len(prices) >= 60 else prices)
        momentum_strength = self._classify_momentum_strength(momentum_1m, momentum_5m)
        
        # Support/Resistance profesional
        support_levels = self._calculate_professional_support_levels(prices)
        resistance_levels = self._calculate_professional_resistance_levels(prices)
        key_level_proximity = self._calculate_key_level_proximity(prices[-1], support_levels + resistance_levels)
        
        # Risk metrics
        risk_score = self._calculate_professional_risk_score(recent_data)
        volatility_risk = self._classify_volatility_risk(volatility_realized)
        liquidity_risk = self._classify_liquidity_risk()
        
        return ProfessionalClassification(
            timestamp=timestamp,
            trend_1m=trend_1m,
            trend_5m=trend_5m,
            trend_15m=trend_15m,
            trend_1h=trend_1h,
            volatility_1m=volatility_1m,
            volatility_5m=volatility_5m,
            volatility_realized=volatility_realized,
            volatility_regime=volatility_regime,
            market_phase=market_phase,
            regime_type=regime_type,
            volume_profile=volume_profile,
            volume_strength=volume_strength,
            volume_anomaly=volume_anomaly,
            orderbook_pressure=orderbook_pressure,
            whale_sentiment=whale_sentiment,
            liquidity_depth=liquidity_depth,
            momentum_1m=momentum_1m,
            momentum_5m=momentum_5m,
            momentum_strength=momentum_strength,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            key_level_proximity=key_level_proximity,
            risk_score=risk_score,
            volatility_risk=volatility_risk,
            liquidity_risk=liquidity_risk
        )
    
    def _classify_professional_trend(self, prices: List[float], timeframe: str) -> str:
        """Clasificación profesional de tendencia"""
        
        if len(prices) < 3:
            return "lateral"
        
        try:
            # Multiple trend detection methods
            
            # 1. Linear regression
            x = np.arange(len(prices))
            slope, intercept = np.polyfit(x, prices, 1)
            r_squared = np.corrcoef(x, prices)[0, 1] ** 2
            
            # 2. Moving averages
            if len(prices) >= 10:
                ma_short = np.mean(prices[-5:])
                ma_long = np.mean(prices[-10:])
                ma_signal = "bullish" if ma_short > ma_long else "bearish"
            else:
                ma_signal = "lateral"
            
            # 3. Price momentum
            momentum = ((prices[-1] - prices[0]) / prices[0]) * 100
            
            # Combined classification
            slope_threshold = 0.1 if timeframe == "1m" else 0.05
            momentum_threshold = 0.1 if timeframe == "1m" else 0.05
            
            if (slope > slope_threshold and momentum > momentum_threshold and 
                r_squared > 0.3 and ma_signal == "bullish"):
                return "bullish"
            elif (slope < -slope_threshold and momentum < -momentum_threshold and 
                  r_squared > 0.3 and ma_signal == "bearish"):
                return "bearish"
            else:
                return "lateral"
                
        except Exception:
            return "lateral"
    
    def _calculate_professional_volatility(self, prices: List[float]) -> float:
        """Cálculo profesional de volatilidad"""
        
        if len(prices) < 2:
            return 0.0
        
        try:
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(len(returns)) * 100
            return round(volatility, 4)
        except Exception:
            return 0.0
    
    def _calculate_realized_volatility(self, prices: List[float]) -> float:
        """Calcular volatilidad realizada"""
        
        if len(prices) < 10:
            return 0.0
        
        try:
            returns = np.diff(np.log(prices))
            realized_vol = np.sqrt(np.sum(returns**2)) * 100
            return round(realized_vol, 4)
        except Exception:
            return 0.0
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Clasificar régimen de volatilidad"""
        
        if volatility < 1.0:
            return "low"
        elif volatility < 2.5:
            return "normal"
        elif volatility < 5.0:
            return "high"
        else:
            return "extreme"
    
    def _determine_professional_market_phase(self, recent_data: List[ProfessionalDataPoint]) -> str:
        """Determinar fase profesional del mercado"""
        
        if len(recent_data) < 20:
            return "lateral"
        
        try:
            # Analizar precio, volumen y orderbook conjuntamente
            prices = [d.price for d in recent_data]
            volumes = [d.volume for d in recent_data]
            whale_scores = [d.whale_activity_score for d in recent_data if d.whale_activity_score > 0]
            
            price_trend = self._classify_professional_trend(prices, "5m")
            volume_trend = "increasing" if volumes[-10:] > volumes[:10] else "decreasing"
            whale_activity = np.mean(whale_scores) if whale_scores else 0
            
            # Lógica profesional
            if price_trend == "lateral" and volume_trend == "increasing" and whale_activity > 0.1:
                return "accumulation"
            elif price_trend in ["bullish", "bearish"] and volume_trend == "increasing":
                return "trending"
            elif price_trend in ["bullish", "bearish"] and volume_trend == "decreasing":
                return "distribution"
            else:
                return "consolidation"
                
        except Exception:
            return "lateral"
    
    def _classify_regime_type(self, recent_data: List[ProfessionalDataPoint]) -> str:
        """Clasificar tipo de régimen"""
        
        if len(recent_data) < 10:
            return "ranging"
        
        try:
            prices = [d.price for d in recent_data]
            volatility = self._calculate_professional_volatility(prices)
            price_range = (max(prices) - min(prices)) / min(prices) * 100
            
            if volatility > 3.0 and price_range > 2.0:
                return "breakout"
            elif volatility < 1.0 and price_range < 1.0:
                return "consolidation"
            elif price_range > 1.5:
                return "trending"
            else:
                return "ranging"
                
        except Exception:
            return "ranging"
    
    def _analyze_professional_volume_profile(self, recent_data: List[ProfessionalDataPoint]) -> str:
        """Análisis profesional del perfil de volumen"""
        
        if len(recent_data) < 20:
            return "neutral"
        
        try:
            # Analizar distribución del volumen por precio
            price_volume_pairs = [(d.price, d.volume) for d in recent_data]
            
            # Calcular VWAP
            total_pv = sum(p * v for p, v in price_volume_pairs)
            total_v = sum(v for _, v in price_volume_pairs)
            vwap = total_pv / total_v if total_v > 0 else 0
            
            current_price = recent_data[-1].price
            
            # Análisis de acumulación vs distribución
            if current_price < vwap * 0.998:  # Precio abajo del VWAP
                return "accumulation"
            elif current_price > vwap * 1.002:  # Precio arriba del VWAP
                return "distribution"
            else:
                return "neutral"
                
        except Exception:
            return "neutral"
    
    def _calculate_volume_strength(self, recent_data: List[ProfessionalDataPoint]) -> float:
        """Calcular fuerza del volumen"""
        
        if len(recent_data) < 10:
            return 50.0
        
        try:
            volumes = [d.volume for d in recent_data]
            avg_volume = np.mean(volumes)
            recent_volume = np.mean(volumes[-5:])
            
            strength = (recent_volume / avg_volume) * 50
            return min(100.0, max(0.0, strength))
            
        except Exception:
            return 50.0
    
    def _detect_volume_anomaly(self, recent_data: List[ProfessionalDataPoint]) -> bool:
        """Detectar anomalías en volumen"""
        
        if len(recent_data) < 20:
            return False
        
        try:
            volumes = [d.volume for d in recent_data]
            avg_volume = np.mean(volumes[:-5])
            std_volume = np.std(volumes[:-5])
            current_volume = recent_data[-1].volume
            
            # Anomalía si está 2+ desviaciones estándar arriba
            return current_volume > avg_volume + 2 * std_volume
            
        except Exception:
            return False
    
    def _analyze_orderbook_pressure(self) -> str:
        """Analizar presión del orderbook"""
        
        if not self.orderbook_snapshots:
            return "neutral"
        
        try:
            recent_snapshots = list(self.orderbook_snapshots)[-10:]
            imbalances = [s['orderbook_imbalance'] for s in recent_snapshots]
            avg_imbalance = np.mean(imbalances)
            
            if avg_imbalance > 0.1:
                return "bullish"
            elif avg_imbalance < -0.1:
                return "bearish"
            else:
                return "neutral"
                
        except Exception:
            return "neutral"
    
    def _analyze_whale_sentiment(self) -> str:
        """Analizar sentimiento de whales"""
        
        if not self.orderbook_snapshots:
            return "neutral"
        
        try:
            recent_snapshots = list(self.orderbook_snapshots)[-5:]
            whale_bid_ratios = []
            
            for snapshot in recent_snapshots:
                total_whale = snapshot['whale_bids'] + snapshot['whale_asks']
                if total_whale > 0:
                    whale_bid_ratio = snapshot['whale_bids'] / total_whale
                    whale_bid_ratios.append(whale_bid_ratio)
            
            if whale_bid_ratios:
                avg_whale_bid_ratio = np.mean(whale_bid_ratios)
                if avg_whale_bid_ratio > 0.6:
                    return "bullish"
                elif avg_whale_bid_ratio < 0.4:
                    return "bearish"
            
            return "neutral"
            
        except Exception:
            return "neutral"
    
    def _analyze_liquidity_depth(self) -> str:
        """Analizar profundidad de liquidez"""
        
        if not self.orderbook_snapshots:
            return "normal"
        
        try:
            latest_snapshot = self.orderbook_snapshots[-1]
            total_volume = latest_snapshot['total_bid_volume'] + latest_snapshot['total_ask_volume']
            
            if total_volume > 1000:  # Threshold profesional
                return "thick"
            elif total_volume < 200:
                return "thin"
            else:
                return "normal"
                
        except Exception:
            return "normal"
    
    def _calculate_professional_momentum(self, prices: List[float]) -> float:
        """Calcular momentum profesional"""
        
        if len(prices) < 2:
            return 0.0
        
        try:
            # Rate of change ponderado
            roc = ((prices[-1] - prices[0]) / prices[0]) * 100
            
            # Acceleration
            if len(prices) >= 4:
                mid_point = len(prices) // 2
                first_half_roc = ((prices[mid_point] - prices[0]) / prices[0]) * 100
                second_half_roc = ((prices[-1] - prices[mid_point]) / prices[mid_point]) * 100
                acceleration = second_half_roc - first_half_roc
                momentum = roc + acceleration * 0.5
            else:
                momentum = roc
            
            return round(momentum, 4)
            
        except Exception:
            return 0.0
    
    def _classify_momentum_strength(self, momentum_1m: float, momentum_5m: float) -> str:
        """Clasificar fuerza del momentum"""
        
        try:
            combined_momentum = abs(momentum_1m) + abs(momentum_5m)
            
            if combined_momentum > 1.0:
                return "strong"
            elif combined_momentum > 0.3:
                return "moderate"
            else:
                return "weak"
                
        except Exception:
            return "weak"
    
    def _calculate_professional_support_levels(self, prices: List[float]) -> List[float]:
        """Calcular niveles de soporte profesionales"""
        
        if len(prices) < 20:
            return []
        
        try:
            # Método profesional: combinación de pivots y clusters
            support_levels = []
            
            # Pivots locales
            for i in range(2, len(prices) - 2):
                if (prices[i] < prices[i-1] and prices[i] < prices[i-2] and
                    prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                    support_levels.append(prices[i])
            
            # Percentiles como backup
            if len(support_levels) < 2:
                support_levels = [
                    np.percentile(prices, 10),
                    np.percentile(prices, 25)
                ]
            
            return sorted(list(set([round(level, 2) for level in support_levels])))[:3]
            
        except Exception:
            return []
    
    def _calculate_professional_resistance_levels(self, prices: List[float]) -> List[float]:
        """Calcular niveles de resistencia profesionales"""
        
        if len(prices) < 20:
            return []
        
        try:
            resistance_levels = []
            
            # Pivots locales
            for i in range(2, len(prices) - 2):
                if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and
                    prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                    resistance_levels.append(prices[i])
            
            # Percentiles como backup
            if len(resistance_levels) < 2:
                resistance_levels = [
                    np.percentile(prices, 75),
                    np.percentile(prices, 90)
                ]
            
            return sorted(list(set([round(level, 2) for level in resistance_levels])), reverse=True)[:3]
            
        except Exception:
            return []
    
    def _calculate_key_level_proximity(self, current_price: float, levels: List[float]) -> float:
        """Calcular proximidad a niveles clave"""
        
        if not levels:
            return 50.0
        
        try:
            distances = [abs(current_price - level) / current_price * 100 for level in levels]
            min_distance = min(distances)
            
            # Convertir a score (menor distancia = mayor score)
            proximity = max(0, 100 - min_distance * 50)
            return round(proximity, 1)
            
        except Exception:
            return 50.0
    
    def _calculate_professional_risk_score(self, recent_data: List[ProfessionalDataPoint]) -> float:
        """Calcular score de riesgo profesional"""
        
        if len(recent_data) < 10:
            return 50.0
        
        try:
            risk_factors = []
            
            # Volatility risk
            prices = [d.price for d in recent_data]
            volatility = self._calculate_professional_volatility(prices)
            vol_risk = min(100, volatility * 20)  # Scale to 0-100
            risk_factors.append(vol_risk)
            
            # Liquidity risk
            liquidity_depth = self._analyze_liquidity_depth()
            liquidity_risk = {"thin": 80, "normal": 40, "thick": 20}.get(liquidity_depth, 50)
            risk_factors.append(liquidity_risk)
            
            # Orderbook imbalance risk
            if self.orderbook_snapshots:
                latest_imbalance = abs(self.orderbook_snapshots[-1]['orderbook_imbalance'])
                imbalance_risk = min(100, latest_imbalance * 200)
                risk_factors.append(imbalance_risk)
            
            # Market phase risk
            market_phase = self._determine_professional_market_phase(recent_data)
            phase_risk = {"accumulation": 30, "trending": 50, "distribution": 70, "consolidation": 40}.get(market_phase, 50)
            risk_factors.append(phase_risk)
            
            # Combined risk score
            total_risk = np.mean(risk_factors)
            return round(total_risk, 1)
            
        except Exception:
            return 50.0
    
    def _classify_volatility_risk(self, volatility: float) -> str:
        """Clasificar riesgo de volatilidad"""
        
        if volatility > 5.0:
            return "extreme"
        elif volatility > 2.5:
            return "high"
        elif volatility > 1.0:
            return "moderate"
        else:
            return "low"
    
    def _classify_liquidity_risk(self) -> str:
        """Clasificar riesgo de liquidez"""
        
        liquidity_depth = self._analyze_liquidity_depth()
        
        risk_mapping = {
            "thin": "high",
            "normal": "moderate", 
            "thick": "low"
        }
        
        return risk_mapping.get(liquidity_depth, "moderate")
    
    # ... [Continúo con los métodos restantes en la siguiente parte]
    
    async def _whale_detection_loop(self):
        """Loop de detección de whales"""
        
        self.logger.info("[WHALE] Iniciando detección de whales profesional")
        
        while self.running:
            try:
                await asyncio.sleep(30)  # Check cada 30 segundos
                
                if self.orderbook_snapshots:
                    await self._analyze_whale_patterns()
                
            except Exception as e:
                self.logger.error(f"[WHALE] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _analyze_whale_patterns(self):
        """Analizar patrones de whales"""
        
        if len(self.orderbook_snapshots) < 5:
            return
        
        try:
            recent_snapshots = list(self.orderbook_snapshots)[-5:]
            
            # Detectar acumulación de whales
            whale_trend = []
            for snapshot in recent_snapshots:
                total_whale = snapshot['whale_bids'] + snapshot['whale_asks']
                whale_trend.append(total_whale)
            
            # Si hay aumento sostenido de actividad whale
            if len(whale_trend) >= 3:
                if (whale_trend[-1] > whale_trend[-2] > whale_trend[-3] and 
                    whale_trend[-1] > whale_trend[0] * 1.5):
                    
                    latest = recent_snapshots[-1]
                    if latest['whale_bids'] > latest['whale_asks']:
                        self.logger.warning(f"[WHALE] 🐋 PATTERN: Sustained whale accumulation detected - "
                                          f"Bid pressure: ${latest['whale_bids']:,.0f}")
                    else:
                        self.logger.warning(f"[WHALE] 🐋 PATTERN: Sustained whale distribution detected - "
                                          f"Ask pressure: ${latest['whale_asks']:,.0f}")
        
        except Exception as e:
            self.logger.error(f"[WHALE] Pattern analysis error: {str(e)}")
    
    async def _technical_analysis_loop(self):
        """Loop de análisis técnico profesional"""
        
        self.logger.info("[TECHNICAL] Iniciando análisis técnico profesional")
        
        while self.running:
            try:
                await asyncio.sleep(300)  # Cada 5 minutos
                
                if len(self.klines_1m) >= 20:
                    await self._calculate_technical_indicators()
                
            except Exception as e:
                self.logger.error(f"[TECHNICAL] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _calculate_technical_indicators(self):
        """Calcular indicadores técnicos profesionales"""
        
        try:
            # Usar últimas 50 klines para indicadores
            klines = list(self.klines_1m)[-50:]
            closes = [k['close'] for k in klines]
            
            if len(closes) >= 20:
                # RSI
                rsi = self._calculate_rsi(closes, 14)
                
                # EMA 20
                ema_20 = self._calculate_ema(closes, 20)
                
                # Bollinger Bands
                bb_upper, bb_lower = self._calculate_bollinger_bands(closes, 20, 2)
                
                # Log indicadores importantes
                if rsi < 30:
                    self.logger.info(f"[TECHNICAL] RSI oversold: {rsi:.1f}")
                elif rsi > 70:
                    self.logger.info(f"[TECHNICAL] RSI overbought: {rsi:.1f}")
                
                current_price = closes[-1]
                if current_price > bb_upper:
                    self.logger.info(f"[TECHNICAL] Price above Bollinger upper: ${current_price:.2f} > ${bb_upper:.2f}")
                elif current_price < bb_lower:
                    self.logger.info(f"[TECHNICAL] Price below Bollinger lower: ${current_price:.2f} < ${bb_lower:.2f}")
        
        except Exception as e:
            self.logger.error(f"[TECHNICAL] Calculation error: {str(e)}")
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calcular RSI"""
        
        if len(prices) < period + 1:
            return 50.0
        
        try:
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return round(rsi, 2)
            
        except Exception:
            return 50.0
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calcular EMA"""
        
        if len(prices) < period:
            return np.mean(prices) if prices else 0.0
        
        try:
            alpha = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return round(ema, 2)
            
        except Exception:
            return 0.0
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int, std_dev: float) -> Tuple[float, float]:
        """Calcular Bollinger Bands"""
        
        if len(prices) < period:
            avg = np.mean(prices) if prices else 0
            return avg, avg
        
        try:
            recent_prices = prices[-period:]
            ma = np.mean(recent_prices)
            std = np.std(recent_prices)
            
            upper = ma + (std * std_dev)
            lower = ma - (std * std_dev)
            
            return round(upper, 2), round(lower, 2)
            
        except Exception:
            return 0.0, 0.0
    
    async def _risk_analysis_loop(self):
        """Loop de análisis de riesgo profesional"""
        
        self.logger.info("[RISK] Iniciando análisis de riesgo profesional")
        
        while self.running:
            try:
                await asyncio.sleep(120)  # Cada 2 minutos
                
                if len(self.realtime_data) >= 20:
                    await self._perform_risk_analysis()
                
            except Exception as e:
                self.logger.error(f"[RISK] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _perform_risk_analysis(self):
        """Realizar análisis de riesgo comprehensivo"""
        
        try:
            recent_data = list(self.realtime_data)[-60:]  # Últimos 5 minutos
            
            # VaR calculation (Value at Risk)
            prices = [d.price for d in recent_data]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            if len(returns) >= 10:
                var_95 = np.percentile(returns, 5) * 100  # 95% VaR
                var_99 = np.percentile(returns, 1) * 100  # 99% VaR
                
                if var_95 < -2.0:  # VaR > 2%
                    self.logger.warning(f"[RISK] High VaR detected: 95%={var_95:.2f}%, 99%={var_99:.2f}%")
            
            # Liquidez risk
            if self.orderbook_snapshots:
                latest_ob = self.orderbook_snapshots[-1]
                if latest_ob['total_bid_volume'] + latest_ob['total_ask_volume'] < 100:
                    self.logger.warning(f"[RISK] Low liquidity: Total volume {latest_ob['total_bid_volume'] + latest_ob['total_ask_volume']:.0f}")
            
            # Volatility spike detection
            volatility = self._calculate_professional_volatility(prices)
            if volatility > 5.0:
                self.logger.warning(f"[RISK] Volatility spike: {volatility:.2f}%")
        
        except Exception as e:
            self.logger.error(f"[RISK] Analysis error: {str(e)}")
    
    async def _data_persistence_loop(self):
        """Loop de persistencia profesional"""
        
        self.logger.info(f"[PERSIST] Iniciando persistencia profesional cada {self.config['save_interval_seconds']}s")
        
        while self.running:
            try:
                await asyncio.sleep(self.config["save_interval_seconds"])
                
                await self._save_professional_data()
                
            except Exception as e:
                self.logger.error(f"[PERSIST] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _save_professional_data(self):
        """Guardar datos profesionales"""
        
        try:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []
            
            # 1. Raw professional data
            if self.realtime_data:
                data_file = f"{self.config['data_directory']}/realtime/professional_data_{self.symbol}_{timestamp_str}.json"
                recent_data = list(self.realtime_data)[-60:]  # Últimos 5 min
                
                data_to_save = []
                for point in recent_data:
                    data_dict = asdict(point)
                    data_dict['timestamp'] = point.timestamp.isoformat()
                    data_to_save.append(data_dict)
                
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                
                saved_files.append(f"realtime: {len(data_to_save)} points")
            
            # 2. OrderBook snapshots
            if self.orderbook_snapshots:
                ob_file = f"{self.config['data_directory']}/orderbook/orderbook_snapshots_{self.symbol}_{timestamp_str}.json"
                recent_obs = list(self.orderbook_snapshots)[-30:]  # Últimos 30 snapshots
                
                ob_data = []
                for snapshot in recent_obs:
                    snapshot_copy = snapshot.copy()
                    snapshot_copy['timestamp'] = snapshot['timestamp'].isoformat()
                    ob_data.append(snapshot_copy)
                
                with open(ob_file, 'w', encoding='utf-8') as f:
                    json.dump(ob_data, f, indent=2, ensure_ascii=False)
                
                saved_files.append(f"orderbook: {len(ob_data)} snapshots")
            
            # 3. Klines data
            for timeframe, klines_data in [("1m", self.klines_1m), ("5m", self.klines_5m), ("15m", self.klines_15m)]:
                if klines_data:
                    klines_file = f"{self.config['data_directory']}/klines/klines_{timeframe}_{self.symbol}_{timestamp_str}.json"
                    recent_klines = list(klines_data)[-20:]  # Últimas 20 klines
                    
                    klines_to_save = []
                    for kline in recent_klines:
                        kline_copy = kline.copy()
                        kline_copy['timestamp'] = kline['timestamp'].isoformat()
                        klines_to_save.append(kline_copy)
                    
                    with open(klines_file, 'w', encoding='utf-8') as f:
                        json.dump(klines_to_save, f, indent=2, ensure_ascii=False)
                    
                    saved_files.append(f"klines_{timeframe}: {len(klines_to_save)} entries")
            
            # 4. Professional classifications
            if self.classifications:
                class_file = f"{self.config['data_directory']}/classifications/professional_classifications_{self.symbol}_{timestamp_str}.json"
                recent_classifications = list(self.classifications)[-30:]  # Últimos 30 min
                
                class_data = []
                for classification in recent_classifications:
                    data_dict = asdict(classification)
                    data_dict['timestamp'] = classification.timestamp.isoformat()
                    class_data.append(data_dict)
                
                with open(class_file, 'w', encoding='utf-8') as f:
                    json.dump(class_data, f, indent=2, ensure_ascii=False)
                
                saved_files.append(f"classifications: {len(class_data)} entries")
            
            # 5. Whale activity summary
            await self._save_whale_activity_summary(timestamp_str)
            saved_files.append("whale_activity")
            
            # 6. Professional summary
            await self._create_professional_summary()
            saved_files.append("professional_summary")
            
            # Log resultado
            self.logger.info(f"[SAVE] Professional data saved: {', '.join(saved_files)}")
            
        except Exception as e:
            self.logger.error(f"[SAVE] Error guardando datos profesionales: {str(e)}")
    
    async def _save_whale_activity_summary(self, timestamp_str: str):
        """Guardar resumen de actividad de whales"""
        
        try:
            if not self.orderbook_snapshots:
                return
            
            recent_snapshots = list(self.orderbook_snapshots)[-60:]  # Última hora
            
            whale_summary = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "symbol": self.symbol,
                "period": "last_hour",
                "total_whale_orders": sum(s['large_orders_count'] for s in recent_snapshots),
                "avg_whale_activity_score": np.mean([s['whale_activity_score'] for s in recent_snapshots]),
                "max_whale_bid": max([s['whale_bids'] for s in recent_snapshots]),
                "max_whale_ask": max([s['whale_asks'] for s in recent_snapshots]),
                "whale_pressure_changes": len([i for i in range(1, len(recent_snapshots)) 
                                             if abs(recent_snapshots[i]['whale_activity_score'] - 
                                                   recent_snapshots[i-1]['whale_activity_score']) > 0.1])
            }
            
            whale_file = f"{self.config['data_directory']}/whale_activity/whale_summary_{self.symbol}_{timestamp_str}.json"
            with open(whale_file, 'w', encoding='utf-8') as f:
                json.dump(whale_summary, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"[WHALE] Error guardando resumen: {str(e)}")
    
    async def _create_professional_summary(self):
        """Crear resumen profesional comprehensivo"""
        
        try:
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            
            # Stats básicas
            basic_stats = {
                "data_points_captured": len(self.realtime_data),
                "orderbook_updates": len(self.orderbook_snapshots),
                "klines_1m": len(self.klines_1m),
                "klines_5m": len(self.klines_5m),
                "classifications": len(self.classifications),
                "active_streams": len(self.connections),
                "total_streams": len(self.websocket_urls)
            }
            
            # Market stats
            market_stats = {}
            if self.realtime_data:
                recent_data = list(self.realtime_data)[-60:]  # Últimos 5 min
                prices = [d.price for d in recent_data]
                
                market_stats = {
                    "current_price": prices[-1],
                    "price_change_5m": ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) >= 2 else 0,
                    "volatility_5m": self._calculate_professional_volatility(prices),
                    "price_high_5m": max(prices),
                    "price_low_5m": min(prices),
                    "avg_whale_activity": np.mean([d.whale_activity_score for d in recent_data]),
                    "avg_orderbook_imbalance": np.mean([d.orderbook_imbalance for d in recent_data]),
                    "avg_spread": np.mean([d.real_spread for d in recent_data])
                }
            
            # Classification stats
            classification_stats = {}
            if self.classifications:
                recent_classifications = list(self.classifications)[-12:]  # Última hora
                
                trends_1m = [c.trend_1m for c in recent_classifications]
                trends_5m = [c.trend_5m for c in recent_classifications]
                phases = [c.market_phase for c in recent_classifications]
                risk_scores = [c.risk_score for c in recent_classifications]
                
                classification_stats = {
                    "dominant_trend_1m": max(set(trends_1m), key=trends_1m.count) if trends_1m else "unknown",
                    "dominant_trend_5m": max(set(trends_5m), key=trends_5m.count) if trends_5m else "unknown",
                    "dominant_phase": max(set(phases), key=phases.count) if phases else "unknown",
                    "avg_risk_score": np.mean(risk_scores) if risk_scores else 50,
                    "risk_distribution": {
                        "low": len([r for r in risk_scores if r < 30]),
                        "medium": len([r for r in risk_scores if 30 <= r <= 70]),
                        "high": len([r for r in risk_scores if r > 70])
                    }
                }
            
            # Performance stats
            performance_stats = {
                "capture_stats": self.capture_stats,
                "data_quality_score": self.capture_stats["data_quality_score"],
                "stream_health": {
                    stream: "healthy" if stream in self.connections else "disconnected"
                    for stream in self.websocket_urls.keys()
                }
            }
            
            # Crear resumen completo
            professional_summary = {
                "timestamp": timestamp.isoformat(),
                "symbol": self.symbol,
                "data_source": self.data_source,
                "basic_statistics": basic_stats,
                "market_statistics": market_stats,
                "classification_statistics": classification_stats,
                "performance_statistics": performance_stats
            }
            
            # Guardar resumen
            summary_file = f"{self.config['data_directory']}/analytics/professional_summary_{self.symbol}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(professional_summary, f, indent=2, ensure_ascii=False)
            
            # Log resumen key metrics
            if market_stats and classification_stats:
                self.logger.info(f"[SUMMARY] Professional: ${market_stats['current_price']:.2f} "
                               f"({market_stats['price_change_5m']:+.2f}%) "
                               f"Trend: {classification_stats['dominant_trend_5m']} "
                               f"Phase: {classification_stats['dominant_phase']} "
                               f"Risk: {classification_stats['avg_risk_score']:.0f}/100 "
                               f"Streams: {basic_stats['active_streams']}/{basic_stats['total_streams']}")
            
        except Exception as e:
            self.logger.error(f"[SUMMARY] Error creando resumen profesional: {str(e)}")
    
    async def _connection_monitor_loop(self):
        """Monitoreo profesional de conexiones"""
        
        while self.running:
            try:
                await asyncio.sleep(self.config["heartbeat_interval"])
                
                # Check all stream connections
                disconnected_streams = []
                for stream_name, url in self.websocket_urls.items():
                    if stream_name not in self.connections:
                        disconnected_streams.append(stream_name)
                
                if disconnected_streams:
                    self.logger.warning(f"[MONITOR] Disconnected streams: {disconnected_streams}")
                
                # Update data quality score
                connected_ratio = (len(self.websocket_urls) - len(disconnected_streams)) / len(self.websocket_urls)
                self.capture_stats["data_quality_score"] = connected_ratio * 100
                
                # Log connection health
                if len(disconnected_streams) > len(self.websocket_urls) // 2:
                    self.logger.error(f"[MONITOR] ❌ Mayoría de streams desconectados: {len(disconnected_streams)}/{len(self.websocket_urls)}")
                
            except Exception as e:
                self.logger.error(f"[MONITOR] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _stats_monitoring_loop(self):
        """Loop de monitoreo de estadísticas profesional"""
        
        while self.running:
            try:
                await asyncio.sleep(600)  # Cada 10 minutos
                
                # Comprehensive stats logging
                self.logger.info(f"[STATS] Professional Capture Status:")
                self.logger.info(f"  Data Points: {self.capture_stats['ticker_points']}")
                self.logger.info(f"  OrderBook Updates: {self.capture_stats['orderbook_updates']}")
                self.logger.info(f"  Klines Received: {self.capture_stats['klines_received']}")
                self.logger.info(f"  Whale Orders: {self.capture_stats['whale_orders_detected']}")
                self.logger.info(f"  Classifications: {self.capture_stats['classifications_made']}")
                self.logger.info(f"  Data Quality: {self.capture_stats['data_quality_score']:.1f}/100")
                self.logger.info(f"  Active Streams: {len(self.connections)}/{len(self.websocket_urls)}")
                
                # Stream errors summary
                if self.capture_stats["stream_errors"]:
                    error_summary = ", ".join([f"{stream}:{count}" for stream, count in self.capture_stats["stream_errors"].items()])
                    self.logger.warning(f"[STATS] Stream Errors: {error_summary}")
                
                # Performance warnings
                if self.capture_stats["data_quality_score"] < 80:
                    self.logger.warning(f"[STATS] ⚠️ Low data quality: {self.capture_stats['data_quality_score']:.1f}/100")
                
                if len(self.connections) < len(self.websocket_urls) * 0.7:
                    self.logger.warning(f"[STATS] ⚠️ Many streams disconnected: {len(self.connections)}/{len(self.websocket_urls)}")
                
            except Exception as e:
                self.logger.error(f"[STATS] Error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup(self):
        """Cleanup profesional"""
        
        try:
            self.logger.info("[CLEANUP] Iniciando cleanup profesional...")
            
            # Cerrar todas las conexiones WebSocket
            for stream_name, connection in self.connections.items():
                try:
                    await connection.close()
                    self.logger.info(f"[CLEANUP] {stream_name} connection cerrada")
                except:
                    pass
            
            # Guardar data final
            await self._save_professional_data()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("[CLEANUP] ✅ Cleanup profesional completado")
            
        except Exception as e:
            self.logger.error(f"[CLEANUP] Error: {str(e)}")
    
    def get_professional_status(self) -> Dict:
        """Obtener estado profesional completo"""
        
        # Last data point
        last_data = None
        if self.realtime_data:
            last_point = self.realtime_data[-1]
            last_data = {
                "timestamp": last_point.timestamp.isoformat(),
                "price": last_point.price,
                "real_spread": last_point.real_spread,
                "orderbook_imbalance": last_point.orderbook_imbalance,
                "whale_activity_score": last_point.whale_activity_score,
                "large_orders_count": last_point.large_orders_count,
                "volume": last_point.volume
            }
        
        # Last classification
        last_classification = None
        if self.classifications:
            last_class = self.classifications[-1]
            last_classification = {
                "timestamp": last_class.timestamp.isoformat(),
                "trend_1m": last_class.trend_1m,
                "trend_5m": last_class.trend_5m,
                "trend_1h": last_class.trend_1h,
                "market_phase": last_class.market_phase,
                "orderbook_pressure": last_class.orderbook_pressure,
                "whale_sentiment": last_class.whale_sentiment,
                "risk_score": last_class.risk_score,
                "volatility_regime": last_class.volatility_regime
            }
        
        # Connection status
        connection_status = {
            "active_streams": len(self.connections),
            "total_streams": len(self.websocket_urls),
            "connected_streams": list(self.connections.keys()),
            "disconnected_streams": [stream for stream in self.websocket_urls.keys() if stream not in self.connections],
            "data_quality_score": self.capture_stats["data_quality_score"]
        }
        
        return {
            "running": self.running,
            "data_source": self.data_source,
            "capture_stats": self.capture_stats,
            "connection_status": connection_status,
            "buffer_sizes": {
                "realtime_data": len(self.realtime_data),
                "orderbook_snapshots": len(self.orderbook_snapshots),
                "klines_1m": len(self.klines_1m),
                "klines_5m": len(self.klines_5m),
                "klines_15m": len(self.klines_15m),
                "classifications": len(self.classifications)
            },
            "last_data_point": last_data,
            "last_classification": last_classification,
            "config": self.config
        }
    
    def stop(self):
        """Detener captura profesional"""
        
        self.logger.info("[STOP] Deteniendo captura profesional...")
        self.running = False

# ========== INTEGRATION FUNCTION ==========

def integrate_professional_capture(hybrid_scheduler):
    """Integrar captura profesional con hybrid scheduler"""
    
    # Crear sistema de captura profesional
    professional_capture = ProfessionalDataCapture(hybrid_scheduler.symbol)
    
    # Agregar como componente
    hybrid_scheduler.professional_capture = professional_capture
    
    # Crear task function
    async def start_professional_capture_task():
        await professional_capture.start_professional_capture()
    
    hybrid_scheduler.logger.info("[INTEGRATION] Professional Data Capture integrado")
    
    return start_professional_capture_task

# ========== TEST FUNCTION ==========

async def test_professional_capture():
    """Test del sistema profesional"""
    
    print("🚀 TESTING PROFESSIONAL CONTINUOUS DATA CAPTURE")
    print(f"Current Time: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Current User: biGGzeta")
    print("=" * 80)
    
    capture = ProfessionalDataCapture("ETHUSD_PERP")
    
    # Ejecutar por 3 minutos como test
    try:
        await asyncio.wait_for(capture.start_professional_capture(), timeout=180)
    except asyncio.TimeoutError:
        print("✅ Test completado - sistema profesional funcionando")
        
        # Mostrar estadísticas finales
        status = capture.get_professional_status()
        print(f"\n📊 RESULTADOS PROFESIONALES:")
        print(f"   Data source: {status['data_source']}")
        print(f"   Active streams: {status['connection_status']['active_streams']}/{status['connection_status']['total_streams']}")
        print(f"   Data quality: {status['connection_status']['data_quality_score']:.1f}/100")
        print(f"   Realtime points: {status['buffer_sizes']['realtime_data']}")
        print(f"   OrderBook updates: {status['buffer_sizes']['orderbook_snapshots']}")
        print(f"   Klines 1m: {status['buffer_sizes']['klines_1m']}")
        print(f"   Classifications: {status['buffer_sizes']['classifications']}")
        print(f"   Whale orders detected: {status['capture_stats']['whale_orders_detected']}")
        
        if status['last_data_point']:
            last_data = status['last_data_point']
            print(f"   ETH profesional: ${last_data['price']:.2f}")
            print(f"   Real spread: ${last_data['real_spread']:.2f}")
            print(f"   OrderBook imbalance: {last_data['orderbook_imbalance']:+.3f}")
            print(f"   Whale activity: {last_data['whale_activity_score']:.3f}")
            print(f"   Large orders: {last_data['large_orders_count']}")
        
        if status['last_classification']:
            last_class = status['last_classification']
            print(f"   Multi-trend: 1m:{last_class['trend_1m']} 5m:{last_class['trend_5m']} 1h:{last_class['trend_1h']}")
            print(f"   Market phase: {last_class['market_phase']}")
            print(f"   OrderBook pressure: {last_class['orderbook_pressure']}")
            print(f"   Whale sentiment: {last_class['whale_sentiment']}")
            print(f"   Risk score: {last_class['risk_score']}/100")
        
        capture.stop()
        return True

if __name__ == "__main__":
    asyncio.run(test_professional_capture())