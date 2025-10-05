#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Trading Scheduler - VERSIÓN FINAL COMPLETA INTEGRADA
Scheduler inteligente + OrderBook Intelligence + Adaptive ML Engine + Continuous Data Capture
Autor: biGGzeta
Fecha: 2025-10-05 14:37:17 UTC
Versión: 2.1.0 (Continuous Data Integrated)
"""

import asyncio
import datetime
import json
import os
import logging
import signal
import sys
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import websockets
import threading
from concurrent.futures import ThreadPoolExecutor
import schedule
import time

# Imports del sistema existente
from enhanced_system import EnhancedTradingSystem

@dataclass
class ScheduleConfig:
    """Configuración del scheduler"""
    deep_analysis_interval_hours: int = 3
    realtime_monitoring_enabled: bool = True
    websocket_url: str = "wss://fstream.binance.com/ws/ethusd_perp@ticker"
    max_concurrent_tasks: int = 4  # Aumentado para continuous capture
    log_level: str = "INFO"
    auto_restart_on_error: bool = True
    max_restart_attempts: int = 5
    enable_emoji_logging: bool = False  # False para Windows compatibility
    enable_orderbook_intelligence: bool = True  # OrderBook Intelligence
    enable_ml: bool = True  # ML Engine enable/disable
    enable_continuous_capture: bool = True  # NUEVO: Continuous Data Capture

@dataclass
class SystemState:
    """Estado actual del sistema híbrido"""
    last_deep_analysis: Optional[datetime.datetime] = None
    last_deep_analysis_results: Optional[Dict] = None
    current_signals: Dict = None
    realtime_adjustments: Dict = None
    system_status: str = "STARTING"
    errors_count: int = 0
    restart_attempts: int = 0
    
    def __post_init__(self):
        if self.current_signals is None:
            self.current_signals = {}
        if self.realtime_adjustments is None:
            self.realtime_adjustments = {}

class HybridTradingScheduler:
    """
    Scheduler híbrido inteligente para trading - VERSIÓN FINAL COMPLETA
    
    Combina:
    - Deep Analysis cada N horas (sistema completo)
    - Real-time monitoring continuo (WebSocket + ML basic)
    - OrderBook Intelligence (whale detection + market conditions)
    - Adaptive ML Engine (learning + predictions)
    - Continuous Data Capture (NUEVO - captura continua para clasificación)
    - State management inteligente
    - Error recovery automático
    """
    
    def __init__(self, symbol: str = "ETHUSD_PERP", config: Optional[ScheduleConfig] = None):
        self.symbol = symbol
        self.config = config or ScheduleConfig()
        self.state = SystemState()
        
        # Sistemas core
        self.enhanced_system = EnhancedTradingSystem(symbol)
        
        # OrderBook Intelligence
        self.orderbook_intel = None
        self.orderbook_enabled = self.config.enable_orderbook_intelligence
        
        # ML Engine
        self.ml_engine = None
        self.ml_enabled = self.config.enable_ml
        
        # NUEVO: Continuous Data Capture
        self.continuous_capture = None
        self.continuous_enabled = self.config.enable_continuous_capture
        
        # Threading y async
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        self.running = False
        self.websocket_task = None
        self.scheduler_task = None
        self.orderbook_task = None
        self.continuous_task = None  # NUEVO
        
        # Logs y estado
        self.setup_logging()
        self.setup_directories()
        
        # Handlers para graceful shutdown
        self.setup_signal_handlers()
        
        # Log messages (con/sin emojis según config)
        if self.config.enable_emoji_logging:
            self.logger.info(f"🚀 Hybrid Trading Scheduler FINAL COMPLETO iniciado para {symbol}")
            self.logger.info(f"⚙️ Config: Deep analysis cada {self.config.deep_analysis_interval_hours}h")
            self.logger.info(f"🔴 Real-time monitoring: {'✅ Enabled' if self.config.realtime_monitoring_enabled else '❌ Disabled'}")
            self.logger.info(f"📚 OrderBook Intelligence: {'✅ Enabled' if self.orderbook_enabled else '❌ Disabled'}")
            self.logger.info(f"🤖 ML Engine: {'✅ Enabled' if self.ml_enabled else '❌ Disabled'}")
            self.logger.info(f"🔄 Continuous Data Capture: {'✅ Enabled' if self.continuous_enabled else '❌ Disabled'}")
        else:
            self.logger.info(f"[INIT] Hybrid Trading Scheduler FINAL COMPLETO iniciado para {symbol}")
            self.logger.info(f"[CONFIG] Deep analysis cada {self.config.deep_analysis_interval_hours}h")
            self.logger.info(f"[REALTIME] Real-time monitoring: {'Enabled' if self.config.realtime_monitoring_enabled else 'Disabled'}")
            self.logger.info(f"[ORDERBOOK] OrderBook Intelligence: {'Enabled' if self.orderbook_enabled else 'Disabled'}")
            self.logger.info(f"[ML] ML Engine: {'Enabled' if self.ml_enabled else 'Disabled'}")
            self.logger.info(f"[CONTINUOUS] Continuous Data Capture: {'Enabled' if self.continuous_enabled else 'Disabled'}")
    
    def setup_logging(self):
        """Configurar logging del sistema con encoding UTF-8"""
        
        log_dir = "logs/scheduler"
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger("HybridScheduler")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler con UTF-8
        log_file = os.path.join(log_dir, f"hybrid_scheduler_{self.symbol}_{datetime.datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Configurar console stream para UTF-8 en Windows si es posible
        try:
            if hasattr(console_handler.stream, 'reconfigure'):
                console_handler.stream.reconfigure(encoding='utf-8')
        except Exception:
            # Fallback para sistemas que no soportan reconfigure
            pass
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def setup_directories(self):
        """Crear directorios necesarios"""
        
        directories = [
            "logs/scheduler",
            "logs/realtime", 
            "logs/state",
            "logs/orderbook",
            "logs/continuous",  # NUEVO
            "ml_models/orderbook",
            "data/realtime",
            "data/continuous",  # NUEVO
            "data/continuous/raw",  # NUEVO
            "data/continuous/classified",  # NUEVO
            "data/continuous/trends",  # NUEVO
            "data/continuous/patterns",  # NUEVO
            "data/continuous/summaries"  # NUEVO
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_signal_handlers(self):
        """Configurar handlers para shutdown graceful"""
        
        def signal_handler(signum, frame):
            if self.config.enable_emoji_logging:
                self.logger.info(f"📶 Signal {signum} recibido. Iniciando shutdown graceful...")
            else:
                self.logger.info(f"[SIGNAL] Signal {signum} recibido. Iniciando shutdown graceful...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def log(self, emoji: str, message: str, level: str = "info"):
        """Helper para logging con/sin emojis"""
        
        if self.config.enable_emoji_logging:
            full_message = f"{emoji} {message}"
        else:
            # Convertir emoji a tag
            emoji_tags = {
                "🚀": "[START]",
                "📊": "[ANALYSIS]", 
                "🔴": "[REALTIME]",
                "✅": "[SUCCESS]",
                "❌": "[ERROR]",
                "⚠️": "[WARNING]",
                "🎯": "[TARGET]",
                "🛡️": "[RISK]",
                "⏰": "[SCHEDULE]",
                "🔗": "[CONNECT]",
                "⚡": "[ALERT]",
                "💾": "[SAVE]",
                "📚": "[ORDERBOOK]",
                "🤖": "[ML]",
                "🐋": "[WHALE]",
                "🔄": "[CONTINUOUS]",
                "📈": "[TREND]",
                "📉": "[CLASSIFY]"
            }
            tag = emoji_tags.get(emoji, "[INFO]")
            full_message = f"{tag} {message}"
        
        getattr(self.logger, level.lower())(full_message)
    
    async def start(self):
        """Iniciar el scheduler híbrido COMPLETO"""
        
        self.log("🚀", "INICIANDO HYBRID TRADING SCHEDULER FINAL COMPLETO")
        self.logger.info("=" * 80)
        
        self.running = True
        self.state.system_status = "RUNNING"
        
        try:
            # FASE PREVIA: Inicializar componentes adicionales
            await self._initialize_all_components()
            
            # Crear tasks concurrentes
            tasks = []
            
            # Task 1: Deep analysis scheduler
            deep_analysis_task = asyncio.create_task(self._run_deep_analysis_scheduler())
            tasks.append(deep_analysis_task)
            
            # Task 2: Real-time monitoring (si está habilitado)
            if self.config.realtime_monitoring_enabled:
                realtime_task = asyncio.create_task(self._run_realtime_monitoring())
                tasks.append(realtime_task)
            
            # Task 3: State management y health monitoring
            state_task = asyncio.create_task(self._run_state_manager())
            tasks.append(state_task)
            
            # Task 4: OrderBook Intelligence (si está habilitado)
            if self.orderbook_intel:
                orderbook_task = asyncio.create_task(self.orderbook_intel.start_monitoring())
                tasks.append(orderbook_task)
                self.orderbook_task = orderbook_task
            
            # Task 5: NUEVO - Continuous Data Capture (si está habilitado)
            if self.continuous_capture:
                continuous_task = asyncio.create_task(self.continuous_capture.start_continuous_capture())
                tasks.append(continuous_task)
                self.continuous_task = continuous_task
            
            # Ejecutar todos los tasks concurrentemente
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.log("❌", f"Error en scheduler principal: {str(e)}", "error")
            if self.config.auto_restart_on_error:
                await self._attempt_restart()
            else:
                raise
    
    async def _initialize_all_components(self):
        """Inicializar TODOS los componentes del sistema"""
        
        self.log("🔧", "Inicializando TODOS los componentes del sistema...")
        
        # 1. Inicializar OrderBook Intelligence
        if self.orderbook_enabled:
            try:
                from orderbook_intelligence import OrderBookIntelligence
                self.orderbook_intel = OrderBookIntelligence(self.symbol)
                self.log("📚", "OrderBook Intelligence inicializado")
            except ImportError as e:
                self.log("⚠️", f"OrderBook Intelligence no disponible: {str(e)}", "warning")
                self.orderbook_enabled = False
            except Exception as e:
                self.log("⚠️", f"Error inicializando OrderBook Intelligence: {str(e)}", "warning")
                self.orderbook_enabled = False
        
        # 2. Inicializar ML Engine
        if self.ml_enabled:
            try:
                from adaptive_ml_engine import integrate_ml_safely
                
                # Integrar ML con OrderBook Intelligence si está disponible
                if self.orderbook_intel:
                    self.ml_engine = integrate_ml_safely(self.orderbook_intel, self, enable_ml=True)
                    ml_status = self.ml_engine.get_status() if hasattr(self.ml_engine, 'get_status') else {"status": "fallback"}
                    self.log("🤖", f"ML Engine integrado: {ml_status.get('enabled', False)}")
                else:
                    self.log("⚠️", "ML Engine requiere OrderBook Intelligence - disabled", "warning")
                    self.ml_enabled = False
                    
            except ImportError as e:
                self.log("⚠️", f"ML Engine no disponible: {str(e)}", "warning")
                self.ml_enabled = False
            except Exception as e:
                self.log("⚠️", f"Error inicializando ML Engine: {str(e)}", "warning")
                self.ml_enabled = False
        
        # 3. NUEVO: Inicializar Continuous Data Capture
        if self.continuous_enabled:
            try:
                from continuous_data_capture import ContinuousDataCapture
                self.continuous_capture = ContinuousDataCapture(self.symbol)
                self.log("🔄", "Continuous Data Capture inicializado")
                
                # Configurar integración con otros componentes
                if self.orderbook_intel:
                    # Link continuous capture con orderbook intel para data sharing
                    self.continuous_capture.orderbook_intel = self.orderbook_intel
                    self.log("🔗", "Continuous Capture integrado con OrderBook Intelligence")
                
            except ImportError as e:
                self.log("⚠️", f"Continuous Data Capture no disponible: {str(e)}", "warning")
                self.continuous_enabled = False
            except Exception as e:
                self.log("⚠️", f"Error inicializando Continuous Data Capture: {str(e)}", "warning")
                self.continuous_enabled = False
        
        # 4. Reportar estado final de TODOS los componentes
        components_status = {
            "Enhanced System": "✅ Active",
            "Real-time Monitoring": "✅ Active" if self.config.realtime_monitoring_enabled else "❌ Disabled",
            "OrderBook Intelligence": "✅ Active" if self.orderbook_intel else "❌ Disabled",
            "ML Engine": "✅ Active" if self.ml_engine and hasattr(self.ml_engine, 'get_status') else "❌ Disabled",
            "Continuous Data Capture": "✅ Active" if self.continuous_capture else "❌ Disabled"
        }
        
        active_components = sum(1 for v in components_status.values() if '✅' in v)
        total_components = len(components_status)
        
        self.log("📊", f"Componentes inicializados: {active_components}/{total_components} activos")
        
        for component, status in components_status.items():
            self.logger.info(f"   {component}: {status}")
        
        # Log configuración detallada
        self.logger.info(f"\n[CONFIG] Configuración Final:")
        self.logger.info(f"   Deep Analysis: cada {self.config.deep_analysis_interval_hours}h")
        self.logger.info(f"   Max Concurrent Tasks: {self.config.max_concurrent_tasks}")
        self.logger.info(f"   Auto Restart: {self.config.auto_restart_on_error}")
        self.logger.info(f"   Log Level: {self.config.log_level}")
    
    async def _run_deep_analysis_scheduler(self):
        """Ejecutar análisis profundo cada N horas"""
        
        self.log("📊", f"Deep Analysis Scheduler iniciado (cada {self.config.deep_analysis_interval_hours}h)")
        
        # Ejecutar análisis inicial inmediatamente
        await self._execute_deep_analysis()
        
        # Scheduler loop
        while self.running:
            try:
                # Calcular próxima ejecución
                next_run = self._calculate_next_deep_analysis()
                wait_seconds = (next_run - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                
                if wait_seconds > 0:
                    self.log("⏰", f"Próximo análisis profundo en {wait_seconds/3600:.1f}h ({next_run.strftime('%H:%M:%S UTC')})")
                    await asyncio.sleep(wait_seconds)
                
                if self.running:
                    await self._execute_deep_analysis()
                    
            except Exception as e:
                self.log("❌", f"Error en deep analysis scheduler: {str(e)}", "error")
                await asyncio.sleep(300)  # Wait 5 min before retry
    
    async def _execute_deep_analysis(self):
        """Ejecutar análisis profundo completo"""
        
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        self.log("📊", f"EJECUTANDO ANÁLISIS PROFUNDO - {timestamp.strftime('%H:%M:%S')} UTC")
        
        try:
            # Ejecutar enhanced system en thread separado para no bloquear
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor, 
                self._run_enhanced_system_sync
            )
            
            # Actualizar state
            self.state.last_deep_analysis = timestamp
            self.state.last_deep_analysis_results = results
            
            # Extraer señales principales
            if results and results.get("final_recommendation", {}).get("status") == "success":
                final_rec = results["final_recommendation"]
                
                self.state.current_signals = {
                    "action": final_rec["final_action"],
                    "confidence": final_rec["final_confidence"],
                    "source": final_rec["confidence_source"],
                    "timestamp": timestamp.isoformat(),
                    "setup_type": results.get("setup_detection", {}).get("detected_setup", "N/A"),
                    "risk_level": final_rec.get("risk_assessment", {}).get("risk_level", "UNKNOWN")
                }
                
                self.log("✅", f"Análisis completado: {final_rec['final_action']} ({final_rec['final_confidence']}/100)")
                self.log("🎯", f"Setup: {self.state.current_signals['setup_type']}")
                self.log("🛡️", f"Riesgo: {self.state.current_signals['risk_level']}")
                
                # Evaluar con OrderBook Intelligence si está disponible
                if self.orderbook_intel:
                    await self._evaluate_with_orderbook_intelligence()
                
                # NUEVO: Integrar con Continuous Data Capture
                if self.continuous_capture:
                    await self._integrate_with_continuous_capture(results)
                
            else:
                self.log("⚠️", "Análisis profundo completado con errores", "warning")
            
            # Guardar estado
            await self._save_state()
            
            return results
            
        except Exception as e:
            self.log("❌", f"Error ejecutando análisis profundo: {str(e)}", "error")
            self.state.errors_count += 1
            return None
    
    async def _evaluate_with_orderbook_intelligence(self):
        """Evaluar señales con OrderBook Intelligence"""
        
        try:
            if not self.orderbook_intel or not self.state.current_signals:
                return
            
            # Evaluar contexto de orderbook para las señales actuales
            orderbook_context = self.orderbook_intel.evaluate_signal_context(self.state.current_signals)
            
            if orderbook_context.get("status") == "success":
                # Aplicar ajustes sugeridos por orderbook
                suggested_adjustments = orderbook_context.get("suggested_adjustments", {})
                signal_support_score = orderbook_context.get("signal_support_score", 50)
                identified_risks = orderbook_context.get("identified_risks", [])
                
                if suggested_adjustments or identified_risks:
                    self.state.realtime_adjustments["orderbook_intelligence"] = {
                        "signal_support_score": signal_support_score,
                        "suggested_adjustments": suggested_adjustments,
                        "identified_risks": identified_risks,
                        "orderbook_context": orderbook_context.get("orderbook_context", {})
                    }
                    
                    self.log("📚", f"OrderBook evaluation: Support {signal_support_score}/100, Risks: {len(identified_risks)}")
                    
                    # Log riesgos importantes
                    if "high_whale_activity" in identified_risks:
                        self.log("🐋", "High whale activity detected - increased caution")
                    if "recent_disruptions" in identified_risks:
                        self.log("⚡", "Recent market disruptions detected")
            
        except Exception as e:
            self.log("⚠️", f"Error en OrderBook evaluation: {str(e)}", "warning")
    
    async def _integrate_with_continuous_capture(self, analysis_results: Dict):
        """Integrar resultados del análisis con Continuous Data Capture"""
        
        try:
            if not self.continuous_capture:
                return
            
            # Extraer información relevante del análisis
            analysis_summary = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "analysis_type": "deep_analysis",
                "final_action": analysis_results.get("final_recommendation", {}).get("final_action", "WAIT"),
                "confidence": analysis_results.get("final_recommendation", {}).get("final_confidence", 0),
                "setup_type": analysis_results.get("setup_detection", {}).get("detected_setup", "N/A"),
                "risk_level": analysis_results.get("final_recommendation", {}).get("risk_assessment", {}).get("risk_level", "UNKNOWN"),
                "enhanced_score": analysis_results.get("enhanced_signal", {}).get("enhanced_score", 0)
            }
            
            # Agregar a continuous capture como training data si tiene métodos de ML integration
            if hasattr(self.continuous_capture, 'add_analysis_result'):
                self.continuous_capture.add_analysis_result(analysis_summary)
                self.log("🔄", "Análisis integrado con Continuous Data Capture")
            
            # Log integración
            self.log("🔗", f"Deep analysis shared con Continuous Capture: {analysis_summary['final_action']} ({analysis_summary['confidence']}/100)")
            
        except Exception as e:
            self.log("⚠️", f"Error integrando con Continuous Data Capture: {str(e)}", "warning")
    
    def _run_enhanced_system_sync(self):
        """Wrapper sync para ejecutar enhanced system"""
        try:
            system = EnhancedTradingSystem(self.symbol)
            return system.run_complete_enhanced_analysis()
        except Exception as e:
            self.log("❌", f"Error en enhanced system: {str(e)}", "error")
            return None
    
    async def _run_realtime_monitoring(self):
        """Ejecutar monitoring real-time"""
        
        self.log("🔴", "Real-time Monitoring iniciado")
        
        while self.running:
            try:
                # WebSocket connection
                uri = self.config.websocket_url
                self.log("🔗", f"Conectando WebSocket: {uri}")
                
                async with websockets.connect(uri) as websocket:
                    self.log("✅", "WebSocket conectado")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        await self._process_realtime_data(message)
                        
            except websockets.exceptions.ConnectionClosed:
                self.log("⚠️", "WebSocket desconectado. Reconectando en 5s...", "warning")
                await asyncio.sleep(5)
            except Exception as e:
                self.log("❌", f"Error en real-time monitoring: {str(e)}", "error")
                await asyncio.sleep(10)
    
    async def _process_realtime_data(self, message: str):
        """Procesar datos real-time del WebSocket"""
        
        try:
            data = json.loads(message)
            
            # Extraer datos relevantes (Binance ticker format)
            if 'c' in data:  # Current price
                current_price = float(data['c'])
                price_change_pct = float(data['P'])
                volume = float(data['v'])
                
                # Basic real-time analysis
                await self._analyze_realtime_data({
                    "price": current_price,
                    "price_change_pct": price_change_pct,
                    "volume": volume,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc)
                })
                
        except Exception as e:
            self.logger.debug(f"Error procesando mensaje WebSocket: {str(e)}")
    
    async def _analyze_realtime_data(self, tick_data: Dict):
        """Análisis básico de datos real-time"""
        
        # Para versión actual: logging básico y detección de cambios significativos
        price = tick_data["price"]
        change_pct = tick_data["price_change_pct"]
        
        # Detectar movimientos significativos (reducido threshold para más logs)
        if abs(change_pct) >= 0.5:  # Movimiento >0.5% (reducido de 2.0%)
            self.log("⚡", f"Movimiento detectado: ${price:.2f} ({change_pct:+.2f}%)")
            
            # Enhanced analysis con componentes adicionales
            await self._evaluate_signal_adjustments(tick_data)
    
    async def _evaluate_signal_adjustments(self, tick_data: Dict):
        """Evaluar si se deben ajustar señales basado en datos real-time - ENHANCED"""
        
        # Evaluación básica original
        if not self.state.current_signals:
            return
        
        current_action = self.state.current_signals.get("action", "WAIT")
        current_confidence = self.state.current_signals.get("confidence", 0)
        
        # Example: Upgrade signal si hay confirmación
        price_change = tick_data["price_change_pct"]
        
        base_adjustments = {}
        
        if current_action == "WEAK_BUY" and price_change > 1.0:
            base_adjustments = {
                "original_action": current_action,
                "suggested_action": "BUY",
                "reason": f"Price momentum confirmation (+{price_change:.1f}%)",
                "timestamp": tick_data["timestamp"].isoformat(),
                "confidence_boost": 10
            }
        elif current_action == "WEAK_SELL" and price_change < -1.0:
            base_adjustments = {
                "original_action": current_action,
                "suggested_action": "SELL",
                "reason": f"Price momentum confirmation ({price_change:.1f}%)",
                "timestamp": tick_data["timestamp"].isoformat(),
                "confidence_boost": 10
            }
        
        # Enhanced analysis con ML Engine si está disponible
        if self.ml_engine and hasattr(self.ml_engine, 'is_ready') and self.ml_engine.is_ready():
            try:
                # Los ML insights ya se aplicaron en el OrderBook processing
                # Aquí podemos agregar análisis adicional de real-time data
                
                # Ejemplo: Detectar patrones en real-time data
                if abs(price_change) > 1.5:
                    # Agregar training sample para ML
                    if hasattr(self.ml_engine, 'add_training_sample_safe'):
                        # TODO: Preparar features del tick data para ML
                        pass
                
            except Exception as e:
                self.log("⚠️", f"Error en ML real-time analysis: {str(e)}", "warning")
        
        # Aplicar ajustes si los hay
        if base_adjustments:
            self.state.realtime_adjustments["signal_upgrade"] = base_adjustments
            self.log("🔥", f"Signal upgrade sugerido: {current_action} → {base_adjustments.get('suggested_action')}")
    
    async def _run_state_manager(self):
        """Gestionar estado del sistema y health monitoring"""
        
        self.log("📊", "State Manager iniciado")
        
        while self.running:
            try:
                # Health check cada 30 segundos
                await self._perform_health_check()
                await asyncio.sleep(30)
                
            except Exception as e:
                self.log("❌", f"Error en state manager: {str(e)}", "error")
                await asyncio.sleep(60)
    
    async def _perform_health_check(self):
        """Realizar health check del sistema"""
        
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Check si deep analysis está al día
        if self.state.last_deep_analysis:
            hours_since_analysis = (now - self.state.last_deep_analysis).total_seconds() / 3600
            
            if hours_since_analysis > (self.config.deep_analysis_interval_hours + 0.5):
                self.log("⚠️", f"Deep analysis atrasado: {hours_since_analysis:.1f}h desde último", "warning")
        
        # Check error count
        if self.state.errors_count > 10:
            self.log("⚠️", f"Muchos errores detectados: {self.state.errors_count}", "warning")
        
        # Check estado de TODOS los componentes cada 10 minutos
        if now.minute % 10 == 0:
            await self._check_all_components_health()
        
        # Log estado cada 5 minutos
        if now.minute % 5 == 0 and now.second < 30:
            await self._log_comprehensive_system_status()
    
    async def _check_all_components_health(self):
        """Check health de TODOS los componentes"""
        
        component_status = {}
        
        # Enhanced System health (siempre activo)
        component_status["enhanced_system"] = {"status": "active"}
        
        # OrderBook Intelligence health
        if self.orderbook_intel:
            try:
                component_status["orderbook"] = {
                    "status": "active",
                    "snapshots": len(self.orderbook_intel.orderbook_snapshots),
                    "conditions": len(self.orderbook_intel.market_conditions),
                    "disruptions": len(self.orderbook_intel.disruption_alerts)
                }
            except Exception as e:
                component_status["orderbook"] = {"status": "error", "error": str(e)}
        
        # ML Engine health
        if self.ml_engine and hasattr(self.ml_engine, 'get_status'):
            try:
                ml_status = self.ml_engine.get_status()
                component_status["ml_engine"] = {
                    "status": "active" if ml_status.get("ready", False) else "not_ready",
                    "trained": ml_status.get("models_trained", False),
                    "samples": ml_status.get("training_samples", 0)
                }
            except Exception as e:
                component_status["ml_engine"] = {"status": "error", "error": str(e)}
        
        # NUEVO: Continuous Data Capture health
        if self.continuous_capture:
            try:
                capture_status = self.continuous_capture.get_current_status()
                component_status["continuous_capture"] = {
                    "status": "active" if capture_status.get("running", False) else "stopped",
                    "data_source": capture_status.get("data_source", "unknown"),
                    "total_points": capture_status.get("capture_stats", {}).get("total_points", 0),
                    "data_points": capture_status.get("buffer_sizes", {}).get("realtime_data", 0),
                    "classifications": capture_status.get("buffer_sizes", {}).get("classifications", 0)
                }
                
                # Log estadísticas importantes de continuous capture
                if capture_status.get("last_classification"):
                    last_class = capture_status["last_classification"]
                    self.log("🔄", f"Continuous: {last_class.get('trend_1m', 'unknown')} trend, "
                           f"{last_class.get('market_phase', 'unknown')} phase")
                
            except Exception as e:
                component_status["continuous_capture"] = {"status": "error", "error": str(e)}
        
        # Log component health summary
        active_components = sum(1 for comp in component_status.values() if comp.get("status") == "active")
        total_components = len(component_status)
        
        self.log("🔧", f"Component health: {active_components}/{total_components} active")
        
        # Log detalles importantes
        for component, status in component_status.items():
            if status.get("status") == "error":
                self.log("⚠️", f"{component}: ERROR - {status.get('error', 'unknown')}", "warning")
            elif status.get("status") == "active" and component == "continuous_capture":
                total_points = status.get("total_points", 0)
                if total_points > 0:
                    self.log("📈", f"Continuous Capture: {total_points} points captured, "
                           f"{status.get('data_points', 0)} in buffer")
    
    async def _log_comprehensive_system_status(self):
        """Log del estado comprehensivo del sistema"""
        
        # Estado básico
        basic_status = {
            "system_status": self.state.system_status,
            "last_analysis": self.state.last_deep_analysis.isoformat() if self.state.last_deep_analysis else "Never",
            "current_signals": len(self.state.current_signals),
            "realtime_adjustments": len(self.state.realtime_adjustments),
            "errors_count": self.state.errors_count
        }
        
        # Estado de componentes
        component_status = {
            "enhanced_system_enabled": True,
            "realtime_monitoring_enabled": self.config.realtime_monitoring_enabled,
            "orderbook_enabled": self.orderbook_enabled,
            "ml_enabled": self.ml_enabled,
            "continuous_capture_enabled": self.continuous_enabled
        }
        
        # Combinar estado
        full_status = {**basic_status, **component_status}
        
        self.log("📊", f"System Status Comprehensive: {json.dumps(full_status, indent=2)}")
        
        # Log señales actuales si existen
        if self.state.current_signals:
            current_action = self.state.current_signals.get("action", "UNKNOWN")
            confidence = self.state.current_signals.get("confidence", 0)
            setup_type = self.state.current_signals.get("setup_type", "N/A")
            risk_level = self.state.current_signals.get("risk_level", "UNKNOWN")
            
            self.log("🎯", f"Current Signal: {current_action} ({confidence}/100) - {setup_type} - Risk: {risk_level}")
    
    def _calculate_next_deep_analysis(self) -> datetime.datetime:
        """Calcular próxima ejecución de análisis profundo"""
        
        if not self.state.last_deep_analysis:
            return datetime.datetime.now(datetime.timezone.utc)
        
        return self.state.last_deep_analysis + datetime.timedelta(hours=self.config.deep_analysis_interval_hours)
    
    async def _save_state(self):
        """Guardar estado actual del sistema"""
        
        state_file = f"logs/state/hybrid_state_{self.symbol}_{datetime.datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            # Estado básico
            state_data = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "symbol": self.symbol,
                "config": asdict(self.config),
                "state": asdict(self.state)
            }
            
            # Agregar estado de OrderBook Intelligence
            if self.orderbook_intel:
                state_data["orderbook_status"] = {
                    "snapshots_count": len(self.orderbook_intel.orderbook_snapshots),
                    "market_conditions_count": len(self.orderbook_intel.market_conditions),
                    "disruption_alerts_count": len(self.orderbook_intel.disruption_alerts),
                    "current_regime": getattr(self.orderbook_intel, 'current_market_regime', 'unknown')
                }
            
            # Agregar estado de ML Engine
            if self.ml_engine and hasattr(self.ml_engine, 'get_status'):
                state_data["ml_status"] = self.ml_engine.get_status()
            
            # NUEVO: Agregar estado de Continuous Data Capture
            if self.continuous_capture:
                capture_status = self.continuous_capture.get_current_status()
                state_data["continuous_capture_status"] = {
                    "running": capture_status.get("running", False),
                    "data_source": capture_status.get("data_source", "unknown"),
                    "total_points_captured": capture_status.get("capture_stats", {}).get("total_points", 0),
                    "buffer_sizes": capture_status.get("buffer_sizes", {}),
                    "last_classification": capture_status.get("last_classification")
                }
            
            def json_serializer(obj):
                if isinstance(obj, datetime.datetime):
                    return obj.isoformat()
                return str(obj)
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=json_serializer, ensure_ascii=False)
                
            self.log("💾", f"Estado completo guardado: {state_file}")
                
        except Exception as e:
            self.log("❌", f"Error guardando estado: {str(e)}", "error")
    
    async def _attempt_restart(self):
        """Intentar restart automático del sistema"""
        
        if self.state.restart_attempts >= self.config.max_restart_attempts:
            self.log("❌", f"Máximo de restart attempts alcanzado ({self.config.max_restart_attempts})", "error")
            self.stop()
            return
        
        self.state.restart_attempts += 1
        self.log("🔄", f"Intentando restart #{self.state.restart_attempts}")
        
        await asyncio.sleep(30)  # Wait before restart
        await self.start()
    
    def stop(self):
        """Detener el scheduler"""
        
        self.log("🛑", "Deteniendo Hybrid Trading Scheduler COMPLETO...")
        self.running = False
        self.state.system_status = "STOPPING"
        
        # Cancel ALL tasks
        tasks_to_cancel = [
            ("websocket_task", self.websocket_task),
            ("scheduler_task", self.scheduler_task),
            ("orderbook_task", self.orderbook_task),
            ("continuous_task", self.continuous_task)  # NUEVO
        ]
        
        for task_name, task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                self.log("🔄", f"{task_name} cancelled")
        
        # Stop continuous capture explicitly
        if self.continuous_capture:
            self.continuous_capture.stop()
            self.log("🔄", "Continuous Data Capture stopped")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.log("✅", "Hybrid Trading Scheduler COMPLETO detenido")
    
    def get_current_status(self) -> Dict:
        """Obtener estado actual del sistema"""
        
        basic_status = {
            "running": self.running,
            "state": asdict(self.state),
            "config": asdict(self.config),
            "symbol": self.symbol
        }
        
        # Estado de TODOS los componentes
        component_status = {
            "enhanced_system": {"enabled": True, "active": True},
            "orderbook_intelligence": {
                "enabled": self.orderbook_enabled,
                "active": self.orderbook_intel is not None
            },
            "ml_engine": {
                "enabled": self.ml_enabled,
                "active": self.ml_engine is not None and hasattr(self.ml_engine, 'get_status')
            },
            "continuous_data_capture": {  # NUEVO
                "enabled": self.continuous_enabled,
                "active": self.continuous_capture is not None
            }
        }
        
        return {**basic_status, "components": component_status}

# ========== CONFIGURACIONES PREDEFINIDAS ==========

def create_production_config() -> ScheduleConfig:
    """Configuración para producción"""
    return ScheduleConfig(
        deep_analysis_interval_hours=3,
        realtime_monitoring_enabled=True,
        log_level="INFO",
        enable_emoji_logging=False,  # Mejor para logs en producción
        auto_restart_on_error=True,
        max_restart_attempts=5,
        enable_orderbook_intelligence=True,
        enable_ml=True,
        enable_continuous_capture=True  # NUEVO
    )

def create_development_config() -> ScheduleConfig:
    """Configuración para desarrollo"""
    return ScheduleConfig(
        deep_analysis_interval_hours=1,  # Más frecuente para testing
        realtime_monitoring_enabled=True,
        log_level="DEBUG",
        enable_emoji_logging=True,  # Visual para desarrollo
        auto_restart_on_error=False,  # Manual restart en dev
        max_restart_attempts=3,
        enable_orderbook_intelligence=True,
        enable_ml=True,
        enable_continuous_capture=True  # NUEVO
    )

def create_testing_config() -> ScheduleConfig:
    """Configuración para testing"""
    return ScheduleConfig(
        deep_analysis_interval_hours=0.1,  # 6 minutos para testing rápido
        realtime_monitoring_enabled=False,  # Sin WebSocket en tests
        log_level="DEBUG",
        enable_emoji_logging=False,
        auto_restart_on_error=False,
        max_restart_attempts=1,
        enable_orderbook_intelligence=True,
        enable_ml=False,  # Disabled en tests para velocidad
        enable_continuous_capture=True  # NUEVO
    )

def create_safe_config() -> ScheduleConfig:
    """Configuración segura sin componentes adicionales"""
    return ScheduleConfig(
        deep_analysis_interval_hours=3,
        realtime_monitoring_enabled=True,
        log_level="INFO",
        enable_emoji_logging=False,
        auto_restart_on_error=True,
        max_restart_attempts=5,
        enable_orderbook_intelligence=False,  # DISABLED
        enable_ml=False,  # DISABLED
        enable_continuous_capture=False  # DISABLED
    )

def create_continuous_only_config() -> ScheduleConfig:
    """Configuración solo para continuous data capture"""
    return ScheduleConfig(
        deep_analysis_interval_hours=6,  # Menos frecuente
        realtime_monitoring_enabled=False,  # Solo continuous capture
        log_level="INFO",
        enable_emoji_logging=False,
        auto_restart_on_error=True,
        max_restart_attempts=5,
        enable_orderbook_intelligence=False,
        enable_ml=False,
        enable_continuous_capture=True  # SOLO ESTO ACTIVO
    )

# ========== RUNNER FUNCTIONS ==========

async def run_hybrid_scheduler(symbol: str = "ETHUSD_PERP", config: Optional[ScheduleConfig] = None):
    """Ejecutar scheduler híbrido"""
    
    if config is None:
        config = create_production_config()
    
    scheduler = HybridTradingScheduler(symbol, config)
    await scheduler.start()

def run_scheduler_blocking(symbol: str = "ETHUSD_PERP", mode: str = "production"):
    """Ejecutar scheduler en modo blocking"""
    
    # Seleccionar configuración
    if mode == "production":
        config = create_production_config()
    elif mode == "development":
        config = create_development_config()
    elif mode == "testing":
        config = create_testing_config()
    elif mode == "safe":
        config = create_safe_config()
    elif mode == "continuous":
        config = create_continuous_only_config()
    else:
        config = create_production_config()
    
    print(f"INICIANDO HYBRID TRADING SCHEDULER FINAL COMPLETO")
    print(f"Símbolo: {symbol}")
    print(f"Modo: {mode}")
    print(f"Timestamp: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Usuario: biGGzeta")
    print("=" * 80)
    
    # Mostrar configuración
    print(f"Configuración Final:")
    print(f"  Deep Analysis: cada {config.deep_analysis_interval_hours}h")
    print(f"  Real-time Monitoring: {config.realtime_monitoring_enabled}")
    print(f"  OrderBook Intelligence: {config.enable_orderbook_intelligence}")
    print(f"  ML Engine: {config.enable_ml}")
    print(f"  Continuous Data Capture: {config.enable_continuous_capture}")
    print(f"  Max Concurrent Tasks: {config.max_concurrent_tasks}")
    print("=" * 80)
    
    asyncio.run(run_hybrid_scheduler(symbol, config))

def run_in_background(symbol: str = "ETHUSD_PERP", mode: str = "production"):
    """Ejecutar scheduler en background thread"""
    
    def background_runner():
        run_scheduler_blocking(symbol, mode)
    
    thread = threading.Thread(target=background_runner, daemon=True)
    thread.start()
    return thread

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Trading Scheduler FINAL COMPLETO')
    parser.add_argument('--symbol', default='ETHUSD_PERP', help='Trading symbol')
    parser.add_argument('--mode', choices=['production', 'development', 'testing', 'safe', 'continuous'], 
                       default='production', help='Execution mode')
    
    args = parser.parse_args()
    
    # Ejecutar scheduler
    run_scheduler_blocking(args.symbol, args.mode)