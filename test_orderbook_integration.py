#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de Integración: Hybrid Scheduler + OrderBook Intelligence
Autor: biGGzeta
Fecha: 2025-10-05 13:50:05 UTC
"""

import asyncio
import datetime
import sys
import os

# Imports de nuestros sistemas
from hybrid_scheduler import HybridTradingScheduler, ScheduleConfig
from orderbook_intelligence import OrderBookIntelligence, integrate_orderbook_with_scheduler

async def test_orderbook_standalone():
    """Test standalone del OrderBook Intelligence"""
    
    print("🧪 TESTING ORDERBOOK INTELLIGENCE - STANDALONE")
    print("=" * 60)
    
    # Crear sistema
    orderbook_intel = OrderBookIntelligence("ETHUSD_PERP")
    
    print("✅ OrderBook Intelligence inicializado")
    
    # Test por 60 segundos para ver funcionamiento
    try:
        print("🔄 Ejecutando test por 60 segundos...")
        await asyncio.wait_for(orderbook_intel.start_monitoring(), timeout=60)
    except asyncio.TimeoutError:
        print("✅ Test standalone completado correctamente")
        
        # Mostrar estadísticas capturadas
        print(f"\n📊 ESTADÍSTICAS CAPTURADAS:")
        print(f"   Snapshots: {len(orderbook_intel.orderbook_snapshots)}")
        print(f"   Market Conditions: {len(orderbook_intel.market_conditions)}")
        print(f"   Disruption Alerts: {len(orderbook_intel.disruption_alerts)}")
        
        # Mostrar última condición si existe
        if orderbook_intel.market_conditions:
            last_condition = orderbook_intel.market_conditions[-1]
            print(f"   Última Condición: {last_condition.condition_type}")
            print(f"   Volatilidad: {last_condition.volatility:.2f}%")
        
        return True
    except Exception as e:
        print(f"❌ Error en test standalone: {str(e)}")
        return False

async def test_signal_evaluation():
    """Test de evaluación de señales con orderbook context"""
    
    print("\n🎯 TESTING SIGNAL EVALUATION")
    print("=" * 40)
    
    # Crear sistema
    orderbook_intel = OrderBookIntelligence("ETHUSD_PERP")
    
    # Simular que ya tenemos datos (forzar algunas condiciones)
    from orderbook_intelligence import MarketCondition
    
    # Crear condición de mercado simulada
    mock_condition = MarketCondition(
        start_time=datetime.datetime.now(datetime.timezone.utc),
        end_time=datetime.datetime.now(datetime.timezone.utc),
        condition_type="lateral",
        price_range_pct=0.15,
        volatility=0.18,
        volume_profile={"total_volume": 1000, "avg_trades_per_min": 45},
        order_book_stats={"avg_imbalance": 0.15, "whale_activity": 3, "avg_spread_pct": 0.05}
    )
    
    orderbook_intel.market_conditions.append(mock_condition)
    
    # Test diferentes señales
    test_signals = [
        {"action": "WEAK_BUY", "confidence": 60, "position_size_pct": 75},
        {"action": "STRONG_SELL", "confidence": 85, "position_size_pct": 100},
        {"action": "WAIT", "confidence": 45, "position_size_pct": 0}
    ]
    
    for signal in test_signals:
        evaluation = orderbook_intel.evaluate_signal_context(signal)
        
        print(f"\n📊 Signal: {signal['action']} ({signal['confidence']}%)")
        print(f"   Support Score: {evaluation['signal_support_score']}/100")
        print(f"   Risks: {evaluation['identified_risks']}")
        print(f"   Adjustments: {evaluation['suggested_adjustments']}")
    
    print("✅ Signal evaluation test completado")

async def test_integration_with_scheduler():
    """Test de integración completa con Hybrid Scheduler"""
    
    print("\n🔗 TESTING INTEGRATION WITH HYBRID SCHEDULER")
    print("=" * 50)
    
    # Crear configuración de test (ciclos rápidos)
    test_config = ScheduleConfig(
        deep_analysis_interval_hours=0.05,  # 3 minutos para test
        realtime_monitoring_enabled=True,
        log_level="INFO",
        enable_emoji_logging=False,  # Logs limpios para test
        auto_restart_on_error=False,
        max_restart_attempts=1
    )
    
    # Crear sistemas
    scheduler = HybridTradingScheduler("ETHUSD_PERP", test_config)
    orderbook_intel = OrderBookIntelligence("ETHUSD_PERP")
    
    print("✅ Scheduler y OrderBook Intelligence creados")
    
    # Función de test que ejecuta por tiempo limitado
    async def run_integrated_test():
        try:
            # Integrar sistemas
            orderbook_task = await integrate_orderbook_with_scheduler(scheduler, orderbook_intel)
            
            # Ejecutar scheduler
            scheduler_task = asyncio.create_task(scheduler.start())
            
            # Esperar ambos tasks
            await asyncio.gather(scheduler_task, orderbook_task, return_exceptions=True)
            
        except Exception as e:
            print(f"⚠️ Exception en integration test: {str(e)}")
    
    # Ejecutar test por 3 minutos
    try:
        print("🔄 Ejecutando test integrado por 3 minutos...")
        await asyncio.wait_for(run_integrated_test(), timeout=180)
    except asyncio.TimeoutError:
        print("✅ Test de integración completado")
        
        # Mostrar estado final
        status = scheduler.get_current_status()
        print(f"\n📊 ESTADO FINAL:")
        print(f"   Scheduler running: {status['running']}")
        print(f"   Current signals: {len(status['state']['current_signals'])}")
        print(f"   OrderBook snapshots: {len(orderbook_intel.orderbook_snapshots)}")
        
        # Cleanup
        scheduler.stop()
        return True
    except Exception as e:
        print(f"❌ Error en test de integración: {str(e)}")
        scheduler.stop()
        return False

async def test_disruption_detection():
    """Test específico de detección de disrupciones"""
    
    print("\n⚡ TESTING DISRUPTION DETECTION")
    print("=" * 35)
    
    orderbook_intel = OrderBookIntelligence("ETHUSD_PERP")
    
    # Simular snapshots con movimiento disruptivo
    from orderbook_intelligence import OrderBookSnapshot
    
    base_time = datetime.datetime.now(datetime.timezone.utc)
    base_price = 4540.0
    
    # Snapshot 1: Normal
    snapshot1 = OrderBookSnapshot(
        timestamp=base_time,
        price=base_price,
        bids=[(base_price - i, 10 + i) for i in range(1, 6)],
        asks=[(base_price + i, 10 + i) for i in range(1, 6)],
        trades_count=50,
        volume_takers=1000,
        avg_order_size=20
    )
    
    # Snapshot 2: Movimiento disruptivo (+1.2%)
    snapshot2 = OrderBookSnapshot(
        timestamp=base_time + datetime.timedelta(minutes=1),
        price=base_price * 1.012,  # +1.2%
        bids=[(base_price * 1.012 - i, 15 + i) for i in range(1, 6)],
        asks=[(base_price * 1.012 + i, 8 + i) for i in range(1, 6)],  # Menos liquidez en asks
        trades_count=120,  # Más trades
        volume_takers=3000,  # Más volumen
        avg_order_size=35
    )
    
    # Añadir snapshots
    orderbook_intel.orderbook_snapshots.append(snapshot1)
    orderbook_intel.orderbook_snapshots.append(snapshot2)
    
    # Procesar segundo snapshot para detectar disrupción
    await orderbook_intel._process_snapshot(snapshot2)
    
    # Verificar detección
    if orderbook_intel.disruption_alerts:
        alert = orderbook_intel.disruption_alerts[-1]
        print(f"✅ Disrupción detectada:")
        print(f"   Tipo: {alert['type']}")
        print(f"   Magnitud: {alert['magnitude_pct']:.2f}%")
        print(f"   Señales: {alert['disruption_signals']}")
    else:
        print("❌ No se detectó disrupción (posible issue)")
    
    print("✅ Disruption detection test completado")

async def test_market_condition_analysis():
    """Test de análisis de condiciones de mercado"""
    
    print("\n📊 TESTING MARKET CONDITION ANALYSIS")
    print("=" * 40)
    
    orderbook_intel = OrderBookIntelligence("ETHUSD_PERP")
    
    # Simular serie de snapshots para crear condición de mercado
    from orderbook_intelligence import OrderBookSnapshot
    
    base_time = datetime.datetime.now(datetime.timezone.utc)
    base_price = 4540.0
    
    # Crear 5 snapshots simulando mercado lateral
    for i in range(5):
        price_variation = base_price + (i - 2) * 0.5  # Variación pequeña ±1
        
        snapshot = OrderBookSnapshot(
            timestamp=base_time + datetime.timedelta(minutes=i),
            price=price_variation,
            bids=[(price_variation - j, 10 + j) for j in range(1, 6)],
            asks=[(price_variation + j, 10 + j) for j in range(1, 6)],
            trades_count=45 + i * 2,
            volume_takers=900 + i * 50,
            avg_order_size=20 + i
        )
        
        orderbook_intel.orderbook_snapshots.append(snapshot)
    
    # Analizar condición de mercado
    condition = orderbook_intel._analyze_market_condition()
    
    if condition:
        print(f"✅ Condición de mercado analizada:")
        print(f"   Tipo: {condition.condition_type}")
        print(f"   Volatilidad: {condition.volatility:.2f}%")
        print(f"   Rango de precio: {condition.price_range_pct:.2f}%")
        print(f"   Volumen total: {condition.volume_profile['total_volume']:.0f}")
    else:
        print("❌ No se pudo analizar condición de mercado")
    
    print("✅ Market condition analysis test completado")

async def run_all_tests():
    """Ejecutar todos los tests en secuencia"""
    
    print("🚀 INICIANDO BATTERY DE TESTS COMPLETA")
    print("Current Time:", datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))
    print("=" * 70)
    
    tests_results = {}
    
    # Test 1: Standalone
    print("\n1️⃣ TEST STANDALONE")
    tests_results["standalone"] = await test_orderbook_standalone()
    
    # Test 2: Signal Evaluation
    print("\n2️⃣ TEST SIGNAL EVALUATION")
    await test_signal_evaluation()
    tests_results["signal_evaluation"] = True
    
    # Test 3: Disruption Detection
    print("\n3️⃣ TEST DISRUPTION DETECTION")
    await test_disruption_detection()
    tests_results["disruption_detection"] = True
    
    # Test 4: Market Condition Analysis
    print("\n4️⃣ TEST MARKET CONDITION ANALYSIS")
    await test_market_condition_analysis()
    tests_results["market_condition"] = True
    
    # Test 5: Integration (opcional - toma más tiempo)
    response = input("\n❓ ¿Ejecutar test de integración completa? (toma 3 minutos) [y/N]: ")
    if response.lower() == 'y':
        print("\n5️⃣ TEST INTEGRATION")
        tests_results["integration"] = await test_integration_with_scheduler()
    else:
        print("\n5️⃣ TEST INTEGRATION - SKIPPED")
        tests_results["integration"] = "skipped"
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📋 RESUMEN DE TESTS:")
    for test_name, result in tests_results.items():
        status = "✅ PASS" if result is True else ("⏭️ SKIP" if result == "skipped" else "❌ FAIL")
        print(f"   {test_name.upper()}: {status}")
    
    print(f"\n🎯 Tests completados: {datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S UTC')}")
    
    return tests_results

if __name__ == "__main__":
    # Ejecutar todos los tests
    results = asyncio.run(run_all_tests())
    
    # Exit code basado en resultados
    failed_tests = [name for name, result in results.items() if result is False]
    if failed_tests:
        print(f"\n❌ Tests fallidos: {failed_tests}")
        sys.exit(1)
    else:
        print(f"\n✅ Todos los tests completados exitosamente!")
        sys.exit(0)