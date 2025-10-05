#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para verificar que todos los analizadores base funcionan correctamente
Autor: biGGzeta
Fecha: 2025-10-04
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Agregar la ruta del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports de los analizadores base
try:
    from analysis.max_min_analyzer import MaxMinAnalyzer
    from analysis.percentage_analyzer import PercentageAnalyzer
    from analysis.low_time_analyzer import LowTimeAnalyzer
    from analysis.high_time_analyzer import HighTimeAnalyzer
    from analysis.panorama_analyzer import PanoramaAnalyzer
    from analysis.weekly_analyzer import WeeklyAnalyzer
    from data.csv_handler import BinanceDataDownloader
    print("✅ Todos los imports exitosos")
except ImportError as e:
    print(f"❌ Error en imports: {e}")
    sys.exit(1)

def create_test_data(hours: int = 3, symbol: str = "ETHUSD_PERP") -> pd.DataFrame:
    """Crear datos de prueba simulados"""
    
    # Crear timestamps cada minuto
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Simular datos de precio (tendencia alcista suave con volatilidad)
    base_price = 4400
    n_points = len(timestamps)
    
    # Crear trend + ruido
    trend = np.linspace(0, 50, n_points)  # Tendencia alcista de $50
    noise = np.random.normal(0, 5, n_points)  # Ruido de ±$5
    
    close_prices = base_price + trend + noise
    
    # Crear OHLC realista
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, close_prices)):
        # High/Low basado en close con algo de spread
        high = close + np.random.uniform(1, 8)
        low = close - np.random.uniform(1, 8)
        open_price = close_prices[i-1] if i > 0 else close
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(100, 1000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def test_analyzer(analyzer_class, analyzer_name: str, test_data: pd.DataFrame):
    """Probar un analizador específico"""
    
    print(f"\n🔍 Probando {analyzer_name}...")
    
    try:
        # Crear instancia
        analyzer = analyzer_class("ETHUSD_PERP")
        
        # Cargar datos
        analyzer.load_data(test_data)
        print(f"   ✅ Datos cargados: {len(test_data)} puntos")
        
        # Probar métodos específicos
        if analyzer_name == "MaxMinAnalyzer":
            result, detailed = analyzer.analyze_max_min_last_hours(3)
            print(f"   📊 Resultado: {result}")
            
        elif analyzer_name == "PercentageAnalyzer":
            result, detailed = analyzer.analyze_range_percentage(3)
            print(f"   📊 Resultado: {result}")
            
        elif analyzer_name == "LowTimeAnalyzer":
            result, detailed = analyzer.analyze_low_time_minimums(3, 15)
            print(f"   📊 Resultado: {result}")
            
        elif analyzer_name == "HighTimeAnalyzer":
            result, detailed = analyzer.analyze_high_time_maximums(3, 15)
            print(f"   📊 Resultado: {result}")
            
        elif analyzer_name == "PanoramaAnalyzer":
            result, detailed = analyzer.analyze_48h_panorama(3)  # Usar 3h para test
            print(f"   📊 Resultado: {result}")
            
        elif analyzer_name == "WeeklyAnalyzer":
            result, detailed = analyzer.analyze_weekly_with_recent_extremes(1)  # 1 día para test
            print(f"   📊 Resultado: {result}")
        
        print(f"   ✅ {analyzer_name} funciona correctamente")
        return True
        
    except Exception as e:
        print(f"   ❌ Error en {analyzer_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal de pruebas"""
    
    print("🚀 INICIANDO VERIFICACIÓN DE ANALIZADORES BASE")
    print("=" * 60)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 Usuario: biGGzeta")
    print("=" * 60)
    
    # Crear datos de prueba
    print("\n📊 Creando datos de prueba...")
    test_data = create_test_data(hours=3)
    print(f"✅ Datos creados: {len(test_data)} puntos desde {test_data.index[0]} hasta {test_data.index[-1]}")
    print(f"💰 Rango de precios: ${test_data['low'].min():.2f} - ${test_data['high'].max():.2f}")
    
    # Lista de analizadores a probar
    analyzers_to_test = [
        (MaxMinAnalyzer, "MaxMinAnalyzer"),
        (PercentageAnalyzer, "PercentageAnalyzer"), 
        (LowTimeAnalyzer, "LowTimeAnalyzer"),
        (HighTimeAnalyzer, "HighTimeAnalyzer"),
        (PanoramaAnalyzer, "PanoramaAnalyzer"),
        (WeeklyAnalyzer, "WeeklyAnalyzer")
    ]
    
    # Probar cada analizador
    results = {}
    for analyzer_class, name in analyzers_to_test:
        success = test_analyzer(analyzer_class, name, test_data)
        results[name] = success
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE VERIFICACIÓN:")
    print("=" * 60)
    
    all_passed = True
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 TODOS LOS ANALIZADORES FUNCIONAN CORRECTAMENTE")
        print("✅ Sistema base verificado - Listo para construir lógica de señales")
    else:
        print("⚠️ ALGUNOS ANALIZADORES TIENEN PROBLEMAS")
        print("🔧 Revisa los errores antes de continuar")
    print("=" * 60)

if __name__ == "__main__":
    main()