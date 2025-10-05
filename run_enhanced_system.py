#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced System Runner - Ejecutor del Sistema Mejorado
Ejecuta el sistema base + volume layer integrados
Autor: biGGzeta
Fecha: 2025-10-05 11:53:14 UTC
"""

import argparse
import sys
import os
from enhanced_system import EnhancedTradingSystem

def main():
    """Función principal"""
    
    parser = argparse.ArgumentParser(description='Enhanced Trading System - Base + Volume')
    parser.add_argument('--symbol', default='ETHUSD_PERP', help='Trading symbol')
    parser.add_argument('--capital', type=float, default=1000.0, help='Trading capital')
    parser.add_argument('--mode', choices=['test', 'analysis'], default='analysis', 
                       help='Execution mode')
    
    args = parser.parse_args()
    
    print(f"🚀 INICIANDO ENHANCED TRADING SYSTEM")
    print(f"📊 Símbolo: {args.symbol}")
    print(f"💰 Capital: ${args.capital:,.2f}")
    print(f"🔧 Modo: {args.mode}")
    print("=" * 60)
    
    # Inicializar sistema
    system = EnhancedTradingSystem(args.symbol, args.capital)
    
    # Ejecutar análisis
    if args.mode == 'test':
        print("🧪 Ejecutando en modo TEST...")
    else:
        print("📊 Ejecutando análisis completo...")
    
    results = system.run_enhanced_analysis()
    
    # Resultado final
    if results.get("enhanced_signal", {}).get("status") == "success":
        enhanced_signal = results["enhanced_signal"]
        print(f"\n🎯 RESULTADO FINAL:")
        print(f"   Acción: {enhanced_signal['enhanced_action']}")
        print(f"   Score: {enhanced_signal['enhanced_score']}/100")
        print(f"   Mejora vs base: {enhanced_signal['fusion_breakdown']['volume_boost']:+} puntos")
    else:
        print(f"\n❌ Error en análisis: {results.get('message', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    main()