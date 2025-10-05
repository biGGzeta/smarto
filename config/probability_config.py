#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuración para Motor de Interpretación Probabilística
Autor: biGGzeta
Fecha: 2025-10-04
"""

class ProbabilityConfig:
    """Configuración para interpretación probabilística"""
    
    # Umbrales de clasificación
    TIGHT_RANGE_THRESHOLD = 1.5  # % para considerar rango compacto
    WIDE_RANGE_THRESHOLD = 3.0   # % para considerar rango amplio
    
    # Probabilidades base
    BASE_BREAKOUT_PROB_TIGHT = 78
    BASE_BREAKOUT_PROB_WIDE = 45
    BASE_BREAKOUT_PROB_NORMAL = 60
    
    # Multiplicadores de magnitud
    TIGHT_RANGE_MULTIPLIER = 2.8
    WIDE_RANGE_MULTIPLIER = 1.2
    NORMAL_RANGE_MULTIPLIER = 2.0
    
    # Umbrales de soporte/resistencia
    STRONG_SUPPORT_THRESHOLD = 75  # % probabilidad de hold
    WEAK_SUPPORT_THRESHOLD = 60   # % probabilidad de hold
    STRONG_RESISTANCE_THRESHOLD = 8.0  # Rating de fuerza
    
    # Umbrales estructurales
    HIGH_ZONE_THRESHOLD = 80  # % del rango semanal
    LOW_ZONE_THRESHOLD = 20   # % del rango semanal
    
    # Configuración de escenarios
    MIN_SCENARIO_PROBABILITY = 5    # % mínimo para incluir escenario
    MAX_SCENARIOS = 3               # Número máximo de escenarios
    HIGH_CONFIDENCE_THRESHOLD = 70  # % para alta confianza
    
    # Factores de ajuste
    ASYMMETRY_WEIGHT = 1.5  # Peso para asimetría de liquidez
    MOMENTUM_WEIGHT = 1.2   # Peso para momentum
    STRUCTURE_WEIGHT = 1.8  # Peso para bias estructural
    
    # Configuración de logs
    SAVE_DETAILED_LOGS = True
    SAVE_EXTERNAL_SIGNALS = True
    SAVE_SCENARIO_SUMMARIES = True
    
    # Directorios
    LOGS_BASE_DIR = "logs/probabilities"
    SIGNALS_DIR = "signals"