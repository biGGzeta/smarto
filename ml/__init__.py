"""
Módulo de Machine Learning para Trading Adaptativo

Incluye:
- Feature Engineering: Extracción de características del análisis adaptativo
- Decision Tree: Árbol de decisiones para señales de trading
- ML System: Sistema integrado completo
"""

from .feature_engineering import FeatureExtractor, MarketFeatures
from .decision_tree import AdaptiveDecisionTree, TradingSignal, SignalType
from .adaptive_ml_system import AdaptiveMLSystem

__all__ = [
    'FeatureExtractor',
    'MarketFeatures', 
    'AdaptiveDecisionTree',
    'TradingSignal',
    'SignalType',
    'AdaptiveMLSystem'
]

__version__ = "1.0.0"