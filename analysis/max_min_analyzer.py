from .base_analyzer import BaseAnalyzer
from typing import Dict, Any
import pandas as pd

class MaxMinAnalyzer(BaseAnalyzer):
    """Analizador específico para máximos y mínimos con timeframe 1m"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Implementación del método abstracto requerido"""
        # Este método lo usaremos más adelante para análisis genéricos
        return "Análisis genérico no implementado aún"
    
    def analyze_max_min_last_hours(self, hours: float) -> tuple[str, Dict[str, Any]]:
        """
        Pregunta específica: ¿Cuáles son los máximos y mínimos de las últimas X horas?
        
        Returns:
            tuple: (respuesta_simple, datos_detallados)
        """
        if self.data is None or self.data.empty:
            return "Sin datos", {}
            
        range_data = self.get_time_range_data(hours)
        
        if range_data.empty:
            return f"Sin datos para {hours}h", {}
        
        # Análisis de máximos y mínimos
        max_price = range_data['high'].max()
        min_price = range_data['low'].min()
        max_time = range_data[range_data['high'] == max_price].index[0]
        min_time = range_data[range_data['low'] == min_price].index[0]
        
        # Respuesta simple para terminal
        base_symbol = self.symbol.replace('USD_PERP', '')
        simple_answer = f"{base_symbol} {max_price:,.0f}-{min_price:,.0f}"
        
        # Datos detallados para logs y señales
        detailed_data = {
            "analysis_type": "max_min_range",
            "max_price": max_price,
            "min_price": min_price,
            "max_time": max_time,
            "min_time": min_time,
            "data_points": len(range_data),
            "time_range_hours": hours,
            "timeframe": "1m",
            "percentage_range": ((max_price - min_price) / min_price) * 100,
            "first_data_point": range_data.index[0],
            "last_data_point": range_data.index[-1],
            "current_price": range_data['close'].iloc[-1]
        }
        
        return simple_answer, detailed_data