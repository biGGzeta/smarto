from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, Tuple
import pandas as pd

class PercentageAnalyzer(BaseAnalyzer):
    """Analizador específico para cálculos de porcentaje de rango"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Implementación del método abstracto requerido"""
        return "Análisis genérico no implementado aún"
    
    def analyze_range_percentage(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """
        Pregunta específica: ¿De cuánto porcentaje fue el rango?
        
        Respuesta esperada: "subida del X%" o "bajada del X%"
        """
        if self.data is None or self.data.empty:
            return "Sin datos", {}
            
        range_data = self.get_time_range_data(hours)
        
        if range_data.empty:
            return f"Sin datos para {hours}h", {}
        
        # Obtener máximo y mínimo
        max_price = range_data['high'].max()
        min_price = range_data['low'].min()
        
        # Calcular porcentaje del rango
        percentage_range = ((max_price - min_price) / min_price) * 100
        
        # Determinar si fue subida o bajada general
        first_price = range_data['open'].iloc[0]
        last_price = range_data['close'].iloc[-1]
        
        # Determinar dirección principal
        if last_price > first_price:
            direction = "subida"
            net_change = ((last_price - first_price) / first_price) * 100
        else:
            direction = "bajada" 
            net_change = ((first_price - last_price) / first_price) * 100
        
        # Respuesta simple para terminal
        simple_answer = f"{direction} del {percentage_range:.2f}%"
        
        # Datos detallados para logs y señales
        detailed_data = {
            "analysis_type": "percentage_range",
            "max_price": max_price,
            "min_price": min_price,
            "first_price": first_price,
            "last_price": last_price,
            "percentage_range": percentage_range,
            "net_percentage_change": net_change,
            "direction": direction,
            "time_range_hours": hours,
            "volatility_classification": self._classify_range_volatility(percentage_range)
        }
        
        return simple_answer, detailed_data
    
    def _classify_range_volatility(self, percentage: float) -> str:
        """Clasifica la volatilidad del rango"""
        if percentage < 0.5:
            return "VERY_LOW"
        elif percentage < 1.0:
            return "LOW"
        elif percentage < 2.0:
            return "MODERATE"
        elif percentage < 5.0:
            return "HIGH"
        else:
            return "EXTREME"