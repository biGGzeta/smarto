from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta

class BaseAnalyzer(ABC):
    """Clase base para todos los analizadores"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data: pd.DataFrame = None
        
    @abstractmethod
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Método abstracto para realizar análisis específico"""
        pass
        
    def load_data(self, data: pd.DataFrame):
        """Carga los datos para análisis"""
        self.data = data.copy()
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data.set_index('timestamp', inplace=True)
    
    def get_time_range_data(self, hours: float) -> pd.DataFrame:
        """Obtiene datos de las últimas X horas"""
        if self.data is None or self.data.empty:
            raise ValueError("No hay datos cargados")
            
        end_time = self.data.index.max()
        start_time = end_time - timedelta(hours=hours)
        
        return self.data[self.data.index >= start_time].copy()

class PriceRangeAnalyzer(BaseAnalyzer):
    """Analizador para preguntas sobre rangos de precio"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """
        Analiza rangos de precio según los parámetros de la pregunta
        
        question_params ejemplo:
        {
            "type": "max_min_range",
            "hours": 3,
            "include_percentage": True,
            "include_time_distribution": True
        }
        """
        hours = question_params.get("hours", 24)
        range_data = self.get_time_range_data(hours)
        
        if question_params.get("type") == "max_min_range":
            return self._analyze_max_min_range(range_data, question_params)
        elif question_params.get("type") == "time_in_ranges":
            return self._analyze_time_in_ranges(range_data, question_params)
            
        return "Tipo de análisis no reconocido"
    
    def _analyze_max_min_range(self, data: pd.DataFrame, params: Dict) -> str:
        """Analiza máximos y mínimos en el rango de tiempo"""
        max_price = data['high'].max()
        min_price = data['low'].min()
        
        # Encontrar timestamps de máximos y mínimos
        max_time = data[data['high'] == max_price].index[0]
        min_time = data[data['low'] == min_price].index[0]
        
        response_parts = []
        response_parts.append(f"Máximo: ${max_price:,.2f} a las {max_time.strftime('%H:%M:%S')}")
        response_parts.append(f"Mínimo: ${min_price:,.2f} a las {min_time.strftime('%H:%M:%S')}")
        
        if params.get("include_percentage", False):
            percentage_change = ((max_price - min_price) / min_price) * 100
            response_parts.append(f"Rango: {percentage_change:.2f}% de variación")
            
        return " | ".join(response_parts)
    
    def _analyze_time_in_ranges(self, data: pd.DataFrame, params: Dict) -> str:
        """Analiza tiempo que el precio estuvo en diferentes rangos"""
        min_price = data['low'].min()
        max_price = data['high'].max()
        
        # Dividir en 3 rangos automáticamente
        range_size = (max_price - min_price) / 3
        ranges = [
            (min_price, min_price + range_size),
            (min_price + range_size, min_price + 2*range_size),
            (min_price + 2*range_size, max_price)
        ]
        
        range_times = []
        total_duration = (data.index.max() - data.index.min()).total_seconds()
        
        for i, (range_min, range_max) in enumerate(ranges):
            # Datos donde el precio estuvo en este rango (usando precio de cierre)
            in_range = data[(data['close'] >= range_min) & (data['close'] <= range_max)]
            
            if not in_range.empty:
                # Calcular duración total en este rango
                range_duration = len(in_range) * (total_duration / len(data))
                hours = int(range_duration // 3600)
                minutes = int((range_duration % 3600) // 60)
                seconds = int(range_duration % 60)
                
                range_times.append(
                    f"Rango ${range_min:,.0f}-${range_max:,.0f}: {hours:02d}:{minutes:02d}:{seconds:02d}"
                )
        
        return " | ".join(range_times)