from .base_analyzer import BaseAnalyzer
from .adaptive_parameters import AdaptiveParameters
from typing import Dict, Any, Tuple
import pandas as pd

class AdaptiveBaseAnalyzer(BaseAnalyzer):
    """Clase base que incluye capacidades adaptativas"""
    
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.adaptive_params = AdaptiveParameters(symbol)
        self.current_parameters = {}
        self.market_conditions = {}
    
    def load_data_and_analyze_conditions(self, data: pd.DataFrame, timeframe: str = "3h"):
        """Cargar datos y analizar condiciones del mercado automáticamente"""
        self.load_data(data)
        
        # Analizar condiciones del mercado
        self.market_conditions = self.adaptive_params.analyze_market_conditions(data, timeframe)
        
        # Obtener parámetros adaptativos
        self.current_parameters = self.adaptive_params.get_adaptive_parameters()
        
        print(f"🎯 Sistema adaptativo activado para {self.symbol}")
    
    def get_adaptive_window_sizes(self) -> list:
        """Obtener tamaños de ventana adaptativos"""
        return self.current_parameters.get("window_sizes", [3, 5, 7])
    
    def get_adaptive_wick_threshold(self) -> float:
        """Obtener umbral de wick adaptativo"""
        return self.current_parameters.get("wick_min_pct", 0.05)
    
    def get_adaptive_time_threshold(self) -> int:
        """Obtener umbral de tiempo adaptativo"""
        return self.current_parameters.get("max_time_minutes", 15)
    
    def get_adaptive_stability_threshold(self) -> float:
        """Obtener umbral de estabilidad adaptativo"""
        return self.current_parameters.get("stability_threshold", 0.5)
    
    def get_adaptive_zone_thresholds(self) -> Tuple[float, float]:
        """Obtener umbrales de zona adaptativos"""
        zone_low = self.current_parameters.get("zone_low", 20)
        zone_high = self.current_parameters.get("zone_high", 80)
        return zone_low, zone_high
    
    def get_significance_weights(self) -> Dict[str, float]:
        """Obtener pesos para cálculo de significancia"""
        return {
            "depth": self.current_parameters.get("depth_weight", 0.33),
            "time": self.current_parameters.get("time_weight", 0.33),
            "context": self.current_parameters.get("context_weight", 0.34)
        }