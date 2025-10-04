import logging
import os
import json
from datetime import datetime
from config.settings import LogConfig
import pandas as pd

class QALogger:
    """Logger dual: simple para terminal, completo para archivos + señales"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Configura el logger con archivos separados"""
        logger_name = f"QA_{self.symbol}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, LogConfig.LOG_LEVEL))
        
        if logger.handlers:
            return logger
            
        os.makedirs(LogConfig.LOGS_DIR, exist_ok=True)
        
        # Handler para logs detallados (para otros códigos)
        detailed_file = os.path.join(LogConfig.LOGS_DIR, f"detailed_{self.symbol.lower()}_{datetime.now().strftime('%Y%m%d')}.log")
        detailed_handler = logging.FileHandler(detailed_file, encoding='utf-8')
        detailed_format = logging.Formatter("%(asctime)s | %(message)s")
        detailed_handler.setFormatter(detailed_format)
        logger.addHandler(detailed_handler)
        
        return logger
    
    def log_qa(self, question: str, simple_answer: str, detailed_data: dict = None):
        """
        Registra Q&A con formato dual:
        - Terminal: respuesta simple
        - Archivo: datos completos + señales
        """
        timestamp = datetime.now()
        
        # LOG SIMPLE PARA TERMINAL
        print(f"\n[{timestamp.strftime('%H:%M:%S')}] {self.symbol}")
        print(f"Q: {question}")
        print(f"A: {simple_answer}")
        print("-" * 40)
        
        # LOG DETALLADO PARA ARCHIVO + SEÑALES
        if detailed_data:
            # Convertir timestamps de pandas a strings para JSON
            detailed_data_json = self._make_json_serializable(detailed_data)
            
            log_entry = {
                "timestamp": timestamp.isoformat(),
                "symbol": self.symbol,
                "question": question,
                "simple_answer": simple_answer,
                "detailed_data": detailed_data_json,
                "signal_data": self._generate_signal(detailed_data_json)
            }
            
            self.logger.info(json.dumps(log_entry, indent=2))
    
    def _make_json_serializable(self, data: dict) -> dict:
        """Convierte objetos no serializables a JSON recursivamente"""
        json_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                json_data[key] = self._make_json_serializable(value)
            elif isinstance(value, list):
                json_data[key] = [self._convert_item_to_json(item) for item in value]
            else:
                json_data[key] = self._convert_item_to_json(value)
        return json_data
    
    def _convert_item_to_json(self, item):
        """Convierte un item individual a formato JSON serializable"""
        if isinstance(item, dict):
            return self._make_json_serializable(item)
        elif isinstance(item, (pd.Timestamp, pd.DatetimeIndex)):
            return item.isoformat() if hasattr(item, 'isoformat') else str(item)
        elif hasattr(item, 'isoformat'):  # datetime objects
            return item.isoformat()
        elif isinstance(item, (pd.Series, pd.DataFrame)):
            return str(item)  # Convertir pandas objects a string
        else:
            return item
    
    def _generate_signal(self, detailed_data: dict) -> dict:
        """Genera señales para otros códigos basadas en los datos"""
        if not detailed_data:
            return {}
        
        signal = {
            "type": "price_range_signal",
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol
        }
        
        # Señales específicas según el tipo de análisis
        if "total_volatility" in detailed_data:
            total_volatility = detailed_data["total_volatility"]
            
            signal.update({
                "volatility_level": self._classify_volatility_from_percentage(total_volatility),
                "total_volatility": total_volatility
            })
            
            # Si hay ciclos, agregar información del ciclo dominante
            if "cycles" in detailed_data and detailed_data["cycles"]:
                dominant_cycle = detailed_data["cycles"][0]  # El primero ya está ordenado por tiempo
                signal.update({
                    "dominant_range_min": dominant_cycle.get("range_min"),
                    "dominant_range_max": dominant_cycle.get("range_max"),
                    "dominant_cycle_duration": dominant_cycle.get("time_minutes")
                })
        
        return signal
    
    def _classify_volatility_from_percentage(self, percentage: float) -> str:
        """Clasifica la volatilidad desde porcentaje"""
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
    
    def _detect_trend(self, data: dict) -> str:
        """Detecta tendencia básica para señales"""
        if "max_time" in data and "min_time" in data:
            # Los timestamps ya están convertidos a string, comparar como strings
            max_time_str = data["max_time"]
            min_time_str = data["min_time"]
            
            if max_time_str > min_time_str:
                return "BULLISH"
            else:
                return "BEARISH"
        
        return "NEUTRAL"