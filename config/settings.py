import os
from typing import Optional

class BinanceConfig:
    """Configuración para conexión con Binance"""
    
    # URLs de la API
    BASE_URL = "https://api.binance.com/api/v3"
    FUTURES_URL = "https://fapi.binance.com/fapi/v1"
    
    # Configuración de API (opcional para datos públicos)
    API_KEY: Optional[str] = os.getenv('BINANCE_API_KEY')
    SECRET_KEY: Optional[str] = os.getenv('BINANCE_SECRET_KEY')
    
    # Configuración de requests
    TIMEOUT = 30
    MAX_RETRIES = 3
    RATE_LIMIT_SLEEP = 0.2  # segundos entre requests
    
    # Límites de datos
    DEFAULT_LIMIT = 1000  # registros por request
    MAX_LIMIT = 1500

class LogConfig:
    """Configuración de logging"""
    
    LOG_LEVEL = "INFO"
    LOGS_DIR = "logs"
    
    # Formato de logs
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Archivos de log
    DETAILED_LOG_FILE = "detailed_trading.log"
    ERROR_LOG_FILE = "errors.log"
    
    # Rotación de logs
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5

class TradingConfig:
    """Configuración general de trading"""
    
    # Símbolos por defecto
    DEFAULT_SYMBOL = "ETHUSD_PERP"
    SUPPORTED_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", 
        "BTCUSDT_PERP", "ETHUSD_PERP", "BNBUSDT_PERP"
    ]
    
    # Timeframes soportados
    SUPPORTED_TIMEFRAMES = [
        "1m", "3m", "5m", "15m", "30m", 
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1M"
    ]
    
    # Configuración de análisis
    DEFAULT_ANALYSIS_HOURS = 3
    DEFAULT_PANORAMA_HOURS = 48
    DEFAULT_WEEKLY_HOURS = 168  # 7 días
    
    # Umbrales de volatilidad
    LOW_VOLATILITY_THRESHOLD = 0.5  # %
    MEDIUM_VOLATILITY_THRESHOLD = 1.5  # %
    HIGH_VOLATILITY_THRESHOLD = 3.0  # %
    
    # Configuración de detección de extremos
    WICK_MIN_PERCENTAGE = 0.05  # %
    SHORT_TIME_MAX_MINUTES = 15
    STABILITY_THRESHOLD = 0.5  # % para considerar extremos estables