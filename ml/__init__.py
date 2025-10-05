def __init__(self, symbol: str = "ETHUSD_PERP", enable_ml: bool = True, enable_logging: bool = True):
    self.symbol = symbol
    self.enable_ml = enable_ml
    self.enable_logging = enable_logging
    
    # Inicializar componentes principales (usando tus módulos existentes)
    self.downloader = BinanceDataDownloader(symbol)
    self.max_min_analyzer = MaxMinAnalyzer(symbol)
    self.percentage_analyzer = PercentageAnalyzer(symbol)
    self.low_time_analyzer = LowTimeAnalyzer(symbol)
    self.high_time_analyzer = HighTimeAnalyzer(symbol)
    self.panorama_analyzer = PanoramaAnalyzer(symbol)
    self.weekly_analyzer = WeeklyAnalyzer(symbol)
    
    # Logger QA
    if enable_logging:
        self.logger = QALogger(symbol)
    
    # Componentes ML - CORREGIDO
    if enable_ml:
        self.ml_system = AdaptiveMLSystem(symbol)  # ✅ SIN save_data=True
    
    # Variables para reutilizar datos
    self.last_data = None
    self.last_48h_data = None
    self.last_weekly_data = None
    
    print(f"🎯 Sistema inicializado para {symbol}")
    if enable_ml:
        print(f"   🤖 ML: ✅ Habilitado")
    if enable_logging:
        print(f"   📝 Logging: ✅ Habilitado")