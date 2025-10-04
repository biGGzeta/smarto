import asyncio
from data.csv_handler import BinanceDataDownloader
from analysis.max_min_analyzer import MaxMinAnalyzer
from analysis.percentage_analyzer import PercentageAnalyzer
from analysis.low_time_analyzer import LowTimeAnalyzer
from analysis.high_time_analyzer import HighTimeAnalyzer
from analysis.panorama_analyzer import PanoramaAnalyzer
from analysis.weekly_analyzer import WeeklyAnalyzer
from utils.logger import QALogger
from config.settings import BinanceConfig

class TradingQASystem:
    """Sistema principal de preguntas y respuestas de trading"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP"):
        self.symbol = symbol
        self.downloader = BinanceDataDownloader(symbol)
        self.max_min_analyzer = MaxMinAnalyzer(symbol)
        self.percentage_analyzer = PercentageAnalyzer(symbol)
        self.low_time_analyzer = LowTimeAnalyzer(symbol)
        self.high_time_analyzer = HighTimeAnalyzer(symbol)
        self.panorama_analyzer = PanoramaAnalyzer(symbol)
        self.weekly_analyzer = WeeklyAnalyzer(symbol)
        self.logger = QALogger(symbol)
        self.last_data = None
        self.last_48h_data = None
        self.last_weekly_data = None
        
    def ask_max_min_last_hours(self, hours: float):
        """Pregunta 1: ¿Cuáles son los máximos y mínimos de las últimas X horas?"""
        question = f"¿Cuáles son los máximos y mínimos de las últimas {hours} horas?"
        
        print(f"🔄 Descargando datos de {self.symbol} (1m) para las últimas {hours} horas...")
        
        # Descargar datos en timeframe 1m
        data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer
        
        # Guardar datos para reutilizar
        self.last_data = data.copy()
        
        # Cargar datos y analizar
        self.max_min_analyzer.load_data(data)
        simple_answer, detailed_data = self.max_min_analyzer.analyze_max_min_last_hours(hours)
        
        # Registrar en logs duales
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data
    
    def ask_range_percentage(self, hours: float):
        """Pregunta 2: ¿De cuánto porcentaje fue el rango?"""
        question = f"¿De cuánto porcentaje fue el rango de las últimas {hours} horas?"
        
        # Reutilizar datos si ya los tenemos
        if self.last_data is not None and not self.last_data.empty:
            print(f"🔄 Reutilizando datos existentes para análisis de porcentaje...")
            data = self.last_data
        else:
            print(f"🔄 Descargando datos de {self.symbol} (1m) para análisis de porcentaje...")
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer
        
        # Cargar datos y analizar
        self.percentage_analyzer.load_data(data)
        simple_answer, detailed_data = self.percentage_analyzer.analyze_range_percentage(hours)
        
        # Registrar en logs duales
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data
    
    def ask_low_time_minimums(self, hours: float, max_time_minutes: int = 15):
        """Pregunta 3: ¿Cuáles fueron los mínimos más bajos en los que el precio estuvo poco tiempo?"""
        question = f"¿Cuáles fueron los mínimos más bajos en los que el precio estuvo poco tiempo (≤{max_time_minutes}min)?"
        
        # Reutilizar datos si ya los tenemos
        if self.last_data is not None and not self.last_data.empty:
            print(f"🔄 Reutilizando datos existentes para análisis de mínimos rápidos...")
            data = self.last_data
        else:
            print(f"🔄 Descargando datos de {self.symbol} (1m) para análisis de mínimos...")
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer
        
        # Cargar datos y analizar
        self.low_time_analyzer.load_data(data)
        simple_answer, detailed_data = self.low_time_analyzer.analyze_low_time_minimums(hours, max_time_minutes)
        
        # Registrar en logs duales
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data
    
    def ask_high_time_maximums(self, hours: float, max_time_minutes: int = 15):
        """Pregunta 4: ¿Cuáles fueron los máximos más altos en los que el precio estuvo poco tiempo?"""
        question = f"¿Cuáles fueron los máximos más altos en los que el precio estuvo poco tiempo (≤{max_time_minutes}min)?"
        
        # Reutilizar datos si ya los tenemos
        if self.last_data is not None and not self.last_data.empty:
            print(f"🔄 Reutilizando datos existentes para análisis de máximos rápidos...")
            data = self.last_data
        else:
            print(f"🔄 Descargando datos de {self.symbol} (1m) para análisis de máximos...")
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer
        
        # Cargar datos y analizar
        self.high_time_analyzer.load_data(data)
        simple_answer, detailed_data = self.high_time_analyzer.analyze_high_time_maximums(hours, max_time_minutes)
        
        # Registrar en logs duales
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data
    
    def ask_48h_panorama(self):
        """Pregunta 5: Panorama de las últimas 48 horas"""
        question = "¿Cuál es el panorama de las últimas 48 horas?"
        
        print(f"🔄 Descargando datos de {self.symbol} (1m) para panorama de 48 horas...")
        
        # Descargar datos de 48 horas
        data = self.downloader.get_klines("1m", 48)
        
        if data.empty:
            simple_answer = "Sin datos disponibles para 48h"
            self.logger.log_qa(question, simple_answer)
            return simple_answer
        
        # Guardar datos de 48h para reutilizar
        self.last_48h_data = data.copy()
        
        # Cargar datos y analizar
        self.panorama_analyzer.load_data(data)
        simple_answer, detailed_data = self.panorama_analyzer.analyze_48h_panorama(48)
        
        # Registrar en logs duales
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data
    
    def ask_weekly_analysis(self):
        """Pregunta 6: Análisis semanal completo con tendencia de extremos recientes"""
        question = "¿Cuál es el análisis semanal completo con tendencia de extremos recientes?"
        
        print(f"🔄 Descargando datos de {self.symbol} (1m) para análisis semanal...")
        
        # Descargar datos de 1 semana (168 horas)
        data = self.downloader.get_klines("1m", 168)
        
        if data.empty:
            simple_answer = "Sin datos disponibles para análisis semanal"
            self.logger.log_qa(question, simple_answer)
            return simple_answer
        
        # Guardar datos semanales para reutilizar
        self.last_weekly_data = data.copy()
        
        # Cargar datos y analizar
        self.weekly_analyzer.load_data(data)
        # En el método ask_weekly_analysis, cambiar la llamada:
        simple_answer, detailed_data = self.weekly_analyzer.analyze_weekly_with_recent_extremes(3)  # 3 días en lugar de 2
        
        # Registrar en logs duales
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        return simple_answer, detailed_data

def test_system():
    """Prueba dinámica del sistema con 6 preguntas"""
    print("🚀 Iniciando sistema de Trading Q&A")
    print("=" * 50)
    
    # Inicializar sistema con ETH
    system = TradingQASystem("ETHUSD_PERP")
    
    # Pregunta 1: Máximos y mínimos de las últimas 3 horas
    print("\n📊 PREGUNTA 1: Máximos y mínimos")
    result1 = system.ask_max_min_last_hours(3)
    
    if isinstance(result1, tuple):
        simple_answer1, detailed_data1 = result1
        print(f"\n✅ Respuesta 1: {simple_answer1}")
        print(f"📈 Volatilidad: {detailed_data1.get('percentage_range', 0):.2f}%")
    
    # Pregunta 2: Porcentaje del rango
    print("\n📊 PREGUNTA 2: Porcentaje del rango")
    result2 = system.ask_range_percentage(3)
    
    if isinstance(result2, tuple):
        simple_answer2, detailed_data2 = result2
        print(f"\n✅ Respuesta 2: {simple_answer2}")
    
    # Pregunta 3: Mínimos con poco tiempo
    print("\n📊 PREGUNTA 3: Mínimos rápidos")
    result3 = system.ask_low_time_minimums(3, 15)
    
    if isinstance(result3, tuple):
        simple_answer3, detailed_data3 = result3
        print(f"\n✅ Respuesta 3: {simple_answer3}")
    
    # Pregunta 4: Máximos con poco tiempo
    print("\n📊 PREGUNTA 4: Máximos rápidos")
    result4 = system.ask_high_time_maximums(3, 15)
    
    if isinstance(result4, tuple):
        simple_answer4, detailed_data4 = result4
        print(f"\n✅ Respuesta 4: {simple_answer4}")
    
    # Pregunta 5: Panorama de 48 horas
    print("\n📊 PREGUNTA 5: Panorama 48h")
    result5 = system.ask_48h_panorama()
    
    if isinstance(result5, tuple):
        simple_answer5, detailed_data5 = result5
        print(f"\n✅ Respuesta 5: {simple_answer5}")
    
    # Pregunta 6: Análisis semanal
    print("\n📊 PREGUNTA 6: Análisis semanal")
    result6 = system.ask_weekly_analysis()
    
    if isinstance(result6, tuple):
        simple_answer6, detailed_data6 = result6
        print(f"\n✅ Respuesta 6: {simple_answer6}")
        if detailed_data6.get('maximos_trend'):
            print(f"🎯 Máximos 2h: {detailed_data6['maximos_trend']['trend']}")
            print(f"🎯 Mínimos 2h: {detailed_data6['minimos_trend']['trend']}")
    
    print("\n" + "=" * 50)
    print("🎯 Seis preguntas validadas. Sistema COMPLETO funcionando!")

if __name__ == "__main__":
    test_system()