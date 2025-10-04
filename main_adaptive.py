import asyncio
import pandas as pd
from typing import Dict, Any, Tuple
from data.csv_handler import BinanceDataDownloader
from analysis.adaptive_low_time_analyzer import AdaptiveLowTimeAnalyzer
from analysis.adaptive_high_time_analyzer import AdaptiveHighTimeAnalyzer
from analysis.adaptive_percentage_analyzer import AdaptivePercentageAnalyzer
from analysis.adaptive_panorama_analyzer import AdaptivePanoramaAnalyzer
from analysis.adaptive_weekly_analyzer import AdaptiveWeeklyAnalyzer
from analysis.max_min_analyzer import MaxMinAnalyzer  # Mantener el original para comparación
from utils.logger import QALogger
from config.settings import BinanceConfig

class AdaptiveTradingQASystem:
    """Sistema principal de Trading Q&A con parámetros auto-adaptativos"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP"):
        self.symbol = symbol
        self.downloader = BinanceDataDownloader(symbol)
        
        # Analizadores adaptativos
        self.adaptive_low_analyzer = AdaptiveLowTimeAnalyzer(symbol)
        self.adaptive_high_analyzer = AdaptiveHighTimeAnalyzer(symbol)
        self.adaptive_percentage_analyzer = AdaptivePercentageAnalyzer(symbol)
        self.adaptive_panorama_analyzer = AdaptivePanoramaAnalyzer(symbol)
        self.adaptive_weekly_analyzer = AdaptiveWeeklyAnalyzer(symbol)
        
        # Mantener analizador original para comparación
        self.original_max_min_analyzer = MaxMinAnalyzer(symbol)
        
        self.logger = QALogger(symbol)
        self.analysis_cache = {}
        
    def run_adaptive_analysis_suite(self, hours_3h: float = 3, hours_48h: float = 48) -> Dict[str, Any]:
        """Ejecutar suite completo de análisis adaptativo"""
        
        print("🚀 Iniciando Sistema Adaptativo de Trading Q&A")
        print("=" * 60)
        print(f"📊 Símbolo: {self.symbol}")
        print(f"🕐 Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 60)
        
        results = {}
        
        # Pregunta 1: Máximos y mínimos (mantener original)
        print("\n📊 PREGUNTA 1: Máximos y mínimos básicos")
        result1 = self.ask_max_min_last_hours(hours_3h)
        results["max_min_basic"] = result1
        
        # Pregunta 2: Análisis de porcentajes adaptativo
        print("\n📊 PREGUNTA 2: Análisis de porcentajes adaptativo")
        result2 = self.ask_adaptive_range_percentage(hours_3h)
        results["range_percentage_adaptive"] = result2
        
        # Pregunta 3: Mínimos adaptativos
        print("\n📊 PREGUNTA 3: Mínimos adaptativos")
        result3 = self.ask_adaptive_low_time_minimums(hours_3h)
        results["low_minimums_adaptive"] = result3
        
        # Pregunta 4: Máximos adaptativos
        print("\n📊 PREGUNTA 4: Máximos adaptativos")
        result4 = self.ask_adaptive_high_time_maximums(hours_3h)
        results["high_maximums_adaptive"] = result4
        
        # Pregunta 5: Panorama 48h adaptativo
        print("\n📊 PREGUNTA 5: Panorama 48h adaptativo")
        result5 = self.ask_adaptive_48h_panorama()
        results["panorama_48h_adaptive"] = result5
        
        # Pregunta 6: Análisis semanal adaptativo
        print("\n📊 PREGUNTA 6: Análisis semanal adaptativo")
        result6 = self.ask_adaptive_weekly_analysis()
        results["weekly_adaptive"] = result6
        
        # Resumen adaptativo
        print("\n" + "=" * 60)
        self._print_adaptive_summary(results)
        print("🎯 Suite de análisis adaptativo completado!")
        
        return results
    
    def ask_max_min_last_hours(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 1: Máximos y mínimos básicos (sin cambios)"""
        question = f"¿Cuáles son los máximos y mínimos de las últimas {hours} horas?"
        
        print(f"🔄 Descargando datos de {self.symbol} (1m) para las últimas {hours} horas...")
        
        data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        self.analysis_cache["data_3h"] = data.copy()
        
        self.original_max_min_analyzer.load_data(data)
        simple_answer, detailed_data = self.original_max_min_analyzer.analyze_max_min_last_hours(hours)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        if isinstance(detailed_data, dict):
            print(f"✅ Respuesta 1: {simple_answer}")
            print(f"📈 Volatilidad: {detailed_data.get('percentage_range', 0):.2f}%")
        
        return simple_answer, detailed_data
    
    def ask_adaptive_range_percentage(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 2: Análisis de porcentajes con parámetros adaptativos"""
        question = f"¿Cuál es el análisis adaptativo de porcentajes de las últimas {hours} horas?"
        
        # Reutilizar datos si están disponibles
        if "data_3h" in self.analysis_cache:
            print(f"🔄 Reutilizando datos existentes para análisis adaptativo...")
            data = self.analysis_cache["data_3h"]
        else:
            print(f"🔄 Descargando datos de {self.symbol} para análisis adaptativo...")
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        # 🔧 CARGAR DATOS EN EL ANALIZADOR
        self.adaptive_percentage_analyzer.load_data(data)
        
        simple_answer, detailed_data = self.adaptive_percentage_analyzer.analyze_range_percentage_adaptive(hours)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        print(f"✅ Respuesta 2: {simple_answer}")
        
        return simple_answer, detailed_data
    
    def ask_adaptive_low_time_minimums(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 3: Mínimos adaptativos"""
        question = f"¿Cuáles son los mínimos adaptativos de las últimas {hours} horas?"
        
        if "data_3h" in self.analysis_cache:
            print(f"🔄 Reutilizando datos para análisis adaptativo de mínimos...")
            data = self.analysis_cache["data_3h"]
        else:
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        # 🔧 CARGAR DATOS EN EL ANALIZADOR
        self.adaptive_low_analyzer.load_data(data)
        
        simple_answer, detailed_data = self.adaptive_low_analyzer.analyze_low_time_minimums_adaptive(hours)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        print(f"✅ Respuesta 3: {simple_answer}")
        
        return simple_answer, detailed_data
    
    def ask_adaptive_high_time_maximums(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 4: Máximos adaptativos"""
        question = f"¿Cuáles son los máximos adaptativos de las últimas {hours} horas?"
        
        if "data_3h" in self.analysis_cache:
            print(f"🔄 Reutilizando datos para análisis adaptativo de máximos...")
            data = self.analysis_cache["data_3h"]
        else:
            data = self.downloader.get_klines("1m", hours)
        
        if data.empty:
            simple_answer = "Sin datos disponibles"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        # 🔧 CARGAR DATOS EN EL ANALIZADOR
        self.adaptive_high_analyzer.load_data(data)
        
        simple_answer, detailed_data = self.adaptive_high_analyzer.analyze_high_time_maximums_adaptive(hours)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        print(f"✅ Respuesta 4: {simple_answer}")
        
        return simple_answer, detailed_data
    
    def ask_adaptive_48h_panorama(self) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 5: Panorama 48h adaptativo"""
        question = "¿Cuál es el panorama adaptativo de las últimas 48 horas?"
        
        print(f"🔄 Descargando datos de {self.symbol} para panorama adaptativo 48h...")
        
        data = self.downloader.get_klines("1m", 48)
        
        if data.empty:
            simple_answer = "Sin datos disponibles para 48h"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        self.analysis_cache["data_48h"] = data.copy()
        
        # 🔧 CARGAR DATOS EN EL ANALIZADOR
        self.adaptive_panorama_analyzer.load_data(data)
        
        simple_answer, detailed_data = self.adaptive_panorama_analyzer.analyze_48h_panorama_adaptive(48)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        print(f"✅ Respuesta 5: {simple_answer}")
        
        return simple_answer, detailed_data
    
    def ask_adaptive_weekly_analysis(self) -> Tuple[str, Dict[str, Any]]:
        """Pregunta 6: Análisis semanal adaptativo"""
        question = "¿Cuál es el análisis semanal adaptativo completo?"
        
        print(f"🔄 Descargando datos de {self.symbol} para análisis semanal adaptativo...")
        
        data = self.downloader.get_klines("1m", 168)
        
        if data.empty:
            simple_answer = "Sin datos disponibles para análisis semanal"
            self.logger.log_qa(question, simple_answer)
            return simple_answer, {}
        
        self.analysis_cache["data_weekly"] = data.copy()
        
        # 🔧 CARGAR DATOS EN EL ANALIZADOR
        self.adaptive_weekly_analyzer.load_data(data)
        
        simple_answer, detailed_data = self.adaptive_weekly_analyzer.analyze_weekly_with_recent_extremes_adaptive(3)
        
        self.logger.log_qa(question, simple_answer, detailed_data)
        
        print(f"✅ Respuesta 6: {simple_answer}")
        
        if isinstance(detailed_data, dict):
            maximos_trend = detailed_data.get('maximos_trend', {})
            minimos_trend = detailed_data.get('minimos_trend', {})
            print(f"🎯 Máximos 3d: {maximos_trend.get('trend', 'unknown')}")
            print(f"🎯 Mínimos 3d: {minimos_trend.get('trend', 'unknown')}")
        
        return simple_answer, detailed_data
    
    def _print_adaptive_summary(self, results: Dict[str, Any]):
        """Imprimir resumen del análisis adaptativo"""
        
        print("🎯 RESUMEN ADAPTATIVO:")
        print("-" * 40)
        
        # Extraer condiciones del mercado de cualquier análisis
        sample_result = None
        for key, result in results.items():
            if isinstance(result, tuple) and len(result) > 1:
                sample_result = result[1]
                break
        
        if sample_result and isinstance(sample_result, dict):
            conditions = sample_result.get("market_conditions", {})
            parameters = sample_result.get("adaptive_parameters", {})
            
            print(f"📊 Régimen del mercado: {conditions.get('market_regime', 'unknown')}")
            print(f"🔥 Volatilidad: {conditions.get('volatility', 0):.2f}%")
            print(f"📈 Rango: {conditions.get('price_range_pct', 0):.2f}%")
            print(f"🌊 Densidad: {conditions.get('direction_density', 0):.1f}")
            print(f"📊 Tendencia: {conditions.get('trend_strength', 0):.2f}")
            
            print("\n🔧 Parámetros adaptativos aplicados:")
            print(f"   Windows: {parameters.get('window_sizes', [])}")
            print(f"   Wick threshold: {parameters.get('wick_min_pct', 0):.3f}%")
            print(f"   Max time: {parameters.get('max_time_minutes', 0)}min")
            print(f"   Zonas: {parameters.get('zone_low', 0):.0f}%-{parameters.get('zone_high', 0):.0f}%")

def test_adaptive_system():
    """Probar el sistema adaptativo completo"""
    
    # Inicializar sistema adaptativo
    system = AdaptiveTradingQASystem("ETHUSD_PERP")
    
    # Ejecutar suite completo
    results = system.run_adaptive_analysis_suite()
    
    return results

if __name__ == "__main__":
    test_adaptive_system()