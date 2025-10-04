from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

class PanoramaAnalyzer(BaseAnalyzer):
    """Analizador para panorama general de 48 horas con zonas de precio"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Implementación del método abstracto requerido"""
        return "Análisis genérico no implementado aún"
    
    def analyze_48h_panorama(self, hours: float = 48) -> Tuple[str, Dict[str, Any]]:
        """
        Pregunta específica: Panorama de las últimas 48 horas
        
        Analiza: máximo, mínimo, % rango, tiempo en zona alta (>80%) y zona baja (<20%)
        """
        if self.data is None or self.data.empty:
            return "Sin datos", {}
            
        range_data = self.get_time_range_data(hours)
        
        if range_data.empty:
            return f"Sin datos para {hours}h", {}
        
        # Análisis básico del rango
        max_price = range_data['high'].max()
        min_price = range_data['low'].min()
        price_range = max_price - min_price
        percentage_range = (price_range / min_price) * 100
        
        # Definir zonas del rango
        zone_80_threshold = min_price + (price_range * 0.8)  # Zona alta >80%
        zone_20_threshold = min_price + (price_range * 0.2)  # Zona baja <20%
        
        # Analizar tiempo en zonas altas y bajas
        high_zone_analysis = self._analyze_zone_time(range_data, zone_80_threshold, 'above', 'alta')
        low_zone_analysis = self._analyze_zone_time(range_data, zone_20_threshold, 'below', 'baja')
        
        # Detectar ciclos y rebotes en zonas extremas
        high_zone_cycles = self._detect_zone_cycles(range_data, zone_80_threshold, 'above')
        low_zone_cycles = self._detect_zone_cycles(range_data, zone_20_threshold, 'below')
        
        # Calcular distribución general del precio
        price_distribution = self._calculate_price_distribution(range_data, min_price, price_range)
        
        # Formatear respuesta
        simple_answer = self._format_panorama_response(
            max_price, min_price, percentage_range, 
            high_zone_analysis, low_zone_analysis,
            high_zone_cycles, low_zone_cycles
        )
        
        detailed_data = {
            "analysis_type": "48h_panorama",
            "time_range_hours": hours,
            "max_price": max_price,
            "min_price": min_price,
            "price_range": price_range,
            "percentage_range": percentage_range,
            "zone_80_threshold": zone_80_threshold,
            "zone_20_threshold": zone_20_threshold,
            "high_zone_analysis": high_zone_analysis,
            "low_zone_analysis": low_zone_analysis,
            "high_zone_cycles": high_zone_cycles,
            "low_zone_cycles": low_zone_cycles,
            "price_distribution": price_distribution
        }
        
        return simple_answer, detailed_data
    
    def _analyze_zone_time(self, data: pd.DataFrame, threshold: float, direction: str, zone_name: str) -> Dict:
        """Analiza cuánto tiempo estuvo el precio en una zona específica"""
        
        if direction == 'above':
            # Zona alta: precio por encima del umbral
            in_zone_mask = (data['high'] >= threshold) | (data['close'] >= threshold)
        else:
            # Zona baja: precio por debajo del umbral
            in_zone_mask = (data['low'] <= threshold) | (data['close'] <= threshold)
        
        in_zone_data = data[in_zone_mask]
        total_time_in_zone = len(in_zone_data)  # minutos
        
        # Calcular porcentaje del tiempo total
        total_time = len(data)
        time_percentage = (total_time_in_zone / total_time) * 100
        
        # Convertir a horas y minutos
        hours = total_time_in_zone // 60
        minutes = total_time_in_zone % 60
        
        return {
            "zone_name": zone_name,
            "threshold": threshold,
            "direction": direction,
            "total_minutes": total_time_in_zone,
            "time_formatted": f"{hours}:{minutes:02d}h",
            "time_percentage": time_percentage,
            "total_touches": len(in_zone_data)
        }
    
    def _detect_zone_cycles(self, data: pd.DataFrame, threshold: float, direction: str) -> List[Dict]:
        """Detecta ciclos/rebotes en zonas extremas"""
        cycles = []
        
        if direction == 'above':
            in_zone_mask = (data['high'] >= threshold) | (data['close'] >= threshold)
        else:
            in_zone_mask = (data['low'] <= threshold) | (data['close'] <= threshold)
        
        # Detectar períodos continuos en la zona
        current_cycle = None
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            is_in_zone = in_zone_mask.iloc[i]
            
            if is_in_zone and current_cycle is None:
                # Inicio de un nuevo ciclo
                current_cycle = {
                    'start_time': timestamp,
                    'start_price': row['close'],
                    'zone_direction': direction,
                    'touches': 1
                }
            elif is_in_zone and current_cycle is not None:
                # Continúa el ciclo actual
                current_cycle['touches'] += 1
                current_cycle['end_time'] = timestamp
                current_cycle['end_price'] = row['close']
            elif not is_in_zone and current_cycle is not None:
                # Fin del ciclo actual
                current_cycle['end_time'] = current_cycle.get('end_time', timestamp)
                current_cycle['end_price'] = current_cycle.get('end_price', row['close'])
                current_cycle['duration_minutes'] = current_cycle['touches']
                
                # Formatear duración
                hours = current_cycle['duration_minutes'] // 60
                minutes = current_cycle['duration_minutes'] % 60
                current_cycle['duration_formatted'] = f"{hours}:{minutes:02d}h"
                
                # Calcular tipo de movimiento
                if direction == 'above':
                    current_cycle['movement_type'] = 'resistencia' if current_cycle['touches'] > 10 else 'rebote_alto'
                else:
                    current_cycle['movement_type'] = 'soporte' if current_cycle['touches'] > 10 else 'rebote_bajo'
                
                cycles.append(current_cycle)
                current_cycle = None
        
        # Cerrar último ciclo si está abierto
        if current_cycle is not None:
            current_cycle['end_time'] = data.index[-1]
            current_cycle['end_price'] = data['close'].iloc[-1]
            current_cycle['duration_minutes'] = current_cycle['touches']
            
            hours = current_cycle['duration_minutes'] // 60
            minutes = current_cycle['duration_minutes'] % 60
            current_cycle['duration_formatted'] = f"{hours}:{minutes:02d}h"
            
            if direction == 'above':
                current_cycle['movement_type'] = 'resistencia' if current_cycle['touches'] > 10 else 'rebote_alto'
            else:
                current_cycle['movement_type'] = 'soporte' if current_cycle['touches'] > 10 else 'rebote_bajo'
            
            cycles.append(current_cycle)
        
        # Filtrar ciclos muy cortos (menos de 5 minutos)
        significant_cycles = [c for c in cycles if c['duration_minutes'] >= 5]
        
        return significant_cycles
    
    def _calculate_price_distribution(self, data: pd.DataFrame, min_price: float, price_range: float) -> Dict:
        """Calcula la distribución del precio en el rango"""
        
        # Dividir el rango en 5 zonas (0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
        zones = []
        zone_names = ["Zona Baja (0-20%)", "Zona Baja-Media (20-40%)", "Zona Media (40-60%)", 
                     "Zona Media-Alta (60-80%)", "Zona Alta (80-100%)"]
        
        for i in range(5):
            zone_min = min_price + (price_range * i / 5)
            zone_max = min_price + (price_range * (i + 1) / 5)
            
            # Contar tiempo en esta zona
            in_zone_count = 0
            for _, row in data.iterrows():
                if zone_min <= row['close'] <= zone_max:
                    in_zone_count += 1
            
            zone_percentage = (in_zone_count / len(data)) * 100
            
            zones.append({
                "zone_name": zone_names[i],
                "zone_min": zone_min,
                "zone_max": zone_max,
                "time_minutes": in_zone_count,
                "time_percentage": zone_percentage
            })
        
        return zones
    
    def _format_panorama_response(self, max_price: float, min_price: float, percentage_range: float,
                                high_zone_analysis: Dict, low_zone_analysis: Dict,
                                high_zone_cycles: List[Dict], low_zone_cycles: List[Dict]) -> str:
        """Formatea la respuesta del panorama de 48h"""
        
        # Información básica
        basic_info = f"rango 48h: ${min_price:,.0f}-${max_price:,.0f} ({percentage_range:.2f}%)"
        
        # Tiempo en zonas extremas
        high_zone_info = f"zona alta (>80%): {high_zone_analysis['time_formatted']} ({high_zone_analysis['time_percentage']:.1f}%)"
        low_zone_info = f"zona baja (<20%): {low_zone_analysis['time_formatted']} ({low_zone_analysis['time_percentage']:.1f}%)"
        
        # Ciclos más significativos
        cycles_info = []
        
        # Agregar ciclos de zona alta
        significant_high_cycles = [c for c in high_zone_cycles if c['duration_minutes'] >= 15][:2]
        for cycle in significant_high_cycles:
            cycles_info.append(f"{cycle['movement_type']} {cycle['duration_formatted']}")
        
        # Agregar ciclos de zona baja
        significant_low_cycles = [c for c in low_zone_cycles if c['duration_minutes'] >= 15][:2]
        for cycle in significant_low_cycles:
            cycles_info.append(f"{cycle['movement_type']} {cycle['duration_formatted']}")
        
        cycles_text = ", ".join(cycles_info) if cycles_info else "sin ciclos significativos"
        
        return f"{basic_info}; {high_zone_info}; {low_zone_info}; ciclos: {cycles_text}"