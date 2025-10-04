from .adaptive_base_analyzer import AdaptiveBaseAnalyzer
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

class AdaptivePanoramaAnalyzer(AdaptiveBaseAnalyzer):
    """Analizador de panorama 48h con parámetros auto-adaptativos"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Implementación del método abstracto requerido"""
        hours = question_params.get('hours', 48)
        simple_answer, _ = self.analyze_48h_panorama_adaptive(hours)
        return simple_answer
    
    def analyze_48h_panorama_adaptive(self, hours: float = 48) -> Tuple[str, Dict[str, Any]]:
        """
        Análisis de panorama 48h con zonas y ciclos adaptativos
        """
        if self.data is None or self.data.empty:
            return "Sin datos", {}
            
        range_data = self.get_time_range_data(hours)
        
        if range_data.empty:
            return f"Sin datos para {hours}h", {}
        
        print(f"🔄 Iniciando análisis adaptativo de panorama {hours}h...")
        
        # Analizar condiciones y obtener parámetros adaptativos
        self.load_data_and_analyze_conditions(range_data, f"{hours}h")
        
        # Análisis básico del rango
        max_price = range_data['high'].max()
        min_price = range_data['low'].min()
        price_range = max_price - min_price
        percentage_range = (price_range / min_price) * 100
        
        # Obtener umbrales de zona adaptativos
        zone_low, zone_high = self.get_adaptive_zone_thresholds()
        
        # Definir zonas adaptativas
        zone_80_threshold = min_price + (price_range * (zone_high / 100))
        zone_20_threshold = min_price + (price_range * (zone_low / 100))
        
        # Analizar tiempo en zonas con criterios adaptativos
        high_zone_analysis = self._analyze_adaptive_zone_time(
            range_data, zone_80_threshold, 'above', f'alta (>{zone_high:.0f}%)'
        )
        low_zone_analysis = self._analyze_adaptive_zone_time(
            range_data, zone_20_threshold, 'below', f'baja (<{zone_low:.0f}%)'
        )
        
        # Detectar ciclos simplificados
        high_zone_cycles = self._detect_simple_zone_cycles(range_data, zone_80_threshold, 'above')
        low_zone_cycles = self._detect_simple_zone_cycles(range_data, zone_20_threshold, 'below')
        
        # Formatear respuesta
        simple_answer = self._format_adaptive_panorama_response(
            max_price, min_price, percentage_range, 
            high_zone_analysis, low_zone_analysis,
            high_zone_cycles, low_zone_cycles
        )
        
        detailed_data = {
            "analysis_type": "adaptive_48h_panorama",
            "time_range_hours": hours,
            "market_conditions": self.market_conditions,
            "adaptive_parameters": self.current_parameters,
            "max_price": max_price,
            "min_price": min_price,
            "price_range": price_range,
            "percentage_range": percentage_range,
            "zone_high_threshold": zone_80_threshold,
            "zone_low_threshold": zone_20_threshold,
            "zone_percentages": {"high": zone_high, "low": zone_low},
            "high_zone_analysis": high_zone_analysis,
            "low_zone_analysis": low_zone_analysis,
            "high_zone_cycles": high_zone_cycles,
            "low_zone_cycles": low_zone_cycles
        }
        
        return simple_answer, detailed_data
    
    def _analyze_adaptive_zone_time(self, data: pd.DataFrame, threshold: float, 
                                   direction: str, zone_name: str) -> Dict:
        """Analizar tiempo en zona con criterios adaptativos - versión simplificada"""
        
        regime = self.market_conditions.get("market_regime", "smooth_ranging")
        
        # Criterio simplificado para estar en zona
        if direction == 'above':
            in_zone_mask = data['close'] >= threshold
        else:  # below
            in_zone_mask = data['close'] <= threshold
        
        in_zone_data = data[in_zone_mask]
        total_time_in_zone = len(in_zone_data)
        
        total_time = len(data)
        time_percentage = (total_time_in_zone / total_time) * 100
        
        hours = total_time_in_zone // 60
        minutes = total_time_in_zone % 60
        
        return {
            "zone_name": zone_name,
            "threshold": threshold,
            "direction": direction,
            "total_minutes": total_time_in_zone,
            "time_formatted": f"{hours}:{minutes:02d}h",
            "time_percentage": round(time_percentage, 1),
            "total_touches": len(in_zone_data),
            "quality_metric": 75,  # Valor fijo simplificado
            "quality_name": "estabilidad",
            "regime": regime,
            "adaptive_criteria": True
        }
    
    def _detect_simple_zone_cycles(self, data: pd.DataFrame, threshold: float, direction: str) -> List[Dict]:
        """Detectar ciclos en zonas de forma simplificada"""
        cycles = []
        regime = self.market_conditions.get("market_regime", "smooth_ranging")
        
        # Buscar períodos continuos en la zona
        if direction == 'above':
            in_zone_mask = data['close'] >= threshold
        else:
            in_zone_mask = data['close'] <= threshold
        
        # Encontrar segmentos continuos
        current_segment = None
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            is_in_zone = in_zone_mask.iloc[i]
            
            if is_in_zone and current_segment is None:
                current_segment = {
                    'start_time': timestamp,
                    'start_price': row['close'],
                    'touches': 1
                }
            elif is_in_zone and current_segment is not None:
                current_segment['touches'] += 1
                current_segment['end_time'] = timestamp
                current_segment['end_price'] = row['close']
            elif not is_in_zone and current_segment is not None:
                # Fin del segmento
                if current_segment['touches'] >= 10:  # Mínimo 10 minutos
                    
                    duration_hours = current_segment['touches'] / 60
                    
                    if current_segment['touches'] > 30:
                        movement_type = 'consolidación' if direction == 'above' else 'soporte'
                    else:
                        movement_type = 'rebote_alto' if direction == 'above' else 'rebote_bajo'
                    
                    cycles.append({
                        'start_time': current_segment['start_time'],
                        'end_time': current_segment['end_time'],
                        'duration_minutes': current_segment['touches'],
                        'duration_formatted': f"{int(duration_hours)}:{int((duration_hours % 1) * 60):02d}h",
                        'movement_type': movement_type,
                        'efficiency': 80,  # Valor fijo
                        'regime': regime
                    })
                
                current_segment = None
        
        return cycles
    
    def _format_adaptive_panorama_response(self, max_price: float, min_price: float, percentage_range: float,
                                         high_zone_analysis: Dict, low_zone_analysis: Dict,
                                         high_zone_cycles: List[Dict], low_zone_cycles: List[Dict]) -> str:
        """Formatear respuesta del panorama adaptativo"""
        
        zone_high = self.current_parameters.get("zone_high", 80)
        zone_low = self.current_parameters.get("zone_low", 20)
        regime = self.market_conditions.get("market_regime", "unknown")
        volatility = self.market_conditions.get("volatility", 0)
        
        basic_info = f"rango 48h adaptativo: ${min_price:,.0f}-${max_price:,.0f} ({percentage_range:.2f}%)"
        
        high_zone_info = f"zona {high_zone_analysis['zone_name']}: {high_zone_analysis['time_formatted']} ({high_zone_analysis['time_percentage']}%)"
        low_zone_info = f"zona {low_zone_analysis['zone_name']}: {low_zone_analysis['time_formatted']} ({low_zone_analysis['time_percentage']}%)"
        
        cycles_info = []
        
        # Agregar ciclos significativos
        significant_high_cycles = [c for c in high_zone_cycles if c['duration_minutes'] >= 10][:2]
        for cycle in significant_high_cycles:
            cycles_info.append(f"{cycle['movement_type']} {cycle['duration_formatted']}")
        
        significant_low_cycles = [c for c in low_zone_cycles if c['duration_minutes'] >= 10][:2]
        for cycle in significant_low_cycles:
            cycles_info.append(f"{cycle['movement_type']} {cycle['duration_formatted']}")
        
        cycles_text = ", ".join(cycles_info) if cycles_info else "sin ciclos significativos"
        
        return f"{basic_info} ({regime}, vol {volatility:.1f}%); {high_zone_info}; {low_zone_info}; ciclos: {cycles_text}"