from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

class PercentageAnalyzer(BaseAnalyzer):
    """Analizador específico para cálculos de porcentaje de rango con ciclos secuenciales"""
    
    def analyze(self, question_params: Dict[str, Any]) -> str:
        """Implementación del método abstracto requerido"""
        return "Análisis genérico no implementado aún"
    
    def analyze_range_percentage(self, hours: float) -> Tuple[str, Dict[str, Any]]:
        """
        Pregunta específica: ¿De cuánto porcentaje fue el rango?
        
        Analiza ciclos secuenciales sin overlapping
        """
        if self.data is None or self.data.empty:
            return "Sin datos", {}
            
        range_data = self.get_time_range_data(hours)
        
        if range_data.empty:
            return f"Sin datos para {hours}h", {}
        
        # Análisis básico
        max_price = range_data['high'].max()
        min_price = range_data['low'].min()
        percentage_range = ((max_price - min_price) / min_price) * 100
        
        # Detectar ciclos secuenciales sin overlapping
        simple_answer, detailed_data = self._analyze_sequential_cycles(range_data, hours, percentage_range)
        
        return simple_answer, detailed_data
    
    def _analyze_sequential_cycles(self, data: pd.DataFrame, hours: float, total_volatility: float) -> Tuple[str, Dict]:
        """Análisis de ciclos secuenciales basados en movimientos rápidos"""
        
        # Detectar puntos de inflexión (movimientos rápidos)
        inflection_points = self._detect_inflection_points(data)
        
        if not inflection_points:
            return f"poca volatilidad {total_volatility:.2f}%, sin ciclos detectados", {
                "analysis_type": "no_cycles",
                "total_volatility": total_volatility,
                "time_range_hours": hours
            }
        
        # Crear ciclos secuenciales
        cycles = self._create_sequential_cycles(data, inflection_points)
        
        # Determinar tendencia general
        trend = self._determine_overall_trend(data, cycles)
        
        # Formatear respuesta
        simple_answer = self._format_sequential_response(cycles, total_volatility, trend)
        
        detailed_data = {
            "analysis_type": "sequential_cycles",
            "total_volatility": total_volatility,
            "trend": trend,
            "cycles": cycles,
            "inflection_points": inflection_points,
            "time_range_hours": hours,
            "cycles_count": len(cycles)
        }
        
        return simple_answer, detailed_data
    
    def _detect_inflection_points(self, data: pd.DataFrame) -> List[Dict]:
        """Detecta puntos de inflexión basados en movimientos rápidos - versión ampliada"""
        data = data.copy()
        inflection_points = []
        
        # Calcular movimientos acumulativos en ventanas más amplias
        for window in [5, 10, 15, 20, 30]:  # Ventanas más grandes
            data[f'change_{window}m'] = ((data['close'] - data['close'].shift(window)) / data['close'].shift(window)) * 100
        
        # Estrategia escalonada: empezar con umbrales bajos y subir gradualmente
        threshold_attempts = [0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
        
        for threshold in threshold_attempts:
            temp_points = []
            
            # Buscar movimientos en todas las ventanas
            for window in [5, 10, 15, 20, 30]:
                col = f'change_{window}m'
                if col in data.columns:
                    significant_moves = data[abs(data[col]) >= threshold].copy()
                    
                    for idx, row in significant_moves.iterrows():
                        if not self._is_point_too_close_enhanced(temp_points, idx, window):
                            temp_points.append({
                                'index': idx,
                                'timestamp': idx,
                                'price': row['close'],
                                'change_pct': row[col],
                                'window_minutes': window,
                                'threshold_used': threshold
                            })
            
            # Ordenar y limpiar puntos temporales
            temp_points = sorted(temp_points, key=lambda x: x['timestamp'])
            temp_points = self._remove_close_points_enhanced(temp_points)
            
            # Si tenemos entre 3-8 puntos bien distribuidos, usar estos
            if self._are_points_well_distributed(temp_points, data):
                inflection_points = temp_points
                break
            
            # Si tenemos muchos puntos, tomar una muestra bien distribuida
            if len(temp_points) > 8:
                inflection_points = self._sample_distributed_points(temp_points, data)
                break
        
        # Si no encontramos suficientes puntos, crear divisiones temporales
        if len(inflection_points) < 2:
            inflection_points = self._create_temporal_divisions(data)
        
        return inflection_points
    
    def _is_point_too_close_enhanced(self, existing_points: List[Dict], new_index, window_minutes: int) -> bool:
        """Versión mejorada para verificar proximidad de puntos"""
        min_separation = max(15, window_minutes)  # Mínimo 15 minutos de separación
        
        for point in existing_points:
            time_diff = abs((new_index - point['timestamp']).total_seconds() / 60)
            if time_diff < min_separation:
                return True
        return False
    
    def _remove_close_points_enhanced(self, points: List[Dict]) -> List[Dict]:
        """Elimina puntos muy cercanos con lógica mejorada"""
        if len(points) <= 1:
            return points
        
        filtered_points = [points[0]]
        
        for i in range(1, len(points)):
            current_point = points[i]
            should_keep = True
            
            for kept_point in filtered_points:
                time_diff = abs((current_point['timestamp'] - kept_point['timestamp']).total_seconds() / 60)
                
                # Mantener separación mínima de 20 minutos
                if time_diff < 20:
                    # Mantener el que tenga mayor cambio porcentual
                    if abs(current_point['change_pct']) <= abs(kept_point['change_pct']):
                        should_keep = False
                        break
                    else:
                        # Remover el punto anterior y mantener el actual
                        filtered_points = [p for p in filtered_points if p != kept_point]
            
            if should_keep:
                filtered_points.append(current_point)
        
        return filtered_points
    
    def _are_points_well_distributed(self, points: List[Dict], data: pd.DataFrame) -> bool:
        """Verifica si los puntos están bien distribuidos en el tiempo"""
        if len(points) < 2:
            return False
        
        total_duration = (data.index[-1] - data.index[0]).total_seconds() / 60  # minutos
        
        # Verificar que tengamos al menos 2-6 puntos
        if not (2 <= len(points) <= 6):
            return False
        
        # Verificar que cubran al menos el 70% del rango temporal
        first_point_time = points[0]['timestamp']
        last_point_time = points[-1]['timestamp']
        covered_duration = (last_point_time - first_point_time).total_seconds() / 60
        
        coverage_ratio = covered_duration / total_duration
        return coverage_ratio >= 0.5  # Al menos 50% de cobertura
    
    def _sample_distributed_points(self, points: List[Dict], data: pd.DataFrame) -> List[Dict]:
        """Toma una muestra bien distribuida de puntos"""
        if len(points) <= 6:
            return points
        
        # Dividir el tiempo en segmentos y tomar el mejor punto de cada segmento
        total_duration = (data.index[-1] - data.index[0]).total_seconds() / 60
        num_segments = min(6, len(points) // 2)
        segment_duration = total_duration / num_segments
        
        sampled_points = []
        start_time = data.index[0]
        
        for i in range(num_segments):
            segment_start = start_time + pd.Timedelta(minutes=i * segment_duration)
            segment_end = start_time + pd.Timedelta(minutes=(i + 1) * segment_duration)
            
            # Encontrar el mejor punto en este segmento
            segment_points = [
                p for p in points 
                if segment_start <= p['timestamp'] <= segment_end
            ]
            
            if segment_points:
                # Tomar el punto con mayor cambio porcentual
                best_point = max(segment_points, key=lambda x: abs(x['change_pct']))
                sampled_points.append(best_point)
        
        return sampled_points
    
    def _create_temporal_divisions(self, data: pd.DataFrame) -> List[Dict]:
        """Crea divisiones temporales cuando no hay suficientes puntos de inflexión"""
        points = []
        total_duration = len(data)  # minutos
        
        # Crear 3-4 divisiones temporales equidistantes
        num_divisions = 3 if total_duration < 120 else 4
        division_size = total_duration // num_divisions
        
        for i in range(1, num_divisions):
            division_index = data.index[i * division_size]
            division_row = data.loc[division_index]
            
            points.append({
                'index': division_index,
                'timestamp': division_index,
                'price': division_row['close'],
                'change_pct': 0.0,  # Sin cambio específico
                'window_minutes': 0,
                'threshold_used': 0.0,
                'type': 'temporal_division'
            })
        
        return points
    
    def _create_sequential_cycles(self, data: pd.DataFrame, inflection_points: List[Dict]) -> List[Dict]:
        """Crea ciclos secuenciales basados en puntos de inflexión"""
        if len(inflection_points) < 2:
            return []
        
        cycles = []
        data_start = data.index[0]
        data_end = data.index[-1]
        
        # Crear ciclos entre puntos de inflexión
        cycle_start = data_start
        
        for i, point in enumerate(inflection_points):
            cycle_end = point['timestamp']
            
            # Crear ciclo desde el inicio hasta este punto de inflexión
            if cycle_start < cycle_end:
                cycle_data = data.loc[cycle_start:cycle_end]
                cycle = self._create_cycle_info(cycle_data, cycle_start, cycle_end, i + 1)
                cycles.append(cycle)
                cycle_start = cycle_end
        
        # Crear último ciclo desde el último punto de inflexión hasta el final
        if cycle_start < data_end:
            cycle_data = data.loc[cycle_start:data_end]
            cycle = self._create_cycle_info(cycle_data, cycle_start, data_end, len(cycles) + 1)
            cycles.append(cycle)
        
        return cycles
    
    def _create_cycle_info(self, cycle_data: pd.DataFrame, start_time, end_time, cycle_number: int) -> Dict:
        """Crea información detallada de un ciclo"""
        if cycle_data.empty:
            return {}
        
        duration_minutes = len(cycle_data)
        hours_calc = duration_minutes // 60
        minutes = duration_minutes % 60
        
        start_price = cycle_data['close'].iloc[0]
        end_price = cycle_data['close'].iloc[-1]
        min_price = cycle_data['low'].min()
        max_price = cycle_data['high'].max()
        
        # Determinar dirección del ciclo
        price_change = ((end_price - start_price) / start_price) * 100
        if price_change > 0.05:
            direction = "alcista"
        elif price_change < -0.05:
            direction = "bajista"
        else:
            direction = "lateral"
        
        return {
            "cycle_number": cycle_number,
            "start_time": start_time,
            "end_time": end_time,
            "start_time_str": start_time.strftime('%H:%M'),
            "end_time_str": end_time.strftime('%H:%M'),
            "duration_minutes": duration_minutes,
            "duration_formatted": f"{hours_calc}:{minutes:02d}",
            "start_price": start_price,
            "end_price": end_price,
            "min_price": min_price,
            "max_price": max_price,
            "price_change_pct": price_change,
            "direction": direction,
            "range_label": f"${min_price:,.0f}-${max_price:,.0f}"
        }
    
    def _determine_overall_trend(self, data: pd.DataFrame, cycles: List[Dict]) -> str:
        """Determina la tendencia general del período"""
        if not cycles:
            return "lateral"
        
        first_price = data['close'].iloc[0]
        last_price = data['close'].iloc[-1]
        
        total_change = ((last_price - first_price) / first_price) * 100
        
        if total_change > 0.1:
            return "alcista"
        elif total_change < -0.1:
            return "bajista"
        else:
            return "lateral"
    
    def _format_sequential_response(self, cycles: List[Dict], total_volatility: float, trend: str) -> str:
        """Formatea respuesta con ciclos secuenciales"""
        if not cycles:
            return f"poca volatilidad {total_volatility:.2f}%, sin ciclos detectados"
        
        # Tomar los primeros 3-4 ciclos más significativos
        significant_cycles = [c for c in cycles if c.get('duration_minutes', 0) >= 5][:4]
        
        if not significant_cycles:
            return f"volatilidad {total_volatility:.2f}%, movimientos muy cortos"
        
        cycle_descriptions = []
        for cycle in significant_cycles:
            cycle_descriptions.append(
                f"{cycle['range_label']} ({cycle['start_time_str']}-{cycle['end_time_str']}, {cycle['duration_formatted']}h, {cycle['direction']})"
            )
        
        base_response = f"tendencia {trend} {total_volatility:.2f}%"
        cycles_detail = " → ".join(cycle_descriptions)
        
        return f"{base_response}: {cycles_detail}"