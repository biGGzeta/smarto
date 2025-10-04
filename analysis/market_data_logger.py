import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
from ml.feature_engineering import MarketFeatures
from ml.decision_tree import TradingSignal

class MarketDataLogger:
    """Logger completo para análisis profundo del mercado y decisiones"""
    
    def __init__(self, symbol: str = "ETHUSD_PERP", output_dir: str = "market_analysis"):
        self.symbol = symbol
        self.output_dir = output_dir
        self.current_session = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio si no existe
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Creado directorio de análisis: {self.output_dir}")
        
        # Archivos de logging
        self.session_file = os.path.join(self.output_dir, f"session_{self.current_session}.json")
        self.market_data_file = os.path.join(self.output_dir, f"market_data_{self.current_session}.csv")
        self.decisions_file = os.path.join(self.output_dir, f"decisions_{self.current_session}.json")
        self.summary_file = os.path.join(self.output_dir, "analysis_summary.json")
        
        # Datos de la sesión
        self.session_data = {
            "session_id": self.current_session,
            "symbol": self.symbol,
            "start_time": datetime.utcnow().isoformat(),
            "user": "biGGzeta",
            "timezone": "UTC",
            "version": "1.0"
        }
        
        print(f"📝 Iniciando logging de análisis profundo - Sesión: {self.current_session}")
    
    def log_complete_analysis(self, adaptive_results: Dict[str, Any], 
                            ml_results: Dict[str, Any], 
                            integrated_analysis: Dict[str, Any]) -> str:
        """Log completo de todo el análisis"""
        
        timestamp = datetime.utcnow()
        
        # 1. Extraer datos del mercado
        market_data = self._extract_market_data(adaptive_results, ml_results, timestamp)
        
        # 2. Extraer datos de decisiones
        decision_data = self._extract_decision_data(ml_results, integrated_analysis, timestamp)
        
        # 3. Guardar datos del mercado en CSV
        self._save_market_data_csv(market_data)
        
        # 4. Guardar datos de decisiones en JSON
        self._save_decision_data_json(decision_data)
        
        # 5. Actualizar resumen de sesión
        self._update_session_summary(market_data, decision_data)
        
        # 6. Crear análisis detallado
        analysis_report = self._create_detailed_analysis_report(
            adaptive_results, ml_results, integrated_analysis, timestamp
        )
        
        # 7. Guardar reporte completo
        report_file = os.path.join(self.output_dir, f"full_report_{self.current_session}.json")
        with open(report_file, 'w') as f:
            json.dump(analysis_report, f, indent=2)
        
        print(f"💾 Análisis completo guardado:")
        print(f"   📊 Datos mercado: {self.market_data_file}")
        print(f"   🎯 Decisiones: {self.decisions_file}")
        print(f"   📋 Reporte completo: {report_file}")
        
        return report_file
    
    def _extract_market_data(self, adaptive_results: Dict[str, Any], 
                           ml_results: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """Extraer todos los datos del mercado para CSV"""
        
        # Datos básicos
        max_min_data = adaptive_results.get('max_min_basic', ('', {}))[1] if isinstance(adaptive_results.get('max_min_basic'), tuple) else {}
        weekly_data = adaptive_results.get('weekly_adaptive', ('', {}))[1] if isinstance(adaptive_results.get('weekly_adaptive'), tuple) else {}
        panorama_data = adaptive_results.get('panorama_48h_adaptive', ('', {}))[1] if isinstance(adaptive_results.get('panorama_48h_adaptive'), tuple) else {}
        
        # Features ML si están disponibles
        features = ml_results.get('features') if 'features' in ml_results else None
        ml_data = ml_results.get('ml_data', {})
        
        market_data = {
            # Timestamp y meta
            'timestamp': timestamp.isoformat(),
            'session_id': self.current_session,
            'symbol': self.symbol,
            'user': 'biGGzeta',
            
            # Datos básicos del mercado
            'current_price': features.current_price if features else max_min_data.get('current_price', 0),
            'max_3h': max_min_data.get('max_price', 0),
            'min_3h': max_min_data.get('min_price', 0),
            'range_3h_pct': max_min_data.get('percentage_range', 0),
            
            # Datos semanales
            'week_min': weekly_data.get('week_min', 0),
            'week_max': weekly_data.get('week_max', 0),
            'week_range_pct': weekly_data.get('week_range_pct', 0),
            'price_position_pct': weekly_data.get('range_position_pct', 50),
            
            # Datos 48h
            'range_48h_pct': panorama_data.get('percentage_range', 0),
            'zone_high_time_pct': panorama_data.get('high_zone_analysis', {}).get('time_percentage', 0),
            'zone_low_time_pct': panorama_data.get('low_zone_analysis', {}).get('time_percentage', 0),
            
            # Condiciones del mercado
            'regime': features.regime if features else 'unknown',
            'volatility': features.volatility if features else 0,
            'trend_strength': features.trend_strength if features else 0,
            'direction_density': features.direction_density if features else 0,
            
            # Extremos y tendencias
            'maximos_trend': features.maximos_trend if features else 'unknown',
            'minimos_trend': features.minimos_trend if features else 'unknown',
            'maximos_strength': features.maximos_strength if features else 0,
            'minimos_strength': features.minimos_strength if features else 0,
            'extremes_alignment': features.extremes_alignment if features else False,
            'extremes_count_max': features.extremes_count_max if features else 0,
            'extremes_count_min': features.extremes_count_min if features else 0,
            
            # Momentum
            'momentum_1d': features.momentum_1d if features else 0,
            'momentum_3d': features.momentum_3d if features else 0,
            'momentum_7d': features.momentum_7d if features else 0,
            'momentum_strength': features.momentum_strength if features else 0,
            'momentum_direction': features.momentum_direction if features else 'unknown',
            
            # Actividad de wicks
            'wickdowns_count_3h': features.wickdowns_count_3h if features else 0,
            'wickups_count_3h': features.wickups_count_3h if features else 0,
            'total_wicks_3h': features.total_wicks_3h if features else 0,
            'strongest_wick_pct': features.strongest_wick_pct if features else 0,
            'wick_activity_ratio': features.wick_activity_ratio if features else 0,
            
            # Parámetros adaptativos
            'wick_threshold_used': features.wick_threshold_used if features else 0,
            'stability_threshold_used': features.stability_threshold_used if features else 0,
            'max_time_used': features.max_time_used if features else 0,
            'avg_window_size': features.to_ml_features().get('avg_window_size', 0) if features else 0,
            
            # Features derivadas
            'market_stress': features.market_stress if features else 0,
            'trend_momentum_alignment': features.trend_momentum_alignment if features else 0,
            'zone_pressure': features.zone_pressure if features else 'unknown',
            
            # Ciclos
            'cycles_count': features.cycles_count if features else 0,
            'cycles_quality_avg': features.cycles_quality_avg if features else 0,
            'cycle_consistency': features.cycle_consistency if features else 0,
            'dominant_cycle_direction': features.dominant_cycle_direction if features else 'unknown',
            
            # Targets ML
            'target_multiclass': ml_data.get('target_multiclass', 1),
            'target_binary_buy': ml_data.get('target_binary_buy', 0),
            'target_binary_sell': ml_data.get('target_binary_sell', 0),
            'target_confidence': ml_data.get('target_confidence', 50)
        }
        
        return market_data
    
    def _extract_decision_data(self, ml_results: Dict[str, Any], 
                             integrated_analysis: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """Extraer datos de decisiones y reasoning"""
        
        signal = ml_results.get('signal') if 'signal' in ml_results else None
        trading_signal = integrated_analysis.get('trading_signal', {})
        
        decision_data = {
            'timestamp': timestamp.isoformat(),
            'session_id': self.current_session,
            
            # Señal principal
            'signal_type': signal.signal_type.value if signal else 'UNKNOWN',
            'confidence': signal.confidence if signal else 0,
            'expected_move_pct': signal.expected_move_pct if signal else None,
            'risk_reward_ratio': signal.risk_reward_ratio if signal else None,
            'price_target': signal.price_target if signal else None,
            'stop_loss': signal.stop_loss if signal else None,
            'timeframe': signal.timeframe if signal else 'unknown',
            
            # Reasoning path completo
            'decision_reasoning': signal.reasoning if signal else [],
            'decision_path_length': len(signal.reasoning) if signal else 0,
            
            # Recomendaciones
            'trading_recommendations': integrated_analysis.get('trading_recommendations', {}),
            'confidence_analysis': integrated_analysis.get('confidence_analysis', {}),
            
            # Análisis de calidad
            'quality_metrics': ml_results.get('quality_metrics', {}),
            
            # Resumen ejecutivo
            'executive_summary': integrated_analysis.get('executive_summary', ''),
            
            # Condiciones que llevaron a la decisión
            'market_conditions_at_decision': {
                'regime': ml_results.get('features').regime if 'features' in ml_results else 'unknown',
                'volatility': ml_results.get('features').volatility if 'features' in ml_results else 0,
                'momentum_3d': ml_results.get('features').momentum_3d if 'features' in ml_results else 0,
                'price_position': ml_results.get('features').price_position_pct if 'features' in ml_results else 0,
                'extremes_alignment': ml_results.get('features').extremes_alignment if 'features' in ml_results else False
            }
        }
        
        return decision_data
    
    def _save_market_data_csv(self, market_data: Dict[str, Any]):
        """Guardar datos del mercado en CSV acumulativo"""
        
        # Convertir a DataFrame
        df_new = pd.DataFrame([market_data])
        
        # Si el archivo existe, agregarlo; si no, crearlo
        if os.path.exists(self.market_data_file):
            df_existing = pd.read_csv(self.market_data_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        # Guardar
        df_combined.to_csv(self.market_data_file, index=False)
    
    def _save_decision_data_json(self, decision_data: Dict[str, Any]):
        """Guardar datos de decisiones en JSON acumulativo"""
        
        # Si el archivo existe, cargarlo; si no, crear nuevo
        if os.path.exists(self.decisions_file):
            with open(self.decisions_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {'decisions': []}
        
        # Agregar nueva decisión
        existing_data['decisions'].append(decision_data)
        existing_data['total_decisions'] = len(existing_data['decisions'])
        existing_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Guardar
        with open(self.decisions_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def _update_session_summary(self, market_data: Dict[str, Any], decision_data: Dict[str, Any]):
        """Actualizar resumen de la sesión"""
        
        self.session_data.update({
            'last_analysis': datetime.utcnow().isoformat(),
            'total_analyses': self.session_data.get('total_analyses', 0) + 1,
            'current_market_state': {
                'price': market_data['current_price'],
                'regime': market_data['regime'],
                'volatility': market_data['volatility'],
                'momentum_3d': market_data['momentum_3d'],
                'position_pct': market_data['price_position_pct']
            },
            'latest_signal': {
                'type': decision_data['signal_type'],
                'confidence': decision_data['confidence'],
                'reasoning_summary': decision_data['decision_reasoning'][-1] if decision_data['decision_reasoning'] else 'No reasoning'
            }
        })
        
        # Guardar sesión actualizada
        with open(self.session_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
    
    def _create_detailed_analysis_report(self, adaptive_results: Dict[str, Any], 
                                       ml_results: Dict[str, Any], 
                                       integrated_analysis: Dict[str, Any], 
                                       timestamp: datetime) -> Dict[str, Any]:
        """Crear reporte detallado de análisis"""
        
        report = {
            'metadata': {
                'timestamp': timestamp.isoformat(),
                'session_id': self.current_session,
                'symbol': self.symbol,
                'user': 'biGGzeta',
                'analysis_version': '1.0',
                'timezone': 'UTC'
            },
            
            'market_analysis': {
                'adaptive_results_summary': self._summarize_adaptive_results(adaptive_results),
                'ml_features_summary': self._summarize_ml_features(ml_results),
                'decision_tree_path': ml_results.get('decision_path', []),
                'quality_assessment': ml_results.get('quality_metrics', {})
            },
            
            'trading_analysis': {
                'signal_details': self._extract_signal_details(ml_results),
                'recommendations': integrated_analysis.get('trading_recommendations', {}),
                'confidence_factors': integrated_analysis.get('confidence_analysis', {}),
                'executive_summary': integrated_analysis.get('executive_summary', '')
            },
            
            'technical_details': {
                'adaptive_parameters_used': self._extract_adaptive_parameters(adaptive_results),
                'feature_engineering_stats': self._extract_feature_stats(ml_results),
                'decision_tree_stats': self._extract_decision_stats(ml_results)
            },
            
            'raw_data': {
                'adaptive_results_full': adaptive_results,
                'ml_results_full': ml_results,
                'integrated_analysis_full': integrated_analysis
            }
        }
        
        return report
    
    def _summarize_adaptive_results(self, adaptive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Resumir resultados del análisis adaptativo"""
        
        summary = {}
        
        for key, result in adaptive_results.items():
            if isinstance(result, tuple) and len(result) >= 2:
                simple_answer = result[0]
                detailed_data = result[1] if isinstance(result[1], dict) else {}
                
                summary[key] = {
                    'simple_answer': simple_answer,
                    'key_metrics': {
                        'regime': detailed_data.get('market_conditions', {}).get('market_regime', 'unknown'),
                        'volatility': detailed_data.get('market_conditions', {}).get('volatility', 0),
                        'data_points': detailed_data.get('total_data_points', 0),
                        'adaptive_params': detailed_data.get('adaptive_parameters', {})
                    }
                }
        
        return summary
    
    def _summarize_ml_features(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Resumir features ML"""
        
        features = ml_results.get('features')
        if not features:
            return {}
        
        return {
            'total_features': len(features.to_ml_features()),
            'regime': features.regime,
            'key_indicators': {
                'momentum_3d': features.momentum_3d,
                'price_position_pct': features.price_position_pct,
                'volatility': features.volatility,
                'trend_strength': features.trend_strength,
                'extremes_alignment': features.extremes_alignment
            },
            'market_structure': {
                'maximos_trend': features.maximos_trend,
                'minimos_trend': features.minimos_trend,
                'maximos_strength': features.maximos_strength,
                'minimos_strength': features.minimos_strength
            }
        }
    
    def _extract_signal_details(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer detalles de la señal"""
        
        signal = ml_results.get('signal')
        if not signal:
            return {}
        
        return {
            'type': signal.signal_type.value,
            'confidence': signal.confidence,
            'expected_move_pct': signal.expected_move_pct,
            'risk_reward_ratio': signal.risk_reward_ratio,
            'price_target': signal.price_target,
            'stop_loss': signal.stop_loss,
            'timeframe': signal.timeframe,
            'reasoning_count': len(signal.reasoning),
            'full_reasoning': signal.reasoning
        }
    
    def _extract_adaptive_parameters(self, adaptive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer parámetros adaptativos usados"""
        
        params = {}
        
        for key, result in adaptive_results.items():
            if isinstance(result, tuple) and len(result) >= 2:
                detailed_data = result[1] if isinstance(result[1], dict) else {}
                adaptive_params = detailed_data.get('adaptive_parameters', {})
                if adaptive_params:
                    params[key] = adaptive_params
        
        return params
    
    def _extract_feature_stats(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer estadísticas de feature engineering"""
        
        ml_data = ml_results.get('ml_data', {})
        
        return {
            'feature_count': ml_data.get('feature_count', 0),
            'target_multiclass': ml_data.get('target_multiclass', 0),
            'target_confidence': ml_data.get('target_confidence', 0),
            'market_conditions': ml_data.get('market_conditions', {})
        }
    
    def _extract_decision_stats(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer estadísticas del árbol de decisiones"""
        
        return {
            'decision_path_length': len(ml_results.get('decision_path', [])),
            'quality_metrics': ml_results.get('quality_metrics', {})
        }
    
    def create_analysis_dashboard_data(self) -> str:
        """Crear datos para dashboard de análisis"""
        
        # Leer datos históricos de la sesión
        if not os.path.exists(self.market_data_file):
            return "No hay datos disponibles para dashboard"
        
        df = pd.read_csv(self.market_data_file)
        
        dashboard_data = {
            'session_summary': self.session_data,
            'total_analyses': len(df),
            'price_evolution': df[['timestamp', 'current_price', 'price_position_pct']].to_dict('records'),
            'signal_distribution': df['target_multiclass'].value_counts().to_dict(),
            'regime_distribution': df['regime'].value_counts().to_dict(),
            'momentum_stats': {
                'avg_momentum_3d': df['momentum_3d'].mean(),
                'max_momentum_3d': df['momentum_3d'].max(),
                'min_momentum_3d': df['momentum_3d'].min(),
                'momentum_volatility': df['momentum_3d'].std()
            },
            'volatility_stats': {
                'avg_volatility': df['volatility'].mean(),
                'max_volatility': df['volatility'].max(),
                'regime_volatility_by_regime': df.groupby('regime')['volatility'].mean().to_dict()
            }
        }
        
        dashboard_file = os.path.join(self.output_dir, f"dashboard_data_{self.current_session}.json")
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        print(f"📊 Dashboard data creado: {dashboard_file}")
        return dashboard_file
    
    def generate_session_report(self) -> str:
        """Generar reporte final de la sesión"""
        
        if not os.path.exists(self.market_data_file):
            return "No hay datos para generar reporte"
        
        df = pd.read_csv(self.market_data_file)
        
        report = f"""
# REPORTE DE SESIÓN DE ANÁLISIS - {self.current_session}

## Información General
- **Usuario**: biGGzeta
- **Símbolo**: {self.symbol}
- **Fecha**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Total de Análisis**: {len(df)}

## Resumen del Mercado
- **Precio Promedio**: ${df['current_price'].mean():.2f}
- **Rango de Precios**: ${df['current_price'].min():.2f} - ${df['current_price'].max():.2f}
- **Posición Promedio en Rango Semanal**: {df['price_position_pct'].mean():.1f}%
- **Volatilidad Promedio**: {df['volatility'].mean():.2f}%

## Análisis de Momentum
- **Momentum 3d Promedio**: {df['momentum_3d'].mean():.2f}%
- **Momentum Máximo**: {df['momentum_3d'].max():.2f}%
- **Momentum Mínimo**: {df['momentum_3d'].min():.2f}%

## Distribución de Regímenes
{df['regime'].value_counts().to_string()}

## Distribución de Señales
{df.groupby('target_multiclass').size().to_string()}

## Calidad de Decisiones
- **Confianza Promedio**: {df['target_confidence'].mean():.1f}%
- **Alignment de Extremos**: {(df['extremes_alignment'].sum() / len(df) * 100):.1f}% del tiempo

---
Generado automáticamente por el Sistema de Trading Adaptativo ML
"""
        
        report_file = os.path.join(self.output_dir, f"session_report_{self.current_session}.md")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📋 Reporte de sesión generado: {report_file}")
        return report_file