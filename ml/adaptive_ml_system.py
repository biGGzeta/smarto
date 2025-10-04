from .feature_engineering import FeatureExtractor, MarketFeatures
from .decision_tree import AdaptiveDecisionTree, TradingSignal
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

class AdaptiveMLSystem:
    """Sistema completo que convierte análisis adaptativo en features y señales ML"""
    
    def __init__(self, save_data: bool = True, data_dir: str = "ml_data"):
        self.feature_extractor = FeatureExtractor()
        self.decision_tree = AdaptiveDecisionTree()
        self.save_data = save_data
        self.data_dir = data_dir
        
        # Crear directorio de datos si no existe
        if self.save_data and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"📁 Creado directorio: {self.data_dir}")
        
        self.pipeline_history = []
        
    def analyze_and_signal(self, adaptive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline completo: análisis adaptativo → features → señal ML"""
        
        print("🚀 Iniciando pipeline ML completo...")
        
        try:
            # Paso 1: Extraer features estructuradas
            print("📊 Paso 1: Extrayendo features...")
            features = self.feature_extractor.extract_features(adaptive_results)
            
            # Paso 2: Evaluar através del árbol de decisiones
            print("🌳 Paso 2: Evaluando árbol de decisiones...")
            signal = self.decision_tree.evaluate(features)
            
            # Paso 3: Preparar datos para ML futuro
            print("🤖 Paso 3: Preparando datos ML...")
            ml_data = self._prepare_ml_data(features, signal)
            
            # Paso 4: Análisis de calidad del pipeline
            quality_metrics = self._analyze_pipeline_quality(features, signal)
            
            # Resultado completo
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "features": features,
                "signal": signal,
                "ml_data": ml_data,
                "decision_path": signal.reasoning,
                "quality_metrics": quality_metrics,
                "pipeline_version": "1.0"
            }
            
            # Guardar datos si está habilitado
            if self.save_data:
                self._save_pipeline_data(result)
            
            # Guardar en historial
            self.pipeline_history.append(result)
            
            print(f"✅ Pipeline ML completado: {signal.signal_type.value} (confianza: {signal.confidence}%)")
            
            return result
            
        except Exception as e:
            print(f"❌ Error en pipeline ML: {str(e)}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "status": "failed"
            }
    
    def _prepare_ml_data(self, features: MarketFeatures, signal: TradingSignal) -> Dict[str, Any]:
        """Preparar datos en formato óptimo para ML futuro"""
        
        # Features numéricas para ML
        ml_features = features.to_ml_features()
        
        # Agregar información de la señal como target
        signal_encoding = {
            "STRONG_BUY": 4,
            "BUY": 3, 
            "WEAK_BUY": 2,
            "HOLD": 1,
            "WEAK_SELL": 0,
            "SELL": -1,
            "STRONG_SELL": -2
        }
        
        # Target para entrenar modelos futuros
        target_numeric = signal_encoding.get(signal.signal_type.value, 1)
        target_binary_buy = 1 if target_numeric > 1 else 0  # Para clasificación binaria
        target_binary_sell = 1 if target_numeric < 1 else 0
        
        return {
            # Features de entrada
            "features": ml_features,
            "feature_names": list(ml_features.keys()),
            "feature_count": len(ml_features),
            
            # Targets para diferentes tipos de modelos
            "target_multiclass": target_numeric,  # Para clasificación 7-clases
            "target_binary_buy": target_binary_buy,  # Para modelo buy/no-buy
            "target_binary_sell": target_binary_sell,  # Para modelo sell/no-sell
            "target_confidence": signal.confidence,  # Para regresión de confianza
            
            # Información adicional
            "signal_type": signal.signal_type.value,
            "expected_move": signal.expected_move_pct,
            "risk_reward": signal.risk_reward_ratio,
            
            # Metadatos
            "regime": features.regime,
            "symbol": features.symbol,
            "market_conditions": {
                "volatility": features.volatility,
                "trend_strength": features.trend_strength,
                "price_position": features.price_position_pct
            }
        }
    
    def _analyze_pipeline_quality(self, features: MarketFeatures, signal: TradingSignal) -> Dict[str, Any]:
        """Analizar calidad y consistencia del pipeline"""
        
        quality_metrics = {
            "feature_completeness": 0,
            "signal_confidence": signal.confidence,
            "decision_path_length": len(signal.reasoning),
            "risk_reward_quality": 0,
            "market_condition_clarity": 0,
            "data_quality_score": 0
        }
        
        # 1. Completitud de features
        ml_features = features.to_ml_features()
        non_zero_features = sum(1 for v in ml_features.values() if v != 0)
        quality_metrics["feature_completeness"] = (non_zero_features / len(ml_features)) * 100
        
        # 2. Calidad de risk-reward
        if signal.risk_reward_ratio:
            if signal.risk_reward_ratio >= 2:
                quality_metrics["risk_reward_quality"] = 100
            elif signal.risk_reward_ratio >= 1.5:
                quality_metrics["risk_reward_quality"] = 75
            elif signal.risk_reward_ratio >= 1:
                quality_metrics["risk_reward_quality"] = 50
            else:
                quality_metrics["risk_reward_quality"] = 25
        else:
            quality_metrics["risk_reward_quality"] = 0
        
        # 3. Claridad de condiciones del mercado
        if features.regime != "unknown" and features.volatility > 0:
            clarity = min(100, features.trend_strength * 100 + 25)
            quality_metrics["market_condition_clarity"] = clarity
        else:
            quality_metrics["market_condition_clarity"] = 25
        
        # 4. Score general de calidad de datos
        extremes_quality = 50
        if features.extremes_alignment:
            extremes_quality += 25
        if features.extremes_count_max > 2 and features.extremes_count_min > 2:
            extremes_quality += 25
        
        quality_metrics["data_quality_score"] = min(100, extremes_quality)
        
        return quality_metrics
    
    def _save_pipeline_data(self, result: Dict[str, Any]):
        """Guardar datos del pipeline para análisis futuro"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # 1. Guardar features en CSV (acumulativo)
            features_file = os.path.join(self.data_dir, "features_history.csv")
            self._save_features_to_csv(result["features"], features_file)
            
            # 2. Guardar señal individual en JSON
            signal_file = os.path.join(self.data_dir, f"signal_{timestamp}.json")
            self._save_signal_to_json(result["signal"], signal_file)
            
            # 3. Guardar resultado completo del pipeline
            pipeline_file = os.path.join(self.data_dir, f"pipeline_{timestamp}.json")
            self._save_pipeline_result(result, pipeline_file)
            
            # 4. Actualizar archivo de resumen
            summary_file = os.path.join(self.data_dir, "pipeline_summary.json")
            self._update_pipeline_summary(result, summary_file)
            
            print(f"💾 Datos del pipeline guardados en {self.data_dir}")
            
        except Exception as e:
            print(f"⚠️ Error guardando datos del pipeline: {str(e)}")
            # No fallar el pipeline completo por un error de guardado
    
    def _save_features_to_csv(self, features: MarketFeatures, filepath: str):
        """Guardar features en CSV acumulativo"""
        try:
            # Preparar fila de datos
            row_data = features.to_ml_features()
            row_data.update({
                'timestamp': features.timestamp.isoformat(),
                'symbol': features.symbol,
                'regime': features.regime,
                'maximos_trend': features.maximos_trend,
                'minimos_trend': features.minimos_trend,
                'momentum_direction': features.momentum_direction,
                'zone_pressure': features.zone_pressure,
                'dominant_cycle_direction': features.dominant_cycle_direction
            })
            
            # Crear DataFrame con la nueva fila
            new_row_df = pd.DataFrame([row_data])
            
            # Si el archivo existe, agregarlo; si no, crearlo
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath)
                combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            else:
                combined_df = new_row_df
            
            # Guardar archivo actualizado
            combined_df.to_csv(filepath, index=False)
            
        except Exception as e:
            print(f"⚠️ Error guardando features CSV: {str(e)}")
    
    def _save_signal_to_json(self, signal: TradingSignal, filepath: str):
        """Guardar señal individual en JSON"""
        try:
            signal_data = signal.to_dict()
            
            with open(filepath, 'w') as f:
                json.dump(signal_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error guardando señal JSON: {str(e)}")
    
    def _save_pipeline_result(self, result: Dict[str, Any], filepath: str):
        """Guardar resultado completo del pipeline"""
        try:
            # Preparar datos serializables
            serializable_result = {
                "timestamp": result["timestamp"],
                "pipeline_version": result.get("pipeline_version", "1.0"),
                "quality_metrics": result.get("quality_metrics", {}),
                "ml_data": {
                    "feature_count": result["ml_data"].get("feature_count", 0),
                    "target_multiclass": result["ml_data"].get("target_multiclass", 0),
                    "target_binary_buy": result["ml_data"].get("target_binary_buy", 0),
                    "target_binary_sell": result["ml_data"].get("target_binary_sell", 0),
                    "target_confidence": result["ml_data"].get("target_confidence", 0),
                    "signal_type": result["ml_data"].get("signal_type", "UNKNOWN"),
                    "regime": result["ml_data"].get("regime", "unknown")
                },
                "market_conditions": {
                    "regime": result["features"].regime,
                    "volatility": result["features"].volatility,
                    "trend_strength": result["features"].trend_strength,
                    "price_position_pct": result["features"].price_position_pct,
                    "momentum_3d": result["features"].momentum_3d,
                    "extremes_alignment": result["features"].extremes_alignment
                },
                "signal_summary": {
                    "type": result["signal"].signal_type.value,
                    "confidence": result["signal"].confidence,
                    "timeframe": result["signal"].timeframe,
                    "expected_move_pct": result["signal"].expected_move_pct,
                    "risk_reward_ratio": result["signal"].risk_reward_ratio,
                    "reasoning_count": len(result["signal"].reasoning)
                },
                "decision_path": result["decision_path"]
            }
            
            with open(filepath, 'w') as f:
                json.dump(serializable_result, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error guardando resultado pipeline: {str(e)}")
    
    def _update_pipeline_summary(self, result: Dict[str, Any], filepath: str):
        """Actualizar archivo de resumen del pipeline"""
        try:
            # Cargar resumen existente o crear nuevo
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    summary = json.load(f)
            else:
                summary = {
                    "total_executions": 0,
                    "signal_distribution": {},
                    "regime_distribution": {},
                    "average_confidence": 0,
                    "first_execution": None,
                    "last_execution": None,
                    "quality_trends": []
                }
            
            # Actualizar estadísticas
            summary["total_executions"] += 1
            summary["last_execution"] = result["timestamp"]
            
            if summary["first_execution"] is None:
                summary["first_execution"] = result["timestamp"]
            
            # Distribución de señales
            signal_type = result["signal"].signal_type.value
            if signal_type not in summary["signal_distribution"]:
                summary["signal_distribution"][signal_type] = 0
            summary["signal_distribution"][signal_type] += 1
            
            # Distribución de regímenes
            regime = result["features"].regime
            if regime not in summary["regime_distribution"]:
                summary["regime_distribution"][regime] = 0
            summary["regime_distribution"][regime] += 1
            
            # Confianza promedio
            total_confidence = summary.get("total_confidence", 0) + result["signal"].confidence
            summary["average_confidence"] = total_confidence / summary["total_executions"]
            summary["total_confidence"] = total_confidence
            
            # Tendencias de calidad
            quality_entry = {
                "timestamp": result["timestamp"],
                "feature_completeness": result["quality_metrics"]["feature_completeness"],
                "signal_confidence": result["quality_metrics"]["signal_confidence"],
                "market_clarity": result["quality_metrics"]["market_condition_clarity"]
            }
            summary["quality_trends"].append(quality_entry)
            
            # Mantener solo últimas 100 entradas de calidad
            if len(summary["quality_trends"]) > 100:
                summary["quality_trends"] = summary["quality_trends"][-100:]
            
            # Guardar resumen actualizado
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error actualizando resumen pipeline: {str(e)}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema ML"""
        
        feature_stats = {
            "total_feature_extractions": len(self.feature_extractor.feature_history),
            "last_extraction": self.feature_extractor.feature_history[-1].timestamp.isoformat() if self.feature_extractor.feature_history else None
        }
        
        decision_stats = self.decision_tree.get_performance_stats()
        
        pipeline_stats = {
            "total_executions": len(self.pipeline_history),
            "success_rate": 100.0,  # Por ahora
            "last_execution": self.pipeline_history[-1]["timestamp"] if self.pipeline_history else None
        }
        
        return {
            "system_version": "1.0",
            "pipeline_stats": pipeline_stats,
            "feature_extraction_stats": feature_stats,
            "decision_tree_stats": decision_stats,
            "data_directory": self.data_dir,
            "save_data_enabled": self.save_data
        }
    
    def load_historical_data(self) -> Dict[str, Any]:
        """Cargar datos históricos guardados"""
        try:
            result = {
                "features_data": None,
                "pipeline_summary": None,
                "available_signals": []
            }
            
            # Cargar features históricas
            features_file = os.path.join(self.data_dir, "features_history.csv")
            if os.path.exists(features_file):
                result["features_data"] = pd.read_csv(features_file)
                print(f"📊 Cargadas {len(result['features_data'])} filas de features históricas")
            
            # Cargar resumen del pipeline
            summary_file = os.path.join(self.data_dir, "pipeline_summary.json")
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    result["pipeline_summary"] = json.load(f)
                print(f"📈 Cargado resumen con {result['pipeline_summary']['total_executions']} ejecuciones")
            
            # Listar señales disponibles
            signal_files = [f for f in os.listdir(self.data_dir) if f.startswith("signal_") and f.endswith(".json")]
            result["available_signals"] = sorted(signal_files)
            
            return result
            
        except Exception as e:
            print(f"❌ Error cargando datos históricos: {str(e)}")
            return {"error": str(e)}
    
    def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """Analizar importancia de features basado en datos históricos"""
        try:
            historical_data = self.load_historical_data()
            
            if historical_data["features_data"] is None:
                return {"error": "No hay datos históricos disponibles"}
            
            df = historical_data["features_data"]
            
            # Análisis básico de correlaciones y estadísticas
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            analysis = {
                "total_samples": len(df),
                "feature_stats": {},
                "regime_distribution": df['regime'].value_counts().to_dict() if 'regime' in df.columns else {},
                "correlation_with_confidence": {},
                "top_varying_features": {}
            }
            
            # Estadísticas por feature
            for col in numeric_columns:
                if col in df.columns:
                    analysis["feature_stats"][col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "non_zero_pct": float((df[col] != 0).mean() * 100)
                    }
            
            # Correlación con confianza de señales (si está disponible)
            if 'target_confidence' in df.columns:
                for col in numeric_columns[:20]:  # Top 20 features
                    if col != 'target_confidence':
                        corr = df[col].corr(df['target_confidence'])
                        if not np.isnan(corr):
                            analysis["correlation_with_confidence"][col] = float(corr)
            
            # Features con más variación
            feature_variations = {}
            for col in numeric_columns:
                if col in df.columns and df[col].std() > 0:
                    cv = df[col].std() / abs(df[col].mean()) if df[col].mean() != 0 else float('inf')
                    feature_variations[col] = cv
            
            # Top 10 features más variables
            sorted_variations = sorted(feature_variations.items(), key=lambda x: x[1], reverse=True)
            analysis["top_varying_features"] = dict(sorted_variations[:10])
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error en análisis de importancia: {str(e)}")
            return {"error": str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Limpiar datos antiguos del directorio ML"""
        try:
            import time
            from pathlib import Path
            
            current_time = time.time()
            cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
            
            deleted_files = []
            
            for file_path in Path(self.data_dir).glob("*"):
                if file_path.is_file():
                    file_time = file_path.stat().st_mtime
                    
                    # Mantener archivos principales
                    if file_path.name in ["features_history.csv", "pipeline_summary.json"]:
                        continue
                    
                    # Eliminar archivos antiguos
                    if file_time < cutoff_time:
                        file_path.unlink()
                        deleted_files.append(file_path.name)
            
            print(f"🧹 Limpieza completada: {len(deleted_files)} archivos eliminados")
            return {"deleted_files": deleted_files, "days_kept": days_to_keep}
            
        except Exception as e:
            print(f"❌ Error en limpieza: {str(e)}")
            return {"error": str(e)}