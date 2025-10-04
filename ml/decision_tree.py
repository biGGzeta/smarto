import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from .feature_engineering import MarketFeatures
import json
from datetime import datetime

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TradingSignal:
    """Señal de trading generada por el árbol"""
    signal_type: SignalType
    confidence: float  # 0-100
    reasoning: List[str]  # Lista de condiciones que se cumplieron
    features_used: MarketFeatures
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    timeframe: str = "short"  # short, medium, long
    expected_move_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para logging"""
        return {
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "timeframe": self.timeframe,
            "expected_move_pct": self.expected_move_pct,
            "risk_reward_ratio": self.risk_reward_ratio,
            "timestamp": datetime.utcnow().isoformat()
        }

@dataclass
class DecisionNode:
    """Nodo individual del árbol de decisiones"""
    name: str
    condition: Callable[[MarketFeatures], bool]
    true_action: Union['DecisionNode', Callable[[MarketFeatures], TradingSignal]]
    false_action: Union['DecisionNode', Callable[[MarketFeatures], TradingSignal]]
    confidence_weight: float = 1.0
    description: str = ""

class AdaptiveDecisionTree:
    """Árbol de decisiones que usa features adaptativos para generar señales"""
    
    def __init__(self):
        self.tree = self._build_decision_tree()
        self.signal_history = []
        self.decision_stats = {
            "total_evaluations": 0,
            "signal_distribution": {},
            "accuracy_by_signal": {}
        }
        
    def _build_decision_tree(self) -> DecisionNode:
        """Construir árbol de decisiones completo basado en lógica de trading"""
        
        print("🌳 Construyendo árbol de decisiones adaptativo...")
        
        # Nodo raíz: Análisis de régimen del mercado
        root_node = DecisionNode(
            name="market_regime_analysis",
            condition=lambda f: f.regime in ["high_volatility", "trending"],
            true_action=self._build_volatile_trending_branch(),
            false_action=self._build_calm_ranging_branch(),
            confidence_weight=1.0,
            description="Determinar estrategia principal según régimen del mercado"
        )
        
        print("✅ Árbol de decisiones construido")
        return root_node
    
    def _build_volatile_trending_branch(self) -> DecisionNode:
        """Branch para mercados volátiles o en tendencia fuerte"""
        
        # Nivel 2A: Análisis de momentum en mercados volátiles
        momentum_node = DecisionNode(
            name="volatile_momentum_check",
            condition=lambda f: f.momentum_3d > 3 and f.trend_strength > 0.7,
            true_action=self._build_strong_momentum_branch(),
            false_action=DecisionNode(
                name="volatile_position_check",
                condition=lambda f: f.price_position_pct < 20,  # Zona muy baja
                true_action=self._build_oversold_volatile_branch(),
                false_action=DecisionNode(
                    name="volatile_high_position",
                    condition=lambda f: f.price_position_pct > 80,  # Zona muy alta
                    true_action=self._build_overbought_volatile_branch(),
                    false_action=self._build_volatile_middle_branch(),
                    description="Análisis zona alta en mercado volátil"
                ),
                description="Análisis posición en mercado volátil sin momentum fuerte"
            ),
            description="Verificar momentum fuerte en mercado volátil"
        )
        
        return momentum_node
    
    def _build_calm_ranging_branch(self) -> DecisionNode:
        """Branch para mercados calmos o laterales"""
        
        # Nivel 2B: Análisis de estructura de extremos
        structure_node = DecisionNode(
            name="extremes_structure_analysis",
            condition=lambda f: f.maximos_trend == "crecientes" and f.minimos_trend == "crecientes",
            true_action=self._build_bullish_structure_branch(),
            false_action=DecisionNode(
                name="bearish_structure_check",
                condition=lambda f: f.maximos_trend == "decrecientes" and f.minimos_trend == "decrecientes",
                true_action=self._build_bearish_structure_branch(),
                false_action=DecisionNode(
                    name="mixed_structure_check",
                    condition=lambda f: f.extremes_alignment == False,  # Extremos divergentes
                    true_action=self._build_divergent_structure_branch(),
                    false_action=self._build_stable_structure_branch(),
                    description="Análisis estructura divergente"
                ),
                description="Verificar estructura bajista consistente"
            ),
            description="Análisis de estructura de extremos en mercado calmo"
        )
        
        return structure_node
    
    def _build_strong_momentum_branch(self) -> DecisionNode:
        """Branch para momentum fuerte en mercados volátiles"""
        
        return DecisionNode(
            name="strong_momentum_direction",
            condition=lambda f: f.momentum_direction == "alcista",
            true_action=DecisionNode(
                name="bullish_momentum_confirmation",
                condition=lambda f: f.zone_pressure != "high" and f.wickdowns_count_3h > f.wickups_count_3h,
                true_action=self._create_buy_signal("STRONG_BULLISH_MOMENTUM", 85, "strong_momentum"),
                false_action=self._create_buy_signal("BULLISH_MOMENTUM_WEAK", 70, "momentum_with_resistance"),
                description="Confirmación momentum alcista fuerte"
            ),
            false_action=DecisionNode(
                name="bearish_momentum_confirmation", 
                condition=lambda f: f.zone_pressure != "low" and f.wickups_count_3h > f.wickdowns_count_3h,
                true_action=self._create_sell_signal("STRONG_BEARISH_MOMENTUM", 85, "strong_momentum"),
                false_action=self._create_sell_signal("BEARISH_MOMENTUM_WEAK", 70, "momentum_with_support"),
                description="Confirmación momentum bajista fuerte"
            ),
            description="Análisis dirección del momentum fuerte"
        )
    
    def _build_oversold_volatile_branch(self) -> DecisionNode:
        """Branch para condiciones oversold en mercado volátil"""
        
        return DecisionNode(
            name="oversold_volatile_confirmation",
            condition=lambda f: (f.zone_low_time_pct > 15 and 
                                f.wickdowns_count_3h >= 3 and 
                                f.strongest_wick_pct > 0.1),
            true_action=DecisionNode(
                name="oversold_momentum_check",
                condition=lambda f: f.momentum_1d > -2,  # No está cayendo fuertemente
                true_action=self._create_buy_signal("OVERSOLD_BOUNCE_VOLATILE", 80, "oversold_bounce"),
                false_action=self._create_hold_signal("OVERSOLD_FALLING_KNIFE", 30, "catching_falling_knife"),
                description="Verificar que no sea falling knife"
            ),
            false_action=self._create_hold_signal("OVERSOLD_WEAK_SIGNALS", 45, "oversold_but_weak"),
            description="Confirmación condiciones oversold en volatilidad"
        )
    
    def _build_overbought_volatile_branch(self) -> DecisionNode:
        """Branch para condiciones overbought en mercado volátil"""
        
        return DecisionNode(
            name="overbought_volatile_confirmation",
            condition=lambda f: (f.zone_high_time_pct > 15 and 
                                f.wickups_count_3h >= 3 and 
                                f.strongest_wick_pct > 0.1),
            true_action=DecisionNode(
                name="overbought_momentum_check",
                condition=lambda f: f.momentum_1d < 2,  # No está subiendo fuertemente
                true_action=self._create_sell_signal("OVERBOUGHT_REVERSAL_VOLATILE", 80, "overbought_reversal"),
                false_action=self._create_hold_signal("OVERBOUGHT_STRONG_TREND", 30, "trend_too_strong"),
                description="Verificar que momentum no sea demasiado fuerte"
            ),
            false_action=self._create_hold_signal("OVERBOUGHT_WEAK_SIGNALS", 45, "overbought_but_weak"),
            description="Confirmación condiciones overbought en volatilidad"
        )
    
    def _build_volatile_middle_branch(self) -> DecisionNode:
        """Branch para zona media en mercados volátiles"""
        
        return DecisionNode(
            name="volatile_middle_breakout",
            condition=lambda f: f.wick_activity_ratio > 0.15,  # Alta actividad de wicks
            true_action=DecisionNode(
                name="breakout_direction_volatile",
                condition=lambda f: f.momentum_1d > 1,
                true_action=self._create_buy_signal("VOLATILE_BREAKOUT_UP", 65, "volatile_breakout"),
                false_action=DecisionNode(
                    name="breakdown_check_volatile",
                    condition=lambda f: f.momentum_1d < -1,
                    true_action=self._create_sell_signal("VOLATILE_BREAKDOWN", 65, "volatile_breakdown"),
                    false_action=self._create_hold_signal("VOLATILE_CONSOLIDATION", 35, "consolidation"),
                    description="Verificar breakdown en volatilidad"
                ),
                description="Determinar dirección breakout volátil"
            ),
            false_action=self._create_hold_signal("VOLATILE_LOW_ACTIVITY", 25, "low_activity"),
            description="Análisis breakout en zona media volátil"
        )
    
    def _build_bullish_structure_branch(self) -> DecisionNode:
        """Branch para estructura alcista (extremos crecientes) - MEJORADO"""
        
        return DecisionNode(
            name="bullish_structure_strength",
            condition=lambda f: f.maximos_strength > 60 and f.minimos_strength > 60,
            true_action=DecisionNode(
                name="bullish_momentum_alignment",
                condition=lambda f: f.momentum_3d > 1 and f.trend_momentum_alignment > 0.5,
                true_action=DecisionNode(
                    name="bullish_momentum_strength_check",
                    # 🔧 NUEVO: Verificar si momentum es muy fuerte
                    condition=lambda f: f.momentum_3d > 7,
                    true_action=DecisionNode(
                        name="bullish_super_strong_momentum",
                        # Con momentum >7%, permitir entrada hasta en 90% del rango
                        condition=lambda f: f.price_position_pct < 90,
                        true_action=self._create_buy_signal("BULLISH_SUPER_MOMENTUM", 90, "super_momentum"),
                        false_action=self._create_buy_signal("BULLISH_MOMENTUM_TOP", 75, "momentum_at_top"),
                        description="Momentum súper fuerte - override posición"
                    ),
                    false_action=DecisionNode(
                        name="bullish_strong_momentum",
                        # Con momentum 5-7%, permitir hasta 85%
                        condition=lambda f: f.momentum_3d > 5,
                        true_action=DecisionNode(
                            name="bullish_strong_position_check",
                            condition=lambda f: f.price_position_pct < 85,
                            true_action=self._create_buy_signal("BULLISH_STRONG_MOMENTUM", 85, "strong_momentum"),
                            false_action=self._create_buy_signal("BULLISH_MOMENTUM_HIGH", 70, "momentum_high_position"),
                            description="Momentum fuerte con ajuste de posición"
                        ),
                        false_action=DecisionNode(
                            name="bullish_normal_momentum",
                            # Con momentum 1-5%, criterio normal hasta 75%
                            condition=lambda f: f.price_position_pct < 75,
                            true_action=self._create_buy_signal("BULLISH_STRUCTURE_STRONG", 80, "bullish_structure"),
                            false_action=self._create_hold_signal("BULLISH_HIGH_POSITION", 60, "high_position"),
                            description="Momentum normal - criterio estándar"
                        ),
                        description="Momentum 1-5% - análisis estándar"
                    ),
                    description="Clasificar fuerza del momentum alcista"
                ),
                false_action=self._create_buy_signal("BULLISH_STRUCTURE_WEAK_MOMENTUM", 65, "weak_momentum"),
                description="Verificar alineación momentum-tendencia"
            ),
            false_action=DecisionNode(
                name="bullish_weak_structure",
                condition=lambda f: f.price_position_pct < 50,
                true_action=self._create_buy_signal("BULLISH_EARLY_STRUCTURE", 70, "early_structure"),
                false_action=self._create_hold_signal("BULLISH_STRUCTURE_WAIT", 50, "wait_for_clarity"),
                description="Estructura alcista débil"
            ),
            description="Evaluar fuerza de estructura alcista"
        )
    
    def _build_bearish_structure_branch(self) -> DecisionNode:
        """Branch para estructura bajista (extremos decrecientes) - MEJORADO"""
        
        return DecisionNode(
            name="bearish_structure_strength",
            condition=lambda f: f.maximos_strength > 60 and f.minimos_strength > 60,
            true_action=DecisionNode(
                name="bearish_momentum_alignment",
                condition=lambda f: f.momentum_3d < -1 and f.trend_momentum_alignment > 0.5,
                true_action=DecisionNode(
                    name="bearish_momentum_strength_check",
                    # 🔧 NUEVO: Verificar si momentum bajista es muy fuerte
                    condition=lambda f: f.momentum_3d < -7,
                    true_action=DecisionNode(
                        name="bearish_super_strong_momentum",
                        # Con momentum <-7%, permitir venta hasta en 10% del rango
                        condition=lambda f: f.price_position_pct > 10,
                        true_action=self._create_sell_signal("BEARISH_SUPER_MOMENTUM", 90, "super_momentum"),
                        false_action=self._create_sell_signal("BEARISH_MOMENTUM_BOTTOM", 75, "momentum_at_bottom"),
                        description="Momentum bajista súper fuerte"
                    ),
                    false_action=DecisionNode(
                        name="bearish_strong_momentum",
                        # Con momentum -5 a -7%, permitir hasta 15%
                        condition=lambda f: f.momentum_3d < -5,
                        true_action=DecisionNode(
                            name="bearish_strong_position_check",
                            condition=lambda f: f.price_position_pct > 15,
                            true_action=self._create_sell_signal("BEARISH_STRONG_MOMENTUM", 85, "strong_momentum"),
                            false_action=self._create_sell_signal("BEARISH_MOMENTUM_LOW", 70, "momentum_low_position"),
                            description="Momentum bajista fuerte"
                        ),
                        false_action=DecisionNode(
                            name="bearish_normal_momentum",
                            # Con momentum -1 a -5%, criterio normal hasta 25%
                            condition=lambda f: f.price_position_pct > 25,
                            true_action=self._create_sell_signal("BEARISH_STRUCTURE_STRONG", 80, "bearish_structure"),
                            false_action=self._create_hold_signal("BEARISH_LOW_POSITION", 60, "low_position"),
                            description="Momentum bajista normal"
                        ),
                        description="Momentum bajista 1-5%"
                    ),
                    description="Clasificar fuerza del momentum bajista"
                ),
                false_action=self._create_sell_signal("BEARISH_STRUCTURE_WEAK_MOMENTUM", 65, "weak_momentum"),
                description="Verificar alineación momentum-tendencia bajista"
            ),
            false_action=DecisionNode(
                name="bearish_weak_structure",
                condition=lambda f: f.price_position_pct > 50,
                true_action=self._create_sell_signal("BEARISH_EARLY_STRUCTURE", 70, "early_structure"),
                false_action=self._create_hold_signal("BEARISH_STRUCTURE_WAIT", 50, "wait_for_clarity"),
                description="Estructura bajista débil"
            ),
            description="Evaluar fuerza de estructura bajista"
        )
    
    def _build_divergent_structure_branch(self) -> DecisionNode:
        """Branch para estructuras divergentes (máximos y mínimos en direcciones opuestas)"""
        
        return DecisionNode(
            name="divergence_type_analysis",
            condition=lambda f: f.maximos_trend == "crecientes" and f.minimos_trend == "decrecientes",
            true_action=DecisionNode(
                name="bearish_divergence_confirmation",
                condition=lambda f: f.price_position_pct > 60 and f.momentum_3d < 1,
                true_action=self._create_sell_signal("BEARISH_DIVERGENCE", 75, "bearish_divergence"),
                false_action=self._create_hold_signal("BEARISH_DIVERGENCE_WEAK", 40, "weak_divergence"),
                description="Confirmación divergencia bajista"
            ),
            false_action=DecisionNode(
                name="bullish_divergence_confirmation",
                condition=lambda f: f.price_position_pct < 40 and f.momentum_3d > -1,
                true_action=self._create_buy_signal("BULLISH_DIVERGENCE", 75, "bullish_divergence"),
                false_action=self._create_hold_signal("BULLISH_DIVERGENCE_WEAK", 40, "weak_divergence"),
                description="Confirmación divergencia alcista"
            ),
            description="Análisis tipo de divergencia"
        )
    
    def _build_stable_structure_branch(self) -> DecisionNode:
        """Branch para estructuras estables (extremos estables)"""
        
        return DecisionNode(
            name="stable_structure_momentum",
            condition=lambda f: abs(f.momentum_3d) < 1.5,  # Momentum bajo
            true_action=DecisionNode(
                name="stable_zone_analysis",
                condition=lambda f: f.zone_pressure == "balanced",
                true_action=self._create_hold_signal("STABLE_BALANCED", 20, "stable_market"),
                false_action=DecisionNode(
                    name="stable_zone_imbalance",
                    condition=lambda f: f.zone_pressure == "low",
                    true_action=self._create_buy_signal("STABLE_OVERSOLD", 60, "stable_oversold"),
                    false_action=self._create_sell_signal("STABLE_OVERBOUGHT", 60, "stable_overbought"),
                    description="Análisis desbalance en mercado estable"
                ),
                description="Análisis zonas en mercado estable"
            ),
            false_action=DecisionNode(
                name="stable_momentum_direction",
                condition=lambda f: f.momentum_3d > 0,
                true_action=self._create_buy_signal("STABLE_BULLISH_MOMENTUM", 65, "stable_bullish"),
                false_action=self._create_sell_signal("STABLE_BEARISH_MOMENTUM", 65, "stable_bearish"),
                description="Dirección momentum en mercado estable"
            ),
            description="Análisis momentum en estructura estable"
        )
    
    def _create_buy_signal(self, reason: str, confidence: float, category: str) -> Callable[[MarketFeatures], TradingSignal]:
        """Crear función que genera señal de compra - MEJORADO"""
        
        def signal_generator(features: MarketFeatures) -> TradingSignal:
            # Determinar tipo de señal según confianza
            if confidence >= 85:
                signal_type = SignalType.STRONG_BUY
            elif confidence >= 70:
                signal_type = SignalType.BUY  
            else:
                signal_type = SignalType.WEAK_BUY
            
            # Calcular targets y stop loss ADAPTATIVOS
            current_price = features.current_price
            momentum_3d = features.momentum_3d
            
            # 🔧 TARGETS ADAPTATIVOS basados en momentum
            if category == "super_momentum":
                expected_move = min(6.0, momentum_3d * 0.75)  # Cap en 6%
                target = current_price * (1 + expected_move/100)
                stop = current_price * 0.975  # 2.5% stop
            elif category == "strong_momentum":
                expected_move = min(4.0, momentum_3d * 0.6)   # Cap en 4%
                target = current_price * (1 + expected_move/100)
                stop = current_price * 0.98   # 2% stop
            elif category == "bullish_structure":
                expected_move = min(3.0, max(2.0, momentum_3d * 0.4))  # Entre 2-3%
                target = current_price * (1 + expected_move/100)
                stop = current_price * 0.985  # 1.5% stop
            elif category == "oversold_bounce":
                expected_move = 2.0
                target = current_price * 1.02
                stop = current_price * 0.99
            else:  # Default
                expected_move = 1.5
                target = current_price * 1.015
                stop = current_price * 0.995
            
            risk_reward = expected_move / ((current_price - stop) / current_price * 100)
            
            return TradingSignal(
                signal_type=signal_type,
                confidence=confidence,
                reasoning=[f"BUY: {reason} (momentum 3d: +{momentum_3d:.1f}%, categoría: {category})"],
                features_used=features,
                price_target=target,
                stop_loss=stop,
                timeframe="short",
                expected_move_pct=expected_move,
                risk_reward_ratio=risk_reward
            )
        
        return signal_generator
    
    def _create_sell_signal(self, reason: str, confidence: float, category: str) -> Callable[[MarketFeatures], TradingSignal]:
        """Crear función que genera señal de venta - MEJORADO"""
        
        def signal_generator(features: MarketFeatures) -> TradingSignal:
            # Determinar tipo de señal según confianza
            if confidence >= 85:
                signal_type = SignalType.STRONG_SELL
            elif confidence >= 70:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.WEAK_SELL
            
            # Calcular targets y stop loss ADAPTATIVOS
            current_price = features.current_price
            momentum_3d = abs(features.momentum_3d)  # Valor absoluto para momentum bajista
            
            # 🔧 TARGETS ADAPTATIVOS basados en momentum
            if category == "super_momentum":
                expected_move = -min(6.0, momentum_3d * 0.75)
                target = current_price * (1 + expected_move/100)
                stop = current_price * 1.025  # 2.5% stop
            elif category == "strong_momentum":
                expected_move = -min(4.0, momentum_3d * 0.6)
                target = current_price * (1 + expected_move/100)
                stop = current_price * 1.02   # 2% stop
            elif category == "bearish_structure":
                expected_move = -min(3.0, max(2.0, momentum_3d * 0.4))
                target = current_price * (1 + expected_move/100)
                stop = current_price * 1.015  # 1.5% stop
            elif category == "overbought_reversal":
                expected_move = -2.0
                target = current_price * 0.98
                stop = current_price * 1.01
            else:  # Default
                expected_move = -1.5
                target = current_price * 0.985
                stop = current_price * 1.005
            
            risk_reward = abs(expected_move) / ((stop - current_price) / current_price * 100)
            
            return TradingSignal(
                signal_type=signal_type,
                confidence=confidence,
                reasoning=[f"SELL: {reason} (momentum 3d: {features.momentum_3d:.1f}%, categoría: {category})"],
                features_used=features,
                price_target=target,
                stop_loss=stop,
                timeframe="short",
                expected_move_pct=expected_move,
                risk_reward_ratio=risk_reward
            )
        
        return signal_generator
    
    def _create_hold_signal(self, reason: str, confidence: float, category: str) -> Callable[[MarketFeatures], TradingSignal]:
        """Crear función que genera señal de hold"""
        
        def signal_generator(features: MarketFeatures) -> TradingSignal:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=confidence,
                reasoning=[f"HOLD: {reason} (categoría: {category})"],
                features_used=features,
                timeframe="short"
            )
        
        return signal_generator
    
    def evaluate(self, features: MarketFeatures) -> TradingSignal:
        """Evaluar features através del árbol de decisiones"""
        
        print(f"🌳 Evaluando features través del árbol de decisiones...")
        print(f"📊 Datos clave: Régimen={features.regime}, Momentum 3d={features.momentum_3d:.1f}%, Posición={features.price_position_pct:.1f}%")
        
        reasoning_path = []
        current_node = self.tree
        nodes_visited = []
        
        # Navegar através del árbol
        while isinstance(current_node, DecisionNode):
            nodes_visited.append(current_node.name)
            reasoning_path.append(f"📊 {current_node.description}")
            
            try:
                condition_result = current_node.condition(features)
                
                if condition_result:
                    reasoning_path.append(f"✅ {current_node.name}: Condición cumplida")
                    current_node = current_node.true_action
                else:
                    reasoning_path.append(f"❌ {current_node.name}: Condición no cumplida")
                    current_node = current_node.false_action
                    
            except Exception as e:
                reasoning_path.append(f"⚠️ Error evaluando {current_node.name}: {str(e)}")
                # Fallback a hold
                return TradingSignal(
                    signal_type=SignalType.HOLD,
                    confidence=25,
                    reasoning=reasoning_path + [f"Error en evaluación: {str(e)}"],
                    features_used=features
                )
        
        # current_node ahora debería ser una función que retorna TradingSignal
        if callable(current_node):
            try:
                signal = current_node(features)
                signal.reasoning = reasoning_path + signal.reasoning
                
                # Actualizar estadísticas
                self._update_stats(signal)
                
                # Guardar en historial
                self.signal_history.append(signal)
                
                print(f"✅ Señal generada: {signal.signal_type.value} (confianza: {signal.confidence}%)")
                if signal.expected_move_pct:
                    print(f"🎯 Movimiento esperado: {signal.expected_move_pct:.1f}%")
                
                return signal
                
            except Exception as e:
                reasoning_path.append(f"⚠️ Error generando señal: {str(e)}")
                
        # Fallback en caso de error
        return TradingSignal(
            signal_type=SignalType.HOLD,
            confidence=25,
            reasoning=reasoning_path + ["Fallback por error en árbol"],
            features_used=features
        )
    
    def _update_stats(self, signal: TradingSignal):
        """Actualizar estadísticas de decisiones"""
        self.decision_stats["total_evaluations"] += 1
        
        signal_type = signal.signal_type.value
        if signal_type not in self.decision_stats["signal_distribution"]:
            self.decision_stats["signal_distribution"][signal_type] = 0
        self.decision_stats["signal_distribution"][signal_type] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de performance"""
        return {
            "decision_stats": self.decision_stats,
            "total_signals": len(self.signal_history),
            "recent_signals": [s.to_dict() for s in self.signal_history[-10:]],  # Últimas 10
            "signal_distribution": self.decision_stats["signal_distribution"]
        }
    
    def save_signal_history(self, filepath: str = "signal_history.json"):
        """Guardar historial de señales"""
        history_data = [signal.to_dict() for signal in self.signal_history]
        
        with open(filepath, 'w') as f:
            json.dump({
                "signals": history_data,
                "stats": self.decision_stats,
                "total_signals": len(history_data)
            }, f, indent=2)
        
        print(f"💾 Historial de señales guardado en {filepath}")