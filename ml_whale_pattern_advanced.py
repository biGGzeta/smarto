import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WhalePatternML:
    def __init__(self):
        # Inicialización de modelos
        self.pattern_classifier = RandomForestClassifier(n_estimators=120, max_depth=10, class_weight='balanced', random_state=42)
        self.breakout_classifier = GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, random_state=42)
        self.fakeout_classifier = GradientBoostingClassifier(n_estimators=60, learning_rate=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def feature_matrix_and_labels(self, df):
        """
        Toma un DataFrame con columnas de features y etiquetas y devuelve X, y_pattern, y_breakout, y_fakeout
        """
        feature_cols = [
            # Whale/orderbook/volume/price features
            "whale_bids", "whale_asks", "orderbook_imbalance", "depth_ratio",
            "spread", "total_bid_volume", "total_ask_volume", "volume",
            "taker_buy_volume", "taker_sell_volume", "buy_sell_ratio",
            "price", "momentum", "volatility"
        ]
        X = df[feature_cols].values
        y_pattern = df['pattern_label'].values   # e.g. "accumulation", "distribution", "lateral", "none"
        y_breakout = df['breakout_label'].values # 0/1
        y_fakeout = df['fakeout_label'].values   # 0/1
        return X, y_pattern, y_breakout, y_fakeout

    def fit(self, df):
        X, y_pattern, y_breakout, y_fakeout = self.feature_matrix_and_labels(df)
        X_scaled = self.scaler.fit_transform(X)
        self.pattern_classifier.fit(X_scaled, y_pattern)
        self.breakout_classifier.fit(X_scaled, y_breakout)
        self.fakeout_classifier.fit(X_scaled, y_fakeout)
        self.is_trained = True

    def predict(self, features_dict):
        """
        features_dict: dict de {feature: value}
        """
        feature_cols = [
            "whale_bids", "whale_asks", "orderbook_imbalance", "depth_ratio",
            "spread", "total_bid_volume", "total_ask_volume", "volume",
            "taker_buy_volume", "taker_sell_volume", "buy_sell_ratio",
            "price", "momentum", "volatility"
        ]
        X = np.array([[features_dict.get(col, 0) for col in feature_cols]])
        X_scaled = self.scaler.transform(X)
        ml_pattern = self.pattern_classifier.predict(X_scaled)[0]
        breakout_prob = self.breakout_classifier.predict_proba(X_scaled)[0][1]
        fakeout_prob = self.fakeout_classifier.predict_proba(X_scaled)[0][1]
        # Setup score puede ser heurístico: whale activity + imbalance + breakout_prob
        setup_score = int(100 * (features_dict.get('whale_activity_score', 0) + abs(features_dict.get('orderbook_imbalance', 0))) * breakout_prob)
        # Probabilidad de evento relevante: combinación ponderada
        event_probability = float(0.7 * breakout_prob + 0.3 * setup_score/100)
        return {
            "ml_whale_pattern": ml_pattern,
            "ml_breakout_risk": float(breakout_prob),
            "ml_fakeout_risk": float(fakeout_prob),
            "ml_setup_score": int(setup_score),
            "ml_event_probability": round(event_probability, 3)
        }