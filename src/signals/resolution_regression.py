"""
src/signals/resolution_regression.py
Signal B — Resolution-time regression proxy.
Trains an XGBoost regressor to predict resolution_hours from ticket features,
then maps predicted hours → severity quartile (1–4).
"""
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb

logger = logging.getLogger(__name__)

# Severity thresholds (hours) — calibrated on dataset statistics
# These are updated after fitting via quartile analysis
DEFAULT_THRESHOLDS = [24.0, 72.0, 168.0]  # Low|Med, Med|High, High|Critical


class ResolutionRegressor:
    """
    Trains an XGBoost regressor on available resolution time data,
    then predicts severity buckets for all tickets.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                     sublinear_tf=True, min_df=2)
        self.scaler = StandardScaler()
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.channel_enc = LabelEncoder()
        self.type_enc = LabelEncoder()
        self.thresholds: list = DEFAULT_THRESHOLDS
        self._fitted = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    # ── Feature builder ─────────────────────────────────────────
    def _build_features(self, df: pd.DataFrame, fit: bool = False):
        text = (df["ticket_subject"].fillna("") + " " +
                df["ticket_description"].fillna(""))

        if fit:
            tfidf_feats = self.tfidf.fit_transform(text)
            ch_enc = self.channel_enc.fit_transform(
                df["ticket_channel"].fillna("Unknown"))
            ty_enc = self.type_enc.fit_transform(
                df["ticket_type"].fillna("Unknown"))
        else:
            tfidf_feats = self.tfidf.transform(text)
            ch_enc = self.channel_enc.transform(
                df["ticket_channel"].fillna("Unknown")
                  .apply(lambda x: x if x in self.channel_enc.classes_ else "Unknown"))
            ty_enc = self.type_enc.transform(
                df["ticket_type"].fillna("Unknown")
                  .apply(lambda x: x if x in self.type_enc.classes_ else "Unknown"))

        meta = np.column_stack([
            ch_enc,
            ty_enc,
            df["is_enterprise"].fillna(0).values,
        ])
        meta_sparse = csr_matrix(meta)
        return hstack([tfidf_feats, meta_sparse])

    # ── Fit ─────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "ResolutionRegressor":
        """
        Train on rows with known resolution_hours.
        """
        train_df = df.dropna(subset=["resolution_hours"]).copy()
        if len(train_df) < 50:
            logger.warning("Too few samples with resolution_hours. Using heuristic thresholds.")
            self._fitted = False
            return self

        logger.info(f"Training resolution regressor on {len(train_df)} samples…")
        X = self._build_features(train_df, fit=True)
        y = np.log1p(train_df["resolution_hours"].values)  # log-transform for skew

        self.xgb_model.fit(X, y)

        # Calibrate thresholds from quartiles of actual resolution times
        q25, q50, q75 = np.percentile(
            train_df["resolution_hours"].values, [25, 50, 75])
        self.thresholds = [q25, q50, q75]
        logger.info(f"Resolution time quartile thresholds: "
                    f"Q25={q25:.1f}h, Q50={q50:.1f}h, Q75={q75:.1f}h")

        self._fitted = True
        return self

    # ── Predict ─────────────────────────────────────────────────
    def predict_severity(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns array of severity integers 1–4 for each ticket.
        """
        if not self._fitted:
            # Fallback: use actual resolution_hours if available
            return df["resolution_hours"].apply(
                self._hours_to_severity_default).values

        X = self._build_features(df, fit=False)
        log_pred = self.xgb_model.predict(X)
        hours_pred = np.expm1(log_pred)

        return np.array([self._hours_to_severity(h) for h in hours_pred])

    def _hours_to_severity(self, hours: float) -> int:
        t = self.thresholds
        if hours <= t[0]:
            return 1
        elif hours <= t[1]:
            return 2
        elif hours <= t[2]:
            return 3
        else:
            return 4

    @staticmethod
    def _hours_to_severity_default(hours) -> int:
        if pd.isna(hours):
            return 2
        if hours <= 24:
            return 1
        elif hours <= 72:
            return 2
        elif hours <= 168:
            return 3
        else:
            return 4

    # ── Persistence ─────────────────────────────────────────────
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"ResolutionRegressor saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
        logger.info(f"ResolutionRegressor loaded from {path}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Return top XGBoost feature importances (TF-IDF terms)."""
        if not self._fitted:
            return pd.DataFrame()
        importances = self.xgb_model.feature_importances_
        tfidf_terms = self.tfidf.get_feature_names_out().tolist()
        meta_terms = ["channel", "ticket_type", "is_enterprise"]
        all_terms = tfidf_terms + meta_terms
        n = min(len(importances), len(all_terms))
        df = pd.DataFrame({
            "feature": all_terms[:n],
            "importance": importances[:n]
        }).sort_values("importance", ascending=False)
        return df.head(20)
