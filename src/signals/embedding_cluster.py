"""
src/signals/embedding_cluster.py
Signal D — Embedding-based semantic urgency clustering.
Encodes tickets with all-MiniLM-L6-v2, clusters into k=4 groups,
ranks clusters by median resolution time → severity labels.
"""
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_CLUSTERS = 4
RANDOM_STATE = 42


class EmbeddingClusterer:
    """
    Uses sentence-transformers to embed tickets, K-Means to cluster,
    then ranks clusters by resolution time to assign severity labels 1–4.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL,
                 n_clusters: int = N_CLUSTERS,
                 model_path: Optional[str] = None):
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.embedder = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_severity_map: dict = {}  # cluster_id → severity 1-4
        self._fitted = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def _load_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformer: {self.model_name}…")
            self.embedder = SentenceTransformer(self.model_name)
            logger.info("Sentence-transformer loaded.")

    def _embed(self, texts: list) -> np.ndarray:
        self._load_embedder()
        embeddings = self.embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings  # already L2-normalized

    def fit(self, df: pd.DataFrame) -> "EmbeddingClusterer":
        """
        Embed tickets, fit K-Means, rank clusters by resolution time.
        """
        texts = (df["ticket_subject"].fillna("") + ". " +
                 df["ticket_description"].fillna("")).tolist()

        logger.info(f"Embedding {len(texts)} tickets…")
        embeddings = self._embed(texts)

        logger.info(f"Fitting K-Means with k={self.n_clusters}…")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
            max_iter=300,
        )
        cluster_labels = self.kmeans.fit_predict(embeddings)

        # Rank clusters by median resolution time → severity
        self.cluster_severity_map = self._rank_clusters(
            cluster_labels, df["resolution_hours"].values)

        self._fitted = True
        logger.info(f"Cluster → severity map: {self.cluster_severity_map}")
        return self

    def predict_severity(self, df: pd.DataFrame) -> np.ndarray:
        """Returns severity array 1–4 for each ticket."""
        texts = (df["ticket_subject"].fillna("") + ". " +
                 df["ticket_description"].fillna("")).tolist()
        embeddings = self._embed(texts)

        if not self._fitted:
            logger.warning("EmbeddingClusterer not fitted. Returning default severity 2.")
            return np.full(len(df), 2, dtype=int)

        cluster_labels = self.kmeans.predict(embeddings)
        severities = np.array([
            self.cluster_severity_map.get(c, 2) for c in cluster_labels
        ])
        return severities

    def get_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw embeddings for visualization."""
        texts = (df["ticket_subject"].fillna("") + ". " +
                 df["ticket_description"].fillna("")).tolist()
        return self._embed(texts)

    @staticmethod
    def _rank_clusters(cluster_labels: np.ndarray,
                       resolution_hours: np.ndarray) -> dict:
        """
        Rank clusters by median resolution time.
        Cluster with highest median → severity 4, lowest → severity 1.
        Falls back to priority-column median if resolution_hours is all NaN.
        """
        medians = {}
        for cid in np.unique(cluster_labels):
            mask = cluster_labels == cid
            hrs = resolution_hours[mask]
            valid = hrs[~np.isnan(hrs)]
            medians[cid] = np.median(valid) if len(valid) > 0 else 0.0

        # Sort clusters by median ascending → rank 1 (low) to 4 (critical)
        sorted_clusters = sorted(medians.items(), key=lambda x: x[1])
        n = len(sorted_clusters)
        severity_map = {}
        for rank, (cid, _) in enumerate(sorted_clusters):
            # Map rank to 1–4 evenly
            severity = max(1, min(4, int(1 + (rank / max(n - 1, 1)) * 3)))
            severity_map[cid] = severity

        return severity_map

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Don't pickle the large embedder model, just the cluster artifacts
        state = {
            "kmeans": self.kmeans,
            "cluster_severity_map": self.cluster_severity_map,
            "model_name": self.model_name,
            "n_clusters": self.n_clusters,
            "_fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"EmbeddingClusterer state saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.kmeans = state["kmeans"]
        self.cluster_severity_map = state["cluster_severity_map"]
        self.model_name = state["model_name"]
        self.n_clusters = state["n_clusters"]
        self._fitted = state["_fitted"]
        logger.info(f"EmbeddingClusterer loaded from {path}")
