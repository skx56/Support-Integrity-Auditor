"""
src/stage1_pseudo_labels.py
Stage 1: Self-supervised pseudo-label generation.

Pipeline:
  1. Run 4 independent signals → per-ticket severity scores 1–4
  2. Fuse via weighted ensemble
  3. Compare fused score to assigned priority → binary mismatch label
  4. Report pairwise signal agreement (ablation)

Output columns added to DataFrame:
  - severity_llm, severity_resolution, severity_rules, severity_cluster
  - severity_fused (1–4 integer)
  - inferred_severity_label (Low/Medium/High/Critical)
  - mismatch (0/1)
  - mismatch_type (Hidden Crisis / False Alarm / Consistent)
  - severity_delta (signed int)
"""
import logging
import json
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.utils import (
    load_dataset, numeric_to_severity_label, PRIORITY_MAP, SEVERITY_MAP_INV
)
from src.signals.llm_scorer import LLMScorer
from src.signals.resolution_regression import ResolutionRegressor
from src.signals.nlp_rules import RuleBasedScorer
from src.signals.embedding_cluster import EmbeddingClusterer

logger = logging.getLogger(__name__)

# ── Fusion weights (must sum to 1.0) ────────────────────────────
FUSION_WEIGHTS = {
    "llm":        0.40,
    "resolution": 0.30,
    "rules":      0.20,
    "cluster":    0.10,
}

# Mismatch threshold: |fused - assigned| >= this → mismatch
MISMATCH_DELTA_THRESHOLD = 2


class PseudoLabelPipeline:
    """
    Orchestrates all 4 signals and produces pseudo-labeled data.
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_model: str = "microsoft/Phi-3-mini-4k-instruct",
        clusterer_cache: Optional[str] = None,
        regressor_cache: Optional[str] = None,
        ablation_mode: bool = False,  # If True, also run per-signal-only experiments
    ):
        self.use_llm = use_llm
        self.ablation_mode = ablation_mode

        logger.info("Initialising pseudo-label pipeline…")

        # Signal B: Resolution regressor
        self.regressor = ResolutionRegressor(model_path=regressor_cache)

        # Signal C: Rule-based NLP
        self.rules_scorer = RuleBasedScorer()

        # Signal D: Embedding clusterer
        self.clusterer = EmbeddingClusterer(model_path=clusterer_cache)

        # Signal A: LLM (optional, can be skipped for speed)
        self.llm_scorer: Optional[LLMScorer] = None
        if use_llm:
            self.llm_scorer = LLMScorer(model_name=llm_model)

    # ── Main entry point ─────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full pseudo-label pipeline on a preprocessed DataFrame.
        Returns df with added severity + mismatch columns.
        """
        df = df.copy()
        n = len(df)
        logger.info(f"Running pseudo-label pipeline on {n} tickets…")

        # ── Signal B: Resolution Regression ──────────────────────
        logger.info("Signal B: Training resolution regressor…")
        self.regressor.fit(df)
        df["severity_resolution"] = self.regressor.predict_severity(df)

        # ── Signal C: Rule-Based NLP ─────────────────────────────
        logger.info("Signal C: Rule-based NLP scoring…")
        rules_results = self.rules_scorer.score_batch(df)
        df["severity_rules"] = [r["severity"] for r in rules_results]
        df["_rules_keywords"] = [r["matched_keywords"] for r in rules_results]
        df["_rules_negated"] = [r["negated_keywords"] for r in rules_results]
        df["_rules_raw_score"] = [r["raw_score"] for r in rules_results]

        # ── Signal D: Embedding Clustering ───────────────────────
        logger.info("Signal D: Embedding + clustering…")
        self.clusterer.fit(df)
        df["severity_cluster"] = self.clusterer.predict_severity(df)

        # ── Signal A: LLM Zero-Shot ──────────────────────────────
        if self.use_llm and self.llm_scorer is not None:
            logger.info("Signal A: LLM zero-shot scoring (this may take a while)…")
            llm_results = self.llm_scorer.score_batch(df)
            df["severity_llm"] = [r["severity"] for r in llm_results]
            df["_llm_reason"] = [r.get("reason", "") for r in llm_results]
        else:
            logger.info("Signal A: LLM skipped — using rules as proxy.")
            # Use rules as proxy for LLM weight
            df["severity_llm"] = df["severity_rules"]
            df["_llm_reason"] = ""

        # ── Fusion ────────────────────────────────────────────────
        logger.info("Fusing signals…")
        df["severity_fused"] = self._fuse(df)
        df["inferred_severity_label"] = df["severity_fused"].apply(numeric_to_severity_label)
        df["severity_delta"] = df["severity_fused"] - df["priority_numeric"]

        # ── Mismatch labeling ─────────────────────────────────────
        df["mismatch"] = (df["severity_delta"].abs() >= MISMATCH_DELTA_THRESHOLD).astype(int)
        df["mismatch_type"] = df.apply(self._classify_mismatch, axis=1)

        # ── Agreement metrics ─────────────────────────────────────
        agreement = self._compute_signal_agreement(df)
        logger.info(f"Signal agreement metrics:\n{json.dumps(agreement, indent=2)}")
        df.attrs["signal_agreement"] = agreement

        # ── Ablation (optional) ───────────────────────────────────
        if self.ablation_mode:
            self._run_ablation(df)

        mismatch_rate = df["mismatch"].mean()
        logger.info(f"Mismatch rate: {mismatch_rate:.2%} "
                    f"({df['mismatch'].sum()} tickets flagged out of {n})")
        logger.info(f"Mismatch type breakdown:\n{df['mismatch_type'].value_counts()}")

        return df

    # ── Fusion ───────────────────────────────────────────────────
    @staticmethod
    def _fuse(df: pd.DataFrame) -> pd.Series:
        raw = (
            FUSION_WEIGHTS["llm"]        * df["severity_llm"] +
            FUSION_WEIGHTS["resolution"] * df["severity_resolution"] +
            FUSION_WEIGHTS["rules"]      * df["severity_rules"] +
            FUSION_WEIGHTS["cluster"]    * df["severity_cluster"]
        )
        return raw.round().clip(1, 4).astype(int)

    @staticmethod
    def _classify_mismatch(row) -> str:
        delta = row["severity_delta"]
        if abs(delta) < MISMATCH_DELTA_THRESHOLD:
            return "Consistent"
        elif delta > 0:
            return "Hidden Crisis"   # inferred > assigned (underprioritised)
        else:
            return "False Alarm"     # inferred < assigned (overprioritised)

    # ── Signal agreement ─────────────────────────────────────────
    @staticmethod
    def _compute_signal_agreement(df: pd.DataFrame) -> Dict[str, float]:
        """
        Pairwise Cohen's Kappa + raw agreement between signals.
        """
        signals = ["severity_llm", "severity_resolution", "severity_rules", "severity_cluster"]
        available = [s for s in signals if s in df.columns]
        agreement = {}

        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                s1, s2 = available[i], available[j]
                kappa = cohen_kappa_score(df[s1], df[s2])
                raw_agree = (df[s1] == df[s2]).mean()
                key = f"{s1}_vs_{s2}"
                agreement[key] = {
                    "cohen_kappa": round(float(kappa), 4),
                    "raw_agreement": round(float(raw_agree), 4),
                }

        return agreement

    # ── Ablation ─────────────────────────────────────────────────
    def _run_ablation(self, df: pd.DataFrame):
        """
        For each signal, compute the mismatch rate using ONLY that signal.
        Helps justify the fusion strategy in the README.
        """
        logger.info("\n=== ABLATION: Single-signal mismatch rates ===")
        results = {}
        for signal_col, weight_key in [
            ("severity_llm", "llm"),
            ("severity_resolution", "resolution"),
            ("severity_rules", "rules"),
            ("severity_cluster", "cluster"),
        ]:
            if signal_col not in df.columns:
                continue
            delta = df[signal_col] - df["priority_numeric"]
            mismatch_rate = (delta.abs() >= MISMATCH_DELTA_THRESHOLD).mean()
            agreement_with_fused = (df[signal_col] == df["severity_fused"]).mean()
            results[weight_key] = {
                "mismatch_rate": round(float(mismatch_rate), 4),
                "agreement_with_fused": round(float(agreement_with_fused), 4),
            }
            logger.info(f"  {signal_col}: mismatch_rate={mismatch_rate:.2%}, "
                        f"agreement_with_fused={agreement_with_fused:.2%}")

        df.attrs["ablation_results"] = results
        return results


# ── Standalone runner ─────────────────────────────────────────
def generate_pseudo_labels(
    data_path: str,
    output_path: str,
    use_llm: bool = True,
    ablation: bool = True,
    regressor_cache: str = "models/regressor.pkl",
    clusterer_cache: str = "models/clusterer.pkl",
) -> pd.DataFrame:
    """
    Full pseudo-label generation pipeline.
    Saves enriched DataFrame to output_path as CSV.
    """
    from src.utils import load_dataset
    df = load_dataset(data_path)

    pipeline = PseudoLabelPipeline(
        use_llm=use_llm,
        clusterer_cache=clusterer_cache,
        regressor_cache=regressor_cache,
        ablation_mode=ablation,
    )
    df_labeled = pipeline.run(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(output_path, index=False)
    logger.info(f"Pseudo-labeled data saved to {output_path}")

    # Save agreement + ablation metrics
    meta_path = str(Path(output_path).with_suffix("")) + "_metadata.json"
    meta = {
        "fusion_weights": FUSION_WEIGHTS,
        "mismatch_delta_threshold": MISMATCH_DELTA_THRESHOLD,
        "signal_agreement": df_labeled.attrs.get("signal_agreement", {}),
        "ablation_results": df_labeled.attrs.get("ablation_results", {}),
        "mismatch_rate": float(df_labeled["mismatch"].mean()),
        "n_tickets": len(df_labeled),
        "n_mismatched": int(df_labeled["mismatch"].sum()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")

    return df_labeled
