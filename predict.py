#!/usr/bin/env python3
"""
predict.py
Inference script for the Support Integrity Auditor (SIA).

Accepts a CSV (single ticket or batch) and outputs:
  - predictions.csv  — original data + mismatch prediction + confidence
  - dossiers.jsonl   — Evidence Dossiers for all mismatched tickets

Usage:
    # Batch CSV
    python predict.py --input data/test.csv --model models/deberta_lora --output results/

    # Single ticket (interactive)
    python predict.py --interactive --model models/deberta_lora

    # With LLM re-scoring for dossier evidence
    python predict.py --input data/test.csv --model models/deberta_lora --use-llm
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("predict")


def load_artifacts(model_dir: str, artifacts_dir: str, use_llm: bool):
    """Load all trained artifacts."""
    from src.stage2_classifier import MismatchClassifier
    from src.signals.resolution_regression import ResolutionRegressor
    from src.signals.embedding_cluster import EmbeddingClusterer
    from src.signals.nlp_rules import RuleBasedScorer
    from src.signals.llm_scorer import LLMScorer

    artifacts = Path(artifacts_dir)

    classifier = MismatchClassifier(model_dir=model_dir)
    classifier.load()

    regressor = ResolutionRegressor(model_path=str(artifacts / "regressor.pkl"))
    clusterer = EmbeddingClusterer(model_path=str(artifacts / "clusterer.pkl"))
    rules = RuleBasedScorer()
    llm = LLMScorer() if use_llm else None

    return classifier, regressor, clusterer, rules, llm


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as training."""
    from src.utils import load_dataset
    import tempfile, os

    # Save to temp and reload through preprocessing pipeline
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    try:
        processed = load_dataset(tmp_path)
    finally:
        os.unlink(tmp_path)

    return processed


def run_signals(df, regressor, clusterer, rules, llm):
    """Re-run signals to populate severity columns needed for dossier."""
    from src.signals.nlp_rules import RuleBasedScorer
    from src.utils import numeric_to_severity_label
    from src.stage1_pseudo_labels import FUSION_WEIGHTS, MISMATCH_DELTA_THRESHOLD

    # Signal B
    df["severity_resolution"] = regressor.predict_severity(df)

    # Signal C
    rules_results = rules.score_batch(df)
    df["severity_rules"] = [r["severity"] for r in rules_results]
    df["_rules_keywords"] = [r["matched_keywords"] for r in rules_results]
    df["_rules_negated"] = [r["negated_keywords"] for r in rules_results]
    df["_rules_raw_score"] = [r["raw_score"] for r in rules_results]

    # Signal D
    df["severity_cluster"] = clusterer.predict_severity(df)

    # Signal A (LLM or proxy)
    if llm is not None:
        llm_results = llm.score_batch(df)
        df["severity_llm"] = [r["severity"] for r in llm_results]
        df["_llm_reason"] = [r.get("reason", "") for r in llm_results]
    else:
        df["severity_llm"] = df["severity_rules"]
        df["_llm_reason"] = ""

    # Fuse
    import numpy as np
    raw = (
        FUSION_WEIGHTS["llm"] * df["severity_llm"] +
        FUSION_WEIGHTS["resolution"] * df["severity_resolution"] +
        FUSION_WEIGHTS["rules"] * df["severity_rules"] +
        FUSION_WEIGHTS["cluster"] * df["severity_cluster"]
    )
    df["severity_fused"] = raw.round().clip(1, 4).astype(int)
    df["inferred_severity_label"] = df["severity_fused"].apply(numeric_to_severity_label)
    df["severity_delta"] = df["severity_fused"] - df["priority_numeric"]

    def classify_mismatch(row):
        d = row["severity_delta"]
        if abs(d) < MISMATCH_DELTA_THRESHOLD:
            return "Consistent"
        return "Hidden Crisis" if d > 0 else "False Alarm"

    df["mismatch_type"] = df.apply(classify_mismatch, axis=1)

    return df


def predict_batch(input_path: str, model_dir: str, artifacts_dir: str,
                  output_dir: str, use_llm: bool = False,
                  threshold: float = 0.5):
    """Main batch inference function."""
    logger.info(f"Loading input from {input_path}…")
    raw_df = pd.read_csv(input_path)
    df = preprocess_input(raw_df)

    logger.info("Loading trained artifacts…")
    classifier, regressor, clusterer, rules, llm = load_artifacts(
        model_dir, artifacts_dir, use_llm)

    logger.info("Running signals…")
    df = run_signals(df, regressor, clusterer, rules, llm)

    logger.info("Running classifier…")
    preds, probs = classifier.predict(df, threshold=threshold)
    df["mismatch"] = preds
    df["mismatch_confidence"] = probs

    # Generate dossiers
    logger.info("Generating Evidence Dossiers…")
    from src.stage3_dossier import DossierGenerator
    generator = DossierGenerator()
    dossiers = generator.generate_batch(df, probs)

    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pred_cols = [
        "ticket_id", "ticket_subject", "ticket_priority",
        "inferred_severity_label", "mismatch", "mismatch_confidence",
        "mismatch_type", "severity_delta",
    ]
    available = [c for c in pred_cols if c in df.columns]
    df[available].to_csv(str(output_path / "predictions.csv"), index=False)

    from src.utils import save_jsonl
    save_jsonl(dossiers, str(output_path / "dossiers.jsonl"))

    # Summary
    summary = {
        "n_total": len(df),
        "n_mismatch": int(df["mismatch"].sum()),
        "mismatch_rate": float(df["mismatch"].mean()),
        "hidden_crisis": int((df["mismatch_type"] == "Hidden Crisis").sum()),
        "false_alarm": int((df["mismatch_type"] == "False Alarm").sum()),
    }
    with open(str(output_path / "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  Total tickets: {summary['n_total']}")
    logger.info(f"  Mismatches:    {summary['n_mismatch']} ({summary['mismatch_rate']:.1%})")
    logger.info(f"  Hidden Crisis: {summary['hidden_crisis']}")
    logger.info(f"  False Alarm:   {summary['false_alarm']}")
    logger.info(f"  Dossiers:      {output_dir}/dossiers.jsonl")
    logger.info(f"{'='*50}")

    return df, dossiers


def predict_single_interactive(model_dir: str, artifacts_dir: str, use_llm: bool):
    """Interactive single-ticket prediction."""
    logger.info("Interactive single-ticket mode")
    print("\n" + "="*60)
    print("  Support Integrity Auditor — Interactive Mode")
    print("="*60)

    ticket = {
        "ticket_id": input("Ticket ID [default: T001]: ").strip() or "T001",
        "ticket_subject": input("Subject: ").strip(),
        "ticket_description": input("Description: ").strip(),
        "ticket_priority": input("Assigned Priority (Low/Medium/High/Critical): ").strip(),
        "ticket_channel": input("Channel (email/chat/phone/social media/web): ").strip() or "email",
        "ticket_type": input("Ticket Type: ").strip() or "General",
        "customer_email": input("Customer Email: ").strip() or "user@example.com",
        "resolution_time_(in_hours)": input("Resolution Time (hours, leave blank if unknown): ").strip() or None,
    }

    df = pd.DataFrame([ticket])
    df = preprocess_input(df)

    classifier, regressor, clusterer, rules, llm = load_artifacts(
        model_dir, artifacts_dir, use_llm)

    df = run_signals(df, regressor, clusterer, rules, llm)
    preds, probs = classifier.predict(df)
    df["mismatch"] = preds
    df["mismatch_confidence"] = probs

    row = df.iloc[0]
    print(f"\n{'='*60}")
    print(f"  RESULT: {'⚠️  PRIORITY MISMATCH DETECTED' if preds[0] == 1 else '✅ CONSISTENT'}")
    print(f"  Assigned Priority:  {row.get('ticket_priority', '?')}")
    print(f"  Inferred Severity:  {row.get('inferred_severity_label', '?')}")
    print(f"  Mismatch Type:      {row.get('mismatch_type', '?')}")
    print(f"  Confidence:         {probs[0]:.1%}")
    print(f"{'='*60}")

    if preds[0] == 1:
        from src.stage3_dossier import DossierGenerator
        generator = DossierGenerator()
        dossier = generator.generate(row, float(probs[0]))
        print("\nEvidence Dossier:")
        print(json.dumps(dossier, indent=2))


def main():
    parser = argparse.ArgumentParser(description="SIA Inference Engine")
    parser.add_argument("--input", help="Input CSV path (batch mode)")
    parser.add_argument("--model", default="models/deberta_lora",
                        help="Path to trained model directory")
    parser.add_argument("--artifacts", default="models",
                        help="Path to artifacts directory (regressor, clusterer)")
    parser.add_argument("--output", default="results",
                        help="Output directory for predictions and dossiers")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive single-ticket mode")
    parser.add_argument("--use-llm", action="store_true",
                        help="Re-run LLM scoring during inference (slower)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Mismatch classification threshold")
    args = parser.parse_args()

    if args.interactive:
        predict_single_interactive(args.model, args.artifacts, args.use_llm)
    elif args.input:
        predict_batch(
            input_path=args.input,
            model_dir=args.model,
            artifacts_dir=args.artifacts,
            output_dir=args.output,
            use_llm=args.use_llm,
            threshold=args.threshold,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
