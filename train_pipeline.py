#!/usr/bin/env python3
"""
train_pipeline.py
Standalone training script for the Support Integrity Auditor (SIA).

Usage:
    python train_pipeline.py --data data/raw/tickets.csv [OPTIONS]

Options:
    --data          Path to raw CSV (required)
    --output-dir    Directory to save models and processed data [default: .]
    --no-llm        Skip LLM scoring (faster, uses rule-based proxy instead)
    --epochs        Number of training epochs [default: 5]
    --batch-size    Training batch size [default: 16]
    --no-ablation   Skip ablation analysis
    --seed          Random seed [default: 42]
"""
import argparse
import logging
import sys
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_pipeline.log"),
    ]
)
logger = logging.getLogger("train_pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="SIA Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to raw CRM CSV")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM scoring (CPU-friendly mode)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for mismatch")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data" / "processed"
    model_dir = output_dir / "models" / "deberta_lora"
    artifact_dir = output_dir / "models"

    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ── STAGE 1: Pseudo-label generation ─────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1: Pseudo-Label Generation")
    logger.info("=" * 60)

    from src.stage1_pseudo_labels import PseudoLabelPipeline
    from src.utils import load_dataset

    df = load_dataset(args.data)

    pipeline = PseudoLabelPipeline(
        use_llm=not args.no_llm,
        clusterer_cache=str(artifact_dir / "clusterer.pkl"),
        regressor_cache=str(artifact_dir / "regressor.pkl"),
        ablation_mode=not args.no_ablation,
    )
    df_labeled = pipeline.run(df)

    labeled_path = str(data_dir / "pseudo_labeled.csv")
    df_labeled.to_csv(labeled_path, index=False)
    logger.info(f"Pseudo-labeled data saved to {labeled_path}")

    # Save pipeline artifacts
    pipeline.regressor.save(str(artifact_dir / "regressor.pkl"))
    pipeline.clusterer.save(str(artifact_dir / "clusterer.pkl"))

    # Save metadata
    meta = {
        "signal_agreement": df_labeled.attrs.get("signal_agreement", {}),
        "ablation_results": df_labeled.attrs.get("ablation_results", {}),
        "mismatch_rate": float(df_labeled["mismatch"].mean()),
        "n_tickets": len(df_labeled),
        "n_mismatched": int(df_labeled["mismatch"].sum()),
    }
    with open(str(artifact_dir / "stage1_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Stage 1 complete. Mismatch rate: {meta['mismatch_rate']:.2%}")
    logger.info(f"  Hidden Crisis: {(df_labeled['mismatch_type'] == 'Hidden Crisis').sum()}")
    logger.info(f"  False Alarm:   {(df_labeled['mismatch_type'] == 'False Alarm').sum()}")

    # ── STAGE 2: Classifier training ─────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2: Classifier Training (DeBERTa-v3-small + LoRA)")
    logger.info("=" * 60)

    from src.stage2_classifier import MismatchClassifier

    classifier = MismatchClassifier(model_dir=str(model_dir))
    metrics = classifier.train(
        df_labeled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2 RESULTS:")
    logger.info(f"  Binary Accuracy:     {metrics.get('eval_accuracy', metrics.get('accuracy', '?')):.4f}")
    logger.info(f"  Macro F1:            {metrics.get('eval_macro_f1', metrics.get('macro_f1', '?')):.4f}")
    logger.info(f"  Recall (Consistent): {metrics.get('eval_recall_consistent', '?')}")
    logger.info(f"  Recall (Mismatch):   {metrics.get('eval_recall_mismatch', '?')}")
    logger.info("=" * 60)

    # Check thresholds
    acc = metrics.get("eval_accuracy", metrics.get("accuracy", 0))
    f1 = metrics.get("eval_macro_f1", metrics.get("macro_f1", 0))
    r_c = metrics.get("eval_recall_consistent", 0)
    r_m = metrics.get("eval_recall_mismatch", 0)

    passes = acc >= 0.83 and f1 >= 0.82 and r_c >= 0.78 and r_m >= 0.78
    if passes:
        logger.info("✅ All verification thresholds met!")
    else:
        logger.warning("⚠️  Some thresholds not met. Consider adjusting threshold or training longer.")
        if acc < 0.83:
            logger.warning(f"  Accuracy {acc:.4f} < 0.83 target")
        if f1 < 0.82:
            logger.warning(f"  Macro F1 {f1:.4f} < 0.82 target")
        if r_c < 0.78:
            logger.warning(f"  Recall(Consistent) {r_c} < 0.78 target")
        if r_m < 0.78:
            logger.warning(f"  Recall(Mismatch) {r_m} < 0.78 target")

    logger.info(f"\nTraining complete! Model saved to: {model_dir}")
    logger.info(f"Run inference with: python predict.py --model {model_dir} --input <CSV>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
