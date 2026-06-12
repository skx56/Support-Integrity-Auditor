"""
src/stage2_classifier.py
Stage 2: Fine-tuned binary mismatch classifier.

Model: microsoft/deberta-v3-small + LoRA (PEFT)
Input: Concatenated text fields + structured metadata tokens
Output: Binary label (0=Consistent, 1=Mismatch)
Imbalance: Weighted cross-entropy loss + stratified split
"""
import logging
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# ── Model config ────────────────────────────────────────────────
BASE_MODEL = "microsoft/deberta-v3-small"
MAX_SEQ_LEN = 512
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"],
    bias="none",
)

LABEL2ID = {"Consistent": 0, "Mismatch": 1}
ID2LABEL = {0: "Consistent", 1: "Mismatch"}

# Resolution time bins for metadata tokens
RESOLUTION_BINS = [(0, 24, "fast"), (24, 72, "moderate"),
                   (72, 168, "slow"), (168, 1e9, "very_slow")]


def _resolution_bin(hours: float) -> str:
    if pd.isna(hours):
        return "unknown"
    for lo, hi, label in RESOLUTION_BINS:
        if lo <= hours < hi:
            return label
    return "very_slow"


# ── Dataset ─────────────────────────────────────────────────────
class TicketDataset(Dataset):
    """
    Tokenises tickets for DeBERTa.
    Input format:
      [CLS] {subject} [SEP] {description} [SEP]
      channel:{channel} type:{type} time:{res_bin} enterprise:{0/1} [SEP]
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = df["mismatch"].values.astype(int)
        self.texts = [self._build_text(row) for _, row in df.iterrows()]

    @staticmethod
    def _build_text(row) -> str:
        subject = str(row.get("ticket_subject", ""))[:200]
        description = str(row.get("ticket_description", ""))[:600]
        channel = str(row.get("ticket_channel", "unknown"))
        ttype = str(row.get("ticket_type", "unknown"))
        res_bin = _resolution_bin(row.get("resolution_hours", np.nan))
        enterprise = int(row.get("is_enterprise", 0))
        meta = f"channel:{channel} type:{ttype} time:{res_bin} enterprise:{enterprise}"
        return f"{subject} {description} {meta}"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        encoding["labels"] = int(self.labels[idx])
        return encoding


# ── Metrics ─────────────────────────────────────────────────────
def compute_metrics(eval_pred) -> Dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    recall_consistent = recall_score(labels, preds, pos_label=0, zero_division=0)
    recall_mismatch = recall_score(labels, preds, pos_label=1, zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "recall_consistent": round(recall_consistent, 4),
        "recall_mismatch": round(recall_mismatch, 4),
    }


# ── Trainer with weighted loss ──────────────────────────────────
class WeightedTrainer(Trainer):
    """Custom Trainer that applies class-weighted cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Main classifier class ────────────────────────────────────────
class MismatchClassifier:
    """
    DeBERTa-v3-small + LoRA binary classifier for priority mismatch detection.
    """

    def __init__(self, model_dir: str = "models/deberta_lora",
                 base_model: str = BASE_MODEL):
        self.model_dir = model_dir
        self.base_model = base_model
        self.tokenizer = None
        self.model = None
        self._trained = False

    def train(
        self,
        df: pd.DataFrame,
        val_size: float = 0.15,
        test_size: float = 0.15,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        seed: int = 42,
    ) -> Dict:
        """
        Full training loop. Returns evaluation metrics on test split.
        """
        logger.info(f"Training MismatchClassifier on {len(df)} samples…")

        # Split
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df["mismatch"], random_state=seed)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size / (1 - test_size),
            stratify=train_val_df["mismatch"], random_state=seed)

        logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        logger.info(f"Train mismatch rate: {train_df['mismatch'].mean():.2%}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # Datasets
        train_dataset = TicketDataset(train_df, self.tokenizer)
        val_dataset = TicketDataset(val_df, self.tokenizer)
        test_dataset = TicketDataset(test_df, self.tokenizer)

        # Class weights
        class_weights = self._compute_class_weights(train_df["mismatch"].values)
        logger.info(f"Class weights: {class_weights}")

        # Load base model + apply LoRA
        base = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self.model = get_peft_model(base, LORA_CONFIG)
        self.model.print_trainable_parameters()

        # Training args
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            seed=seed,
            dataloader_num_workers=0,
            report_to="none",
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        logger.info("Starting training…")
        trainer.train()

        # Evaluate on test set
        logger.info("Evaluating on test set…")
        test_results = trainer.evaluate(test_dataset)

        # Full classification report
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        report = classification_report(
            test_dataset.labels, preds, target_names=["Consistent", "Mismatch"])
        logger.info(f"\nTest Classification Report:\n{report}")

        # Save model + tokenizer
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

        # Save metrics
        metrics = {
            **test_results,
            "classification_report": report,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "mismatch_rate_train": float(train_df["mismatch"].mean()),
        }
        with open(f"{self.model_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        self._trained = True
        logger.info(f"Training complete. Test Macro F1: {test_results.get('eval_macro_f1', '?')}")
        return metrics

    def load(self):
        """Load trained model from disk."""
        logger.info(f"Loading MismatchClassifier from {self.model_dir}…")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        base = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
        self.model = PeftModel.from_pretrained(base, self.model_dir)
        self.model.eval()
        self._trained = True
        logger.info("Model loaded.")

    def predict(self, df: pd.DataFrame, batch_size: int = 32,
                threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predictions, probabilities) for the Mismatch class.
        """
        if not self._trained:
            raise RuntimeError("Model not trained or loaded.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()

        dataset = TicketDataset(df, self.tokenizer)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                outputs = self.model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy())

        probs = np.array(all_probs)
        preds = (probs >= threshold).astype(int)
        return preds, probs

    @staticmethod
    def _compute_class_weights(labels: np.ndarray) -> torch.Tensor:
        classes = np.unique(labels)
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weight_tensor = torch.zeros(2)
        for cls, w in zip(classes, weights):
            weight_tensor[cls] = w
        return weight_tensor
