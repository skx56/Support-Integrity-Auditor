"""
src/utils.py
Shared helpers: data loading, preprocessing, priority normalization,
schema validation, ROUGE-L hallucination check.
"""
import re
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Priority / Severity mappings
# ──────────────────────────────────────────────────────────────
PRIORITY_MAP = {"low": 1, "medium": 2, "high": 3, "critical": 4}
SEVERITY_MAP_INV = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}

RESOLUTION_BINS = {  # hours → severity bucket
    "Low": (0, 24),
    "Medium": (24, 72),
    "High": (72, 168),
    "Critical": (168, float("inf")),
}

# ──────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ──────────────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    """Load the CRM CSV and apply standard cleaning.
    Handles both the original schema and the actual Kaggle dataset schema:
      Ticket_ID, Ticket_Subject, Ticket_Description, Priority_Level,
      Ticket_Channel, Issue_Category, Resolution_Time_Hours, Customer_Email
    """
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} tickets from {path}")

    # Normalise column names to lowercase_underscore
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Map actual Kaggle column names → standard internal names ──
    COLUMN_ALIASES = {
        # ticket id
        "ticket_id":              "ticket_id",
        # subject
        "ticket_subject":         "ticket_subject",
        "subject":                "ticket_subject",
        # description
        "ticket_description":     "ticket_description",
        "description":            "ticket_description",
        # priority  (actual dataset uses priority_level)
        "priority_level":         "ticket_priority",
        "ticket_priority":        "ticket_priority",
        "priority":               "ticket_priority",
        # channel
        "ticket_channel":         "ticket_channel",
        "channel":                "ticket_channel",
        # type / category  (actual dataset uses issue_category)
        "issue_category":         "ticket_type",
        "ticket_type":            "ticket_type",
        "category":               "ticket_type",
        # resolution time  (actual dataset uses resolution_time_hours)
        "resolution_time_hours":  "resolution_time_(in_hours)",
        "resolution_time_(in_hours)": "resolution_time_(in_hours)",
        "resolution_time":        "resolution_time_(in_hours)",
        # email
        "customer_email":         "customer_email",
        "email":                  "customer_email",
    }

    rename_map = {}
    for col in df.columns:
        target = COLUMN_ALIASES.get(col)
        if target and target not in df.columns:
            rename_map[col] = target

    df = df.rename(columns=rename_map)
    logger.info(f"Columns after rename: {df.columns.tolist()}")

    # Ensure ticket_id exists
    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"T{i:05d}" for i in range(len(df))]

    # Fill missing text fields
    df["ticket_description"] = df["ticket_description"].fillna("") if "ticket_description" in df.columns else ""
    df["ticket_subject"]     = df["ticket_subject"].fillna("")     if "ticket_subject"     in df.columns else ""
    df["ticket_type"]        = df["ticket_type"].fillna("Unknown") if "ticket_type"        in df.columns else "Unknown"
    df["ticket_channel"]     = df["ticket_channel"].fillna("Unknown") if "ticket_channel"  in df.columns else "Unknown"
    df["customer_email"]     = df["customer_email"].fillna("user@unknown.com") if "customer_email" in df.columns else "user@unknown.com"

    # Clean text
    df["ticket_description"] = df["ticket_description"].apply(_clean_text)
    df["ticket_subject"]     = df["ticket_subject"].apply(_clean_text)

    # Priority numeric  (Low=1, Medium=2, High=3, Critical=4)
    if "ticket_priority" not in df.columns:
        raise ValueError("Could not find a priority column. Got: " + str(df.columns.tolist()))

    df["priority_numeric"] = (
        df["ticket_priority"].astype(str).str.strip().str.lower().map(PRIORITY_MAP)
    )
    missing_priority = df["priority_numeric"].isna().sum()
    if missing_priority > 0:
        logger.warning(f"{missing_priority} rows have unrecognised priority — dropping them.")
        logger.warning(f"Unique priority values found: {df['ticket_priority'].unique()[:10]}")
    df = df.dropna(subset=["priority_numeric"])
    df["priority_numeric"] = df["priority_numeric"].astype(int)

    # Resolution time
    rt_col = "resolution_time_(in_hours)"
    if rt_col in df.columns:
        df["resolution_hours"] = pd.to_numeric(df[rt_col], errors="coerce")
    else:
        df["resolution_hours"] = np.nan

    # Customer domain tier
    df["customer_domain"] = df["customer_email"].apply(_extract_domain)
    df["is_enterprise"]   = df["customer_domain"].apply(_is_enterprise_domain).astype(int)

    # Combined text field
    df["full_text"] = df["ticket_subject"] + ". " + df["ticket_description"]

    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    logger.info(f"Priority distribution:\n{df['ticket_priority'].value_counts()}")
    return df.reset_index(drop=True)


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"https?://\S+", " ", text)     # strip URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_domain(email: str) -> str:
    if not isinstance(email, str):
        return "unknown"
    m = re.search(r"@([\w.]+)", email)
    return m.group(1).lower() if m else "unknown"


ENTERPRISE_TLDS = {"com", "org", "io", "co", "net"}
FREE_EMAIL_PROVIDERS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"}

def _is_enterprise_domain(domain: str) -> int:
    return 0 if domain in FREE_EMAIL_PROVIDERS else 1


# ──────────────────────────────────────────────────────────────
# Severity helpers
# ──────────────────────────────────────────────────────────────
def numeric_to_severity_label(n: int) -> str:
    return SEVERITY_MAP_INV.get(int(np.clip(round(n), 1, 4)), "Medium")


def severity_label_to_numeric(label: str) -> int:
    return PRIORITY_MAP.get(label.strip().lower(), 2)


# ──────────────────────────────────────────────────────────────
# Hallucination guard
# ──────────────────────────────────────────────────────────────
_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def hallucination_check(generated_text: str, source_text: str, threshold: float = 0.25) -> bool:
    """
    Returns True if generated_text is sufficiently grounded in source_text.
    Uses ROUGE-L recall to detect if key spans in generated text appear in source.
    """
    if not generated_text or not source_text:
        return False
    score = _scorer.score(source_text, generated_text)["rougeL"].recall
    return score >= threshold


# ──────────────────────────────────────────────────────────────
# Dossier JSON schema validation
# ──────────────────────────────────────────────────────────────
DOSSIER_SCHEMA = {
    "type": "object",
    "required": [
        "ticket_id", "assigned_priority", "inferred_severity",
        "mismatch_type", "severity_delta", "feature_evidence",
        "constraint_analysis", "confidence"
    ],
    "properties": {
        "ticket_id": {"type": "string"},
        "assigned_priority": {"type": "string"},
        "inferred_severity": {"type": "string"},
        "mismatch_type": {"type": "string", "enum": ["Hidden Crisis", "False Alarm"]},
        "severity_delta": {"type": "string"},
        "feature_evidence": {"type": "array", "minItems": 1},
        "constraint_analysis": {"type": "string"},
        "confidence": {"type": "string"},
    }
}

def validate_dossier(dossier: dict) -> bool:
    try:
        import jsonschema
        jsonschema.validate(dossier, DOSSIER_SCHEMA)
        return True
    except Exception as e:
        logger.warning(f"Dossier validation failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────────────────
def save_jsonl(records: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved {len(records)} records to {path}")


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]
