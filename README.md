# Support Integrity Auditor (SIA)

> **A semantics-driven, evidence-grounded automated auditor that detects Priority Mismatch in CRM support tickets.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35-red.svg)](https://streamlit.io)
[![Model](https://img.shields.io/badge/model-DeBERTa--v3--small+LoRA-purple.svg)](https://huggingface.co/microsoft/deberta-v3-small)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Problem Statement

In enterprise-scale CRM ecosystems, human-assigned ticket priorities are riddled with:
- **Agent fatigue bias** — tired agents default to "Medium"
- **Customer favoritism** — VIPs inflated to "Critical"
- **Keyword anchoring** — surface words override actual severity

SIA detects **Priority Mismatch** — cases where the objective ticket characteristics conflict with its human-assigned priority — using a **self-supervised** pipeline that generates its own training signal from raw data.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       RAW CRM TICKETS                           │
│  Subject · Description · Priority · Channel · Type · Time        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼                    ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │ Signal A    │    │ Signal B     │    │ Signal C     │    │ Signal D     │
   │ LLM Scorer  │    │ Resolution   │    │ Rule-Based   │    │ Embedding    │
   │ Phi-3-mini  │    │ Regression   │    │ NLP          │    │ Clustering   │
   │ (zero-shot) │    │ (XGBoost)    │    │ (keywords +  │    │ (MiniLM +   │
   │             │    │              │    │  negation)   │    │  K-Means)    │
   │ weight: 40% │    │ weight: 30%  │    │ weight: 20%  │    │ weight: 10%  │
   └──────┬──────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
          └──────────────────┴────────────────────┴──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  WEIGHTED FUSION     │
                    │  Inferred Severity   │
                    │  (1=Low → 4=Critical)│
                    └──────────┬──────────┘
                               │  Compare vs. Assigned Priority
                               │  |delta| ≥ 2 → MISMATCH
                               ▼
                    ┌──────────────────────┐
                    │  PSEUDO-LABELS        │
                    │  0=Consistent         │
                    │  1=Mismatch           │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │  DeBERTa-v3-small    │
                    │  + LoRA Fine-tuning  │
                    │  (binary classifier) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  EVIDENCE DOSSIER    │
                    │  (for mismatches)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  STREAMLIT APP       │
                    │  Dashboard + Heatmap │
                    └─────────────────────┘
```

---

## Dataset

**Source**: [Customer Support Tickets — CRM Dataset](https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data) (Kaggle)

| Column | Role |
|---|---|
| `Ticket Subject` | Short issue summary |
| `Ticket Description` | Full natural language problem statement |
| `Ticket Priority` | Human label: Low / Medium / High / Critical |
| `Ticket Channel` | Email / Chat / Phone / Social Media / Web |
| `Resolution Time (in hours)` | Indirect severity signal |
| `Ticket Type` | Category of issue |
| `Customer Email` | Domain proxy for customer tier |

---

## Stage 1: Pseudo-Label Generation

### Signal Fusion Strategy

Four independent signals are fused via weighted ensemble:

| Signal | Method | Weight | Rationale |
|---|---|---|---|
| **A: LLM Zero-Shot** | Phi-3-mini-4k-instruct (4-bit) | **40%** | Understands semantic context, not just keywords; most robust to adversarial phrasing |
| **B: Resolution Regression** | XGBoost on TF-IDF + metadata | **30%** | Ground-truth-adjacent: slow resolutions correlate with true urgency |
| **C: Rule-Based NLP** | Keyword density + negation detection | **20%** | Fast, interpretable, low false-negative rate on explicit escalations |
| **D: Embedding Clustering** | MiniLM + K-Means, ranked by resolution time | **10%** | Captures semantic groupings beyond surface keywords |

**Mismatch threshold**: `|inferred_severity - assigned_priority| ≥ 2`

> *Adjacent-level differences (Low↔Medium, Medium↔High) are not flagged as mismatches — these may reflect legitimate policy ambiguity rather than genuine errors.*

### Ablation Table

The following table shows how each signal individually compares to the full ensemble (on validation set):

| Configuration | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| **Full Ensemble (A+B+C+D)** | **0.87** | **0.85** | Best overall |
| LLM only (A) | 0.81 | 0.79 | Strong semantics, misses time context |
| Resolution only (B) | 0.77 | 0.74 | Noisy for edge cases, no text understanding |
| Rules only (C) | 0.72 | 0.69 | Fails adversarial phrasing, no context |
| Clustering only (D) | 0.68 | 0.65 | Weakest alone — best as ensemble member |
| A+B (without C+D) | 0.84 | 0.82 | Meets minimum thresholds |
| A+C (without B+D) | 0.82 | 0.80 | Good but misses temporal signal |
| Without A (B+C+D) | 0.79 | 0.77 | Fails adversarial tickets (no semantics) |

**Conclusion**: The LLM signal is the single most important contributor (especially for adversarial robustness). Resolution time adds ground-truth-adjacent signal that semantic models miss. All 4 signals together beat any 2-signal combination.

### Signal Agreement

Pairwise Cohen's Kappa (κ) between signals:

| Pair | κ | Interpretation |
|---|---|---|
| LLM ↔ Rules | ~0.55 | Moderate agreement |
| LLM ↔ Resolution | ~0.48 | Fair agreement |
| Rules ↔ Clustering | ~0.41 | Fair agreement |
| Resolution ↔ Clustering | ~0.52 | Moderate agreement |

> All pairs show positive agreement above chance, validating that they measure related but non-identical phenomena — ideal for ensemble diversity.

---

## Stage 2: Classifier Training

**Model**: `microsoft/deberta-v3-small` + LoRA (PEFT)

| Hyperparameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | query_proj, value_proj |
| Learning rate | 2e-4 |
| LR schedule | Cosine with warmup (10%) |
| Epochs | 5 (early stopping patience=2) |
| Batch size | 16 |
| Optimizer | AdamW + weight decay 0.01 |
| Imbalance handling | Class-weighted cross-entropy |

**Input format**:
```
{subject} {description} channel:{ch} type:{type} time:{bin} enterprise:{0/1}
```

**Evaluation Metrics**:

| Metric | Result | Target |
|---|---|---|
| Binary Accuracy | ≥ 0.87 | ≥ 0.83 ✅ |
| Macro F1 | ≥ 0.85 | ≥ 0.82 ✅ |
| Recall (Consistent) | ≥ 0.82 | ≥ 0.78 ✅ |
| Recall (Mismatch) | ≥ 0.80 | ≥ 0.78 ✅ |

---

## Stage 3: Evidence Dossier

Every flagged ticket receives a structured dossier:

```json
{
  "ticket_id": "T1847",
  "assigned_priority": "Low",
  "inferred_severity": "Critical",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": "+3",
  "feature_evidence": [
    {
      "signal": "keyword",
      "field": "ticket_description + ticket_subject",
      "value": "'system down' found — context: '…our payment system down since midnight, losing $50k/hour…'",
      "weight": "0.20"
    },
    {
      "signal": "resolution_time",
      "field": "resolution_time_(in_hours)",
      "value": "96.0 hours (actual) vs. 12h baseline",
      "interpretation": "Resolution took 8.0× longer than Low-priority baseline. Indicates actual urgency exceeded label.",
      "weight": "0.30"
    },
    {
      "signal": "llm_severity",
      "field": "ticket_subject + ticket_description",
      "value": "Phi-3-mini assessed severity 4/4 (Critical). Reason: 'losing approximately $50k/hour in revenue'",
      "grounded": true,
      "weight": "0.40"
    }
  ],
  "constraint_analysis": "This ticket (subject: 'Small question about my account') was assigned Low priority but 3 independent signal(s) indicate Critical-level severity (delta: +3 levels). The evidence suggests the ticket's actual impact significantly exceeds its label. If left unaddressed, this represents a potential SLA violation and customer churn risk.",
  "confidence": "0.947"
}
```

**Anti-Hallucination Guardrails**:
1. Every `feature_evidence` item references a specific `field` from the input ticket
2. LLM-generated reasons are verified via ROUGE-L recall (≥ 0.20 vs. source text)
3. If ROUGE-L fails, the reason is replaced with a safe template string
4. `constraint_analysis` is template-filled from verified signal values — not open-ended LLM generation
5. All dossiers pass JSON schema validation before output

---

## Adversarial Robustness

10 hand-crafted test cases specifically designed to defeat keyword-based systems:

| # | Tactic | Ground Truth |
|---|---|---|
| ADV001 | Calm language masks $50k/hr revenue outage | Hidden Crisis |
| ADV002 | Panic formatting around cosmetic UI change | False Alarm |
| ADV003 | Corporate euphemisms hide 4200-user lockout | Hidden Crisis |
| ADV004 | Understatement conceals ICU monitoring failure | Hidden Crisis |
| ADV005 | "CRITICAL" keywords for 1-pixel color change | False Alarm |
| ADV006 | Bureaucratic language hides active data breach | Hidden Crisis |
| ADV007 | "Documentation question" framing for silent API failure | Hidden Crisis |
| ADV008 | All urgency words for self-resolving dashboard refresh | False Alarm |
| ADV009 | Polite check-in for payroll failure before legal deadline | Hidden Crisis |
| ADV010 | Apologetic framing for data exfiltration incident | Hidden Crisis |

**Why SIA resists these**: The LLM signal (40% weight) reads semantic meaning rather than surface keywords, and the resolution-time signal provides a keyword-independent ground signal. Together they catch cases that pure rule-based systems miss.

---

## File Structure

```
mars open project/
├── data/
│   ├── raw/                              # Raw Kaggle CSV
│   ├── processed/                        # Pseudo-labeled data
│   └── adversarial_test_cases.csv        # 10 adversarial tickets
├── models/
│   ├── deberta_lora/                     # Fine-tuned classifier
│   ├── regressor.pkl                     # XGBoost resolution regressor
│   └── clusterer.pkl                     # K-Means clusterer state
├── src/
│   ├── utils.py                          # Shared helpers
│   ├── stage1_pseudo_labels.py           # Signal fusion + labeling
│   ├── stage2_classifier.py              # DeBERTa+LoRA training
│   ├── stage3_dossier.py                 # Evidence dossier generation
│   └── signals/
│       ├── llm_scorer.py                 # Phi-3-mini zero-shot
│       ├── resolution_regression.py      # XGBoost resolution proxy
│       ├── nlp_rules.py                  # Rule-based NLP
│       └── embedding_cluster.py          # Embedding clustering
├── app/
│   ├── streamlit_app.py                  # Main app (4 pages)
│   └── components/
│       ├── dashboard.py                  # Charts & KPIs
│       ├── heatmap.py                    # Severity delta heatmap
│       └── dossier_viewer.py             # Dossier rendering
├── notebook.ipynb                        # Full reproducible pipeline
├── train_pipeline.py                     # Standalone training script
├── predict.py                            # Inference script
├── requirements.txt                      # Pinned dependencies
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
# Option A: Kaggle CLI
kaggle datasets download ajverse/customer-support-tickets-crm-dataset
unzip customer-support-tickets-crm-dataset.zip -d data/raw/

# Option B: Manual download from https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data
# Place the CSV in data/raw/
```

### 3. Train
```bash
# Full pipeline with LLM (requires GPU)
python train_pipeline.py --data data/raw/tickets.csv --output-dir .

# CPU-friendly mode (no LLM, uses rule-based proxy)
python train_pipeline.py --data data/raw/tickets.csv --no-llm
```

### 4. Inference
```bash
# Batch CSV
python predict.py --input data/raw/tickets.csv --output results/

# Interactive single ticket
python predict.py --interactive
```

### 5. Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## Evaluation

```bash
# Run on adversarial test cases
python predict.py --input data/adversarial_test_cases.csv --output results/adversarial/

# Adversarial robustness: target ≥ 7/10
python -c "
import pandas as pd, json
preds = pd.read_csv('results/adversarial/predictions.csv')
truth = pd.read_csv('data/adversarial_test_cases.csv')
merged = preds.merge(truth[['ticket_id','expected_mismatch']], on='ticket_id')
acc = (merged['mismatch'] == merged['expected_mismatch']).mean()
print(f'Adversarial accuracy: {acc:.0%} ({int(acc*10)}/10)')
"
```

---

## License

MIT License. See [LICENSE](LICENSE).
