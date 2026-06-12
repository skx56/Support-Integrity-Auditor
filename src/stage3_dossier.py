"""
src/stage3_dossier.py
Stage 3: Evidence Dossier Generation.

For every ticket classified as mismatch=1, produces a structured
JSON dossier grounded strictly in actual ticket field values.
Anti-hallucination: every evidence item traceable to a source field.
ROUGE-L check enforces LLM reason strings trace back to source text.
"""
import json
import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from src.utils import (
    hallucination_check, validate_dossier,
    numeric_to_severity_label, PRIORITY_MAP
)
from src.signals.nlp_rules import RuleBasedScorer

logger = logging.getLogger(__name__)

# Priority numeric → expected resolution hours baseline (for interpretation)
PRIORITY_RESOLUTION_BASELINE = {
    1: 12.0,    # Low
    2: 48.0,    # Medium
    3: 120.0,   # High
    4: 4.0,     # Critical
}


class DossierGenerator:
    """
    Generates hallucination-free Evidence Dossiers for mismatched tickets.
    """

    def __init__(self):
        self.rules_scorer = RuleBasedScorer()

    def generate(self, row: pd.Series, confidence: float) -> Dict[str, Any]:
        """
        Build a single dossier for a flagged ticket.

        Args:
            row: A DataFrame row with all ticket + signal columns.
            confidence: Model's predicted probability for Mismatch class.

        Returns:
            Validated dossier dict.
        """
        ticket_id = str(row.get("ticket_id", row.name))
        assigned_priority = str(row.get("ticket_priority", "Unknown"))
        inferred_severity = str(row.get("inferred_severity_label", "Unknown"))
        severity_delta = int(row.get("severity_delta", 0))
        mismatch_type = str(row.get("mismatch_type", "Hidden Crisis"))

        # Build evidence items (strictly field-traceable)
        feature_evidence = []

        # Evidence 1: Keyword signal
        kw_evidence = self._keyword_evidence(row)
        if kw_evidence:
            feature_evidence.append(kw_evidence)

        # Evidence 2: Resolution time signal
        rt_evidence = self._resolution_evidence(row)
        if rt_evidence:
            feature_evidence.append(rt_evidence)

        # Evidence 3: LLM signal
        llm_evidence = self._llm_evidence(row)
        if llm_evidence:
            feature_evidence.append(llm_evidence)

        # Evidence 4: Channel signal (if relevant)
        ch_evidence = self._channel_evidence(row)
        if ch_evidence:
            feature_evidence.append(ch_evidence)

        # Evidence 5: Enterprise tier
        ent_evidence = self._enterprise_evidence(row)
        if ent_evidence:
            feature_evidence.append(ent_evidence)

        # Constraint analysis (template-based, not open-ended LLM)
        constraint_analysis = self._build_constraint_analysis(
            row, mismatch_type, severity_delta, feature_evidence)

        dossier = {
            "ticket_id": ticket_id,
            "assigned_priority": assigned_priority,
            "inferred_severity": inferred_severity,
            "mismatch_type": mismatch_type,
            "severity_delta": f"{severity_delta:+d}",
            "feature_evidence": feature_evidence,
            "constraint_analysis": constraint_analysis,
            "confidence": f"{confidence:.3f}",
        }

        # Validate schema
        if not validate_dossier(dossier):
            logger.warning(f"Dossier validation failed for ticket {ticket_id}")

        return dossier

    def generate_batch(self, df: pd.DataFrame, probabilities: np.ndarray) -> List[Dict]:
        """
        Generate dossiers for all mismatched tickets.
        Only processes rows where mismatch==1.
        """
        dossiers = []
        mismatch_df = df[df["mismatch"] == 1].copy()
        mismatch_probs = probabilities[df["mismatch"] == 1]

        logger.info(f"Generating dossiers for {len(mismatch_df)} mismatched tickets…")

        for (_, row), prob in zip(mismatch_df.iterrows(), mismatch_probs):
            try:
                dossier = self.generate(row, float(prob))
                dossiers.append(dossier)
            except Exception as e:
                logger.warning(f"Error generating dossier for ticket {row.get('ticket_id', '?')}: {e}")

        logger.info(f"Generated {len(dossiers)} dossiers.")
        return dossiers

    # ── Evidence builders ────────────────────────────────────────
    def _keyword_evidence(self, row: pd.Series) -> Optional[Dict]:
        """Evidence from rule-based NLP matched keywords."""
        keywords = row.get("_rules_keywords", [])
        if not keywords:
            # Re-run rules on this ticket
            result = self.rules_scorer.score_ticket(
                subject=str(row.get("ticket_subject", "")),
                description=str(row.get("ticket_description", "")),
                channel=str(row.get("ticket_channel", "")),
            )
            keywords = result.get("matched_keywords", [])

        if not keywords:
            return None

        # Get context snippet from actual text (traceable to source)
        source_text = str(row.get("ticket_subject", "")) + " " + str(row.get("ticket_description", ""))
        top_kw = keywords[0]
        snippet = self.rules_scorer.extract_evidence_snippet(
            str(row.get("ticket_subject", "")),
            str(row.get("ticket_description", "")),
            top_kw,
        )

        return {
            "signal": "keyword",
            "field": "ticket_description + ticket_subject",
            "value": f"'{top_kw}' found in ticket — context: \"{snippet[:120]}\"",
            "all_matched": keywords[:5],
            "weight": "0.20",
        }

    def _resolution_evidence(self, row: pd.Series) -> Optional[Dict]:
        """Evidence from resolution time vs. baseline."""
        actual_hours = row.get("resolution_hours", np.nan)
        priority_num = row.get("priority_numeric", 2)
        baseline = PRIORITY_RESOLUTION_BASELINE.get(int(priority_num), 48.0)

        if pd.isna(actual_hours):
            return None

        ratio = actual_hours / baseline if baseline > 0 else 1.0
        if ratio >= 1.5:
            interpretation = (f"Resolution took {actual_hours:.1f}h — "
                              f"{ratio:.1f}× longer than the {baseline:.0f}h baseline "
                              f"for {numeric_to_severity_label(priority_num)}-priority tickets. "
                              f"Indicates actual urgency exceeded label.")
        elif ratio <= 0.4:
            interpretation = (f"Resolution took only {actual_hours:.1f}h — "
                              f"{1/ratio:.1f}× faster than the {baseline:.0f}h baseline "
                              f"for {numeric_to_severity_label(priority_num)}-priority tickets. "
                              f"Suggests ticket was less severe than labeled.")
        else:
            return None  # Resolution time consistent with priority, not strong evidence

        return {
            "signal": "resolution_time",
            "field": "resolution_time_(in_hours)",
            "value": f"{actual_hours:.1f} hours (actual) vs. {baseline:.0f}h baseline",
            "interpretation": interpretation,
            "weight": "0.30",
        }

    def _llm_evidence(self, row: pd.Series) -> Optional[Dict]:
        """Evidence from LLM zero-shot scoring."""
        llm_severity = row.get("severity_llm")
        llm_reason = str(row.get("_llm_reason", ""))
        priority_num = row.get("priority_numeric", 2)

        if pd.isna(llm_severity) or not llm_reason:
            return None

        # Hallucination check: reason must trace back to ticket text
        source = str(row.get("ticket_subject", "")) + " " + str(row.get("ticket_description", ""))
        is_grounded = hallucination_check(llm_reason, source, threshold=0.20)

        if not is_grounded:
            # Fall back to a safe, unquoted paraphrase
            llm_reason = f"LLM severity assessment: {numeric_to_severity_label(int(llm_severity))}"

        return {
            "signal": "llm_severity",
            "field": "ticket_subject + ticket_description",
            "value": (f"Phi-3-mini assessed severity {int(llm_severity)}/4 "
                      f"({numeric_to_severity_label(int(llm_severity))}). "
                      f"Reason: \"{llm_reason[:200]}\""),
            "grounded": is_grounded,
            "weight": "0.40",
        }

    def _channel_evidence(self, row: pd.Series) -> Optional[Dict]:
        """Evidence from ticket channel (social media = public escalation risk)."""
        channel = str(row.get("ticket_channel", "")).lower()
        inferred_sev = int(row.get("severity_fused", 2))
        priority_num = int(row.get("priority_numeric", 2))

        HIGH_URGENCY_CHANNELS = {"social media", "phone"}
        LOW_URGENCY_CHANNELS = {"web", "portal"}

        if channel in HIGH_URGENCY_CHANNELS and inferred_sev > priority_num:
            return {
                "signal": "channel",
                "field": "ticket_channel",
                "value": f"Channel: '{channel}' — high-urgency intake path",
                "interpretation": (f"Tickets via {channel} carry elevated urgency risk "
                                   f"(public visibility / real-time escalation)."),
                "weight": "0.10",
            }
        elif channel in LOW_URGENCY_CHANNELS and inferred_sev < priority_num:
            return {
                "signal": "channel",
                "field": "ticket_channel",
                "value": f"Channel: '{channel}' — self-service/low-urgency intake",
                "interpretation": f"Tickets via {channel} typically represent lower-urgency issues.",
                "weight": "0.10",
            }
        return None

    def _enterprise_evidence(self, row: pd.Series) -> Optional[Dict]:
        """Enterprise customer tier as evidence."""
        is_enterprise = int(row.get("is_enterprise", 0))
        inferred_sev = int(row.get("severity_fused", 2))
        priority_num = int(row.get("priority_numeric", 2))
        domain = str(row.get("customer_domain", "unknown"))

        if is_enterprise and inferred_sev > priority_num:
            return {
                "signal": "customer_tier",
                "field": "customer_email",
                "value": f"Enterprise customer (domain: {domain})",
                "interpretation": ("Enterprise SLA obligations increase the business impact "
                                   "of even seemingly minor issues."),
                "weight": "0.05",
            }
        return None

    # ── Constraint analysis ──────────────────────────────────────
    @staticmethod
    def _build_constraint_analysis(
        row: pd.Series,
        mismatch_type: str,
        delta: int,
        evidence: List[Dict],
    ) -> str:
        """
        Template-based constraint analysis. Strictly grounded — no open-ended LLM.
        """
        assigned = str(row.get("ticket_priority", "Unknown"))
        inferred = str(row.get("inferred_severity_label", "Unknown"))
        subject = str(row.get("ticket_subject", ""))[:80]
        signal_count = len(evidence)

        if mismatch_type == "Hidden Crisis":
            return (
                f"This ticket (subject: \"{subject}\") was assigned {assigned} priority "
                f"but {signal_count} independent signal(s) indicate {inferred}-level severity "
                f"(delta: {delta:+d} levels). The evidence suggests the ticket's actual impact "
                f"significantly exceeds its label. If left unaddressed, this represents "
                f"a potential SLA violation and customer churn risk."
            )
        else:  # False Alarm
            return (
                f"This ticket (subject: \"{subject}\") was assigned {assigned} priority "
                f"but {signal_count} independent signal(s) indicate only {inferred}-level severity "
                f"(delta: {delta:+d} levels). The high label may be causing unnecessary resource "
                f"allocation and queue disruption for genuinely critical tickets."
            )
