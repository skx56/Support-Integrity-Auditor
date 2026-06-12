"""
src/signals/nlp_rules.py
Signal C — Rule-based NLP severity scoring.
Combines escalation keyword density, negation detection,
intensifier detection, and channel-based modifiers.
"""
import re
import logging
from typing import Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

# ── Keyword Lexicons ────────────────────────────────────────────
CRITICAL_KEYWORDS = [
    # Outage / downtime
    "system down", "complete outage", "total outage", "service outage",
    "production down", "not working", "completely broken", "unavailable",
    # Data / security
    "data loss", "data breach", "security breach", "unauthorized access",
    "credentials exposed", "account hacked", "phishing", "ransomware",
    # Business impact
    "revenue impact", "losing customers", "sla breach", "sla violation",
    "critical failure", "mission critical", "all users affected",
    "cannot access", "locked out", "cannot login", "unable to login",
    # Time pressure
    "emergency", "asap", "immediately", "right now", "this moment",
    "escalate to", "executive", "ceo", "cto", "vp",
]

HIGH_KEYWORDS = [
    "error", "broken", "fails", "failure", "disruption", "degraded",
    "slow", "performance issue", "intermittent", "recurring issue",
    "multiple users", "several users", "team cannot", "blocking",
    "workaround", "no workaround", "deadline", "impacted",
    "incorrect data", "wrong data", "missing data", "corrupt",
]

MEDIUM_KEYWORDS = [
    "issue", "problem", "not working correctly", "unexpected behavior",
    "sometimes", "occasionally", "some users", "one user",
    "help needed", "question", "how to", "need assistance",
]

LOW_KEYWORDS = [
    "minor", "cosmetic", "typo", "small issue", "slight",
    "enhancement", "feature request", "suggestion", "nice to have",
    "when convenient", "low priority", "no rush", "feedback",
    "improvement", "would be nice", "wondering if",
]

# Negation window: words before a term that negate it
NEGATION_WORDS = [
    "not", "no", "never", "without", "resolved", "fixed", "working",
    "works fine", "working now", "already", "don't", "doesn't", "isn't",
    "wasn't", "haven't", "has been resolved", "was resolved",
]

# Intensifiers (boost severity score)
INTENSIFIERS = [
    "very", "extremely", "highly", "absolutely", "completely",
    "totally", "severely", "badly", "terribly", "desperately",
]

# Channel multipliers (some channels carry more urgency)
CHANNEL_WEIGHTS = {
    "phone": 1.15,
    "chat": 1.05,
    "email": 1.00,
    "social media": 1.20,  # public escalation risk
    "web": 0.95,
    "portal": 0.95,
    "unknown": 1.00,
}


class RuleBasedScorer:
    """
    Computes a rule-based severity score (1–4) using keyword density,
    negation detection, intensifier boosting, and channel weighting.
    """

    def score_ticket(self, subject: str, description: str,
                     channel: str = "unknown") -> dict:
        """
        Returns:
            {"severity": int 1-4, "matched_keywords": list, "negated_keywords": list,
             "raw_score": float}
        """
        text = (subject + " " + description).lower()

        critical_hits, negated_critical = self._match_with_negation(text, CRITICAL_KEYWORDS)
        high_hits, negated_high = self._match_with_negation(text, HIGH_KEYWORDS)
        medium_hits, _ = self._match_with_negation(text, MEDIUM_KEYWORDS)
        low_hits, _ = self._match_with_negation(text, LOW_KEYWORDS)

        intensifier_boost = self._count_intensifiers(text)

        # Weighted raw score (0–10 scale)
        raw_score = (
            len(critical_hits) * 3.0 +
            len(high_hits) * 1.5 +
            len(medium_hits) * 0.5 -
            len(low_hits) * 0.8 +
            intensifier_boost * 0.5
        )

        # Channel multiplier
        ch_key = channel.lower().strip()
        multiplier = CHANNEL_WEIGHTS.get(ch_key, 1.0)
        raw_score *= multiplier

        # Map to 1–4
        severity = self._score_to_severity(raw_score, critical_hits, low_hits)

        all_matched = critical_hits + high_hits + medium_hits
        all_negated = negated_critical + negated_high

        return {
            "severity": severity,
            "matched_keywords": all_matched[:5],  # top 5 for dossier
            "negated_keywords": all_negated[:3],
            "raw_score": round(raw_score, 3),
            "intensifiers_found": intensifier_boost,
        }

    def score_batch(self, df) -> list:
        results = []
        for _, row in df.iterrows():
            result = self.score_ticket(
                subject=str(row.get("ticket_subject", "")),
                description=str(row.get("ticket_description", "")),
                channel=str(row.get("ticket_channel", "unknown")),
            )
            results.append(result)
        return results

    # ── Helpers ─────────────────────────────────────────────────
    @staticmethod
    def _match_with_negation(text: str, keywords: List[str],
                             window: int = 5) -> Tuple[List[str], List[str]]:
        """
        Returns (confirmed_hits, negated_hits).
        A hit is negated if a negation word appears within `window` words before it.
        """
        confirmed = []
        negated = []
        words = re.split(r"\W+", text)

        for kw in keywords:
            kw_words = kw.split()
            kw_len = len(kw_words)
            if kw not in text:
                continue

            # Find all occurrences
            for i, word in enumerate(words):
                if words[i:i + kw_len] == kw_words:
                    # Check negation window
                    window_start = max(0, i - window)
                    preceding = words[window_start:i]
                    is_negated = any(
                        neg in " ".join(preceding)
                        for neg in NEGATION_WORDS
                    )
                    if is_negated:
                        negated.append(kw)
                    else:
                        confirmed.append(kw)
                    break  # count each keyword once

        return confirmed, negated

    @staticmethod
    def _count_intensifiers(text: str) -> int:
        return sum(1 for intensifier in INTENSIFIERS if intensifier in text)

    @staticmethod
    def _score_to_severity(raw_score: float,
                           critical_hits: List[str],
                           low_hits: List[str]) -> int:
        """Map raw score to 1–4 severity bucket."""
        if len(critical_hits) >= 2 or raw_score >= 5.0:
            return 4
        elif len(critical_hits) == 1 or raw_score >= 2.5:
            return 3
        elif raw_score >= 0.5 or (raw_score > 0 and len(low_hits) == 0):
            return 2
        else:
            return 1

    def extract_evidence_snippet(self, subject: str, description: str,
                                  keyword: str) -> str:
        """Extract surrounding context for a matched keyword (for dossier)."""
        text = subject + " " + description
        text_lower = text.lower()
        idx = text_lower.find(keyword)
        if idx == -1:
            return subject[:100]
        start = max(0, idx - 30)
        end = min(len(text), idx + len(keyword) + 50)
        return "…" + text[start:end].strip() + "…"
