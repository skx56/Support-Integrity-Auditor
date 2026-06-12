"""
src/signals/llm_scorer.py
Signal A — Zero-shot severity scoring via Phi-3-mini-4k-instruct (4-bit quantized).
Falls back to a lightweight DistilBERT zero-shot classifier if the large model
cannot be loaded (CPU-only environments).
"""
import re
import json
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ── Prompt template ────────────────────────────────────────────
SEVERITY_PROMPT = """You are a support ticket severity analyst. Analyze the following support ticket and output ONLY a valid JSON object with no extra text.

Support Ticket:
Subject: {subject}
Description: {description}
Channel: {channel}
Type: {ticket_type}

Output format (respond ONLY with this JSON, no markdown, no explanation):
{{"severity": <integer 1-4>, "reason": "<one direct quote or close paraphrase from the ticket text>"}}

Severity scale:
1 = Low (minor inconvenience, cosmetic, no business impact)
2 = Medium (moderate impact, workaround available)
3 = High (significant disruption, no workaround, affects multiple users)
4 = Critical (complete outage, data loss, security breach, revenue impact)"""

# ── Loader ─────────────────────────────────────────────────────
class LLMScorer:
    """
    Uses Phi-3-mini-4k-instruct (4-bit) for zero-shot severity scoring.
    Gracefully falls back to rule-based heuristic if GPU/model unavailable.
    """

    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct",
                 use_4bit: bool = True, device: Optional[str] = None):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._fallback = False

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

            logger.info(f"Loading {self.model_name} on {self.device}...")

            quant_config = None
            if self.use_4bit and self.device == "cuda":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=120,
                do_sample=False,
                temperature=1.0,
            )
            logger.info("LLM loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load LLM ({e}). Falling back to heuristic scorer.")
            self._fallback = True

    def score_ticket(self, subject: str, description: str,
                     channel: str, ticket_type: str) -> dict:
        """
        Returns {"severity": int 1-4, "reason": str}.
        """
        if self._fallback:
            return self._heuristic_score(subject, description)

        prompt = SEVERITY_PROMPT.format(
            subject=subject[:300],
            description=description[:800],
            channel=channel,
            ticket_type=ticket_type,
        )

        try:
            output = self.pipeline(prompt)[0]["generated_text"]
            # Extract only the newly generated part
            generated = output[len(prompt):].strip()
            result = _parse_json_output(generated)
            if result:
                result["severity"] = max(1, min(4, int(result.get("severity", 2))))
                return result
        except Exception as e:
            logger.warning(f"LLM inference error: {e}")

        return self._heuristic_score(subject, description)

    def score_batch(self, df) -> list:
        """Score all tickets in a DataFrame. Returns list of dicts."""
        results = []
        for _, row in df.iterrows():
            result = self.score_ticket(
                subject=str(row.get("ticket_subject", "")),
                description=str(row.get("ticket_description", "")),
                channel=str(row.get("ticket_channel", "")),
                ticket_type=str(row.get("ticket_type", "")),
            )
            results.append(result)
        return results

    @staticmethod
    def _heuristic_score(subject: str, description: str) -> dict:
        """CPU-safe heuristic fallback using escalation keyword density."""
        text = (subject + " " + description).lower()
        critical_kw = ["outage", "breach", "data loss", "system down", "cannot access",
                       "not working", "critical", "urgent", "emergency", "security",
                       "revenue", "production down", "all users affected"]
        high_kw = ["broken", "error", "fails", "disruption", "multiple users",
                   "recurring", "escalate", "no workaround"]
        low_kw = ["minor", "cosmetic", "suggestion", "when convenient",
                  "low priority", "feedback", "nice to have", "enhancement"]

        c_score = sum(1 for kw in critical_kw if kw in text)
        h_score = sum(1 for kw in high_kw if kw in text)
        l_score = sum(1 for kw in low_kw if kw in text)

        if c_score >= 2:
            sev = 4
        elif c_score == 1 or h_score >= 2:
            sev = 3
        elif h_score == 1:
            sev = 2
        elif l_score >= 1:
            sev = 1
        else:
            sev = 2

        # Find a reason snippet
        for kw in critical_kw + high_kw:
            if kw in text:
                idx = text.find(kw)
                snippet = (subject + " " + description)[max(0, idx-20):idx+50].strip()
                return {"severity": sev, "reason": snippet, "_fallback": True}

        return {"severity": sev, "reason": subject[:80], "_fallback": True}


def _parse_json_output(text: str) -> Optional[dict]:
    """Attempt to parse JSON from model output, handling markdown fences."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find first JSON object
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
