"""Rule-based case handling for negative customer interactions.

Given a negative interaction, this module works out:
  * WHY the customer is unhappy (keywords in the transcript),
  * HOW urgent it is (fusion score, plus escalation words),
  * WHAT the agent should do next (a playbook per reason),
  * and, on the customer's next interaction, WHETHER things improved.

Like fusion.py it is deterministic and explainable: no extra model is needed.
"""

import re

from app.fusion import POSITIVE_THRESHOLD
from app.schemas import Attention, EmotionResult, FollowUp, FusionResult

# reason key -> (phrase used in the alert, regex patterns, suggested actions)
REASONS: dict[str, dict] = {
    "delay": {
        "label": "a delay or long wait",
        "patterns": [
            r"\blate\b", r"\bdelay", r"\bwait", r"\bdeliver", r"\bshipp",
            r"hasn'?t arrived", r"not arrived", r"taking too long",
            r"still (haven'?t|not|waiting)", r"where is my",
        ],
        "actions": [
            "Check the order or ticket status and explain what caused the delay",
            "Give a realistic delivery or resolution time",
            "Offer expedited handling or compensation if applicable",
        ],
    },
    "billing": {
        "label": "a billing or payment issue",
        "patterns": [
            r"\bbill", r"\bcharg", r"\brefund", r"\bpayment", r"\binvoice",
            r"\bovercharg", r"\bmoney", r"credit card", r"\bsubscription",
        ],
        "actions": [
            "Review the disputed charge on the customer's account",
            "Explain the billing clearly, line by line",
            "Offer a correction or refund if the charge is wrong",
        ],
    },
    "service": {
        "label": "poor service",
        "patterns": [
            r"\brude", r"\bunhelp", r"\bnobody", r"\bno one", r"\bignor",
            r"\bworst", r"\bterrible", r"\buseless", r"\bunprofessional",
            r"proper work", r"can'?t even", r"bad time", r"bad service",
            r"poor service", r"not helpful", r"\bdisrespect", r"\bwasted",
        ],
        "actions": [
            "Acknowledge the poor experience without being defensive",
            "Ask what went wrong and note the details",
            "Offer a goodwill gesture, such as priority support or a discount",
        ],
    },
    "product": {
        "label": "a faulty or damaged product",
        "patterns": [
            r"\bbroken", r"\bdamag", r"\bdefect", r"doesn'?t work", r"not working",
            r"\bfaulty", r"wrong item", r"\bmissing", r"stopped working", r"\bquality",
        ],
        "actions": [
            "Ask for details or a photo of the problem",
            "Offer a replacement or a refund",
            "Arrange a pickup or return",
        ],
    },
    "access": {
        "label": "an account or technical problem",
        "patterns": [
            r"\blog ?in\b", r"\bpassword", r"\blocked", r"\baccount", r"\berror",
            r"can'?t access", r"\bcrash", r"not loading", r"\bwebsite", r"\bapp\b",
        ],
        "actions": [
            "Verify the customer's identity",
            "Walk through the fix or reset their access",
            "Give a ticket number and an expected fix time",
        ],
    },
}

GENERAL = {
    "label": "their experience",
    "actions": [
        "Ask open questions to find out what went wrong",
        "Summarise the problem back to confirm you understood",
        "Provide a solution or an expected resolution time",
    ],
}

# Words that mean the customer may leave, complain formally, or is abusive.
ESCALATION_PATTERNS = [
    r"\bcancel", r"\blawyer", r"\blegal", r"\bmanager", r"\bsupervisor",
    r"never again", r"\bcomplain", r"\bunacceptable", r"\bsue\b",
    r"\bf+u+c+k", r"\bshit", r"\bbullshit",
]

MOODS = {
    "angry": "frustrated",
    "disgust": "dissatisfied",
    "sad": "upset",
    "fearful": "anxious",
}

SEVERITY_ORDER = ["medium", "high", "critical"]
IMPROVEMENT_DELTA = 0.25


def _normalise(text: str) -> str:
    return text.lower().replace("\u2019", "'").replace("\u2018", "'")


def _hits(text: str, patterns: list[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text))


def detect_reason(transcript: str) -> tuple[str, str, list[str]]:
    """Return (reason key, phrase, suggested actions) for a transcript."""
    text = _normalise(transcript)
    best_key, best_hits = "general", 0
    for key, spec in REASONS.items():
        hits = _hits(text, spec["patterns"])
        if hits > best_hits:
            best_key, best_hits = key, hits
    spec = REASONS.get(best_key, GENERAL)
    return best_key, spec["label"], list(spec["actions"])


def severity_for(score: float, transcript: str) -> str:
    if score <= -0.75:
        level = "critical"
    elif score <= -0.5:
        level = "high"
    else:
        level = "medium"
    if _hits(_normalise(transcript), ESCALATION_PATTERNS):
        level = max(level, "high", key=SEVERITY_ORDER.index)
    return level


def build_attention(
    transcript: str, emotion: EmotionResult, fusion: FusionResult
) -> Attention | None:
    """Steps 1-4: detect, find the reason, rate urgency, suggest a response."""
    if fusion.state != "negative":
        return None

    key, label, reason_actions = detect_reason(transcript)
    severity = severity_for(fusion.score, transcript)

    mood = "unhappy"
    if emotion.status == "ready" and emotion.confidence >= 0.4:
        mood = MOODS.get(emotion.label.lower(), "unhappy")

    actions = ["Apologize sincerely and acknowledge the customer's frustration"]
    actions += reason_actions
    if severity in ("high", "critical"):
        actions.insert(1, "Escalate to a senior agent or supervisor")

    return Attention(
        severity=severity,
        reason=key,
        reason_label=label,
        summary=f"{severity.title()} Attention: Customer appears {mood} about {label}.",
        actions=actions[:5],
    )


def build_followup(previous_event_id: str, previous_score: float, new_score: float) -> FollowUp:
    """Step 5: compare the customer's next interaction with the one that was handled."""
    delta = new_score - previous_score
    if delta >= IMPROVEMENT_DELTA or new_score >= POSITIVE_THRESHOLD:
        outcome = "improved"
    elif delta <= -IMPROVEMENT_DELTA:
        outcome = "worse"
    else:
        outcome = "unchanged"
    return FollowUp(
        previous_event_id=previous_event_id,
        previous_score=round(previous_score, 3),
        new_score=round(new_score, 3),
        outcome=outcome,
    )
