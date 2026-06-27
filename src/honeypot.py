"""
honeypot.py — Stage 5: Detect honeypot candidates.

5 detectors, each producing a fraud probability [0, 1].
Composite score determines penalty multiplier.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.config import (
    HONEYPOT_MULTIPLIER_HIGH,
    HONEYPOT_MULTIPLIER_MED,
    HONEYPOT_THRESHOLD_HIGH,
    HONEYPOT_THRESHOLD_MED,
    NON_TECH_TITLES,
    TIER1_SKILLS,
    TIER2_SKILLS,
)
from src.utils import clamp


def detect_honeypot(
    candidate: dict,
    features: Dict[str, float],
) -> Tuple[float, float, str]:
    """Run all 5 honeypot detectors on a single candidate.

    Args:
        candidate: Raw candidate dict from JSONL.
        features: Pre-computed feature dict from feature_builder.

    Returns:
        Tuple of (honeypot_score, multiplier, explanation).
        - honeypot_score: 0.0 (clean) to 1.0 (definite honeypot)
        - multiplier: applied to final score (1.0 = no penalty, 0.0 = zeroed)
        - explanation: brief text explaining the detection
    """
    scores: Dict[str, float] = {}
    reasons: List[str] = []

    # Detector 1: Title-Description Mismatch
    td_score = features.get("title_description_mismatch", 0.0)
    scores["title_desc"] = td_score
    if td_score > 0.5:
        reasons.append("career descriptions don't match job titles")

    # Detector 2: Keyword Stuffing
    ks_score = features.get("keyword_stuffing_score", 0.0)
    scores["keyword_stuffing"] = ks_score
    if ks_score > 0.5:
        reasons.append("AI skills listed without supporting career evidence or assessments")

    # Detector 3: Summary-Career Mismatch
    sc_score = features.get("summary_career_mismatch", 0.0)
    scores["summary_career"] = sc_score
    if sc_score > 0.5:
        reasons.append("summary references a different role than actual career history")

    # Detector 4: Proficiency-Evidence Mismatch
    ep_score = features.get("endorsement_proficiency_mismatch", 0.0)
    dp_score = features.get("duration_proficiency_mismatch", 0.0)
    ap_score = features.get("assessment_proficiency_mismatch", 0.0)
    prof_score = max(ep_score, dp_score, ap_score)
    scores["proficiency_evidence"] = prof_score
    if prof_score > 0.5:
        reasons.append("claimed proficiency not backed by endorsements, duration, or assessments")

    # Detector 5: Skill-Career Mismatch
    skc_score = features.get("skill_career_mismatch", 0.0)
    scores["skill_career"] = skc_score
    if skc_score > 0.5:
        reasons.append("advanced AI skills listed but career history shows no AI work")

    # Composite honeypot score
    honeypot_score = (
        0.30 * scores["title_desc"]
        + 0.25 * scores["keyword_stuffing"]
        + 0.20 * scores["summary_career"]
        + 0.15 * scores["proficiency_evidence"]
        + 0.10 * scores["skill_career"]
    )
    honeypot_score = clamp(honeypot_score)

    # Determine multiplier
    if honeypot_score >= HONEYPOT_THRESHOLD_HIGH:
        multiplier = HONEYPOT_MULTIPLIER_HIGH
    elif honeypot_score >= HONEYPOT_THRESHOLD_MED:
        multiplier = HONEYPOT_MULTIPLIER_MED
    else:
        multiplier = 1.0

    # Build explanation
    if reasons:
        explanation = "⚠️ Flagged: " + "; ".join(reasons) + "."
    else:
        explanation = ""

    return honeypot_score, multiplier, explanation


def batch_detect_honeypots(
    candidates: List[dict],
    features_list: List[Dict[str, float]],
) -> List[Tuple[float, float, str]]:
    """Run honeypot detection on a batch of candidates.

    Args:
        candidates: List of raw candidate dicts.
        features_list: Corresponding feature dicts.

    Returns:
        List of (honeypot_score, multiplier, explanation) tuples.
    """
    results = []
    for cand, feats in zip(candidates, features_list):
        results.append(detect_honeypot(cand, feats))
    return results
