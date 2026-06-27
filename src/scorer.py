"""
scorer.py — Stage 4: Weighted hybrid scoring.

Combines feature signals into a single ranking score.
No ML model needed — direct weighted combination with penalty multipliers.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.config import (
    INACTIVE_DAYS_THRESHOLD,
    INACTIVE_PENALTY,
    NOTICE_PERIOD_PENALTIES,
    SCORING_WEIGHTS,
    TITLE_RELEVANCE,
    ParsedJD,
)
from src.utils import clamp, safe_div


def compute_candidate_score(
    features: Dict[str, float],
    semantic_sim: float,
    candidate: dict,
    jd: ParsedJD,
) -> Tuple[float, Dict[str, float]]:
    """Compute the final ranking score for a single candidate.

    Args:
        features: Feature dict from feature_builder.
        semantic_sim: Cosine similarity from retrieval (0-1).
        candidate: Raw candidate dict.
        jd: Parsed JD.

    Returns:
        Tuple of (final_score, component_scores) where component_scores
        maps component name → weighted contribution.
    """
    components: Dict[str, float] = {}

    # --- Component scores (each in [0, 1]) ---

    # 1. Mandatory skill coverage
    components["mandatory_skill_coverage"] = features.get(
        "mandatory_skill_coverage", 0.0
    )

    # 2. Semantic similarity (from FAISS retrieval)
    components["semantic_similarity"] = clamp(semantic_sim)

    # 3. Career relevance — weighted combo of career keyword signals
    career_rel = (
        0.35 * features.get("retrieval_keyword_score", 0.0)
        + 0.25 * features.get("production_keyword_score", 0.0)
        + 0.20 * features.get("career_relevance_keywords", 0.0)
        + 0.10 * features.get("has_ml_title", 0.0)
        + 0.10 * features.get("ml_title_duration_months", 0.0) / 60.0
    )
    components["career_relevance"] = clamp(career_rel)

    # 4. Title relevance
    current_title = candidate.get("profile", {}).get("current_title", "")
    title_score = TITLE_RELEVANCE.get(current_title, 0.15)
    components["title_relevance"] = title_score

    # 5. Retrieval experience
    retrieval_exp = (
        0.40 * clamp(features.get("retrieval_skill_count", 0.0) / 5.0)
        + 0.30 * features.get("retrieval_keyword_score", 0.0)
        + 0.30 * clamp(features.get("relevant_skill_duration", 0.0) / 120.0)
    )
    components["retrieval_experience"] = clamp(retrieval_exp)

    # 6. Production experience
    prod_exp = (
        0.50 * features.get("production_keyword_score", 0.0)
        + 0.25 * features.get("has_product_company", 0.0)
        + 0.25 * (1.0 - features.get("consulting_only", 0.0))
    )
    components["production_experience"] = clamp(prod_exp)

    # 7. Behavioral score
    beh_weights = jd.behavioral_weights
    behavioral = (
        beh_weights.get("recruiter_response_rate", 0.3)
            * features.get("recruiter_response_rate", 0.0)
        + beh_weights.get("recency", 0.25)
            * features.get("recency_score", 0.0)
        + beh_weights.get("open_to_work", 0.15)
            * features.get("open_to_work", 0.0)
        + beh_weights.get("profile_completeness", 0.1)
            * features.get("profile_completeness", 0.0)
        + beh_weights.get("interview_completion", 0.1)
            * features.get("interview_completion_rate", 0.0)
        + beh_weights.get("github_activity", 0.1)
            * features.get("github_activity", 0.0)
    )
    components["behavioral_score"] = clamp(behavioral)

    # 8. Evaluation experience
    eval_exp = (
        0.50 * clamp(features.get("ranking_skill_count", 0.0) / 3.0)
        + 0.50 * features.get("career_relevance_keywords", 0.0)
    )
    components["evaluation_experience"] = clamp(eval_exp)

    # 9. Career stability
    stability = (
        0.40 * (1.0 - features.get("job_hopping_flag", 0.0))
        + 0.30 * clamp(features.get("avg_tenure_months", 0.0) / 36.0)
        + 0.30 * features.get("career_progression_score", 0.5)
    )
    components["career_stability"] = clamp(stability)

    # 10. GitHub score
    components["github_score"] = features.get("github_activity", 0.0)

    # 11. Startup exposure
    components["startup_exposure"] = features.get("startup_exposure", 0.0)

    # 12. Education relevance
    edu_rel = (
        0.30 * features.get("ml_relevant_field", 0.0)
        + 0.30 * features.get("highest_degree_ordinal", 0.0)
        + 0.20 * features.get("best_tier", 0.0)
        + 0.20 * features.get("has_postgrad", 0.0)
    )
    components["education_relevance"] = clamp(edu_rel)

    # --- Weighted sum ---
    raw_score = sum(
        SCORING_WEIGHTS[name] * value
        for name, value in components.items()
    )

    # --- Penalty multipliers ---
    multiplier = 1.0

    # Notice period penalty
    notice_days = candidate.get("redrob_signals", {}).get("notice_period_days", 90)
    for threshold, penalty in NOTICE_PERIOD_PENALTIES:
        if notice_days > threshold:
            multiplier *= penalty
            break

    # Inactivity penalty
    days_inactive = features.get("days_since_active", 0.0)
    if days_inactive > INACTIVE_DAYS_THRESHOLD:
        multiplier *= INACTIVE_PENALTY

    # Not open to work penalty
    if not candidate.get("redrob_signals", {}).get("open_to_work_flag", True):
        multiplier *= 0.92

    # Consulting-only penalty
    if features.get("consulting_only", 0.0) > 0.5:
        multiplier *= jd.consulting_penalty

    # Career consistency penalty (anti-honeypot)
    career_consistency = features.get("career_title_consistency", 1.0)
    if career_consistency < 0.5:
        multiplier *= (0.5 + career_consistency)

    # Non-relevant career content penalty
    non_rel = features.get("non_relevant_career_ratio", 0.0)
    if non_rel > 0.7:
        multiplier *= 0.6

    # LangChain-only AI check
    skills_set = {s["name"] for s in candidate.get("skills", [])}
    langchain_only = (
        "LangChain" in skills_set
        and features.get("tier1_skill_count", 0) < 2
        and features.get("retrieval_keyword_score", 0) < 0.2
    )
    if langchain_only:
        multiplier *= jd.langchain_only_penalty

    # Experience band adjustment
    yoe = features.get("years_experience", 0.0)
    if yoe < jd.experience_min:
        multiplier *= max(0.5, yoe / jd.experience_min)
    elif yoe > jd.experience_max:
        multiplier *= max(0.7, 1.0 - (yoe - jd.experience_max) / 20.0)

    # Ultra-rare skill bonus (strong positive signal)
    ultra_rare = features.get("ultra_rare_skill_count", 0.0)
    if ultra_rare > 0:
        multiplier *= (1.0 + 0.05 * ultra_rare)

    final_score = clamp(raw_score * multiplier, 0.0, 1.0)

    return final_score, components


def batch_score_candidates(
    features_list: List[Dict[str, float]],
    semantic_sims: np.ndarray,
    candidates: List[dict],
    jd: ParsedJD,
) -> List[Tuple[float, Dict[str, float]]]:
    """Score a batch of candidates.

    Args:
        features_list: List of feature dicts.
        semantic_sims: Array of cosine similarities.
        candidates: List of raw candidate dicts.
        jd: Parsed JD.

    Returns:
        List of (final_score, component_scores) tuples.
    """
    results = []
    for feats, sim, cand in zip(features_list, semantic_sims, candidates):
        results.append(compute_candidate_score(feats, float(sim), cand, jd))
    return results
