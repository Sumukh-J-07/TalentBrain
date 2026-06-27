"""
reasoning.py — Stage 6: Generate factual 1-2 sentence explanations.

Template-based only. No LLM. No hallucination. Uses only verified candidate data.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from src.config import (
    MANDATORY_SKILLS,
    PREFERRED_SKILLS,
    TIER1_SKILLS,
    ParsedJD,
)


def generate_reasoning(
    candidate: dict,
    features: Dict[str, float],
    score: float,
    rank: int,
    honeypot_explanation: str,
    jd: ParsedJD,
) -> str:
    """Generate a factual 1-2 sentence reasoning for a ranked candidate.

    Args:
        candidate: Raw candidate dict from JSONL.
        features: Feature dict from feature_builder.
        score: Final ranking score.
        rank: Final rank (1-100).
        honeypot_explanation: Honeypot detection explanation (empty if clean).
        jd: Parsed JD.

    Returns:
        1-2 sentence reasoning string. Factual only — no invented information.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    # Extract verified facts
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0.0)

    # Skill overlap with JD
    skill_names: Set[str] = {s["name"] for s in skills}
    mandatory_matches = sorted(skill_names & jd.mandatory_skills)
    preferred_matches = sorted(skill_names & jd.preferred_skills)
    tier1_matches = sorted(skill_names & TIER1_SKILLS)

    # Behavioral facts
    response_rate = signals.get("recruiter_response_rate", 0.0)
    github_score = signals.get("github_activity_score", -1)
    notice_days = signals.get("notice_period_days", 0)
    open_to_work = signals.get("open_to_work_flag", False)

    # Career facts
    career_companies = [j.get("company", "") for j in career[:3]]
    current_tenure = features.get("current_tenure_months", 0)

    # Build reasoning parts
    parts: List[str] = []

    # Part 1: Core profile summary
    skill_summary = ""
    if mandatory_matches:
        top_skills = mandatory_matches[:4]
        skill_summary = f"with {len(mandatory_matches)} core skills ({', '.join(top_skills)})"
    elif tier1_matches:
        top_skills = tier1_matches[:3]
        skill_summary = f"with relevant skills in {', '.join(top_skills)}"
    elif preferred_matches:
        top_skills = preferred_matches[:3]
        skill_summary = f"with preferred skills: {', '.join(top_skills)}"
    else:
        skill_summary = "with limited skill overlap"

    parts.append(
        f"{title} at {company} ({yoe:.1f}Y experience) {skill_summary}"
    )

    # Part 2: Differentiating signals
    signals_parts: List[str] = []

    # Production/retrieval signals from career description
    retrieval_kw = features.get("retrieval_keyword_score", 0.0)
    production_kw = features.get("production_keyword_score", 0.0)
    if retrieval_kw > 0.3:
        signals_parts.append("retrieval/search experience in career history")
    elif production_kw > 0.3:
        signals_parts.append("production deployment experience")

    # GitHub
    if github_score > 50:
        signals_parts.append(f"active GitHub (score: {github_score:.0f})")
    elif github_score > 20:
        signals_parts.append("moderate GitHub activity")

    # Behavioral
    if response_rate > 0.6:
        signals_parts.append(f"high recruiter engagement ({response_rate:.0%})")
    elif response_rate < 0.2:
        signals_parts.append(f"low recruiter response rate ({response_rate:.0%})")

    if signals_parts:
        parts.append("; ".join(signals_parts))

    # Part 3: Concerns (if any)
    concerns: List[str] = []
    if notice_days > 90:
        concerns.append(f"{notice_days}-day notice period")
    if not open_to_work:
        concerns.append("not marked open to work")
    if features.get("consulting_only", 0.0) > 0.5:
        concerns.append("consulting-only career")
    if features.get("job_hopping_flag", 0.0) > 0.5:
        concerns.append("frequent job changes")

    if concerns:
        parts.append("Concerns: " + ", ".join(concerns))

    # Combine
    reasoning = ". ".join(parts) + "."

    # Add honeypot warning if flagged
    if honeypot_explanation:
        reasoning = honeypot_explanation + " " + reasoning

    # Ensure no newlines or commas that break CSV
    reasoning = reasoning.replace("\n", " ").replace("\r", " ")
    # Escape commas for CSV safety — the CSV writer handles quoting,
    # but let's also keep reasoning clean
    reasoning = reasoning.strip()

    # Truncate if too long (keep under 500 chars for CSV readability)
    if len(reasoning) > 500:
        reasoning = reasoning[:497] + "..."

    return reasoning


def batch_generate_reasoning(
    candidates: List[dict],
    features_list: List[Dict[str, float]],
    scores: List[float],
    ranks: List[int],
    honeypot_explanations: List[str],
    jd: ParsedJD,
) -> List[str]:
    """Generate reasoning for a batch of candidates.

    Args:
        candidates: List of raw candidate dicts.
        features_list: Corresponding feature dicts.
        scores: Final scores.
        ranks: Final ranks.
        honeypot_explanations: Honeypot explanations.
        jd: Parsed JD.

    Returns:
        List of reasoning strings.
    """
    return [
        generate_reasoning(cand, feats, score, rank, hp_exp, jd)
        for cand, feats, score, rank, hp_exp
        in zip(candidates, features_list, scores, ranks, honeypot_explanations)
    ]
