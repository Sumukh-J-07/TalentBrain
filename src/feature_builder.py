"""
feature_builder.py — Stage 2: Extract 75 engineered features per candidate.

Processes candidates in streaming chunks. No full dataset in memory.
Each candidate → flat dict of float features.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, FrozenSet, List, Optional, Set

from src.config import (
    ADJACENT_TECH_TITLES,
    CONSULTING_COMPANIES,
    DATA_TITLES,
    DEGREE_ORDINAL,
    ELITE_TITLES,
    EVALUATION_KEYWORDS,
    INSTITUTION_TIER_SCORE,
    LLM_KEYWORDS,
    MANDATORY_SKILLS,
    ML_CORE_TITLES,
    ML_KEYWORDS,
    ML_RELEVANT_FIELDS,
    NOISE_SKILLS,
    NON_RELEVANT_CAREER_KEYWORDS,
    NON_TECH_TITLES,
    PREFERRED_SKILLS,
    PRODUCTION_KEYWORDS,
    REFERENCE_DATE,
    RETRIEVAL_KEYWORDS,
    TIER1_SKILLS,
    TIER2_SKILLS,
    TITLE_RELEVANCE,
    ULTRA_RARE_SKILLS,
    ParsedJD,
)
from src.utils import (
    clamp,
    count_keyword_hits,
    normalize_text,
    parse_date,
    safe_div,
)

_REF_DATE = date.fromisoformat(REFERENCE_DATE)


# ===================================================================
# Main entry point
# ===================================================================

def extract_all_features(candidate: dict, jd: ParsedJD) -> Dict[str, float]:
    """Extract all 75 features for a single candidate.

    Args:
        candidate: Raw candidate dict from JSONL.
        jd: Parsed job description.

    Returns:
        Flat dict mapping feature_name → float value.
    """
    features: Dict[str, float] = {}

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    certs = candidate.get("certifications", [])
    signals = candidate.get("redrob_signals", {})

    # Pre-compute shared data
    skill_names: Set[str] = {s["name"] for s in skills}
    all_career_text = _concat_career_text(career)
    summary_text = normalize_text(profile.get("summary", ""))

    # Extract feature groups
    features.update(_skill_features(skills, skill_names, jd, signals))
    features.update(_career_features(profile, career, skill_names, all_career_text))
    features.update(_behavioral_features(signals))
    features.update(_education_features(education))
    features.update(_consistency_features(
        profile, career, skills, skill_names, signals, summary_text, all_career_text
    ))
    features.update(_text_features(profile, career, jd))

    return features


# ===================================================================
# Feature group: Skills (20 features)
# ===================================================================

def _skill_features(
    skills: List[dict],
    skill_names: Set[str],
    jd: ParsedJD,
    signals: dict,
) -> Dict[str, float]:
    f: Dict[str, float] = {}

    mandatory_overlap = skill_names & jd.mandatory_skills
    preferred_overlap = skill_names & jd.preferred_skills

    f["mandatory_skill_count"] = float(len(mandatory_overlap))
    f["mandatory_skill_coverage"] = safe_div(
        len(mandatory_overlap), len(jd.mandatory_skills)
    )
    f["preferred_skill_count"] = float(len(preferred_overlap))
    f["preferred_skill_coverage"] = safe_div(
        len(preferred_overlap), len(jd.preferred_skills)
    )
    f["total_skill_count"] = float(len(skills))

    f["tier1_skill_count"] = float(len(skill_names & TIER1_SKILLS))
    f["tier2_skill_count"] = float(len(skill_names & TIER2_SKILLS))
    f["ultra_rare_skill_count"] = float(len(skill_names & ULTRA_RARE_SKILLS))
    noise_count = len(skill_names & NOISE_SKILLS)
    f["noise_skill_ratio"] = safe_div(noise_count, len(skills))

    # Proficiency
    prof_map = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
    proficiencies = [prof_map.get(s.get("proficiency", ""), 0) for s in skills]
    f["advanced_skill_count"] = float(sum(1 for p in proficiencies if p >= 2))

    # Endorsements
    endorsements = [s.get("endorsements", 0) for s in skills]
    f["avg_endorsements"] = safe_div(sum(endorsements), len(endorsements))
    f["max_endorsements"] = float(max(endorsements)) if endorsements else 0.0
    f["total_endorsements"] = float(sum(endorsements))

    # Duration
    durations = [s.get("duration_months", 0) for s in skills]
    f["avg_skill_duration_months"] = safe_div(sum(durations), len(durations))

    # Duration weighted by relevance
    relevant_duration = sum(
        s.get("duration_months", 0)
        for s in skills
        if s["name"] in (TIER1_SKILLS | TIER2_SKILLS)
    )
    f["relevant_skill_duration"] = float(relevant_duration)

    # Assessments
    assessments = signals.get("skill_assessment_scores", {})
    scores = list(assessments.values()) if assessments else []
    f["assessment_count"] = float(len(scores))
    f["assessment_avg"] = safe_div(sum(scores), len(scores)) / 100.0 if scores else 0.0
    f["assessment_max"] = max(scores) / 100.0 if scores else 0.0

    # Domain-specific counts
    retrieval_skills = {"FAISS", "Pinecone", "Milvus", "Qdrant", "Weaviate",
                        "Elasticsearch", "OpenSearch", "pgvector", "Vector Search",
                        "Semantic Search", "Embeddings", "Sentence Transformers",
                        "Haystack", "LlamaIndex"}
    ranking_skills = {"Learning to Rank", "BM25", "Information Retrieval",
                      "Recommendation Systems"}
    llm_skills = {"LoRA", "QLoRA", "PEFT", "Fine-tuning LLMs", "LLMs",
                  "Hugging Face Transformers", "LangChain", "RAG",
                  "Prompt Engineering"}

    f["retrieval_skill_count"] = float(len(skill_names & retrieval_skills))
    f["ranking_skill_count"] = float(len(skill_names & ranking_skills))
    f["llm_skill_count"] = float(len(skill_names & llm_skills))

    return f


# ===================================================================
# Feature group: Career (18 features)
# ===================================================================

def _career_features(
    profile: dict,
    career: List[dict],
    skill_names: Set[str],
    all_career_text: str,
) -> Dict[str, float]:
    f: Dict[str, float] = {}

    yoe = profile.get("years_of_experience", 0.0)
    f["years_experience"] = float(yoe)
    f["career_entry_count"] = float(len(career))

    durations = [job.get("duration_months", 0) for job in career]
    f["avg_tenure_months"] = safe_div(sum(durations), len(durations))
    f["max_tenure_months"] = float(max(durations)) if durations else 0.0
    f["min_tenure_months"] = float(min(durations)) if durations else 0.0

    # Current tenure
    current_jobs = [j for j in career if j.get("is_current", False)]
    f["current_tenure_months"] = float(
        current_jobs[0].get("duration_months", 0) if current_jobs else 0
    )

    # Company analysis
    companies = {job.get("company", "") for job in career}
    consulting_set = {c for c in companies if c in CONSULTING_COMPANIES}
    f["consulting_only"] = 1.0 if (consulting_set and consulting_set == companies) else 0.0
    f["has_product_company"] = 1.0 if (companies - CONSULTING_COMPANIES) else 0.0

    # Company size analysis
    sizes = [job.get("company_size", "") for job in career]
    startup_sizes = {"1-10", "11-50", "51-200"}
    f["startup_exposure"] = 1.0 if any(s in startup_sizes for s in sizes) else 0.0
    f["big_tech_exposure"] = 1.0 if any(
        s == "10001+" and job.get("industry", "") in {"Software", "AI", "Technology"}
        for s, job in zip(sizes, career)
    ) else 0.0

    # Title analysis
    titles = [job.get("title", "") for job in career]
    ml_title_kw = {"ML", "Machine Learning", "AI", "Data Scientist", "NLP",
                   "Research Engineer", "Applied Scientist", "Recommendation",
                   "Search Engineer", "Ranking"}
    f["has_ml_title"] = 1.0 if any(
        any(kw.lower() in t.lower() for kw in ml_title_kw) for t in titles
    ) else 0.0

    current_title = profile.get("current_title", "")
    f["current_ml_title"] = 1.0 if any(
        kw.lower() in current_title.lower() for kw in ml_title_kw
    ) else 0.0

    # Total months in ML-titled roles
    ml_duration = sum(
        job.get("duration_months", 0)
        for job in career
        if any(kw.lower() in job.get("title", "").lower() for kw in ml_title_kw)
    )
    f["ml_title_duration_months"] = float(ml_duration)

    # Career progression (simplified: current title is more senior than first)
    f["career_progression_score"] = _compute_career_progression(career)

    # Job hopping: 3+ roles with < 18 months
    short_stints = sum(1 for d in durations if 0 < d < 18)
    f["job_hopping_flag"] = 1.0 if short_stints >= 3 else 0.0

    # Career description keyword analysis
    f["production_keyword_score"] = clamp(
        count_keyword_hits(all_career_text, PRODUCTION_KEYWORDS) / 10.0
    )
    f["retrieval_keyword_score"] = clamp(
        count_keyword_hits(all_career_text, RETRIEVAL_KEYWORDS) / 8.0
    )

    return f


def _compute_career_progression(career: List[dict]) -> float:
    """Score career progression from 0 to 1.

    Higher if titles show seniority growth (Junior → mid → Senior → Lead).
    """
    seniority_kw = {
        "intern": 0, "junior": 1, "associate": 1,
        "analyst": 2, "engineer": 3, "developer": 3,
        "senior": 4, "lead": 5, "staff": 6, "principal": 7,
        "director": 8, "vp": 9,
    }
    if len(career) < 2:
        return 0.5  # neutral

    def _title_level(title: str) -> int:
        title_lower = title.lower()
        best = 3  # default mid-level
        for kw, level in seniority_kw.items():
            if kw in title_lower:
                best = max(best, level)
        return best

    # Career is ordered most-recent-first
    levels = [_title_level(job.get("title", "")) for job in career]
    if levels[0] > levels[-1]:
        return 1.0  # progressed
    elif levels[0] == levels[-1]:
        return 0.5  # flat
    else:
        return 0.2  # regressed


# ===================================================================
# Feature group: Behavioral (15 features)
# ===================================================================

def _behavioral_features(signals: dict) -> Dict[str, float]:
    f: Dict[str, float] = {}

    f["profile_completeness"] = signals.get("profile_completeness_score", 0.0) / 100.0
    f["open_to_work"] = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    f["recruiter_response_rate"] = signals.get("recruiter_response_rate", 0.0)

    avg_resp = signals.get("avg_response_time_hours", 336.0)
    f["response_time_score"] = clamp(1.0 - (avg_resp / 336.0))

    github = signals.get("github_activity_score", -1)
    f["github_activity"] = clamp(max(0, github) / 100.0)
    f["has_github"] = 1.0 if github >= 0 else 0.0

    f["search_appearance_norm"] = clamp(
        signals.get("search_appearance_30d", 0) / 500.0
    )
    f["saved_by_recruiters_norm"] = clamp(
        signals.get("saved_by_recruiters_30d", 0) / 30.0
    )
    f["interview_completion_rate"] = signals.get("interview_completion_rate", 0.0)

    oar = signals.get("offer_acceptance_rate", -1)
    f["offer_acceptance_clean"] = max(0.0, oar) if oar >= 0 else 0.0

    notice = signals.get("notice_period_days", 90)
    f["notice_period_score"] = clamp(1.0 - (notice / 180.0))

    # Recency
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = (_REF_DATE - last_active).days
        f["days_since_active"] = float(max(0, days_inactive))
        f["recency_score"] = clamp(1.0 - (days_inactive / 365.0))
    else:
        f["days_since_active"] = 365.0
        f["recency_score"] = 0.0

    # Verification
    verified_count = sum([
        signals.get("verified_email", False),
        signals.get("verified_phone", False),
        signals.get("linkedin_connected", False),
    ])
    f["verified_score"] = verified_count / 3.0

    f["connection_norm"] = clamp(signals.get("connection_count", 0) / 500.0)

    return f


# ===================================================================
# Feature group: Education (5 features)
# ===================================================================

def _education_features(education: List[dict]) -> Dict[str, float]:
    f: Dict[str, float] = {}

    if not education:
        f["highest_degree_ordinal"] = 0.0
        f["best_tier"] = 0.0
        f["ml_relevant_field"] = 0.0
        f["has_postgrad"] = 0.0
        f["education_count"] = 0.0
        return f

    degrees = [DEGREE_ORDINAL.get(e.get("degree", ""), 1.0) for e in education]
    f["highest_degree_ordinal"] = max(degrees) / 4.0  # normalize to 0-1

    tiers = [
        INSTITUTION_TIER_SCORE.get(e.get("tier", "unknown"), 0.3)
        for e in education
    ]
    f["best_tier"] = max(tiers)

    f["ml_relevant_field"] = 1.0 if any(
        e.get("field_of_study", "") in ML_RELEVANT_FIELDS for e in education
    ) else 0.0

    postgrad_degrees = {"M.Tech", "M.E.", "M.S.", "M.Sc", "MBA", "Ph.D"}
    f["has_postgrad"] = 1.0 if any(
        e.get("degree", "") in postgrad_degrees for e in education
    ) else 0.0

    f["education_count"] = float(len(education))

    return f


# ===================================================================
# Feature group: Consistency / Honeypot signals (12 features)
# ===================================================================

def _consistency_features(
    profile: dict,
    career: List[dict],
    skills: List[dict],
    skill_names: Set[str],
    signals: dict,
    summary_text: str,
    all_career_text: str,
) -> Dict[str, float]:
    f: Dict[str, float] = {}

    current_title = profile.get("current_title", "")

    # 1. Title-description mismatch
    f["title_description_mismatch"] = _title_desc_mismatch(current_title, career)

    # 2. Summary-career mismatch
    f["summary_career_mismatch"] = _summary_career_mismatch(
        summary_text, current_title, career
    )

    # 3. Skill-career mismatch: advanced AI skills but no AI in career
    ai_skill_count = len(skill_names & (TIER1_SKILLS | TIER2_SKILLS))
    ai_in_career = count_keyword_hits(all_career_text, ML_KEYWORDS | RETRIEVAL_KEYWORDS)
    if ai_skill_count >= 5 and ai_in_career < 2:
        f["skill_career_mismatch"] = 1.0
    elif ai_skill_count >= 3 and ai_in_career < 1:
        f["skill_career_mismatch"] = 0.7
    else:
        f["skill_career_mismatch"] = 0.0

    # 4. Timeline overlap
    f["timeline_overlap"] = _detect_timeline_overlap(career)

    # 5. Impossible experience
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    yoe_months = profile.get("years_of_experience", 0) * 12
    gap = abs(yoe_months - total_career_months)
    f["experience_gap_months"] = float(gap)
    f["impossible_experience"] = 1.0 if gap > 36 else clamp(gap / 36.0)

    # 6. Keyword stuffing: non-tech title + many AI skills + no assessments
    assessments = signals.get("skill_assessment_scores", {})
    is_non_tech = current_title in NON_TECH_TITLES
    f["keyword_stuffing_score"] = 0.0
    if is_non_tech and ai_skill_count >= 5:
        if len(assessments) == 0:
            f["keyword_stuffing_score"] = 0.9
        elif len(assessments) < 2:
            f["keyword_stuffing_score"] = 0.6
        else:
            f["keyword_stuffing_score"] = 0.3

    # 7. Endorsement-proficiency mismatch
    expert_zero_endorse = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("endorsements", 0) == 0
    )
    f["endorsement_proficiency_mismatch"] = clamp(expert_zero_endorse / 5.0)

    # 8. Duration-proficiency mismatch
    short_advanced = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months", 0) < 6
    )
    f["duration_proficiency_mismatch"] = clamp(short_advanced / 5.0)

    # 9. Assessment-proficiency mismatch
    assess_mismatch = 0
    for s in skills:
        sname = s.get("name", "")
        if sname in assessments:
            score = assessments[sname]
            prof = s.get("proficiency", "")
            if prof in ("advanced", "expert") and score < 40:
                assess_mismatch += 1
    f["assessment_proficiency_mismatch"] = clamp(assess_mismatch / 3.0)

    # 10. Career title consistency (do titles make sense together?)
    f["career_title_consistency"] = _career_title_consistency(career)

    # 11. Non-relevant career content
    non_rel_hits = count_keyword_hits(all_career_text, NON_RELEVANT_CAREER_KEYWORDS)
    ml_hits = count_keyword_hits(all_career_text, ML_KEYWORDS | RETRIEVAL_KEYWORDS)
    f["non_relevant_career_ratio"] = clamp(
        safe_div(non_rel_hits, non_rel_hits + ml_hits + 1)
    )

    # 12. Composite honeypot signal (pre-computed for later use)
    f["honeypot_raw"] = (
        0.30 * f["title_description_mismatch"]
        + 0.25 * f["keyword_stuffing_score"]
        + 0.20 * f["summary_career_mismatch"]
        + 0.10 * f["endorsement_proficiency_mismatch"]
        + 0.08 * f["skill_career_mismatch"]
        + 0.07 * f["duration_proficiency_mismatch"]
    )

    return f


def _title_desc_mismatch(current_title: str, career: List[dict]) -> float:
    """Detect when career descriptions don't match the job title.

    E.g., title="Marketing Manager" but description talks about
    "streaming data pipelines on Kafka" or "Mechanical engineering design".
    """
    if not career:
        return 0.0

    mismatch_count = 0
    total = 0

    for job in career:
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        if not desc:
            continue
        total += 1

        # Check if description domain matches title domain
        title_domain = _infer_domain(title)
        desc_domain = _infer_domain_from_desc(desc)

        if title_domain and desc_domain and title_domain != desc_domain:
            mismatch_count += 1

    return clamp(safe_div(mismatch_count, max(total, 1)))


def _infer_domain(title: str) -> Optional[str]:
    """Infer professional domain from a job title."""
    title = title.lower()
    domain_map = {
        "engineering_ml": ["ml", "machine learning", "ai", "data scien", "nlp",
                          "research engineer", "recommendation", "search engineer"],
        "engineering_sw": ["software", "developer", "backend", "frontend",
                          "full stack", "devops", "cloud", "sre"],
        "engineering_data": ["data engineer", "analytics", "data analyst"],
        "marketing": ["marketing", "content", "seo", "brand"],
        "sales": ["sales", "account executive", "business development"],
        "hr": ["hr", "human resources", "recruiter", "talent"],
        "ops": ["operations", "supply chain", "logistics", "project manager"],
        "design": ["designer", "graphic", "ux", "ui"],
        "finance": ["accountant", "finance", "audit", "tax"],
        "support": ["support", "customer success", "customer service"],
        "engineering_mech": ["mechanical", "civil", "structural", "electrical"],
    }
    for domain, keywords in domain_map.items():
        if any(kw in title for kw in keywords):
            return domain
    return None


def _infer_domain_from_desc(desc: str) -> Optional[str]:
    """Infer professional domain from a career description."""
    desc = desc.lower()
    domain_indicators = {
        "engineering_ml": ["model", "training", "neural", "embedding", "nlp",
                          "classification", "prediction", "pytorch", "tensorflow",
                          "machine learning", "deep learning", "recommendation"],
        "engineering_sw": ["api", "microservice", "backend", "frontend",
                          "database", "server", "deploy", "cicd"],
        "engineering_data": ["pipeline", "etl", "warehouse", "spark", "airflow",
                            "data quality", "batch processing", "dbt"],
        "marketing": ["marketing", "demand-generation", "paid acquisition",
                     "seo", "content marketing", "brand", "campaign"],
        "sales": ["sales", "quota", "arr", "pipeline", "cold calling",
                 "enterprise sales", "revenue"],
        "hr": ["hiring", "onboarding", "payroll", "talent acquisition",
               "recruitment", "hr"],
        "ops": ["operations", "fulfillment", "warehouse", "logistics",
               "kpi", "continuous improvement", "supply chain"],
        "design": ["design", "brand identity", "logo", "typography",
                  "packaging", "figma", "adobe", "creative direction"],
        "finance": ["accounting", "financial reporting", "gaap", "tax",
                   "audit", "ledger", "month-end close", "statutory"],
        "support": ["support", "ticket", "tier-1", "tier-2", "escalation",
                   "customer-feedback", "knowledge base"],
        "engineering_mech": ["cad", "solidworks", "creo", "ansys", "fea",
                            "mechanical", "prototype", "tooling", "dfm"],
    }
    scores = {}
    for domain, keywords in domain_indicators.items():
        score = sum(1 for kw in keywords if kw in desc)
        if score > 0:
            scores[domain] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def _summary_career_mismatch(
    summary_text: str,
    current_title: str,
    career: List[dict],
) -> float:
    """Detect when summary mentions a different role than actual career.

    E.g., summary says "marketing manager" but title is "Operations Manager".
    """
    if not summary_text:
        return 0.0

    # Check if summary mentions a role different from current title
    title_lower = current_title.lower()
    role_mentions = {
        "marketing manager": "marketing",
        "operations manager": "ops",
        "hr manager": "hr",
        "accountant": "finance",
        "sales executive": "sales",
        "mechanical engineer": "mech",
        "civil engineer": "civil",
        "content writer": "content",
        "graphic designer": "design",
        "customer support": "support",
        "business analyst": "business",
        "project manager": "pm",
    }

    mentioned_roles = []
    for role_text, domain in role_mentions.items():
        if role_text in summary_text:
            mentioned_roles.append(role_text)

    if not mentioned_roles:
        return 0.0

    # Check if any mentioned role differs from actual title
    for mentioned in mentioned_roles:
        if mentioned not in title_lower:
            return 0.8  # summary mentions a different role

    return 0.0


def _detect_timeline_overlap(career: List[dict]) -> float:
    """Detect overlapping career entries (impossible to hold two jobs)."""
    dated_jobs = []
    for job in career:
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        if start:
            dated_jobs.append((start, end or _REF_DATE))

    if len(dated_jobs) < 2:
        return 0.0

    dated_jobs.sort(key=lambda x: x[0])
    overlaps = 0
    for i in range(len(dated_jobs) - 1):
        _, end_i = dated_jobs[i]
        start_j, _ = dated_jobs[i + 1]
        if start_j < end_i:
            overlap_days = (end_i - start_j).days
            if overlap_days > 30:  # > 1 month overlap
                overlaps += 1

    return clamp(overlaps / 2.0)


def _career_title_consistency(career: List[dict]) -> float:
    """Check if career titles follow a logical domain progression.

    Returns 1.0 for consistent, 0.0 for highly inconsistent.
    """
    if len(career) < 2:
        return 1.0

    domains = []
    for job in career:
        title = job.get("title", "")
        domain = _infer_domain(title)
        if domain:
            domains.append(domain)

    if len(domains) < 2:
        return 1.0

    # Count domain switches
    switches = sum(1 for i in range(len(domains) - 1) if domains[i] != domains[i + 1])
    unique_domains = len(set(domains))

    # Many domain switches = inconsistent
    consistency = 1.0 - clamp(switches / max(len(domains) - 1, 1))

    # Penalize > 3 unique domains
    if unique_domains > 3:
        consistency *= 0.5

    return consistency


# ===================================================================
# Feature group: Text (5 features)
# ===================================================================

def _text_features(
    profile: dict,
    career: List[dict],
    jd: ParsedJD,
) -> Dict[str, float]:
    f: Dict[str, float] = {}

    summary = profile.get("summary", "")
    headline = profile.get("headline", "")
    all_career_text = _concat_career_text(career)

    jd_keywords = jd.retrieval_keywords | jd.production_keywords | jd.evaluation_keywords

    f["summary_length"] = clamp(len(summary.split()) / 200.0)
    f["career_desc_total_length"] = clamp(
        len(all_career_text.split()) / 500.0
    )
    f["summary_relevance_keywords"] = clamp(
        count_keyword_hits(summary.lower(), jd_keywords) / 10.0
    )
    f["career_relevance_keywords"] = clamp(
        count_keyword_hits(all_career_text, jd_keywords) / 15.0
    )
    f["headline_relevance"] = clamp(
        count_keyword_hits(headline.lower(), jd_keywords) / 3.0
    )

    return f


# ===================================================================
# Helpers
# ===================================================================

def _concat_career_text(career: List[dict]) -> str:
    """Concatenate all career descriptions into one lowercase string."""
    parts = []
    for job in career:
        desc = job.get("description", "")
        if desc:
            parts.append(desc.lower())
    return " ".join(parts)
