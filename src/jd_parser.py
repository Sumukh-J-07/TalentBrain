"""
jd_parser.py — Stage 1: Parse job_description.docx into a ParsedJD struct.

Rule-based extraction. No external API. No LLM.
The JD is known and fixed — this is a deterministic parser.
"""
from __future__ import annotations

from src.config import (
    CONSULTING_COMPANIES,
    EVALUATION_KEYWORDS,
    MANDATORY_SKILLS,
    PREFERRED_SKILLS,
    PRODUCTION_KEYWORDS,
    RETRIEVAL_KEYWORDS,
    ParsedJD,
)
from src.utils import build_jd_query_text


def parse_jd() -> ParsedJD:
    """Return the ParsedJD for the Redrob hackathon JD.

    Since the JD is fixed and fully analyzed, this returns a hardcoded
    struct.  No file I/O needed at ranking time — the JD understanding
    is baked in.  This keeps ranking runtime at zero for Stage 1.
    """
    jd = ParsedJD(
        title="Senior AI/ML Engineer — Ranking, Retrieval & Matching",

        mandatory_skills=MANDATORY_SKILLS,
        preferred_skills=PREFERRED_SKILLS,

        negative_patterns=frozenset({
            "pure research without production",
            "langchain-only AI experience",
            "consulting-only career",
            "computer vision only without NLP/IR",
            "speech/robotics only without NLP/IR",
            "title chasing (switching every 1.5 years)",
            "no production code in 18 months",
            "framework enthusiast without systems thinking",
        }),

        # JD says "5-9 years" as range, but will consider outside
        experience_min=3.0,
        experience_max=15.0,
        experience_ideal_min=5.0,
        experience_ideal_max=9.0,

        # Weights derived from JD emphasis
        startup_weight=0.6,       # prefers product companies, startup is bonus
        production_weight=0.9,    # very high — "this role writes code"
        shipper_weight=0.8,       # "tilt slightly toward shipper"
        research_penalty=0.50,    # "pure research → will not move forward"
        consulting_penalty=0.60,  # "entire career at consulting → not a fit"
        langchain_only_penalty=0.70,
        cv_speech_only_penalty=0.65,

        location_preferences=frozenset({
            "India", "Pune", "Noida", "Hyderabad", "Mumbai",
            "Delhi NCR", "Gurgaon", "Bengaluru", "Bangalore",
        }),

        notice_period_ideal_max=30,  # "sub-30-day notice" preferred

        behavioral_weights={
            "recruiter_response_rate": 0.30,
            "recency": 0.25,
            "open_to_work": 0.15,
            "profile_completeness": 0.10,
            "interview_completion": 0.10,
            "github_activity": 0.10,
        },

        retrieval_keywords=RETRIEVAL_KEYWORDS,
        production_keywords=PRODUCTION_KEYWORDS,
        evaluation_keywords=EVALUATION_KEYWORDS,
    )

    # Build query text for embedding retrieval
    query_text = build_jd_query_text(jd)
    # Since ParsedJD is frozen, we create a new one with query_text
    jd = ParsedJD(
        title=jd.title,
        mandatory_skills=jd.mandatory_skills,
        preferred_skills=jd.preferred_skills,
        negative_patterns=jd.negative_patterns,
        experience_min=jd.experience_min,
        experience_max=jd.experience_max,
        experience_ideal_min=jd.experience_ideal_min,
        experience_ideal_max=jd.experience_ideal_max,
        startup_weight=jd.startup_weight,
        production_weight=jd.production_weight,
        shipper_weight=jd.shipper_weight,
        research_penalty=jd.research_penalty,
        consulting_penalty=jd.consulting_penalty,
        langchain_only_penalty=jd.langchain_only_penalty,
        cv_speech_only_penalty=jd.cv_speech_only_penalty,
        location_preferences=jd.location_preferences,
        notice_period_ideal_max=jd.notice_period_ideal_max,
        behavioral_weights=jd.behavioral_weights,
        retrieval_keywords=jd.retrieval_keywords,
        production_keywords=jd.production_keywords,
        evaluation_keywords=jd.evaluation_keywords,
        query_text=query_text,
    )

    return jd
