"""
utils.py — Timing, memory monitoring, text processing utilities.
"""
from __future__ import annotations

import functools
import gc
import re
import time
from datetime import date, datetime
from typing import List, Set

import psutil


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """Context-manager / decorator for wall-clock timing."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            print(f"  [{self.label}] {self.elapsed:.2f}s")
        return False

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper


def log_memory(label: str = "") -> float:
    """Log current RSS memory in MB and return the value."""
    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    if label:
        print(f"  [MEM {label}] {rss_mb:.0f} MB")
    return rss_mb


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def count_keyword_hits(text: str, keywords: Set[str] | frozenset) -> int:
    """Count how many keywords from the set appear in the text.

    Both text and keywords should be lowercased.  Keywords can be
    multi-word (e.g. "vector search").  We do substring matching
    so "vector search" matches "built a vector search system".
    """
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def keyword_density(text: str, keywords: Set[str] | frozenset) -> float:
    """Fraction of keywords that appear in text."""
    if not keywords or not text:
        return 0.0
    return count_keyword_hits(text, keywords) / len(keywords)


def parse_date(date_str: str | None) -> date | None:
    """Parse ISO date string to date object. Returns None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def days_between(d1: date | None, d2: date | None) -> int | None:
    """Days between two dates. Returns None if either is None."""
    if d1 is None or d2 is None:
        return None
    return abs((d2 - d1).days)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division with zero-safety."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def build_candidate_text(candidate: dict) -> str:
    """Build a single text string for embedding from candidate profile.

    Format: headline + summary + career descriptions + skills + certs.
    Truncated to ~500 words to stay within MiniLM context.
    """
    parts: List[str] = []

    profile = candidate.get("profile", {})
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])

    for job in candidate.get("career_history", []):
        desc = job.get("description", "")
        title = job.get("title", "")
        company = job.get("company", "")
        if title or company:
            parts.append(f"{title} at {company}")
        if desc:
            parts.append(desc)

    skill_names = [s["name"] for s in candidate.get("skills", [])]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    certs = candidate.get("certifications", [])
    if certs:
        cert_names = [c.get("name", "") for c in certs]
        parts.append("Certifications: " + ", ".join(cert_names))

    full_text = " ".join(parts)
    # Rough truncation to ~500 words
    words = full_text.split()
    if len(words) > 500:
        full_text = " ".join(words[:500])

    return full_text


def build_jd_query_text(jd) -> str:
    """Build a query string from ParsedJD for semantic retrieval."""
    parts = [
        "Senior AI/ML Engineer for ranking and retrieval systems.",
        "Must have experience with embeddings-based retrieval, vector databases,",
        "production deployment, and evaluation frameworks like NDCG and MRR.",
        "Skills needed: " + ", ".join(sorted(jd.mandatory_skills)),
        "Preferred: " + ", ".join(sorted(jd.preferred_skills)),
        "Looking for someone who has shipped ranking systems to production users.",
        "Strong Python, hands-on ML/NLP, startup experience preferred.",
    ]
    return " ".join(parts)


def force_gc():
    """Force garbage collection."""
    gc.collect()
