"""
pipeline.py — End-to-end orchestrator: Stage 1 → 6.

Loads candidates, runs all stages, produces ranked output.
Memory-efficient: streams JSONL, processes in chunks.
"""
from __future__ import annotations

import csv
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    CANDIDATE_LOAD_CHUNK_SIZE,
    RETRIEVAL_TOP_K,
    TOP_N_HONEYPOT,
    TOP_N_OUTPUT,
    TOP_N_SCORING,
    ParsedJD,
)
from src.feature_builder import extract_all_features
from src.honeypot import batch_detect_honeypots
from src.jd_parser import parse_jd
from src.reasoning import batch_generate_reasoning
from src.retrieval import SemanticRetriever
from src.scorer import batch_score_candidates
from src.utils import Timer, build_candidate_text, force_gc, log_memory


def run_pipeline(
    candidates_path: str,
    output_path: str,
    models_dir: str = "models",
    use_cache: bool = True,
) -> List[Dict]:
    """Run the full 6-stage ranking pipeline.

    Args:
        candidates_path: Path to candidates.jsonl.
        output_path: Path to write submission.csv.
        models_dir: Directory for cached embeddings/index.
        use_cache: Whether to use cached embeddings/index.

    Returns:
        List of dicts with {candidate_id, rank, score, reasoning} for top 100.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    total_timer = Timer("TOTAL PIPELINE")
    total_timer.__enter__()

    # ===================================================================
    # STAGE 1: Parse JD
    # ===================================================================
    with Timer("Stage 1: JD Parser"):
        jd = parse_jd()
    print(f"  Mandatory skills: {len(jd.mandatory_skills)}")
    print(f"  Preferred skills: {len(jd.preferred_skills)}")

    # ===================================================================
    # STAGE 2 + STAGE 3 combined: Load candidates, build features & texts
    # ===================================================================
    with Timer("Stage 2+3: Load + Features + Texts"):
        candidates, all_features, candidate_texts = _load_and_featurize(
            candidates_path, jd
        )

    n_candidates = len(candidates)
    n_features = len(all_features[0]) if all_features else 0
    print(f"  Loaded {n_candidates} candidates, {n_features} features each")
    log_memory("after featurization")

    # ===================================================================
    # STAGE 2.5: Feature-based Prefilter (Top 5000)
    # ===================================================================
    with Timer("Stage 2.5: Prefilter"):
        # We compute a lightweight heuristic score to select top 5000 candidates to embed
        # using mandatory skills, retrieval skills, title tier, and profile completeness.
        prefilter_scores = []
        for i, feats in enumerate(all_features):
            score = (
                feats.get("mandatory_skill_count", 0.0) * 1.5 +
                feats.get("retrieval_skill_count", 0.0) * 1.0 +
                feats.get("has_ml_title", 0.0) * 1.0 +
                feats.get("profile_completeness", 0.0) * 0.5
            )
            prefilter_scores.append((i, score))
        
        prefilter_scores.sort(key=lambda x: x[1], reverse=True)
        # Keep top 5000
        top_5000_indices = [idx for idx, _ in prefilter_scores[:5000]]
        
        # Subset candidates, features, and texts
        candidates = [candidates[i] for i in top_5000_indices]
        all_features = [all_features[i] for i in top_5000_indices]
        candidate_texts = [candidate_texts[i] for i in top_5000_indices]
        
    print(f"  Prefiltered to {len(candidates)} candidates for embedding")
    log_memory("after prefilter")

    # ===================================================================
    # STAGE 3: Semantic Retrieval
    # ===================================================================
    with Timer("Stage 3: Semantic Retrieval"):
        retriever = SemanticRetriever()

        emb_cache = models_path / f"embeddings_{len(candidates)}.npy" if use_cache else None
        idx_cache = models_path / f"faiss_index_{len(candidates)}.bin" if use_cache else None

        retriever.encode_candidates(candidate_texts, cache_path=emb_cache)
        retriever.build_index(cache_path=idx_cache)

        # Retrieve top-K
        retrieved_indices, retrieved_distances = retriever.retrieve(
            jd.query_text, top_k=RETRIEVAL_TOP_K
        )

        # Get similarity scores for retrieved candidates
        semantic_sims = retriever.get_similarity_scores(
            jd.query_text, retrieved_indices
        )

        # Free model memory
        retriever.unload_model()

    print(f"  Retrieved top {len(retrieved_indices)} candidates")
    log_memory("after retrieval")

    # Free texts — no longer needed
    del candidate_texts
    force_gc()

    # ===================================================================
    # STAGE 4: Hybrid Scoring (on top-K retrieved candidates)
    # ===================================================================
    with Timer("Stage 4: Hybrid Scoring"):
        # Gather retrieved candidates and their features
        retrieved_candidates = [candidates[i] for i in retrieved_indices]
        retrieved_features = [all_features[i] for i in retrieved_indices]

        # Score all retrieved candidates
        score_results = batch_score_candidates(
            retrieved_features, semantic_sims, retrieved_candidates, jd
        )

        # Sort by score descending
        scored = list(zip(
            retrieved_indices,
            retrieved_candidates,
            retrieved_features,
            semantic_sims,
            score_results,
        ))
        scored.sort(key=lambda x: x[4][0], reverse=True)

        # Take top N for honeypot analysis
        top_n = scored[:TOP_N_HONEYPOT]

    print(f"  Top score: {top_n[0][4][0]:.4f}")
    print(f"  Score range: {top_n[-1][4][0]:.4f} - {top_n[0][4][0]:.4f}")

    # ===================================================================
    # STAGE 5: Honeypot Detection (on top-150)
    # ===================================================================
    with Timer("Stage 5: Honeypot Detection"):
        hp_candidates = [x[1] for x in top_n]
        hp_features = [x[2] for x in top_n]

        honeypot_results = batch_detect_honeypots(hp_candidates, hp_features)

        # Apply honeypot multipliers to scores
        adjusted = []
        honeypot_count = 0
        for i, (idx, cand, feats, sim, (score, components)) in enumerate(top_n):
            hp_score, hp_mult, hp_explanation = honeypot_results[i]

            adjusted_score = score * hp_mult
            adjusted.append((
                idx, cand, feats, sim,
                adjusted_score, components,
                hp_score, hp_mult, hp_explanation,
            ))

            if hp_mult < 1.0:
                honeypot_count += 1

        # Re-sort after honeypot adjustment
        adjusted.sort(key=lambda x: x[4], reverse=True)

        # Take final top 100
        final_100 = adjusted[:TOP_N_OUTPUT]

    print(f"  Honeypots detected: {honeypot_count}/{len(top_n)}")

    # ===================================================================
    # STAGE 6: Reasoning Generation
    # ===================================================================
    with Timer("Stage 6: Reasoning Generation"):
        final_candidates = [x[1] for x in final_100]
        final_features = [x[2] for x in final_100]
        final_scores = [x[4] for x in final_100]
        final_ranks = list(range(1, TOP_N_OUTPUT + 1))
        final_hp_explanations = [x[8] for x in final_100]

        reasonings = batch_generate_reasoning(
            final_candidates, final_features,
            final_scores, final_ranks,
            final_hp_explanations, jd,
        )

    # ===================================================================
    # Build output
    # ===================================================================
    output_rows: List[Dict] = []
    for i, (idx, cand, feats, sim, score, components, hp_score, hp_mult, hp_exp) in enumerate(final_100):
        # Round early to 4 decimals so tie-breaking matches CSV output exactly
        rounded_score = round(score, 4)
        
        output_rows.append({
            "candidate_id": cand.get("candidate_id", ""),
            "internal_rank": i + 1,  # Temporary
            "score": rounded_score,
            "reasoning": reasonings[i],
        })

    # Sort strictly by score DESC, then candidate_id ASC
    output_rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    
    # Re-assign ranks based on final sort order
    for i, row in enumerate(output_rows):
        row["rank"] = i + 1

    # Write CSV
    _write_submission_csv(output_rows, output_path)

    total_timer.__exit__(None, None, None)

    # Free remaining memory
    del candidates, all_features
    force_gc()

    return output_rows


# ===================================================================
# Internal: Load + featurize candidates
# ===================================================================

def _load_and_featurize(
    candidates_path: str,
    jd: ParsedJD,
) -> Tuple[List[dict], List[Dict[str, float]], List[str]]:
    """Load all candidates from JSONL and extract features + text.

    Streams the file and processes in chunks to control memory.

    Returns:
        Tuple of (candidates, features, texts).
    """
    candidates: List[dict] = []
    all_features: List[Dict[str, float]] = []
    candidate_texts: List[str] = []

    print(f"  Loading candidates from {candidates_path}...")
    log_memory("before load")

    with open(candidates_path, "r", encoding="utf-8") as f:
        batch: List[dict] = []
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            batch.append(candidate)

            if len(batch) >= CANDIDATE_LOAD_CHUNK_SIZE:
                _process_batch(batch, jd, candidates, all_features, candidate_texts)
                batch = []

                if (line_num + 1) % 20000 == 0:
                    print(f"    Processed {line_num + 1} candidates...")
                    log_memory(f"chunk {line_num + 1}")

        # Process remaining
        if batch:
            _process_batch(batch, jd, candidates, all_features, candidate_texts)

    print(f"  Total loaded: {len(candidates)}")
    return candidates, all_features, candidate_texts


def _process_batch(
    batch: List[dict],
    jd: ParsedJD,
    candidates: List[dict],
    all_features: List[Dict[str, float]],
    candidate_texts: List[str],
) -> None:
    """Process a batch of candidates: extract features and build text."""
    for cand in batch:
        features = extract_all_features(cand, jd)
        text = build_candidate_text(cand)

        candidates.append(cand)
        all_features.append(features)
        candidate_texts.append(text)


# ===================================================================
# Internal: Post-processing
# ===================================================================

def _write_submission_csv(rows: List[Dict], output_path: str) -> None:
    """Write submission CSV in the exact required format."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for row in rows:
            writer.writerow([
                row["candidate_id"],
                row["rank"],
                f"{row['score']:.4f}",
                row["reasoning"],
            ])

    print(f"\n  ✅ Submission written to {output_path}")
    print(f"     Rows: {len(rows)}")
    print(f"     Rank range: {rows[0]['rank']}-{rows[-1]['rank']}")
    print(f"     Score range: {rows[-1]['score']:.4f}-{rows[0]['score']:.4f}")
