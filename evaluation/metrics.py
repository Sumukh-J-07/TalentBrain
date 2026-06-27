"""
metrics.py — NDCG@k, MAP, Precision@k evaluation metrics.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np


def dcg_at_k(relevances: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at position k.

    Args:
        relevances: List of relevance scores (higher = more relevant).
        k: Cutoff position.

    Returns:
        DCG@k value.
    """
    relevances = relevances[:k]
    return sum(
        rel / math.log2(i + 2)  # i+2 because i is 0-indexed, log2(1) = 0
        for i, rel in enumerate(relevances)
    )


def ndcg_at_k(
    predicted_relevances: List[float],
    ideal_relevances: List[float],
    k: int,
) -> float:
    """Compute Normalized DCG at position k.

    Args:
        predicted_relevances: Relevance scores in predicted order.
        ideal_relevances: All relevance scores (will be sorted for IDCG).
        k: Cutoff position.

    Returns:
        NDCG@k in [0, 1].
    """
    dcg = dcg_at_k(predicted_relevances, k)
    ideal_sorted = sorted(ideal_relevances, reverse=True)
    idcg = dcg_at_k(ideal_sorted, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_average_precision(
    predicted_ids: List[str],
    relevant_ids: set,
) -> float:
    """Compute Mean Average Precision.

    Args:
        predicted_ids: Ordered list of predicted candidate IDs.
        relevant_ids: Set of relevant candidate IDs.

    Returns:
        MAP value.
    """
    if not relevant_ids:
        return 0.0

    hits = 0
    sum_precision = 0.0

    for i, cid in enumerate(predicted_ids):
        if cid in relevant_ids:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precision += precision_at_i

    return sum_precision / len(relevant_ids)


def precision_at_k(
    predicted_ids: List[str],
    relevant_ids: set,
    k: int,
) -> float:
    """Compute Precision at position k.

    Args:
        predicted_ids: Ordered list of predicted candidate IDs.
        relevant_ids: Set of relevant candidate IDs.
        k: Cutoff position.

    Returns:
        P@k in [0, 1].
    """
    top_k = predicted_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k if k > 0 else 0.0


def composite_score(
    ndcg_10: float,
    ndcg_50: float,
    map_score: float,
    p_10: float,
) -> float:
    """Compute the hackathon composite score.

    Formula from submission_spec.docx:
    Final = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10

    Args:
        ndcg_10: NDCG at k=10.
        ndcg_50: NDCG at k=50.
        map_score: Mean Average Precision.
        p_10: Precision at k=10.

    Returns:
        Composite score.
    """
    return (
        0.50 * ndcg_10
        + 0.30 * ndcg_50
        + 0.15 * map_score
        + 0.05 * p_10
    )
