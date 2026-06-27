"""
retrieval.py — Stage 3: Semantic retrieval with MiniLM + FAISS HNSW.

Encodes 100K candidates in chunks, builds HNSW index, retrieves top-1000.
CPU-only. Persists embeddings and index to disk.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    ENCODING_BATCH_SIZE,
    ENCODING_CHUNK_SIZE,
    FAISS_HNSW_EF_CONSTRUCTION,
    FAISS_HNSW_EF_SEARCH,
    FAISS_HNSW_M,
    RETRIEVAL_TOP_K,
)
from src.utils import Timer, build_candidate_text, force_gc, log_memory


class SemanticRetriever:
    """Handles embedding generation, FAISS index build, and retrieval."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model = None
        self.index = None
        self.embeddings: Optional[np.ndarray] = None
        self._model_dir = model_dir

    def load_model(self) -> None:
        """Load the sentence-transformer model (CPU only)."""
        from sentence_transformers import SentenceTransformer

        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        kwargs = {}
        if self._model_dir and os.path.isdir(self._model_dir):
            model_path = self._model_dir
        else:
            model_path = EMBEDDING_MODEL
        self.model = SentenceTransformer(model_path, device="cpu")
        log_memory("model loaded")

    def encode_candidates(
        self,
        candidate_texts: List[str],
        cache_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Encode candidate texts into embeddings.

        Args:
            candidate_texts: List of text strings (one per candidate).
            cache_path: If provided, save/load embeddings from this file.

        Returns:
            NumPy array of shape (N, EMBEDDING_DIM).
        """
        # Try cache first
        if cache_path and cache_path.exists():
            print(f"  Loading cached embeddings from {cache_path}")
            self.embeddings = np.load(str(cache_path))
            
            cached_shape = self.embeddings.shape
            expected_shape = (len(candidate_texts), EMBEDDING_DIM)
            
            if cached_shape == expected_shape:
                print(f"  Cached embeddings: {cached_shape}")
                print(f"  Expected shape: {expected_shape}")
                print("  Using cached embeddings")
                return self.embeddings
                
            print(f"  Cached embeddings: {cached_shape}")
            print(f"  Expected shape: {expected_shape}")
            print("  Cache shape mismatch, re-encoding...")

        if self.model is None:
            self.load_model()

        n = len(candidate_texts)
        print(f"  Encoding {n} candidates (batch_size={ENCODING_BATCH_SIZE})...")
        log_memory("before encoding")

        # Encode in chunks to control memory
        all_embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
        chunk_size = ENCODING_CHUNK_SIZE

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = candidate_texts[start:end]

            with Timer(f"encode chunk {start}-{end}"):
                chunk_emb = self.model.encode(
                    chunk,
                    batch_size=ENCODING_BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # L2-normalize for cosine sim
                )

            all_embeddings[start:end] = chunk_emb
            del chunk_emb
            force_gc()

        self.embeddings = all_embeddings
        log_memory("after encoding")

        # Cache to disk
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), all_embeddings)
            print(f"  Saved embeddings to {cache_path}")

        return all_embeddings

    def build_index(
        self,
        embeddings: Optional[np.ndarray] = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        """Build FAISS HNSW index.

        Args:
            embeddings: Array of shape (N, EMBEDDING_DIM). Uses self.embeddings if None.
            cache_path: If provided, save/load index from this file.
        """
        import faiss

        if embeddings is None:
            embeddings = self.embeddings
        if embeddings is None:
            raise ValueError("No embeddings available. Call encode_candidates first.")

        # Try cache
        if cache_path and cache_path.exists():
            print(f"  Loading cached FAISS index from {cache_path}")
            self.index = faiss.read_index(str(cache_path))
            return

        n, d = embeddings.shape
        print(f"  Building FAISS HNSW index: {n} vectors, dim={d}, M={FAISS_HNSW_M}")

        with Timer("FAISS index build"):
            index = faiss.IndexHNSWFlat(d, FAISS_HNSW_M)
            index.hnsw.efConstruction = FAISS_HNSW_EF_CONSTRUCTION
            index.add(embeddings)

        index.hnsw.efSearch = FAISS_HNSW_EF_SEARCH
        self.index = index

        log_memory("after index build")

        # Cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(cache_path))
            print(f"  Saved FAISS index to {cache_path}")

    def retrieve(
        self,
        query_text: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve top-k candidates for a query.

        Args:
            query_text: The JD query string.
            top_k: Number of candidates to retrieve.

        Returns:
            Tuple of (indices, distances) arrays, each of shape (top_k,).
            distances are L2 distances (lower = more similar for normalized vecs).
        """
        if self.model is None:
            self.load_model()
        if self.index is None:
            raise ValueError("No FAISS index. Call build_index first.")

        # Encode query
        query_emb = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Search
        distances, indices = self.index.search(query_emb, top_k)

        return indices[0], distances[0]

    def get_similarity_scores(
        self,
        query_text: str,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        """Get cosine similarity scores for specific candidates.

        Since embeddings are L2-normalized, cosine_sim = 1 - (L2_dist^2 / 2).
        But since FAISS HNSW uses inner product for normalized vectors,
        we compute dot product directly.

        Args:
            query_text: JD query string.
            candidate_indices: Indices of candidates to score.

        Returns:
            Array of cosine similarities in [0, 1].
        """
        if self.model is None:
            self.load_model()

        query_emb = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        candidate_embs = self.embeddings[candidate_indices]
        similarities = np.dot(candidate_embs, query_emb.T).flatten()

        # Clamp to [0, 1]
        similarities = np.clip(similarities, 0.0, 1.0)
        return similarities

    def unload_model(self) -> None:
        """Free the sentence-transformer model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            force_gc()
            log_memory("after model unload")
