#!/usr/bin/env python3
"""
rank.py — Single entry point for the Redrob ranking pipeline.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
    python rank.py  # uses defaults

Produces submission.csv within 5 min on CPU with 16GB RAM.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Fix OpenMP duplicate lib error and threading segfaults on Mac with PyTorch/FAISS/Tokenizers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_pipeline
from src.utils import log_memory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rank.py
  python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
  python rank.py --no-cache  # force re-compute embeddings
        """,
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="data/candidates.jsonl",
        help="Path to candidates JSONL file (default: data/candidates.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/submission.csv",
        help="Output CSV path (default: outputs/submission.csv)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory for cached models/embeddings (default: models/)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding/index caching (re-compute everything)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run official validator after generating submission",
    )

    args = parser.parse_args()

    # Validate input
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: Candidates file not found: {candidates_path}")
        sys.exit(1)

    print("=" * 60)
    print("  REDROB CANDIDATE RANKING PIPELINE")
    print("=" * 60)
    print(f"  Input:     {args.candidates}")
    print(f"  Output:    {args.out}")
    print(f"  Models:    {args.models_dir}")
    print(f"  Cache:     {'disabled' if args.no_cache else 'enabled'}")
    print("=" * 60)

    log_memory("startup")
    start = time.perf_counter()

    # Run pipeline
    results = run_pipeline(
        candidates_path=str(candidates_path),
        output_path=args.out,
        models_dir=args.models_dir,
        use_cache=not args.no_cache,
    )

    elapsed = time.perf_counter() - start
    log_memory("final")

    print()
    print("=" * 60)
    print(f"  COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Output: {args.out}")
    print(f"  Top 5 candidates:")
    for r in results[:5]:
        print(f"    #{r['rank']}: {r['candidate_id']} (score={r['score']:.4f})")
    print("=" * 60)

    # Optional validation
    if args.validate:
        print("\nRunning official validator...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "data/validate_submission.py", args.out],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
