#!/usr/bin/env python3
"""
profile_dataset.py — Print dataset statistics.
"""
import sys
import json
import collections
from pathlib import Path

def main():
    candidates_path = "data/candidates.jsonl"
    if not Path(candidates_path).exists():
        print(f"Error: {candidates_path} not found.")
        sys.exit(1)
        
    titles = collections.Counter()
    skills = collections.Counter()
    total = 0
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            cand = json.loads(line)
            titles[cand.get("profile", {}).get("current_title", "")] += 1
            for s in cand.get("skills", []):
                skills[s.get("name", "")] += 1
                
    print("=" * 40)
    print(f"Total candidates: {total}")
    print(f"Unique titles: {len(titles)}")
    print(f"Unique skills: {len(skills)}")
    print("=" * 40)
    print("\nTop 10 Titles:")
    for t, c in titles.most_common(10):
        print(f"  {t}: {c}")

if __name__ == "__main__":
    main()
