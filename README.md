# TalentBrain

TalentBrain is a high-performance candidate ranking engine built for the Redrob Candidate Discovery Hackathon. 

The objective of the system is to accurately process a massive dataset of 100,000 candidate profiles, score their relevance against a complex Job Description (specifically targeted at Staff Machine Learning/Search Engineers), and output the top 100 candidates. The system is designed to run entirely on CPU within severe computational constraints (under 5 minutes execution time and under 16GB of RAM). By using a multi-stage funnel approach, the current pipeline consistently executes in under 70 seconds with a peak memory footprint of approximately 2.4GB.

## Architecture

The ranking pipeline is entirely deterministic and relies on local models to bypass the latency of external API calls during inference. The architecture is broken down into the following stages:

1. Job Description Parser
Extracts mandatory skills, preferred skills, negative signals, and behavioral weights directly from the target JD.

2. Feature Engineering
Extracts over 75 dense features from the raw JSON candidate profiles. Features span across several categories including production exposure, retrieval skill coverage, behavioral signals (e.g., GitHub activity, recruiter response rate), and career stability.

3. Heuristic Prefilter
To meet aggressive runtime constraints, computing 100,000 dense embeddings on CPU is not feasible. The prefilter stage scores all candidates using fast heuristics (title relevance, mandatory skill count, and profile completeness) and passes only the top 5,000 candidates to the embedding layer.

4. Semantic Retrieval
Leverages `all-MiniLM-L6-v2` and FAISS HNSW indexing to generate embeddings for the 5,000 prefiltered candidates. It calculates cosine similarity against the parsed Job Description to retrieve the top 1,000 candidates. To further optimize consecutive runs, exact-shape matching is used for embedding caching.

5. Hybrid Scorer
Executes a weighted combination of the semantic similarity score and the 75 engineered features. The scorer applies heavy multiplier penalties for non-ideal signals (e.g., keyword stuffing honeypots, exclusively consulting backgrounds, or candidates lacking the required years of experience).

## Project Structure

* `data/` - Contains schemas, the target job description, and evaluation scripts. (Note: The main candidates dataset is excluded due to size constraints).
* `src/` - Core pipeline modules including the feature builder, JD parser, semantic retriever, and hybrid scorer.
* `models/` - Cache directory for FAISS indices and numpy embeddings.
* `outputs/` - Destination for the final ranked `submission.csv`.
* `evaluation/` - Benchmarking tools to test runtime and memory constraints.
* `rank.py` - The main entrypoint for the pipeline.

## Getting Started

Follow these steps to set up the environment and run the ranking pipeline on your local machine.

### Prerequisites

Ensure you have Python 3.9+ installed.

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Sumukh-J-07/TalentBrain.git
cd TalentBrain
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Dataset:
Because the `candidates.jsonl` dataset is extremely large (over 450MB), it is not hosted in this repository. 
Download `candidates.jsonl` from the official Redrob Hackathon platform and place it directly into the `data/` directory.

### Running the Ranker

To execute the full pipeline and generate the submission file, run:

```bash
python3 rank.py --candidates data/candidates.jsonl --out outputs/submission.csv
```

The script will output progress logs, memory usage at each stage, and completion time. The final ranked candidates will be available in `outputs/submission.csv`.

### Benchmarking

If you want to verify that your environment meets the strict computational constraints, you can run the benchmark script:

```bash
python3 evaluation/benchmark.py
```
