"""
config.py — All constants, skill taxonomies, title tiers, thresholds.
Single source of truth for the entire ranking pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple

# ---------------------------------------------------------------------------
# JD-derived skill taxonomy  (from job_description.docx analysis)
# ---------------------------------------------------------------------------

# Skills the JD says "you absolutely need"
MANDATORY_SKILLS: FrozenSet[str] = frozenset({
    # Embeddings-based retrieval
    "Embeddings", "Sentence Transformers", "Vector Search", "Semantic Search",
    # Vector databases / hybrid search
    "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus", "Elasticsearch",
    "OpenSearch", "pgvector",
    # Python
    "Python",
    # Evaluation frameworks
    "Learning to Rank", "BM25", "Information Retrieval",
})

# Skills the JD says "we'd like you to have"
PREFERRED_SKILLS: FrozenSet[str] = frozenset({
    "Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT",
    "Recommendation Systems",
    "PyTorch", "TensorFlow",
    "Machine Learning", "Deep Learning", "NLP",
    "scikit-learn",
    "Haystack", "LlamaIndex",
    "RAG",
    "Hugging Face Transformers",
    "MLOps", "MLflow", "BentoML", "Kubeflow",
})

# Tier-1: JD-critical retrieval/ranking skills (rare, ~1.3K each in dataset)
TIER1_SKILLS: FrozenSet[str] = frozenset({
    "FAISS", "Milvus", "Qdrant", "Pinecone", "Weaviate",
    "Embeddings", "Sentence Transformers", "Vector Search", "Semantic Search",
    "Learning to Rank", "BM25", "Information Retrieval",
    "Elasticsearch", "OpenSearch", "pgvector",
    "Haystack", "LlamaIndex",
    "Python", "PyTorch", "TensorFlow",
    "NLP", "Machine Learning", "Deep Learning", "scikit-learn",
    "LoRA", "QLoRA", "PEFT",
})

# Tier-2: Supportive AI/ML skills (~4.7K-5.1K each)
TIER2_SKILLS: FrozenSet[str] = frozenset({
    "LLMs", "LangChain", "Hugging Face Transformers",
    "RAG", "Fine-tuning LLMs", "Prompt Engineering",
    "Recommendation Systems", "Information Retrieval",
    "Feature Engineering", "Data Science",
    "MLOps", "MLflow", "BentoML", "Kubeflow",
    "Weights & Biases",
})

# Ultra-rare skills (< 10 occurrences) — strongest organic signal
ULTRA_RARE_SKILLS: FrozenSet[str] = frozenset({
    "Information Retrieval Systems", "Search Backend", "Text Encoders",
    "Vector Representations", "Content Matching", "Model Adaptation",
    "Ranking Systems", "Search & Discovery", "Workflow Orchestration",
    "Search Infrastructure", "Indexing Algorithms",
    "Open-source ML libraries", "Natural Language Processing",
    "Document Processing",
})

# Noise skills (~12K each — appear uniformly across all titles, even HR Managers)
NOISE_SKILLS: FrozenSet[str] = frozenset({
    "HTML", "Databricks", "Redux", "Terraform", "Angular", "Figma",
    "Salesforce CRM", "Vue.js", "Sales", "Accounting", "Agile", "Kafka",
    "Excel", "BigQuery", "CI/CD", "Project Management", "Airflow", "AWS",
    "Flask", "Scrum", "Illustrator", "Kubernetes", "ETL", "CSS", "Docker",
    "Next.js", "Apache Beam", "Java", "Go", "TypeScript", "JavaScript",
    "dbt", "REST APIs", "Spark", "Marketing", "Tally", "GraphQL",
    "Snowflake", "Webpack", "Six Sigma", "SEO", "SAP", "GCP",
    "PostgreSQL", "Rust", "Apache Flink", "gRPC", "Content Writing",
    "SQL", "Hadoop", "Redis", "Tailwind", "Photoshop", "FastAPI",
    "Microservices", "PowerPoint", "Spring Boot", "Data Pipelines",
    "Django", "MongoDB", "Node.js", "Azure", "React",
})

# ---------------------------------------------------------------------------
# Title tiers  (from dataset: 47 unique titles, 100K candidates)
# ---------------------------------------------------------------------------

# Titles with zero relevance to JD
NON_TECH_TITLES: FrozenSet[str] = frozenset({
    "HR Manager", "Marketing Manager", "Accountant", "Sales Executive",
    "Content Writer", "Graphic Designer", "Civil Engineer",
    "Mechanical Engineer", "Customer Support", "Operations Manager",
    "Business Analyst", "Project Manager",
})

# Adjacent tech — could be relevant if career shows ML work
ADJACENT_TECH_TITLES: FrozenSet[str] = frozenset({
    "Software Engineer", "Full Stack Developer", "Cloud Engineer",
    "Java Developer", ".NET Developer", "DevOps Engineer",
    "Mobile Developer", "Frontend Engineer", "QA Engineer",
})

# Data/Analytics — plausible fits
DATA_TITLES: FrozenSet[str] = frozenset({
    "Analytics Engineer", "Data Engineer", "Data Analyst",
    "Backend Engineer", "Senior Data Engineer", "Senior Software Engineer",
})

# ML/AI core — primary candidate pool
ML_CORE_TITLES: FrozenSet[str] = frozenset({
    "ML Engineer", "AI Research Engineer", "Data Scientist",
    "Senior Software Engineer (ML)", "Computer Vision Engineer",
    "Junior ML Engineer", "AI Specialist",
})

# Elite tier — highest signal
ELITE_TITLES: FrozenSet[str] = frozenset({
    "Recommendation Systems Engineer", "Machine Learning Engineer",
    "Applied ML Engineer", "Search Engineer", "AI Engineer",
    "Senior Data Scientist", "NLP Engineer", "Senior NLP Engineer",
    "Senior Machine Learning Engineer", "Staff Machine Learning Engineer",
    "Senior AI Engineer", "Senior Applied Scientist", "Lead AI Engineer",
})

# Title → relevance score mapping
TITLE_RELEVANCE: Dict[str, float] = {}
for t in ELITE_TITLES:
    TITLE_RELEVANCE[t] = 1.0
for t in ML_CORE_TITLES:
    TITLE_RELEVANCE[t] = 0.80
for t in DATA_TITLES:
    TITLE_RELEVANCE[t] = 0.45
for t in ADJACENT_TECH_TITLES:
    TITLE_RELEVANCE[t] = 0.20
for t in NON_TECH_TITLES:
    TITLE_RELEVANCE[t] = 0.0

# Special adjustments within tiers
TITLE_RELEVANCE["Search Engineer"] = 1.0
TITLE_RELEVANCE["Recommendation Systems Engineer"] = 1.0
TITLE_RELEVANCE["NLP Engineer"] = 0.95
TITLE_RELEVANCE["Senior NLP Engineer"] = 0.95
TITLE_RELEVANCE["Computer Vision Engineer"] = 0.55  # JD says no CV-only
TITLE_RELEVANCE["AI Research Engineer"] = 0.65  # JD says no pure research
TITLE_RELEVANCE["Junior ML Engineer"] = 0.70
TITLE_RELEVANCE["AI Specialist"] = 0.70

# ---------------------------------------------------------------------------
# Consulting companies (JD explicit disqualifier)
# ---------------------------------------------------------------------------
CONSULTING_COMPANIES: FrozenSet[str] = frozenset({
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mindtree", "Mphasis", "LTIMindtree",
    "L&T Infotech", "Hexaware",
})

# ---------------------------------------------------------------------------
# Career description keyword sets (for text-based feature extraction)
# ---------------------------------------------------------------------------

RETRIEVAL_KEYWORDS: FrozenSet[str] = frozenset({
    "retrieval", "search", "ranking", "embeddings", "vector",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch", "semantic search", "bm25",
    "information retrieval", "candidate retrieval", "document retrieval",
    "dense retrieval", "hybrid search", "approximate nearest",
    "ann", "hnsw", "ivf", "similarity search", "cosine similarity",
    "sentence-transformers", "sentence transformers", "bi-encoder",
    "cross-encoder", "re-ranking", "reranking",
})

PRODUCTION_KEYWORDS: FrozenSet[str] = frozenset({
    "production", "deployed", "deployment", "serving", "inference",
    "real-time", "latency", "throughput", "scaling", "scale",
    "monitoring", "observability", "a/b test", "ab test",
    "sla", "uptime", "reliability", "mlops", "cicd", "ci/cd",
    "docker", "kubernetes", "microservice", "api", "endpoint",
    "load balancing", "caching", "redis", "kafka",
    "batch processing", "stream processing", "pipeline",
    "users", "customers", "traffic", "requests per second",
    "shipped", "launched", "released",
})

EVALUATION_KEYWORDS: FrozenSet[str] = frozenset({
    "ndcg", "mrr", "map", "precision", "recall", "f1",
    "evaluation", "benchmark", "metric", "offline",
    "a/b test", "ab test", "experiment", "statistical significance",
    "holdout", "cross-validation", "test set",
    "confusion matrix", "roc", "auc",
})

LLM_KEYWORDS: FrozenSet[str] = frozenset({
    "llm", "large language model", "fine-tuning", "fine tuning",
    "lora", "qlora", "peft", "rlhf", "instruction tuning",
    "prompt engineering", "prompt", "gpt", "bert", "transformer",
    "hugging face", "huggingface", "tokenizer",
    "text generation", "language model",
})

ML_KEYWORDS: FrozenSet[str] = frozenset({
    "machine learning", "deep learning", "neural network",
    "model training", "training", "feature engineering",
    "classification", "regression", "clustering",
    "random forest", "xgboost", "lightgbm", "gradient boosting",
    "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "convolutional", "recurrent", "lstm", "attention",
    "nlp", "natural language", "text classification",
    "named entity", "sentiment analysis",
    "recommendation", "collaborative filtering",
    "content-based", "matrix factorization",
})

# Keywords that signal non-relevant career (for honeypot / noise detection)
NON_RELEVANT_CAREER_KEYWORDS: FrozenSet[str] = frozenset({
    "accounting", "financial reporting", "statutory compliance",
    "gaap", "tax filings", "audit", "bookkeeping",
    "brand design", "creative direction", "packaging design",
    "logo", "typography", "adobe suite",
    "mechanical engineering", "cad", "solidworks", "creo",
    "ansys", "fea", "dfm", "dfma", "production tooling",
    "civil engineering", "structural", "construction",
    "sales quota", "arr", "pipeline", "cold calling",
    "hr", "hiring", "onboarding", "payroll", "talent acquisition",
    "content writing", "seo strategy", "editorial calendar",
    "customer support", "tier-1", "tier-2", "tickets",
    "supply chain", "logistics", "warehouse", "fulfillment",
    "marketing", "demand-generation", "paid acquisition",
})

# ---------------------------------------------------------------------------
# Education field relevance
# ---------------------------------------------------------------------------

ML_RELEVANT_FIELDS: FrozenSet[str] = frozenset({
    "Computer Science", "Machine Learning", "Artificial Intelligence",
    "Data Science", "Statistics", "Mathematics", "Applied Mathematics",
    "Information Technology", "Computational Linguistics",
    "Electrical Engineering", "Electronics",
    "Information Systems",
})

# Degree → ordinal value
DEGREE_ORDINAL: Dict[str, float] = {
    "Ph.D": 4.0,
    "M.Tech": 3.5,
    "M.E.": 3.5,
    "M.S.": 3.0,
    "M.Sc": 2.5,
    "MBA": 2.0,
    "B.Tech": 2.0,
    "B.E.": 2.0,
    "B.Sc": 1.5,
    "BCA": 1.5,
    "B.A.": 1.0,
    "Diploma": 0.5,
}

# Tier → score
INSTITUTION_TIER_SCORE: Dict[str, float] = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.50,
    "tier_4": 0.25,
    "unknown": 0.30,
}

# ---------------------------------------------------------------------------
# Scoring weights (Stage 4)
# ---------------------------------------------------------------------------

SCORING_WEIGHTS: Dict[str, float] = {
    "mandatory_skill_coverage":  0.22,
    "semantic_similarity":       0.18,
    "career_relevance":          0.14,
    "title_relevance":           0.10,
    "retrieval_experience":      0.08,
    "production_experience":     0.06,
    "behavioral_score":          0.05,
    "evaluation_experience":     0.05,
    "career_stability":          0.04,
    "github_score":              0.03,
    "startup_exposure":          0.03,
    "education_relevance":       0.02,
}

# ---------------------------------------------------------------------------
# Penalty thresholds
# ---------------------------------------------------------------------------

NOTICE_PERIOD_PENALTIES: List[Tuple[int, float]] = [
    (120, 0.70),   # > 120 days
    (90, 0.85),    # > 90 days
    (60, 0.95),    # > 60 days
    (0, 1.00),     # <= 60 days
]

HONEYPOT_THRESHOLD_HIGH: float = 0.7   # definite honeypot → multiplier 0.0
HONEYPOT_THRESHOLD_MED: float = 0.4    # suspicious → multiplier 0.3
HONEYPOT_MULTIPLIER_HIGH: float = 0.0
HONEYPOT_MULTIPLIER_MED: float = 0.3

INACTIVE_DAYS_THRESHOLD: int = 180  # 6 months
INACTIVE_PENALTY: float = 0.75

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
FAISS_HNSW_M: int = 32
FAISS_HNSW_EF_CONSTRUCTION: int = 200
FAISS_HNSW_EF_SEARCH: int = 64
RETRIEVAL_TOP_K: int = 1000
ENCODING_BATCH_SIZE: int = 64
ENCODING_CHUNK_SIZE: int = 10000
CANDIDATE_LOAD_CHUNK_SIZE: int = 10000

# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------

TOP_N_SCORING: int = 1000    # from retrieval
TOP_N_HONEYPOT: int = 150    # sent to honeypot detector
TOP_N_OUTPUT: int = 100      # final output

# Reference date for computing recency
REFERENCE_DATE: str = "2026-06-15"

# ---------------------------------------------------------------------------
# Parsed JD dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedJD:
    """Structured representation of the job description."""
    title: str
    mandatory_skills: FrozenSet[str]
    preferred_skills: FrozenSet[str]
    negative_patterns: FrozenSet[str]
    experience_min: float
    experience_max: float
    experience_ideal_min: float
    experience_ideal_max: float
    startup_weight: float
    production_weight: float
    shipper_weight: float
    research_penalty: float
    consulting_penalty: float
    langchain_only_penalty: float
    cv_speech_only_penalty: float
    location_preferences: FrozenSet[str]
    notice_period_ideal_max: int
    behavioral_weights: Dict[str, float] = field(default_factory=dict)
    retrieval_keywords: FrozenSet[str] = field(default_factory=frozenset)
    production_keywords: FrozenSet[str] = field(default_factory=frozenset)
    evaluation_keywords: FrozenSet[str] = field(default_factory=frozenset)
    query_text: str = ""
