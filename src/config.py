import os
from pathlib import Path

# ── Root directories ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ── Storage paths ─────────────────────────────────────────────────────────────
CURATED_DATA_DIR   = DATA_DIR / "curated"           # manually kept sample HTMLs
EDGAR_CACHE_DIR    = DATA_DIR / "edgar_cache"        # raw EDGAR .htm downloads
TRAINING_DATA_DIR  = DATA_DIR / "training"           # mined triplet CSVs
DB_PATH            = DATA_DIR / "db" / "metadata.sqlite"
FAISS_INDEX_PATH   = DATA_DIR / "index" / "faiss_index.bin"

# Ensure all directories exist on import
for _p in [CURATED_DATA_DIR, EDGAR_CACHE_DIR, TRAINING_DATA_DIR,
           DB_PATH.parent, FAISS_INDEX_PATH.parent]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Parsing Config ────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # tokens per chunk (matches BGE context window)
CHUNK_OVERLAP = 64    # token overlap so context is not severed at edges

# SEC item titles we care about – used for section-aware chunking labels
SEC_ITEMS_OF_INTEREST = [
    "Item 1",   "Item 1A",  "Item 1B",
    "Item 2",   "Item 3",   "Item 4",
    "Item 5",   "Item 6",
    "Item 7",   "Item 7A",
    "Item 8",   "Item 9",   "Item 9A",
]

# ── Model Config ──────────────────────────────────────────────────────────────
# Default bi-encoder.  To hot-swap to a fine-tuned checkpoint, change this
# value to the local path of the exported model directory.
BASE_ENCODER_MODEL = "BAAI/bge-base-en-v1.5"

RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_MODEL = "google/flan-t5-base"

# ── Training Config ───────────────────────────────────────────────────────────
FINANCEBENCH_DATASET = "PatronusAI/financebench"

# ── Retrieval Config ──────────────────────────────────────────────────────────
RETRIEVAL_TOP_K      = 20   # candidates fetched from FAISS before reranking
RERANK_TOP_K         = 5    # final chunks passed to the generator
EXPECTED_EMBEDDING_DIM = 768

# ── App / API Config ──────────────────────────────────────────────────────────
API_HOST = "127.0.0.1"
API_PORT = 8000
