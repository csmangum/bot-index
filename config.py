"""
config.py - Central configuration for the A360 Bot RAG system.

Adjust these settings to tune the embedding model, storage paths,
and retrieval behaviour without touching any other module.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the project (directory containing this file)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Folder where exported A360 bot JSON files live (may contain sub-folders)
BOTS_DIR = PROJECT_ROOT / "bots"

# Persistent ChromaDB storage location
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# Example bots shipped with the repository (used for demos / tests)
EXAMPLE_BOTS_DIR = PROJECT_ROOT / "example_bots"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

# Sentence-transformers model name.
# Drop-in alternatives:
#   "BAAI/bge-small-en-v1.5"  – slightly better quality, same size
#   "BAAI/bge-base-en-v1.5"   – larger, higher quality
#   "flax-sentence-embeddings/st-codesearch-distilroberta-base"  – code-aware
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Device for inference: "cpu", "cuda", or "mps" (Apple Silicon)
EMBEDDING_DEVICE = "cpu"

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

# Name of the Chroma collection that holds bot chunks
CHROMA_COLLECTION = "a360_bots"

# Distance metric used by Chroma (cosine is recommended for sentence-transformers)
CHROMA_DISTANCE_METRIC = "cosine"

# ---------------------------------------------------------------------------
# Chunking / parsing
# ---------------------------------------------------------------------------

# Maximum recursion depth when traversing nested commands/children.
# Increase if your bots have deeply nested conditional/loop blocks.
MAX_PARSE_DEPTH = 20

# Context overlap: include this many parent-level tokens in each child chunk
# (expressed as a fraction of the parent text length, approximate).
CONTEXT_OVERLAP_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Search & retrieval
# ---------------------------------------------------------------------------

# Default number of vector-search results to return before graph expansion
DEFAULT_TOP_K = 10

# When --ui-elements are provided, search for each element with this many results.
# Uses max(UI_ELEMENT_MIN_TOP_K, DEFAULT_TOP_K // UI_ELEMENT_TOP_K_DIVISOR).
UI_ELEMENT_MIN_TOP_K = 5
UI_ELEMENT_TOP_K_DIVISOR = 2

# Maximum graph hops when computing transitive callers.
# Set to a large number (e.g. 999) for effectively unlimited traversal.
MAX_GRAPH_HOPS = 5

# Minimum cosine similarity score (0–1) to include a result.
# Chroma returns *distance* (lower = more similar for cosine), so threshold
# here is expressed as max distance  (1 − similarity).
MAX_DISTANCE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"  # DEBUG | INFO | WARNING | ERROR
