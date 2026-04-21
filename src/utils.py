"""
src/utils.py - Shared helpers for the A360 Bot RAG system.

Responsibilities:
  - Load and cache the sentence-transformers embedding model.
  - Provide a thin embed() helper that normalises inputs.
  - Configure project-wide logging.
  - Miscellaneous path / string helpers.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    """Initialise root logger with a human-readable format.

    Call this once at application startup (main.py does it automatically).

    Args:
        level: One of "DEBUG", "INFO", "WARNING", "ERROR".
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding model loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str, device: str = "cpu"):
    """Load and cache a sentence-transformers SentenceTransformer model.

    The model is loaded *once* per process and reused for all subsequent
    calls thanks to ``@lru_cache``.

    Args:
        model_name: HuggingFace model identifier, e.g. ``"all-MiniLM-L6-v2"``.
        device:     Torch device string: ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns:
        A ``sentence_transformers.SentenceTransformer`` instance.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from exc

    _log.info("Loading embedding model '%s' on device '%s' …", model_name, device)
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    _log.info("Model loaded in %.1f s", time.time() - t0)
    return model


def embed_texts(texts: List[str], model_name: str, device: str = "cpu") -> List[List[float]]:
    """Embed a list of strings and return plain Python float lists.

    Args:
        texts:      Strings to embed (must be non-empty).
        model_name: Sentence-transformers model identifier.
        device:     Torch device string.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []
    model = get_embedding_model(model_name, device)
    # encode returns a numpy ndarray; convert to nested Python lists for Chroma
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Deterministic document IDs
# ---------------------------------------------------------------------------

def make_doc_id(bot_name: str, chunk_type: str, sequence_path: str) -> str:
    """Create a deterministic ID for a Chroma document using SHA-256.

    Using a content hash guarantees idempotent re-indexing: the same chunk
    always gets the same ID so Chroma can upsert (overwrite) cleanly.

    Args:
        bot_name:      Name of the parent bot.
        chunk_type:    One of ``"bot_summary"``, ``"block"``, ``"action"``.
        sequence_path: A string that uniquely identifies the position within
                       the bot, e.g. ``"cmd-002-05"`` or ``"cmd-001-03/cmd-001-03-02"``.

    Returns:
        A 32-character hex digest string.
    """
    raw = f"{bot_name}::{chunk_type}::{sequence_path}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Collapse excessive whitespace and strip leading/trailing blanks."""
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, max_chars: int = 500) -> str:
    """Return the first *max_chars* characters of *text*, with an ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"


def find_bot_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Return all ``.json`` files under *directory*.

    Args:
        directory: The folder to search.
        recursive: If ``True`` (default), also descend into sub-folders.

    Returns:
        Sorted list of ``Path`` objects.
    """
    if not directory.exists():
        _log.warning("Bot directory does not exist: %s", directory)
        return []
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(directory.glob(pattern))


def format_metadata_for_display(metadata: dict) -> str:
    """Render a metadata dict as a compact, human-readable string.

    Args:
        metadata: A flat dict of string key/value pairs (Chroma metadata).

    Returns:
        A multi-line string for CLI display.
    """
    lines = []
    for key, value in sorted(metadata.items()):
        if value:  # skip empty/None values
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)
