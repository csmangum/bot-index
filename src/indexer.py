"""
src/indexer.py - Indexes parsed A360 bot chunks into ChromaDB.

Workflow:
  1. Load the embedding model (once, cached in utils).
  2. Open (or create) a persistent Chroma collection.
  3. Embed chunk texts in batches.
  4. Upsert documents into Chroma (idempotent by deterministic ID).

The indexer also supports:
  - Re-indexing only bots whose source file has changed (via mtime tracking).
  - Clearing the entire collection and re-indexing from scratch.

Public API::

    index_bots(bots_dir, force_reindex=False)
    index_chunks(chunks)
    get_collection()
    clear_collection()
    collection_stats() -> dict
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------


def _get_chroma_client(db_path: Path):
    """Return a persistent ChromaDB client for *db_path*.

    Args:
        db_path: Directory where ChromaDB will store its files.

    Returns:
        A ``chromadb.PersistentClient`` instance.
    """
    try:
        import chromadb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "chromadb is not installed. Run: pip install chromadb"
        ) from exc

    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def _collection_metadata(distance_metric: str, schema_version: str) -> dict:
    """Build standard metadata for the Chroma collection."""
    return {
        "hnsw:space": distance_metric,
        "index_schema_version": schema_version,
    }


def _ensure_collection_schema(
    db_path: Path, collection_name: str, force_reindex: bool = False
) -> None:
    """Ensure collection metadata matches current index schema version.

    If the on-disk collection schema is outdated and contains data, this
    function requires ``force_reindex=True`` to avoid silent duplicate docs
    when deterministic ID strategy changes.
    """
    from config import CHROMA_DISTANCE_METRIC, INDEX_SCHEMA_VERSION

    client = _get_chroma_client(db_path)
    expected_metadata = _collection_metadata(CHROMA_DISTANCE_METRIC, INDEX_SCHEMA_VERSION)

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        client.create_collection(name=collection_name, metadata=expected_metadata)
        _log.info("Created collection '%s' with schema version %s", collection_name, INDEX_SCHEMA_VERSION)
        return

    actual_metadata = collection.metadata or {}
    actual_version = actual_metadata.get("index_schema_version")
    if actual_version == INDEX_SCHEMA_VERSION:
        return

    count = collection.count()
    if count > 0 and not force_reindex:
        raise RuntimeError(
            "Index schema version mismatch for collection "
            f"'{collection_name}': expected {INDEX_SCHEMA_VERSION!r}, "
            f"found {actual_version!r}. Run 'python main.py index --force' "
            "to rebuild the index with the current ID schema."
        )

    client.delete_collection(collection_name)
    client.create_collection(name=collection_name, metadata=expected_metadata)
    _log.info(
        "Recreated collection '%s' with schema version %s",
        collection_name,
        INDEX_SCHEMA_VERSION,
    )


def get_collection(db_path: Optional[Path] = None, collection_name: Optional[str] = None):
    """Get (or create) the Chroma collection.

    Args:
        db_path:         Override the default ChromaDB directory.
        collection_name: Override the default collection name.

    Returns:
        A ``chromadb.Collection`` instance.
    """
    from config import CHROMA_COLLECTION, CHROMA_DB_DIR, CHROMA_DISTANCE_METRIC, INDEX_SCHEMA_VERSION

    db_path = db_path or CHROMA_DB_DIR
    collection_name = collection_name or CHROMA_COLLECTION

    client = _get_chroma_client(db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata=_collection_metadata(CHROMA_DISTANCE_METRIC, INDEX_SCHEMA_VERSION),
    )
    return collection


def clear_collection(db_path: Optional[Path] = None, collection_name: Optional[str] = None) -> None:
    """Delete all documents from the collection (but keep the collection itself).

    Args:
        db_path:         Override the default ChromaDB directory.
        collection_name: Override the default collection name.
    """
    from config import CHROMA_COLLECTION, CHROMA_DB_DIR, CHROMA_DISTANCE_METRIC, INDEX_SCHEMA_VERSION

    db_path = db_path or CHROMA_DB_DIR
    collection_name = collection_name or CHROMA_COLLECTION

    client = _get_chroma_client(db_path)
    # Delete then recreate to guarantee empty state
    try:
        client.delete_collection(collection_name)
        _log.info("Deleted existing collection '%s'", collection_name)
    except Exception as del_exc:  # collection may not exist yet
        _log.debug("Could not delete collection '%s': %s", collection_name, del_exc)

    client.create_collection(
        name=collection_name,
        metadata=_collection_metadata(CHROMA_DISTANCE_METRIC, INDEX_SCHEMA_VERSION),
    )
    _log.info("Created fresh collection '%s'", collection_name)


# ---------------------------------------------------------------------------
# Core indexing logic
# ---------------------------------------------------------------------------

_BATCH_SIZE = 64  # number of chunks to embed + upsert per batch


def index_chunks(
    chunks: List[dict],
    db_path: Optional[Path] = None,
    collection_name: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> int:
    """Embed and upsert *chunks* into Chroma.

    Uses deterministic IDs so repeated calls are safe (upsert semantics).

    Args:
        chunks:          List of chunk dicts from ``parser.parse_bot_file``.
        db_path:         Override ChromaDB directory.
        collection_name: Override collection name.
        model_name:      Override embedding model.
        device:          Override torch device.

    Returns:
        Number of chunks successfully indexed.
    """
    from config import CHROMA_COLLECTION, CHROMA_DB_DIR, EMBEDDING_DEVICE, EMBEDDING_MODEL
    from src.utils import embed_texts

    db_path = db_path or CHROMA_DB_DIR
    collection_name = collection_name or CHROMA_COLLECTION
    model_name = model_name or EMBEDDING_MODEL
    device = device or EMBEDDING_DEVICE

    if not chunks:
        _log.warning("index_chunks called with 0 chunks – nothing to do.")
        return 0

    collection = get_collection(db_path, collection_name)
    total_indexed = 0
    t0 = time.time()

    for batch_start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[batch_start : batch_start + _BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = embed_texts(texts, model_name=model_name, device=device)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_indexed += len(batch)
        _log.debug(
            "Upserted batch %d–%d (%d chunks)",
            batch_start + 1,
            batch_start + len(batch),
            len(batch),
        )

    elapsed = time.time() - t0
    _log.info(
        "Indexed %d chunk(s) in %.1f s (%.0f chunks/s)",
        total_indexed,
        elapsed,
        total_indexed / max(elapsed, 0.001),
    )
    return total_indexed


def index_bots(
    bots_dir: Optional[Path] = None,
    force_reindex: bool = False,
    db_path: Optional[Path] = None,
    collection_name: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, int]:
    """Parse + index all bots found under *bots_dir*.

    Args:
        bots_dir:        Directory containing ``.json`` bot files (may have sub-folders).
        force_reindex:   If ``True``, clears the collection first; otherwise upserts.
        db_path:         Override ChromaDB directory.
        collection_name: Override collection name.
        model_name:      Override embedding model.
        device:          Override torch device.

    Returns:
        A dict mapping ``bot_name → chunks_indexed``.
    """
    from config import BOTS_DIR, CHROMA_COLLECTION, CHROMA_DB_DIR
    from src.parser import parse_bot_file
    from src.utils import find_bot_files

    bots_dir = bots_dir or BOTS_DIR
    db_path = db_path or CHROMA_DB_DIR
    collection_name = collection_name or CHROMA_COLLECTION

    if force_reindex:
        _log.info("force_reindex=True – clearing existing collection …")
        clear_collection(db_path, collection_name)
    else:
        _ensure_collection_schema(
            db_path=db_path,
            collection_name=collection_name,
            force_reindex=False,
        )

    bot_files = find_bot_files(bots_dir)
    if not bot_files:
        _log.warning("No bot .json files found in '%s'.", bots_dir)
        return {}

    results: Dict[str, int] = {}

    for bf in bot_files:
        _log.info("Processing bot file: %s", bf.name)
        try:
            chunks = parse_bot_file(bf)
        except ValueError as exc:
            _log.error("Skipping '%s': %s", bf.name, exc)
            continue

        if not chunks:
            _log.warning("'%s' produced 0 chunks – skipping.", bf.name)
            continue

        bot_name = chunks[0]["metadata"].get("bot_name", bf.stem)
        n = index_chunks(chunks, db_path=db_path, collection_name=collection_name,
                         model_name=model_name, device=device)
        results[bot_name] = n

    total = sum(results.values())
    _log.info("Indexing complete: %d bot(s), %d chunk(s) total.", len(results), total)
    return results


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def collection_stats(
    db_path: Optional[Path] = None, collection_name: Optional[str] = None
) -> dict:
    """Return basic statistics about the current Chroma collection.

    Returns:
        Dict with keys: ``collection_name``, ``total_chunks``,
        ``unique_bots``, ``chunk_types``.
    """
    collection = get_collection(db_path, collection_name)
    count = collection.count()
    if count == 0:
        return {
            "collection_name": collection.name,
            "total_chunks": 0,
            "unique_bots": 0,
            "chunk_types": {},
        }

    # Fetch all metadata (no embeddings needed)
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    bot_names = set()
    chunk_types: Dict[str, int] = {}
    for meta in metadatas:
        if not meta:
            continue
        bn = meta.get("bot_name", "unknown")
        ct = meta.get("chunk_type", "unknown")
        bot_names.add(bn)
        chunk_types[ct] = chunk_types.get(ct, 0) + 1

    return {
        "collection_name": collection.name,
        "total_chunks": count,
        "unique_bots": len(bot_names),
        "chunk_types": chunk_types,
    }
