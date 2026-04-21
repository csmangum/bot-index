"""
src/search.py - Semantic search and regression-test recommendation.

Workflow:
  1. Embed the query string.
  2. Run a cosine-similarity search in Chroma (optionally filtered by
     metadata predicates such as action_type or window_title).
  3. Extract the unique set of bots from the top-K results.
  4. Optionally expand via the call graph to surface transitive callers.
  5. Return a ranked, evidence-rich recommendation list.

Public API::

    semantic_search(query, top_k, where, ...)  -> List[SearchResult]
    recommend_for_regression(query, ui_elements, top_k, ...) -> RegressionReport
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Single result from a semantic / metadata search."""

    doc_id: str
    bot_name: str
    chunk_type: str
    text: str
    distance: float                  # 0 = identical, 1 = orthogonal
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def similarity(self) -> float:
        """Cosine similarity (1 − distance), clamped to [0, 1]."""
        return max(0.0, min(1.0, 1.0 - self.distance))


@dataclass
class BotRecommendation:
    """A single bot recommended for regression testing."""

    bot_name: str
    score: float                     # higher = more likely impacted
    evidence: List[str]              # human-readable reasons
    matching_selectors: List[str]
    matching_windows: List[str]
    dependency_path: Optional[str]   # e.g. "Main_Invoice → LoginToCustomerPortal"
    is_direct_match: bool            # True if found by vector search


@dataclass
class RegressionReport:
    """Full regression recommendation report."""

    query: str
    ui_elements: List[str]
    recommendations: List[BotRecommendation]
    total_bots_evaluated: int
    search_results: List[SearchResult]


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------


def semantic_search(
    query: str,
    top_k: Optional[int] = None,
    where: Optional[dict] = None,
    collection=None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    max_distance: Optional[float] = None,
) -> List[SearchResult]:
    """Run a semantic (vector + optional metadata) search against the Chroma collection.

    Args:
        query:        Natural-language query string.
        top_k:        Maximum results to return (default from config).
        where:        Optional Chroma ``where`` filter dict for metadata predicates.
                      Example: ``{"action_type": {"$eq": "ActionNode"}}``.
        collection:   Pre-fetched Chroma collection (fetched if ``None``).
        model_name:   Override embedding model.
        device:       Override torch device.
        max_distance: Only return results with distance ≤ this value.

    Returns:
        List of :class:`SearchResult` objects sorted by ascending distance
        (most similar first).
    """
    from config import DEFAULT_TOP_K, EMBEDDING_DEVICE, EMBEDDING_MODEL, MAX_DISTANCE_THRESHOLD
    from src.utils import embed_texts

    top_k = top_k or DEFAULT_TOP_K
    model_name = model_name or EMBEDDING_MODEL
    device = device or EMBEDDING_DEVICE
    max_distance = max_distance if max_distance is not None else MAX_DISTANCE_THRESHOLD

    if collection is None:
        from src.indexer import get_collection
        collection = get_collection()

    if collection.count() == 0:
        _log.warning("Collection is empty – please run 'index' first.")
        return []

    query_embedding = embed_texts([query], model_name=model_name, device=device)[0]

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    raw = collection.query(**query_kwargs)

    results: List[SearchResult] = []
    ids_list = raw.get("ids", [[]])[0]
    docs_list = raw.get("documents", [[]])[0]
    metas_list = raw.get("metadatas", [[]])[0]
    dists_list = raw.get("distances", [[]])[0]

    for doc_id, doc_text, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
        if dist > max_distance:
            continue
        results.append(
            SearchResult(
                doc_id=doc_id,
                bot_name=meta.get("bot_name", "unknown"),
                chunk_type=meta.get("chunk_type", "unknown"),
                text=doc_text or "",
                distance=dist,
                metadata=meta or {},
            )
        )

    _log.info("Semantic search for '%s': %d result(s) (top_k=%d)", query, len(results), top_k)
    return results


# ---------------------------------------------------------------------------
# Regression recommendation
# ---------------------------------------------------------------------------


def recommend_for_regression(
    query: str,
    ui_elements: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    use_graph: bool = True,
    collection=None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    max_distance: Optional[float] = None,
) -> RegressionReport:
    """Recommend bots for regression testing given a change description.

    Algorithm:
      1. Semantic search with the natural-language *query*.
      2. If *ui_elements* are provided, also search for each element and merge.
      3. Collect directly matched bots + their evidence (selectors, windows).
      4. Expand via graph: include transitive callers of matched bots.
      5. Score and rank all candidates.

    Args:
        query:        Description of the UI / logic change, e.g.
                      ``"login button selector changed to #new-login-btn"``.
        ui_elements:  Optional list of specific selectors or window titles to
                      search for, e.g. ``["button#login-submit", "Customer Portal - Login"]``.
        top_k:        Number of vector-search results per query.
        use_graph:    If ``True``, expand results via the call graph.
        collection:   Pre-fetched Chroma collection.
        model_name:   Override embedding model.
        device:       Override torch device.
        max_distance: Maximum cosine distance to include.

    Returns:
        A :class:`RegressionReport` with prioritised bot recommendations.
    """
    from config import DEFAULT_TOP_K, UI_ELEMENT_MIN_TOP_K, UI_ELEMENT_TOP_K_DIVISOR

    top_k = top_k or DEFAULT_TOP_K
    ui_elements = ui_elements or []

    if collection is None:
        from src.indexer import get_collection
        collection = get_collection()

    # ── Step 1: Vector search on the main query ──────────────────────────
    all_search_results: List[SearchResult] = semantic_search(
        query,
        top_k=top_k,
        collection=collection,
        model_name=model_name,
        device=device,
        max_distance=max_distance,
    )

    # ── Step 2: Additional searches for explicit UI elements ─────────────
    for elem in ui_elements:
        elem_results = semantic_search(
            elem,
            top_k=max(UI_ELEMENT_MIN_TOP_K, top_k // UI_ELEMENT_TOP_K_DIVISOR),
            collection=collection,
            model_name=model_name,
            device=device,
            max_distance=max_distance,
        )
        # Merge: keep the better (lower) distance per doc_id
        existing_ids = {r.doc_id for r in all_search_results}
        for r in elem_results:
            if r.doc_id not in existing_ids:
                all_search_results.append(r)
                existing_ids.add(r.doc_id)

    # ── Step 3: Build direct bot evidence map ───────────────────────────
    bot_evidence: Dict[str, dict] = {}  # bot_name → {score, selectors, windows, evidence}

    for result in all_search_results:
        bn = result.bot_name
        if bn not in bot_evidence:
            bot_evidence[bn] = {
                "score": 0.0,
                "selectors": [],
                "windows": [],
                "evidence": [],
            }
        entry = bot_evidence[bn]
        # Score: accumulate inverse distance (higher = more relevant)
        entry["score"] += result.similarity

        sel = result.metadata.get("selector", "")
        win = result.metadata.get("window_title", "")
        if sel and sel not in entry["selectors"]:
            entry["selectors"].append(sel)
        if win and win not in entry["windows"]:
            entry["windows"].append(win)

        # Build human-readable evidence line
        evid = f"[{result.chunk_type}] similarity={result.similarity:.2f}"
        action_type = result.metadata.get("action_type", "")
        if action_type:
            evid += f" | action={action_type}"
        if sel:
            evid += f" | selector={sel}"
        if win:
            evid += f" | window='{win}'"
        entry["evidence"].append(evid)

    # ── Step 4: Graph expansion ─────────────────────────────────────────
    graph_additions: Dict[str, dict] = {}  # bot_name → {score, path, hops}

    if use_graph and bot_evidence:
        try:
            from src.graph import build_call_graph, get_transitive_callers

            call_graph = build_call_graph(collection=collection)

            for directly_matched_bot in list(bot_evidence.keys()):
                callers = get_transitive_callers(call_graph, directly_matched_bot)
                for caller_name, hop_distance in callers.items():
                    assert hop_distance > 0, "BFS hop_distance must be a positive integer"
                    if caller_name in bot_evidence:
                        # Already a direct hit – boost score instead
                        bot_evidence[caller_name]["score"] += 1.0 / hop_distance
                        bot_evidence[caller_name]["evidence"].append(
                            f"[graph] transitive caller of '{directly_matched_bot}' "
                            f"(hop={hop_distance})"
                        )
                    else:
                        existing = graph_additions.get(caller_name)
                        if existing is None or hop_distance < existing["hops"]:
                            graph_additions[caller_name] = {
                                "score": 1.0 / hop_distance,
                                "path": f"{caller_name} → {directly_matched_bot}",
                                "hops": hop_distance,
                            }
        except Exception as exc:
            _log.warning("Graph expansion failed: %s", exc)

    # ── Step 5: Build final recommendation list ──────────────────────────
    recommendations: List[BotRecommendation] = []

    # Direct matches
    for bot_name, ev in bot_evidence.items():
        recommendations.append(
            BotRecommendation(
                bot_name=bot_name,
                score=ev["score"],
                evidence=ev["evidence"],
                matching_selectors=ev["selectors"],
                matching_windows=ev["windows"],
                dependency_path=None,
                is_direct_match=True,
            )
        )

    # Graph-expanded bots (not already in direct matches)
    for bot_name, gdata in graph_additions.items():
        recommendations.append(
            BotRecommendation(
                bot_name=bot_name,
                score=gdata["score"],
                evidence=[f"[graph] caller via dependency path: {gdata['path']}"],
                matching_selectors=[],
                matching_windows=[],
                dependency_path=gdata["path"],
                is_direct_match=False,
            )
        )

    # Sort: direct matches first, then by descending score
    recommendations.sort(key=lambda r: (not r.is_direct_match, -r.score))

    _log.info(
        "Regression recommendation: %d bot(s) recommended (%d direct, %d via graph)",
        len(recommendations),
        sum(1 for r in recommendations if r.is_direct_match),
        sum(1 for r in recommendations if not r.is_direct_match),
    )

    return RegressionReport(
        query=query,
        ui_elements=ui_elements,
        recommendations=recommendations,
        total_bots_evaluated=len(recommendations),
        search_results=all_search_results,
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_search_results(results: List[SearchResult], max_text_len: int = 120) -> None:
    """Pretty-print a list of search results to stdout."""
    if not results:
        print("No results found.")
        return

    print(f"\n{'='*70}")
    print(f"  Search Results  ({len(results)} match(es))")
    print(f"{'='*70}")

    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Bot: {r.bot_name}  |  type: {r.chunk_type}  |  similarity: {r.similarity:.3f}")
        if r.metadata.get("selector"):
            print(f"       Selector:  {r.metadata['selector']}")
        if r.metadata.get("window_title"):
            print(f"       Window:    {r.metadata['window_title']}")
        text_preview = r.text[:max_text_len] + ("…" if len(r.text) > max_text_len else "")
        print(f"       Preview:   {text_preview}")

    print()


def print_regression_report(report: RegressionReport) -> None:
    """Pretty-print a regression report to stdout."""
    print(f"\n{'='*70}")
    print("  REGRESSION TEST RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"  Query:       {report.query}")
    if report.ui_elements:
        print(f"  UI elements: {', '.join(report.ui_elements)}")
    print(f"  Bots found:  {report.total_bots_evaluated}")
    print()

    if not report.recommendations:
        print("  ⚠  No bots matched. Try broadening your query or re-indexing.")
        return

    for i, rec in enumerate(report.recommendations, 1):
        tag = "DIRECT" if rec.is_direct_match else "GRAPH "
        print(f"  [{i}] [{tag}] Bot: {rec.bot_name}  (score: {rec.score:.3f})")
        if rec.dependency_path:
            print(f"         Dependency path: {rec.dependency_path}")
        if rec.matching_selectors:
            print(f"         Selectors:  {', '.join(rec.matching_selectors)}")
        if rec.matching_windows:
            print(f"         Windows:    {', '.join(rec.matching_windows)}")
        for ev in rec.evidence[:3]:  # cap evidence lines
            print(f"         Evidence:   {ev}")
        if len(rec.evidence) > 3:
            print(f"         … +{len(rec.evidence) - 3} more evidence item(s)")

    print()
