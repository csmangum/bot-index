"""
src/graph.py - Build and query the A360 bot call graph.

The call graph is a directed graph where an edge  A → B  means "bot A calls
bot B" (via a ``TaskBot: Run`` / ``taskBotPath`` command).

The graph is built on-the-fly from the Chroma metadata (no separate file
needed), using every stored chunk whose ``chunk_type == "action"`` and
``called_bot != ""``.

Public API::

    build_call_graph(collection=None) -> nx.DiGraph
    get_direct_callers(graph, bot_name)  -> List[str]
    get_transitive_callers(graph, bot_name, max_hops=5) -> Dict[str, int]
    get_direct_callees(graph, bot_name)  -> List[str]
    get_all_bots(graph) -> List[str]
    print_graph_summary(graph) -> None
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_call_graph(collection=None):
    """Build a directed NetworkX graph from the Chroma collection metadata.

    Each node is a bot name.  Edges represent "caller → callee" relationships
    extracted from ``taskBotPath`` fields.

    Args:
        collection: A pre-fetched Chroma collection.  If ``None``, fetches
                    using the default config.

    Returns:
        A ``networkx.DiGraph`` where each node has a ``bot_id`` attribute
        (if available) and each edge has a ``command_id`` attribute.
    """
    try:
        import networkx as nx  # type: ignore
    except ImportError as exc:
        raise ImportError("networkx is not installed. Run: pip install networkx") from exc

    if collection is None:
        from src.indexer import get_collection
        collection = get_collection()

    graph = nx.DiGraph()

    count = collection.count()
    if count == 0:
        _log.warning("Collection is empty – returning empty graph.")
        return graph

    # Fetch all documents (we only need metadatas)
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    for meta in metadatas:
        if not meta:
            continue

        bot_name = meta.get("bot_name", "")
        called_bot = meta.get("called_bot", "")
        chunk_type = meta.get("chunk_type", "")
        bot_id = meta.get("bot_id", "")
        cmd_id = meta.get("command_id", "")

        if not bot_name:
            continue

        # Ensure the caller node exists with attributes
        if bot_name not in graph:
            graph.add_node(bot_name, bot_id=bot_id)

        # Add caller→callee edge for TaskBot: Run chunks
        if chunk_type == "action" and called_bot:
            # Normalise callee name: strip leading path components and extension
            callee_name = _normalise_bot_name(called_bot)
            if callee_name not in graph:
                graph.add_node(callee_name, bot_id="")
            graph.add_edge(bot_name, callee_name, command_id=cmd_id)
            _log.debug("Edge: %s → %s (cmd=%s)", bot_name, callee_name, cmd_id)

    _log.info(
        "Call graph built: %d node(s), %d edge(s)", graph.number_of_nodes(), graph.number_of_edges()
    )
    return graph


def _normalise_bot_name(path: str) -> str:
    """Extract the bot name from a ``taskBotPath`` value.

    A360 taskBotPath values look like ``"bots/LoginToCustomerPortal"`` or
    ``"Automation Anywhere/Bots/LoginToCustomerPortal.atmx"``.
    We want just ``"LoginToCustomerPortal"``.

    Args:
        path: Raw taskBotPath string from the bot JSON.

    Returns:
        Just the bot file name, without extension or leading path.
    """
    import posixpath

    name = posixpath.basename(path.replace("\\", "/"))
    # strip common A360 extensions
    for ext in (".atmx", ".json", ".taskbot"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
    return name


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_direct_callers(graph, bot_name: str) -> List[str]:
    """Return all bots that directly call *bot_name*.

    Args:
        graph:    NetworkX DiGraph produced by :func:`build_call_graph`.
        bot_name: Name of the callee bot.

    Returns:
        List of caller bot names (may be empty).
    """
    if bot_name not in graph:
        return []
    # predecessors = nodes with an edge pointing TO bot_name
    return list(graph.predecessors(bot_name))


def get_transitive_callers(
    graph, bot_name: str, max_hops: Optional[int] = None
) -> Dict[str, int]:
    """Return all bots that transitively call *bot_name*, with their hop depth.

    Performs a breadth-first search backwards through the call graph.

    Args:
        graph:    NetworkX DiGraph produced by :func:`build_call_graph`.
        bot_name: The target (callee) bot name.
        max_hops: Maximum number of hops to traverse.  ``None`` → no limit.

    Returns:
        Dict mapping caller_name → hop_distance (1 = direct caller).
    """
    if max_hops is None:
        from config import MAX_GRAPH_HOPS
        max_hops = MAX_GRAPH_HOPS

    if bot_name not in graph:
        return {}

    visited: Dict[str, int] = {}
    queue = [(bot_name, 0)]

    while queue:
        current, depth = queue.pop(0)
        for predecessor in graph.predecessors(current):
            if predecessor not in visited:
                hop = depth + 1
                if max_hops == 0 or hop <= max_hops:
                    visited[predecessor] = hop
                    queue.append((predecessor, hop))

    return visited


def get_direct_callees(graph, bot_name: str) -> List[str]:
    """Return all bots directly called by *bot_name*.

    Args:
        graph:    NetworkX DiGraph.
        bot_name: Name of the caller bot.

    Returns:
        List of callee bot names.
    """
    if bot_name not in graph:
        return []
    return list(graph.successors(bot_name))


def get_all_bots(graph) -> List[str]:
    """Return all bot names present in the graph (sorted).

    Args:
        graph: NetworkX DiGraph.

    Returns:
        Sorted list of bot names.
    """
    return sorted(graph.nodes())


def print_graph_summary(graph) -> None:
    """Print a human-readable summary of the call graph to stdout.

    Args:
        graph: NetworkX DiGraph.
    """
    bots = get_all_bots(graph)
    if not bots:
        print("Call graph is empty – no bots indexed yet.")
        return

    print(f"\n{'='*60}")
    print(f"  Bot Call Graph  ({graph.number_of_nodes()} bots, {graph.number_of_edges()} edges)")
    print(f"{'='*60}")

    for bot in bots:
        callees = get_direct_callees(graph, bot)
        callers = get_direct_callers(graph, bot)
        callee_str = ", ".join(callees) if callees else "—"
        caller_str = ", ".join(callers) if callers else "—"
        print(f"\n  Bot: {bot}")
        print(f"    Calls:     {callee_str}")
        print(f"    Called by: {caller_str}")

    print()
