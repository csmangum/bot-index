#!/usr/bin/env python3
"""
main.py - CLI entry point for the A360 Bot RAG system.

Usage examples::

    # Index all bots in ./bots/ (re-index everything)
    python main.py index

    # Index example bots shipped with the repo
    python main.py index --bots-dir example_bots

    # Re-index (clears existing collection first)
    python main.py index --force

    # Semantic search
    python main.py search "login button changed"

    # Search with specific UI elements for richer results
    python main.py search "customer portal login" --ui-elements "button#login-submit,input#username"

    # Show the bot call graph
    python main.py graph

    # Show collection status / statistics
    python main.py status
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure the project root is on sys.path so that "src" and
# "config" are importable regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (  # noqa: E402  (import after sys.path tweak)
    BOTS_DIR,
    DEFAULT_TOP_K,
    EXAMPLE_BOTS_DIR,
    LOG_LEVEL,
)
from src.utils import configure_logging  # noqa: E402

# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    """Parse and index bot files."""
    from src.indexer import index_bots

    bots_dir = Path(args.bots_dir) if args.bots_dir else BOTS_DIR
    if not bots_dir.exists():
        print(f"[ERROR] Bots directory not found: {bots_dir}")
        print("        Use --bots-dir to specify a different path, or try:")
        print("        python main.py index --bots-dir example_bots")
        return 1

    print(f"Indexing bots from: {bots_dir}")
    if args.force:
        print("  --force: clearing existing index first …")

    results = index_bots(bots_dir=bots_dir, force_reindex=args.force)

    if not results:
        print("[WARN] No bots were indexed.")
        return 0

    print("\nIndexing summary:")
    total = 0
    for bot_name, n_chunks in sorted(results.items()):
        print(f"  {bot_name}: {n_chunks} chunk(s)")
        total += n_chunks
    print(f"\n  Total: {len(results)} bot(s), {total} chunk(s)")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Semantic search + regression recommendation."""
    from src.search import print_regression_report, recommend_for_regression

    query: str = args.query
    ui_elements = (
        [e.strip() for e in args.ui_elements.split(",") if e.strip()]
        if args.ui_elements
        else []
    )
    top_k: int = args.top_k

    print(f"Searching for: '{query}'")
    if ui_elements:
        print(f"  UI elements: {', '.join(ui_elements)}")

    report = recommend_for_regression(
        query=query,
        ui_elements=ui_elements,
        top_k=top_k,
        use_graph=not args.no_graph,
    )
    print_regression_report(report)
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    """Display the bot call graph."""
    from src.graph import build_call_graph, print_graph_summary
    from src.indexer import get_collection

    collection = get_collection()
    if collection.count() == 0:
        print("[WARN] Collection is empty – please run 'index' first.")
        return 1

    graph = build_call_graph(collection=collection)
    print_graph_summary(graph)
    return 0


def cmd_status(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Show collection statistics."""
    from src.indexer import collection_stats

    stats = collection_stats()
    print(f"\n{'='*50}")
    print("  Collection Status")
    print(f"{'='*50}")
    print(f"  Collection:   {stats['collection_name']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Unique bots:  {stats['unique_bots']}")
    if stats["chunk_types"]:
        print("  Chunk types:")
        for ct, cnt in sorted(stats["chunk_types"].items()):
            print(f"    {ct}: {cnt}")
    print()
    return 0


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="A360 Bot Semantic Search & Regression Recommendation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ── index ──────────────────────────────────────────────────────────────
    p_index = subparsers.add_parser("index", help="Parse and index bot JSON files")
    p_index.add_argument(
        "--bots-dir",
        metavar="DIR",
        default=None,
        help=f"Directory containing bot JSON files (default: {BOTS_DIR})",
    )
    p_index.add_argument(
        "--force",
        action="store_true",
        help="Clear the existing index before re-indexing",
    )
    p_index.set_defaults(func=cmd_index)

    # ── search ─────────────────────────────────────────────────────────────
    p_search = subparsers.add_parser(
        "search", help="Semantic search and regression recommendations"
    )
    p_search.add_argument("query", help="Natural-language change description")
    p_search.add_argument(
        "--ui-elements",
        metavar="SELECTORS",
        default="",
        help="Comma-separated CSS selectors or window titles to include in search",
    )
    p_search.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of vector-search results per query (default: {DEFAULT_TOP_K})",
    )
    p_search.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable call-graph expansion (return direct matches only)",
    )
    p_search.set_defaults(func=cmd_search)

    # ── graph ──────────────────────────────────────────────────────────────
    p_graph = subparsers.add_parser("graph", help="Display the bot call graph")
    p_graph.set_defaults(func=cmd_graph)

    # ── status ─────────────────────────────────────────────────────────────
    p_status = subparsers.add_parser("status", help="Show collection statistics")
    p_status.set_defaults(func=cmd_status)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments and dispatch to the appropriate sub-command."""
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("Unexpected error: %s", exc)
        print(f"\n[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
