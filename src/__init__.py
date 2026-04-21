"""
src/__init__.py

Exposes the public API of the a360-bot-rag-poc source package so that
callers can do ``from src import parser, indexer, graph, search`` without
needing to know the internal module layout.
"""

from src import graph, indexer, parser, search, utils

__all__ = ["parser", "indexer", "graph", "search", "utils"]
