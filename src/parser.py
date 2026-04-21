"""
src/parser.py - Structure-aware parser and chunker for A360 bot JSON files.

A360 bot JSON files follow a hierarchical schema:
  {
    "id": "...",
    "name": "BotName",
    "description": "...",
    "variables": [...],
    "commands": [
      {
        "id": "cmd-id",
        "type": "ActionNode" | "TaskBot: Run" | "ConditionNode" | "LoopNode",
        "sequence": N,
        "properties": { ... },
        "children": [ ... ]   # optional nested commands
      },
      ...
    ]
  }

The parser produces three tiers of chunks per bot:

  1. **Bot-level summary** – one chunk per bot file.
     Contains: name, description, variable names, list of called bots,
     all unique window titles and selectors found anywhere in the bot.

  2. **Block-level chunk** – one chunk per top-level command that has
     ``children`` (i.e. If/Else, Loop, …).
     Contains: block type, parent context, all child action summaries.

  3. **Action-level chunk** – one chunk per leaf command (no children).
     Contains: action type, package, selector, window title, object repo name,
     called bot (if TaskBot: Run), and a snippet of parent block context.

Each chunk is returned as a dict::

    {
        "id":       str,          # deterministic hash-based ID
        "text":     str,          # human-readable text fed to the embedder
        "metadata": dict[str, str]  # flat string metadata for Chroma
    }

Metadata keys (all values are strings):
  bot_name, bot_id, bot_version, bot_description,
  chunk_type ("bot_summary" | "block" | "action"),
  command_id, action_type, package_name,
  selector, selector_type, window_title,
  object_repository_name, called_bot,
  variables (comma-separated), depth (str int),
  parent_command_id, parent_action_type
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils import clean_text, make_doc_id

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Chunk = Dict[str, Any]
CommandNode = Dict[str, Any]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_bot_file(bot_path: Path) -> List[Chunk]:
    """Parse a single A360 bot JSON file and return all chunks.

    Args:
        bot_path: Absolute ``Path`` to the ``.json`` file.

    Returns:
        List of chunk dicts ready for indexing.

    Raises:
        ValueError: If the file cannot be parsed as a valid bot JSON.
    """
    _log.debug("Parsing bot file: %s", bot_path)
    try:
        raw = bot_path.read_text(encoding="utf-8")
        data: dict = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot parse '{bot_path}': {exc}") from exc

    bot_name = data.get("name", bot_path.stem)
    bot_id = data.get("id", "")
    bot_version = data.get("version", "")
    bot_description = data.get("description", "")
    variables: List[dict] = data.get("variables", [])
    if "commands" in data:
        commands: List[CommandNode] = data.get("commands") or []
    else:
        commands = data.get("actions") or []
    # Note: A360 uses "commands" in most versions; older exports may use "actions".

    var_names = [v.get("name", "") for v in variables if v.get("name")]

    # Collect global info via a full tree walk before chunking
    all_selectors, all_window_titles, all_called_bots = _collect_global_info(commands)

    chunks: List[Chunk] = []

    # ── 1. Bot-level summary ────────────────────────────────────────────────
    summary_chunk = _make_bot_summary(
        bot_name=bot_name,
        bot_id=bot_id,
        bot_version=bot_version,
        bot_description=bot_description,
        var_names=var_names,
        all_selectors=all_selectors,
        all_window_titles=all_window_titles,
        all_called_bots=all_called_bots,
    )
    chunks.append(summary_chunk)

    # ── 2 & 3. Block + action chunks (recursive) ───────────────────────────
    base_meta = {
        "bot_name": bot_name,
        "bot_id": bot_id,
        "bot_version": bot_version,
        "bot_description": truncate_meta(bot_description),
        "variables": ", ".join(var_names),
    }
    for cmd in commands:
        chunks.extend(
            _process_command(
                cmd=cmd,
                bot_name=bot_name,
                base_meta=base_meta,
                parent_context="",
                parent_command_id="",
                parent_action_type="",
                depth=0,
            )
        )

    _log.info(
        "Parsed '%s': %d chunk(s) (%d commands at top level)",
        bot_name,
        len(chunks),
        len(commands),
    )
    return chunks


def parse_all_bots(bots_dir: Path, recursive: bool = True) -> List[Chunk]:
    """Parse every ``.json`` file under *bots_dir* and return all chunks.

    Args:
        bots_dir:  Directory containing exported A360 bot JSON files.
        recursive: Descend into sub-folders.

    Returns:
        Flat list of all chunks from all bots.
    """
    from src.utils import find_bot_files

    bot_files = find_bot_files(bots_dir, recursive=recursive)
    if not bot_files:
        _log.warning("No .json bot files found in '%s'", bots_dir)
        return []

    all_chunks: List[Chunk] = []
    for bf in bot_files:
        try:
            all_chunks.extend(parse_bot_file(bf))
        except ValueError as exc:
            _log.error("Skipping '%s': %s", bf.name, exc)
    return all_chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_global_info(
    commands: List[CommandNode],
) -> Tuple[List[str], List[str], List[str]]:
    """Walk the entire command tree and collect selectors, window titles, called bots.

    Returns:
        (unique_selectors, unique_window_titles, unique_called_bots)
    """
    selectors: List[str] = []
    window_titles: List[str] = []
    called_bots: List[str] = []

    def _walk(nodes: List[CommandNode]) -> None:
        for node in nodes:
            props = node.get("properties", {})
            sel = props.get("selector", "")
            win = props.get("windowTitle", "")
            tbot = props.get("taskBotPath", "")

            if sel and sel not in selectors:
                selectors.append(sel)
            if win and win not in window_titles:
                window_titles.append(win)
            if tbot and tbot not in called_bots:
                called_bots.append(tbot)

            children = node.get("children", [])
            if children:
                _walk(children)

    _walk(commands)
    return selectors, window_titles, called_bots


def _make_bot_summary(
    bot_name: str,
    bot_id: str,
    bot_version: str,
    bot_description: str,
    var_names: List[str],
    all_selectors: List[str],
    all_window_titles: List[str],
    all_called_bots: List[str],
) -> Chunk:
    """Build the bot-level summary chunk."""
    parts = [f"Bot: {bot_name}"]
    if bot_description:
        parts.append(f"Description: {bot_description}")
    if bot_version:
        parts.append(f"Version: {bot_version}")
    if var_names:
        parts.append(f"Variables: {', '.join(var_names)}")
    if all_called_bots:
        parts.append(f"Calls: {', '.join(all_called_bots)}")
    if all_window_titles:
        parts.append(f"Windows: {', '.join(all_window_titles)}")
    if all_selectors:
        parts.append(f"Selectors: {', '.join(all_selectors)}")

    text = " | ".join(parts)

    doc_id = make_doc_id(bot_name, "bot_summary", "summary")
    metadata = {
        "bot_name": bot_name,
        "bot_id": bot_id,
        "bot_version": bot_version,
        "bot_description": truncate_meta(bot_description),
        "chunk_type": "bot_summary",
        "variables": ", ".join(var_names),
        "called_bots": ", ".join(all_called_bots),
        "window_titles": ", ".join(all_window_titles),
        "selectors": ", ".join(all_selectors),
        "command_id": "",
        "action_type": "",
        "package_name": "",
        "selector": "",
        "selector_type": "",
        "window_title": "",
        "object_repository_name": "",
        "called_bot": "",
        "depth": "0",
        "parent_command_id": "",
        "parent_action_type": "",
    }
    return {"id": doc_id, "text": clean_text(text), "metadata": metadata}


def _process_command(
    cmd: CommandNode,
    bot_name: str,
    base_meta: dict,
    parent_context: str,
    parent_command_id: str,
    parent_action_type: str,
    depth: int,
) -> List[Chunk]:
    """Recursively build chunks for *cmd* and all its descendants.

    Generates:
    - A **block-level** chunk when *cmd* has ``children``.
    - An **action-level** chunk when *cmd* is a leaf (no children) or
      is a ``TaskBot: Run`` node.

    Args:
        cmd:                The command node dict.
        bot_name:           Name of the parent bot (for IDs and metadata).
        base_meta:          Shared metadata fields (bot_name, bot_id, …).
        parent_context:     Short description of the parent block for overlap.
        parent_command_id:  ID of the direct parent command node.
        parent_action_type: Type string of the direct parent command node.
        depth:              Current recursion depth (0 = top level).

    Returns:
        List of Chunk dicts.
    """
    from config import MAX_PARSE_DEPTH  # local import to avoid circular deps

    if depth > MAX_PARSE_DEPTH:
        _log.warning("Max parse depth (%d) reached – truncating at '%s'", MAX_PARSE_DEPTH, cmd.get("id"))
        return []

    chunks: List[Chunk] = []
    props = cmd.get("properties", {})
    cmd_id = cmd.get("id", "")
    cmd_type = cmd.get("type", "")
    cmd_name = props.get("commandName", cmd_type)
    package = props.get("packageName", "")
    selector = props.get("selector", "")
    selector_type = props.get("selectorType", "")
    window_title = props.get("windowTitle", "")
    obj_repo = props.get("objectRepositoryName", "")
    called_bot = props.get("taskBotPath", "")
    children: List[CommandNode] = cmd.get("children", [])

    # Build a one-line description of this command for use as context in children
    short_desc = _short_description(cmd_type, cmd_name, package, selector, window_title, called_bot)

    # ── Block-level chunk (commands with children) ──────────────────────────
    if children:
        # Collect all child summaries for block text
        child_summaries = [
            _short_description(
                c.get("type", ""),
                c.get("properties", {}).get("commandName", c.get("type", "")),
                c.get("properties", {}).get("packageName", ""),
                c.get("properties", {}).get("selector", ""),
                c.get("properties", {}).get("windowTitle", ""),
                c.get("properties", {}).get("taskBotPath", ""),
            )
            for c in children
        ]

        text_parts = [f"Bot: {bot_name}", f"Block: {cmd_name} ({cmd_type})"]
        if parent_context:
            text_parts.append(f"Parent Context: {parent_context}")
        if window_title:
            text_parts.append(f"Window: {window_title}")
        text_parts.append(f"Contains: {'; '.join(child_summaries)}")

        block_text = clean_text(" | ".join(text_parts))
        block_id = make_doc_id(bot_name, "block", cmd_id)
        block_meta = {
            **base_meta,
            "chunk_type": "block",
            "command_id": cmd_id,
            "action_type": cmd_type,
            "package_name": package,
            "selector": selector,
            "selector_type": selector_type,
            "window_title": window_title,
            "object_repository_name": obj_repo,
            "called_bot": called_bot,
            "depth": str(depth),
            "parent_command_id": parent_command_id,
            "parent_action_type": parent_action_type,
        }
        # Ensure all meta values are strings (Chroma requirement)
        block_meta = _stringify_meta(block_meta)
        chunks.append({"id": block_id, "text": block_text, "metadata": block_meta})

        # Pass this block's short description as context for children (overlap)
        context_for_children = short_desc
        for child_cmd in children:
            chunks.extend(
                _process_command(
                    cmd=child_cmd,
                    bot_name=bot_name,
                    base_meta=base_meta,
                    parent_context=context_for_children,
                    parent_command_id=cmd_id,
                    parent_action_type=cmd_type,
                    depth=depth + 1,
                )
            )

    else:
        # ── Action-level chunk (leaf node) ───────────────────────────────────
        text_parts = [f"Bot: {bot_name}"]
        text_parts.append(f"Action: {cmd_name}")
        if cmd_type and cmd_type != cmd_name:
            text_parts.append(f"Type: {cmd_type}")
        if package:
            text_parts.append(f"Package: {package}")
        if parent_context:
            text_parts.append(f"Parent Context: {parent_context}")
        if selector:
            text_parts.append(f"Selector: {selector}")
        if selector_type:
            text_parts.append(f"Selector Type: {selector_type}")
        if window_title:
            text_parts.append(f"Window: {window_title}")
        if obj_repo:
            text_parts.append(f"Object Repository: {obj_repo}")
        if called_bot:
            text_parts.append(f"Calls Bot: {called_bot}")

        # Include input/output variable mappings for Run Task nodes
        input_vars = props.get("inputVariables", {})
        output_vars = props.get("outputVariables", {})
        if input_vars:
            iv_str = ", ".join(f"{k}={v}" for k, v in input_vars.items())
            text_parts.append(f"Input Variables: {iv_str}")
        if output_vars:
            ov_str = ", ".join(f"{k}={v}" for k, v in output_vars.items())
            text_parts.append(f"Output Variables: {ov_str}")

        action_text = clean_text(" | ".join(text_parts))
        action_id = make_doc_id(bot_name, "action", cmd_id)
        action_meta = {
            **base_meta,
            "chunk_type": "action",
            "command_id": cmd_id,
            "action_type": cmd_type,
            "package_name": package,
            "selector": selector,
            "selector_type": selector_type,
            "window_title": window_title,
            "object_repository_name": obj_repo,
            "called_bot": called_bot,
            "depth": str(depth),
            "parent_command_id": parent_command_id,
            "parent_action_type": parent_action_type,
        }
        action_meta = _stringify_meta(action_meta)
        chunks.append({"id": action_id, "text": action_text, "metadata": action_meta})

    return chunks


def _short_description(
    cmd_type: str,
    cmd_name: str,
    package: str,
    selector: str,
    window_title: str,
    called_bot: str,
) -> str:
    """Build a compact one-line description of a command (used as parent context)."""
    parts: List[str] = []
    label = cmd_name or cmd_type
    if label:
        parts.append(label)
    if package and package not in label:
        parts.append(f"[{package}]")
    if selector:
        parts.append(f"selector={selector}")
    if window_title:
        parts.append(f"window='{window_title}'")
    if called_bot:
        parts.append(f"calls={called_bot}")
    return clean_text(", ".join(parts)) if parts else "unknown"


def _stringify_meta(meta: dict) -> dict:
    """Ensure all metadata values are plain strings (Chroma requirement)."""
    return {k: str(v) if v is not None else "" for k, v in meta.items()}


def truncate_meta(text: str, max_chars: int = 200) -> str:
    """Truncate long strings that go into metadata fields."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
