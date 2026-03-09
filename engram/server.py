"""Engram MCP server — persistent memory for LLMs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from .store import MemoryStore

logger = logging.getLogger("engram")

DB_PATH = os.environ.get("ENGRAM_DB", str(Path.home() / ".engram" / "memory.db"))
store = MemoryStore(DB_PATH)
server = Server("engram")


def _fmt_memory(m) -> str:
    """Format a memory for display."""
    parts = [f"[{m.id}] ({m.memory_type}/{m.category}"]
    if m.key:
        parts[0] += f"/{m.key}"
    parts[0] += f") confidence={m.confidence:.1f}"
    if m.score > 0:
        parts[0] += f" score={m.score:.3f}"
    parts.append(m.content)
    if m.model:
        parts.append(f"model: {m.model}")
    if m.context:
        parts.append(f"context: {m.context}")
    if m.tags:
        parts.append(f"tags: {', '.join(m.tags)}")
    parts.append(f"updated: {m.updated_at} | accessed: {m.accessed_at} ({m.access_count}x)")
    return "\n".join(parts)


def _fmt_history(h) -> str:
    """Format a history entry for display."""
    parts = [f"[{h.created_at}] {h.action}"]
    if h.model:
        parts[0] += f" by {h.model}"
    if h.action == "updated" and h.old_content != h.new_content:
        parts.append(f"  was: {h.old_content[:200] if h.old_content else '(none)'}")
        parts.append(f"  now: {h.new_content[:200] if h.new_content else '(none)'}")
    elif h.action == "created":
        parts.append(f"  content: {h.new_content[:200] if h.new_content else '(none)'}")
    elif h.action == "forgotten":
        parts.append(f"  deleted: {h.old_content[:200] if h.old_content else '(none)'}")
    if h.old_confidence != h.new_confidence and h.old_confidence is not None:
        parts.append(f"  confidence: {h.old_confidence} -> {h.new_confidence}")
    if h.context:
        parts.append(f"  reasoning: {h.context}")
    return "\n".join(parts)


def _fmt_relationship(r) -> str:
    """Format a relationship for display."""
    line = f"[{r.id}] {r.entity_from} --({r.relation_type})--> {r.entity_to}"
    if r.metadata:
        line += f" {json.dumps(r.metadata)}"
    return line


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="remember",
            description=(
                "Store a new memory. Use memory_type='fact' for stable knowledge (configs, IPs, "
                "architecture), 'episode' for experiential context (debugging sessions, decisions), "
                "'preference' for user preferences. Facts with the same category+key are auto-deduplicated."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["fact", "episode", "preference"],
                        "default": "fact",
                        "description": "Type: fact (stable knowledge), episode (experiential), preference (user prefs)",
                    },
                    "category": {
                        "type": "string",
                        "default": "general",
                        "description": "Category for organization (e.g., 'network', 'nas', 'proxmox', 'project')",
                    },
                    "key": {
                        "type": "string",
                        "description": "Unique key within category (for facts). Same category+key = upsert.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for filtering",
                    },
                    "confidence": {
                        "type": "number",
                        "default": 1.0,
                        "description": "Confidence level 0.0-1.0",
                    },
                    "source": {
                        "type": "string",
                        "description": "Where this was learned (e.g., 'user stated', 'observed in logs')",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model identifier (e.g., 'claude-opus-4-6', 'qwen3:32b'). Self-identify here.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Reasoning or justification for storing this memory. Explain WHY you concluded this.",
                    },
                },
                "required": ["content"],
            },
        ),
        types.Tool(
            name="recall",
            description=(
                "Search memories by natural language query. Returns ranked results combining "
                "keyword relevance, recency, access frequency, and confidence. This is your "
                "primary retrieval tool — use it to find what you know about a topic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter to a specific category",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["fact", "episode", "preference"],
                        "description": "Filter to a specific memory type",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to memories with any of these tags",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max results to return",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="forget",
            description="Delete a memory by ID, or by category+key for facts. Records provenance of deletion.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Memory ID to delete",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category (used with key for fact deletion)",
                    },
                    "key": {
                        "type": "string",
                        "description": "Key (used with category for fact deletion)",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model performing the deletion",
                    },
                    "context": {
                        "type": "string",
                        "description": "Why this memory is being deleted",
                    },
                },
            },
        ),
        types.Tool(
            name="update",
            description="Update specific fields of an existing memory. Only provided fields are changed. Records provenance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to update",
                    },
                    "content": {"type": "string"},
                    "category": {"type": "string"},
                    "key": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "model": {
                        "type": "string",
                        "description": "Model performing the update",
                    },
                    "context": {
                        "type": "string",
                        "description": "Reasoning for the update — especially important if changing a conclusion",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        types.Tool(
            name="relate",
            description=(
                "Create or update a relationship between two entities. Useful for tracking "
                "how services, machines, projects, and concepts connect to each other."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_from": {
                        "type": "string",
                        "description": "Source entity (e.g., 'Immich')",
                    },
                    "entity_to": {
                        "type": "string",
                        "description": "Target entity (e.g., 'TrueNAS')",
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "Relationship type (e.g., 'runs_on', 'depends_on', 'connects_to')",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata about the relationship",
                    },
                    "confidence": {
                        "type": "number",
                        "default": 1.0,
                    },
                },
                "required": ["entity_from", "entity_to", "relation_type"],
            },
        ),
        types.Tool(
            name="get_entity",
            description="Get all memories and relationships for a named entity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to look up",
                    },
                },
                "required": ["entity"],
            },
        ),
        types.Tool(
            name="get_context",
            description=(
                "Bootstrap context for a new conversation. Returns user preferences, "
                "recently updated facts, and optionally topic-specific memories. "
                "Call this at the start of a conversation to prime your memory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional topic to focus context retrieval on",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                    },
                },
            },
        ),
        types.Tool(
            name="history",
            description=(
                "View the provenance/audit trail for a memory. Shows who created or modified it, "
                "what changed, and their reasoning. Essential for understanding divergent conclusions "
                "from different models. Can also filter by model to see all contributions from a specific model."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Memory ID to get history for",
                    },
                    "model": {
                        "type": "string",
                        "description": "Filter history by model (shows all that model's changes across all memories)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                    },
                },
            },
        ),
        types.Tool(
            name="list_categories",
            description="List all memory categories with counts.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="stats",
            description="Get memory system statistics: total count, types, staleness, DB path.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_stale",
            description="Find memories not accessed in N days. Use for periodic cleanup.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "default": 30,
                        "description": "Days since last access to consider stale",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        result = _dispatch(name, arguments)
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [types.TextContent(type="text", text=f"Error: {e}")]


def _dispatch(name: str, args: dict) -> str:
    if name == "remember":
        m = store.remember(
            content=args["content"],
            memory_type=args.get("memory_type", "fact"),
            category=args.get("category", "general"),
            key=args.get("key"),
            tags=args.get("tags"),
            confidence=args.get("confidence", 1.0),
            source=args.get("source"),
            model=args.get("model"),
            context=args.get("context"),
        )
        return f"Stored:\n{_fmt_memory(m)}"

    elif name == "recall":
        memories = store.recall(
            query=args["query"],
            category=args.get("category"),
            memory_type=args.get("memory_type"),
            tags=args.get("tags"),
            limit=args.get("limit", 10),
        )
        if not memories:
            return "No memories found matching that query."
        lines = [f"Found {len(memories)} memories:\n"]
        for m in memories:
            lines.append(_fmt_memory(m))
            lines.append("")
        return "\n".join(lines)

    elif name == "forget":
        model = args.get("model")
        context = args.get("context")
        if args.get("memory_id"):
            ok = store.forget(args["memory_id"], model=model, context=context)
        elif args.get("category") and args.get("key"):
            ok = store.forget_by_key(args["category"], args["key"], model=model, context=context)
        else:
            return "Provide either memory_id or both category+key."
        return "Forgotten." if ok else "Memory not found."

    elif name == "update":
        m = store.update(
            memory_id=args["memory_id"],
            content=args.get("content"),
            category=args.get("category"),
            key=args.get("key"),
            tags=args.get("tags"),
            confidence=args.get("confidence"),
            model=args.get("model"),
            context=args.get("context"),
        )
        if not m:
            return "Memory not found."
        return f"Updated:\n{_fmt_memory(m)}"

    elif name == "relate":
        r = store.relate(
            entity_from=args["entity_from"],
            entity_to=args["entity_to"],
            relation_type=args["relation_type"],
            metadata=args.get("metadata"),
            confidence=args.get("confidence", 1.0),
        )
        return f"Related:\n{_fmt_relationship(r)}"

    elif name == "get_entity":
        data = store.get_entity(args["entity"])
        parts = [f"Entity: {data['entity']}\n"]
        if data["memories"]:
            parts.append(f"Memories ({len(data['memories'])}):")
            for m_dict in data["memories"]:
                m = type("M", (), m_dict)()  # quick attr access
                parts.append(f"  [{m_dict['id']}] {m_dict['content'][:120]}")
        if data["relationships"]:
            parts.append(f"\nRelationships ({len(data['relationships'])}):")
            for r_dict in data["relationships"]:
                parts.append(f"  {r_dict['entity_from']} --({r_dict['relation_type']})--> {r_dict['entity_to']}")
        if not data["memories"] and not data["relationships"]:
            parts.append("No memories or relationships found.")
        return "\n".join(parts)

    elif name == "get_context":
        ctx = store.get_context(
            topic=args.get("topic"),
            limit=args.get("limit", 20),
        )
        parts = []
        if ctx["preferences"]:
            parts.append(f"Preferences ({len(ctx['preferences'])}):")
            for p in ctx["preferences"]:
                parts.append(f"  - {p['content'][:120]}")
        if ctx["recent"]:
            parts.append(f"\nRecent facts ({len(ctx['recent'])}):")
            for f in ctx["recent"]:
                parts.append(f"  [{f['category']}/{f.get('key', '?')}] {f['content'][:120]}")
        if ctx.get("topic_memories"):
            parts.append(f"\nTopic memories ({len(ctx['topic_memories'])}):")
            for m in ctx["topic_memories"]:
                parts.append(f"  [{m['id']}] {m['content'][:120]}")
        if not parts:
            parts.append("Memory is empty. Start building it with the 'remember' tool.")
        return "\n".join(parts)

    elif name == "history":
        if args.get("memory_id"):
            entries = store.get_history(args["memory_id"])
            if not entries:
                return "No history found for this memory."
            lines = [f"History for memory {args['memory_id']} ({len(entries)} entries):\n"]
            for h in entries:
                lines.append(_fmt_history(h))
                lines.append("")
            return "\n".join(lines)
        elif args.get("model"):
            entries = store.get_history_by_model(args["model"], limit=args.get("limit", 20))
            if not entries:
                return f"No history found for model {args['model']}."
            lines = [f"History from {args['model']} ({len(entries)} entries):\n"]
            for h in entries:
                lines.append(f"memory:{h.memory_id} — {_fmt_history(h)}")
                lines.append("")
            return "\n".join(lines)
        else:
            return "Provide either memory_id or model to query history."

    elif name == "list_categories":
        cats = store.list_categories()
        if not cats:
            return "No categories yet."
        lines = ["Categories:"]
        for c in cats:
            lines.append(f"  {c['category']} ({c['memory_type']}): {c['count']}")
        return "\n".join(lines)

    elif name == "stats":
        s = store.stats()
        return json.dumps(s, indent=2)

    elif name == "get_stale":
        memories = store.get_stale(
            days=args.get("days", 30),
            limit=args.get("limit", 20),
        )
        if not memories:
            return "No stale memories found."
        lines = [f"Stale memories ({len(memories)}):\n"]
        for m in memories:
            lines.append(f"[{m.id}] ({m.category}) last accessed {m.accessed_at}")
            lines.append(f"  {m.content[:120]}")
            lines.append("")
        return "\n".join(lines)

    else:
        return f"Unknown tool: {name}"


def main():
    import asyncio

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Engram starting with DB at {DB_PATH}")

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
