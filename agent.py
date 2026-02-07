#!/usr/bin/env python3
"""
D&D Campaign Archivist — Claude Agent SDK
Interactive agent that searches campaign transcripts using hybrid
text + vector search. Powered by Claude with custom MCP tools.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import chromadb
import requests
from etl import _load_dotenv

_load_dotenv()  # Load .env before SDK picks up env vars

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

# --- Configuration ---
BASE_DIR = Path(__file__).parent
TRANSCRIPTS_DIR = BASE_DIR / "output_transcripts"
CHROMADB_DIR = BASE_DIR / "data" / "chromadb"
PLAYERS_JSON = BASE_DIR / "config" / "players.json"
VOICE_BANK_PATH = BASE_DIR / "config" / "voice_bank.json"
INPUT_AUDIO_DIR = BASE_DIR / "input_audio"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
DEFAULT_MODEL = "claude-opus-4-6"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".mp4", ".mkv", ".webm"}

SPEAKER_LINE_RE = re.compile(
    r"^\*\*(.+?)\*\*\s+\[(\d{2}:\d{2}:\d{2})\]:\s*(.+)$"
)

SYSTEM_PROMPT = """\
You are the Campaign Archivist, an expert assistant for a D&D campaign.
Your job is to help players recall events, find quotes, track storylines,
and answer questions about what happened in their sessions.

## Rules
- ALWAYS search transcripts before answering — never fabricate campaign details.
- Use `semantic_search` for broad/conceptual queries ("when did the party discuss strategy").
- Use `search_transcripts` for exact names, quotes, or specific words ("what did Narrator say about the tavern").
- Use `list_sessions` to orient yourself on available data.
- Quote dialogue with speaker name + timestamp when citing evidence.
- Clearly distinguish between what transcripts say vs. your inferences.
- If no transcripts exist yet, suggest: `python etl.py && python embed.py`
- Keep answers focused and cite sources. Don't repeat entire transcripts unless asked.
"""


# --- Configurable paths (support --data-dir for remote deployment) ---
_transcripts_dir = TRANSCRIPTS_DIR
_chromadb_dir = CHROMADB_DIR
_chroma_collection = None


def _get_collection() -> chromadb.Collection | None:
    """Lazy-init ChromaDB collection."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if not _chromadb_dir.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(_chromadb_dir))
        _chroma_collection = client.get_or_create_collection(
            name="campaign_transcripts",
            metadata={"hnsw:space": "cosine"},
        )
        return _chroma_collection
    except Exception:
        return None


def _get_embedding(text: str) -> list[float] | None:
    """Get embedding from Ollama. Returns None on failure."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except Exception:
        return None


# ============================================================
# MCP Tools
# ============================================================

@tool("list_sessions", "List all available campaign session transcripts with dates and sizes", {})
async def list_sessions(args: dict[str, Any]) -> dict[str, Any]:
    files = sorted(_transcripts_dir.glob("*.md"))
    if not files:
        return {"content": [{"type": "text", "text": "No transcripts found. Run: python etl.py && python embed.py"}]}

    lines = [f"Found {len(files)} session transcript(s):\n"]
    for f in files:
        size_kb = f.stat().st_size / 1024
        lines.append(f"- **{f.stem}** ({size_kb:.1f} KB)")

    # ChromaDB status
    collection = _get_collection()
    if collection and collection.count() > 0:
        lines.append(f"\nChromaDB: {collection.count()} embedded chunks")
    else:
        lines.append("\nChromaDB: no embeddings yet (run: python embed.py)")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool("search_transcripts", "Exact text search across session transcripts. Use for specific names, quotes, and keywords.", {
    "query": str,
    "speaker": str,
    "max_results": int,
})
async def search_transcripts(args: dict[str, Any]) -> dict[str, Any]:
    search_query = args["query"].lower()
    speaker_filter = args.get("speaker", "").lower()
    max_results = min(args.get("max_results", 20) or 20, 20)

    files = sorted(_transcripts_dir.glob("*.md"))
    if not files:
        return {"content": [{"type": "text", "text": "No transcripts found."}]}

    results = []
    for filepath in files:
        text = filepath.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            if search_query not in line.lower():
                continue
            if speaker_filter:
                match = SPEAKER_LINE_RE.match(line.strip())
                if match and speaker_filter not in match.group(1).lower():
                    continue
            results.append(f"[{filepath.stem}:{i}] {line.strip()}")
            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    if not results:
        return {"content": [{"type": "text", "text": f"No results found for '{args['query']}'"}]}

    header = f"Found {len(results)} match(es) for '{args['query']}':\n"
    return {"content": [{"type": "text", "text": header + "\n".join(results)}]}


@tool("semantic_search", "Vector similarity search for conceptual/thematic queries across campaign transcripts.", {
    "query": str,
    "n_results": int,
    "speaker": str,
})
async def semantic_search(args: dict[str, Any]) -> dict[str, Any]:
    collection = _get_collection()
    if not collection or collection.count() == 0:
        return {"content": [{"type": "text", "text": "No embeddings available. Run: python embed.py"}]}

    search_query = args["query"]
    n_results = min(args.get("n_results", 10) or 10, 10)
    speaker_filter = args.get("speaker", "")

    embedding = _get_embedding(search_query)
    if embedding is None:
        return {"content": [{"type": "text", "text": "Error: Could not generate embedding. Is Ollama running?"}]}

    where_filter = {"speaker": speaker_filter} if speaker_filter else None

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Search error: {e}"}]}

    if not results["documents"] or not results["documents"][0]:
        return {"content": [{"type": "text", "text": f"No semantic matches for '{search_query}'"}]}

    lines = [f"Top {len(results['documents'][0])} semantic matches for '{search_query}':\n"]
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist  # cosine distance → similarity
        lines.append(
            f"[{meta['session']}:{meta['line_number']}] "
            f"(similarity: {similarity:.2f}) "
            f"**{meta['speaker']}** [{meta['timestamp']}]"
        )
        # Show just the current speaker line (last line of chunk)
        last_line = doc.strip().split("\n")[-1]
        lines.append(f"  {last_line}\n")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool("get_session_content", "Read the full or partial content of a specific session transcript.", {
    "session": str,
    "start_line": int,
    "end_line": int,
})
async def get_session_content(args: dict[str, Any]) -> dict[str, Any]:
    session = args["session"]
    # Try exact match, then fuzzy
    filepath = _transcripts_dir / f"{session}.md"
    if not filepath.exists():
        # Try matching by stem
        candidates = list(_transcripts_dir.glob("*.md"))
        matches = [f for f in candidates if session.lower() in f.stem.lower()]
        if not matches:
            available = [f.stem for f in candidates]
            return {"content": [{"type": "text", "text":
                f"Session '{session}' not found. Available: {', '.join(available) or 'none'}"}]}
        filepath = matches[0]

    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()
    total = len(lines)

    start = max(1, args.get("start_line", 1) or 1)
    end = min(total, args.get("end_line", start + 199) or start + 199)

    selected = lines[start - 1:end]
    header = f"**{filepath.stem}** (lines {start}-{end} of {total}):\n\n"
    content = header + "\n".join(selected)

    if end < total:
        content += f"\n\n... ({total - end} more lines. Use start_line={end + 1} to continue)"

    return {"content": [{"type": "text", "text": content}]}


@tool("get_campaign_info", "Get the player/character roster and campaign configuration.", {})
async def get_campaign_info(args: dict[str, Any]) -> dict[str, Any]:
    lines = []

    # Prefer voice bank (persistent speaker identity) over static players.json
    if VOICE_BANK_PATH.exists():
        with open(VOICE_BANK_PATH) as f:
            bank = json.load(f)

        lines.append("**Campaign Roster** (from voice bank):\n")
        for name, info in bank.get("speakers", {}).items():
            n_sessions = len(info.get("embeddings", []))
            lines.append(f"- {info.get('character', name)} (played by {info.get('player', '?')}) — {n_sessions} session(s) enrolled")

        threshold = bank.get("meta", {}).get("threshold", "?")
        lines.append(f"\nVoice matching threshold: {threshold}")
    elif PLAYERS_JSON.exists():
        with open(PLAYERS_JSON) as f:
            players = json.load(f)

        lines.append("**Campaign Roster** (from static config):\n")
        speaker_map = players.get("speaker_map", {})
        for speaker_id, info in speaker_map.items():
            if isinstance(info, dict):
                lines.append(f"- {info.get('character', '?')} (played by {info.get('player', '?')})")
            else:
                lines.append(f"- {info}")

        lines.append(f"\nSpeaker range: {players.get('min_speakers', '?')}-{players.get('max_speakers', '?')}")
    else:
        return {"content": [{"type": "text", "text": "No voice bank or players.json found."}]}

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


# ============================================================
# Agent setup
# ============================================================

def create_campaign_server():
    """Create the MCP server with all campaign tools."""
    return create_sdk_mcp_server(
        name="campaign",
        version="1.0.0",
        tools=[list_sessions, search_transcripts, semantic_search,
               get_session_content, get_campaign_info],
    )


def create_options(model: str = DEFAULT_MODEL) -> ClaudeAgentOptions:
    """Build agent options with campaign tools."""
    server = create_campaign_server()
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        model=model,
        mcp_servers={"campaign": server},
        allowed_tools=[
            "mcp__campaign__list_sessions",
            "mcp__campaign__search_transcripts",
            "mcp__campaign__semantic_search",
            "mcp__campaign__get_session_content",
            "mcp__campaign__get_campaign_info",
            "Read", "Grep", "Glob",
        ],
        permission_mode="bypassPermissions",
        max_turns=15,
        cwd=str(BASE_DIR),
    )


# ============================================================
# UI
# ============================================================

def print_banner():
    """Print startup banner with session info."""
    print("=" * 60)
    print("  D&D Campaign Archivist")
    print("  Powered by Claude + Agent SDK")
    print("=" * 60)

    # Transcript count
    transcripts = list(_transcripts_dir.glob("*.md"))
    print(f"\n  Sessions:    {len(transcripts)} transcript(s)")

    # Pending audio
    if INPUT_AUDIO_DIR.exists():
        audio_files = [f for f in INPUT_AUDIO_DIR.iterdir()
                       if f.suffix.lower() in AUDIO_EXTENSIONS]
        print(f"  Audio files: {len(audio_files)} in input_audio/")

    # ChromaDB stats
    collection = _get_collection()
    if collection and collection.count() > 0:
        print(f"  Embeddings:  {collection.count()} chunks in ChromaDB")
    else:
        print("  Embeddings:  none (run: python embed.py)")

    print(f"\n  Commands: /quit, /cost, /sessions")
    print("=" * 60)
    print()


async def interactive_loop(model: str):
    """Multi-turn interactive REPL."""
    print_banner()
    options = create_options(model)
    total_cost = 0.0

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Special commands
            if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                print("Goodbye!")
                break

            if user_input.lower() == "/cost":
                print(f"  Total API cost this session: ${total_cost:.4f}")
                continue

            if user_input.lower() == "/sessions":
                files = sorted(_transcripts_dir.glob("*.md"))
                if files:
                    for f in files:
                        print(f"  - {f.stem}")
                else:
                    print("  No transcripts found.")
                continue

            # Send query to Claude
            await client.query(user_input)

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            print(f"\nArchivist: {block.text}")
                        elif isinstance(block, ToolUseBlock):
                            print(f"  [searching: {block.name}...]")
                elif isinstance(msg, ResultMessage):
                    if msg.total_cost_usd:
                        total_cost += msg.total_cost_usd

            print()


async def single_query(prompt: str, model: str):
    """One-shot query mode.

    Uses ClaudeSDKClient (streaming mode) so stdin stays open for MCP
    control requests. A plain string prompt closes stdin after sending,
    which breaks MCP tool calls.
    """
    options = create_options(model)

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text)
                    elif isinstance(block, ToolUseBlock):
                        print(f"  [searching: {block.name}...]", file=sys.stderr)
            elif isinstance(msg, ResultMessage):
                if msg.total_cost_usd:
                    print(f"\n[Cost: ${msg.total_cost_usd:.4f}]", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="D&D Campaign Archivist (Claude Agent SDK)")
    parser.add_argument("--query", "-q", type=str, help="Single query mode (non-interactive)")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Claude model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--data-dir", type=Path, help="Base data directory (for remote deployment)")
    args = parser.parse_args()

    # Check for API key (ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN from .env)
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        print("Error: No API credentials found.")
        print("Set one of: ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN (via .env or export)")
        sys.exit(1)

    # Override paths if --data-dir specified
    global _transcripts_dir, _chromadb_dir
    if args.data_dir:
        base = Path(args.data_dir)
        _transcripts_dir = base / "output_transcripts"
        _chromadb_dir = base / "data" / "chromadb"

    if args.query:
        asyncio.run(single_query(args.query, args.model))
    else:
        asyncio.run(interactive_loop(args.model))


if __name__ == "__main__":
    main()
