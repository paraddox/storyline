#!/usr/bin/env python3
"""
D&D Campaign Archivist — CLI for OpenClaw

Standalone CLI that provides campaign search tools as argparse subcommands.
Designed to be called via OpenClaw's `exec` tool. No claude-agent-sdk dependency.

Usage:
    python3 campaign_cli.py sessions
    python3 campaign_cli.py search "query" [--speaker NAME] [--max N]
    python3 campaign_cli.py semantic "query" [--speaker NAME] [--n N]
    python3 campaign_cli.py info
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).parent
TRANSCRIPTS_DIR = BASE_DIR / "output_transcripts"
CHROMADB_DIR = BASE_DIR / "data" / "chromadb"
PLAYERS_JSON = BASE_DIR / "config" / "players.json"
VOICE_BANK_PATH = BASE_DIR / "config" / "voice_bank.json"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"

SPEAKER_LINE_RE = re.compile(
    r"^\*\*(.+?)\*\*\s+\[(\d{2}:\d{2}:\d{2})\]:\s*(.+)$"
)


def _load_dotenv():
    """Load .env file into os.environ (simple, no dependency)."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


# --- ChromaDB / Ollama helpers ---

_chroma_collection = None


def _get_collection():
    """Lazy-init ChromaDB collection. Returns None on failure."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if not CHROMADB_DIR.exists():
        return None
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
        _chroma_collection = client.get_or_create_collection(
            name="campaign_transcripts",
            metadata={"hnsw:space": "cosine"},
        )
        return _chroma_collection
    except Exception:
        return None


def _get_embedding(text):
    """Get embedding from Ollama. Returns None on failure."""
    try:
        import requests
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except Exception:
        return None


# --- Core functions (return strings, used by MCP server and CLI) ---

def _do_sessions() -> str:
    """List all available session transcripts with sizes and ChromaDB status."""
    files = sorted(TRANSCRIPTS_DIR.glob("*.md"))
    if not files:
        return "No transcripts found. Run: python etl.py && python embed.py"

    lines = [f"Found {len(files)} session transcript(s):\n"]
    for f in files:
        size_kb = f.stat().st_size / 1024
        lines.append(f"- **{f.stem}** ({size_kb:.1f} KB)")

    collection = _get_collection()
    if collection and collection.count() > 0:
        lines.append(f"\nChromaDB: {collection.count()} embedded chunks")
    else:
        lines.append("\nChromaDB: no embeddings yet (run: python embed.py)")

    return "\n".join(lines)


def _do_search(query: str, speaker: str = "", max_results: int = 20) -> str:
    """Exact text search across session transcripts."""
    search_query = query.lower()
    speaker_filter = speaker.lower()
    max_results = min(max_results or 20, 20)

    files = sorted(TRANSCRIPTS_DIR.glob("*.md"))
    if not files:
        return "No transcripts found."

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
        return f"No results found for '{query}'"

    header = f"Found {len(results)} match(es) for '{query}':\n"
    return header + "\n".join(results)


def _do_semantic(query: str, n_results: int = 10, speaker: str = "") -> str:
    """Vector similarity search via ChromaDB + Ollama embeddings."""
    collection = _get_collection()
    if not collection or collection.count() == 0:
        return "No embeddings available. Run: python embed.py"

    embedding = _get_embedding(query)
    if embedding is None:
        return "Could not generate embedding. Is Ollama running with nomic-embed-text?\nInstall: ollama pull nomic-embed-text"

    n_results = min(n_results or 10, 10)
    where_filter = {"speaker": speaker} if speaker else None

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return f"Search error: {e}"

    if not results["documents"] or not results["documents"][0]:
        return f"No semantic matches for '{query}'"

    lines = [f"Top {len(results['documents'][0])} semantic matches for '{query}':\n"]
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist
        lines.append(
            f"[{meta['session']}:{meta['line_number']}] "
            f"(similarity: {similarity:.2f}) "
            f"**{meta['speaker']}** [{meta['timestamp']}]"
        )
        last_line = doc.strip().split("\n")[-1]
        lines.append(f"  {last_line}\n")

    return "\n".join(lines)


def _do_session_content(session: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read the full or partial content of a specific session transcript."""
    filepath = TRANSCRIPTS_DIR / f"{session}.md"
    if not filepath.exists():
        candidates = list(TRANSCRIPTS_DIR.glob("*.md"))
        matches = [f for f in candidates if session.lower() in f.stem.lower()]
        if not matches:
            available = [f.stem for f in candidates]
            return f"Session '{session}' not found. Available: {', '.join(available) or 'none'}"
        filepath = matches[0]

    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()
    total = len(lines)

    start = max(1, start_line or 1)
    end = min(total, end_line if end_line and end_line > 0 else start + 199)

    selected = lines[start - 1:end]
    header = f"**{filepath.stem}** (lines {start}-{end} of {total}):\n\n"
    content = header + "\n".join(selected)

    if end < total:
        content += f"\n\n... ({total - end} more lines. Use start_line={end + 1} to continue)"

    return content


def _do_info() -> str:
    """Get the player/character roster and campaign configuration."""
    lines = []

    if VOICE_BANK_PATH.exists():
        with open(VOICE_BANK_PATH) as f:
            bank = json.load(f)

        version = bank.get("meta", {}).get("version", 1)

        if version >= 2:
            lines.append("**Campaign Roster** (from voice bank v2):\n")
            for player_name, info in bank.get("players", {}).items():
                active_char = info.get("active_character", "?")
                n_sessions = len(info.get("embeddings", []))
                chars = info.get("characters", {})
                if len(chars) > 1:
                    char_list = ", ".join(
                        f"{c} *" if c == active_char else c for c in chars.keys())
                    lines.append(f"- {player_name}: {char_list} — {n_sessions} session(s) enrolled")
                else:
                    lines.append(f"- {player_name} as {active_char} — {n_sessions} session(s) enrolled")
        else:
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
        return "No voice bank or players.json found."

    return "\n".join(lines)


# --- CLI subcommands (thin wrappers around _do_* functions) ---

def cmd_sessions(args):
    print(_do_sessions())

def cmd_search(args):
    print(_do_search(args.query, args.speaker or "", args.max or 20))

def cmd_semantic(args):
    print(_do_semantic(args.query, args.n or 10, args.speaker or ""))

def cmd_info(args):
    print(_do_info())


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="D&D Campaign Archivist CLI (for OpenClaw exec)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # sessions
    sub.add_parser("sessions", help="List session transcripts and ChromaDB status")

    # search
    p_search = sub.add_parser("search", help="Exact text search across transcripts")
    p_search.add_argument("query", help="Text to search for (case-insensitive)")
    p_search.add_argument("--speaker", help="Filter to lines from this speaker")
    p_search.add_argument("--max", type=int, default=20, help="Max results (default: 20)")

    # semantic
    p_sem = sub.add_parser("semantic", help="Vector similarity search via ChromaDB")
    p_sem.add_argument("query", help="Natural language search query")
    p_sem.add_argument("--speaker", help="Filter to a specific speaker")
    p_sem.add_argument("--n", type=int, default=10, help="Number of results (default: 10)")

    # info
    sub.add_parser("info", help="Show player/character roster")

    args = parser.parse_args()

    commands = {
        "sessions": cmd_sessions,
        "search": cmd_search,
        "semantic": cmd_semantic,
        "info": cmd_info,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
