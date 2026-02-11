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


# --- Subcommands ---

def cmd_sessions(args):
    """List all available session transcripts with sizes and ChromaDB status."""
    files = sorted(TRANSCRIPTS_DIR.glob("*.md"))
    if not files:
        print("No transcripts found. Run: python etl.py && python embed.py")
        return

    print(f"Found {len(files)} session transcript(s):\n")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"- **{f.stem}** ({size_kb:.1f} KB)")

    collection = _get_collection()
    if collection and collection.count() > 0:
        print(f"\nChromaDB: {collection.count()} embedded chunks")
    else:
        print("\nChromaDB: no embeddings yet (run: python embed.py)")


def cmd_search(args):
    """Exact text search across session transcripts."""
    search_query = args.query.lower()
    speaker_filter = (args.speaker or "").lower()
    max_results = min(args.max or 20, 20)

    files = sorted(TRANSCRIPTS_DIR.glob("*.md"))
    if not files:
        print("No transcripts found.")
        return

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
        print(f"No results found for '{args.query}'")
        return

    print(f"Found {len(results)} match(es) for '{args.query}':\n")
    for r in results:
        print(r)


def cmd_semantic(args):
    """Vector similarity search via ChromaDB + Ollama embeddings."""
    collection = _get_collection()
    if not collection or collection.count() == 0:
        print("No embeddings available. Run: python embed.py")
        return

    embedding = _get_embedding(args.query)
    if embedding is None:
        print("Could not generate embedding. Is Ollama running with nomic-embed-text?")
        print("Install: ollama pull nomic-embed-text")
        return

    n_results = min(args.n or 10, 10)
    speaker_filter = args.speaker or ""
    where_filter = {"speaker": speaker_filter} if speaker_filter else None

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"Search error: {e}")
        return

    if not results["documents"] or not results["documents"][0]:
        print(f"No semantic matches for '{args.query}'")
        return

    print(f"Top {len(results['documents'][0])} semantic matches for '{args.query}':\n")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist
        print(
            f"[{meta['session']}:{meta['line_number']}] "
            f"(similarity: {similarity:.2f}) "
            f"**{meta['speaker']}** [{meta['timestamp']}]"
        )
        last_line = doc.strip().split("\n")[-1]
        print(f"  {last_line}\n")


def cmd_info(args):
    """Get the player/character roster and campaign configuration."""
    if VOICE_BANK_PATH.exists():
        with open(VOICE_BANK_PATH) as f:
            bank = json.load(f)

        version = bank.get("meta", {}).get("version", 1)

        if version >= 2:
            # v2: player-indexed
            print("**Campaign Roster** (from voice bank v2):\n")
            for player_name, info in bank.get("players", {}).items():
                active_char = info.get("active_character", "?")
                n_sessions = len(info.get("embeddings", []))
                chars = info.get("characters", {})
                if len(chars) > 1:
                    char_list = ", ".join(
                        f"{c} *" if c == active_char else c for c in chars.keys())
                    print(f"- {player_name}: {char_list} — {n_sessions} session(s) enrolled")
                else:
                    print(f"- {player_name} as {active_char} — {n_sessions} session(s) enrolled")
        else:
            # v1: character-indexed (legacy)
            print("**Campaign Roster** (from voice bank):\n")
            for name, info in bank.get("speakers", {}).items():
                n_sessions = len(info.get("embeddings", []))
                print(f"- {info.get('character', name)} (played by {info.get('player', '?')}) — {n_sessions} session(s) enrolled")

        threshold = bank.get("meta", {}).get("threshold", "?")
        print(f"\nVoice matching threshold: {threshold}")

    elif PLAYERS_JSON.exists():
        with open(PLAYERS_JSON) as f:
            players = json.load(f)

        print("**Campaign Roster** (from static config):\n")
        speaker_map = players.get("speaker_map", {})
        for speaker_id, info in speaker_map.items():
            if isinstance(info, dict):
                print(f"- {info.get('character', '?')} (played by {info.get('player', '?')})")
            else:
                print(f"- {info}")

        print(f"\nSpeaker range: {players.get('min_speakers', '?')}-{players.get('max_speakers', '?')}")
    else:
        print("No voice bank or players.json found.")


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
