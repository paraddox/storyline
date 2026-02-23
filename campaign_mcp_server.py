#!/usr/bin/env python3
"""
D&D Campaign Archivist — MCP Server

Exposes campaign search tools via stdio MCP transport.
Used by agent.py with `claude -p --mcp-config ...`.

Set CAMPAIGN_DATA_DIR env var to override base data directory.
"""

import os
import sys
from pathlib import Path

# Load .env before anything else
BASE_DIR = Path(__file__).parent
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k, _v = _k.strip(), _v.strip()
            if _k and _k not in os.environ:
                os.environ[_k] = _v

# Override campaign_cli paths if CAMPAIGN_DATA_DIR is set
data_dir = os.environ.get("CAMPAIGN_DATA_DIR")
if data_dir:
    import campaign_cli
    base = Path(data_dir)
    campaign_cli.TRANSCRIPTS_DIR = base / "output_transcripts"
    campaign_cli.CHROMADB_DIR = base / "data" / "chromadb"
    campaign_cli.PLAYERS_JSON = base / "config" / "players.json"
    campaign_cli.VOICE_BANK_PATH = base / "config" / "voice_bank.json"

from mcp.server.fastmcp import FastMCP
from campaign_cli import _do_sessions, _do_search, _do_semantic, _do_session_content, _do_info

mcp = FastMCP("campaign")


@mcp.tool()
def list_sessions() -> str:
    """List all available campaign session transcripts with dates and sizes."""
    return _do_sessions()


@mcp.tool()
def search_transcripts(query: str, speaker: str = "", max_results: int = 20) -> str:
    """Exact text search across session transcripts. Use for specific names, quotes, and keywords."""
    return _do_search(query, speaker, max_results)


@mcp.tool()
def semantic_search(query: str, n_results: int = 10, speaker: str = "") -> str:
    """Vector similarity search for conceptual/thematic queries across campaign transcripts."""
    return _do_semantic(query, n_results, speaker)


@mcp.tool()
def get_session_content(session: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read the full or partial content of a specific session transcript."""
    return _do_session_content(session, start_line, end_line)


@mcp.tool()
def get_campaign_info() -> str:
    """Get the player/character roster and campaign configuration."""
    return _do_info()


if __name__ == "__main__":
    mcp.run(transport="stdio")
