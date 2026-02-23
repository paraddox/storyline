#!/usr/bin/env python3
"""
D&D Campaign Archivist — Claude CLI
Interactive agent that searches campaign transcripts using hybrid
text + vector search. Powered by Claude CLI (`claude -p`) with MCP tools.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from etl import _load_dotenv

_load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
TRANSCRIPTS_DIR = BASE_DIR / "output_transcripts"
INPUT_AUDIO_DIR = BASE_DIR / "input_audio"
DEFAULT_MODEL = "claude-opus-4-6"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".mp4", ".mkv", ".webm"}

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

ALLOWED_TOOLS = [
    "mcp__campaign__list_sessions",
    "mcp__campaign__search_transcripts",
    "mcp__campaign__semantic_search",
    "mcp__campaign__get_session_content",
    "mcp__campaign__get_campaign_info",
    "Read", "Grep", "Glob",
]


def _find_claude_cli() -> str:
    """Find the claude CLI binary. Exits with error if not found."""
    cli_path = shutil.which("claude")
    if not cli_path:
        print("Error: Claude CLI not found. Install it first:")
        print("  npm install -g @anthropic-ai/claude-code")
        sys.exit(1)
    return cli_path


def _build_mcp_config(data_dir: Path | None) -> str:
    """Build inline MCP config JSON for claude -p."""
    env = {}
    if data_dir:
        env["CAMPAIGN_DATA_DIR"] = str(data_dir)
    return json.dumps({
        "mcpServers": {
            "campaign": {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(BASE_DIR / "campaign_mcp_server.py")],
                "env": env,
            }
        }
    })


def _build_cmd(cli_path: str, model: str, data_dir: Path | None,
               session_id: str | None = None) -> list[str]:
    """Build the claude -p command with all flags."""
    cmd = [
        cli_path, "-p",
        "--output-format", "json",
        "--model", model,
        "--max-turns", "15",
        "--permission-mode", "bypassPermissions",
        "--mcp-config", _build_mcp_config(data_dir),
        "--allowedTools", ",".join(ALLOWED_TOOLS),
    ]
    if session_id:
        cmd.extend(["--resume", session_id])
    else:
        cmd.extend(["--system-prompt", SYSTEM_PROMPT])
    return cmd


def _run_query(cmd: list[str], prompt: str) -> dict:
    """Execute claude -p and return parsed JSON response."""
    env = {**os.environ, "CLAUDECODE": ""}  # Unset to allow nested invocation
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(BASE_DIR),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"is_error": True, "result": "Query timed out (300s limit)"}

    if result.returncode != 0:
        stderr = result.stderr.strip()[:500]
        return {"is_error": True, "result": f"Claude CLI error (exit {result.returncode}): {stderr}"}

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        # Fall back to raw text output
        return {"result": result.stdout.strip(), "is_error": False}


# ============================================================
# UI
# ============================================================

def print_banner(transcripts_dir: Path):
    """Print startup banner with session info."""
    print("=" * 60)
    print("  D&D Campaign Archivist")
    print("  Powered by Claude CLI")
    print("=" * 60)

    transcripts = list(transcripts_dir.glob("*.md"))
    print(f"\n  Sessions:    {len(transcripts)} transcript(s)")

    if INPUT_AUDIO_DIR.exists():
        audio_files = [f for f in INPUT_AUDIO_DIR.iterdir()
                       if f.suffix.lower() in AUDIO_EXTENSIONS]
        print(f"  Audio files: {len(audio_files)} in input_audio/")

    print(f"\n  Commands: /quit, /cost, /sessions")
    print("=" * 60)
    print()


def interactive_loop(model: str, data_dir: Path | None):
    """Multi-turn interactive REPL."""
    transcripts_dir = (data_dir / "output_transcripts") if data_dir else TRANSCRIPTS_DIR
    print_banner(transcripts_dir)

    cli_path = _find_claude_cli()
    session_id = None
    total_cost = 0.0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/cost":
            print(f"  Total API cost this session: ${total_cost:.4f}")
            continue

        if user_input.lower() == "/sessions":
            files = sorted(transcripts_dir.glob("*.md"))
            if files:
                for f in files:
                    print(f"  - {f.stem}")
            else:
                print("  No transcripts found.")
            continue

        cmd = _build_cmd(cli_path, model, data_dir, session_id)
        response = _run_query(cmd, user_input)

        if response.get("is_error"):
            print(f"\nArchivist: [Error: {response.get('result', 'unknown error')}]\n")
            continue

        session_id = response.get("session_id", session_id)
        cost = response.get("total_cost_usd", 0)
        if cost:
            total_cost += cost

        text = response.get("result", "")
        if text:
            print(f"\nArchivist: {text}\n")


def single_query(prompt: str, model: str, data_dir: Path | None,
                 output: Path | None = None):
    """One-shot query mode."""
    cli_path = _find_claude_cli()
    cmd = _build_cmd(cli_path, model, data_dir)
    response = _run_query(cmd, prompt)

    if response.get("is_error"):
        print(f"Error: {response.get('result', 'unknown error')}", file=sys.stderr)
        sys.exit(1)

    text = response.get("result", "")
    print(text)

    cost = response.get("total_cost_usd", 0)
    if cost:
        print(f"\n[Cost: ${cost:.4f}]", file=sys.stderr)

    if output and text:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"\nSaved to: {output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="D&D Campaign Archivist (Claude CLI)")
    parser.add_argument("--query", "-q", type=str, help="Single query mode (non-interactive)")
    parser.add_argument("--output", "-o", type=Path, help="Save query output to file (use with --query)")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Claude model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--data-dir", type=Path, help="Base data directory (for remote deployment)")
    args = parser.parse_args()

    # Check for claude CLI
    _find_claude_cli()

    if args.query:
        single_query(args.query, args.model, args.data_dir, args.output)
    else:
        interactive_loop(args.model, args.data_dir)


if __name__ == "__main__":
    main()
