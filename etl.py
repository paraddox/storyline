#!/usr/bin/env python3
"""
D&D Session ETL Pipeline — Unified
Handles enrollment, processing, character switches, and embedding in a single script.
Monitors input_audio/ for new files, transcribes via WhisperX,
identifies speakers via voice bank (player-indexed v2) or enrollment,
detects character switches, formats Markdown, embeds to ChromaDB,
and uploads to Open WebUI.
"""

import json
import os
import sys
import time
import argparse
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import requests

# --- .env loading ---
def _load_dotenv():
    """Load .env file into os.environ (simple, no dependency)."""
    env_path = Path(__file__).parent / ".env"
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

_load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
INPUT_DIR = BASE_DIR / "input_audio"
OUTPUT_DIR = BASE_DIR / "output_transcripts"
PLAYERS_JSON = CONFIG_DIR / "players.json"
PROCESSED_LOG = CONFIG_DIR / "processed.json"
VOICE_BANK_PATH = CONFIG_DIR / "voice_bank.json"

WHISPERX_URL = os.environ.get("WHISPERX_URL", "http://localhost:9876")
OPENWEBUI_URL = os.environ.get("OPENWEBUI_URL", "http://localhost:3456")
OPENWEBUI_API_KEY = os.environ.get("OPENWEBUI_API_KEY", "")
KNOWLEDGE_COLLECTION_ID = os.environ.get("KNOWLEDGE_COLLECTION_ID", "")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".mp4", ".mkv", ".webm"}


# ============================================================
# LLM Helpers
# ============================================================

def has_anthropic_credentials() -> bool:
    """Check if any Anthropic API credentials are available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))


def call_claude_llm(prompt: str, model: str = "claude-sonnet-4-5-20250929", max_tokens: int = 2048) -> str | None:
    """Call Claude and return the text response.

    Uses ANTHROPIC_API_KEY directly if available (faster, no subprocess).
    Falls back to the `claude` CLI for CLAUDE_CODE_OAUTH_TOKEN.
    Returns the response text, or None on any failure.
    """
    import shutil
    import subprocess

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            print(f"  LLM API call failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"  LLM response parsing failed: {e}")
            return None

    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not oauth_token:
        return None

    cli_path = shutil.which("claude")
    if not cli_path:
        print("  Claude CLI not found; cannot use CLAUDE_CODE_OAUTH_TOKEN for LLM calls")
        return None

    try:
        # Use stdin instead of -p to avoid OS argument length limits on long transcripts
        result = subprocess.run(
            [cli_path, "--output-format", "text",
             "--model", model, "--max-turns", "1",
             "--permission-mode", "bypassPermissions"],
            input=prompt,
            capture_output=True, text=True, timeout=300,
            env={**os.environ},
        )
        if result.returncode != 0:
            print(f"  Claude CLI failed (exit {result.returncode}): {result.stderr[:200]}")
            return None
        output = result.stdout.strip()
        if not output:
            print(f"  Claude CLI returned empty output (stderr: {result.stderr[:300]})")
        return output
    except subprocess.TimeoutExpired:
        print("  Claude CLI timed out")
        return None
    except Exception as e:
        print(f"  Claude CLI error: {e}")
        return None


def _parse_llm_json(content: str) -> dict | None:
    """Parse JSON from LLM response, extracting from markdown fences or free text."""
    content = content.strip()

    # Try 1: Direct parse (entire response is JSON)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from markdown code fences
    import re
    fence_match = re.search(r'```(?:json)?\s*\n(.*?)```', content, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try 3: Find all top-level { ... } blocks and try each (largest first)
    candidates = []
    i = 0
    while i < len(content):
        if content[i] == '{':
            depth = 0
            for j in range(i, len(content)):
                if content[j] == '{':
                    depth += 1
                elif content[j] == '}':
                    depth -= 1
                    if depth == 0:
                        candidates.append(content[i:j + 1])
                        i = j + 1
                        break
            else:
                break
        else:
            i += 1

    # Try largest candidate first (most likely the full JSON response)
    for candidate in sorted(candidates, key=len, reverse=True):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    print(f"  LLM response parsing failed: no valid JSON found")
    print(f"  Raw LLM response (first 500 chars): {content[:500]}")
    return None


# ============================================================
# File / Config Helpers
# ============================================================

def load_players() -> dict:
    """Load speaker mapping from players.json."""
    with open(PLAYERS_JSON) as f:
        return json.load(f)


def load_processed() -> dict:
    """Load the log of already-processed files."""
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG) as f:
            return json.load(f)
    return {}


def save_processed(processed: dict):
    """Save the processed files log."""
    with open(PROCESSED_LOG, "w") as f:
        json.dump(processed, f, indent=2)


def file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file for dedup."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================
# Voice Bank v2 — Player-Indexed
# ============================================================

def create_empty_voice_bank() -> dict:
    """Return an empty v2 voice bank structure."""
    return {
        "meta": {
            "version": 2,
            "embedding_dim": 256,
            "threshold": 0.55,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        },
        "players": {},
    }


def compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """Compute arithmetic mean of embedding vectors."""
    arr = np.array(embeddings)
    return arr.mean(axis=0).tolist()


def migrate_voice_bank_v1_to_v2(bank: dict) -> dict:
    """Convert v1 (character-indexed) voice bank to v2 (player-indexed)."""
    if bank.get("meta", {}).get("version", 1) >= 2:
        return bank

    print("  Migrating voice bank v1 -> v2 (player-indexed)...")

    old_speakers = bank.get("speakers", {})
    players = {}

    for character_name, info in old_speakers.items():
        player = info.get("player", character_name)
        embedding_entries = info.get("embeddings", [])
        centroid = info.get("centroid", [])

        if player in players:
            # Player already exists (multiple characters for same player)
            existing = players[player]
            existing["characters"][character_name] = {
                "first_session": embedding_entries[0]["session"] if embedding_entries else "",
                "last_session": embedding_entries[-1]["session"] if embedding_entries else "",
            }
            # Tag existing embeddings with character name
            for entry in embedding_entries:
                entry["character"] = character_name
            existing["embeddings"].extend(embedding_entries)
            # Recompute centroid across all embeddings
            all_vectors = [e["vector"] for e in existing["embeddings"]]
            if all_vectors:
                existing["centroid"] = compute_centroid(all_vectors)
            # Update active_character to the latest
            existing["active_character"] = character_name
        else:
            # Tag embeddings with character name
            for entry in embedding_entries:
                entry["character"] = character_name

            sessions = [e["session"] for e in embedding_entries]
            players[player] = {
                "active_character": character_name,
                "characters": {
                    character_name: {
                        "first_session": sessions[0] if sessions else "",
                        "last_session": sessions[-1] if sessions else "",
                    }
                },
                "embeddings": embedding_entries,
                "centroid": centroid,
            }

    new_bank = {
        "meta": {
            **bank.get("meta", {}),
            "version": 2,
            "last_updated": datetime.now().isoformat(),
        },
        "players": players,
    }

    print(f"  Migrated {len(old_speakers)} character(s) -> {len(players)} player(s)")
    return new_bank


def load_voice_bank() -> dict | None:
    """Load the voice bank, auto-migrating v1 to v2 if needed."""
    if not VOICE_BANK_PATH.exists():
        return None
    with open(VOICE_BANK_PATH) as f:
        bank = json.load(f)

    version = bank.get("meta", {}).get("version", 1)
    if version < 2:
        bank = migrate_voice_bank_v1_to_v2(bank)
        save_voice_bank(bank)

    return bank


def save_voice_bank(bank: dict):
    """Save voice bank to disk with updated timestamp."""
    bank["meta"]["last_updated"] = datetime.now().isoformat()
    with open(VOICE_BANK_PATH, "w") as f:
        json.dump(bank, f, indent=2)


# ============================================================
# Transcription
# ============================================================

def transcribe(audio_path: Path, players: dict) -> dict:
    """Send audio to WhisperX API for transcription + diarization."""
    min_speakers = players.get("min_speakers", 2)
    max_speakers = players.get("max_speakers", 6)

    print(f"  Transcribing: {audio_path.name}")
    print(f"  Speaker range: {min_speakers}-{max_speakers}")

    params = {
        "task": "transcribe",
        "language": "en",
        "diarize": "true",
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "output": "json",
        "return_speaker_embeddings": "true",
    }

    # Pass diarization tuning params from players.json if present
    clustering_threshold = players.get("clustering_threshold")
    if clustering_threshold is not None:
        params["clustering_threshold"] = clustering_threshold
        print(f"  Clustering threshold: {clustering_threshold}")

    diarization_model = players.get("diarization_model")
    if diarization_model:
        params["diarization_model"] = diarization_model
        print(f"  Diarization model: {diarization_model}")

    with open(audio_path, "rb") as f:
        response = requests.post(
            f"{WHISPERX_URL}/asr",
            files={"audio_file": (audio_path.name, f)},
            params=params,
            timeout=3600,
        )

    response.raise_for_status()
    return response.json()


# ============================================================
# Speaker Matching (v2 — player-indexed)
# ============================================================

def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms == 0:
        return 0.0
    return float(dot / norms)


def match_speakers_by_embedding(speaker_embeddings: dict, voice_bank: dict) -> dict:
    """Match SPEAKER_XX labels to known players via multi-embedding scoring.

    Iterates bank["players"], compares embeddings, returns player identity.
    Same 0.6/0.4 blended scoring algorithm as before.

    Returns dict: SPEAKER_XX -> {
        "player": str, "character": str (active_character),
        "confidence": float, "best_session": float, "centroid": float
    }
    """
    threshold = voice_bank.get("meta", {}).get("threshold", 0.55)
    bank_players = voice_bank.get("players", {})

    matches = {}
    for speaker_id, embedding in speaker_embeddings.items():
        best_player = None
        best_score = -1.0
        best_session_score = 0.0
        best_centroid_score = 0.0

        for player_name, info in bank_players.items():
            stored = info.get("embeddings", [])
            centroid = info.get("centroid", [])
            if not stored and not centroid:
                continue

            # Score against every stored embedding
            if stored:
                all_scores = [cosine_similarity(embedding, e["vector"]) for e in stored]
                session_score = max(all_scores)
            else:
                session_score = 0.0

            # Score against centroid
            if centroid:
                centroid_score = cosine_similarity(embedding, centroid)
            else:
                centroid_score = session_score

            # Blend: favor best individual match but anchor with centroid
            score = 0.6 * session_score + 0.4 * centroid_score

            if score > best_score:
                best_score = score
                best_player = player_name
                best_session_score = session_score
                best_centroid_score = centroid_score

        if best_score >= threshold and best_player:
            active_char = bank_players[best_player].get("active_character", best_player)
            matches[speaker_id] = {
                "player": best_player,
                "character": active_char,
                "confidence": best_score,
                "best_session": best_session_score,
                "centroid": best_centroid_score,
            }

    # Note: multiple SPEAKER_XX IDs may legitimately map to the same player
    # (diarization splits, especially for the DM). We allow this — no collision reset.

    return matches


def update_voice_bank(voice_bank: dict, speaker_embeddings: dict, matches: dict,
                      session_name: str, character_overrides: dict | None = None):
    """Append high-confidence embeddings to voice bank and recompute centroids.

    Args:
        character_overrides: optional dict {player_name: new_character} from character switch detection
    """
    update_threshold = 0.60
    updated = False
    bank_players = voice_bank.get("players", {})

    for speaker_id, match in matches.items():
        if match["confidence"] < update_threshold:
            continue
        player = match["player"]
        if player.startswith("SPEAKER_"):
            continue
        if player not in bank_players:
            continue

        embedding = speaker_embeddings.get(speaker_id)
        if embedding is None:
            continue

        player_data = bank_players[player]
        existing_sessions = {e["session"] for e in player_data.get("embeddings", [])}
        if session_name in existing_sessions:
            continue

        # Determine character for this embedding
        character = match["character"]
        if character_overrides and player in character_overrides:
            character = character_overrides[player]

        # Append embedding tagged with character
        player_data["embeddings"].append({
            "session": session_name,
            "vector": embedding,
            "character": character,
            "confidence": match["confidence"],
            "auto": True,
        })

        # Update character tracking
        if character not in player_data.get("characters", {}):
            player_data.setdefault("characters", {})[character] = {
                "first_session": session_name,
                "last_session": session_name,
            }
        else:
            player_data["characters"][character]["last_session"] = session_name

        # Recompute centroid across ALL embeddings (same voice regardless of character)
        all_vectors = [e["vector"] for e in player_data["embeddings"]]
        player_data["centroid"] = compute_centroid(all_vectors)
        updated = True
        print(f"  Voice bank updated: {player}/{character} ({len(all_vectors)} sessions)")

    if updated:
        save_voice_bank(voice_bank)


# ============================================================
# LLM Speaker Identification (processing mode — with voice bank)
# ============================================================

def identify_speakers_with_llm(segments, voice_matches, voice_bank, uncertain_ids, players=None):
    """Use Claude to identify speakers from transcript context.

    Called as a fallback when voice matching leaves unmatched or low-confidence speakers.
    Returns dict: SPEAKER_XX -> {"character": str, "player": str, "confidence": float, "source": "llm"}
    """
    if not has_anthropic_credentials():
        print("  Skipping LLM identification: no ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN")
        return {}

    if not uncertain_ids:
        return {}

    print(f"  Invoking Claude for speaker identification ({len(uncertain_ids)} uncertain)...")

    # Build the identified speakers section
    identified_lines = []
    for sid, match in voice_matches.items():
        if not match.get("character", "").startswith("SPEAKER_"):
            player = match.get("player", "")
            conf = match.get("confidence", 0)
            identified_lines.append(
                f"- {sid} -> {match['character']} (played by {player}), confidence: {conf:.2f}")

    # Build the "need identification" section
    need_id_lines = []
    for sid in uncertain_ids:
        if sid in voice_matches and not voice_matches[sid].get("character", "").startswith("SPEAKER_"):
            conf = voice_matches[sid]["confidence"]
            char = voice_matches[sid]["character"]
            need_id_lines.append(f"- {sid} (low confidence: {conf:.2f}, tentatively matched to \"{char}\")")
        else:
            need_id_lines.append(f"- {sid} (unmatched)")

    # Build the roster from v2 bank (player -> active_character)
    bank_players = voice_bank.get("players", {})
    matched_players = {m["player"] for m in voice_matches.values()
                       if not m.get("player", "").startswith("SPEAKER_") and m.get("confidence", 0) >= 0.70}
    roster_lines = []
    for player_name, info in bank_players.items():
        if player_name not in matched_players:
            active_char = info.get("active_character", player_name)
            roster_lines.append(f"- {active_char} (played by {player_name})")

    # Add players from players.json not yet in voice bank
    if players:
        for speaker_id, info in players.get("speaker_map", {}).items():
            if isinstance(info, dict):
                character = info.get("character", "")
                player = info.get("player", "")
            else:
                character = info
                player = info
            if player and player not in bank_players and player not in matched_players:
                roster_lines.append(f"- {character} (played by {player}) [NEW - not yet enrolled]")

    # Build full transcript
    transcript_lines = []
    for seg in segments:
        start = seg.get("start", 0)
        ts = format_timestamp(start)
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if text:
            transcript_lines.append(f"[{ts}] {speaker}: {text}")

    prompt = f"""You are identifying unknown speakers in a D&D session transcript.

## Already Identified (by voice matching)
{chr(10).join(identified_lines) if identified_lines else "(none)"}

## Need Identification
{chr(10).join(need_id_lines)}

## Campaign Roster (not yet confidently matched)
{chr(10).join(roster_lines) if roster_lines else "(all roster members already matched)"}

## Full Session Transcript
{chr(10).join(transcript_lines)}

## Instructions
For each speaker that needs identification, analyze the full transcript for:
- Someone calling a name and that speaker responding
- DM patterns: scene descriptions, "roll for...", NPC dialogue, narration
- Character-specific speech: "I cast...", "my barbarian...", class abilities
- Out-of-character references: players using real names
- Process of elimination: if all other players are accounted for

A player may voice multiple characters in a session (main character + sidekicks/companions).
Report only their PRIMARY character — the main PC they are playing, not sidekicks or NPCs they temporarily control.

For each speaker, respond with ONLY valid JSON (no markdown fences):
{{"SPEAKER_XX": {{"character": "Name", "player": "PlayerName", "reasoning": "brief explanation"}}}}

Only include speakers you can identify with reasonable confidence. Omit those you cannot."""

    content = call_claude_llm(prompt)
    if not content:
        print("  LLM call returned no content for speaker identification")
        return {}

    llm_result = _parse_llm_json(content)
    if llm_result is None:
        print("  Failed to parse LLM response for speaker identification")
        return {}
    if not llm_result:
        print("  LLM could not identify any uncertain speakers")
        return {}

    # Build full known roster (voice bank + players.json)
    known_players = set(bank_players.keys())
    known_characters = set()
    for info in bank_players.values():
        known_characters.add(info.get("active_character", ""))
        known_characters.update(info.get("characters", {}).keys())
    if players:
        for info in players.get("speaker_map", {}).values():
            if isinstance(info, dict):
                known_characters.add(info.get("character", ""))
                known_players.add(info.get("player", ""))
            else:
                known_characters.add(info)

    results = {}
    print(f"  LLM returned keys: {list(llm_result.keys())}, uncertain_ids: {uncertain_ids}")
    for sid, info in llm_result.items():
        if sid not in uncertain_ids:
            print(f"    {sid}: not in uncertain_ids, skipping")
            continue
        if not isinstance(info, dict):
            print(f"    {sid}: value is not a dict ({type(info).__name__}), skipping")
            continue
        character = info.get("character", "")
        player = info.get("player", character)
        reasoning = info.get("reasoning", "")
        if character in known_characters or player in known_players:
            results[sid] = {
                "character": character,
                "player": player,
                "confidence": 0.50,
                "source": "llm",
            }
            print(f"    {sid} -> {character} (LLM, reasoning: \"{reasoning}\")")
        else:
            print(f"    {sid}: LLM suggested \"{character}\" but not in roster, skipping")

    return results


# ============================================================
# Character Switch Detection
# ============================================================

def verify_characters_with_llm(segments, matches, voice_bank, session_name) -> dict:
    """Lightweight LLM check to detect character switches.

    After voice matching assigns active_character to each player, this function
    verifies against actual dialogue. Returns dict {player_name: new_character}
    for any detected switches. Empty dict if no switches detected.
    """
    if not has_anthropic_credentials():
        return {}

    bank_players = voice_bank.get("players", {})

    # Build current assignments + character history
    assignments = []
    for sid, match in matches.items():
        player = match.get("player", "")
        character = match.get("character", "")
        if player.startswith("SPEAKER_"):
            continue
        player_info = bank_players.get(player, {})
        char_history = list(player_info.get("characters", {}).keys())
        history_str = " -> ".join(char_history) if len(char_history) > 1 else "(no prior switches)"
        assignments.append(
            f"- {sid} = {player}, currently playing: {character}, history: {history_str}")

    if not assignments:
        return {}

    # Build condensed transcript (first/last ~50 lines from each speaker for efficiency)
    transcript_lines = []
    for seg in segments:
        start = seg.get("start", 0)
        ts = format_timestamp(start)
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if text:
            transcript_lines.append(f"[{ts}] {speaker}: {text}")

    # Truncate to ~500 lines for token efficiency
    if len(transcript_lines) > 500:
        # Take first 250 + last 250
        transcript_lines = transcript_lines[:250] + ["... (middle truncated) ..."] + transcript_lines[-250:]

    prompt = f"""You are verifying character assignments in a D&D session transcript.

## Current Assignments
{chr(10).join(assignments)}

## Session: {session_name}

## Transcript Excerpt
{chr(10).join(transcript_lines)}

## Instructions
Check if any player appears to be playing a DIFFERENT primary character than what's listed above.

A player may voice multiple characters in a session (main character + sidekicks/companions/NPCs).
Report only PRIMARY character changes — the main PC they are playing, not sidekicks or NPCs they temporarily control.

If all assignments look correct, respond with: {{"switches": {{}}}}

If a player switched their primary character, respond with:
{{"switches": {{"PlayerName": "NewCharacterName"}}, "reasoning": {{"PlayerName": "brief explanation"}}}}

Respond with ONLY valid JSON (no markdown fences)."""

    print(f"  Verifying character assignments via LLM...")
    content = call_claude_llm(prompt, max_tokens=1024)
    if not content:
        return {}

    result = _parse_llm_json(content)
    if not result:
        return {}

    switches = result.get("switches", {})
    reasoning = result.get("reasoning", {})

    if switches:
        for player, new_char in switches.items():
            reason = reasoning.get(player, "")
            print(f"  Character switch detected: {player} -> {new_char} ({reason})")

            # Update voice bank: add new character, finalize old, update active
            if player in bank_players:
                player_data = bank_players[player]
                old_char = player_data.get("active_character", "")

                # Finalize old character's last_session
                if old_char and old_char in player_data.get("characters", {}):
                    player_data["characters"][old_char]["last_session"] = session_name

                # Add new character
                player_data.setdefault("characters", {})[new_char] = {
                    "first_session": session_name,
                    "last_session": session_name,
                }
                player_data["active_character"] = new_char

        save_voice_bank(voice_bank)
    else:
        print(f"  Character assignments verified: no switches detected")

    return switches


# ============================================================
# Enrollment Functions (absorbed from enroll.py)
# ============================================================

def get_speaker_samples(segments: list, speaker_id: str, n: int = 10) -> list[dict]:
    """Get N evenly-spaced sample dialogue lines for a speaker, with timestamps."""
    speaker_segs = [s for s in segments
                    if s.get("speaker") == speaker_id
                    and len(s.get("text", "").strip()) > 20]
    if not speaker_segs:
        return []
    if len(speaker_segs) <= n:
        chosen = speaker_segs
    else:
        indices = [int(i * (len(speaker_segs) - 1) / (n - 1)) for i in range(n)]
        chosen = [speaker_segs[i] for i in indices]
    return [{"text": s["text"].strip()[:120],
             "start": s.get("start", 0),
             "end": s.get("end", 0)} for s in chosen]


def get_speaker_stats(segments: list) -> dict:
    """Segment count + total speaking time per speaker."""
    stats = {}
    for seg in segments:
        sid = seg.get("speaker", "UNKNOWN")
        if sid not in stats:
            stats[sid] = {"count": 0, "total_time": 0.0}
        stats[sid]["count"] += 1
        stats[sid]["total_time"] += seg.get("end", 0) - seg.get("start", 0)
    return stats


def pre_identify_speakers_for_enrollment(
    segments: list, sorted_speakers: list, stats: dict,
    dm_hint_id: str | None, players: dict,
) -> dict:
    """Use Claude to pre-identify speakers from transcript context during enrollment.

    No voice bank context — identification is purely transcript-based.
    Returns dict: SPEAKER_XX -> {"character": str|None, "player": str|None, "reasoning": str}.
    """
    if not has_anthropic_credentials():
        return {}

    print(f"\n  Invoking Claude for speaker pre-identification ({len(sorted_speakers)} speakers)...")

    # Build speaker stats section
    stats_lines = []
    for sid in sorted_speakers:
        s = stats.get(sid, {"count": 0, "total_time": 0.0})
        total_secs = s["total_time"]
        mins = int(total_secs // 60)
        secs = int(total_secs % 60)
        dm_tag = " [likely DM — most segments by 2x+]" if sid == dm_hint_id else ""
        stats_lines.append(f"- {sid}: {s['count']} segments, {mins}:{secs:02d} speaking time{dm_tag}")

    # Build campaign roster — always include player names, even when character is placeholder
    roster_lines = []
    placeholder_patterns = {"Character ", "Player "}
    for speaker_id, info in players.get("speaker_map", {}).items():
        if isinstance(info, dict):
            character = info.get("character", "")
            player = info.get("player", "")
        else:
            character = info
            player = info
        # Skip fully placeholder entries
        player_is_placeholder = any(player.startswith(p) for p in placeholder_patterns)
        char_is_placeholder = any(character.startswith(p) for p in placeholder_patterns)
        if player_is_placeholder and char_is_placeholder:
            continue
        if char_is_placeholder:
            roster_lines.append(f"- player: {player} (character name unknown)")
        elif character:
            roster_lines.append(f"- {character} (player: {player})")

    # Build full transcript
    transcript_lines = []
    for seg in segments:
        start = seg.get("start", 0)
        ts = format_timestamp(start)
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if text:
            transcript_lines.append(f"[{ts}] {speaker}: {text}")

    roster_section = "\n".join(roster_lines) if roster_lines else "(no roster available)"
    n_roster = len(roster_lines)

    # Build a comma-separated list of known player names for the prompt
    known_player_names = []
    for _sid, info in players.get("speaker_map", {}).items():
        p = info.get("player", "") if isinstance(info, dict) else info
        if p and not any(p.startswith(x) for x in ("Player ", "Character ")):
            known_player_names.append(p)
    player_names_str = ", ".join(known_player_names) if known_player_names else "(unknown)"

    prompt = f"""You are identifying speakers in a D&D session transcript during enrollment.
No voice bank exists yet — you must identify speakers purely from dialogue context.

## Speaker Stats
{chr(10).join(stats_lines)}

## Campaign Roster
{roster_section}

## Full Session Transcript
{chr(10).join(transcript_lines)}

## CRITICAL RULES
1. The "player" field MUST be a real person's name from the Campaign Roster above (e.g. {player_names_str}).
   Do NOT use character names, "Assistant", "DM", or invented names as the player value.
   The player is the real human behind the microphone.

2. Each identified SPEAKER_XX must map to a DIFFERENT player — never assign the same player to
   two different SPEAKER_XX IDs UNLESS one is clearly a diarization duplicate of the other.

3. IMPORTANT: The diarization system sometimes splits one person into multiple SPEAKER_XX IDs
   (especially the DM who uses many voices for NPCs). There may be MORE speaker IDs than actual
   people present. If a speaker's dialogue is clearly narration/NPC voices and overlaps with the DM,
   mark them as the same DM player — do NOT force-assign an absent roster player.

4. Not all roster players may be present in every session. Only assign a player if you have
   evidence they are actually speaking. Do NOT use process of elimination to force-assign a
   player you have no evidence for.

## Instructions
For each SPEAKER_XX, analyze the full transcript for:
- Someone calling a name and that speaker responding in the next line(s)
- DM patterns: scene descriptions, "roll for...", NPC dialogue, narration, running combat
- Character-specific speech: "I cast...", "my barbarian...", referencing class abilities
- Out-of-character references: players using real names

A player may voice multiple characters in a session (main character + sidekicks/companions/NPCs).
Report only their PRIMARY character — the main PC they are playing, not sidekicks or NPCs they temporarily control.

For the DM/Game Master, use character="Narrator" and the DM's real player name from the roster.
If you can identify the character name from dialogue, great. If not, set character to the player name.
Always provide BOTH fields — use your best guess rather than null.

Respond with ONLY valid JSON (no markdown fences):
{{"SPEAKER_XX": {{"character": "Name", "player": "PlayerName", "reasoning": "brief explanation"}}}}

Include ALL speakers you can confidently identify. Omit those you truly cannot identify.
It is OK to assign the same player (especially the DM) to multiple speaker IDs if the evidence supports it."""

    content = call_claude_llm(prompt)
    if not content:
        return {}

    llm_result = _parse_llm_json(content)
    if not llm_result:
        return {}

    results = {}
    for sid, info in llm_result.items():
        if sid not in stats:
            continue
        character = info.get("character") or None
        player = info.get("player") or None
        reasoning = info.get("reasoning", "")
        if character and character.lower() == "null":
            character = None
        if player and player.lower() == "null":
            player = None
        if character or player:
            results[sid] = {
                "character": character,
                "player": player,
                "reasoning": reasoning,
            }

    return results


def enroll_speakers(segments, speaker_embeddings, players, session_name):
    """Enrollment orchestrator: identify speakers, create v2 voice bank.

    Player-first prompts. Creates a v2 bank with player as the primary key.
    Returns (voice_bank, speaker_map) where speaker_map maps SPEAKER_XX -> {character, player}.
    """
    stats = get_speaker_stats(segments)
    sorted_speakers = sorted(
        speaker_embeddings.keys(),
        key=lambda sid: stats.get(sid, {}).get("count", 0),
        reverse=True,
    )

    # Determine if top speaker is likely the DM (>2x segments of next speaker)
    top_counts = [stats.get(sid, {}).get("count", 0) for sid in sorted_speakers]
    dm_hint_id = None
    if len(top_counts) >= 2 and top_counts[0] > 2 * top_counts[1]:
        dm_hint_id = sorted_speakers[0]

    # Build set of known player names from players.json (needed for validation below)
    known_players = set()
    for _sid, info in players.get("speaker_map", {}).items():
        if isinstance(info, dict):
            p = info.get("player", "")
        else:
            p = info
        if p and not any(p.startswith(pat) for pat in ("Player ", "Character ")):
            known_players.add(p)

    # LLM pre-identification
    llm_ids = pre_identify_speakers_for_enrollment(
        segments, sorted_speakers, stats, dm_hint_id, players)
    if llm_ids:
        print("\n  LLM Pre-identification Results:")
        for sid, info in llm_ids.items():
            char = info.get("character") or "?"
            plyr = info.get("player") or "?"
            print(f"    {sid} -> {char} (player: {plyr}) -- {info.get('reasoning', '')}")

    # Validate LLM results: reject entries where player name isn't in the known roster
    invalid_sids = []
    for sid, info in llm_ids.items():
        p = info.get("player") or ""
        if p and p.lower() not in {kp.lower() for kp in known_players}:
            print(f"  Warning: LLM assigned unknown player '{p}' to {sid}, dropping for interactive")
            invalid_sids.append(sid)
    for sid in invalid_sids:
        del llm_ids[sid]

    # When the LLM assigns the same player to multiple SPEAKER_XX IDs (e.g. DM split
    # across diarization segments), keep all of them — they'll be merged under one player
    # entry in the voice bank. Just log it for visibility.
    player_to_sids = {}
    for sid, info in llm_ids.items():
        p = (info.get("player") or "").lower()
        if p:
            player_to_sids.setdefault(p, []).append(sid)
    for p, sids in player_to_sids.items():
        if len(sids) > 1:
            print(f"\n  Note: LLM assigned player '{p}' to {len(sids)} speakers "
                  f"({', '.join(sids)}) — will merge embeddings under one player entry.")

    bank = create_empty_voice_bank()
    speaker_map = {}
    assigned_players = set()

    for speaker_id in sorted_speakers:
        embedding = speaker_embeddings[speaker_id]
        speaker_stats = stats.get(speaker_id, {"count": 0, "total_time": 0.0})
        total_secs = speaker_stats["total_time"]
        mins = int(total_secs // 60)
        secs = int(total_secs % 60)
        time_str = f"{mins}:{secs:02d}"
        dm_tag = " -- likely the DM" if speaker_id == dm_hint_id else ""

        # Try LLM identification first
        if speaker_id in llm_ids:
            llm_info = llm_ids[speaker_id]
            llm_char = llm_info.get("character")
            llm_player = llm_info.get("player")

            print(f"\n  === {speaker_id} === "
                  f"({speaker_stats['count']} segments, {time_str} speaking time{dm_tag})")
            print(f"  LLM identified: player={llm_player or '?'}, character={llm_char or '?'}")

            player = llm_player or llm_char
            character = llm_char or llm_player
            print(f"  -> Auto-accepted: {character} (player: {player})")
        else:
            # Interactive fallback
            print(f"\n  === {speaker_id} === "
                  f"({speaker_stats['count']} segments, {time_str} speaking time{dm_tag})")

            samples = get_speaker_samples(segments, speaker_id)
            if samples:
                print(f"  Sample dialogue:")
                for i, s in enumerate(samples, 1):
                    start_ts = format_timestamp(s["start"])
                    end_ts = format_timestamp(s["end"])
                    print(f"    {i}. [{start_ts} - {end_ts}] \"{s['text']}\"")

            # Show unassigned players from roster
            unassigned = [p for p in known_players if p.lower() not in assigned_players]
            if unassigned:
                print(f"  Unassigned players: {', '.join(sorted(unassigned))}")

            player = input(f"  Player name for {speaker_id} (the real person): ").strip()
            if not player or player.lower() == "skip":
                print(f"  Skipping {speaker_id}")
                continue
            character = input(f"  Character name for {player}: ").strip()
            if not character:
                character = player

        # Record mapping
        speaker_map[speaker_id] = {"character": character, "player": player}
        assigned_players.add(player.lower())

        # Add to v2 voice bank (player-indexed)
        if player in bank["players"]:
            player_data = bank["players"][player]
            player_data["embeddings"].append({
                "session": session_name,
                "vector": embedding,
                "character": character,
            })
            if character not in player_data["characters"]:
                player_data["characters"][character] = {
                    "first_session": session_name,
                    "last_session": session_name,
                }
            player_data["active_character"] = character
            all_vectors = [e["vector"] for e in player_data["embeddings"]]
            player_data["centroid"] = compute_centroid(all_vectors)
            print(f"  Added session embedding for {player}/{character}")
        else:
            bank["players"][player] = {
                "active_character": character,
                "characters": {
                    character: {
                        "first_session": session_name,
                        "last_session": session_name,
                    }
                },
                "embeddings": [{
                    "session": session_name,
                    "vector": embedding,
                    "character": character,
                }],
                "centroid": embedding if isinstance(embedding, list) else embedding.tolist(),
            }
            print(f"  Enrolled: {character} (player: {player})")

    save_voice_bank(bank)

    # Summary
    print(f"\n  Voice Bank Summary:")
    print(f"  {'Player':<15} {'Character':<20} {'Sessions'}")
    print(f"  {'-'*15} {'-'*20} {'-'*8}")
    for player_name, info in bank["players"].items():
        n = len(info["embeddings"])
        char = info["active_character"]
        print(f"  {player_name:<15} {char:<20} {n}")

    return bank, speaker_map


# ============================================================
# Speaker Mapping
# ============================================================

def map_speakers(segments: list, speaker_map: dict) -> list:
    """Replace SPEAKER_XX labels with player/character names."""
    for segment in segments:
        speaker_id = segment.get("speaker", "UNKNOWN")
        if speaker_id in speaker_map:
            mapping = speaker_map[speaker_id]
            if isinstance(mapping, dict):
                segment["speaker_name"] = mapping.get("character", mapping.get("player", speaker_id))
                segment["player_name"] = mapping.get("player", speaker_id)
            else:
                segment["speaker_name"] = mapping
                segment["player_name"] = mapping
        else:
            segment["speaker_name"] = speaker_id
            segment["player_name"] = speaker_id
    return segments


# ============================================================
# Output Formatting
# ============================================================

def segments_to_markdown(segments: list, session_name: str, audio_file: str) -> str:
    """Convert transcription segments to formatted Markdown."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# {session_name}",
        f"",
        f"**Date Processed:** {date_str}",
        f"**Source:** {audio_file}",
        f"",
        f"---",
        f"",
    ]

    current_speaker = None
    current_block = []

    def flush_block():
        if current_speaker and current_block:
            timestamp = current_block[0]["timestamp"]
            text = " ".join(seg["text"].strip() for seg in current_block)
            lines.append(f"**{current_speaker}** [{timestamp}]: {text}")
            lines.append("")

    for segment in segments:
        speaker = segment.get("speaker_name", "UNKNOWN")
        timestamp = format_timestamp(segment.get("start", 0))
        segment["timestamp"] = timestamp

        if speaker != current_speaker:
            flush_block()
            current_speaker = speaker
            current_block = [segment]
        else:
            current_block.append(segment)

    flush_block()

    return "\n".join(lines)


def upload_to_openwebui(markdown_path: Path) -> bool:
    """Upload a markdown transcript to Open WebUI Knowledge Collection."""
    if not OPENWEBUI_API_KEY:
        print("  Skipping upload: OPENWEBUI_API_KEY not set")
        return False
    if not KNOWLEDGE_COLLECTION_ID:
        print("  Skipping upload: KNOWLEDGE_COLLECTION_ID not set")
        return False

    headers = {"Authorization": f"Bearer {OPENWEBUI_API_KEY}"}

    with open(markdown_path, "rb") as f:
        upload_resp = requests.post(
            f"{OPENWEBUI_URL}/api/v1/files/",
            headers=headers,
            files={"file": (markdown_path.name, f, "text/markdown")},
            timeout=120,
        )
    upload_resp.raise_for_status()
    file_id = upload_resp.json()["id"]
    print(f"  Uploaded file: {file_id}")

    add_resp = requests.post(
        f"{OPENWEBUI_URL}/api/v1/knowledge/{KNOWLEDGE_COLLECTION_ID}/file/add",
        headers=headers,
        json={"file_id": file_id},
        timeout=120,
    )
    add_resp.raise_for_status()
    print(f"  Added to knowledge collection: {KNOWLEDGE_COLLECTION_ID}")
    return True


# ============================================================
# Embedding Integration
# ============================================================

def _try_embed_transcript(output_path: Path):
    """Attempt to embed transcript into ChromaDB. Graceful degradation if Ollama is down."""
    try:
        from embed import embed_transcript, get_chroma_collection
    except ImportError:
        print("  Skipping embedding: embed.py not available")
        return

    try:
        collection = get_chroma_collection()
        count = embed_transcript(output_path, collection)
        if count:
            print(f"  Embedded {count} chunks into ChromaDB")
    except requests.exceptions.ConnectionError:
        print("  Skipping embedding: Ollama not available (transcription succeeded)")
    except Exception as e:
        print(f"  Embedding failed (transcription succeeded): {e}")


# ============================================================
# Unified Pipeline Orchestrator
# ============================================================

def process_file(audio_path: Path, players: dict, interactive: bool = False) -> Path | None:
    """Process a single audio file through the full pipeline.

    Auto-detects enrollment vs processing mode:
    - No voice bank (or --interactive): enrollment mode
    - Voice bank exists: processing mode with character switch detection
    """
    session_name = audio_path.stem

    # Transcribe
    result = transcribe(audio_path, players)

    segments = result.get("segments", [])
    if not segments:
        print(f"  Warning: No segments returned for {audio_path.name}")
        return None

    speaker_embeddings = result.get("speaker_embeddings", {})
    voice_bank = load_voice_bank()

    if voice_bank and voice_bank.get("players") and not interactive:
        # === PROCESSING MODE ===
        print(f"  Using voice bank for speaker identification...")

        if speaker_embeddings:
            matches = match_speakers_by_embedding(speaker_embeddings, voice_bank)
        else:
            print(f"  No speaker embeddings returned, using static speaker mapping")
            segments = map_speakers(segments, players.get("speaker_map", {}))
            return _finalize_transcript(segments, session_name, audio_path)

        if matches:
            # Print initial matches
            embedding_map = {}
            for speaker_id, match in matches.items():
                bs = match.get("best_session", match["confidence"])
                cs = match.get("centroid", match["confidence"])
                print(f"    {speaker_id} -> {match['player']}/{match['character']} "
                      f"(confidence: {match['confidence']:.3f}, "
                      f"best_session: {bs:.2f}, centroid: {cs:.2f})")
                embedding_map[speaker_id] = {
                    "character": match["character"],
                    "player": match["player"],
                }

            # Verify character assignments (detect switches)
            character_overrides = verify_characters_with_llm(
                segments, matches, voice_bank, session_name)

            # Apply character overrides to the embedding map
            if character_overrides:
                for sid, match in matches.items():
                    player = match.get("player", "")
                    if player in character_overrides:
                        new_char = character_overrides[player]
                        embedding_map[sid]["character"] = new_char
                        match["character"] = new_char

            # LLM-assisted identification for unmatched/low-confidence speakers
            all_speaker_ids = set(speaker_embeddings.keys())
            matched_ids = set(matches.keys())
            unmatched = [sid for sid in all_speaker_ids if sid not in matched_ids]
            low_confidence = [sid for sid, m in matches.items()
                              if m["confidence"] < 0.70]
            needs_llm = unmatched + low_confidence

            if needs_llm:
                llm_ids = identify_speakers_with_llm(
                    segments, matches, voice_bank, needs_llm, players)
                for sid, info in llm_ids.items():
                    if sid in needs_llm:
                        matches[sid] = info
                        embedding_map[sid] = {
                            "character": info["character"],
                            "player": info["player"],
                        }

                # Auto-enroll LLM-identified players not yet in voice bank
                bank_players = voice_bank.get("players", {})
                for sid, info in llm_ids.items():
                    player = info["player"]
                    character = info["character"]
                    if player not in bank_players:
                        embedding = speaker_embeddings.get(sid)
                        if embedding is None:
                            continue
                        bank_players[player] = {
                            "active_character": character,
                            "characters": {
                                character: {
                                    "first_session": session_name,
                                    "last_session": session_name,
                                }
                            },
                            "embeddings": [{
                                "session": session_name,
                                "vector": embedding,
                                "character": character,
                                "auto": True,
                                "source": "llm",
                            }],
                            "centroid": embedding if isinstance(embedding, list) else embedding,
                        }
                        save_voice_bank(voice_bank)
                        print(f"  New player enrolled: {player}/{character} (identified by LLM)")

            segments = map_speakers(segments, embedding_map)

            # Update voice bank with high-confidence embeddings
            update_voice_bank(voice_bank, speaker_embeddings, matches,
                              session_name, character_overrides)
        else:
            print(f"  No embedding matches found, falling back to static mapping")
            segments = map_speakers(segments, players.get("speaker_map", {}))

    else:
        # === ENROLLMENT MODE ===
        if not speaker_embeddings:
            print("  Error: WhisperX did not return speaker embeddings.")
            print("  Make sure your WhisperX instance supports return_speaker_embeddings=true")
            return None

        print(f"\n  ENROLLMENT MODE: No voice bank found, enrolling speakers...")
        print(f"  Found {len(speaker_embeddings)} speakers with embeddings.")

        voice_bank, enrollment_map = enroll_speakers(
            segments, speaker_embeddings, players, session_name)

        segments = map_speakers(segments, enrollment_map)

    return _finalize_transcript(segments, session_name, audio_path)


def _finalize_transcript(segments, session_name, audio_path) -> Path:
    """Write transcript markdown, upload, and embed."""
    display_name = session_name.replace("_", " ").replace("-", " ").title()
    markdown = segments_to_markdown(segments, display_name, audio_path.name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{audio_path.stem}.md"
    output_path.write_text(markdown)
    print(f"  Transcript saved: {output_path}")

    upload_to_openwebui(output_path)

    # Embed into ChromaDB
    _try_embed_transcript(output_path)

    return output_path


# ============================================================
# Batch Processing
# ============================================================

def find_new_files(processed: dict) -> list[Path]:
    """Find audio files that haven't been processed yet."""
    new_files = []
    for f in sorted(INPUT_DIR.iterdir()):
        if f.suffix.lower() in AUDIO_EXTENSIONS and f.name not in processed:
            new_files.append(f)
    return new_files


def run_once(players: dict, processed: dict, interactive: bool = False):
    """Process all new files once."""
    new_files = find_new_files(processed)
    if not new_files:
        print("No new audio files found.")
        return

    print(f"Found {len(new_files)} new file(s) to process.")
    for audio_path in new_files:
        print(f"\nProcessing: {audio_path.name}")
        try:
            output = process_file(audio_path, players, interactive=interactive)
            if output:
                processed[audio_path.name] = {
                    "hash": file_hash(audio_path),
                    "processed_at": datetime.now().isoformat(),
                    "output": str(output),
                }
                save_processed(processed)
                print(f"  Done: {audio_path.name}")
        except requests.exceptions.RequestException as e:
            print(f"  Error processing {audio_path.name}: {e}")
        except Exception as e:
            print(f"  Unexpected error processing {audio_path.name}: {e}")
            raise


def watch(players: dict, processed: dict, interval: int = 30):
    """Watch input_audio/ for new files and process them."""
    print(f"Watching {INPUT_DIR} for new audio files (every {interval}s)...")
    print("Press Ctrl+C to stop.\n")
    while True:
        run_once(players, processed)
        time.sleep(interval)


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="D&D Session ETL Pipeline")
    parser.add_argument("--watch", action="store_true", help="Continuously watch for new files")
    parser.add_argument("--interval", type=int, default=30, help="Watch polling interval in seconds")
    parser.add_argument("--file", type=Path, help="Process a single specific file")
    parser.add_argument("--interactive", action="store_true",
                        help="Force enrollment mode even with existing voice bank")
    args = parser.parse_args()

    players = load_players()
    processed = load_processed()

    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        print(f"Processing single file: {args.file}")
        process_file(args.file, players, interactive=args.interactive)
    elif args.watch:
        watch(players, processed, args.interval)
    else:
        run_once(players, processed, interactive=args.interactive)


if __name__ == "__main__":
    main()
