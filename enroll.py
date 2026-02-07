#!/usr/bin/env python3
"""
Voice Bank Enrollment — Bootstrap speaker identity from a known session.

Sends a session's audio through WhisperX with speaker embeddings enabled,
then labels each speaker (interactively or via players.json) and saves
the labeled embeddings to config/voice_bank.json.

Also produces a transcript and marks the session as processed, so etl.py
won't re-transcribe the same file.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from etl import (
    map_speakers, segments_to_markdown, upload_to_openwebui,
    save_processed, load_processed, file_hash, OUTPUT_DIR, format_timestamp,
    has_anthropic_credentials, call_claude_llm,
)

# --- Configuration ---
BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
INPUT_DIR = BASE_DIR / "input_audio"
PLAYERS_JSON = CONFIG_DIR / "players.json"
VOICE_BANK_PATH = CONFIG_DIR / "voice_bank.json"
WHISPERX_URL = "http://localhost:9876"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".mp4", ".mkv", ".webm"}


def find_audio_file(session_name: str) -> Path:
    """Find the audio file for a given session name."""
    for ext in AUDIO_EXTENSIONS:
        candidate = INPUT_DIR / f"{session_name}{ext}"
        if candidate.exists():
            return candidate
    # Try partial match
    for f in INPUT_DIR.iterdir():
        if session_name in f.stem and f.suffix.lower() in AUDIO_EXTENSIONS:
            return f
    return None


def transcribe_with_embeddings(audio_path: Path, min_speakers: int = 4, max_speakers: int = 5) -> dict:
    """Transcribe audio via WhisperX with speaker embeddings enabled."""
    print(f"  Transcribing with embeddings: {audio_path.name}")
    print(f"  Speaker range: {min_speakers}-{max_speakers}")
    print(f"  This may take several minutes...")

    with open(audio_path, "rb") as f:
        response = requests.post(
            f"{WHISPERX_URL}/asr",
            files={"audio_file": (audio_path.name, f)},
            params={
                "task": "transcribe",
                "language": "en",
                "diarize": "true",
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "output": "json",
                "return_speaker_embeddings": "true",
            },
            timeout=3600,
        )

    response.raise_for_status()
    return response.json()


def get_speaker_samples(segments: list, speaker_id: str, n: int = 10) -> list[dict]:
    """Get N evenly-spaced sample dialogue lines for a speaker, with timestamps."""
    speaker_segs = [s for s in segments
                    if s.get("speaker") == speaker_id
                    and len(s.get("text", "").strip()) > 20]
    if not speaker_segs:
        return []
    # Pick evenly-spaced samples across the session
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

    Unlike etl.py's identify_speakers_with_llm(), there's no voice bank context here —
    identification is purely transcript-based (name calls, DM patterns, etc.).

    Returns dict mapping SPEAKER_XX -> {"character": str|None, "player": str|None, "reasoning": str}.
    Empty dict on any failure.
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

    # Build campaign roster (filter out placeholders)
    roster_lines = []
    placeholder_patterns = {"Character ", "Player "}
    for speaker_id, info in players.get("speaker_map", {}).items():
        if isinstance(info, dict):
            character = info.get("character", "")
            player = info.get("player", "")
        else:
            character = info
            player = info
        # Skip placeholder entries like "Character 1" / "Player 1"
        is_placeholder = any(character.startswith(p) for p in placeholder_patterns) or \
                         any(player.startswith(p) for p in placeholder_patterns)
        if character and not is_placeholder:
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

    prompt = f"""You are identifying speakers in a D&D session transcript during enrollment.
No voice bank exists yet — you must identify speakers purely from dialogue context.

## Speaker Stats
{chr(10).join(stats_lines)}

## Campaign Roster
{roster_section}

## Full Session Transcript
{chr(10).join(transcript_lines)}

## Instructions
For each SPEAKER_XX, analyze the full transcript for:
- Someone calling a name and that speaker responding in the next line(s)
- DM patterns: scene descriptions, "roll for...", NPC dialogue, narration, running combat
- Character-specific speech: "I cast...", "my barbarian...", referencing class abilities
- Out-of-character references: players using real names
- Process of elimination: if all other players are accounted for

For the DM/Game Master, use character="Narrator" and player="DM".
If you know the character but not the player, set player to the character name.
If you know the player but not the character, set character to the player name.
Always provide BOTH fields — use your best guess rather than null.

Respond with ONLY valid JSON (no markdown fences):
{{"SPEAKER_XX": {{"character": "Name", "player": "PlayerName", "reasoning": "brief explanation"}}}}

Include ALL speakers you can identify with reasonable confidence. Omit those you truly cannot identify."""

    content = call_claude_llm(prompt)
    if not content:
        return {}

    try:
        # Parse JSON (handle possible markdown fences)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        llm_result = json.loads(content)

        # Normalize results: only keep speakers we actually have
        results = {}
        for sid, info in llm_result.items():
            if sid not in stats:
                continue
            character = info.get("character") or None
            player = info.get("player") or None
            reasoning = info.get("reasoning", "")
            # Convert "null" strings to actual None
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

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  LLM response parsing failed: {e}")
        return {}


def compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """Compute arithmetic mean of embedding vectors."""
    arr = np.array(embeddings)
    return arr.mean(axis=0).tolist()


def load_voice_bank() -> dict:
    """Load existing voice bank or return empty structure."""
    if VOICE_BANK_PATH.exists():
        with open(VOICE_BANK_PATH) as f:
            return json.load(f)
    return {
        "meta": {
            "embedding_dim": 256,
            "threshold": 0.55,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        },
        "speakers": {},
    }


def save_voice_bank(bank: dict):
    """Save voice bank to disk."""
    bank["meta"]["last_updated"] = datetime.now().isoformat()
    with open(VOICE_BANK_PATH, "w") as f:
        json.dump(bank, f, indent=2)
    print(f"\n  Voice bank saved: {VOICE_BANK_PATH}")


def produce_transcript(segments: list, labeled: dict, audio_path: Path):
    """Write transcript and mark session as processed using labels from enrollment.

    Args:
        segments: raw WhisperX segments
        labeled: dict mapping speaker_id -> (character, player)
        audio_path: path to the source audio file
    """
    # Build enrollment map for map_speakers()
    enrollment_map = {}
    for speaker_id, (character, player) in labeled.items():
        enrollment_map[speaker_id] = {"character": character, "player": player}

    # Apply labels
    labeled_segments = map_speakers(segments, enrollment_map)

    # Generate markdown
    session_title = audio_path.stem.replace("_", " ").replace("-", " ").title()
    markdown = segments_to_markdown(labeled_segments, session_title, audio_path.name)

    # Write transcript
    output_path = OUTPUT_DIR / f"{audio_path.stem}.md"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)
    print(f"  Transcript saved: {output_path}")

    # Mark as processed so etl.py won't re-process
    processed = load_processed()
    processed[audio_path.name] = {
        "hash": file_hash(audio_path),
        "processed_at": datetime.now().isoformat(),
        "output": str(output_path),
    }
    save_processed(processed)

    # Upload to Open WebUI
    upload_to_openwebui(output_path)


def enroll_from_session(session_name: str, auto_mode: bool = False, add_session: bool = False):
    """Enroll speakers from a session into the voice bank."""
    # Find audio file
    audio_path = find_audio_file(session_name)
    if not audio_path:
        print(f"Error: No audio file found for session '{session_name}' in {INPUT_DIR}")
        sys.exit(1)

    # Load player config for speaker range
    players = {}
    if PLAYERS_JSON.exists():
        with open(PLAYERS_JSON) as f:
            players = json.load(f)

    min_speakers = players.get("min_speakers", 4)
    max_speakers = players.get("max_speakers", 5)

    # Transcribe with embeddings
    result = transcribe_with_embeddings(audio_path, min_speakers, max_speakers)

    segments = result.get("segments", [])
    speaker_embeddings = result.get("speaker_embeddings", {})

    if not speaker_embeddings:
        print("Error: WhisperX did not return speaker embeddings.")
        print("Make sure your WhisperX instance supports return_speaker_embeddings=true")
        sys.exit(1)

    print(f"\n  Found {len(speaker_embeddings)} speakers with embeddings.")

    # Compute stats for display
    stats = get_speaker_stats(segments)

    # Sort speakers by segment count (descending) — DM typically talks the most
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

    # Load or create voice bank
    bank = load_voice_bank() if add_session else {
        "meta": {
            "embedding_dim": 256,
            "threshold": 0.55,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        },
        "speakers": {},
    }

    # Label each speaker
    speaker_map = players.get("speaker_map", {})

    # LLM pre-identification (skip in auto mode — already have players.json)
    llm_ids = {}
    if not auto_mode:
        llm_ids = pre_identify_speakers_for_enrollment(
            segments, sorted_speakers, stats, dm_hint_id, players)
        if llm_ids:
            print("\n  LLM Pre-identification Results:")
            for sid, info in llm_ids.items():
                char = info.get("character") or "?"
                plyr = info.get("player") or "?"
                print(f"    {sid} -> {char} (player: {plyr}) -- {info.get('reasoning', '')}")

    labeled = {}  # speaker_id -> (character, player) for transcript production
    assigned_characters = set()
    assigned_players = set()

    for speaker_id in sorted_speakers:
        embedding = speaker_embeddings[speaker_id]
        samples = get_speaker_samples(segments, speaker_id)
        speaker_stats = stats.get(speaker_id, {"count": 0, "total_time": 0.0})

        # Format speaking time
        total_secs = speaker_stats["total_time"]
        mins = int(total_secs // 60)
        secs = int(total_secs % 60)
        time_str = f"{mins}:{secs:02d}"

        # Branch A: auto_mode — use players.json mapping directly
        if auto_mode and speaker_id in speaker_map:
            mapping = speaker_map[speaker_id]
            if isinstance(mapping, dict):
                character = mapping.get("character", speaker_id)
                player = mapping.get("player", speaker_id)
            else:
                character = mapping
                player = mapping
            print(f"\n  {speaker_id} -> {character} (auto from players.json)")

        # Branch B: LLM pre-identified this speaker
        elif speaker_id in llm_ids:
            llm_info = llm_ids[speaker_id]
            llm_char = llm_info.get("character")
            llm_player = llm_info.get("player")
            reasoning = llm_info.get("reasoning", "")
            dm_tag = " -- likely the DM" if speaker_id == dm_hint_id else ""

            print(f"\n  === {speaker_id} === "
                  f"({speaker_stats['count']} segments, {time_str} speaking time{dm_tag})")
            print(f"  LLM identified: character={llm_char or '?'}, "
                  f"player={llm_player or '?'}")
            print(f"  Reasoning: {reasoning}")

            # Auto-accept, defaulting missing fields
            character = llm_char or llm_player
            player = llm_player or llm_char
            print(f"  -> Auto-accepted: {character} (player: {player})")

        # Branch C: interactive fallback
        else:
            dm_tag = " -- likely the DM" if speaker_id == dm_hint_id else ""
            print(f"\n  === {speaker_id} === "
                  f"({speaker_stats['count']} segments, {time_str} speaking time{dm_tag})")

            if samples:
                print(f"  Sample dialogue:")
                for i, s in enumerate(samples, 1):
                    start_ts = format_timestamp(s["start"])
                    end_ts = format_timestamp(s["end"])
                    print(f"    {i}. [{start_ts} - {end_ts}] \"{s['text']}\"")
            else:
                print(f"  (No dialogue samples found)")

            # Show unassigned roster members as suggestions
            unassigned = []
            for _sid, info in players.get("speaker_map", {}).items():
                if isinstance(info, dict):
                    rc = info.get("character", "")
                    rp = info.get("player", "")
                else:
                    rc = info
                    rp = info
                # Skip placeholders
                if any(rc.startswith(p) for p in ("Character ", "Player ")) or \
                   any(rp.startswith(p) for p in ("Character ", "Player ")):
                    continue
                if rc.lower() not in assigned_characters and rp.lower() not in assigned_players:
                    unassigned.append(f"{rc} ({rp})")
            if unassigned:
                print(f"  Unassigned from roster: {', '.join(unassigned)}")

            if auto_mode:
                print(f"  Warning: {speaker_id} not in players.json, skipping")
                continue

            character = input(f"  Character name for {speaker_id} (or 'skip'): ").strip()
            if character.lower() == "skip" or not character:
                print(f"  Skipping {speaker_id}")
                continue
            player = input(f"  Player name for {character} (or Enter to use character name): ").strip()
            if not player:
                player = character

        # Record label and track assignments
        if speaker_id not in labeled:
            labeled[speaker_id] = (character, player)
        assigned_characters.add(character.lower())
        assigned_players.add(player.lower())

        # Add to voice bank
        entry = {
            "session": session_name,
            "vector": embedding,
        }

        if character in bank["speakers"]:
            # Append embedding to existing speaker
            bank["speakers"][character]["embeddings"].append(entry)
            all_vectors = [e["vector"] for e in bank["speakers"][character]["embeddings"]]
            bank["speakers"][character]["centroid"] = compute_centroid(all_vectors)
            print(f"  Added session embedding for {character} ({len(all_vectors)} total)")
        else:
            # New speaker entry
            bank["speakers"][character] = {
                "player": player,
                "character": character,
                "embeddings": [entry],
                "centroid": embedding,  # Single vector = its own centroid
            }
            print(f"  Enrolled {character} (player: {player})")

    save_voice_bank(bank)

    # Produce transcript using the labels just provided (no second transcription!)
    if labeled:
        print(f"\n  Writing transcript with enrolled speaker names...")
        produce_transcript(segments, labeled, audio_path)

    # Summary
    print(f"\n  Voice Bank Summary:")
    print(f"  {'Character':<20} {'Player':<15} {'Sessions'}")
    print(f"  {'-'*20} {'-'*15} {'-'*8}")
    for name, info in bank["speakers"].items():
        n = len(info["embeddings"])
        print(f"  {name:<20} {info['player']:<15} {n}")


def main():
    parser = argparse.ArgumentParser(description="Voice Bank Enrollment")
    parser.add_argument("--session", required=True, help="Session name (e.g. '2025-12-10 21-43-18')")
    parser.add_argument("--auto", action="store_true", help="Auto-label using players.json mapping")
    parser.add_argument("--add-session", action="store_true", help="Append to existing voice bank")
    args = parser.parse_args()

    enroll_from_session(args.session, auto_mode=args.auto, add_session=args.add_session)


if __name__ == "__main__":
    main()
