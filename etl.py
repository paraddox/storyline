#!/usr/bin/env python3
"""
D&D Session ETL Pipeline
Monitors input_audio/ for new files, transcribes via WhisperX,
maps speakers, formats Markdown, and uploads to Open WebUI Knowledge Collection.
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
            # Don't overwrite existing env vars (explicit exports take priority)
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


def has_anthropic_credentials() -> bool:
    """Check if any Anthropic API credentials are available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))


def call_claude_llm(prompt: str, model: str = "claude-sonnet-4-5-20250929", max_tokens: int = 2048) -> str | None:
    """Call Claude and return the text response.

    Uses ANTHROPIC_API_KEY directly if available (faster, no subprocess).
    Falls back to the `claude` CLI for CLAUDE_CODE_OAUTH_TOKEN (OAuth tokens
    aren't supported by the raw API, but the CLI handles them natively).

    Returns the response text, or None on any failure.
    """
    import shutil
    import subprocess

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        # Direct API call (faster)
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

    # Fall back to claude CLI for OAuth tokens
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not oauth_token:
        return None

    cli_path = shutil.which("claude")
    if not cli_path:
        print("  Claude CLI not found; cannot use CLAUDE_CODE_OAUTH_TOKEN for LLM calls")
        return None

    try:
        result = subprocess.run(
            [cli_path, "-p", prompt, "--output-format", "text",
             "--model", model, "--max-turns", "1",
             "--permission-mode", "bypassPermissions"],
            capture_output=True, text=True, timeout=300,
            env={**os.environ},
        )
        if result.returncode != 0:
            print(f"  Claude CLI failed (exit {result.returncode}): {result.stderr[:200]}")
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("  Claude CLI timed out")
        return None
    except Exception as e:
        print(f"  Claude CLI error: {e}")
        return None


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


def transcribe(audio_path: Path, players: dict) -> dict:
    """Send audio to WhisperX API for transcription + diarization."""
    min_speakers = players.get("min_speakers", 2)
    max_speakers = players.get("max_speakers", 6)

    print(f"  Transcribing: {audio_path.name}")
    print(f"  Speaker range: {min_speakers}-{max_speakers}")

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


def load_voice_bank() -> dict | None:
    """Load the voice bank if it exists."""
    if VOICE_BANK_PATH.exists():
        with open(VOICE_BANK_PATH) as f:
            return json.load(f)
    return None


def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms == 0:
        return 0.0
    return float(dot / norms)


def match_speakers_by_embedding(speaker_embeddings: dict, voice_bank: dict) -> dict:
    """Match SPEAKER_XX labels to known speakers via multi-embedding scoring.

    Compares each new embedding against every stored embedding (not just the centroid),
    then blends the best individual match with the centroid score for robustness.

    Returns a dict mapping SPEAKER_XX -> {"character": ..., "player": ..., "confidence": ...}
    """
    threshold = voice_bank.get("meta", {}).get("threshold", 0.55)
    bank_speakers = voice_bank.get("speakers", {})

    matches = {}
    for speaker_id, embedding in speaker_embeddings.items():
        best_name = None
        best_score = -1.0
        best_session_score = 0.0
        best_centroid_score = 0.0

        for name, info in bank_speakers.items():
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
                best_name = name
                best_session_score = session_score
                best_centroid_score = centroid_score

        if best_score >= threshold and best_name:
            matches[speaker_id] = {
                "character": best_name,
                "player": bank_speakers[best_name].get("player", best_name),
                "confidence": best_score,
                "best_session": best_session_score,
                "centroid": best_centroid_score,
            }

    # Resolve collisions: if two SPEAKER_XX match to the same character,
    # higher confidence wins; the other becomes Unknown
    assigned_names = {}
    for speaker_id, match in sorted(matches.items(), key=lambda x: -x[1]["confidence"]):
        name = match["character"]
        if name in assigned_names:
            # This speaker_id has lower confidence — mark as Unknown
            matches[speaker_id] = {
                "character": speaker_id,
                "player": speaker_id,
                "confidence": match["confidence"],
            }
        else:
            assigned_names[name] = speaker_id

    return matches


def update_voice_bank(voice_bank: dict, speaker_embeddings: dict, matches: dict, session_name: str):
    """Append high-confidence embeddings to voice bank and recompute centroids."""
    update_threshold = 0.60
    updated = False

    for speaker_id, match in matches.items():
        if match["confidence"] < update_threshold:
            continue
        character = match["character"]
        if character.startswith("SPEAKER_"):
            continue
        if character not in voice_bank.get("speakers", {}):
            continue

        embedding = speaker_embeddings.get(speaker_id)
        if embedding is None:
            continue

        # Check if this session is already enrolled
        speaker_data = voice_bank["speakers"][character]
        existing_sessions = {e["session"] for e in speaker_data.get("embeddings", [])}
        if session_name in existing_sessions:
            continue

        # Append with confidence metadata and recompute centroid
        speaker_data["embeddings"].append({
            "session": session_name,
            "vector": embedding,
            "confidence": match["confidence"],
            "auto": True,
        })
        all_vectors = [e["vector"] for e in speaker_data["embeddings"]]
        arr = np.array(all_vectors)
        speaker_data["centroid"] = arr.mean(axis=0).tolist()
        updated = True
        print(f"  Voice bank updated: {character} ({len(all_vectors)} sessions)")

    if updated:
        voice_bank["meta"]["last_updated"] = datetime.now().isoformat()
        with open(VOICE_BANK_PATH, "w") as f:
            json.dump(voice_bank, f, indent=2)


def identify_speakers_with_llm(segments, voice_matches, voice_bank, uncertain_ids, players=None):
    """Use Claude to identify speakers from transcript context.

    Called as a fallback when voice matching leaves unmatched or low-confidence speakers.
    Sends the full transcript with context about known matches and the campaign roster.
    Accepts an optional players dict (from players.json) to include new players not yet
    in the voice bank.
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
        if not match["character"].startswith("SPEAKER_"):
            player = match.get("player", "")
            conf = match.get("confidence", 0)
            identified_lines.append(
                f"- {sid} -> {match['character']} (played by {player}), confidence: {conf:.2f}")

    # Build the "need identification" section
    need_id_lines = []
    for sid in uncertain_ids:
        if sid in voice_matches and not voice_matches[sid]["character"].startswith("SPEAKER_"):
            conf = voice_matches[sid]["confidence"]
            char = voice_matches[sid]["character"]
            need_id_lines.append(f"- {sid} (low confidence: {conf:.2f}, tentatively matched to \"{char}\")")
        else:
            need_id_lines.append(f"- {sid} (unmatched)")

    # Build the roster of unmatched known speakers
    bank_speakers = voice_bank.get("speakers", {})
    matched_characters = {m["character"] for m in voice_matches.values()
                          if not m["character"].startswith("SPEAKER_") and m.get("confidence", 0) >= 0.70}
    roster_lines = []
    for name, info in bank_speakers.items():
        if name not in matched_characters:
            player = info.get("player", name)
            roster_lines.append(f"- {name} (played by {player})")

    # Add players from players.json not yet in voice bank
    if players:
        for speaker_id, info in players.get("speaker_map", {}).items():
            if isinstance(info, dict):
                character = info.get("character", "")
                player = info.get("player", "")
            else:
                character = info
                player = info
            if character and character not in bank_speakers and character not in matched_characters:
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

For each speaker, respond with ONLY valid JSON (no markdown fences):
{{"SPEAKER_XX": {{"character": "Name", "player": "PlayerName", "reasoning": "brief explanation"}}}}

Only include speakers you can identify with reasonable confidence. Omit those you cannot."""

    content = call_claude_llm(prompt)
    if not content:
        return {}

    try:
        # Parse JSON from response (handle possible markdown fences)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        llm_result = json.loads(content)

        # Build full known roster (voice bank + players.json)
        known_characters = set(bank_speakers.keys())
        known_players = {v.get("player") for v in bank_speakers.values()}
        if players:
            for info in players.get("speaker_map", {}).values():
                if isinstance(info, dict):
                    known_characters.add(info.get("character", ""))
                    known_players.add(info.get("player", ""))
                else:
                    known_characters.add(info)

        # Validate and build results
        results = {}
        for sid, info in llm_result.items():
            if sid not in uncertain_ids:
                continue
            character = info.get("character", "")
            player = info.get("player", character)
            reasoning = info.get("reasoning", "")
            # Validate character is in the known roster
            if character in known_characters or player in known_players:
                results[sid] = {
                    "character": character,
                    "player": player,
                    "confidence": 0.50,  # LLM-identified, lower than voice match
                    "source": "llm",
                }
                print(f"    {sid} -> {character} (LLM, reasoning: \"{reasoning}\")")
            else:
                print(f"    {sid}: LLM suggested \"{character}\" but not in roster, skipping")

        return results

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  LLM response parsing failed: {e}")
        return {}


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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

    # Step 1: Upload the file
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

    # Step 2: Add to Knowledge Collection
    add_resp = requests.post(
        f"{OPENWEBUI_URL}/api/v1/knowledge/{KNOWLEDGE_COLLECTION_ID}/file/add",
        headers=headers,
        json={"file_id": file_id},
        timeout=120,
    )
    add_resp.raise_for_status()
    print(f"  Added to knowledge collection: {KNOWLEDGE_COLLECTION_ID}")
    return True


def process_file(audio_path: Path, players: dict) -> Path | None:
    """Process a single audio file through the full pipeline."""
    session_name = audio_path.stem.replace("_", " ").replace("-", " ").title()
    speaker_map = players.get("speaker_map", {})

    # Transcribe
    result = transcribe(audio_path, players)

    # Extract segments
    segments = result.get("segments", [])
    if not segments:
        print(f"  Warning: No segments returned for {audio_path.name}")
        return None

    # Map speakers: prefer voice bank embeddings, fall back to static mapping
    voice_bank = load_voice_bank()
    speaker_embeddings = result.get("speaker_embeddings", {})

    if voice_bank and speaker_embeddings:
        print(f"  Using voice bank for speaker identification...")
        matches = match_speakers_by_embedding(speaker_embeddings, voice_bank)

        if matches:
            # Build speaker_map from embedding matches
            embedding_map = {}
            for speaker_id, match in matches.items():
                bs = match.get("best_session", match["confidence"])
                cs = match.get("centroid", match["confidence"])
                print(f"    {speaker_id} -> {match['character']} "
                      f"(confidence: {match['confidence']:.3f}, "
                      f"best_session: {bs:.2f}, centroid: {cs:.2f})")
                embedding_map[speaker_id] = {
                    "character": match["character"],
                    "player": match["player"],
                }

            # LLM-assisted identification for unmatched/low-confidence speakers
            all_speaker_ids = set(speaker_embeddings.keys())
            matched_ids = set(matches.keys())
            unmatched = [sid for sid in all_speaker_ids if sid not in matched_ids]
            low_confidence = [sid for sid, m in matches.items()
                              if not m["character"].startswith("SPEAKER_")
                              and m["confidence"] < 0.70]

            if unmatched or low_confidence:
                llm_ids = identify_speakers_with_llm(
                    segments, matches, voice_bank, unmatched + low_confidence, players)
                for sid, info in llm_ids.items():
                    if sid in unmatched or sid in low_confidence:
                        matches[sid] = info
                        embedding_map[sid] = {
                            "character": info["character"],
                            "player": info["player"],
                        }

                # Auto-enroll LLM-identified speakers not yet in voice bank
                for sid, info in llm_ids.items():
                    character = info["character"]
                    if character not in voice_bank.get("speakers", {}):
                        embedding = speaker_embeddings.get(sid)
                        if embedding is None:
                            continue
                        voice_bank["speakers"][character] = {
                            "player": info["player"],
                            "character": character,
                            "embeddings": [{
                                "session": audio_path.stem,
                                "vector": embedding,
                                "auto": True,
                                "source": "llm",
                            }],
                            "centroid": embedding,
                        }
                        voice_bank["meta"]["last_updated"] = datetime.now().isoformat()
                        with open(VOICE_BANK_PATH, "w") as f:
                            json.dump(voice_bank, f, indent=2)
                        print(f"  New speaker enrolled in voice bank: {character} "
                              f"(player: {info['player']}, identified by LLM)")

            segments = map_speakers(segments, embedding_map)

            # Auto-update voice bank with high-confidence matches
            update_voice_bank(voice_bank, speaker_embeddings, matches, audio_path.stem)
        else:
            print(f"  No embedding matches found, falling back to static mapping")
            segments = map_speakers(segments, speaker_map)
    else:
        if not voice_bank:
            print(f"  No voice bank found, using static speaker mapping")
        elif not speaker_embeddings:
            print(f"  No speaker embeddings returned, using static speaker mapping")
        segments = map_speakers(segments, speaker_map)

    # Format as Markdown
    markdown = segments_to_markdown(segments, session_name, audio_path.name)

    # Write output
    output_path = OUTPUT_DIR / f"{audio_path.stem}.md"
    output_path.write_text(markdown)
    print(f"  Transcript saved: {output_path}")

    # Upload to Open WebUI
    upload_to_openwebui(output_path)

    return output_path


def find_new_files(processed: dict) -> list[Path]:
    """Find audio files that haven't been processed yet."""
    new_files = []
    for f in sorted(INPUT_DIR.iterdir()):
        if f.suffix.lower() in AUDIO_EXTENSIONS and f.name not in processed:
            new_files.append(f)
    return new_files


def run_once(players: dict, processed: dict):
    """Process all new files once."""
    new_files = find_new_files(processed)
    if not new_files:
        print("No new audio files found.")
        return

    print(f"Found {len(new_files)} new file(s) to process.")
    for audio_path in new_files:
        print(f"\nProcessing: {audio_path.name}")
        try:
            output = process_file(audio_path, players)
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


def main():
    parser = argparse.ArgumentParser(description="D&D Session ETL Pipeline")
    parser.add_argument("--watch", action="store_true", help="Continuously watch for new files")
    parser.add_argument("--interval", type=int, default=30, help="Watch polling interval in seconds")
    parser.add_argument("--file", type=Path, help="Process a single specific file")
    args = parser.parse_args()

    players = load_players()
    processed = load_processed()

    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        print(f"Processing single file: {args.file}")
        process_file(args.file, players)
    elif args.watch:
        watch(players, processed, args.interval)
    else:
        run_once(players, processed)


if __name__ == "__main__":
    main()
