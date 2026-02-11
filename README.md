# Storyline — D&D Campaign Archivist

An end-to-end pipeline that transcribes D&D session recordings, identifies speakers by voice + AI, generates searchable embeddings, and provides an AI-powered campaign archivist via Claude Agent SDK.

```
Audio Files → WhisperX (transcription + diarization) → Speaker Identification
                                                              ↓
                                              Voice Bank ←→ etl.py (matching + LLM fallback)
                                                              ↓
                                                     Markdown Transcripts → embed.py (ChromaDB)
                                                                                  ↓
                                                                            agent.py (Claude Agent SDK)
```

## Requirements

- **Python 3.12+** with a virtual environment
- **Docker** with NVIDIA GPU support (for WhisperX)
- **Ollama** with `nomic-embed-text` model (for embeddings)
- **Anthropic API key** or **Claude Max OAuth token** (for the Claude-powered agent and LLM-assisted identification)
- **HuggingFace account** with accepted terms for [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

## Setup

### 1. Environment

Create a `.env` file:

```bash
HF_TOKEN=hf_...                  # HuggingFace token for pyannote diarization
OPENWEBUI_API_KEY=...             # Optional, for Open WebUI upload

# Authentication (one of the following):
ANTHROPIC_API_KEY=sk-ant-...      # Anthropic API key
# OR
CLAUDE_CODE_OAUTH_TOKEN=...       # OAuth token from `claude setup-token` (Claude Max)
```

### 2. Services

```bash
docker compose up -d          # Starts WhisperX + Open WebUI
ollama pull nomic-embed-text  # Embedding model
```

### 3. Python Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-agent.txt
```

### 4. Player Configuration

Copy the example and fill in your campaign's player roster:

```bash
cp config/players.json.example config/players.json
```

Then edit `config/players.json` with your players' real names and character names:

```json
{
  "speaker_map": {
    "SPEAKER_00": {"player": "DM", "character": "Narrator"},
    "SPEAKER_01": {"player": "Alice", "character": "Elara"},
    "SPEAKER_02": {"player": "Bob", "character": "Kai"},
    "SPEAKER_03": {"player": "Charlie", "character": "Zeno"},
    "SPEAKER_04": {"player": "Dana", "character": "Nokapa"}
  },
  "min_speakers": 4,
  "max_speakers": 5,
  "clustering_threshold": 0.45
}
```

| Field | Purpose |
|-------|---------|
| `speaker_map` | Player roster used as context for LLM identification. Maps player real names to character names. The SPEAKER_XX keys are placeholders — actual mapping is done by voice matching. |
| `min_speakers` / `max_speakers` | Passed to pyannote diarization to constrain speaker count. |
| `clustering_threshold` | Pyannote clustering sensitivity. Lower = more speaker splits (default: 0.6). Use 0.4-0.5 if the DM is being merged with players. |

> `config/players.json` is gitignored since it contains your campaign's real names. Only `players.json.example` is tracked.

## Usage

### Single Command Pipeline

Drop audio files into `input_audio/` and run:

```bash
python etl.py
```

That's it. `etl.py` handles everything automatically:

1. **First session** (no voice bank): enters **enrollment mode** — uses Claude to auto-identify speakers from dialogue context (name calls, DM patterns, character abilities), creates the voice bank, writes the transcript, and embeds to ChromaDB.

2. **Subsequent sessions** (voice bank exists): enters **processing mode** — matches speakers by voice embedding, verifies character assignments via LLM, falls back to LLM for uncertain speakers, writes the transcript, and embeds to ChromaDB.

```bash
python etl.py                    # Process all new files
python etl.py --file path        # Process a specific file
python etl.py --watch            # Watch for new files continuously
python etl.py --interactive      # Force enrollment mode (re-enroll speakers)
```

### How Speaker Identification Works

The pipeline uses a two-phase approach: **voice matching** for speed and consistency, with **LLM fallback** for ambiguous cases.

#### Phase 1: Voice Matching

Each speaker embedding from WhisperX is compared against every stored embedding in the voice bank using cosine similarity:

```
score = 0.6 * best_session_score + 0.4 * centroid_score
```

- **best_session_score**: Highest cosine similarity against any individual stored embedding for that player.
- **centroid_score**: Cosine similarity against the player's centroid (mean of all their stored embeddings across sessions).
- **Threshold**: 0.55 (configurable in voice bank). Speakers below this are unmatched.

The blending favors strong individual matches while the centroid anchors against drift as more sessions are processed.

**DM splits**: Diarization often splits the DM into multiple speaker IDs (different NPC voices, narration vs. dialogue). The system allows multiple SPEAKER_XX IDs to map to the same player — no collision resolution forces them apart.

High-confidence matches (above 0.60) automatically add their embedding to the voice bank, making future matching more robust.

#### Phase 2: LLM Fallback

Speakers that are unmatched or have low confidence (below 0.70) are sent to Claude for identification. The LLM receives:
- Already-identified speakers with confidence scores
- The full session transcript with timestamps
- The campaign roster (unmatched players from voice bank + players.json)

Claude analyzes dialogue for contextual clues: name calls, DM patterns ("roll for..."), character abilities ("I cast..."), out-of-character references, and process of elimination. Players identified by the LLM are auto-enrolled into the voice bank.

### Voice Bank (v2)

The voice bank (`config/voice_bank.json`) is **player-indexed** — keyed by the player's real name, not their character. This supports character switches: the same physical voice is always recognized regardless of which character they're playing.

```json
{
  "meta": {"version": 2, "embedding_dim": 256, "threshold": 0.55},
  "players": {
    "Alice": {
      "active_character": "Zara",
      "characters": {
        "Elara": {"first_session": "Session 01", "last_session": "Session 15"},
        "Zara": {"first_session": "Session 16", "last_session": "Session 20"}
      },
      "embeddings": [
        {"session": "Session 01", "vector": [...], "character": "Elara", "confidence": 0.93},
        {"session": "Session 16", "vector": [...], "character": "Zara", "confidence": 0.91}
      ],
      "centroid": [...]
    }
  }
}
```

Each player entry tracks:
- **active_character**: Current character (updated on switch detection)
- **characters**: Map of all characters played with session ranges
- **embeddings**: Per-session voice vectors tagged with character name and confidence
- **centroid**: Mean of all embeddings (recomputed on each update)

The voice bank auto-migrates from v1 (character-indexed) to v2 (player-indexed) if an older format is detected.

### Character Switches

When a player switches characters mid-campaign (e.g., Elara -> Zara), `etl.py` detects this automatically via a lightweight LLM verification after voice matching. The voice bank tracks players (physical voices), not characters — so the same voice is always recognized, regardless of which character they're playing.

The system distinguishes between:
- **Permanent character switches** (Elara -> Zara) — updates `active_character` in the voice bank
- **Sidekick/companion control** (player voices an NPC temporarily) — no change, keeps the primary PC

### Diarization Tuning

WhisperX uses pyannote for speaker diarization. The pipeline patches the WhisperX container (`config/whisperx_main.py`) to expose pyannote's clustering threshold, which controls how aggressively it merges speaker segments.

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| 0.6 (default) | Fewer speakers, more merging | Clean recordings, few speakers |
| 0.45 | More speakers, less merging | D&D sessions where DM voices NPCs |
| 0.3 | Maximum separation | Debugging, when players are being merged |

Set `clustering_threshold` in `config/players.json`. The Docker service runs with `COMPUTE_TYPE=float32` and `BATCH_SIZE=8` for maximum accuracy.

### Embed (standalone)

`etl.py` embeds transcripts automatically. Use `embed.py` standalone for re-indexing or status:

```bash
python embed.py              # Embed all un-embedded transcripts
python embed.py --file path  # Embed a specific transcript
python embed.py --reindex    # Re-embed everything from scratch
python embed.py --status     # Show embedding status
```

Embeddings are stored in ChromaDB at `data/chromadb/`. If Ollama is unavailable when `etl.py` runs, transcription succeeds and embedding is skipped with a warning.

### Query

```bash
# Interactive REPL (multi-turn conversation)
python agent.py

# Single query
python agent.py --query "What happened in the last session?"

# Save output to file
python agent.py --query "Summarize session 2025-12-10" --output output_transcripts/summary.md

# Use a different model
python agent.py --model claude-opus-4-6
```

#### Interactive Commands

| Command | Description |
|---------|-------------|
| `/quit` | Exit the agent |
| `/cost` | Show API cost for this session |
| `/sessions` | List available transcripts |

#### Agent Tools

The agent has 5 search tools plus built-in file access:

| Tool | Use For |
|------|---------|
| `list_sessions` | See what transcripts are available |
| `search_transcripts` | Exact text/speaker search — names, quotes, keywords |
| `semantic_search` | Conceptual queries — "when did the party discuss strategy" |
| `get_session_content` | Read full or partial session content |
| `get_campaign_info` | Player/character roster |

## Deployment Modes

### Single Machine

Everything runs on one machine with a GPU:

```
┌──────────────────────────────────┐
│  WhisperX (Docker, GPU)          │
│  Ollama + nomic-embed-text (GPU) │
│  etl.py → embed.py → agent.py   │
│  ChromaDB (data/chromadb/)       │
└──────────────────────────────────┘
```

```bash
# Full pipeline on one machine (etl.py handles embedding automatically)
python etl.py
python agent.py
```

### Two Machines (GPU + Query)

Machine A (GPU) handles transcription and embedding. Machine B runs the agent and queries.

```
┌─ Machine A (GPU) ────────┐              ┌─ Machine B ─────────────┐
│ WhisperX (Docker)         │    rsync     │ ChromaDB (synced)       │
│ Ollama + nomic-embed-text │ ──────────→  │ Transcripts (synced)    │
│ etl.py → embed.py         │  transcripts │ agent.py (Claude)       │
│ ChromaDB (source)         │  + chromadb  │                         │
└───────────────────────────┘              └─────────────────────────┘
```

**On Machine A** (after transcription + embedding):

```bash
./sync.sh user@machineB           # Sync transcripts + ChromaDB + config
./sync.sh user@machineB ~/mydir   # Custom destination directory
```

**On Machine B:**

```bash
pip install claude-agent-sdk chromadb requests
export ANTHROPIC_API_KEY='sk-ant-...'
python agent.py --data-dir ~/storyline
```

The `--data-dir` flag tells the agent where to find the synced transcripts and ChromaDB data.

### OpenClaw + Discord

For Discord integration, use [OpenClaw](https://github.com/openclaw) as the gateway. The Campaign Archivist runs as an OpenClaw agent — no separate `agent.py` process needed (single Claude session, half the token cost).

```
┌─ Machine A (GPU) ────────┐              ┌─ Remote (Mini PC) ──────────┐
│ WhisperX (Docker)         │    rsync     │ OpenClaw (Discord gateway)  │
│ Ollama + nomic-embed-text │ ──────────→  │   └── "archivist" agent     │
│ etl.py → embed.py         │  transcripts │       ├── AGENTS.md (prompt)│
│ ChromaDB (source)         │  + chromadb  │       ├── read (transcripts)│
└───────────────────────────┘              │       └── exec campaign_cli │
                                           │ Ollama + nomic-embed-text   │
                                           │ ChromaDB (synced)           │
                                           └─────────────────────────────┘
```

**One-time setup on the remote machine:**

```bash
# 1. Clone the repo and install minimal deps
git clone <repo> ~/storyline
cd ~/storyline
pip install -r requirements-openclaw.txt

# 2. Install Ollama + embedding model (small, CPU-only is fine)
# See: https://ollama.com/download
ollama pull nomic-embed-text  # ~274MB, runs on CPU

# 3. Sync data from GPU machine
# (on Machine A): ./sync.sh user@remote
```

4. Configure OpenClaw (for example in `~/.openclaw/openclaw.json`) with docs-style keys:

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/storyline",
      "skipBootstrap": true,
      "sandbox": { "mode": "off" }
    }
  },
  "tools": {
    "allow": ["read", "exec"]
  }
}
```

If you decide against installing Ollama on the remote, `campaign_cli.py semantic` gracefully degrades with a clear message. Text search (`campaign_cli.py search`) and direct transcript reading still work fully.

## Project Structure

```
storyline/
├── input_audio/           # Drop session recordings here
├── output_transcripts/    # Generated Markdown transcripts
├── data/
│   ├── chromadb/          # Vector embeddings (ChromaDB)
│   ├── whisperx-cache/    # WhisperX model cache
│   └── open-webui/        # Open WebUI data
├── config/
│   ├── players.json       # Player roster + diarization settings
│   ├── voice_bank.json    # Player-indexed voice embeddings (v2, generated by etl.py)
│   ├── whisperx_main.py   # Patched WhisperX service (exposes clustering params)
│   └── processed.json     # ETL dedup log
├── etl.py                 # Unified pipeline: transcribe, enroll, identify, embed
├── embed.py               # Standalone embedding tool (re-index, status)
├── agent.py               # Claude Agent SDK campaign archivist (local CLI)
├── campaign_cli.py        # Standalone CLI for OpenClaw exec (no SDK dependency)
├── AGENTS.md              # OpenClaw system prompt (Campaign Archivist persona)
├── skills/
│   └── campaign-archivist/
│       └── SKILL.md       # OpenClaw skill definition
├── sync.sh                # rsync data to remote machine
├── docker-compose.yml     # WhisperX + Open WebUI services
├── requirements-agent.txt # Python dependencies (local CLI with claude-agent-sdk)
├── requirements-openclaw.txt # Minimal deps for remote (chromadb + requests only)
└── .env                   # HF_TOKEN, API keys (not in git)
```

## Transcript Format

Transcripts are Markdown with speaker-attributed, timestamped dialogue:

```markdown
# Session Name

**Date Processed:** 2026-02-07
**Source:** recording.mp3

---

**Narrator** [00:05:23]: You enter a dark tavern.

**Elara** [00:05:45]: I approach the bartender and ask about local rumors.
```
