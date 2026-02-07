# Storyline — D&D Campaign Archivist

An end-to-end pipeline that transcribes D&D session recordings, generates searchable embeddings, and provides an AI-powered campaign archivist via Claude Agent SDK.

```
Audio Files → WhisperX (transcription + diarization) → Markdown Transcripts
                                                            ↓
                                                       embed.py (ChromaDB)
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

Edit `config/players.json` with your player roster and speaker count:

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
  "max_speakers": 5
}
```

> `speaker_map` is a static fallback — voice bank enrollment (below) is the primary identification method.

## Usage

### First Session (one-time setup)

WhisperX assigns `SPEAKER_XX` labels non-deterministically — `SPEAKER_00` in one session may be a different person in the next. Enrollment solves this by creating a **voice bank** of labeled embeddings.

**1. Drop your audio file into `input_audio/`**

**2. Enroll speakers** — transcribes, uses Claude to auto-identify speakers from dialogue context (name calls, DM patterns, character abilities), and prompts only for any remaining unknowns:

```bash
python enroll.py --session "2025-12-10 21-43-18"
```

This does three things in one step:
- Creates `config/voice_bank.json` with labeled voice embeddings
- Writes the transcript to `output_transcripts/` with correct speaker names
- Marks the session as processed (so `etl.py` won't re-transcribe it)

**3. Embed the transcript** for semantic search:

```bash
python embed.py
```

That's it — your first session is fully indexed and queryable.

### Future Sessions (fully automatic)

Once the voice bank exists, future sessions are hands-free:

```bash
# 1. Drop audio into input_audio/
# 2. Transcribe — auto-matches speakers via voice bank
python etl.py
# 3. Embed
python embed.py
```

`etl.py` uses multi-embedding scoring to match new speakers against every stored voice sample, blending best-session and centroid scores for robustness. High-confidence matches automatically grow the voice bank.

If a speaker can't be matched by voice (below threshold or low confidence), `etl.py` falls back to **LLM-assisted identification** — Claude analyzes the transcript for contextual clues (name mentions, DM patterns, character abilities) to fill in the gaps. Requires `ANTHROPIC_API_KEY` to be set; skipped gracefully if not.

```bash
python etl.py              # Transcribe all new files
python etl.py --file path  # Transcribe a specific file
python etl.py --watch      # Watch for new files continuously
```

### New Player Joins

If `etl.py` encounters an unmatched voice, it appears as `SPEAKER_XX` in the transcript. Run enrollment on that session to label the new speaker and update the voice bank:

```bash
python enroll.py --session "2025-12-17 21-19-25" --add-session
```

### Auto-enrollment

If you already have a correct `players.json` mapping for a session, skip the interactive prompts:

```bash
python enroll.py --session "2025-12-10 21-43-18" --auto
```

### Embed

```bash
python embed.py              # Embed all un-embedded transcripts
python embed.py --file path  # Embed a specific transcript
python embed.py --reindex    # Re-embed everything from scratch
python embed.py --status     # Show embedding status
```

Embeddings are stored in ChromaDB at `data/chromadb/`.

### Query

```bash
# Interactive REPL (multi-turn conversation)
python agent.py

# Single query
python agent.py --query "What happened in the last session?"

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
# Full pipeline on one machine
python etl.py && python embed.py
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
│   ├── players.json       # Speaker → character mapping (static fallback)
│   ├── voice_bank.json    # Persistent voice embeddings (generated by enroll.py)
│   └── processed.json     # ETL dedup log
├── enroll.py              # Voice bank enrollment (bootstrap speaker identity)
├── etl.py                 # Audio → Markdown transcript pipeline
├── embed.py               # Transcript → ChromaDB embeddings
├── agent.py               # Claude Agent SDK campaign archivist
├── sync.sh                # rsync to remote machine
├── docker-compose.yml     # WhisperX + Open WebUI services
├── requirements-agent.txt # Python dependencies
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
