---
name: campaign-archivist
description: D&D campaign transcript search via campaign_cli.py
metadata: {"openclaw":{"requires":{"bins":["python3"]}}}
---

# Campaign Archivist Skill

Search and retrieve D&D campaign session transcripts using `campaign_cli.py`.

## Subcommands

### List Sessions

```bash
exec python3 campaign_cli.py sessions
```

Shows all available transcripts with file sizes and ChromaDB embedding status. Run this first to see what data exists.

### Exact Text Search

```bash
exec python3 campaign_cli.py search "query"
exec python3 campaign_cli.py search "query" --speaker "Narrator"
exec python3 campaign_cli.py search "query" --max 10
```

Case-insensitive text search across all transcripts. Best for specific names, quotes, and keywords. Use `--speaker` to filter to a specific character.

### Semantic Search

```bash
exec python3 campaign_cli.py semantic "query"
exec python3 campaign_cli.py semantic "query" --speaker "Narrator"
exec python3 campaign_cli.py semantic "query" --n 5
```

Vector similarity search via ChromaDB + Ollama embeddings. Best for conceptual/thematic queries like "moments of betrayal" or "party strategy discussion". Requires Ollama with `nomic-embed-text` model.

### Campaign Info

```bash
exec python3 campaign_cli.py info
```

Shows player/character roster and voice bank metadata. Use this instead of reading `voice_bank.json` directly (which contains 41K+ tokens of embedding vectors).

## Reading Full Transcripts

Use the native `read` tool to access full transcript files:

```
read output_transcripts/2025-12-10 21-43-18.md
```

## When to Use Which Tool

| Question Type | Tool |
|---|---|
| "What sessions exist?" | `sessions` |
| "What did X say about Y?" | `search "Y" --speaker X` |
| "Find mentions of the artifact" | `search "artifact"` |
| "When did the party discuss strategy?" | `semantic "party strategy discussion"` |
| "Tell me about the full session" | `read` the transcript file |
| "Who are the players?" | `info` |
