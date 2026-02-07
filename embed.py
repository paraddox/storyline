#!/usr/bin/env python3
"""
D&D Transcript Embedding Pipeline
Chunks transcripts into speaker-aware segments, generates embeddings
via Ollama's nomic-embed-text, and stores in ChromaDB.
"""

import argparse
import re
import sys
from pathlib import Path

import chromadb
import requests

# --- Configuration ---
BASE_DIR = Path(__file__).parent
TRANSCRIPTS_DIR = BASE_DIR / "output_transcripts"
CHROMADB_DIR = BASE_DIR / "data" / "chromadb"
PLAYERS_JSON = BASE_DIR / "config" / "players.json"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "campaign_transcripts"

# Regex to parse speaker lines: **Speaker** [HH:MM:SS]: text
SPEAKER_LINE_RE = re.compile(
    r"^\*\*(.+?)\*\*\s+\[(\d{2}:\d{2}:\d{2})\]:\s*(.+)$"
)


def parse_transcript(filepath: Path) -> list[dict]:
    """Parse a Markdown transcript into speaker-turn chunks.

    Each chunk contains:
      - text: the speaker line (with optional previous line as context)
      - session: filename stem
      - speaker: character name
      - timestamp: HH:MM:SS
      - line_number: 1-based line number in file
    """
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()
    session = filepath.stem

    chunks = []
    prev_line = None

    for i, line in enumerate(lines, start=1):
        match = SPEAKER_LINE_RE.match(line.strip())
        if not match:
            continue

        speaker = match.group(1)
        timestamp = match.group(2)
        dialogue = match.group(3)

        # Build chunk text: include previous speaker line as overlap context
        if prev_line:
            chunk_text = f"{prev_line}\n{line.strip()}"
        else:
            chunk_text = line.strip()

        chunks.append({
            "text": chunk_text,
            "session": session,
            "speaker": speaker,
            "timestamp": timestamp,
            "line_number": i,
        })

        prev_line = line.strip()

    return chunks


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"embeddings": [[...]]} for /api/embed
    return data["embeddings"][0]


def get_chroma_collection(data_dir: Path = None) -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    db_path = data_dir or CHROMADB_DIR
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_embedded_sessions(collection: chromadb.Collection) -> set[str]:
    """Return set of session names already in the collection."""
    if collection.count() == 0:
        return set()
    # Get all unique session values from metadata
    results = collection.get(include=["metadatas"])
    sessions = {m["session"] for m in results["metadatas"] if "session" in m}
    return sessions


def embed_transcript(filepath: Path, collection: chromadb.Collection) -> int:
    """Chunk and embed a single transcript file. Returns chunk count."""
    chunks = parse_transcript(filepath)
    if not chunks:
        print(f"  No speaker lines found in {filepath.name}")
        return 0

    session = filepath.stem
    print(f"  Embedding {len(chunks)} chunks from {filepath.name}...")

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{session}:{chunk['line_number']}"
        embedding = get_embedding(chunk["text"])

        ids.append(chunk_id)
        documents.append(chunk["text"])
        embeddings.append(embedding)
        metadatas.append({
            "session": chunk["session"],
            "speaker": chunk["speaker"],
            "timestamp": chunk["timestamp"],
            "line_number": chunk["line_number"],
        })

        # Batch upsert every 50 chunks
        if len(ids) >= 50:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            ids, documents, embeddings, metadatas = [], [], [], []

    # Upsert remaining
    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"  Done: {len(chunks)} chunks embedded for {session}")
    return len(chunks)


def remove_session(session: str, collection: chromadb.Collection):
    """Remove all chunks for a session from the collection."""
    if collection.count() == 0:
        return
    results = collection.get(
        where={"session": session},
        include=[],
    )
    if results["ids"]:
        collection.delete(ids=results["ids"])
        print(f"  Removed {len(results['ids'])} existing chunks for {session}")


def main():
    parser = argparse.ArgumentParser(description="Embed D&D transcripts into ChromaDB")
    parser.add_argument("--file", type=Path, help="Embed a specific transcript file")
    parser.add_argument("--reindex", action="store_true", help="Re-embed all transcripts")
    parser.add_argument("--data-dir", type=Path, help="ChromaDB storage directory")
    parser.add_argument("--status", action="store_true", help="Show embedding status")
    args = parser.parse_args()

    data_dir = args.data_dir or CHROMADB_DIR
    collection = get_chroma_collection(data_dir)

    if args.status:
        embedded = get_embedded_sessions(collection)
        transcripts = list(TRANSCRIPTS_DIR.glob("*.md"))
        print(f"ChromaDB: {collection.count()} chunks across {len(embedded)} sessions")
        print(f"Transcripts on disk: {len(transcripts)}")
        for t in sorted(transcripts):
            status = "embedded" if t.stem in embedded else "pending"
            print(f"  [{status}] {t.name}")
        return

    # Determine which files to process
    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        files = [args.file]
    else:
        files = sorted(TRANSCRIPTS_DIR.glob("*.md"))
        if not files:
            print(f"No transcripts found in {TRANSCRIPTS_DIR}")
            print("Run: python etl.py  (to transcribe audio first)")
            return

    embedded_sessions = get_embedded_sessions(collection)
    total_chunks = 0

    for filepath in files:
        session = filepath.stem
        if session in embedded_sessions and not args.reindex:
            print(f"  Skipping {filepath.name} (already embedded)")
            continue

        if args.reindex and session in embedded_sessions:
            remove_session(session, collection)

        try:
            count = embed_transcript(filepath, collection)
            total_chunks += count
        except requests.exceptions.ConnectionError:
            print(f"\nError: Cannot connect to Ollama at {OLLAMA_URL}")
            print("Make sure Ollama is running: ollama serve")
            sys.exit(1)
        except Exception as e:
            print(f"  Error embedding {filepath.name}: {e}")
            raise

    print(f"\nTotal: {total_chunks} new chunks embedded")
    print(f"Collection now has {collection.count()} chunks total")


if __name__ == "__main__":
    main()
