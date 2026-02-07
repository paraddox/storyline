#!/bin/bash
# Sync transcripts + ChromaDB + config to a remote machine
# Usage: ./sync.sh user@host [dest_dir]

set -euo pipefail

REMOTE="${1:?Usage: sync.sh user@host [dest_dir]}"
DEST="${2:-~/storyline}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Syncing storyline data to ${REMOTE}:${DEST}"

rsync -avz --progress \
  "${SCRIPT_DIR}/output_transcripts/" \
  "${REMOTE}:${DEST}/output_transcripts/"

rsync -avz --progress \
  "${SCRIPT_DIR}/data/chromadb/" \
  "${REMOTE}:${DEST}/data/chromadb/"

rsync -avz --progress \
  "${SCRIPT_DIR}/config/players.json" \
  "${REMOTE}:${DEST}/config/players.json"

# Sync voice bank if it exists
if [ -f "${SCRIPT_DIR}/config/voice_bank.json" ]; then
  rsync -avz --progress \
    "${SCRIPT_DIR}/config/voice_bank.json" \
    "${REMOTE}:${DEST}/config/voice_bank.json"
fi

echo ""
echo "Sync complete. On the remote machine run:"
echo "  cd ${DEST} && python agent.py --data-dir ${DEST}/data/chromadb"
