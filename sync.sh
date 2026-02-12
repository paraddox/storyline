#!/bin/bash
# Sync data files (transcripts, embeddings, config) to a remote machine.
# The remote should already have the repo cloned — this only copies data.
# Usage: ./sync.sh user@host [dest_dir]

set -euo pipefail

REMOTE="${1:?Usage: sync.sh user@host [dest_dir]}"
DEST="${2:-~/storyline}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Syncing storyline data to ${REMOTE}:${DEST}"

rsync -avz --delete --progress \
  --include='output_transcripts/***' \
  --include='data/chromadb/***' \
  --include='config/players.json' \
  --include='config/voice_bank.json' \
  --include='data/' \
  --include='config/' \
  --exclude='*' \
  "${SCRIPT_DIR}/" "${REMOTE}:${DEST}/"

echo ""
echo "Sync complete. Campaign data is ready on the remote."
echo "  OpenClaw: data available at ${DEST} (agent reads via AGENTS.md)"
echo "  Standalone: cd ${DEST} && python agent.py"
