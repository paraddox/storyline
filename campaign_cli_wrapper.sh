#!/bin/bash
# Wrapper to run campaign_cli.py with the correct venv Python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/campaign_cli.py" "$@"
