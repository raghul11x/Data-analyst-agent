#!/bin/bash
# Agentic Analyser — startup script
set -e
cd "$(dirname "$0")"

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║         Agentic Analyser             ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

echo "→ Installing dependencies…"
pip install -r requirements.txt -q

echo "→ Starting at http://localhost:5050"
echo ""
python server.py
