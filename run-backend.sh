#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/backend"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -r requirements.txt
if [ ! -f .env ]; then cp .env.example .env; echo "Edit backend/.env and add GROQ_API_KEY, then run this script again."; exit 1; fi
uvicorn app:app --reload --host 127.0.0.1 --port 8000
