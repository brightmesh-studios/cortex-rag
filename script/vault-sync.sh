#!/bin/bash

VAULT=/Users/path/to/vault
RAG=/Users/path/to/cortex-rag

cd $VAULT

prettier --write .

git pull --rebase --autostash

if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
  git add .
  git commit -m "vault: auto-sync $(date '+%Y-%m-%d %H:%M')"
  git push
  echo "[$(date '+%Y-%m-%d %H:%M')] vault synced"
else
  echo "[$(date '+%Y-%m-%d %H:%M')] vault unchanged, skipping commit"
fi

cd $RAG

source .venv/bin/activate

python src/ingest.py --changed-only

echo "[$(date '+%Y-%m-%d %H:%M')] RAG index updated"%  