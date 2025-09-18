#!/usr/bin/env bash
set -euo pipefail

DB_URI="${SURREAL_URI:-file:./data/surreal.db}"
PORT="${SURREAL_PORT:-8000}"
USER="${SURREALDB_USER:-root}"
PASS="${SURREALDB_PASS:-root}"

mkdir -p ./data
exec surreal start --user "$USER" --pass "$PASS" --bind 127.0.0.1:"$PORT" "$DB_URI"

