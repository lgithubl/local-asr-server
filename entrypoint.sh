#!/usr/bin/env sh
set -eu

ASR_HOST="${ASR_HOST:-0.0.0.0}"
ASR_PORT="${ASR_PORT:-9000}"
ASR_BACKEND="${ASR_BACKEND:-faster-whisper}"
ASR_MODEL_PATH="${ASR_MODEL_PATH:-/models}"
ASR_INPUT_DIR="${ASR_INPUT_DIR:-/inputs}"
ASR_OUTPUT_DIR="${ASR_OUTPUT_DIR:-/outputs}"

if [ "$ASR_BACKEND" != "mock" ] && [ ! -e "$ASR_MODEL_PATH" ]; then
  echo "Model path not found: $ASR_MODEL_PATH" >&2
  echo "Mount a model volume at /models or set ASR_MODEL_PATH." >&2
  exit 64
fi

if [ ! -d "$ASR_INPUT_DIR" ]; then
  echo "Input directory not found: $ASR_INPUT_DIR" >&2
  exit 64
fi

mkdir -p "$ASR_OUTPUT_DIR"

exec uvicorn asr_server.main:app --host "$ASR_HOST" --port "$ASR_PORT"
