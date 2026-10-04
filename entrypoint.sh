#!/usr/bin/env sh
set -eu

ASR_HOST="${ASR_HOST:-0.0.0.0}"
ASR_PORT="${ASR_PORT:-9000}"
ASR_BACKEND="${ASR_BACKEND:-faster-whisper}"
ASR_MODEL_PATH="${ASR_MODEL_PATH:-/models}"
ASR_INPUT_DIR="${ASR_INPUT_DIR:-/inputs}"
ASR_OUTPUT_DIR="${ASR_OUTPUT_DIR:-/outputs}"
ASR_APP_DIR="${ASR_APP_DIR:-/app}"
ASR_EXTERNAL_APP_DIR="${ASR_EXTERNAL_APP_DIR:-/workspace/local-asr-server}"
ASR_REQUIRE_EXTERNAL_CODE="${ASR_REQUIRE_EXTERNAL_CODE:-0}"

if [ -f "$ASR_EXTERNAL_APP_DIR/asr_server/main.py" ]; then
  ASR_APP_DIR="$ASR_EXTERNAL_APP_DIR"
  echo "Using external ASR code from: $ASR_APP_DIR"
elif [ "$ASR_REQUIRE_EXTERNAL_CODE" = "1" ]; then
  echo "External ASR code not found: $ASR_EXTERNAL_APP_DIR/asr_server/main.py" >&2
  exit 64
else
  echo "Using bundled ASR code from: $ASR_APP_DIR"
fi

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

cd "$ASR_APP_DIR"
export PYTHONPATH="$ASR_APP_DIR${PYTHONPATH:+:$PYTHONPATH}"

exec uvicorn asr_server.main:app --host "$ASR_HOST" --port "$ASR_PORT"
