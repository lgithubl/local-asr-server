# local-asr-server

HTTP subtitle generation service for local audio files. It exposes a small API over `faster-whisper` and writes finished subtitle files atomically for an external file-watcher.

The image does not include models. Mount a CTranslate2/faster-whisper model directory into `/models` and choose it with environment variables. Switching models is done by changing env vars and restarting the container.

The image includes a bundled copy of the app code so it can run standalone. For later code-only fixes, unpack the code artifact from Actions and mount it at `/workspace/local-asr-server:ro`; the container will use the mounted code before the bundled copy.

## API

```bash
curl -fsS -X POST http://127.0.0.1:9000/v1/subtitles \
  -H 'Content-Type: application/json' \
  -d '{
    "input_path": "/inputs/sample.wav",
    "language": "ja",
    "output_format": "srt",
    "output_dir": "rj123456",
    "uniq_key_name": "sample-ja.srt"
  }'
```

The server writes `/outputs/rj123456/sample-ja.srt.tmp` first. After transcription and fsync complete, it atomically renames it to `/outputs/rj123456/sample-ja.srt`. External systems should watch for the final file name only.

Supported output formats: `srt`, `vtt`, `json`, `txt`.

Request options:

- `output_dir`: optional relative directory under `ASR_OUTPUT_DIR`, for example `rj123456` or `rj123456/part01`. The server creates it automatically. Absolute paths, `..`, backslashes, and unsafe characters are rejected.
- `vad_filter`: optional boolean. Overrides `ASR_VAD_FILTER` for one request.
- `async` or `async_mode`: optional boolean. When true, the API returns after validation and the server continues writing the output file in the background. Completion is still detected by the final output file appearing.
- `segmenter`: optional, `none` or `asmr-onnx`. When `asmr-onnx`, the server uses the mounted `Whisper-Vad-EncDec-ASMR-onnx` model to split speech before ASR.
- `asmr_vad`: optional boolean shortcut. `true` means `segmenter=asmr-onnx`.

Async example:

```bash
curl -fsS -X POST http://127.0.0.1:9000/v1/subtitles \
  -H 'Content-Type: application/json' \
  -d '{
    "input_path": "/inputs/sample.wav",
    "language": "ja",
    "output_format": "srt",
    "uniq_key_name": "sample-ja.srt",
    "vad_filter": true,
    "segmenter": "asmr-onnx",
    "async": true
  }'
```

Queue status:

```bash
curl -fsS http://127.0.0.1:9000/v1/jobs
```

Only `doing` and `pending` jobs are kept in `/v1/jobs`. Finished jobs disappear from the queue. If a background job fails, the server writes `<final-output-path>.err.log` in the same output directory; successful jobs only produce the final subtitle file.

Trace and stats endpoints:

```bash
curl -fsS http://127.0.0.1:9000/v1/stats
curl -fsS http://127.0.0.1:9000/v1/traces
curl -fsS http://127.0.0.1:9000/v1/traces/<job_id>
```

Trace data is memory-only and resets when the container restarts. It records recent done/error/doing/pending jobs and stage timings such as `validate_request`, `queue_wait`, `load_model`, `decode_audio`, `asmr_vad_segment`, `faster_whisper_transcribe`, `render_subtitle`, `write_tmp_file`, `rename_final_file`, and `write_error_log`. Subtitle text is not stored in traces.

## Run on Docker

```bash
docker run -d \
  --name local-asr-ja \
  --restart unless-stopped \
  --gpus all \
  -p 9001:9000 \
  -v /data/asr-inputs:/inputs:ro \
  -v /data/asr-outputs:/outputs \
  -v /data/asr-models/kotoba-whisper-v2.1-ct2:/models:ro \
  -e ASR_MODEL_PATH=/models \
  -e ASR_DEVICE=cuda \
  -e ASR_COMPUTE_TYPE=float32 \
  -e ASR_DEFAULT_LANGUAGE=ja \
  -e ASR_IDLE_UNLOAD_SECONDS=300 \
  ghcr.io/lgithubl/local-asr-server:m40
```

Run with external code mounted:

```bash
docker run -d \
  --name local-asr-ja \
  --restart unless-stopped \
  --gpus all \
  -p 9001:9000 \
  -v /data/asr-inputs:/inputs:ro \
  -v /data/asr-outputs:/outputs \
  -v /data/asr-models/asr-model-ja-large-v3:/models:ro \
  -v /data/local-asr-server-code/local-asr-server-code-m40:/workspace/local-asr-server:ro \
  -e ASR_REQUIRE_EXTERNAL_CODE=1 \
  -e ASR_MODEL_PATH=/models \
  -e ASR_DEVICE=cuda \
  -e ASR_COMPUTE_TYPE=float32 \
  -e ASR_DEFAULT_LANGUAGE=ja \
  -e ASR_VAD_FILTER=0 \
  -e ASR_IDLE_UNLOAD_SECONDS=300 \
  ghcr.io/lgithubl/local-asr-server:m40
```

## Environment

- `ASR_MODEL_PATH`: model directory, default `/models`
- `ASR_INPUT_DIR`: read-only audio root, default `/inputs`
- `ASR_OUTPUT_DIR`: subtitle output root, default `/outputs`
- `ASR_APP_DIR`: bundled app directory, default `/app`
- `ASR_EXTERNAL_APP_DIR`: optional external app-code mount, default `/workspace/local-asr-server`
- `ASR_REQUIRE_EXTERNAL_CODE`: default `0`. Set `1` to fail startup unless external code is mounted.
- `ASR_DEVICE`: `cuda` or `cpu`, default `cuda`
- `ASR_COMPUTE_TYPE`: default `float32`, safest for Tesla M40. Use `int8` only on GPUs/backends that support it.
- `ASR_DEFAULT_LANGUAGE`: default `ja`; use `zh` or `en` for other instances
- `ASR_VAD_FILTER`: default `0`. Keep it off for subtitle completeness, especially quiet Japanese/ASMR audio.
- `ASR_BACKEND`: `faster-whisper` or `mock`; `mock` is for CI tests only
- `ASR_IDLE_UNLOAD_SECONDS`: default `0` disables unloading. Set `300` to unload the ASR model after 5 idle minutes.
- `ASR_MAX_QUEUE_SIZE`: async job queue size, default `64`
- `ASR_TRACE_ENABLED`: default `1`. Enables in-memory job trace and stage timing endpoints.
- `ASR_TRACE_MAX_ITEMS`: default `500`. Number of recent trace records retained in memory.
- `ASR_ASMR_VAD_MODEL_PATH`: optional path to `Whisper-Vad-EncDec-ASMR-onnx` `model.onnx`
- `ASR_ASMR_VAD_METADATA_PATH`: optional path to `model_metadata.json`
- `ASR_ASMR_VAD_FEATURE_EXTRACTOR_PATH`: optional path containing Whisper feature extractor files

## Recommended instances

- Japanese: `kotoba-tech/kotoba-whisper-v2.1` converted or published as faster-whisper/CTranslate2
- Chinese: Whisper `large-v3` / `large-v3-turbo`, or a separate FunASR service if Chinese becomes dominant
- English: Whisper `large-v3` / `large-v3-turbo`

For stable production routing, run one container per model/language and route externally.

## Image and code artifacts from Actions

Run **Build Image** from GitHub Actions for the first install. It uploads these artifacts by default:

- `local-asr-server-m40`: Docker image tarball, load with `gzip -dc local-asr-server-m40.tar.gz | docker load`
- `local-asr-server-code-m40`: mountable app code tarball for code-only updates

Unpack the code artifact like this:

```bash
mkdir -p /data/local-asr-server-code
tar -C /data/local-asr-server-code -xzf local-asr-server-code-m40.tar.gz
```

Then add this mount to the Docker or k3s spec:

```bash
-v /data/local-asr-server-code/local-asr-server-code-m40:/workspace/local-asr-server:ro \
-e ASR_REQUIRE_EXTERNAL_CODE=1
```

If you do not mount external code, the container uses the code bundled in the image.

For code-only updates after the first install, run **Build Code Pack** instead. It uploads only `local-asr-server-code-m40`; unpack it over the mounted code directory and restart the container. The large Docker image does not need to be reloaded.

## Model artifacts from Actions

Run **Build Model Pack** from GitHub Actions. It uploads two artifacts:

- `${artifact_name}`: compressed `tar.zst` model-only package
- `${artifact_name}-runtime`: ready-to-run package with `models/`, empty `inputs/`, empty `outputs/`, `docker.env`, and `run-docker.sh`

For Japanese, use the artifact name `asr-model-ja-kotoba` and `default_language=ja`. For Chinese, use `asr-model-zh-large-v3` and `default_language=zh`. Download `${artifact_name}-runtime`, unzip it, put audio files into `inputs/`, and run:

```bash
cd asr-model-ja-kotoba-runtime
./run-docker.sh ghcr.io/lgithubl/local-asr-server:m40
```

If you only download the model-only artifact, unpack it as:

```bash
mkdir -p /data/asr-models/asr-model-ja-kotoba
tar -C /data/asr-models/asr-model-ja-kotoba -I zstd -xf asr-model-ja-kotoba.tar.zst
```

Run **Build ASMR VAD Pack** to download `Whisper-Vad-EncDec-ASMR-onnx` plus the required Whisper feature extractor files. Mount it like this:

```bash
-v /data/asr-models/asr-asmr-vad-onnx:/asmr-vad:ro \
-e ASR_ASMR_VAD_MODEL_PATH=/asmr-vad/model.onnx \
-e ASR_ASMR_VAD_METADATA_PATH=/asmr-vad/model_metadata.json \
-e ASR_ASMR_VAD_FEATURE_EXTRACTOR_PATH=/asmr-vad
```

## k3s with Tesla M40

The node must already expose `nvidia.com/gpu`. See `examples/k3s-pod.yaml` for a minimal pod.

Load an exported image artifact:

```bash
gzip -dc local-asr-server-m40.tar.gz | docker load
docker image inspect ghcr.io/lgithubl/local-asr-server:m40
```

Then import it into your k3s runtime if needed, for example with containerd:

```bash
gzip -dc local-asr-server-m40.tar.gz | sudo k3s ctr images import -
```

## Development smoke test

```bash
ASR_BACKEND=mock ASR_INPUT_DIR=/tmp/asr-inputs ASR_OUTPUT_DIR=/tmp/asr-outputs \
  uvicorn asr_server.main:app --host 127.0.0.1 --port 9000
```

Then open `http://127.0.0.1:9000/` or call `/health`.
