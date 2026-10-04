# local-asr-server

HTTP subtitle generation service for local audio files. It exposes a small API over `faster-whisper` and writes finished subtitle files atomically for an external file-watcher.

The image does not include models. Mount a CTranslate2/faster-whisper model directory into `/models` and choose it with environment variables. Switching models is done by changing env vars and restarting the container.

## API

```bash
curl -fsS -X POST http://127.0.0.1:9000/v1/subtitles \
  -H 'Content-Type: application/json' \
  -d '{
    "input_path": "/inputs/sample.wav",
    "language": "ja",
    "output_format": "srt",
    "uniq_key_name": "sample-ja.srt"
  }'
```

The server writes `/outputs/sample-ja.srt.tmp` first. After transcription and fsync complete, it atomically renames it to `/outputs/sample-ja.srt`. External systems should watch for the final file name only.

Supported output formats: `srt`, `vtt`, `json`, `txt`.

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
  -e ASR_COMPUTE_TYPE=int8 \
  -e ASR_DEFAULT_LANGUAGE=ja \
  ghcr.io/lgithubl/local-asr-server:m40
```

## Environment

- `ASR_MODEL_PATH`: model directory, default `/models`
- `ASR_INPUT_DIR`: read-only audio root, default `/inputs`
- `ASR_OUTPUT_DIR`: subtitle output root, default `/outputs`
- `ASR_DEVICE`: `cuda` or `cpu`, default `cuda`
- `ASR_COMPUTE_TYPE`: default `int8`, good for Tesla M40
- `ASR_DEFAULT_LANGUAGE`: default `ja`; use `zh` or `en` for other instances
- `ASR_BACKEND`: `faster-whisper` or `mock`; `mock` is for CI tests only

## Recommended instances

- Japanese: `kotoba-tech/kotoba-whisper-v2.1` converted or published as faster-whisper/CTranslate2
- Chinese: Whisper `large-v3` / `large-v3-turbo`, or a separate FunASR service if Chinese becomes dominant
- English: Whisper `large-v3` / `large-v3-turbo`

For stable production routing, run one container per model/language and route externally.

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
