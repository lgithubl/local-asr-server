from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import settings
from .paths import PathValidationError, atomic_write, resolve_input_path, validate_output_key
from .subtitles import render_subtitles
from .transcriber import Transcriber


app = FastAPI(title="local-asr-server", version=__version__)
transcriber = Transcriber(settings)


class SubtitleRequest(BaseModel):
    input_path: str = Field(..., description="Absolute path under ASR_INPUT_DIR, or relative path inside it")
    language: str | None = Field(None, description="ja, zh, en, or any Whisper language code")
    output_format: Literal["srt", "vtt", "json", "txt"] = "srt"
    uniq_key_name: str | None = Field(None, description="Final output file name, written under ASR_OUTPUT_DIR")


def default_output_key(audio_path: Path, language: str | None, output_format: str) -> str:
    stat = audio_path.stat()
    digest = hashlib.sha256(f"{audio_path.name}:{stat.st_size}:{stat.st_mtime_ns}:{language}:{output_format}".encode()).hexdigest()[:16]
    return f"{audio_path.stem}.{digest}.{output_format}"


@app.on_event("startup")
def startup() -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    if settings.preload_model:
        transcriber.preload()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <!doctype html>
    <html><head><title>local-asr-server</title></head>
    <body>
      <h1>local-asr-server</h1>
      <p>POST <code>/v1/subtitles</code> with a local input path to generate subtitles.</p>
      <p>See <a href="/docs">/docs</a> and <a href="/health">/health</a>.</p>
    </body></html>
    """


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "version": __version__,
        "backend": settings.backend,
        "model_path": settings.model_path,
        "model_loaded": transcriber.model_loaded,
        "device": settings.device,
        "compute_type": settings.compute_type,
        "default_language": settings.default_language,
        "input_dir": str(settings.input_dir),
        "output_dir": str(settings.output_dir),
        "output_dir_writable": settings.output_dir.exists() and settings.output_dir.is_dir(),
    }


@app.post("/v1/subtitles")
def create_subtitles(request: SubtitleRequest) -> dict:
    try:
        audio_path = resolve_input_path(settings.input_dir, request.input_path)
        output_key = validate_output_key(
            request.uniq_key_name or default_output_key(audio_path, request.language, request.output_format)
        )
        segments = transcriber.transcribe(audio_path, request.language)
        content = render_subtitles(segments, request.output_format)
        output_path = atomic_write(settings.output_dir, output_key, content)
    except PathValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ok": True,
        "output_key": output_key,
        "output_path": str(output_path),
        "segments": len(segments),
        "tmp_path": str(output_path) + ".tmp",
    }
