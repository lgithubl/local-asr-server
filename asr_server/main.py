from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Literal
import queue

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import AliasChoices, BaseModel, Field

from . import __version__
from .config import settings
from .jobs import JobQueue, SubtitleJob, format_exception, new_job_id
from .paths import PathValidationError, atomic_write, resolve_input_path, resolve_output_path, validate_output_key, validate_output_subdir
from .subtitles import render_subtitles
from .tracing import trace_store
from .transcriber import Transcriber


app = FastAPI(title="local-asr-server", version=__version__)
transcriber = Transcriber(settings)
job_queue: JobQueue | None = None
idle_unload_timer: threading.Timer | None = None


class SubtitleRequest(BaseModel):
    input_path: str = Field(..., description="Absolute path under ASR_INPUT_DIR, or relative path inside it")
    language: str | None = Field(None, description="ja, zh, en, or any Whisper language code")
    output_format: Literal["srt", "vtt", "json", "txt"] = "srt"
    output_dir: str | None = Field(None, description="Optional relative directory under ASR_OUTPUT_DIR")
    uniq_key_name: str | None = Field(None, description="Final output file name, written under ASR_OUTPUT_DIR")
    vad_filter: bool | None = Field(None, description="Override ASR_VAD_FILTER for this request")
    segmenter: Literal["none", "asmr-onnx"] = Field("none", description="Optional pre-segmenter before ASR")
    asmr_vad: bool | None = Field(None, description="Shortcut for segmenter=asmr-onnx when true")
    async_mode: bool = Field(
        False,
        validation_alias=AliasChoices("async", "async_mode"),
        description="Return immediately and continue writing the output file in the background",
    )


def default_output_key(audio_path: Path, language: str | None, output_format: str) -> str:
    stat = audio_path.stat()
    digest = hashlib.sha256(f"{audio_path.name}:{stat.st_size}:{stat.st_mtime_ns}:{language}:{output_format}".encode()).hexdigest()[:16]
    return f"{audio_path.stem}.{digest}.{output_format}"


def write_subtitle_file(
    audio_path: Path,
    output_subdir: str,
    output_key: str,
    language: str | None,
    output_format: str,
    vad_filter: bool | None,
    segmenter: str,
) -> tuple[Path, int]:
    segments = transcriber.transcribe(audio_path, language, vad_filter=vad_filter, segmenter=segmenter)
    with trace_store.stage("render_subtitle"):
        content = render_subtitles(segments, output_format)
    output_path = atomic_write(settings.output_dir, output_key, content, output_subdir=output_subdir)
    return output_path, len(segments)


def schedule_idle_unload() -> None:
    global idle_unload_timer
    if settings.idle_unload_seconds <= 0 or settings.backend == "mock":
        return
    if idle_unload_timer is not None:
        idle_unload_timer.cancel()
    idle_unload_timer = threading.Timer(settings.idle_unload_seconds, unload_model_if_idle)
    idle_unload_timer.daemon = True
    idle_unload_timer.start()


def cancel_idle_unload() -> None:
    global idle_unload_timer
    if idle_unload_timer is not None:
        idle_unload_timer.cancel()
        idle_unload_timer = None


def unload_model_if_idle() -> None:
    global idle_unload_timer
    idle_unload_timer = None
    if job_queue is not None:
        snapshot = job_queue.snapshot()
        if snapshot["doing"] is not None or snapshot["pending_count"]:
            return
    transcriber.unload()


def err_log_key(output_key: str) -> str:
    return validate_output_key(f"{output_key}.err.log")


def atomic_write_error_log(job: SubtitleJob, exc: BaseException) -> None:
    content = format_exception(exc)
    record = trace_store.get(job.trace_id)
    token = trace_store.set_current(record)
    try:
        with trace_store.stage("write_error_log"):
            atomic_write(settings.output_dir, err_log_key(job.output_key), content, output_subdir=job.output_dir)
    finally:
        trace_store.reset_current(token)


def run_job(job: SubtitleJob) -> None:
    cancel_idle_unload()
    record = trace_store.get(job.trace_id)
    token = trace_store.set_current(record)
    try:
        output_path, segment_count = write_subtitle_file(
            Path(job.input_path),
            job.output_dir,
            job.output_key,
            job.language,
            job.output_format,
            job.vad_filter,
            job.segmenter,
        )
        if record is not None:
            record.output_path = str(output_path)
            record.segment_count = segment_count
    finally:
        trace_store.reset_current(token)


def on_job_start(job: SubtitleJob) -> None:
    trace_store.mark_started(job.trace_id, job.started_at)


def on_job_success(job: SubtitleJob) -> None:
    record = trace_store.get(job.trace_id)
    trace_store.mark_done(job.trace_id, record.segment_count if record else None)


def on_job_error(job: SubtitleJob, exc: BaseException) -> None:
    trace_store.mark_error(job.trace_id, exc)


def model_snapshot() -> dict:
    return {
        "loaded": transcriber.model_loaded,
        "idle_unload_seconds": settings.idle_unload_seconds,
        "backend": settings.backend,
        "device": settings.device,
        "compute_type": settings.compute_type,
    }


def resolve_segmenter(request: SubtitleRequest) -> str:
    if request.asmr_vad is True:
        return "asmr-onnx"
    if request.asmr_vad is False and request.segmenter == "asmr-onnx":
        return "none"
    return request.segmenter


def fsync_unlink(path: Path) -> None:
    try:
        path.unlink()
        dir_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except FileNotFoundError:
        return


@app.on_event("startup")
def startup() -> None:
    global job_queue
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    trace_store.enabled = settings.trace_enabled
    trace_store.max_items = max(1, settings.trace_max_items)
    job_queue = JobQueue(
        settings.max_queue_size,
        run_job,
        atomic_write_error_log,
        schedule_idle_unload,
        start_callback=on_job_start,
        success_callback=on_job_success,
        error_callback=on_job_error,
    )
    job_queue.start()
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
    # worker 线程死掉后队列会永久停摆，而进程照常响应——必须在这里暴露出来，
    # 否则只能等客户端轮询超时才发现。
    worker_alive = job_queue.worker_alive() if job_queue is not None else True
    output_dir_writable = (
        settings.output_dir.is_dir() and os.access(settings.output_dir, os.W_OK)
    )
    return {
        "ok": worker_alive and output_dir_writable,
        "worker_alive": worker_alive,
        "version": __version__,
        "backend": settings.backend,
        "model_path": settings.model_path,
        "model_loaded": transcriber.model_loaded,
        "device": settings.device,
        "compute_type": settings.compute_type,
        "default_language": settings.default_language,
        "vad_filter": settings.vad_filter,
        "idle_unload_seconds": settings.idle_unload_seconds,
        "queue": job_queue.snapshot() if job_queue else None,
        "asmr_vad_model_path": settings.asmr_vad_model_path,
        "input_dir": str(settings.input_dir),
        "output_dir": str(settings.output_dir),
        "output_dir_writable": output_dir_writable,
    }


@app.get("/v1/jobs")
def list_jobs() -> dict:
    if job_queue is None:
        return {"max_size": settings.max_queue_size, "doing": None, "pending": [], "pending_count": 0, "doing_count": 0}
    return job_queue.snapshot()


@app.get("/v1/stats")
def stats() -> dict:
    snapshot = job_queue.snapshot() if job_queue else None
    return trace_store.summary(snapshot, model_snapshot())


@app.get("/v1/traces")
def traces(status: str | None = None, limit: int | None = None) -> dict:
    safe_limit = None if limit is None else max(0, min(limit, settings.trace_max_items))
    return {"items": trace_store.list_records(status=status, limit=safe_limit)}


@app.get("/v1/traces/{job_id}")
def trace_detail(job_id: str) -> dict:
    record = trace_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return record.to_dict()


@app.post("/v1/subtitles")
def create_subtitles(request: SubtitleRequest) -> dict:
    job_id = new_job_id()
    record = trace_store.create(
        job_id,
        status="pending" if request.async_mode else "doing",
        input_path=request.input_path,
        output_dir=request.output_dir or "",
        language=request.language,
        async_mode=request.async_mode,
        vad_filter=request.vad_filter,
        segmenter=request.segmenter,
        output_format=request.output_format,
    )
    token = trace_store.set_current(record)
    try:
        with trace_store.stage("validate_request"):
            audio_path = resolve_input_path(settings.input_dir, request.input_path)
            output_subdir = validate_output_subdir(request.output_dir)
            output_key = validate_output_key(
                request.uniq_key_name or default_output_key(audio_path, request.language, request.output_format)
            )
            output_path = resolve_output_path(settings.output_dir, output_subdir, output_key)
        tmp_path = Path(str(output_path) + ".tmp")
        error_path = resolve_output_path(settings.output_dir, output_subdir, err_log_key(output_key))
        segmenter = resolve_segmenter(request)
        if record is not None:
            record.input_path = str(audio_path)
            record.output_dir = output_subdir
            record.output_key = output_key
            record.output_path = str(output_path)
            record.segmenter = segmenter

        if request.async_mode:
            if job_queue is None:
                raise RuntimeError("job queue is not initialized")
            cancel_idle_unload()
            fsync_unlink(error_path)
            job = SubtitleJob(
                job_id=job_id,
                input_path=str(audio_path),
                trace_id=job_id,
                output_dir=output_subdir,
                output_key=output_key,
                output_path=str(output_path),
                tmp_path=str(tmp_path),
                err_log_path=str(error_path),
                language=request.language,
                output_format=request.output_format,
                vad_filter=request.vad_filter,
                segmenter=segmenter,
            )
            job_queue.enqueue(job)
            return {
                "ok": True,
                "accepted": True,
                "async": True,
                "job_id": job.job_id,
                "trace_id": job.trace_id,
                "output_dir": output_subdir,
                "output_key": output_key,
                "output_path": str(output_path),
                "tmp_path": str(tmp_path),
                "err_log_path": str(error_path),
            }

        trace_store.mark_started(job_id)
        output_path, segment_count = write_subtitle_file(
            audio_path,
            output_subdir,
            output_key,
            request.language,
            request.output_format,
            request.vad_filter,
            segmenter,
        )
        schedule_idle_unload()
        trace_store.mark_done(job_id, segment_count)
    except PathValidationError as exc:
        trace_store.mark_error(job_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except queue.Full as exc:
        trace_store.mark_error(job_id, "job queue is full")
        raise HTTPException(status_code=429, detail="job queue is full") from exc
    except ValueError as exc:
        trace_store.mark_error(job_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        trace_store.mark_error(job_id, exc)
        if "output_key" in locals():
            fallback_job = SubtitleJob(
                job_id=job_id,
                input_path=str(audio_path) if "audio_path" in locals() else request.input_path,
                trace_id=job_id,
                output_dir=output_subdir if "output_subdir" in locals() else "",
                output_key=output_key,
                output_path=str(output_path) if "output_path" in locals() else str(settings.output_dir / output_key),
                tmp_path=str(tmp_path) if "tmp_path" in locals() else str(settings.output_dir / f"{output_key}.tmp"),
                err_log_path=str(error_path) if "error_path" in locals() else str(settings.output_dir / err_log_key(output_key)),
                language=request.language,
                output_format=request.output_format,
                vad_filter=request.vad_filter,
                segmenter=segmenter if "segmenter" in locals() else "none",
            )
            atomic_write_error_log(fallback_job, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        trace_store.reset_current(token)

    return {
        "ok": True,
        "accepted": False,
        "async": False,
        "job_id": job_id,
        "trace_id": job_id,
        "output_dir": output_subdir,
        "output_key": output_key,
        "output_path": str(output_path),
        "segments": segment_count,
        "tmp_path": str(output_path) + ".tmp",
        "err_log_path": str(error_path),
    }
