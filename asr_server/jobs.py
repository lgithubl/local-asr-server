from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal


JobStatus = Literal["pending", "doing"]


@dataclass
class SubtitleJob:
    job_id: str
    input_path: str
    output_key: str
    output_path: str
    tmp_path: str
    err_log_path: str
    language: str | None
    output_format: str
    vad_filter: bool | None
    segmenter: str
    status: JobStatus = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None

    def public_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "input_path": self.input_path,
            "output_key": self.output_key,
            "output_path": self.output_path,
            "tmp_path": self.tmp_path,
            "err_log_path": self.err_log_path,
            "language": self.language,
            "output_format": self.output_format,
            "vad_filter": self.vad_filter,
            "segmenter": self.segmenter,
            "created_at": self.created_at,
            "started_at": self.started_at,
        }


class JobQueue:
    def __init__(
        self,
        max_size: int,
        handler: Callable[[SubtitleJob], None],
        error_writer: Callable[[SubtitleJob, BaseException], None],
        idle_callback: Callable[[], None] | None = None,
    ):
        self.max_size = max_size
        self.handler = handler
        self.error_writer = error_writer
        self.idle_callback = idle_callback
        self._queue: queue.Queue[SubtitleJob | None] = queue.Queue(maxsize=max_size)
        self._pending: list[SubtitleJob] = []
        self._doing: SubtitleJob | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="subtitle-job-worker", daemon=True)
        self._thread.start()

    def enqueue(self, job: SubtitleJob) -> None:
        with self._lock:
            if len(self._pending) + (1 if self._doing else 0) >= self.max_size:
                raise queue.Full
            if any(existing.output_key == job.output_key for existing in self._pending) or (
                self._doing is not None and self._doing.output_key == job.output_key
            ):
                raise ValueError("a job with the same output key is already pending or running")
            self._pending.append(job)
        self._queue.put(job)

    def snapshot(self) -> dict:
        with self._lock:
            doing = self._doing.public_dict() if self._doing else None
            pending = [job.public_dict() for job in self._pending]
        return {
            "max_size": self.max_size,
            "doing": doing,
            "pending": pending,
            "pending_count": len(pending),
            "doing_count": 1 if doing else 0,
        }

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            with self._lock:
                self._pending = [pending for pending in self._pending if pending.job_id != job.job_id]
                job.status = "doing"
                job.started_at = time.time()
                self._doing = job
            try:
                self.handler(job)
            except Exception as exc:
                self.error_writer(job, exc)
            finally:
                with self._lock:
                    if self._doing and self._doing.job_id == job.job_id:
                        self._doing = None
                    became_idle = self._doing is None and not self._pending
                self._queue.task_done()
                if became_idle and self.idle_callback is not None:
                    self.idle_callback()


def new_job_id() -> str:
    return uuid.uuid4().hex


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
