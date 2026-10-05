from __future__ import annotations

import contextlib
import contextvars
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator


_current_trace: contextvars.ContextVar["TraceRecord | None"] = contextvars.ContextVar("asr_trace", default=None)


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class TraceRecord:
    job_id: str
    status: str
    input_path: str | None = None
    output_key: str | None = None
    output_dir: str = ""
    output_path: str | None = None
    language: str | None = None
    async_mode: bool = False
    vad_filter: bool | None = None
    segmenter: str = "none"
    output_format: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    stages_ms: dict[str, int] = field(default_factory=dict)
    segment_count: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        total_ms = None
        if self.finished_at is not None:
            total_ms = max(0, int((self.finished_at - self.created_at) * 1000))
        elif self.started_at is not None:
            total_ms = max(0, int((time.time() - self.created_at) * 1000))
        return {
            "job_id": self.job_id,
            "status": self.status,
            "input_path": self.input_path,
            "output_key": self.output_key,
            "output_dir": self.output_dir,
            "output_path": self.output_path,
            "language": self.language,
            "async": self.async_mode,
            "vad_filter": self.vad_filter,
            "segmenter": self.segmenter,
            "output_format": self.output_format,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "queue_wait_ms": self.stages_ms.get("queue_wait"),
            "total_ms": total_ms,
            "stages": dict(self.stages_ms),
            "segment_count": self.segment_count,
            "error": self.error,
        }


class TraceStore:
    def __init__(self, enabled: bool = True, max_items: int = 500):
        self.enabled = enabled
        self.max_items = max(1, max_items)
        self._records: deque[TraceRecord] = deque()
        self._by_id: dict[str, TraceRecord] = {}
        self._lock = threading.Lock()
        self._submitted = 0
        self._done = 0
        self._error = 0

    def create(self, job_id: str, **fields) -> TraceRecord | None:
        if not self.enabled:
            return None
        status = fields.pop("status", "pending")
        record = TraceRecord(job_id=job_id, status=status, **fields)
        with self._lock:
            self._submitted += 1
            self._records.append(record)
            self._by_id[job_id] = record
            while len(self._records) > self.max_items:
                old = self._records.popleft()
                self._by_id.pop(old.job_id, None)
        return record

    def get(self, job_id: str) -> TraceRecord | None:
        with self._lock:
            return self._by_id.get(job_id)

    def set_current(self, record: TraceRecord | None):
        return _current_trace.set(record)

    def reset_current(self, token) -> None:
        _current_trace.reset(token)

    def mark_started(self, job_id: str, started_at: float | None = None) -> None:
        record = self.get(job_id)
        if record is None:
            return
        started_at = started_at or time.time()
        with self._lock:
            record.status = "doing"
            record.started_at = started_at
            record.stages_ms["queue_wait"] = max(0, int((started_at - record.created_at) * 1000))

    def mark_done(self, job_id: str, segment_count: int | None = None) -> None:
        record = self.get(job_id)
        if record is None:
            return
        with self._lock:
            record.status = "done"
            record.finished_at = time.time()
            if segment_count is not None:
                record.segment_count = segment_count
            self._done += 1

    def mark_error(self, job_id: str, error: BaseException | str) -> None:
        record = self.get(job_id)
        if record is None:
            return
        with self._lock:
            record.status = "error"
            record.finished_at = time.time()
            record.error = str(error)
            self._error += 1

    def add_stage_ms(self, name: str, elapsed_ms: int) -> None:
        record = _current_trace.get()
        if record is None:
            return
        with self._lock:
            record.stages_ms[name] = record.stages_ms.get(name, 0) + max(0, elapsed_ms)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled or _current_trace.get() is None:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_stage_ms(name, int((time.perf_counter() - start) * 1000))

    def list_records(self, status: str | None = None, limit: int | None = None) -> list[dict]:
        with self._lock:
            rows = list(self._records)
        if status:
            rows = [row for row in rows if row.status == status]
        if limit is not None:
            rows = rows[-max(0, limit) :]
        return [row.to_dict() for row in reversed(rows)]

    def summary(self, queue_snapshot: dict | None, model: dict | None = None) -> dict:
        with self._lock:
            rows = list(self._records)
            totals = {"submitted": self._submitted, "done": self._done, "error": self._error}
        done_rows = [row for row in rows if row.finished_at is not None]

        def values(name: str) -> list[int]:
            if name == "total":
                return [max(0, int((row.finished_at - row.created_at) * 1000)) for row in done_rows if row.finished_at is not None]
            return [row.stages_ms[name] for row in done_rows if name in row.stages_ms]

        def avg(items: list[int]) -> int | None:
            return int(sum(items) / len(items)) if items else None

        def p95(items: list[int]) -> int | None:
            if not items:
                return None
            ordered = sorted(items)
            index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
            return ordered[index]

        latency = {}
        for name in ["total", "queue_wait", "load_model", "decode_audio", "asmr_vad_segment", "faster_whisper_transcribe", "render_subtitle", "write_tmp_file", "rename_final_file"]:
            items = values(name)
            latency[f"{name}_avg"] = avg(items)
            latency[f"{name}_p95"] = p95(items)

        return {
            "trace_enabled": self.enabled,
            "trace_max_items": self.max_items,
            "queue": queue_snapshot,
            "totals": totals,
            "latency_ms": latency,
            "model": model,
        }


trace_store = TraceStore()


def current_trace() -> TraceRecord | None:
    return _current_trace.get()
