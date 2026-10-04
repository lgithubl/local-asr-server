from __future__ import annotations

import threading
import gc
from pathlib import Path

import numpy as np

from .asmr_vad import AsmrVadSegmenter
from .config import Settings
from .subtitles import Segment


class Transcriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()
        self._asmr_vad = AsmrVadSegmenter(settings)

    @property
    def model_loaded(self) -> bool:
        return self.settings.backend == "mock" or self._model is not None

    def unload(self) -> None:
        with self._lock:
            self._model = None
            gc.collect()

    def preload(self) -> None:
        if self.settings.backend != "mock":
            self._load_model()

    def _load_model(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.settings.model_path,
            device=self.settings.device,
            compute_type=self.settings.compute_type,
        )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        vad_filter: bool | None = None,
        segmenter: str = "none",
    ) -> list[Segment]:
        if self.settings.backend == "mock":
            chosen_language = language or self.settings.default_language
            return [Segment(start=0.0, end=1.5, text=f"mock transcription for {audio_path.name} ({chosen_language})")]

        if segmenter == "asmr-onnx":
            return self._transcribe_with_asmr_vad(audio_path, language, vad_filter)

        with self._lock:
            model = self._load_model()
            raw_segments, _info = model.transcribe(
                str(audio_path),
                language=language or self.settings.default_language,
                beam_size=self.settings.beam_size,
                vad_filter=self.settings.vad_filter if vad_filter is None else vad_filter,
            )
            return [Segment(start=s.start, end=s.end, text=s.text) for s in raw_segments]

    def _transcribe_with_asmr_vad(self, audio_path: Path, language: str | None, vad_filter: bool | None) -> list[Segment]:
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(audio_path), sampling_rate=16000)
        windows = self._asmr_vad.segment(audio)
        if not windows:
            return []

        segments: list[Segment] = []
        with self._lock:
            model = self._load_model()
            for window in windows:
                start_sample = max(0, int(window.start * 16000))
                end_sample = min(len(audio), int(window.end * 16000))
                chunk = np.ascontiguousarray(audio[start_sample:end_sample])
                if chunk.size == 0:
                    continue
                raw_segments, _info = model.transcribe(
                    chunk,
                    language=language or self.settings.default_language,
                    beam_size=self.settings.beam_size,
                    vad_filter=self.settings.vad_filter if vad_filter is None else vad_filter,
                )
                for segment in raw_segments:
                    segments.append(
                        Segment(
                            start=segment.start + window.start,
                            end=segment.end + window.start,
                            text=segment.text,
                        )
                    )
        return segments
