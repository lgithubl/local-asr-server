from __future__ import annotations

import threading
from pathlib import Path

from .config import Settings
from .subtitles import Segment


class Transcriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self.settings.backend == "mock" or self._model is not None

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

    def transcribe(self, audio_path: Path, language: str | None) -> list[Segment]:
        if self.settings.backend == "mock":
            chosen_language = language or self.settings.default_language
            return [Segment(start=0.0, end=1.5, text=f"mock transcription for {audio_path.name} ({chosen_language})")]

        with self._lock:
            model = self._load_model()
            raw_segments, _info = model.transcribe(
                str(audio_path),
                language=language or self.settings.default_language,
                beam_size=self.settings.beam_size,
                vad_filter=self.settings.vad_filter,
            )
            return [Segment(start=s.start, end=s.end, text=s.text) for s in raw_segments]
