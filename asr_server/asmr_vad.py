from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings


@dataclass(frozen=True)
class SpeechWindow:
    start: float
    end: float


class AsmrVadSegmenter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._session = None
        self._feature_extractor = None
        self._input_name = ""
        self._output_names: list[str] = []
        self._frame_duration_ms = 20
        self._chunk_duration_ms = 30000
        self._sample_rate = 16000

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _resolve_model_path(self) -> Path:
        model_path = self.settings.asmr_vad_model_path
        if not model_path:
            raise ValueError("ASR_ASMR_VAD_MODEL_PATH is required when segmenter=asmr-onnx")
        path = Path(model_path)
        if path.is_dir():
            path = path / "model.onnx"
        if not path.is_file():
            raise ValueError(f"ASR ASMR VAD model not found: {path}")
        return path

    def _load(self) -> None:
        if self._session is not None:
            return

        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        model_path = self._resolve_model_path()
        metadata_path = Path(self.settings.asmr_vad_metadata_path) if self.settings.asmr_vad_metadata_path else model_path.with_name("model_metadata.json")
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        feature_extractor_path = self.settings.asmr_vad_feature_extractor_path or str(model_path.parent)
        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(feature_extractor_path)

        options = ort.SessionOptions()
        options.inter_op_num_threads = self.settings.asmr_vad_threads
        options.intra_op_num_threads = self.settings.asmr_vad_threads
        providers = ["CPUExecutionProvider"]
        if not self.settings.asmr_vad_force_cpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self._session = ort.InferenceSession(str(model_path), providers=providers, sess_options=options)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]
        self._frame_duration_ms = int(metadata.get("frame_duration_ms", 20))
        self._chunk_duration_ms = int(metadata.get("total_duration_ms", 30000))

    def segment(self, audio: np.ndarray) -> list[SpeechWindow]:
        with self._lock:
            self._load()
            probabilities = self._audio_forward(audio)

        return self._timestamps_from_probabilities(probabilities, len(audio) / self._sample_rate)

    def _audio_forward(self, audio: np.ndarray) -> np.ndarray:
        assert self._session is not None
        assert self._feature_extractor is not None
        chunk_samples = int(self._chunk_duration_ms * self._sample_rate / 1000)
        all_probabilities = []

        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")

            inputs = self._feature_extractor(chunk, sampling_rate=self._sample_rate, return_tensors="np")
            outputs = self._session.run(self._output_names, {self._input_name: inputs.input_features})
            logits = outputs[0][0]
            all_probabilities.append(1 / (1 + np.exp(-logits)))

        if not all_probabilities:
            return np.array([], dtype=np.float32)
        return np.concatenate(all_probabilities).astype(np.float32)

    def _timestamps_from_probabilities(self, probabilities: np.ndarray, audio_duration: float) -> list[SpeechWindow]:
        if probabilities.size == 0:
            return []

        threshold = self.settings.asmr_vad_threshold
        neg_threshold = max(threshold - 0.15, 0.01)
        min_speech_frames = max(1, int(self.settings.asmr_vad_min_speech_ms / self._frame_duration_ms))
        min_silence_frames = max(1, int(self.settings.asmr_vad_min_silence_ms / self._frame_duration_ms))
        pad_frames = max(0, int(self.settings.asmr_vad_speech_pad_ms / self._frame_duration_ms))
        max_speech_frames = math.inf
        if self.settings.asmr_vad_max_speech_s > 0:
            max_speech_frames = int(self.settings.asmr_vad_max_speech_s * 1000 / self._frame_duration_ms)

        triggered = False
        temp_end = 0
        speeches: list[dict[str, int]] = []
        current: dict[str, int] = {}

        for index, probability in enumerate(probabilities):
            if probability >= threshold and not triggered:
                triggered = True
                current = {"start": index}
                temp_end = 0
                continue

            if triggered and "start" in current and index - current["start"] > max_speech_frames:
                current["end"] = current["start"] + int(max_speech_frames)
                speeches.append(current)
                current = {}
                triggered = False
                temp_end = 0
                continue

            if probability < neg_threshold and triggered:
                if not temp_end:
                    temp_end = index
                if index - temp_end >= min_silence_frames:
                    current["end"] = temp_end
                    if current["end"] - current["start"] >= min_speech_frames:
                        speeches.append(current)
                    current = {}
                    triggered = False
                    temp_end = 0
            elif probability >= threshold and temp_end:
                temp_end = 0

        if triggered and "start" in current:
            current["end"] = len(probabilities)
            if current["end"] - current["start"] >= min_speech_frames:
                speeches.append(current)

        for index, speech in enumerate(speeches):
            if index == 0:
                speech["start"] = max(0, speech["start"] - pad_frames)
            else:
                speech["start"] = max(speeches[index - 1]["end"], speech["start"] - pad_frames)

            if index < len(speeches) - 1:
                speech["end"] = min(speeches[index + 1]["start"], speech["end"] + pad_frames)
            else:
                speech["end"] = min(len(probabilities), speech["end"] + pad_frames)

        windows = []
        for speech in speeches:
            start = max(0.0, speech["start"] * self._frame_duration_ms / 1000)
            end = min(audio_duration, speech["end"] * self._frame_duration_ms / 1000)
            if end > start:
                windows.append(SpeechWindow(start=start, end=end))
        return windows
