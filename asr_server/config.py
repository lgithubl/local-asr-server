from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("ASR_HOST", "0.0.0.0")
    port: int = int(os.getenv("ASR_PORT", "9000"))
    backend: str = os.getenv("ASR_BACKEND", "faster-whisper")
    model_path: str = os.getenv("ASR_MODEL_PATH", "/models")
    input_dir: Path = Path(os.getenv("ASR_INPUT_DIR", "/inputs"))
    output_dir: Path = Path(os.getenv("ASR_OUTPUT_DIR", "/outputs"))
    device: str = os.getenv("ASR_DEVICE", "cuda")
    compute_type: str = os.getenv("ASR_COMPUTE_TYPE", "float32")
    default_language: str = os.getenv("ASR_DEFAULT_LANGUAGE", "ja")
    default_format: str = os.getenv("ASR_DEFAULT_FORMAT", "srt")
    beam_size: int = int(os.getenv("ASR_BEAM_SIZE", "5"))
    vad_filter: bool = _bool_env("ASR_VAD_FILTER", False)
    preload_model: bool = _bool_env("ASR_PRELOAD_MODEL", True)
    idle_unload_seconds: int = int(os.getenv("ASR_IDLE_UNLOAD_SECONDS", "0"))
    max_queue_size: int = int(os.getenv("ASR_MAX_QUEUE_SIZE", "64"))
    trace_enabled: bool = _bool_env("ASR_TRACE_ENABLED", True)
    trace_max_items: int = int(os.getenv("ASR_TRACE_MAX_ITEMS", "500"))
    asmr_vad_model_path: str = os.getenv("ASR_ASMR_VAD_MODEL_PATH", "")
    asmr_vad_metadata_path: str = os.getenv("ASR_ASMR_VAD_METADATA_PATH", "")
    asmr_vad_feature_extractor_path: str = os.getenv("ASR_ASMR_VAD_FEATURE_EXTRACTOR_PATH", "")
    asmr_vad_force_cpu: bool = _bool_env("ASR_ASMR_VAD_FORCE_CPU", True)
    asmr_vad_threads: int = int(os.getenv("ASR_ASMR_VAD_THREADS", "1"))
    asmr_vad_threshold: float = float(os.getenv("ASR_ASMR_VAD_THRESHOLD", "0.5"))
    asmr_vad_min_speech_ms: int = int(os.getenv("ASR_ASMR_VAD_MIN_SPEECH_MS", "250"))
    asmr_vad_min_silence_ms: int = int(os.getenv("ASR_ASMR_VAD_MIN_SILENCE_MS", "100"))
    asmr_vad_speech_pad_ms: int = int(os.getenv("ASR_ASMR_VAD_SPEECH_PAD_MS", "300"))
    asmr_vad_max_speech_s: float = float(os.getenv("ASR_ASMR_VAD_MAX_SPEECH_S", "30"))


settings = Settings()
