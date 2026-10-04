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


settings = Settings()
