from __future__ import annotations

import os
import re
from pathlib import Path


SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,190}$")


class PathValidationError(ValueError):
    pass


def resolve_input_path(input_dir: Path, requested_path: str) -> Path:
    if not requested_path:
        raise PathValidationError("input_path is required")
    path = Path(requested_path)
    if not path.is_absolute():
        path = input_dir / path
    resolved_input_dir = input_dir.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if os.path.commonpath([str(resolved_input_dir), str(resolved_path)]) != str(resolved_input_dir):
        raise PathValidationError("input_path must stay inside ASR_INPUT_DIR")
    if not resolved_path.is_file():
        raise PathValidationError("input_path must point to a file")
    return resolved_path


def validate_output_key(output_key: str) -> str:
    if not output_key:
        raise PathValidationError("uniq_key_name is required")
    if "/" in output_key or "\\" in output_key or ".." in output_key:
        raise PathValidationError("uniq_key_name must be a plain file name")
    if not SAFE_OUTPUT_NAME.fullmatch(output_key):
        raise PathValidationError("uniq_key_name contains unsupported characters")
    return output_key


def atomic_write(output_dir: Path, output_key: str, content: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / validate_output_key(output_key)
    tmp_path = output_dir / f"{output_key}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, final_path)
    dir_fd = os.open(output_dir, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return final_path
