from __future__ import annotations

import os
import re
from pathlib import Path


SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,190}$")
SAFE_OUTPUT_DIR_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$")


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


def validate_output_subdir(output_subdir: str | None) -> str:
    if not output_subdir:
        return ""
    if output_subdir.startswith("/") or output_subdir.startswith("\\"):
        raise PathValidationError("output_dir must be a relative directory")
    if "\\" in output_subdir:
        raise PathValidationError("output_dir must use forward slashes")
    parts = [part for part in output_subdir.split("/") if part]
    if not parts:
        return ""
    if len(parts) > 8:
        raise PathValidationError("output_dir is too deep")
    for part in parts:
        if part in {".", ".."} or ".." in part:
            raise PathValidationError("output_dir contains unsupported path segments")
        if not SAFE_OUTPUT_DIR_PART.fullmatch(part):
            raise PathValidationError("output_dir contains unsupported characters")
    return "/".join(parts)


def resolve_output_path(output_dir: Path, output_subdir: str | None, output_key: str) -> Path:
    safe_subdir = validate_output_subdir(output_subdir)
    safe_key = validate_output_key(output_key)
    base = output_dir.resolve(strict=False)
    final_dir = base / safe_subdir if safe_subdir else base
    final_path = final_dir / safe_key
    resolved_final = final_path.resolve(strict=False)
    if os.path.commonpath([str(base), str(resolved_final)]) != str(base):
        raise PathValidationError("output path must stay inside ASR_OUTPUT_DIR")
    return final_path


def atomic_write(output_dir: Path, output_key: str, content: str, output_subdir: str | None = None) -> Path:
    from .tracing import trace_store

    final_path = resolve_output_path(output_dir, output_subdir, output_key)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.parent / f"{final_path.name}.tmp"
    with trace_store.stage("write_tmp_file"):
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
    with trace_store.stage("rename_final_file"):
        os.replace(tmp_path, final_path)
    dir_fd = os.open(final_path.parent, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return final_path
