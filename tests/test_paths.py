from pathlib import Path

import pytest

from asr_server.paths import PathValidationError, atomic_write, resolve_input_path, resolve_output_path, validate_output_key, validate_output_subdir


def test_resolve_input_path_allows_files_inside_root(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")

    assert resolve_input_path(tmp_path, "sample.wav") == audio.resolve()
    assert resolve_input_path(tmp_path, str(audio)) == audio.resolve()


def test_resolve_input_path_blocks_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(b"RIFF")

    with pytest.raises(PathValidationError):
        resolve_input_path(tmp_path, str(outside))


def test_validate_output_key_accepts_plain_names() -> None:
    assert validate_output_key("job-001.ja.srt") == "job-001.ja.srt"


def test_validate_output_subdir_accepts_relative_dirs() -> None:
    assert validate_output_subdir("rj123/sub-01") == "rj123/sub-01"
    assert validate_output_subdir("") == ""
    assert validate_output_subdir(None) == ""


@pytest.mark.parametrize("name", ["../rj", "/rj", "rj/../x", "rj\\x", " rj", "rj?x"])
def test_validate_output_subdir_rejects_unsafe_dirs(name: str) -> None:
    with pytest.raises(PathValidationError):
        validate_output_subdir(name)


def test_resolve_output_path_allows_relative_subdir(tmp_path: Path) -> None:
    assert resolve_output_path(tmp_path, "rj123", "job.srt") == tmp_path.resolve() / "rj123" / "job.srt"


def test_atomic_write_creates_subdir_and_renames_tmp(tmp_path: Path) -> None:
    path = atomic_write(tmp_path, "job.srt", "hello", output_subdir="rj123")
    assert path == tmp_path.resolve() / "rj123" / "job.srt"
    assert path.read_text() == "hello"
    assert not (tmp_path / "rj123" / "job.srt.tmp").exists()


@pytest.mark.parametrize("name", ["../x.srt", "dir/x.srt", "", " x.srt", "x?.srt"])
def test_validate_output_key_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(PathValidationError):
        validate_output_key(name)
