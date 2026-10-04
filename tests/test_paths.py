from pathlib import Path

import pytest

from asr_server.paths import PathValidationError, resolve_input_path, validate_output_key


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


@pytest.mark.parametrize("name", ["../x.srt", "dir/x.srt", "", " x.srt", "x?.srt"])
def test_validate_output_key_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(PathValidationError):
        validate_output_key(name)
