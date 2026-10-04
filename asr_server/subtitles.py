from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable, Literal


OutputFormat = Literal["srt", "vtt", "json", "txt"]


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def format_timestamp(seconds: float, *, vtt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def render_srt(segments: Iterable[Segment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
            f"{segment.text.strip()}"
        )
    return "\n\n".join(blocks).strip() + "\n"


def render_vtt(segments: Iterable[Segment]) -> str:
    blocks = ["WEBVTT", ""]
    for segment in segments:
        blocks.append(
            f"{format_timestamp(segment.start, vtt=True)} --> {format_timestamp(segment.end, vtt=True)}\n"
            f"{segment.text.strip()}"
        )
        blocks.append("")
    return "\n".join(blocks).strip() + "\n"


def render_json(segments: Iterable[Segment]) -> str:
    return json.dumps({"segments": [asdict(segment) for segment in segments]}, ensure_ascii=False, indent=2) + "\n"


def render_txt(segments: Iterable[Segment]) -> str:
    return "\n".join(segment.text.strip() for segment in segments if segment.text.strip()) + "\n"


def render_subtitles(segments: list[Segment], output_format: str) -> str:
    if output_format == "srt":
        return render_srt(segments)
    if output_format == "vtt":
        return render_vtt(segments)
    if output_format == "json":
        return render_json(segments)
    if output_format == "txt":
        return render_txt(segments)
    raise ValueError("output_format must be one of: srt, vtt, json, txt")
