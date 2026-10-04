from asr_server.subtitles import Segment, render_subtitles


def test_render_srt() -> None:
    assert render_subtitles([Segment(0, 1.25, "hello")], "srt") == "1\n00:00:00,000 --> 00:00:01,250\nhello\n"


def test_render_vtt() -> None:
    assert render_subtitles([Segment(0, 1.25, "hello")], "vtt") == "WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nhello\n"
