from asr_server.tracing import TraceStore


def test_trace_store_records_stage_and_summary() -> None:
    store = TraceStore(enabled=True, max_items=10)
    record = store.create("job-1", input_path="/inputs/a.wav", output_key="a.srt")
    token = store.set_current(record)
    try:
        with store.stage("render_subtitle"):
            pass
    finally:
        store.reset_current(token)
    store.mark_started("job-1")
    store.mark_done("job-1", segment_count=2)

    detail = store.get("job-1").to_dict()
    assert detail["status"] == "done"
    assert detail["segment_count"] == 2
    assert "render_subtitle" in detail["stages"]

    summary = store.summary(queue_snapshot={"pending_count": 0, "doing_count": 0})
    assert summary["totals"]["submitted"] == 1
    assert summary["totals"]["done"] == 1
