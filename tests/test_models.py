from datetime import datetime, timezone

from internet_explorer.models import RunSummary


def test_run_summary_persists_datetime_end_fields() -> None:
    started_at = datetime.now(timezone.utc)
    finished_at = datetime.now(timezone.utc)

    summary = RunSummary(run_id="run_1", intent="test intent", started_at=started_at)
    summary.finished_at = finished_at
    summary.completed_at = finished_at

    payload = summary.model_dump()
    assert payload["started_at"] == started_at
    assert payload["finished_at"] == finished_at
    assert payload["completed_at"] == finished_at
