from __future__ import annotations

import hashlib
import json
from itertools import count
from time import perf_counter
from typing import Any

from internet_explorer.persistence import MongoPersistence


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class Telemetry:
    def __init__(self, persistence: MongoPersistence, run_id: str, intent_id: str) -> None:
        self.persistence = persistence
        self.run_id = run_id
        self.intent_id = intent_id
        self._counter = count(1)

    def emit(
        self,
        *,
        phase: str,
        actor: str,
        output_summary: Any,
        decision: str = "",
        strategy_id: str | None = None,
        query_id: str | None = None,
        url_id: str | None = None,
        latency_ms: int | None = None,
        input_payload: Any | None = None,
        error_code: str | None = None,
        retry_no: int = 0,
        token_usage: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        step_no = next(self._counter)
        event = {
            "run_id": self.run_id,
            "intent_id": self.intent_id,
            "strategy_id": strategy_id,
            "query_id": query_id,
            "url_id": url_id,
            "phase": phase,
            "actor": actor,
            "step_no": step_no,
            "input_hash": _hash_payload(input_payload) if input_payload is not None else None,
            "output_summary": output_summary,
            "decision": decision,
            "latency_ms": latency_ms,
            "error_code": error_code,
            "retry_no": retry_no,
            "token_usage": token_usage or {},
        }
        if extra:
            event.update(extra)
        self.persistence.log_event(event)

    def timed(self) -> float:
        return perf_counter()

    def elapsed_ms(self, started_at: float) -> int:
        return int((perf_counter() - started_at) * 1000)

