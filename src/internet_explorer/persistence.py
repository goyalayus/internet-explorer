from __future__ import annotations

from datetime import datetime
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection

from internet_explorer.config import AppConfig
from internet_explorer.models import RunSummary, UrlEvaluation


class MongoPersistence:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = MongoClient(config.mongodb_uri)
        self.db = self.client[config.mongodb_db]
        self.runs: Collection = self.db[config.mongodb_runs_collection]
        self.url_summaries: Collection = self.db[config.mongodb_url_summaries_collection]
        self.events: Collection = self.db[config.mongodb_events_collection]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.runs.create_index("run_id", unique=True)
        self.url_summaries.create_index([("run_id", 1), ("url_id", 1)], unique=True)
        self.url_summaries.create_index([("run_id", 1), ("domain", 1)])
        self.events.create_index([("run_id", 1), ("step_no", 1)])
        self.events.create_index([("run_id", 1), ("phase", 1)])

    def create_run(self, run: RunSummary, metadata: dict[str, Any]) -> None:
        doc = run.model_dump()
        doc.update(
            {
                "metadata": metadata,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        )
        self.runs.insert_one(doc)

    def update_run(self, run_id: str, fields: dict[str, Any]) -> None:
        fields = dict(fields)
        fields["updated_at"] = datetime.utcnow()
        self.runs.update_one({"run_id": run_id}, {"$set": fields}, upsert=False)

    def log_event(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", datetime.utcnow())
        self.events.insert_one(payload)

    def upsert_url_summary(self, run_id: str, evaluation: UrlEvaluation, extra: dict[str, Any] | None = None) -> None:
        doc = evaluation.model_dump(mode="json")
        doc["run_id"] = run_id
        doc["updated_at"] = datetime.utcnow()
        if extra:
            doc.update(extra)
        self.url_summaries.update_one(
            {"run_id": run_id, "url_id": evaluation.url_id},
            {"$set": doc, "$setOnInsert": {"created_at": datetime.utcnow()}},
            upsert=True,
        )

