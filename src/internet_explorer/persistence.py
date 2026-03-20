from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pymongo import MongoClient
from pymongo.collection import Collection

from internet_explorer.config import AppConfig
from internet_explorer.models import RunSummary, UrlEvaluation

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_TLS_CA_QUERY_KEYS = {"tlscafile", "ssl_ca_certs"}


def _sanitize_bson(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if value < _INT64_MIN or value > _INT64_MAX:
            return str(value)
        return value
    if isinstance(value, dict):
        return {str(key): _sanitize_bson(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_bson(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_bson(item) for item in value]
    if isinstance(value, set):
        return [_sanitize_bson(item) for item in value]
    return value


def _strip_tls_ca_file_from_uri(uri: str) -> str:
    raw_uri = (uri or "").strip()
    if not raw_uri:
        return ""

    parsed = urlsplit(raw_uri)
    if not parsed.query:
        return raw_uri

    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    filtered_query = [
        (key, value)
        for key, value in query_items
        if key.lower() not in _TLS_CA_QUERY_KEYS
    ]
    if len(filtered_query) == len(query_items):
        return raw_uri

    cleaned = parsed._replace(query=urlencode(filtered_query, doseq=True))
    return urlunsplit(cleaned)


def _mongo_client_settings(config: AppConfig) -> tuple[str, dict[str, Any]]:
    uri = config.mongodb_uri
    client_kwargs: dict[str, Any] = {}

    if config.mongodb_tls_ca_file is None:
        return uri, client_kwargs

    client_kwargs["tlsCAFile"] = str(config.mongodb_tls_ca_file)
    return _strip_tls_ca_file_from_uri(uri), client_kwargs


def ping_mongo(config: AppConfig, *, server_selection_timeout_ms: int = 15_000) -> dict[str, Any]:
    mongodb_uri, client_kwargs = _mongo_client_settings(config)
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=server_selection_timeout_ms, **client_kwargs)
    try:
        return client.admin.command("ping")
    finally:
        client.close()


class MongoPersistence:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        mongodb_uri, client_kwargs = _mongo_client_settings(config)
        self.client = MongoClient(mongodb_uri, **client_kwargs)
        # Force an immediate connection attempt so startup fails fast if Mongo is unreachable.
        self.client.admin.command("ping")
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
        self.runs.insert_one(_sanitize_bson(doc))

    def update_run(self, run_id: str, fields: dict[str, Any]) -> None:
        fields = dict(fields)
        fields["updated_at"] = datetime.utcnow()
        self.runs.update_one({"run_id": run_id}, {"$set": _sanitize_bson(fields)}, upsert=False)

    def log_event(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", datetime.utcnow())
        self.events.insert_one(_sanitize_bson(payload))

    def upsert_url_summary(self, run_id: str, evaluation: UrlEvaluation, extra: dict[str, Any] | None = None) -> None:
        doc = evaluation.model_dump(mode="json")
        doc["run_id"] = run_id
        doc["updated_at"] = datetime.utcnow()
        if extra:
            doc.update(extra)
        self.url_summaries.update_one(
            {"run_id": run_id, "url_id": evaluation.url_id},
            {"$set": _sanitize_bson(doc), "$setOnInsert": {"created_at": datetime.utcnow()}},
            upsert=True,
        )
