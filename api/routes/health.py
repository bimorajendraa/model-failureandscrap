"""Cek kesiapan aplikasi."""

from __future__ import annotations

from fastapi import APIRouter

import data_reader
from api import db_pool
from api.errors import ModelUnavailable
from api.schemas import HealthResponse
from api.services import batch_service, model_registry

router = APIRouter(tags=["health"])

API_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
def health(check_database: bool = False) -> dict:
    """Status aplikasi.

    Pemeriksaan database TIDAK dijalankan secara default: /health sering
    dipanggil health checker setiap beberapa detik, dan satu query per
    panggilan hanya membebani database tanpa menambah informasi. Pakai
    ?check_database=true kalau memang ingin memastikan koneksinya hidup.
    """
    try:
        versions: dict[str, str | None] = dict(model_registry.versions())
        model_ok = True
    except ModelUnavailable:
        versions = {"failure": None, "scrap": None}
        model_ok = False

    database = "unchecked"
    if check_database:
        try:
            with data_reader.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            database = "reachable"
        except Exception:  # noqa: BLE001 - detailnya tidak boleh bocor ke client
            database = "unreachable"

    cached = batch_service.cached_scores()
    return {
        "status": "ok" if model_ok and database != "unreachable" else "degraded",
        "api_version": API_VERSION,
        "model_version": versions,
        "database": database,
        "connection_pool": db_pool.stats(),
        "batch_cache": {
            "ready": cached is not None,
            "rows": int(len(cached.frame)) if cached else 0,
            "computed_seconds_ago": int(cached.age_seconds) if cached else None,
            "data_through": str(cached.data_end) if cached else None,
        },
    }
