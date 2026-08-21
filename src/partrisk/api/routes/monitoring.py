"""Endpoint monitoring - metrik offline (training) dan live (populasi aktif)
untuk tiap model. Lihat partrisk/api/services/monitoring_service.py untuk
definisi lengkap keduanya dan kenapa dipisah tegas."""

from __future__ import annotations

from fastapi import APIRouter

from partrisk.api.services import monitoring_service

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@router.get("/metrics")
def metrics() -> dict:
    """Snapshot metrik monitoring untuk kedua model."""
    return monitoring_service.summary()


@router.get("/metrics/failure")
def failure_metrics() -> dict:
    return monitoring_service.failure_monitoring()


@router.get("/metrics/scrap")
def scrap_metrics() -> dict:
    return monitoring_service.scrap_monitoring()
