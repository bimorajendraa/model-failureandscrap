"""Keterangan model production yang sedang dipakai."""

from __future__ import annotations

from fastapi import APIRouter

from api.services import model_registry

router = APIRouter(prefix="/api/v1", tags=["model"])


@router.get("/model")
def model_info() -> dict:
    """Versi, target, fitur, ambang risiko, dan metrik uji kedua model.

    Seluruhnya dibaca dari metadata.json yang ditulis train.py /
    train_scrap.py - tidak ada angka yang dihitung ulang di sini.
    """
    return model_registry.describe()
