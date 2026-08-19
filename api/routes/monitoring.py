"""Fondasi monitoring - metrik untuk diamati, bukan alert atau retraining.

Dua kelompok metrik dikirim untuk tiap model:

- `offline`: hasil evaluasi SAAT TRAINING (PR-AUC, ROC-AUC, Precision/
  Recall@kapasitas, Brier) - dibaca apa adanya dari metadata.json, TIDAK
  dihitung ulang di sini.
- `live`: kondisi populasi PART aktif SEKARANG (sebaran skor, jumlah HIGH/
  MEDIUM, pangsa kategori tak dikenal, ringkasan fitur) - tidak ada label
  ground-truth untuk PART yang sedang aktif, jadi PR-AUC/ROC-AUC LIVE
  secara matematis tidak ada di sini.

Retraining otomatis SENGAJA tidak dibangun di atas endpoint ini - monitoring
harus terbukti stabil dulu sebelum keputusan otomatis dibangun di atasnya.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.services import monitoring_service

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
