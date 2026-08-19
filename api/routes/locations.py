"""Sebaran risiko menurut lokasi - bahan peta.

Koordinatnya dari OpenStreetMap Nominatim (lihat geocoding_service.py), bukan
dari database - database ini hanya punya NAMA lokasi, bukan GPS. Hasilnya
disaring ketat: lokasi yang tidak lolos disiplin geografis Jabodetabek TIDAK
ditampilkan sebagai pin, melainkan dilaporkan terpisah di `unresolved` supaya
petanya jujur tentang apa yang tidak diketahuinya.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from api import settings
from api.schemas import LocationMapResponse
from api.services import batch_service, geocoding_service

router = APIRouter(prefix="/api/v1", tags=["locations"])


@router.get("/locations/map", response_model=LocationMapResponse)
def locations_map(
    resolve: bool = Query(
        True,
        description=(
            "Coba geocode lokasi yang belum ada di cache. Matikan untuk "
            "jawaban instan dari cache saja."
        ),
    ),
    budget_seconds: float = Query(
        settings.GEOCODE_BUDGET_SECONDS_DEFAULT,
        ge=0,
        description="Anggaran waktu untuk geocoding lokasi baru pada panggilan ini.",
    ),
) -> dict:
    """Ringkasan risiko per lokasi, dipasangkan dengan koordinat kalau ada.

    Cache geocoding disk mengingat lokasi yang sudah pernah dicoba, jadi
    panggilan berikutnya untuk lokasi yang sama tidak memanggil jaringan lagi.
    Lokasi baru (belum pernah dicoba) di-geocode di sini, dibatasi
    `budget_seconds` supaya satu request tidak menggantung lama; sisanya baru
    diproses pada panggilan berikutnya.
    """
    scores = batch_service.score_active_parts()
    summary = batch_service.location_summary(scores.frame)
    locations = summary.index.tolist()

    if resolve and locations:
        capped = min(budget_seconds, settings.GEOCODE_BUDGET_SECONDS_MAX)
        geocoding_service.resolve_missing(locations, budget_seconds=capped)

    coordinates = geocoding_service.known_coordinates(locations)

    resolved, unresolved = [], []
    for location, row in summary.iterrows():
        counts = row.to_dict()
        entry = coordinates.get(location)
        if entry and entry.get("resolved"):
            resolved.append({"location": location, "lat": entry["lat"], "lon": entry["lon"], **counts})
        else:
            unresolved.append({"location": location, "checked": entry is not None, **counts})

    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "scored_at": scores.scored_at,
    }
