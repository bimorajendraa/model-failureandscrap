"""Daftar prioritas hasil batch scoring.

Semua endpoint di sini membaca satu hasil batch yang sama (lihat
inference/batch_predictor.py). Batch dihitung sekali lalu dipakai ulang
selama masih segar, jadi permintaan filter/paging tidak pernah memicu
skoring ulang seluruh armada.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Query

from api import settings
from api.schemas import (
    FiltersResponse,
    OverviewResponse,
    RecommendationListResponse,
)
from inference import batch_predictor

router = APIRouter(prefix="/api/v1", tags=["recommendations"])

# Kolom internal yang tidak perlu keluar ke client.
_INTERNAL_COLUMNS = ["tier_score"]


def _rows(frame: pd.DataFrame) -> list[dict]:
    """DataFrame -> list dict yang aman di-JSON-kan (NaN jadi null)."""
    clean = frame.drop(columns=_INTERNAL_COLUMNS, errors="ignore")
    return [
        {key: (None if pd.isna(value) else value) for key, value in record.items()}
        for record in clean.to_dict(orient="records")
    ]


@router.get("/recommendations", response_model=RecommendationListResponse)
def recommendations(
    risk: str | None = Query(None, description="Saring kelompok risiko kerusakan: LOW/MEDIUM/HIGH"),
    priority: str | None = Query(None, description="Saring prioritas: LOW/MEDIUM/HIGH/CRITICAL"),
    item_type: str | None = Query(None, description="Saring jenis PART, mis. MOTOR"),
    client: str | None = Query(None, description="Saring client"),
    location: str | None = Query(None, description="Saring lokasi terakhir tercatat"),
    search: str | None = Query(
        None, description="Cari sebagian ID PART, mis. 0112011 (tidak harus lengkap)"
    ),
    replacement_candidates_only: bool = Query(
        False,
        description=(
            "Hanya PART dengan risiko kerusakan MEDIUM/HIGH sekaligus risiko "
            "scrap HIGH - kandidat perencanaan penggantian."
        ),
    ),
    limit: int = Query(settings.DEFAULT_RECOMMENDATION_LIMIT, ge=1),
    offset: int = Query(0, ge=0),
) -> dict:
    """PART yang paling perlu diperhatikan, terurut dari yang paling berisiko."""
    limit = min(limit, settings.MAX_RECOMMENDATION_LIMIT)
    scores = batch_predictor.score_active_parts()
    selected = batch_predictor.filter_scores(
        scores.frame,
        risk=risk,
        priority=priority,
        item_type=item_type,
        client=client,
        location=location,
        search=search,
        replacement_candidates_only=replacement_candidates_only,
    )
    page = selected.iloc[offset : offset + limit]
    return {
        "total": int(len(selected)),
        "returned": int(len(page)),
        "offset": offset,
        "scored_at": scores.scored_at,
        "items": _rows(page),
    }


@router.get("/overview", response_model=OverviewResponse)
def overview(
    top: int = Query(10, ge=1, le=100, description="Berapa PART teratas yang ikut dikirim"),
) -> dict:
    """Angka ringkas seluruh armada + daftar teratas, untuk halaman overview."""
    scores = batch_predictor.score_active_parts()
    return {
        "summary": batch_predictor.summary(scores.frame),
        "scored_at": scores.scored_at,
        "top_priority": _rows(scores.frame.head(top)),
    }


@router.get("/filters", response_model=FiltersResponse)
def filters() -> dict:
    """Nilai filter yang benar-benar ada di data, untuk dropdown dashboard."""
    scores = batch_predictor.score_active_parts()
    return batch_predictor.facets(scores.frame)
