"""Prediksi untuk satu PART.

Route sengaja tipis: seluruh keputusan ada di api/services/prediction_service.py.
Di sini hanya penerjemahan hasil service menjadi bentuk HTTP.

PART yang ada tetapi tidak bisa diskor TIDAK dianggap error - jawabannya 200
dengan status NOT_SCORABLE dan alasannya, karena "PART ini sedang tidak
terpasang" adalah jawaban yang sah, bukan kegagalan sistem.
"""

from __future__ import annotations

from fastapi import APIRouter, Path, Query

from api.errors import PartNotScorable
from api.schemas import AssessmentResponse, FailureResponse, HistoryResponse, ScrapResponse
from api.services import prediction_service

router = APIRouter(prefix="/api/v1/parts", tags=["parts"])

_ITEM_ID = Path(
    description="ID PART, mis. 011201100101164. Fitur ML dibangun otomatis.",
    min_length=1,
    max_length=100,
)


def _not_scorable(error: PartNotScorable) -> dict:
    return {
        "item_id": error.item_id,
        "status": "NOT_SCORABLE",
        "reason": error.reason,
    }


@router.get("/{item_id}/failure", response_model=FailureResponse)
def failure(item_id: str = _ITEM_ID) -> dict:
    """Peluang PART rusak dalam 30/60/90/120 hari ke depan.

    Ini PELUANG, bukan perkiraan tanggal kerusakan.
    """
    try:
        prediction = prediction_service.predict_failure(item_id)
    except PartNotScorable as error:
        return _not_scorable(error)
    return {"item_id": prediction["item_id"], "status": "SCORED", "failure": prediction}


@router.get("/{item_id}/scrap", response_model=ScrapResponse)
def scrap(item_id: str = _ITEM_ID) -> dict:
    """Kalau PART ini rusak, peluang tidak bisa diperbaiki.

    BERSYARAT terhadap kerusakan - bukan peluang PART ini rusak.
    """
    try:
        prediction = prediction_service.predict_scrap(item_id)
    except PartNotScorable as error:
        return _not_scorable(error)
    return {"item_id": prediction["item_id"], "status": "SCORED", "scrap": prediction}


@router.get("/{item_id}/history", response_model=HistoryResponse)
def history(item_id: str = _ITEM_ID) -> dict:
    """Tanggal kerusakan dan lokasi yang pernah tercatat untuk satu PART.

    Dari catatan event apa adanya, bukan dihitung ulang - mendukung faktor
    risiko di /assessment yang berupa hitungan (mis. "2 kerusakan dalam 365
    hari terakhir") dengan tanggal sesungguhnya.
    """
    return prediction_service.item_history(item_id)


@router.get("/{item_id}/assessment", response_model=AssessmentResponse)
def assessment(
    item_id: str = _ITEM_ID,
    explain: bool = Query(
        True,
        description=(
            "Sertakan faktor risiko. Perlu satu putaran pembacaan riwayat "
            "tambahan, matikan kalau hanya butuh angkanya."
        ),
    ),
) -> dict:
    """Gabungan risiko kerusakan + risiko scrap + rekomendasi tindakan."""
    try:
        return prediction_service.get_part_assessment(item_id, include_explanation=explain)
    except PartNotScorable as error:
        return _not_scorable(error)
