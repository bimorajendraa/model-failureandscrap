"""Bentuk request dan response API.

Nama field sengaja mengikuti apa yang benar-benar dikeluarkan model
(failure_probability_30d, scrap_probability, ...) - tidak ada field yang
dikarang dan tidak ada yang diganti namanya, supaya jawaban API bisa
dicocokkan langsung dengan keluaran predict.py / predict_scrap.py.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# `model_` adalah prefix yang dilindungi pydantic; kolom kita memang bernama
# model_version, jadi perlindungan itu dimatikan di seluruh skema.
_CONFIG = ConfigDict(protected_namespaces=(), extra="allow")

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
Priority = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ScoringStatus = Literal["SCORED", "NOT_SCORABLE"]


class HealthResponse(BaseModel):
    model_config = _CONFIG

    status: Literal["ok", "degraded"]
    api_version: str
    model_version: dict[str, str | None]
    database: Literal["reachable", "unreachable", "unchecked"]
    connection_pool: dict
    batch_cache: dict


class FailurePrediction(BaseModel):
    """Keluaran predict.predict() apa adanya.

    Angkanya adalah PELUANG kerusakan dalam N hari ke depan. Model tidak
    memperkirakan tanggal kerusakan.
    """

    model_config = _CONFIG

    item_id: str
    failure_probability_30d: float = Field(ge=0.0, le=1.0)
    failure_probability_60d: float = Field(ge=0.0, le=1.0)
    failure_probability_90d: float = Field(ge=0.0, le=1.0)
    failure_probability_120d: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    model_version: str
    as_of: str


class ScrapPrediction(BaseModel):
    """Keluaran predict_scrap.predict_scrap() apa adanya.

    BERSYARAT: peluang PART tidak bisa diperbaiki JIKA rusak - bukan peluang
    PART ini rusak.
    """

    model_config = _CONFIG

    item_id: str
    scrap_probability: float = Field(ge=0.0, le=1.0)
    scrap_risk_level: RiskLevel
    scrap_risk_basis: str
    item_type: str | None = None
    item_type_known_to_model: bool
    model_version: str
    as_of: str


class Recommendation(BaseModel):
    model_config = _CONFIG

    priority: Priority
    action: str
    message: str
    based_on: dict


class RiskFactor(BaseModel):
    model_config = _CONFIG

    code: str
    direction: Literal["RISK_FACTOR", "MITIGATING", "CONTEXT"]
    label: str
    value: float | int | None = None


class Explanation(BaseModel):
    model_config = _CONFIG

    disclaimer: str
    factors: list[RiskFactor]
    # Keterangan cara membaca angkanya - mis. bahwa kerusakan di riwayat tidak
    # berarti PART berhenti dipakai.
    notes: list[str] = []
    caveats: list[str] = []


class FailureResponse(BaseModel):
    model_config = _CONFIG

    item_id: str
    status: ScoringStatus
    reason: str | None = None
    failure: FailurePrediction | None = None


class ScrapResponse(BaseModel):
    model_config = _CONFIG

    item_id: str
    status: ScoringStatus
    reason: str | None = None
    scrap: ScrapPrediction | None = None


class AssessmentResponse(BaseModel):
    model_config = _CONFIG

    item_id: str
    status: ScoringStatus
    reason: str | None = None
    as_of: str | None = None
    failure: FailurePrediction | None = None
    # Boleh kosong: PART yang riwayatnya belum cukup tetap dapat penilaian
    # kerusakan, hanya tanpa sumbu scrap.
    scrap: ScrapPrediction | None = None
    death_probability_30d: float | None = None
    recommendation: Recommendation | None = None
    replacement_candidate: bool | None = None
    explanation: Explanation | None = None
    model_version: dict[str, str | None] | None = None


class PriorityItem(BaseModel):
    """Satu baris daftar prioritas hasil batch scoring."""

    model_config = _CONFIG

    rank: int
    item_id: str
    item_type: str | None = None
    item_model_code: str | None = None
    client: str | None = None
    location: str | None = None
    installation_age_days: float | None = None
    failure_probability_30d: float
    failure_probability_60d: float
    failure_probability_90d: float
    failure_probability_120d: float
    failure_risk_level: RiskLevel
    scrap_probability: float | None = None
    scrap_risk_level: RiskLevel | None = None
    death_probability_30d: float | None = None
    priority: Priority
    recommended_action: str
    recommendation_message: str
    replacement_candidate: bool


class ScoredAt(BaseModel):
    """Kapan daftar ini dihitung dan sampai kapan datanya."""

    model_config = _CONFIG

    data_through: str
    computed_seconds_ago: int
    model_version: dict[str, str]


class RecommendationListResponse(BaseModel):
    model_config = _CONFIG

    total: int
    returned: int
    offset: int
    scored_at: ScoredAt
    items: list[PriorityItem]


class OverviewResponse(BaseModel):
    model_config = _CONFIG

    summary: dict
    scored_at: ScoredAt
    top_priority: list[PriorityItem]


class FiltersResponse(BaseModel):
    model_config = _CONFIG

    risk_levels: list[str]
    priorities: list[str]
    item_types: list[str]
    clients: list[str]
    locations: list[str]


class FailureHistoryItem(BaseModel):
    model_config = _CONFIG

    date: str
    location: str | None = None
    status: str


class LocationHistoryItem(BaseModel):
    model_config = _CONFIG

    location: str
    first_seen: str
    last_seen: str
    events: int


class HistoryResponse(BaseModel):
    model_config = _CONFIG

    item_id: str
    failures: list[FailureHistoryItem]
    locations: list[LocationHistoryItem]


class ResolvedLocation(BaseModel):
    model_config = _CONFIG

    location: str
    lat: float
    lon: float
    active_parts: int
    high_risk_parts: int
    medium_risk_parts: int
    replacement_candidates: int


class UnresolvedLocation(BaseModel):
    model_config = _CONFIG

    location: str
    active_parts: int
    high_risk_parts: int
    medium_risk_parts: int
    replacement_candidates: int
    # False = belum pernah dicoba sama sekali (kehabisan anggaran waktu).
    # True = sudah dicoba, tidak ada hasil yang lolos penyaringan Jabodetabek.
    checked: bool


class LocationMapResponse(BaseModel):
    model_config = _CONFIG

    resolved: list[ResolvedLocation]
    unresolved: list[UnresolvedLocation]
    scored_at: ScoredAt


class ErrorResponse(BaseModel):
    model_config = _CONFIG

    status: str
    message: str
    item_id: str | None = None
