"""Skoring SELURUH PART aktif sekaligus.

Dashboard tidak bertanya "berapa risiko PART X", melainkan "PART mana yang
paling perlu diperhatikan". Menjawabnya dengan memanggil predict() belasan
ribu kali berarti belasan ribu kali query database dan belasan ribu kali
membangun potret armada - tidak masuk akal.

Di sini seluruh data dibaca sekali, fitur dibangun sekali sebagai DataFrame,
lalu model dijalankan pada semua baris sekaligus.

FITUR DAN MATEMATIKANYA SAMA PERSIS dengan predict.py / predict_scrap.py:

- fitur kerusakan dibangun feature_builder.project_features() yang sama;
- perantaian hazard memakai urutan langkah yang sama;
- kelompok risiko memakai fungsi _risk_level() milik masing-masing model;
- fitur scrap dibangun scrap_features.build_features() yang sama.

Satu-satunya yang ditulis ulang di sini adalah penyusunan kolom mentah scrap
untuk banyak PART sekaligus (scrap_features.current_state() hanya melayani
satu PART per panggilan). Kesamaannya dijaga tests/test_parity.py, yang
membandingkan hasil batch dengan hasil single untuk PART yang sama.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psycopg

import config
import data_reader
import feature_builder
import predict as failure_model
import predict_scrap as scrap_model
import scrap_features
from inference import data_state, explanation, recommendation, settings
from inference.errors import DataSourceUnavailable

_HORIZONS = config.PREDICTION_HORIZON_DAYS


@dataclass
class BatchScores:
    """Hasil satu kali skoring seluruh PART aktif."""

    frame: pd.DataFrame
    # Nilai fitur mentah per PART, dipakai halaman detail untuk menjelaskan
    # faktor risiko tanpa perlu membaca ulang database.
    snapshot: pd.DataFrame
    data_end: pd.Timestamp
    model_version: dict
    # Penanda versi data saat hasil ini dihitung.
    generation: int = 0
    computed_at: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.computed_at

    @property
    def scored_at(self) -> dict:
        """Bentuk ScoredAt (lihat api/schemas.py) - dipakai semua route yang
        menyertakan hasil batch, supaya bentuknya tidak diketik ulang."""
        return {
            "data_through": str(self.data_end),
            "computed_seconds_ago": int(self.age_seconds),
            "model_version": self.model_version,
        }

    def is_stale(self, generation: int) -> bool:
        """Basi kalau umurnya lewat, ATAU kalau database sudah bertambah.

        Dua-duanya perlu: batas umur menjaga hasil tidak dipakai selamanya,
        sedangkan penanda versi data membuat data baru langsung terlihat tanpa
        menunggu batas umur habis.
        """
        return (
            self.age_seconds > settings.BATCH_CACHE_TTL_SECONDS
            or generation != self.generation
        )


_CACHE: BatchScores | None = None
_LOCK = threading.Lock()


def score_active_parts(force_refresh: bool = False) -> BatchScores:
    """Skor seluruh PART aktif; hasilnya dipakai ulang selama masih segar.

    Dikunci supaya beberapa request yang datang bersamaan tidak masing-masing
    menghitung batch yang sama.
    """
    global _CACHE
    # Di luar kunci: pemeriksaan ini juga yang membuang potret armada basi,
    # dan harus terjadi sebelum model dipakai.
    generation = data_state.generation() if _CACHE is None else _fresh_generation()
    with _LOCK:
        if force_refresh or _CACHE is None or _CACHE.is_stale(generation):
            _CACHE = _compute(generation)
        return _CACHE


def _fresh_generation() -> int:
    data_state.current_data_end()
    return data_state.generation()


def cached_scores() -> BatchScores | None:
    """Hasil batch yang tersimpan, tanpa memicu perhitungan baru."""
    return _CACHE


def _compute(generation: int) -> BatchScores:
    data_state.current_data_end()
    try:
        cycles = data_reader.get_cycles()
        events = data_reader.get_events()
    except psycopg.Error as error:
        raise DataSourceUnavailable(
            f"Database tidak bisa dibaca ({type(error).__name__})."
        ) from error

    data_end = pd.Timestamp(cycles["dataset_max_event_on"].max())

    failure, snapshot = _score_failure(cycles, events, data_end)
    scrap = _score_scrap(events, cycles, data_end, failure["item_id"])

    frame = failure.merge(scrap, on="item_id", how="left")
    frame = _attach_context(frame, events)
    frame = _attach_recommendation(frame)
    frame = frame.sort_values("tier_score", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))

    return BatchScores(
        frame=frame,
        snapshot=snapshot,
        data_end=data_end,
        generation=generation,
        model_version={
            "failure": failure_model.load_model()[2]["model_version"],
            "scrap": scrap_model.load_model()[2]["model_version"],
        },
    )


# ---------------------------------------------------------------------------
# Risiko kerusakan
# ---------------------------------------------------------------------------


def _score_failure(
    cycles: pd.DataFrame, events: pd.DataFrame, data_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perantaian hazard yang sama seperti predict(), untuk semua PART aktif.

    Urutan panggilannya sengaja dibuat identik dengan predict.predict():
    current_observations -> attach_history -> attach_fleet_snapshot ->
    part_model_support -> project_features per langkah 30 hari.

    Mengembalikan skor DAN nilai fitur mentahnya. Yang kedua dipakai halaman
    detail untuk menjelaskan faktor risiko - fitur itu sudah dihitung di sini,
    jadi menyimpannya jauh lebih murah daripada membaca ulang database untuk
    satu PART.
    """
    model, calibrator, metadata = failure_model.load_model()

    snapshot = feature_builder.current_observations(cycles)
    snapshot = feature_builder.attach_history(snapshot, events)
    snapshot = feature_builder.attach_fleet_snapshot(
        snapshot, failure_model.fleet_snapshot(data_end)
    )
    support = feature_builder.part_model_support(
        snapshot, metadata["part_model_support"]
    )

    steps = max(_HORIZONS) // config.OBSERVATION_STEP_DAYS
    survival = np.ones(len(snapshot), dtype=float)
    # Skor mentah langkah pertama - dipakai mengurutkan daftar (kolom "rank"),
    # bukan menentukan kelompok risiko (itu urusan failure_probability_30d di
    # bawah). Resolusinya jauh lebih halus daripada probabilitas terkalibrasi,
    # jadi urutan PART yang skornya berdekatan tetap bisa dibedakan.
    tier_score = np.zeros(len(snapshot), dtype=float)
    cumulative: dict[int, np.ndarray] = {}
    for step in range(steps):
        features = feature_builder.project_features(snapshot, support, step)
        raw = model.predict_proba(features)[:, 1]
        if step == 0:
            tier_score = raw
        hazard = calibrator.predict(raw)
        survival = survival * (1.0 - hazard)
        cumulative[(step + 1) * config.OBSERVATION_STEP_DAYS] = 1.0 - survival

    cutoffs = metadata["risk_cutoffs"]
    result = pd.DataFrame({
        "item_id": snapshot["item_identifier_clean"].to_numpy(),
        "item_model_code": snapshot["item_model_code_clean"].to_numpy(),
        "client": snapshot["installed_client_clean"].to_numpy(),
        "installation_age_days": snapshot["days_since_installation"].round(1).to_numpy(),
        "tier_score": tier_score,
    })
    for days in _HORIZONS:
        result[f"failure_probability_{days}d"] = np.round(cumulative[days], 4)
    # Fungsi milik model sendiri, bukan salinan aturannya. Sama seperti
    # predict(): kelompok risiko dari probabilitas 30-hari terkalibrasi.
    result["failure_risk_level"] = [
        failure_model.risk_level(score, cutoffs)
        for score in result["failure_probability_30d"]
    ]

    features_by_item = snapshot[explanation.SOURCE_COLUMNS].copy()
    features_by_item.index = pd.Index(
        snapshot["item_identifier_clean"].to_numpy(), name="item_id"
    )
    return result, features_by_item


# ---------------------------------------------------------------------------
# Risiko scrap
# ---------------------------------------------------------------------------


def _score_scrap(
    events: pd.DataFrame,
    cycles: pd.DataFrame,
    data_end: pd.Timestamp,
    items: pd.Series,
) -> pd.DataFrame:
    model, calibrator, metadata = scrap_model.load_model()

    state = _scrap_states(events, cycles, data_end, items)
    if state.empty:
        return pd.DataFrame(columns=[
            "item_id", "item_type", "scrap_probability", "scrap_risk_level",
            "item_type_known_to_model",
        ])

    features = scrap_features.build_features(state, metadata["known_item_types"])
    raw = model.predict_proba(features)[:, 1]
    probability = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    cutoffs = metadata["risk_cutoffs"]
    return pd.DataFrame({
        "item_id": state["item_identifier_clean"].to_numpy(),
        "item_type": state["item_type_clean"].to_numpy(),
        "scrap_probability": np.round(probability, 4),
        "scrap_risk_level": [
            scrap_model.risk_level(value, cutoffs) for value in probability
        ],
        "item_type_known_to_model": state["item_type_clean"]
        .isin(metadata["known_item_types"])
        .to_numpy(),
    })


def _scrap_states(
    events: pd.DataFrame,
    cycles: pd.DataFrame,
    data_end: pd.Timestamp,
    items: pd.Series,
) -> pd.DataFrame:
    """Kondisi "seandainya rusak sekarang" untuk banyak PART sekaligus.

    Menghasilkan kolom yang sama persis dengan scrap_features.current_state(),
    hanya dihitung sekali lewat groupby alih-alih satu PART per panggilan.
    Setiap kolom di bawah adalah terjemahan langsung dari baris di fungsi itu;
    tests/test_parity.py membandingkan keduanya baris per baris.
    """
    moment = pd.Timestamp(data_end)
    wanted = pd.Index(pd.unique(pd.Series(items)))

    seen = events.loc[events["item_identifier_clean"].isin(wanted)].copy()
    seen["created_on"] = pd.to_datetime(seen["created_on"])
    seen = seen.loc[seen["created_on"] <= moment]
    if seen.empty:
        return pd.DataFrame()

    status = seen["status_clean"].fillna("")
    seen["_is_repaired"] = status.eq(config.REPAIR_COMPLETED_STATUS)
    seen["_is_failure"] = seen["is_failure_onset"].fillna(False).astype(bool)

    # Event sudah terurut (item, created_on, journey_id) dari data_reader, dan
    # GroupBy.last() melewati nilai kosong - sama dengan dropna().iloc[-1].
    grouped = seen.groupby("item_identifier_clean", sort=False)
    state = pd.DataFrame({
        "item_type_clean": grouped["item_type_clean"].last(),
        "first_seen_on": grouped["created_on"].min(),
        "prior_repaired_count": grouped["_is_repaired"].sum().astype("int64"),
        "prior_failure_count": grouped["_is_failure"].sum().astype("int64"),
    })

    installs = cycles.loc[cycles["item_identifier_clean"].isin(wanted)].copy()
    installs["installed_on"] = pd.to_datetime(installs["installed_on"])
    installs = installs.loc[installs["installed_on"] <= moment]
    last_install = installs.groupby("item_identifier_clean")["installed_on"].max()

    state = state.reindex(wanted.intersection(state.index))
    state["failure_onset_on"] = moment
    state["age_total_days"] = (
        moment - state["first_seen_on"]
    ).dt.total_seconds() / 86400.0
    state["cycle_age_days"] = (
        moment - pd.DatetimeIndex(state.index.map(last_install))
    ).total_seconds() / 86400.0
    return state.reset_index(names="item_identifier_clean")


# ---------------------------------------------------------------------------
# Konteks dan rekomendasi
# ---------------------------------------------------------------------------


def _attach_context(frame: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Lokasi terakhir yang tercatat, dipakai sebagai filter dashboard."""
    known = events.loc[events["place_canonical_clean"].notna()]
    location = known.groupby("item_identifier_clean")["place_canonical_clean"].last()
    frame["location"] = frame["item_id"].map(location)
    return frame


def _attach_recommendation(frame: pd.DataFrame) -> pd.DataFrame:
    """Terjemahkan kelompok risiko jadi tindakan, memakai engine yang sama
    dengan endpoint satu PART."""
    horizon = config.TARGET_HORIZON_DAYS
    scrap_levels = [
        None if pd.isna(level) else level for level in frame["scrap_risk_level"]
    ]
    decisions = [
        recommendation.recommend(failure_level, scrap_level)
        for failure_level, scrap_level in zip(frame["failure_risk_level"], scrap_levels)
    ]
    frame["priority"] = [decision["priority"] for decision in decisions]
    frame["recommended_action"] = [decision["action"] for decision in decisions]
    frame["recommendation_message"] = [decision["message"] for decision in decisions]
    frame["replacement_candidate"] = [
        recommendation.is_replacement_candidate(failure_level, scrap_level)
        for failure_level, scrap_level in zip(frame["failure_risk_level"], scrap_levels)
    ]
    # Rumus sama dengan predict_scrap.predict_death_risk().
    frame[f"death_probability_{horizon}d"] = (
        frame[f"failure_probability_{horizon}d"] * frame["scrap_probability"]
    ).round(5)
    return frame


# ---------------------------------------------------------------------------
# Query di atas hasil batch
# ---------------------------------------------------------------------------


def filter_scores(
    frame: pd.DataFrame,
    risk: str | None = None,
    priority: str | None = None,
    item_type: str | None = None,
    client: str | None = None,
    location: str | None = None,
    search: str | None = None,
    replacement_candidates_only: bool = False,
) -> pd.DataFrame:
    """Penyaringan sederhana di atas DataFrame hasil batch."""
    result = frame
    if search:
        # Cocok sebagian, bukan persis: ID PART panjang dan orang biasanya
        # hanya ingat sebagian atau menyalinnya dari lembar kerja.
        result = result[
            result["item_id"].str.contains(search.strip().upper(), regex=False, na=False)
        ]
    if risk:
        result = result[result["failure_risk_level"].eq(risk.upper())]
    if priority:
        result = result[result["priority"].eq(priority.upper())]
    if item_type:
        result = result[result["item_type"].fillna("").str.upper().eq(item_type.upper())]
    if client:
        result = result[result["client"].fillna("").str.upper().eq(client.upper())]
    if location:
        result = result[result["location"].fillna("").str.upper().eq(location.upper())]
    if replacement_candidates_only:
        result = result[result["replacement_candidate"]]
    return result


def summary(frame: pd.DataFrame) -> dict:
    """Angka ringkas untuk halaman overview."""
    levels = frame["failure_risk_level"].value_counts()
    return {
        "active_parts": int(len(frame)),
        "high_risk_parts": int(levels.get("HIGH", 0)),
        "medium_risk_parts": int(levels.get("MEDIUM", 0)),
        "low_risk_parts": int(levels.get("LOW", 0)),
        "replacement_candidates": int(frame["replacement_candidate"].sum()),
        "priority_counts": {
            str(name): int(count)
            for name, count in frame["priority"].value_counts().items()
        },
    }


def location_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan risiko per lokasi terakhir tercatat - bahan peta.

    Hanya agregasi di atas frame yang sudah dihitung; tidak ada query
    tambahan, tidak ada angka baru yang dikarang.
    """
    known = frame.loc[frame["location"].notna()]
    grouped = known.groupby("location").agg(
        active_parts=("item_id", "count"),
        high_risk_parts=("failure_risk_level", lambda s: int((s == "HIGH").sum())),
        medium_risk_parts=("failure_risk_level", lambda s: int((s == "MEDIUM").sum())),
        replacement_candidates=("replacement_candidate", "sum"),
    )
    grouped["replacement_candidates"] = grouped["replacement_candidates"].astype(int)
    return grouped.sort_values("high_risk_parts", ascending=False)


def facets(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Nilai filter yang benar-benar ada di data, untuk dropdown dashboard."""

    def values(column: str) -> list[str]:
        return sorted(frame[column].dropna().astype(str).unique().tolist())

    return {
        "risk_levels": ["HIGH", "MEDIUM", "LOW"],
        "priorities": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        "item_types": values("item_type"),
        "clients": values("client"),
        "locations": values("location"),
    }
