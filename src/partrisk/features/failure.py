"""Feature engineering: kolom mentah -> 18 fitur final model failure.

SATU-SATUNYA tempat fitur dihitung. Training dan prediction sama-sama memanggil
`build_features`, jadi tidak mungkin ada perbedaan antara fitur yang dipelajari
model dan fitur yang dipakai saat production.

Semua hitungan count/durasi memakai LN(1+x): distribusinya sangat right-skewed,
dan tanpa transformasi ini beberapa outlier akan mendominasi.

Sebelumnya satu file `feature_builder.py` (Fase B2 restrukturisasi memecahnya
jadi `observations.py`/`history.py`/`fleet.py`/`support.py`/`transforms.py` -
logic TIDAK diubah, cuma dikelompokkan ulang per subjek). Modul INI
mengimpor-ulang SEMUA nama yang dulu diekspor `feature_builder` (termasuk
`_HISTORY_COUNTS`/`_count_before`/`_log1p` yang dipakai beberapa pemanggil di
survival_model/ walau namanya berawalan underscore) supaya pemanggil yang
mengimpor modul ini sebagai `feature_builder` tidak perlu berubah sama sekali
di luar baris import-nya sendiri.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from partrisk import config
from partrisk.features.fleet import _count_before, attach_fleet, attach_fleet_snapshot, fleet_snapshot
from partrisk.features.history import _HISTORY_COUNTS, attach_history
from partrisk.features.observations import current_observations, training_observations
from partrisk.features.support import cumulative_support, part_model_support, support_totals
from partrisk.features.transforms import _age_band, _log1p

__all__ = [
    "training_observations",
    "current_observations",
    "attach_history",
    "attach_fleet",
    "fleet_snapshot",
    "attach_fleet_snapshot",
    "cumulative_support",
    "support_totals",
    "part_model_support",
    "build_features",
    "project_features",
    "_HISTORY_COUNTS",
    "_count_before",
    "_log1p",
]


def build_features(raw: pd.DataFrame, support: pd.Series) -> pd.DataFrame:
    """Bangun 18 fitur model dari kolom mentah hasil data_reader.

    `support` adalah dukungan historis tipe PART per baris: saat training
    dihitung point-in-time, saat prediction diambil dari metadata model.
    """
    features = pd.DataFrame(index=raw.index)

    # --- Identitas & konteks -------------------------------------------------
    model_code = raw["item_model_code_clean"]
    # Tipe PART yang riwayatnya masih sangat sedikit digabung jadi satu
    # kategori supaya model tidak menghafal pola dari sampel kecil.
    features["part_model_category"] = np.where(
        model_code.isna(),
        config.UNKNOWN_LABEL,
        np.where(
            pd.to_numeric(support, errors="coerce").fillna(0)
            < config.MIN_PART_MODEL_SUPPORT,
            config.LOW_SUPPORT_LABEL,
            model_code.astype("string").fillna(config.UNKNOWN_LABEL),
        ),
    )
    features["client_category"] = (
        raw["installed_client_clean"].fillna(config.UNKNOWN_LABEL).astype(str)
    )

    # --- Umur pemasangan -----------------------------------------------------
    days_since_installation = pd.to_numeric(
        raw["days_since_installation"], errors="coerce"
    ).fillna(0.0)
    features["installation_age_band"] = _age_band(days_since_installation)
    features["log_days_since_installation"] = _log1p(days_since_installation)

    # --- Intensitas dan jenis riwayat ---------------------------------------
    features["log_total_prior_events"] = _log1p(raw["total_prior_events"])
    features["log_prior_failure_count"] = _log1p(raw["prior_failure_count"])
    features["has_prior_failure"] = (
        pd.to_numeric(raw["prior_failure_count"], errors="coerce").fillna(0) > 0
    )
    features["log_prior_corrective_count"] = _log1p(raw["prior_corrective_count"])
    features["has_prior_corrective"] = (
        pd.to_numeric(raw["prior_corrective_count"], errors="coerce").fillna(0) > 0
    )
    features["log_days_since_last_corrective"] = _log1p(raw["days_since_last_corrective"])
    features["log_prior_distinct_places"] = _log1p(raw["prior_distinct_places"])

    # --- Riwayat dalam jendela waktu tertentu -------------------------------
    features["log_prior_corrective_30d"] = _log1p(raw["prior_corrective_30d"])
    features["log_prior_failure_365d"] = _log1p(raw["prior_failure_365d"])
    features["log_prior_events_180d"] = _log1p(raw["prior_events_180d"])

    # --- Lifecycle antar-siklus ---------------------------------------------
    features["log_previous_cycle_lifetime_mean"] = _log1p(
        raw["previous_cycle_lifetime_mean"]
    )
    features["has_previous_cycle"] = raw["has_previous_cycle"].fillna(False).astype(bool)

    # --- Musiman ------------------------------------------------------------
    # Representasi siklik supaya Desember tidak dianggap jauh dari Januari.
    month = pd.to_datetime(raw["observation_on"]).dt.month
    features["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0)
    features["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)

    # --- Kondisi armada ------------------------------------------------------
    # Sudah dihitung attach_fleet (training) atau attach_fleet_snapshot
    # (prediction); di sini hanya disalin supaya kedua jalur memakai kolom
    # yang sama persis.
    for column in config.FLEET_FEATURES:
        features[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0.0)

    features[config.CATEGORICAL_FEATURES] = features[config.CATEGORICAL_FEATURES].astype(str)
    numeric = config.NUMERIC_FEATURES + config.FLEET_FEATURES
    features[numeric] = features[numeric].astype(float)
    return features[config.FEATURE_COLUMNS]


def project_features(raw: pd.DataFrame, support: pd.Series, steps_ahead: int) -> pd.DataFrame:
    """Fitur seandainya waktu maju `steps_ahead` x 30 hari tanpa kejadian baru.

    Dipakai prediction multi-horizon: umur pemasangan, waktu sejak corrective
    terakhir, dan bulan ikut bertambah sesuai waktu yang berlalu, sementara
    hitungan riwayat dibekukan. Asumsi "tidak ada kejadian baru" adalah
    penyederhanaan yang disengaja - tidak ada cara jujur untuk mengetahui
    kejadian yang belum terjadi.
    """
    if steps_ahead == 0:
        return build_features(raw, support)

    elapsed_days = steps_ahead * config.OBSERVATION_STEP_DAYS
    shifted = raw.copy()
    shifted["days_since_installation"] = (
        pd.to_numeric(raw["days_since_installation"], errors="coerce").fillna(0.0)
        + elapsed_days
    )
    # Kolom ini kosong kalau PART belum pernah kena corrective; harus tetap
    # kosong setelah diproyeksikan, bukan berubah jadi angka.
    shifted["days_since_last_corrective"] = (
        pd.to_numeric(raw["days_since_last_corrective"], errors="coerce") + elapsed_days
    )
    shifted["observation_on"] = pd.to_datetime(raw["observation_on"]) + pd.to_timedelta(
        elapsed_days, unit="D"
    )
    return build_features(shifted, support)
