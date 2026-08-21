"""Kondisi armada (fitur lintas-PART) - diekstrak dari `feature_builder.py`
(Fase B2 restrukturisasi), logic TIDAK diubah.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from partrisk import config


def _count_before(sorted_times: dict, keys: pd.Series, at: np.ndarray) -> np.ndarray:
    """Berapa kejadian milik kelompok yang sama terjadi SEBELUM tiap titik waktu."""
    result = np.zeros(len(at), dtype="int64")
    for key, rows in keys.groupby(keys, sort=False).indices.items():
        times = sorted_times.get(key)
        if times is not None:
            result[rows] = np.searchsorted(times, at[rows], side="left")
    return result


def attach_fleet(
    observations: pd.DataFrame, cycles: pd.DataFrame, failures: pd.DataFrame
) -> pd.DataFrame:
    """Kondisi armada model PART pada tiap tanggal observasi (jalur training).

    Dihitung point-in-time: hanya kerusakan yang terjadi SEBELUM tanggal
    observasi, dan jumlah unit yang sedang terpasang pada tanggal itu. Tidak
    ada informasi masa depan yang ikut.

    Laju dinormalkan per jumlah unit supaya model dengan banyak unit tidak
    otomatis terlihat bermasalah hanya karena jumlahnya banyak.
    """
    observations = observations.reset_index(drop=True)
    at = observations["observation_on"].to_numpy("datetime64[ns]")
    window = at - np.timedelta64(config.FLEET_WINDOW_DAYS, "D")
    keys = observations["item_model_code_clean"].fillna(config.UNKNOWN_LABEL)

    def sort_by_model(frame: pd.DataFrame, column: str) -> dict:
        usable = frame.loc[frame[column].notna()]
        grouped = usable.groupby(
            usable["item_model_code_clean"].fillna(config.UNKNOWN_LABEL), sort=False
        )
        return {
            name: np.sort(group[column].to_numpy("datetime64[ns]"))
            for name, group in grouped
        }

    cohort = cycles.loc[cycles["is_initial_model_cohort"].fillna(False)]
    eligible_failures = failures.loc[failures["is_initial_model_cohort"].fillna(False)]

    failure_times = sort_by_model(eligible_failures, "failure_onset_on")
    installed = sort_by_model(cohort, "installed_on")
    ended = sort_by_model(cohort, "cycle_end_on")

    recent = _count_before(failure_times, keys, at) - _count_before(failure_times, keys, window)
    fleet = np.maximum(
        _count_before(installed, keys, at) - _count_before(ended, keys, at), 0
    )
    return _fleet_columns(observations, np.maximum(recent, 0), fleet)


def fleet_snapshot(
    cycles: pd.DataFrame, episodes: pd.DataFrame, at: pd.Timestamp
) -> pd.DataFrame:
    """Kondisi armada tiap model PART pada satu titik waktu.

    Sengaja dihitung lewat attach_fleet yang SAMA PERSIS dengan jalur
    training - bukan rumus terpisah. Dua implementasi yang seharusnya sama
    pernah menyebabkan bug: yang satu menghitung siklus aktif sebagai
    "berakhir >= sekarang", yang lain "berakhir > sekarang", sehingga seluruh
    PART aktif terhitung nol dan lajunya meledak.
    """
    models = pd.Series(cycles["item_model_code_clean"].dropna().unique(), name="model")
    frame = pd.DataFrame({"item_model_code_clean": models.to_numpy()})
    frame["observation_on"] = pd.Timestamp(at)
    return attach_fleet(frame, cycles, episodes)[
        ["item_model_code_clean", *config.FLEET_FEATURES]
    ]


def attach_fleet_snapshot(
    observations: pd.DataFrame, snapshot: pd.DataFrame
) -> pd.DataFrame:
    """Tempelkan kondisi armada dari potret yang sudah dihitung.

    Titik observasi prediction selalu kejadian terbaru di database, yaitu
    tanggal potret itu dibuat - jadi hasilnya identik dengan yang dihitung
    point-in-time oleh attach_fleet.
    """
    observations = observations.reset_index(drop=True)
    lookup = snapshot.set_index("item_model_code_clean")
    model = observations["item_model_code_clean"]
    for column in config.FLEET_FEATURES:
        observations[column] = model.map(lookup[column]).fillna(0.0).astype(float)
    return observations


def _fleet_columns(
    observations: pd.DataFrame, recent: np.ndarray, fleet: np.ndarray
) -> pd.DataFrame:
    observations["log_model_failures_90d"] = np.log1p(recent)
    observations["model_failure_rate_90d"] = recent / np.maximum(fleet, 1)
    observations["log_model_fleet_size"] = np.log1p(fleet)
    return observations
