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


# ---------------------------------------------------------------------------
# Local failure density - generalisasi attach_fleet() ke kolom kategori
# apa pun (bukan hanya item_model_code_clean) - lihat
# reports/local_density_experiment.md. item_type_at_install TERBUKTI
# menaikkan ROC-AUC/PR-AUC/Brier/Recall&Presisi@kapasitas sekaligus;
# client/place DITOLAK (join episode->cycle cuma 87,8% cakupan, sinyal
# jadi berisik) - lihat laporan yang sama, jangan diulang tanpa bukti baru.
# ---------------------------------------------------------------------------

ITEM_TYPE_DENSITY_WINDOWS = (90, 180)


def local_density(
    observations: pd.DataFrame, cycles: pd.DataFrame, failures: pd.DataFrame,
    group_column: str, window_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalisasi MEKANISME attach_fleet() (point-in-time, dinormalkan per
    unit aktif) ke `group_column` APA PUN - bukan logic baru, cuma
    diparameterisasi (pola sama seperti categorical_support.py
    menggeneralisasi cumulative_support). `cycles`/`failures`/`observations`
    HARUS sudah punya `group_column`. Return (recent_count, fleet_size)."""
    observations = observations.reset_index(drop=True)
    at = observations["observation_on"].to_numpy("datetime64[ns]")
    window = at - np.timedelta64(int(window_days), "D")
    keys = observations[group_column].fillna(config.UNKNOWN_LABEL)

    def sort_by_group(frame: pd.DataFrame, time_col: str) -> dict:
        usable = frame.loc[frame[time_col].notna() & frame[group_column].notna()]
        grouped = usable.groupby(
            usable[group_column].fillna(config.UNKNOWN_LABEL), sort=False
        )
        return {
            name: np.sort(group[time_col].to_numpy("datetime64[ns]"))
            for name, group in grouped
        }

    cohort = cycles.loc[cycles["is_initial_model_cohort"].fillna(False)]
    eligible_failures = failures.loc[failures["is_initial_model_cohort"].fillna(False)]

    failure_times = sort_by_group(eligible_failures, "failure_onset_on")
    installed = sort_by_group(cohort, "installed_on")
    ended = sort_by_group(cohort, "cycle_end_on")

    recent = np.maximum(
        _count_before(failure_times, keys, at) - _count_before(failure_times, keys, window), 0
    )
    fleet = np.maximum(
        _count_before(installed, keys, at) - _count_before(ended, keys, at), 0
    )
    return recent, fleet


def _item_type_density_columns(frame: pd.DataFrame, cycles_aug: pd.DataFrame, episodes_aug: pd.DataFrame) -> pd.DataFrame:
    for window in ITEM_TYPE_DENSITY_WINDOWS:
        recent, fleet = local_density(frame, cycles_aug, episodes_aug, "item_type_at_install", window)
        frame[f"log_item_type_failures_{window}d"] = np.log1p(recent)
        frame[f"item_type_failure_rate_{window}d"] = recent / np.maximum(fleet, 1)
    return frame


def attach_item_type_density(
    observations: pd.DataFrame, events: pd.DataFrame, cycles: pd.DataFrame, episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Laju kerusakan point-in-time per item_type_at_install (90/180 hari) -
    jalur TRAINING (banyak observation_on berbeda per baris). Menempelkan
    item_type_at_install ke `observations`/`cycles` sendiri lewat
    `install_context.attach_install_context()` (REUSE join point-in-time
    yang sudah dipakai survival) - pemanggil tidak perlu melakukannya duluan.
    """
    from partrisk.features.survival import install_context

    observations = install_context.attach_install_context(observations, events)
    cycles_aug = install_context.attach_install_context(cycles, events)
    episodes_aug = episodes.rename(columns={"item_type_clean": "item_type_at_install"})
    return _item_type_density_columns(observations, cycles_aug, episodes_aug)


def item_type_density_snapshot(
    cycles: pd.DataFrame, events: pd.DataFrame, episodes: pd.DataFrame, at: pd.Timestamp,
) -> pd.DataFrame:
    """Snapshot laju kerusakan per item_type_at_install pada SATU titik
    waktu - jalur SERVING (predict satu-PART/batch), sama pola dengan
    `fleet_snapshot()` (per item_model_code_clean): dihitung sekali, dipakai
    ulang lewat lookup (`attach_item_type_density_snapshot`), bukan dihitung
    ulang tiap request."""
    from partrisk.features.survival import install_context

    cycles_aug = install_context.attach_install_context(cycles, events)
    episodes_aug = episodes.rename(columns={"item_type_clean": "item_type_at_install"})
    types = pd.Series(cycles_aug["item_type_at_install"].dropna().unique(), name="item_type_at_install")
    frame = pd.DataFrame({"item_type_at_install": types.to_numpy()})
    frame["observation_on"] = pd.Timestamp(at)
    frame = _item_type_density_columns(frame, cycles_aug, episodes_aug)
    density_columns = [c for c in frame.columns if c not in ("item_type_at_install", "observation_on")]
    return frame[["item_type_at_install", *density_columns]]


def attach_item_type_density_snapshot(
    observations: pd.DataFrame, events: pd.DataFrame, snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """Tempelkan snapshot density yang sudah dihitung - jalur SERVING.
    Menempelkan item_type_at_install ke `observations` sendiri dulu (REUSE
    join point-in-time), lalu lookup dari snapshot beku - pola sama dengan
    `attach_fleet_snapshot()`."""
    from partrisk.features.survival import install_context

    observations = install_context.attach_install_context(observations, events).reset_index(drop=True)
    lookup = snapshot.set_index("item_type_at_install")
    key = observations["item_type_at_install"]
    density_columns = [c for c in snapshot.columns if c != "item_type_at_install"]
    for column in density_columns:
        observations[column] = key.map(lookup[column]).fillna(0.0).astype(float)
    return observations
