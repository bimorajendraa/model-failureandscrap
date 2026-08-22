"""Agregat riwayat item point-in-time - diekstrak dari `feature_builder.py`
(Fase B2 restrukturisasi), logic TIDAK diubah.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DAY = np.timedelta64(1, "D")

# Kolom agregat riwayat yang dihasilkan attach_history.
_HISTORY_COUNTS = [
    "total_prior_events",
    "prior_failure_count",
    "prior_corrective_count",
    "prior_corrective_30d",
    "prior_failure_365d",
    "prior_events_180d",
    "prior_distinct_places",
]


def attach_history(observations: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan ringkasan riwayat item pada setiap observasi.

    HANYA memakai event pada atau sebelum observation_on, jadi secara
    konstruksi tidak mungkin mengambil informasi masa depan.
    """
    observations = observations.reset_index(drop=True)
    total = len(observations)
    counts = {name: np.zeros(total, dtype="int64") for name in _HISTORY_COUNTS}
    last_corrective = np.full(total, np.datetime64("NaT"), dtype="datetime64[ns]")
    observed_at = observations["observation_on"].to_numpy("datetime64[ns]")

    if total and len(events):
        events = events.sort_values(
            ["item_identifier_clean", "created_on"], kind="stable"
        )
        event_times = events["created_on"].to_numpy("datetime64[ns]")
        is_corrective = events["wo_type_clean"].eq("CORRECTIVE").to_numpy()
        is_failure = events["is_failure_onset"].fillna(False).to_numpy(dtype=bool)
        is_new_place = _first_occurrence(events)
        event_rows = events.groupby("item_identifier_clean", sort=False).indices

        window_30 = np.timedelta64(30, "D")
        window_180 = np.timedelta64(180, "D")
        window_365 = np.timedelta64(365, "D")

        for item, rows in observations.groupby(
            "item_identifier_clean", sort=False
        ).indices.items():
            slot = event_rows.get(item)
            if slot is None:
                continue
            times = event_times[slot]
            cumulative_failure = np.cumsum(is_failure[slot])
            cumulative_corrective = np.cumsum(is_corrective[slot])
            cumulative_place = np.cumsum(is_new_place[slot])
            corrective_times = times[is_corrective[slot]]

            at = observed_at[rows]
            seen = np.searchsorted(times, at, side="right")
            seen_30 = np.searchsorted(times, at - window_30, side="right")
            seen_180 = np.searchsorted(times, at - window_180, side="right")
            seen_365 = np.searchsorted(times, at - window_365, side="right")

            failure_to_date = _at(cumulative_failure, seen)
            corrective_to_date = _at(cumulative_corrective, seen)

            counts["total_prior_events"][rows] = seen
            counts["prior_failure_count"][rows] = failure_to_date
            counts["prior_corrective_count"][rows] = corrective_to_date
            counts["prior_distinct_places"][rows] = _at(cumulative_place, seen)
            counts["prior_events_180d"][rows] = seen - seen_180
            counts["prior_corrective_30d"][rows] = corrective_to_date - _at(
                cumulative_corrective, seen_30
            )
            counts["prior_failure_365d"][rows] = failure_to_date - _at(
                cumulative_failure, seen_365
            )

            has_corrective = corrective_to_date > 0
            if has_corrective.any():
                position = np.maximum(corrective_to_date - 1, 0)
                last_corrective[rows] = np.where(
                    has_corrective, corrective_times[position], np.datetime64("NaT")
                )

    for name, values in counts.items():
        observations[name] = values
    # Kosong berarti PART belum pernah kena corrective sama sekali.
    observations["days_since_last_corrective"] = (observed_at - last_corrective) / _DAY
    return observations


def attach_degradation_history(
    observations: pd.DataFrame, cycles: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Tambahkan riwayat degradasi point-in-time: umur fisik kumulatif siklus
    sebelumnya, tren jarak antar-kerusakan, dan jendela corrective 60/90 hari.

    Dedup sengaja - REUSE fungsi `features.survival.dynamic_history` apa
    adanya (sudah terbukti aman point-in-time di jalur event-based), bukan
    ditulis ulang di sini. Perbedaan satu-satunya dari pemakaian di jalur
    survival: `cycles` boleh berupa riwayat SATU item saja (bukan seluruh
    armada) - `cumulative_cycle_age`/`corrective_degradation_trend`/
    `windowed_corrective_extra` semuanya group-by `item_identifier_clean`
    secara internal, jadi baris item lain (kalau ada) tidak pernah
    mempengaruhi hasil baris item ini (lihat predict/survival.py untuk
    pemakaian serupa yang sudah diverifikasi bit-identik).

    Terbukti menaikkan ROC-AUC/PR-AUC/Recall&Presisi@kapasitas sekaligus di
    populasi TEST yang sama (lihat reports/degradation_features_experiment.md)
    - bukan spekulasi.
    """
    from partrisk.features.survival import dynamic_history

    observations = observations.reset_index(drop=True)
    # `cumulative_cycle_age()` mengembalikan kolom RAW `cumulative_prior_cycle_days`/
    # `previous_cycle_count` (bukan di-merge langsung ke `observations`) -
    # jalur survival (features/survival/builder.py attach_dynamic_extra)
    # MEMBUTUHKAN nama kolom RAW yang SAMA PERSIS untuk merge internalnya
    # sendiri, dan `observations`/`full_snapshot` di sini SERING objek yang
    # sama dipakai ulang lintas model (lihat serving/batch_predictor.py) -
    # kalau kolom RAW ikut ditempelkan di sini, merge survival belakangan
    # bentrok nama (pandas otomatis jadi `_x`/`_y`, KeyError senyap di jalur
    # itu). Konversi ke log1p SEGERA di sini dan JANGAN simpan nama RAW-nya.
    cum = dynamic_history.cumulative_cycle_age(cycles)
    cum_lookup = cum.set_index("installation_cycle_id")
    matched = observations["installation_cycle_id"].map(cum_lookup["cumulative_prior_cycle_days"])
    count_matched = observations["installation_cycle_id"].map(cum_lookup["previous_cycle_count"])
    trend = dynamic_history.corrective_degradation_trend(observations, events)
    windowed = dynamic_history.windowed_corrective_extra(observations, events)

    observations["log_cumulative_prior_cycle_days"] = np.log1p(
        pd.to_numeric(matched, errors="coerce").fillna(0.0).clip(lower=0.0)
    ).to_numpy()
    observations["log_previous_cycle_count"] = np.log1p(
        pd.to_numeric(count_matched, errors="coerce").fillna(0.0)
    ).to_numpy()
    observations["has_failure_interval_trend"] = trend["has_failure_interval_trend"].to_numpy()
    observations["log_failure_interval_mean_days"] = trend["log_failure_interval_mean_days"].to_numpy()
    observations["failure_interval_trend_ratio"] = trend["failure_interval_trend_ratio"].to_numpy()
    observations["log_prior_corrective_60d"] = windowed["log_prior_corrective_60d"].to_numpy()
    observations["log_prior_corrective_90d"] = windowed["log_prior_corrective_90d"].to_numpy()
    return observations


def _at(cumulative: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Nilai kumulatif setelah `position` event; 0 kalau belum ada event."""
    return np.where(position > 0, cumulative[np.maximum(position - 1, 0)], 0)


def _first_occurrence(events: pd.DataFrame) -> np.ndarray:
    """Tandai event yang memperkenalkan lokasi baru untuk item tersebut.

    Jumlah lokasi berbeda sampai suatu titik waktu = jumlah penanda ini
    sampai titik tersebut, tanpa perlu menghitung ulang himpunan lokasi.
    """
    frame = events[["item_identifier_clean", "place_canonical_clean"]]
    known = frame["place_canonical_clean"].notna()
    return (known & ~frame.duplicated()).to_numpy()
