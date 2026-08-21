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
