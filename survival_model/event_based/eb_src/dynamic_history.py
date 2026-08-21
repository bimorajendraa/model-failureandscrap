"""Fitur dynamic TAMBAHAN untuk event-based (sesi lanjutan): degradation
trend + cumulative physical usage across cycles + jendela corrective
60/90 hari. Semua dihitung point-in-time terhadap `observation_on` tiap
landmark, dengan mekanisme YANG SAMA (searchsorted/cumsum per item) dengan
`feature_builder.attach_history` - tidak ada logic leakage baru, hanya
jendela/statistik tambahan yang belum ada di fitur final saat ini.

Tiga kelompok:

1. `cumulative_cycle_age()` - KONSTAN per lifecycle (bicara tentang siklus
   SEBELUMNYA, bukan siklus berjalan, sama seperti `previous_cycle.py`):
   total hari FISIK yang sudah dijalani item ini di SEMUA siklus
   sebelumnya (durasi asli `cycle_end_on-installed_on`, APA PUN cara
   berakhirnya - beda dengan `previous_cycle_confirmed_failure_lifetime_mean`
   yang SENGAJA hanya menghitung siklus FAILURE: di sini pertanyaannya
   "berapa lama fisik part ini sudah dipakai", bukan "berapa lama part
   sejenis biasanya bertahan sampai gagal" - jadi end-reason TIDAK relevan,
   part yang direinstall tanpa failure tercatat TETAP benar-benar sudah aus
   secara fisik selama siklus itu).

2. `corrective_degradation_trend()` - DINAMIS per landmark: apakah jarak
   antar kejadian gagal (`is_failure_onset`) MENGECIL (memburuk) atau
   TIDAK, dibanding rata-rata historisnya sendiri.

3. `windowed_corrective_extra()` - DINAMIS per landmark: jumlah corrective
   60/90 hari terakhir (melengkapi `prior_corrective_30d` yang sudah ada).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import feature_builder


def cumulative_cycle_age(cycles: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per installation_cycle_id: `cumulative_prior_cycle_days`
    (total hari fisik SEMUA siklus SEBELUMNYA item yang sama - bukan
    termasuk siklus berjalan) dan `previous_cycle_count` (berapa siklus
    sebelumnya). `cycles` = populasi PENUH dari `data_reader.get_cycles()`
    (sama dengan yang dipakai `previous_cycle.py`), BUKAN dibatasi cohort
    eligible - riwayat fisik part valid dihitung dari populasi penuh."""
    frame = cycles.reset_index(drop=True).copy()
    frame["_sequence"] = frame["installation_cycle_id"].str.rsplit(":", n=1).str[-1].astype(int)
    frame = frame.sort_values(["item_identifier_clean", "_sequence"], kind="stable")

    duration_days = (frame["cycle_end_on"] - frame["installed_on"]) / np.timedelta64(1, "D")
    frame["_duration"] = duration_days.clip(lower=0.0)

    grouped = frame.groupby("item_identifier_clean", sort=False)["_duration"]
    # cumsum() dulu, BARU di-shift per grup (bukan shift() polos di atas hasil
    # cumsum - itu shift GLOBAL lintas baris, bukan per item, dan akan
    # membocorkan total item SEBELUMNYA ke baris siklus PERTAMA item
    # berikutnya di frame yang sudah diurutkan - diverifikasi lewat unit
    # check manual sebelum dipakai: bug itu mengenai 19.239/24.045 baris
    # pada percobaan pertama). groupby(...).shift(1) di sini memastikan
    # shift-nya terhenti di batas tiap item, sama seperti pola shift-dulu-
    # baru-agregat di previous_cycle.py.
    cumulative = grouped.cumsum()
    frame["cumulative_prior_cycle_days"] = (
        cumulative.groupby(frame["item_identifier_clean"], sort=False).shift(1).fillna(0.0)
    )
    frame["previous_cycle_count"] = frame.groupby("item_identifier_clean", sort=False).cumcount()

    return frame[["installation_cycle_id", "cumulative_prior_cycle_days", "previous_cycle_count"]].sort_index()


def corrective_degradation_trend(landmarks: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per landmark (butuh `item_identifier_clean` + `observation_on`):

    - `failure_interval_mean_days`: rata-rata jarak (hari) antar kejadian
      `is_failure_onset` SEBELUM observation_on ini (butuh >=2 kejadian).
    - `failure_interval_last_days`: jarak DUA kejadian PALING BARU sebelum
      observation_on ini.
    - `failure_interval_trend_ratio`: `last/mean` - **di bawah 1 berarti
      jarak terbaru LEBIH PENDEK dari rata-rata historis (memburuk/makin
      sering rusak)**, di atas 1 berarti membaik/melambat. `has_*` menandai
      baris yang cukup riwayat (>=2 interval, artinya >=3 kejadian) untuk
      dihitung - selain itu diisi 0/False, BUKAN diasumsikan stabil.
    """
    landmarks = landmarks.reset_index(drop=True)
    n = len(landmarks)
    mean_gap = np.zeros(n)
    last_gap = np.zeros(n)
    trend_ratio = np.zeros(n)
    has_trend = np.zeros(n, dtype=bool)

    failures = events.loc[events["is_failure_onset"].fillna(False)].sort_values(
        ["item_identifier_clean", "created_on"], kind="stable"
    )
    failure_times_by_item = {
        item: sub["created_on"].to_numpy("datetime64[ns]")
        for item, sub in failures.groupby("item_identifier_clean", sort=False)
    }

    at = landmarks["observation_on"].to_numpy("datetime64[ns]")
    items = landmarks["item_identifier_clean"].to_numpy()
    day = np.timedelta64(1, "D")

    rows_by_item = landmarks.groupby("item_identifier_clean", sort=False).indices
    for item, rows in rows_by_item.items():
        times = failure_times_by_item.get(item)
        if times is None or len(times) < 3:
            continue  # butuh >=3 kejadian untuk >=2 interval
        gaps = (times[1:] - times[:-1]) / day  # interval ke-i = antara kejadian i dan i+1
        cum_mean = np.cumsum(gaps) / np.arange(1, len(gaps) + 1)  # rata-rata interval SAMPAI posisi ini

        rows_arr = rows.to_numpy() if hasattr(rows, "to_numpy") else np.asarray(rows)
        # pos = jumlah kejadian is_failure_onset STRICT sebelum observation_on
        pos = np.searchsorted(times, at[rows_arr], side="left")
        # butuh >=3 kejadian sudah terjadi (pos>=3) supaya ADA >=2 interval utuh
        eligible = pos >= 3
        idx = np.clip(pos - 2, 0, len(gaps) - 1)  # indeks interval TERAKHIR yang sudah utuh (0-based ke-(pos-1))
        mean_gap[rows_arr[eligible]] = cum_mean[idx[eligible]]
        last_gap[rows_arr[eligible]] = gaps[idx[eligible]]
        has_trend[rows_arr[eligible]] = True

    valid_mean = has_trend & (mean_gap > 0)
    trend_ratio[valid_mean] = last_gap[valid_mean] / mean_gap[valid_mean]

    out = pd.DataFrame(index=landmarks.index)
    out["has_failure_interval_trend"] = has_trend
    out["log_failure_interval_mean_days"] = np.log1p(np.clip(mean_gap, 0, None))
    out["log_failure_interval_last_days"] = np.log1p(np.clip(last_gap, 0, None))
    out["failure_interval_trend_ratio"] = np.where(valid_mean, np.clip(trend_ratio, 0, 10), 1.0)
    return out


def windowed_corrective_extra(landmarks: pd.DataFrame, events: pd.DataFrame, windows=(60, 90)) -> pd.DataFrame:
    """Jumlah corrective (`wo_type_clean=='CORRECTIVE'`) dalam N hari
    terakhir sebelum observation_on, untuk tiap N di `windows` - melengkapi
    `prior_corrective_30d` yang sudah ada di fitur final (hanya 30 hari).

    Jendela 7/14 hari SEMPAT dicoba (reports/short_window.md) - ablation di
    atas CACHE lama terlihat menang (VAL t0-only C-index 0,7985->0,8058,
    AUC-30d naik juga), TAPI retrain penuh dengan database FRESH (cache
    dihapus) menunjukkan REGRESI di semua metrik (VAL t0 turun ke 0,7974,
    Recall@kapasitas turun ke 0,3345) - DIBATALKAN. Kemungkinan penyebab:
    jendela sesempit 7/14 hari sangat sensitif terhadap kapan tepatnya data
    ditarik dari database yang terus berubah (live) - hasil ablation di
    snapshot lama tidak representatif untuk snapshot baru. Pelajaran: fitur
    jendela SANGAT pendek butuh validasi ulang pada snapshot data yang
    SAMA PERSIS dengan yang dipakai retrain akhir, tidak cukup divalidasi
    sekali di cache yang bisa basi."""
    landmarks = landmarks.reset_index(drop=True)
    n = len(landmarks)
    out = pd.DataFrame(index=landmarks.index)
    for w in windows:
        out[f"prior_corrective_{w}d"] = np.zeros(n, dtype="int64")

    corrective = events.loc[events["wo_type_clean"].eq("CORRECTIVE")].sort_values(
        ["item_identifier_clean", "created_on"], kind="stable"
    )
    times_by_item = {
        item: sub["created_on"].to_numpy("datetime64[ns]")
        for item, sub in corrective.groupby("item_identifier_clean", sort=False)
    }

    at = landmarks["observation_on"].to_numpy("datetime64[ns]")
    items = landmarks["item_identifier_clean"].to_numpy()
    rows_by_item = landmarks.groupby("item_identifier_clean", sort=False).indices

    for item, rows in rows_by_item.items():
        times = times_by_item.get(item)
        if times is None or not len(times):
            continue
        rows_arr = rows.to_numpy() if hasattr(rows, "to_numpy") else np.asarray(rows)
        query = at[rows_arr]
        seen = np.searchsorted(times, query, side="right")
        for w in windows:
            window_start = query - np.timedelta64(w, "D")
            seen_window = np.searchsorted(times, window_start, side="right")
            out.loc[rows_arr, f"prior_corrective_{w}d"] = seen - seen_window

    for w in windows:
        out[f"log_prior_corrective_{w}d"] = np.log1p(out[f"prior_corrective_{w}d"])
    return out


def fleet_hierarchy_features(
    landmarks: pd.DataFrame, cycles_with_type: pd.DataFrame, episodes: pd.DataFrame, window_days: int = 90
) -> pd.DataFrame:
    """SAMA PERSIS mekanismenya dengan `feature_builder.attach_fleet()`
    (dipakai lewat `feature_builder._count_before`, TIDAK disalin ulang),
    TAPI dikelompokkan per `item_type_at_install` (tipe PART, lebih luas
    dari `item_model_code_clean`) alih-alih per model persis - part model
    dengan sampel unit sedikit (fleet kecil) mendapat sinyal laju kerusakan
    yang jauh lebih stabil dari tipe yang sama, alih-alih noise dari
    beberapa unit saja.

    `cycles_with_type` = `cycles` PENUH (bukan cohort/eligible saja - SAMA
    populasi dengan attach_fleet) yang SUDAH ditempel `item_type_at_install`
    (`install_context.attach_install_context`). `episodes` = `data_reader.
    get_failure_episodes()` APA ADANYA - sudah punya `item_type_clean`
    native (tidak perlu attach_install_context terpisah, item PART jarang
    berganti tipe, penyederhanaan yang wajar untuk fitur level ARMADA, bukan
    fitur per-item yang sensitif leakage)."""
    at = landmarks["observation_on"].to_numpy("datetime64[ns]")
    window = at - np.timedelta64(window_days, "D")
    keys = landmarks["item_type_at_install"].fillna("UNKNOWN")

    def sort_by_type(frame: pd.DataFrame, type_column: str, time_column: str) -> dict:
        usable = frame.loc[frame[time_column].notna()]
        grouped = usable.groupby(usable[type_column].fillna("UNKNOWN"), sort=False)
        return {name: np.sort(group[time_column].to_numpy("datetime64[ns]")) for name, group in grouped}

    cohort = cycles_with_type.loc[cycles_with_type["is_initial_model_cohort"].fillna(False)]
    eligible_failures = episodes.loc[episodes["is_initial_model_cohort"].fillna(False)]

    failure_times = sort_by_type(eligible_failures, "item_type_clean", "failure_onset_on")
    installed = sort_by_type(cohort, "item_type_at_install", "installed_on")
    ended = sort_by_type(cohort, "item_type_at_install", "cycle_end_on")

    recent = feature_builder._count_before(failure_times, keys, at) - feature_builder._count_before(
        failure_times, keys, window
    )
    fleet = np.maximum(
        feature_builder._count_before(installed, keys, at) - feature_builder._count_before(ended, keys, at), 0
    )
    recent = np.maximum(recent, 0)

    out = pd.DataFrame(index=landmarks.index)
    out["log_type_failures_90d"] = np.log1p(recent)
    out["type_failure_rate_90d"] = recent / np.maximum(fleet, 1)
    out["log_type_fleet_size"] = np.log1p(fleet)
    return out
