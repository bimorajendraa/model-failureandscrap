"""Helper kecil dipakai lintas script survival_model: batas split temporal
berbasis lifecycle, dan pembacaan kurva S(t) hasil model survival.

Tidak menyentuh apa pun di luar survival_model/ - hanya membaca config.py
milik project utama untuk MIN_OBSERVATION_DATE (satu-satunya konstanta yang
di-reuse di sini, supaya batas bawah data sama dengan model classification).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

TRAIN, VALIDATION, TEST, EXCLUDED_TOO_OLD = "TRAIN", "VALIDATION", "TEST", "EXCLUDED_TOO_OLD"


def lifecycle_split_bounds(data_end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Batas VALIDATION/TEST, dihitung dari tahun data_end - formula yang SAMA
    dengan train.assign_split() di model classification, supaya kedua model
    diuji pada window kalender yang sama persis."""
    test_start = pd.Timestamp(year=data_end.year, month=1, day=1)
    validation_start = test_start - pd.DateOffset(years=1)
    return validation_start, test_start


def assign_lifecycle_split(installed_on: pd.Series, data_end: pd.Timestamp) -> pd.Series:
    """Split berdasar installed_on (awal lifecycle), bukan observation_on -
    unit datanya sudah lifecycle-level. Tanpa embargo bergaya classification;
    lihat README.md bagian "Leakage prevention" untuk alasannya."""
    validation_start, test_start = lifecycle_split_bounds(data_end)
    installed_on = pd.to_datetime(installed_on)
    split = pd.Series(EXCLUDED_TOO_OLD, index=installed_on.index)
    split[installed_on >= pd.Timestamp(config.MIN_OBSERVATION_DATE)] = TRAIN
    split[installed_on >= validation_start] = VALIDATION
    split[installed_on >= test_start] = TEST
    return split


# ---------------------------------------------------------------------------
# Kurva survival: dari objek StepFunction scikit-survival ke array yang mudah
# dievaluasi pada usia berapa pun (termasuk umur PART aktif sekarang).
# ---------------------------------------------------------------------------


def survival_curve_arrays(fitted_model, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Grid waktu (hari, dari t=0=installed_on) dan matriks S(t) (n_baris x n_waktu).

    scikit-survival mengembalikan StepFunction per baris dengan domain waktu
    yang sama (grid kejadian unik saat training) - diambil manual di sini
    (bukan lewat StepFunction.__call__) supaya bisa mengekstrapolasi rata di
    luar rentang training sendiri, alih-alih melempar ValueError.
    """
    step_functions = fitted_model.predict_survival_function(features)
    times = np.asarray(step_functions[0].x, dtype=float)
    curves = np.vstack([np.asarray(fn.y, dtype=float) for fn in step_functions])
    return times, curves


def eval_survival_at(times: np.ndarray, curve: np.ndarray, t: float) -> float:
    """S(t) dari satu kurva step-function.

    t sebelum grid pertama -> 1.0 (belum ada kejadian tercatat, S(0)=1 by
    definition). t melewati grid terakhir -> nilai terakhir yang diketahui
    (ekstrapolasi RATA, bukan ditebak turun/naik) - didokumentasikan sebagai
    keterbatasan di README, bukan disembunyikan sebagai presisi palsu.
    """
    if t <= 0 or t <= times[0]:
        return 1.0
    idx = int(np.searchsorted(times, t, side="right")) - 1
    idx = min(max(idx, 0), len(curve) - 1)
    return float(curve[idx])


def conditional_risk(times: np.ndarray, curve: np.ndarray, age_days: float, horizon_days: float) -> float:
    """P(failure <= age+horizon | selamat sampai age) = 1 - S(age+horizon)/S(age).

    Cara standar memakai kurva survival (dilatih dari t=0=installed_on) untuk
    subjek yang SUDAH berjalan sebagian - satu-satunya penyesuaian adalah
    berlalunya waktu, BUKAN fitur yang di-refresh (lihat README bagian
    "Keterbatasan: baseline instalasi vs kondisi sekarang").
    """
    s_age = eval_survival_at(times, curve, age_days)
    if s_age <= 1e-9:
        return 1.0
    s_future = eval_survival_at(times, curve, age_days + horizon_days)
    return float(np.clip(1.0 - s_future / s_age, 0.0, 1.0))


def step_eval_matrix(times: np.ndarray, curves: np.ndarray, query_times: list[float]) -> np.ndarray:
    """eval_survival_at(), divektorkan untuk banyak baris x banyak titik
    waktu sekaligus (dipakai evaluasi Brier/AUC per horizon). Step function
    (nilai konstan di antara event), BUKAN interpolasi linear - S(t) memang
    turun tangga, bukan garis lurus."""
    query_times = np.asarray(query_times, dtype=float)
    result = np.empty((curves.shape[0], len(query_times)))
    for j, t in enumerate(query_times):
        if t <= 0 or t <= times[0]:
            result[:, j] = 1.0
            continue
        idx = int(np.searchsorted(times, t, side="right")) - 1
        idx = min(max(idx, 0), curves.shape[1] - 1)
        result[:, j] = curves[:, idx]
    return result


def median_survival_time(times: np.ndarray, curve: np.ndarray) -> float | None:
    """Umur saat S(t) pertama kali <= 0.5, atau None kalau kurva belum turun
    sampai separuh dalam rentang follow-up training (tidak diekstrapolasi -
    lebih baik tidak menjawab daripada menjawab dengan menebak)."""
    below = np.where(curve <= 0.5)[0]
    if len(below) == 0:
        return None
    return float(times[int(below[0])])
