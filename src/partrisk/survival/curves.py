"""Kurva survival: dari objek StepFunction scikit-survival ke array yang mudah
dievaluasi pada usia berapa pun (termasuk umur PART aktif sekarang).

Diekstrak dari `survival_model/src/utils.py` (Fase C1 restrukturisasi) -
bagian batas split temporal (TRAIN/VALIDATION/TEST) pindah ke
`features/survival/lifecycle.py` (dipakai bersama assign_lifecycle_outcome()
di file yang sama); logic di sini murni model-agnostic, tidak diubah.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def calibrate_curve(times: np.ndarray, curve_values: np.ndarray, calibrators: dict) -> np.ndarray:
    """S(t) TERKALIBRASI di SELURUH grid waktu - bukan cuma 4 titik horizon
    (30/60/90/120) seperti `predict/survival.py::_calibrate_risk()`.

    Dibutuhkan karena `median_survival_time()`/`survival_time_at_threshold()`
    dipanggil pada SELURUH kurva (S bisa turun ke 0,5/0,9 di titik waktu
    mana pun, bukan cuma di 4 horizon terlatih) - membaca median dari kurva
    MENTAH sementara `calibrated_risk_Nd` sudah dari kurva terkalibrasi adalah
    inkonsistensi (lihat reports/rsf_median_curve_baseline.md &
    rsf_median_curve_calibration_result.md untuk bukti empirisnya: median
    mentah bias optimis +751,9 hari, MAE turun ~40-53% setelah kurva penuh
    dikalibrasi konsisten).

    Metode: raw_risk(t)=1-S(t) dipetakan lewat isotonic per horizon TERLATIH
    (calibrators, TIDAK dilatih ulang di sini) - interpolasi LINEAR antara
    dua horizon terdekat untuk t di antaranya, flat-extrapolation calibrator
    ujung (horizon terkecil/terbesar) di luar rentang terlatih, lalu cummax
    WAJIB di SELURUH grid (bukan cuma 4 titik) - kalibrasi per-titik tidak
    menjamin hasil interpolasi tetap monoton walau tiap calibrator sendiri
    monoton.

    Setiap titik grid masuk TEPAT SATU region setengah-terbuka (h_lo, h_hi] -
    penting karena grid harian s/d 120 hari HAMPIR PASTI memuat titik yang
    PERSIS sama dengan horizon terlatih (mis. t=60, t=90); region tertutup-
    terbuka yang salah (mis. keduanya '<'/'>' ketat) membuat titik itu tidak
    tercakup region manapun (bug nyata yang sempat ditemukan saat prototyping
    - lihat assert di bawah, sengaja bukan silent no-op)."""
    horizons = sorted(calibrators)
    n_rows, n_grid = curve_values.shape
    raw_risk = 1.0 - curve_values
    calibrated_risk = np.full_like(raw_risk, np.nan)

    mask = times <= horizons[0]
    if mask.any():
        calibrated_risk[:, mask] = calibrators[horizons[0]].predict(raw_risk[:, mask].ravel()).reshape(n_rows, mask.sum())
    mask = times > horizons[-1]
    if mask.any():
        calibrated_risk[:, mask] = calibrators[horizons[-1]].predict(raw_risk[:, mask].ravel()).reshape(n_rows, mask.sum())
    for h_lo, h_hi in zip(horizons[:-1], horizons[1:]):
        mask = (times > h_lo) & (times <= h_hi)
        if not mask.any():
            continue
        t_sub = times[mask]
        weight = (t_sub - h_lo) / (h_hi - h_lo)
        sub_raw = raw_risk[:, mask]
        r_lo = calibrators[h_lo].predict(sub_raw.ravel()).reshape(n_rows, mask.sum())
        r_hi = calibrators[h_hi].predict(sub_raw.ravel()).reshape(n_rows, mask.sum())
        calibrated_risk[:, mask] = (1 - weight)[None, :] * r_lo + weight[None, :] * r_hi

    assert not np.isnan(calibrated_risk).any(), "calibrate_curve(): ada titik grid yang tidak tercakup region manapun"
    calibrated_risk = np.maximum.accumulate(calibrated_risk, axis=1)
    return 1.0 - calibrated_risk


def survival_time_at_threshold(times: np.ndarray, curve: np.ndarray, threshold: float) -> float | None:
    """Umur saat S(t) pertama kali <= threshold, atau None kalau kurva belum
    turun sampai situ dalam rentang follow-up training (tidak diekstrapolasi -
    lebih baik tidak menjawab daripada menjawab dengan menebak).

    Ambang tinggi (mis. 0,9) tercapai jauh lebih sering daripada ambang
    rendah (mis. 0,5 - "median") - lihat `days_until_survival_90pct` di
    predict/survival.py: kebanyakan PART aktif belum cukup lama untuk S(t)
    turun sampai separuh, jadi median_days_to_failure sering None. Ambang
    90% adalah field yang JAUH lebih sering terisi dan tetap actionable
    ("berapa hari lagi sampai risikonya mulai naik", bukan "kapan separuh
    populasi ini gagal")."""
    below = np.where(curve <= threshold)[0]
    if len(below) == 0:
        return None
    return float(times[int(below[0])])


def median_survival_time(times: np.ndarray, curve: np.ndarray) -> float | None:
    """Umur saat S(t) pertama kali <= 0.5 - lihat `survival_time_at_threshold()`."""
    return survival_time_at_threshold(times, curve, 0.5)
