"""Metrik evaluasi survival NATIVE (Lapis 1 - README bagian Evaluasi):
C-index, Integrated Brier Score, Brier per horizon, time-dependent AUC.
Semua lewat scikit-survival langsung, tidak ada rumus custom.

Horizon yang melebihi window follow-up TRAIN/eval TIDAK dipaksakan - kalau
sksurv menolak (ValueError, khas untuk horizon yang melebihi rentang data),
horizon itu dilaporkan sebagai "tidak dapat dihitung" (dict kosong/None),
bukan diam-diam dilewati atau membuat evaluasi gagal total.
"""

from __future__ import annotations

import numpy as np
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from . import curves

HORIZONS_DAYS = [30, 60, 90, 120]


def _usable_horizons(y_train, y_eval, horizons: list[int] = HORIZONS_DAYS) -> list[int]:
    """Horizon aman untuk IBS/Brier/AUC: harus lebih pendek dari follow-up
    maksimum TRAIN maupun follow-up eval, dan lebih besar dari follow-up
    minimum eval (syarat estimasi IPCW di scikit-survival)."""
    max_train = float(y_train["time"].max())
    max_eval = float(y_eval["time"].max())
    min_eval = float(y_eval["time"].min())
    limit = min(max_train, max_eval)
    return [h for h in horizons if min_eval < h < limit]


def native_metrics(model, y_train, x_eval, y_eval, risk_sign: int = 1) -> dict:
    """C-index + (kalau follow-up cukup panjang) IBS/Brier/AUC per horizon
    30/60/90/120 hari, dievaluasi dari t=0=installed_on - cara standar
    survival dievaluasi, BUKAN metrik operasional 30-hari (itu Lapis 2,
    lihat evaluate.py compare_with_classification()).

    `risk_sign`: sebagian besar model survival (RSF, ExtraSurvivalTrees, Cox
    PH, GradientBoostingSurvivalAnalysis loss='coxph') mengikuti konvensi
    predict()="skor lebih tinggi = lebih berisiko" - tapi
    GradientBoostingSurvivalAnalysis dengan loss='ipcwls'/'squared'
    predict()-nya mengembalikan PERKIRAAN WAKTU (arah terbalik). `risk_sign`
    (dari src/model_fit.MODEL_REGISTRY) membalik arah itu SEBELUM dipakai di
    concordance_index_censored/ipcw & cumulative_dynamic_auc, supaya seluruh
    model dibandingkan pada konvensi yang sama. Default 1 (tidak dibalik) -
    aman untuk model lama (RSF/Cox) yang sudah sesuai konvensi."""
    risk = risk_sign * model.predict(x_eval)
    c_index = concordance_index_censored(y_eval["event"], y_eval["time"], risk)[0]

    result: dict = {
        "rows": int(len(y_eval)),
        "events": int(y_eval["event"].sum()),
        "c_index": float(c_index),
        "max_followup_days": float(y_eval["time"].max()),
    }

    # Harrell C-index bisa bias optimis kalau censoring TIDAK acak terhadap
    # fitur (plausibel di sini: lifecycle yang installed_on-nya belakangan
    # otomatis lebih sering censored - lihat README bagian "Base rate
    # menurun antar split"). Uno/IPCW C-index menimbang ulang lewat model
    # censoring, jadi kurang bias oleh pola itu - dilaporkan berdampingan,
    # bukan menggantikan Harrell. tau dibatasi ke rentang follow-up yang
    # sama dipakai IBS/Brier/AUC di bawah supaya konsisten satu file ini.
    limit = min(float(y_train["time"].max()), float(y_eval["time"].max()))
    try:
        uno_c_index = concordance_index_ipcw(y_train, y_eval, risk, tau=limit * 0.99)[0]
        result["uno_c_index"] = float(uno_c_index)
    except ValueError:
        result["uno_c_index"] = None

    horizons = _usable_horizons(y_train, y_eval)
    result["horizons_evaluable_days"] = horizons
    if not horizons:
        result["integrated_brier_score"] = None
        result["brier_at_horizon"] = {}
        result["time_dependent_auc_at_horizon"] = {}
        return result

    times_grid, curve_values = curves.survival_curve_arrays(model, x_eval)
    surv_at_horizons = curves.step_eval_matrix(times_grid, curve_values, horizons)

    try:
        result["integrated_brier_score"] = float(
            integrated_brier_score(y_train, y_eval, surv_at_horizons, horizons)
        )
    except ValueError:
        result["integrated_brier_score"] = None

    try:
        _, brier_scores = brier_score(y_train, y_eval, surv_at_horizons, horizons)
        result["brier_at_horizon"] = {int(h): float(b) for h, b in zip(horizons, brier_scores)}
    except ValueError:
        result["brier_at_horizon"] = {}

    try:
        auc_scores, _ = cumulative_dynamic_auc(y_train, y_eval, risk, horizons)
        result["time_dependent_auc_at_horizon"] = {int(h): float(a) for h, a in zip(horizons, auc_scores)}
    except ValueError:
        result["time_dependent_auc_at_horizon"] = {}

    return result


def bootstrap_c_index(
    model, y_train, x_eval, y_eval, risk_sign: int = 1, n_boot: int = 200, seed: int = 42
) -> dict:
    """Interval kepercayaan C-index (Harrell) lewat bootstrap baris eval.

    README/audit sebelumnya melaporkan C-index sebagai titik tunggal - itu
    menyembunyikan seberapa jauh dua angka bisa dibedakan secara berarti.
    VALIDATION hanya punya 385 event (lihat metadata.json) - bootstrap di
    sini mengukur SEBERAPA LEBAR ketidakpastian itu, supaya kandidat model/
    fitur baru hanya dianggap "menang" kalau naiknya di luar rentang ini,
    bukan menang tipis 0,001 yang bisa jadi murni noise resampling.

    Risk score dihitung SEKALI di luar loop resampling (predict() tidak
    berubah antar resample - hanya baris mana yang dipakai concordance_index
    yang berubah), supaya 200 resample tidak memanggil ulang model.predict()
    200 kali."""
    risk = risk_sign * model.predict(x_eval)
    event = np.asarray(y_eval["event"])
    time = np.asarray(y_eval["time"])
    n = len(event)

    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            scores[i] = concordance_index_censored(event[idx], time[idx], risk[idx])[0]
        except ZeroDivisionError:
            # Resample tanpa pasangan comparable sama sekali (langka, hanya
            # mungkin pada n_boot besar/n kecil) - dibuang dari CI, bukan
            # dipaksa jadi 0.5 yang akan menyesatkan lebar interval.
            scores[i] = np.nan

    valid = scores[~np.isnan(scores)]
    return {
        "point_estimate": float(concordance_index_censored(event, time, risk)[0]),
        "bootstrap_mean": float(np.mean(valid)),
        "ci_lower_2_5": float(np.percentile(valid, 2.5)),
        "ci_upper_97_5": float(np.percentile(valid, 97.5)),
        "std": float(np.std(valid)),
        "n_boot_valid": int(len(valid)),
    }
