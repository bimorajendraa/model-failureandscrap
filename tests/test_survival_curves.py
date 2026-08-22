"""curves.calibrate_curve() - kurva S(t) terkalibrasi di SELURUH grid waktu.

Lahir dari bug nyata yang ditemukan saat prototyping (lihat
reports/rsf_median_curve_calibration_result.md): interval region yang salah
(strict '<'/'>' di kedua ujung) membuat titik grid yang PERSIS jatuh di
horizon terlatih (mis. t=60, t=90 - grid harian s/d 120 hari HAMPIR PASTI
memuat titik itu persis) tidak tercakup region manapun, diisi memori
`np.empty_like()` yang tidak diinisialisasi - lalu ikut ter-cummax ke titik
setelahnya. Test di sini menjaga regresi itu tidak terulang tanpa perlu
database (logic murni, calibrators sintetis).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from partrisk.survival import curves


def _synthetic_calibrators(horizons=(30, 60, 90, 120), seed=0) -> dict:
    """Isotonic per horizon, dilatih pada data sintetis (raw_risk naik -> label naik)."""
    rng = np.random.default_rng(seed)
    calibrators = {}
    for h in horizons:
        raw = np.sort(rng.random(200))
        label = (raw + rng.normal(0, 0.05, size=200) > 0.5).astype(float)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw, label)
        calibrators[h] = calibrator
    return calibrators


def _synthetic_curve(times: np.ndarray, n_rows: int = 5, seed: int = 1) -> np.ndarray:
    """S(t) sintetis, monoton turun per baris (step function wajar)."""
    rng = np.random.default_rng(seed)
    rate = rng.uniform(0.001, 0.01, size=n_rows)
    return np.exp(-np.outer(rate, times))


def test_calibrate_curve_titik_persis_di_horizon_terlatih_tidak_nan():
    """Regresi bug: t=60/t=90 (persis horizon terlatih) HARUS tercakup,
    bukan diam-diam jadi NaN/uninitialized."""
    times = np.array([1.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 200.0])
    curve = _synthetic_curve(times)
    calibrators = _synthetic_calibrators()
    result = curves.calibrate_curve(times, curve, calibrators)
    assert not np.isnan(result).any()
    assert result.shape == curve.shape


def test_calibrate_curve_grid_padat_harian_tidak_nan():
    """Grid harian penuh (meniru resolusi RSF production s/d 120 hari) -
    setiap kolom harus tercakup tepat satu region."""
    times = np.arange(1, 400, dtype=float)
    curve = _synthetic_curve(times)
    calibrators = _synthetic_calibrators()
    result = curves.calibrate_curve(times, curve, calibrators)
    assert not np.isnan(result).any()


def test_calibrate_curve_hasil_monoton_non_increasing():
    """S(t) terkalibrasi tidak boleh naik - cummax pada risk = cummin pada
    survival, WAJIB walau tiap calibrator sendiri monoton (bisa saling
    silang antar horizon - lihat docstring calibrate_curve())."""
    times = np.arange(1, 400, dtype=float)
    curve = _synthetic_curve(times, n_rows=20, seed=7)
    calibrators = _synthetic_calibrators()
    result = curves.calibrate_curve(times, curve, calibrators)
    assert (np.diff(result, axis=1) <= 1e-9).all()


def test_calibrate_curve_hasil_dalam_rentang_0_1():
    times = np.arange(1, 400, dtype=float)
    curve = _synthetic_curve(times)
    calibrators = _synthetic_calibrators()
    result = curves.calibrate_curve(times, curve, calibrators)
    assert (result >= -1e-9).all() and (result <= 1.0 + 1e-9).all()


def test_calibrate_curve_di_titik_horizon_pas_sama_dengan_calibrator_langsung():
    """Di t PERSIS SAMA dengan horizon terkecil (30, ujung kiri, tanpa
    interpolasi), hasil harus SAMA PERSIS dengan memanggil calibrator itu
    langsung - bukan cuma "dekat" lewat interpolasi."""
    times = np.array([30.0, 60.0, 90.0, 120.0])
    curve = _synthetic_curve(times, n_rows=3, seed=3)
    calibrators = _synthetic_calibrators()

    result = curves.calibrate_curve(times, curve, calibrators)

    raw_risk_at_30 = 1.0 - curve[:, 0]
    expected_calibrated_risk_at_30 = calibrators[30].predict(raw_risk_at_30)
    # cummax di titik PERTAMA grid = nilai kalibrator itu sendiri (tidak ada
    # yang di-akumulasi sebelumnya).
    np.testing.assert_allclose(1.0 - result[:, 0], expected_calibrated_risk_at_30)
