"""Faktor risiko: apa yang ditulis harus sama dengan apa yang dihitung.

Logic murni, tidak menyentuh database maupun model.

Test di sini lahir dari salah baca yang nyata: label lama menulis "1 kerusakan
seumur hidup PART", yang terbaca sebagai "PART rusak permanen" padahal
maksudnya "1 kerusakan sepanjang riwayatnya" - dan PART itu justru sedang
terpasang dan bekerja normal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.services import explanation


def _row(**overrides) -> pd.Series:
    """Baris snapshot dengan nilai netral, lalu ditimpa seperlunya."""
    base = {
        "item_model_code_clean": "0521201",
        "days_since_installation": 20.0,
        "total_prior_events": 18,
        "prior_failure_count": 0,
        "prior_failure_365d": 0,
        "prior_corrective_count": 0,
        "prior_corrective_30d": 0,
        "days_since_last_corrective": np.nan,
        "prior_distinct_places": 1,
        "previous_cycle_lifetime_mean": 0.0,
        "has_previous_cycle": False,
        "log_model_failures_90d": 0.0,
        "model_failure_rate_90d": 0.0,
        "log_model_fleet_size": 0.0,
    }
    base.update(overrides)
    return pd.Series(base)


def _codes(row: pd.Series) -> set[str]:
    return {factor["code"] for factor in explanation.risk_factors(row)}


def _label(row: pd.Series, code: str) -> str:
    return next(f["label"] for f in explanation.risk_factors(row) if f["code"] == code)


def test_kerusakan_lama_tidak_terbaca_sebagai_rusak_permanen():
    row = _row(prior_failure_count=1, prior_failure_365d=0)
    label = _label(row, "OLDER_FAILURE_HISTORY")
    assert "sepanjang riwayat" in label
    # "seumur hidup" terbaca sebagai vonis, bukan sebagai hitungan kumulatif.
    assert "seumur hidup" not in label


def test_hitungan_korektif_disebut_sebagai_catatan_bukan_pekerjaan():
    """Satu work order menghasilkan beberapa catatan (permintaan, pengeluaran,
    pengiriman, pemasangan), jadi menyebutnya "4 pekerjaan" melebih-lebihkan."""
    row = _row(prior_corrective_30d=4, prior_corrective_count=9,
               days_since_last_corrective=20.0)
    recent = _label(row, "RECENT_CORRECTIVE_MAINTENANCE")
    history = _label(row, "CORRECTIVE_HISTORY")
    assert "catatan" in recent and "pekerjaan korektif" not in recent
    assert "catatan" in history and "seumur hidup" not in history


def test_kerusakan_baru_dan_lama_tidak_muncul_bersamaan():
    """Dua-duanya sekaligus akan membingungkan; hanya yang relevan dipakai."""
    recent = _row(prior_failure_count=3, prior_failure_365d=2)
    assert "RECENT_FAILURE_HISTORY" in _codes(recent)
    assert "OLDER_FAILURE_HISTORY" not in _codes(recent)

    older = _row(prior_failure_count=3, prior_failure_365d=0)
    assert "OLDER_FAILURE_HISTORY" in _codes(older)
    assert "RECENT_FAILURE_HISTORY" not in _codes(older)


def test_belum_pernah_rusak_ditandai_meringankan():
    factors = explanation.risk_factors(_row(prior_failure_count=0))
    no_failure = next(f for f in factors if f["code"] == "NO_FAILURE_HISTORY")
    assert no_failure["direction"] == explanation.MITIGATING


def test_faktor_hanya_muncul_kalau_datanya_ada():
    """Tidak boleh ada alasan yang dikarang untuk PART tanpa riwayat."""
    codes = _codes(_row())
    assert "RECENT_CORRECTIVE_MAINTENANCE" not in codes
    assert "CORRECTIVE_HISTORY" not in codes
    assert "FLEET_CONDITION" not in codes
    assert "LOCATION_CHANGES" not in codes
    assert "PREVIOUS_CYCLE_LIFETIME" not in codes
    # Umur pemasangan selalu ada, jadi selalu boleh ditampilkan.
    assert "INSTALLATION_AGE" in codes


def test_umur_pemasangan_jadi_faktor_risiko_hanya_di_kelompok_tertua():
    """Ambangnya memakai kelompok umur milik model, bukan angka karangan."""
    import config

    muda = explanation.risk_factors(_row(days_since_installation=20.0))
    tua = explanation.risk_factors(
        _row(days_since_installation=float(config.AGE_BAND_THRESHOLDS[-1] + 1))
    )
    assert next(f for f in muda if f["code"] == "INSTALLATION_AGE")["direction"] == (
        explanation.CONTEXT
    )
    assert next(f for f in tua if f["code"] == "INSTALLATION_AGE")["direction"] == (
        explanation.RISK
    )


def test_kondisi_armada_dilaporkan_dengan_jumlah_unit():
    row = _row(
        log_model_failures_90d=float(np.log1p(11)),
        log_model_fleet_size=float(np.log1p(848)),
        model_failure_rate_90d=11 / 848,
    )
    label = _label(row, "FLEET_CONDITION")
    assert "11 kerusakan" in label
    assert "848 unit" in label


def test_caveat_muncul_untuk_model_part_berdukungan_rendah():
    import config

    sedikit = explanation.caveats(_row(), {"0521201": 5})
    banyak = explanation.caveats(_row(), {"0521201": config.MIN_PART_MODEL_SUPPORT + 1})
    assert sedikit and "sedikit" in sedikit[0]
    assert banyak == []


def test_catatan_pembacaan_tersedia():
    assert "bukan berarti" in explanation.FAILURE_HISTORY_NOTE
    assert "CATATAN" in explanation.CORRECTIVE_NOTE
