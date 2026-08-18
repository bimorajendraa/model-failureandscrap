"""Recommendation engine - logic murni, tidak menyentuh database atau model."""

from __future__ import annotations

import itertools

import pytest

from api.services.recommendation_service import (
    PRIORITY_ORDER,
    RISK_LEVELS,
    is_replacement_candidate,
    recommend,
)


def test_setiap_kombinasi_risiko_punya_rekomendasi():
    """Tidak boleh ada kombinasi yang jatuh ke KeyError saat production."""
    for failure_level, scrap_level in itertools.product(RISK_LEVELS, RISK_LEVELS):
        decision = recommend(failure_level, scrap_level)
        assert decision["priority"] in PRIORITY_ORDER
        assert decision["action"]
        assert decision["message"]
        assert decision["based_on"] == {
            "failure_risk_level": failure_level,
            "scrap_risk_level": scrap_level,
        }


def test_risiko_scrap_boleh_kosong():
    """PART yang riwayatnya belum cukup tetap dapat rekomendasi, tanpa menebak."""
    for failure_level in RISK_LEVELS:
        decision = recommend(failure_level, None)
        assert decision["based_on"]["scrap_risk_level"] is None
        assert "scrap belum bisa dinilai" in decision["message"]


def test_kelompok_risiko_tidak_dikenal_ditolak():
    """Lebih baik gagal terang-terangan daripada diam-diam menyarankan MONITOR."""
    with pytest.raises(ValueError):
        recommend("SANGAT_TINGGI", "LOW")
    with pytest.raises(ValueError):
        recommend("HIGH", "EXTREME")


def test_prioritas_naik_mengikuti_risiko_kerusakan():
    """Risiko kerusakan lebih tinggi tidak boleh menghasilkan prioritas lebih rendah."""
    for scrap_level in (*RISK_LEVELS, None):
        ranks = [
            PRIORITY_ORDER[recommend(failure_level, scrap_level)["priority"]]
            for failure_level in ("LOW", "MEDIUM", "HIGH")
        ]
        assert ranks == sorted(ranks, reverse=True)


def test_scrap_tinggi_saja_tidak_menaikkan_prioritas():
    """Risiko scrap bersifat BERSYARAT terhadap kerusakan.

    PART yang kecil kemungkinannya rusak tidak jadi mendesak hanya karena
    seandainya rusak sulit diperbaiki.
    """
    assert recommend("LOW", "HIGH")["priority"] == "LOW"
    assert recommend("LOW", "HIGH")["action"] == "MONITOR"


def test_risiko_kerusakan_dan_scrap_sama_sama_tinggi_jadi_kritis():
    decision = recommend("HIGH", "HIGH")
    assert decision["priority"] == "CRITICAL"
    assert decision["action"] == "INSPECT_AND_PREPARE_REPLACEMENT"


def test_kandidat_penggantian_hanya_saat_dua_risiko_bertemu():
    assert is_replacement_candidate("HIGH", "HIGH")
    assert is_replacement_candidate("MEDIUM", "HIGH")
    assert not is_replacement_candidate("LOW", "HIGH")
    assert not is_replacement_candidate("HIGH", "MEDIUM")
    assert not is_replacement_candidate("HIGH", None)
