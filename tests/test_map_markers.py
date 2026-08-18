"""Warna dan ukuran titik peta - logic murni, tidak menyentuh Streamlit/API.

Ditarik keluar dari halaman Peta Risiko supaya bisa diuji tanpa harus
mensimulasikan klik sungguhan pada peta (AppTest tidak bisa melakukan itu -
sama seperti keterbatasannya pada seleksi baris dataframe).
"""

from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import ui  # noqa: E402


def test_lokasi_dengan_risiko_tinggi_berwarna_merah():
    assert ui.risk_marker_color(high_risk_parts=3, medium_risk_parts=5) == ui.MAP_HIGH_COLOR


def test_lokasi_hanya_risiko_sedang_berwarna_oranye():
    assert ui.risk_marker_color(high_risk_parts=0, medium_risk_parts=2) == ui.MAP_MEDIUM_COLOR


def test_lokasi_tanpa_risiko_tinggi_sedang_berwarna_biru():
    assert ui.risk_marker_color(high_risk_parts=0, medium_risk_parts=0) == ui.MAP_LOW_COLOR


def test_risiko_tinggi_menang_atas_risiko_sedang():
    """Kombinasi tinggi+sedang tetap merah - bukan dicampur atau rata-rata."""
    assert ui.risk_marker_color(high_risk_parts=1, medium_risk_parts=10) == ui.MAP_HIGH_COLOR


def test_radius_naik_mengikuti_jumlah_risiko_tinggi():
    small = ui.risk_marker_radius(high_risk_parts=0)
    big = ui.risk_marker_radius(high_risk_parts=10)
    assert big > small


def test_radius_selalu_positif_walau_tanpa_risiko():
    assert ui.risk_marker_radius(high_risk_parts=0) > 0


def test_warna_adalah_rgba_valid():
    for color in (ui.MAP_HIGH_COLOR, ui.MAP_MEDIUM_COLOR, ui.MAP_LOW_COLOR):
        assert len(color) == 4
        assert all(0 <= channel <= 255 for channel in color)
