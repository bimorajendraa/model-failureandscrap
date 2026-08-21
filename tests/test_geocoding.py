"""Penyaringan geocoding lokasi - logic murni, jaringan di-mock.

Lahir dari kejadian nyata saat dikembangkan: geocoding polos untuk "SERVICE
CENTER" (nama fasilitas internal) menempatkannya di gerai retail yang sama
sekali tidak terkait, hanya karena kebetulan ada di dalam kotak Jabodetabek.
Test di sini menjaga dua lapis penyaringan yang menutup celah itu: nama harus
berpola stasiun publik SEBELUM dikirim ke Nominatim, dan hasilnya harus jatuh
di dalam kotak Jabodetabek SESUDAH kembali.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from partrisk.api.services import geocoding_service as gs


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    """Cache disk terpisah per test, supaya test tidak saling memengaruhi
    dan tidak menyentuh .cache/geocode.json sungguhan.

    Sengaja tidak memakai fixture `tmp_path` bawaan pytest: di mesin ini
    fixture itu gagal karena folder temp bersama pytest tidak bisa
    dibersihkan (izin Windows), tidak terkait dengan test ini sama sekali.
    """
    directory = Path(tempfile.mkdtemp(prefix="geocode_test_"))
    monkeypatch.setattr(gs, "CACHE_PATH", directory / "geocode.json")
    monkeypatch.setattr(gs, "_last_request_at", 0.0)
    yield
    shutil.rmtree(directory, ignore_errors=True)


def _fake_response(payload: list[dict]):
    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    return _Response()


# ---------------------------------------------------------------------------
# Nama yang boleh dikirim ke Nominatim sama sekali
# ---------------------------------------------------------------------------


def test_nama_stasiun_lolos_saringan_pola():
    assert gs._looks_like_public_station("STASIUN JUANDA")
    assert gs._looks_like_public_station("stasiun juanda")  # tidak peka huruf besar/kecil
    assert gs._looks_like_public_station("BATU CEPER (KA BANDARA)")


def test_fasilitas_internal_ditolak_sebelum_ke_jaringan():
    for name in ("GUDANG NI", "SERVICE CENTER", "DIPO DEPOK", "IT KCI JUANDA"):
        assert not gs._looks_like_public_station(name), name


def test_fasilitas_internal_tidak_pernah_memanggil_nominatim(monkeypatch):
    """Ini pemeriksaan paling penting: nama non-stasiun tidak boleh sampai
    memanggil requests.get sama sekali - bukan cuma hasilnya dibuang."""
    called = []
    monkeypatch.setattr(gs.requests, "get", lambda *a, **k: called.append(1) or _fake_response([]))

    entry = gs._resolve_one("GUDANG NI")

    assert called == []
    assert entry["resolved"] is False
    assert "bukan nama stasiun publik" in entry["reason"]


def test_kalimat_pencarian_membuang_akhiran_ka_bandara():
    assert gs._search_query("BATU CEPER (KA BANDARA)") == "Stasiun BATU CEPER"
    assert gs._search_query("STASIUN JUANDA") == "STASIUN JUANDA"


# ---------------------------------------------------------------------------
# Penyaringan geografis Jabodetabek
# ---------------------------------------------------------------------------


def test_hasil_di_luar_jabodetabek_ditolak(monkeypatch):
    """Kejadian nyata: 'SERVICE CENTER' ketemu tapi di Semarang. Simulasikan
    nama STASIUN yang kebetulan hasilnya di luar Jabodetabek."""
    semarang = [{"lat": "-6.9932", "lon": "110.4203", "display_name": "Semarang"}]
    monkeypatch.setattr(gs, "_query_nominatim", lambda name: semarang)

    entry = gs._resolve_one("STASIUN TIDAK_DIKENAL")

    assert entry["resolved"] is False


def test_hasil_di_dalam_jabodetabek_diterima(monkeypatch):
    jakarta = [{"lat": "-6.1667", "lon": "106.8305", "display_name": "Juanda, Jakarta"}]
    monkeypatch.setattr(gs, "_query_nominatim", lambda name: jakarta)

    entry = gs._resolve_one("STASIUN JUANDA")

    assert entry["resolved"] is True
    assert entry["lat"] == pytest.approx(-6.1667)
    assert entry["lon"] == pytest.approx(106.8305)


def test_kandidat_pertama_di_luar_kotak_kandidat_kedua_di_dalam(monkeypatch):
    """Ambil kandidat pertama yang lolos, bukan berhenti di kandidat pertama
    apa pun hasilnya."""
    candidates = [
        {"lat": "-6.9932", "lon": "110.4203", "display_name": "Salah, Semarang"},
        {"lat": "-6.1667", "lon": "106.8305", "display_name": "Benar, Jakarta"},
    ]
    monkeypatch.setattr(gs, "_query_nominatim", lambda name: candidates)

    entry = gs._resolve_one("STASIUN JUANDA")

    assert entry["resolved"] is True
    assert "Benar" in entry["matched_name"]


def test_tidak_ada_kandidat_sama_sekali(monkeypatch):
    monkeypatch.setattr(gs, "_query_nominatim", lambda name: [])
    entry = gs._resolve_one("STASIUN JUANDA")
    assert entry["resolved"] is False
    assert entry["retry"] is False


def test_kegagalan_jaringan_ditandai_boleh_dicoba_lagi(monkeypatch):
    import requests

    def boom(name):
        raise requests.RequestException("timeout")

    monkeypatch.setattr(gs, "_query_nominatim", boom)
    entry = gs._resolve_one("STASIUN JUANDA")
    assert entry["resolved"] is False
    assert entry["retry"] is True


# ---------------------------------------------------------------------------
# Cache dan anggaran waktu
# ---------------------------------------------------------------------------


def test_lokasi_yang_sudah_di_cache_tidak_dicoba_lagi(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gs, "_resolve_one",
        lambda name: (calls.append(name), {"resolved": True, "lat": 0.0, "lon": 0.0})[1],
    )

    gs.resolve_missing(["STASIUN A"], budget_seconds=10)
    assert calls == ["STASIUN A"]

    gs.resolve_missing(["STASIUN A"], budget_seconds=10)
    assert calls == ["STASIUN A"], "lokasi yang sudah berhasil di-cache dipanggil lagi"


def test_kegagalan_boleh_dicoba_lagi_lain_kali(monkeypatch):
    monkeypatch.setattr(gs, "_resolve_one", lambda name: {"resolved": False, "retry": True})

    gs.resolve_missing(["STASIUN GAGAL"], budget_seconds=10)
    gs.resolve_missing(["STASIUN GAGAL"], budget_seconds=10)

    # Tidak melempar error dan hasilnya tetap konsisten - retry=True berarti
    # boleh dicoba lagi, bukan disimpan permanen sebagai gagal.
    entry = gs.known_coordinates(["STASIUN GAGAL"])["STASIUN GAGAL"]
    assert entry["retry"] is True


def test_anggaran_waktu_menghentikan_lebih_awal(monkeypatch):
    import time as time_module

    def slow_resolve(name):
        time_module.sleep(0.05)
        return {"resolved": True, "lat": 0.0, "lon": 0.0}

    monkeypatch.setattr(gs, "_resolve_one", slow_resolve)

    processed = gs.resolve_missing(
        [f"STASIUN {i}" for i in range(100)], budget_seconds=0.12
    )
    assert 0 < processed < 100


def test_koordinat_yang_belum_pernah_dicoba_kembalikan_none():
    result = gs.known_coordinates(["STASIUN BELUM_PERNAH_DICOBA"])
    assert result["STASIUN BELUM_PERNAH_DICOBA"] is None
