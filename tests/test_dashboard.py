"""Halaman dashboard benar-benar dijalankan, bukan hanya diimpor.

Streamlit menjalankan skrip halaman saat ada sesi yang membuka, bukan saat
server start - jadi `streamlit run` yang berhasil naik TIDAK membuktikan
halamannya bisa dirender. AppTest menjalankan skripnya sungguhan.

Sengaja TIDAK menambahkan dashboard/ ke sys.path: kalau Streamlit sendiri
tidak melakukannya, `streamlit run` juga akan gagal, dan test ini harus ikut
gagal alih-alih menutupinya.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import needs_database, needs_internet, needs_models

# needs_internet: halaman Peta Risiko memanggil /api/v1/locations/map, yang
# men-geocode lokasi lewat internet (lihat api/services/geocoding_service.py).
pytestmark = [needs_database, needs_models, needs_internet]

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
PAGES = [
    DASHBOARD_DIR / "app.py",
    DASHBOARD_DIR / "pages" / "1_Prioritas_Perawatan.py",
    DASHBOARD_DIR / "pages" / "2_Detail_PART.py",
    DASHBOARD_DIR / "pages" / "3_Perencanaan_Penggantian.py",
    DASHBOARD_DIR / "pages" / "4_Peta_Risiko.py",
]


@pytest.fixture(scope="module")
def api_url(client) -> str:
    """Arahkan dashboard ke TestClient, bukan ke server sungguhan.

    Dengan begitu test tidak menuntut `uvicorn` sudah jalan di latar.
    """
    return "testclient"


@pytest.fixture(autouse=True)
def route_dashboard_to_testclient(monkeypatch, client):
    """Ganti pemanggil HTTP milik dashboard supaya memanggil aplikasi langsung.

    Menambal `api_client._get`, BUKAN `requests.get` global: `requests` adalah
    satu modul yang sama untuk seluruh proses, jadi menambal `requests.get`
    lewat `api_client.requests` juga ikut menimpa pemanggilan `requests.get`
    milik modul lain (geocoding_service.py memanggil Nominatim sungguhan lewat
    modul `requests` yang persis sama) - bukan hanya milik dashboard.
    """
    import sys

    sys.path.insert(0, str(DASHBOARD_DIR))
    import api_client

    def fake_get(path, params=None):
        response = client.get(path, params=params)
        if response.status_code == 404:
            return response.json()
        if response.status_code >= 400:
            content_type = response.headers.get("content-type", "")
            body = response.json() if content_type.startswith("application/json") else {}
            raise api_client.ApiError(
                body.get("message", f"API menjawab {response.status_code}.")
            )
        return response.json()

    monkeypatch.setattr(api_client, "_get", fake_get)
    api_client.health.clear()
    api_client.overview.clear()
    api_client.filters.clear()
    api_client.recommendations.clear()
    api_client.assessment.clear()
    api_client.history.clear()
    api_client.locations_map.clear()
    yield


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.stem)
def test_halaman_bisa_dirender(page):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(page), default_timeout=300).run()
    assert not app.exception, [error.value for error in app.exception]
    assert not app.error, [box.value for box in app.error]
    assert app.title, "halaman tidak menampilkan judul"


def test_detail_part_menampilkan_angka(scorable_item):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(DASHBOARD_DIR / "pages" / "2_Detail_PART.py"), default_timeout=300
    ).run()
    app.text_input[0].set_value(scorable_item).run()

    assert not app.exception, [error.value for error in app.exception]
    labels = [metric.label for metric in app.metric]
    # Selalu "dalam N hari" - model tidak memperkirakan tanggal kerusakan, dan
    # tampilan tidak boleh membuatnya terbaca lain.
    assert "Rusak dalam 30 hari" in labels
    assert "Rusak dalam 120 hari" in labels
    assert any("Faktor risiko" in header.value for header in app.subheader)


def test_filter_lokasi_terisi_otomatis_dari_peta():
    """Simulasikan datang dari tombol "Lihat daftar PART" di peta - filter
    lokasi di halaman Prioritas Perawatan harus langsung terisi, bukan
    kembali ke "Semua"."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(DASHBOARD_DIR / "pages" / "1_Prioritas_Perawatan.py"), default_timeout=300
    )

    # Perlu tahu satu nama lokasi yang benar-benar ada sebelum bisa mengujinya.
    first_run = app.run()
    assert not first_run.exception
    location_box = next(
        box for box in first_run.selectbox if box.label == "Lokasi"
    )
    available_location = location_box.options[1] if len(location_box.options) > 1 else None
    if available_location is None:
        pytest.skip("tidak ada data lokasi untuk diuji")

    app.session_state["priority_location_filter"] = available_location
    second = app.run()

    assert not second.exception
    location_box = next(box for box in second.selectbox if box.label == "Lokasi")
    assert location_box.value == available_location
    # Sekali pakai - dikeluarkan dari session_state setelah dibaca.
    assert "priority_location_filter" not in app.session_state


def test_detail_part_menjelaskan_yang_tidak_bisa_diskor(not_scorable_item):
    """Halaman harus menerangkan sebabnya, bukan menampilkan angka kosong."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(DASHBOARD_DIR / "pages" / "2_Detail_PART.py"), default_timeout=300
    ).run()
    app.text_input[0].set_value(not_scorable_item).run()

    assert not app.exception, [error.value for error in app.exception]
    assert app.warning, "tidak ada keterangan kenapa PART tidak bisa dinilai"
    assert not app.metric, "menampilkan angka untuk PART yang tidak bisa diskor"
