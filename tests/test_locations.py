"""Endpoint peta lokasi: bentuk jawaban dan penyaringan geografis yang benar.

Menyentuh database (untuk data batch) dan internet sungguhan (Nominatim) -
anggaran waktunya sengaja kecil supaya test tetap cepat; cache di disk yang
dibangun antar-run membuat run berikutnya makin cepat.
"""

from __future__ import annotations

import pytest

from partrisk.api.services import geocoding_service
from tests.conftest import needs_database, needs_internet, needs_models

pytestmark = [needs_database, needs_models, needs_internet]


@pytest.fixture(scope="module")
def response(client):
    result = client.get(
        "/api/v1/locations/map", params={"resolve": True, "budget_seconds": 8}
    )
    assert result.status_code == 200
    return result.json()


def test_bentuk_jawaban(response):
    assert "resolved" in response
    assert "unresolved" in response
    assert response["scored_at"]["data_through"]


def test_titik_yang_ditampilkan_selalu_di_dalam_jabodetabek(response):
    box = geocoding_service.JABODETABEK_BBOX
    for item in response["resolved"]:
        assert box["south"] <= item["lat"] <= box["north"], item
        assert box["west"] <= item["lon"] <= box["east"], item


def test_setiap_lokasi_punya_hitungan_risiko(response):
    for item in response["resolved"] + response["unresolved"]:
        assert item["active_parts"] >= 1
        assert item["high_risk_parts"] >= 0
        assert item["medium_risk_parts"] >= 0


def test_tanpa_resolve_tidak_mengubah_cache(client):
    """?resolve=false harus murni membaca cache, tidak memanggil geocoding baru."""
    before = client.get(
        "/api/v1/locations/map", params={"resolve": False}
    ).json()
    after = client.get(
        "/api/v1/locations/map", params={"resolve": False}
    ).json()
    assert len(before["resolved"]) == len(after["resolved"])
    assert len(before["unresolved"]) == len(after["unresolved"])


def test_fasilitas_internal_tidak_pernah_muncul_sebagai_pin(response):
    """Kejadian nyata yang ditemukan saat dikembangkan: nama fasilitas
    internal (bukan stasiun publik) tidak boleh diberi koordinat sama sekali,
    walau kebetulan ada hasil pencarian yang jatuh di dalam Jabodetabek."""
    resolved_names = {item["location"] for item in response["resolved"]}
    for internal_name in ("GUDANG NI", "SERVICE CENTER", "DIPO DEPOK"):
        assert internal_name not in resolved_names
