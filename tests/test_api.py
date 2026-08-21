"""Endpoint HTTP: bentuk jawaban, status code, dan penanganan kesalahan."""

from __future__ import annotations

import pytest

from partrisk import config
from tests.conftest import needs_database, needs_models

pytestmark = [needs_database, needs_models]

HORIZONS = config.PREDICTION_HORIZON_DAYS


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert body["model_version"]["failure"]
    assert body["model_version"]["scrap"]


def test_health_dengan_cek_database(client):
    body = client.get("/health", params={"check_database": True}).json()
    assert body["database"] == "reachable"


def test_model_info(client):
    body = client.get("/api/v1/model").json()
    assert body["failure"]["model_version"]
    assert body["scrap"]["model_version"]
    # Ambang risiko harus ikut dilaporkan: tanpa itu angka LOW/MEDIUM/HIGH
    # tidak bisa ditelusuri kembali ke dasarnya.
    assert "risk_cutoffs" in body["failure"]
    assert "risk_cutoffs" in body["scrap"]
    assert body["failure"]["horizons_days"] == HORIZONS


def test_prediksi_kerusakan_satu_part(client, scorable_item):
    body = client.get(f"/api/v1/parts/{scorable_item}/failure").json()
    assert body["status"] == "SCORED"
    failure = body["failure"]
    assert failure["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    for days in HORIZONS:
        assert 0.0 <= failure[f"failure_probability_{days}d"] <= 1.0


def test_risiko_kumulatif_tidak_menurun(client, scorable_item):
    failure = client.get(f"/api/v1/parts/{scorable_item}/failure").json()["failure"]
    values = [failure[f"failure_probability_{days}d"] for days in HORIZONS]
    assert values == sorted(values)


def test_prediksi_scrap_satu_part(client, scorable_item):
    body = client.get(f"/api/v1/parts/{scorable_item}/scrap").json()
    assert body["status"] == "SCORED"
    scrap = body["scrap"]
    assert 0.0 <= scrap["scrap_probability"] <= 1.0
    assert scrap["scrap_risk_level"] in ("LOW", "MEDIUM", "HIGH")
    # Sifat bersyaratnya harus ikut terkirim, bukan disimpan di dokumentasi
    # saja - angka ini gampang salah dibaca sebagai peluang PART rusak.
    assert scrap["scrap_risk_basis"]


def test_assessment_gabungan(client, scorable_item):
    body = client.get(f"/api/v1/parts/{scorable_item}/assessment").json()
    assert body["status"] == "SCORED"
    assert body["failure"]["risk_level"]
    assert body["recommendation"]["action"]
    assert body["recommendation"]["based_on"]["failure_risk_level"] == (
        body["failure"]["risk_level"]
    )
    assert body["model_version"]["failure"]
    assert body["explanation"]["disclaimer"]
    assert isinstance(body["explanation"]["factors"], list)


def test_assessment_tanpa_penjelasan(client, scorable_item):
    body = client.get(
        f"/api/v1/parts/{scorable_item}/assessment", params={"explain": False}
    ).json()
    assert body["status"] == "SCORED"
    assert body["explanation"] is None


def test_part_tidak_ditemukan(client):
    response = client.get("/api/v1/parts/PART-YANG-TIDAK-PERNAH-ADA/assessment")
    assert response.status_code == 404
    body = response.json()
    assert body["status"] == "NOT_FOUND"
    # Pesan kesalahan tidak boleh membocorkan isi dapur.
    assert "password" not in response.text.lower()
    assert "psycopg" not in response.text.lower()


def test_part_ada_tapi_tidak_bisa_diskor(client, not_scorable_item):
    """Bukan error: PART yang tidak terpasang memang tidak punya risiko
    kerusakan yang perlu diperkirakan."""
    response = client.get(f"/api/v1/parts/{not_scorable_item}/assessment")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "NOT_SCORABLE"
    assert body["reason"]
    assert body["failure"] is None
    # Data yang tidak ada TIDAK boleh diganti angka karangan supaya prediksi
    # tetap keluar.
    assert body["recommendation"] is None


def test_daftar_rekomendasi(client):
    body = client.get("/api/v1/recommendations", params={"limit": 5}).json()
    assert body["returned"] <= 5
    assert body["total"] >= body["returned"]
    assert body["scored_at"]["data_through"]
    ranks = [item["rank"] for item in body["items"]]
    assert ranks == sorted(ranks)
    for item in body["items"]:
        assert item["recommended_action"]
        assert item["priority"]


def test_saring_rekomendasi_berdasar_risiko(client):
    body = client.get(
        "/api/v1/recommendations", params={"risk": "HIGH", "limit": 20}
    ).json()
    assert all(item["failure_risk_level"] == "HIGH" for item in body["items"])


def test_saring_rekomendasi_berdasar_jenis_part(client):
    filters = client.get("/api/v1/filters").json()
    if not filters["item_types"]:
        return
    item_type = filters["item_types"][0]
    body = client.get(
        "/api/v1/recommendations", params={"item_type": item_type, "limit": 10}
    ).json()
    assert all(item["item_type"] == item_type for item in body["items"])


def test_cari_sebagian_item_id(client):
    """ID PART panjang; orang biasanya hanya ingat sebagian."""
    full = client.get("/api/v1/recommendations", params={"limit": 1}).json()["items"][0]
    fragment = full["item_id"][:7]
    body = client.get(
        "/api/v1/recommendations", params={"search": fragment, "limit": 50}
    ).json()
    assert body["total"] >= 1
    assert all(fragment in item["item_id"] for item in body["items"])
    assert any(item["item_id"] == full["item_id"] for item in body["items"])


def test_cari_yang_tidak_cocok_mengembalikan_kosong(client):
    body = client.get(
        "/api/v1/recommendations", params={"search": "TIDAK-ADA-INI"}
    ).json()
    assert body["total"] == 0
    assert body["items"] == []


def test_cors_tertutup_secara_bawaan(client):
    """Origin browser harus disebutkan eksplisit, bukan dibuka untuk semua."""
    from partrisk.api import settings

    if settings.CORS_ALLOW_ORIGINS:
        pytest.skip("CORS memang sedang dikonfigurasi di environment ini")
    response = client.get("/health", headers={"Origin": "http://jahat.example"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_aktif_saat_origin_didaftarkan(monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    import partrisk.api.main
    from partrisk.api import settings

    monkeypatch.setattr(settings, "CORS_ALLOW_ORIGINS", ["http://localhost:3000"])
    module = importlib.reload(partrisk.api.main)
    try:
        with TestClient(module.app) as configured:
            response = configured.get(
                "/health", headers={"Origin": "http://localhost:3000"}
            )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    finally:
        monkeypatch.undo()
        importlib.reload(partrisk.api.main)


def test_paging_rekomendasi(client):
    first = client.get("/api/v1/recommendations", params={"limit": 3}).json()
    second = client.get(
        "/api/v1/recommendations", params={"limit": 3, "offset": 3}
    ).json()
    assert first["total"] == second["total"]
    ids = {item["item_id"] for item in first["items"]}
    assert ids.isdisjoint({item["item_id"] for item in second["items"]})


def test_kandidat_penggantian(client):
    body = client.get(
        "/api/v1/recommendations",
        params={"replacement_candidates_only": True, "limit": 50},
    ).json()
    for item in body["items"]:
        assert item["replacement_candidate"] is True
        assert item["scrap_risk_level"] == "HIGH"
        assert item["failure_risk_level"] in ("MEDIUM", "HIGH")


def test_overview(client):
    body = client.get("/api/v1/overview", params={"top": 5}).json()
    summary = body["summary"]
    assert summary["active_parts"] > 0
    assert (
        summary["high_risk_parts"]
        + summary["medium_risk_parts"]
        + summary["low_risk_parts"]
        == summary["active_parts"]
    )
    assert len(body["top_priority"]) <= 5


def test_batas_limit_dijaga(client):
    from partrisk.api import settings

    body = client.get(
        "/api/v1/recommendations", params={"limit": settings.MAX_RECOMMENDATION_LIMIT * 10}
    ).json()
    assert body["returned"] <= settings.MAX_RECOMMENDATION_LIMIT


def test_tidak_ada_endpoint_training(client):
    """Training dan inference harus tetap terpisah."""
    for path in ("/train", "/api/v1/train", "/api/v1/model/train"):
        assert client.post(path).status_code in (404, 405)


def test_assessment_cocok_dengan_daftar_prioritas(client):
    """Angka di halaman detail harus sama dengan angka di daftar prioritas."""
    listed = client.get("/api/v1/recommendations", params={"limit": 1}).json()["items"][0]
    detail = client.get(f"/api/v1/parts/{listed['item_id']}/assessment").json()
    assert detail["status"] == "SCORED"
    for days in HORIZONS:
        column = f"failure_probability_{days}d"
        assert detail["failure"][column] == listed[column]
    assert detail["failure"]["risk_level"] == listed["failure_risk_level"]
    assert detail["recommendation"]["action"] == listed["recommended_action"]
    assert detail["scrap"]["scrap_probability"] == listed["scrap_probability"]
