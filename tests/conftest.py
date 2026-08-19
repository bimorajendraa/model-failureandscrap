"""Perkakas bersama untuk test.

Sebagian besar test di sini menyentuh database dan model production sungguhan
- itu memang tujuannya: yang ingin dijaga adalah bahwa lapisan serving
menghasilkan angka yang SAMA dengan ML core, dan itu tidak bisa dibuktikan
dengan data palsu.

Kalau database atau model tidak tersedia, test yang bergantung padanya
di-skip, bukan gagal - supaya `pytest` tetap berguna di mesin yang hanya
memeriksa logic murni (mis. tests/test_recommendation.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _database_available() -> bool:
    try:
        import data_reader

        with data_reader.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001
        return False


def _models_available() -> bool:
    try:
        from inference import model_loader

        model_loader.versions()
        return True
    except Exception:  # noqa: BLE001
        return False


def _internet_available() -> bool:
    try:
        import requests

        requests.get("https://nominatim.openstreetmap.org/status", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


# Di CI, test yang di-skip diam-diam lebih berbahaya daripada test yang gagal:
# hasilnya terbaca "semua lulus" padahal tidak ada yang benar-benar diuji.
# Set REQUIRE_DATABASE=1 supaya ketidaktersediaan jadi kegagalan.
_STRICT = os.getenv("REQUIRE_DATABASE", "").lower() in ("1", "true", "yes")

_HAS_DATABASE = _database_available()
_HAS_MODELS = _models_available()

if _STRICT and not (_HAS_DATABASE and _HAS_MODELS):
    missing = []
    if not _HAS_DATABASE:
        missing.append("database")
    if not _HAS_MODELS:
        missing.append("model production")
    raise RuntimeError(
        "REQUIRE_DATABASE diaktifkan tetapi " + " dan ".join(missing) + " tidak tersedia."
    )

needs_database = pytest.mark.skipif(
    not _HAS_DATABASE, reason="database tidak bisa dihubungi"
)
needs_models = pytest.mark.skipif(
    not _HAS_MODELS, reason="model production belum ada di models/"
)
needs_internet = pytest.mark.skipif(
    not _internet_available(), reason="tidak ada akses internet (geocoding)"
)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def batch():
    """Hasil batch scoring, dihitung sekali untuk seluruh sesi test."""
    from inference import batch_predictor

    return batch_predictor.score_active_parts()


@pytest.fixture(scope="session")
def scorable_item(batch) -> str:
    """Satu PART aktif yang pasti bisa diskor."""
    return str(batch.frame["item_id"].iloc[0])


@pytest.fixture(scope="session")
def not_scorable_item(batch) -> str:
    """PART yang ada di database tetapi tidak bisa diskor.

    Dicari dari data, bukan ditulis tetap: ID yang hari ini tidak terpasang
    bisa saja terpasang lagi besok.
    """
    import data_reader

    active = set(batch.frame["item_id"])
    events = data_reader.get_events()
    for item in events["item_identifier_clean"].dropna().unique():
        if item not in active:
            return str(item)
    pytest.skip("semua PART di database sedang aktif")
