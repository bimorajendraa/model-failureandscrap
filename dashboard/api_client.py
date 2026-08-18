"""Satu-satunya pintu dashboard ke data.

Dashboard TIDAK pernah menyentuh database atau memuat model. Semua angka
datang lewat HTTP dari FastAPI, sehingga aturan bisnis, ambang risiko, dan
kredensial database hanya ada di satu tempat.

Hasilnya di-cache sebentar supaya berpindah halaman tidak memicu permintaan
baru untuk data yang sama.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# Batch scoring di sisi API memakan waktu puluhan detik saat cache-nya dingin.
REQUEST_TIMEOUT = (5, 180)
CACHE_TTL_SECONDS = 300


class ApiError(RuntimeError):
    """Permintaan ke API gagal."""


def _get(path: str, params: dict | None = None) -> dict:
    try:
        response = requests.get(
            f"{API_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as error:
        raise ApiError(
            f"Tidak bisa menghubungi API di {API_BASE_URL}. "
            f"Pastikan `uvicorn api.main:app` sedang jalan. ({type(error).__name__})"
        ) from error

    if response.status_code == 404:
        return response.json()
    if response.status_code >= 400:
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        raise ApiError(body.get("message", f"API menjawab {response.status_code}."))
    return response.json()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def health() -> dict:
    return _get("/health")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def model_info() -> dict:
    return _get("/api/v1/model")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Menghitung risiko seluruh PART aktif...")
def overview(top: int = 10) -> dict:
    return _get("/api/v1/overview", {"top": top})


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Menghitung risiko seluruh PART aktif...")
def filters() -> dict:
    return _get("/api/v1/filters")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Mengambil daftar prioritas...")
def recommendations(**params) -> dict:
    return _get("/api/v1/recommendations", {k: v for k, v in params.items() if v not in (None, "", "Semua")})


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Menilai PART...")
def assessment(item_id: str, explain: bool = True) -> dict:
    return _get(f"/api/v1/parts/{item_id}/assessment", {"explain": explain})


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def history(item_id: str) -> dict:
    return _get(f"/api/v1/parts/{item_id}/history")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Mencari koordinat lokasi...")
def locations_map(resolve: bool = True, budget_seconds: int = 60) -> dict:
    return _get(
        "/api/v1/locations/map", {"resolve": resolve, "budget_seconds": budget_seconds}
    )


def as_frame(items: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(items)


def percent(value: float | None) -> str:
    """Probabilitas -> persen. Kosong ditampilkan apa adanya, bukan 0%."""
    return "-" if value is None or pd.isna(value) else f"{value:.1%}"
