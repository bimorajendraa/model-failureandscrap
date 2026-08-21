"""Pengaturan lapisan serving.

Terpisah dari config.py: config.py memegang konstanta MODEL (ambang, fitur,
hyperparameter) yang berasal dari research dan tidak boleh berubah karena
alasan operasional. Yang di sini murni urusan server.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

import config

load_dotenv(config.ENV_FILE)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# Hitung batch saat aplikasi start (bukan saat request pertama datang).
WARMUP_BATCH_ON_STARTUP = os.getenv("WARMUP_BATCH_ON_STARTUP", "false").lower() in (
    "1", "true", "yes"
)

# Batas jumlah baris yang boleh diminta sekali panggil.
MAX_RECOMMENDATION_LIMIT = _int("MAX_RECOMMENDATION_LIMIT", 500)

# Batas atas anggaran waktu untuk geocoding lokasi per panggilan
# /api/v1/locations/map, apa pun yang diminta lewat ?budget_seconds=. Menjaga
# satu request tidak menggantung lama walau ada banyak lokasi baru yang
# belum pernah dicoba.
GEOCODE_BUDGET_SECONDS_DEFAULT = _int("GEOCODE_BUDGET_SECONDS_DEFAULT", 60)
GEOCODE_BUDGET_SECONDS_MAX = _int("GEOCODE_BUDGET_SECONDS_MAX", 90)

# Origin yang boleh memanggil API dari browser. Streamlit tidak memerlukannya
# (panggilannya server-ke-server), tetapi frontend browser mana pun akan
# diblokir tanpa ini. Kosong = tidak ada origin browser yang diizinkan.
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
DEFAULT_RECOMMENDATION_LIMIT = _int("DEFAULT_RECOMMENDATION_LIMIT", 50)

# Dipakai dashboard untuk menemukan API.
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
