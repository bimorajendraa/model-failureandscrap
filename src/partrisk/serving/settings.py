"""Pengaturan runtime inferensi - bukan konstanta model (lihat config.py)."""

from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# Berapa lama hasil batch scoring dipakai ulang sebelum dihitung ulang. Data
# sumber hanya bertambah beberapa kali sehari, dan satu kali batch memakan
# waktu menit-menitan, jadi menghitungnya per request tidak masuk akal.
BATCH_CACHE_TTL_SECONDS = _int("BATCH_CACHE_TTL_SECONDS", 3600)

# Seberapa sering batas waktu data diperiksa ulang. Pemeriksaannya satu query
# ringan, tetapi dipanggil di setiap request - jadi hasilnya ditahan sebentar.
# Ini juga yang menentukan seberapa cepat data baru terlihat oleh aplikasi.
DATA_FRESHNESS_TTL_SECONDS = _int("DATA_FRESHNESS_TTL_SECONDS", 60)
